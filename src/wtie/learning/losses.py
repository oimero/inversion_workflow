"""wtie.learning.losses: 子波学习损失函数模块。

本模块提供确定性重建损失、变分损失及其辅助 KL 散度计算，
用于训练与超参数搜索阶段的误差评估。

边界说明
--------
- 本模块不负责网络结构定义、训练循环与优化器调度。
- 本模块不负责数据预处理，仅对输入张量执行损失计算。

核心公开对象
------------
1. ReconstructionLoss: 子波重建损失（可选叠加地震一致性项）。
2. VariationalLoss: 重建项与变分项的加权组合损失。

Examples
--------
>>> loss_fn = ReconstructionLoss(parameters)
>>> loss = loss_fn(data_batch, predicted_wavelet)
>>> total = loss["total"]
"""

import math

import torch

from wtie.processing.spectral import pt_convolution
from wtie.utils.types_ import Dict, Tensor


class AbstractLoss(object):
    """损失函数抽象基类。

    子类必须提供包含 total 键的 key_names，用于训练反向传播入口。

    Attributes
    ----------
    key_names : tuple of str
        当前损失对象可返回的损失项键名集合。
    """

    key_names = None

    def __init__(self):
        """初始化抽象损失基类并校验键名约束。

        Raises
        ------
        NotImplementedError
            当子类未设置 key_names 或缺少 total 键时抛出。
        """
        if self.key_names == None:
            raise NotImplementedError("Losses subcalsses must implement `key_names` attribute")

        if "total" not in self.key_names:
            raise NotImplementedError("The key `total` must be present for backprop.")


class ReconstructionLoss(AbstractLoss):
    """重建损失。

    该损失包含子波重建误差，并可按 beta 权重加入由反射率卷积得到的地震一致性误差。

    Attributes
    ----------
    key_names : tuple of str
        返回字典中固定损失键名，包含 total、wavelet、seismic。
    wavelet_loss_type : str
        子波误差类型，支持 mse 或 mae。
    seismic_loss_type : str
        地震误差类型，支持 mse 或 mae。
    beta : float
        地震误差权重；当 beta 为 None 或小于等于 0 时不启用地震项。
    is_validation_error : bool
        是否额外输出验证统计指标。
    """

    key_names = ("total", "wavelet", "seismic")

    def __init__(self, parameters: dict, is_validation_error: bool = True):
        """初始化重建损失。

        Parameters
        ----------
        parameters : dict
            参数字典，需包含 wavelet_loss_type、seismic_loss_type、beta。
        is_validation_error : bool, default=True
            是否在返回字典中附加 validation_error_mean 与 validation_error_std。

        Raises
        ------
        KeyError
            当 parameters 缺少必要键时抛出。
        """
        self.key_names = ReconstructionLoss.key_names
        super().__init__()

        self.wavelet_loss_type = parameters["wavelet_loss_type"]
        self.seismic_loss_type = parameters["seismic_loss_type"]
        self.beta = parameters["beta"]

        self.is_validation_error = is_validation_error

    def __call__(
        self,
        data_batch: dict,
        predicted_wavelet: Tensor,
    ) -> Dict[str, float]:
        """计算重建损失与可选验证统计。

        Parameters
        ----------
        data_batch : dict
            批数据字典，需至少包含：

            - wavelet: 真实子波，shape 为 (batch, n_samples)。
            - reflectivity: 反射率，shape 为 (batch, 1, n_samples)（待确认）。
        predicted_wavelet : Tensor
            预测子波，shape 为 (batch, n_samples)。

        Returns
        -------
        dict
            包含以下键：

            - total: 总损失。
            - wavelet: 子波重建损失。
            - seismic: 地震一致性损失（未启用时为零张量）。
            - validation_error_mean: 验证误差均值（仅 is_validation_error=True 时）。
            - validation_error_std: 验证误差标准差（仅 is_validation_error=True 时）。

        Raises
        ------
        KeyError
            当 data_batch 缺少 wavelet 或 reflectivity 时抛出。
        ValueError
            当 wavelet_loss_type 或 seismic_loss_type 非 mse/mae 时抛出。
        """
        # recons_loss = F.mse_loss(inp, out, reduction='sum')
        # torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction: str = 'mean', beta: float = 1.0)

        # ---------------
        # Wavelet error
        # ---------------
        label_wavelet = data_batch["wavelet"]
        if self.wavelet_loss_type == "mse":
            loss_wavelet = torch.mean(torch.sum((label_wavelet - predicted_wavelet).pow(2), dim=1), dim=0)
        elif self.wavelet_loss_type == "mae":
            loss_wavelet = torch.mean(torch.sum(torch.abs(label_wavelet - predicted_wavelet), dim=1), dim=0)
        else:
            raise ValueError("Wrong wavelet loss type %s" % str(self.wavelet_loss_type))

        # ---------------
        # Seismic error
        # ---------------
        beta = self.beta
        cond = (beta is not None) and beta > 0.0
        if cond:
            # add channel = 1
            true_seismic = pt_convolution(torch.unsqueeze(label_wavelet, 1), data_batch["reflectivity"])
            pred_seismic = pt_convolution(torch.unsqueeze(predicted_wavelet, 1), data_batch["reflectivity"])
            # shape [batch, 1, N]
            if self.seismic_loss_type == "mse":
                loss_seismic = torch.mean(torch.sum((true_seismic - pred_seismic).pow(2), dim=2), dim=(0, 1))
            elif self.seismic_loss_type == "mae":
                loss_seismic = torch.mean(torch.sum(torch.abs(true_seismic - pred_seismic), dim=2), dim=(0, 1))
            else:
                raise ValueError("Wrong seismic loss type %s" % str(self.seismic_loss_type))

        # -----------------
        # Total error
        # -----------------
        if cond:
            total_loss = (1 - beta) * loss_wavelet + beta * loss_seismic
            loss = {"total": total_loss, "wavelet": loss_wavelet, "seismic": loss_seismic}
        else:
            loss = {
                "total": loss_wavelet,
                "wavelet": loss_wavelet,
                "seismic": torch.zeros((1,), dtype=torch.float32, requires_grad=False),
            }

        # ------------------
        # Validation error (used for hyper-parameter search, not for training)
        # ------------------
        if self.is_validation_error:
            # validation_error = torch.sum(torch.abs(label_wavelet - predicted_wavelet), dim=1)
            validation_error = torch.sum((label_wavelet - predicted_wavelet).pow(2), dim=1)
            loss["validation_error_mean"] = torch.mean(validation_error)
            loss["validation_error_std"] = torch.std(validation_error)

        return loss  # type: ignore


class _CenteredUnitGaussianLoss(AbstractLoss):
    """零均值单位高斯先验下的 KL 散度损失。"""

    key_names = ("variational", "total")

    def __init__(self, parameters: dict = None):  # type: ignore
        """初始化 KL 散度损失。

        Parameters
        ----------
        parameters : dict, optional
            预留参数，当前实现未使用。
        """
        self.key_names = _CenteredUnitGaussianLoss.key_names
        super().__init__()

    def __call__(self, mu: Tensor, log_var: Tensor) -> Dict[str, float]:
        """计算批级 KL 散度均值。

        Parameters
        ----------
        mu : Tensor
            潜变量分布均值，shape 为 (batch, latent_dim)。
        log_var : Tensor
            潜变量分布方差对数，shape 为 (batch, latent_dim)。

        Returns
        -------
        dict
            包含 variational 与 total 两个键，值相同。
        """
        batch_kld_loss = _batch_kl_div_with_unit_gaussian(mu, log_var)

        kld_loss = torch.mean(batch_kld_loss, dim=0)

        return {"variational": kld_loss, "total": kld_loss}  # type: ignore


class VariationalLoss(AbstractLoss):
    """变分总损失。

    总损失按 alpha 加权组合重建损失与 KL 散度损失，并返回用于超参数搜索的验证误差统计。

    Attributes
    ----------
    key_names : tuple of str
        返回字典中固定损失键名，包含 total、wavelet、seismic、variational。
    reconstruction : ReconstructionLoss
        重建损失对象。
    kl_div : _CenteredUnitGaussianLoss
        KL 散度损失对象。
    alpha : float
        变分项权重，建议范围 [0, 1]（代码未强制约束）。
    """

    key_names = ("total", "wavelet", "seismic", "variational")

    def __init__(self, parameters: dict):
        """初始化变分损失。

        Parameters
        ----------
        parameters : dict
            参数字典，需包含 alpha_init 及 ReconstructionLoss 所需键。

        Raises
        ------
        KeyError
            当 parameters 缺少必要键时抛出。
        """
        self.key_names = VariationalLoss.key_names
        super().__init__()

        self.reconstruction = ReconstructionLoss(parameters, is_validation_error=False)
        self.kl_div = _CenteredUnitGaussianLoss()

        # weighting between reconstruction and variation
        self._alpha = parameters["alpha_init"]

    def __call__(self, data_batch: dict, predicted_wavelet: Tensor, mu: Tensor, log_var: Tensor) -> Dict[str, float]:
        """计算变分总损失与验证统计。

        Parameters
        ----------
        data_batch : dict
            批数据字典，需至少包含 wavelet 与 reflectivity。
        predicted_wavelet : Tensor
            预测子波，shape 为 (batch, n_samples)。
        mu : Tensor
            潜变量分布均值，shape 为 (batch, latent_dim)。
        log_var : Tensor
            潜变量分布方差对数，shape 为 (batch, latent_dim)。

        Returns
        -------
        dict
            包含重建项、变分项、总损失与 validation_error。
            其中 validation_error 是字典，含：

            - reconstruction_error: (mean, std)
            - variation_error: (mean, std)

        Raises
        ------
        KeyError
            当 data_batch 缺少必要键时抛出。
        """

        loss_dict = {}
        alpha = self.alpha

        # compute losses
        reconstruction_dict = self.reconstruction(data_batch, predicted_wavelet)
        kl_dict = self.kl_div(mu, log_var)

        # total loss
        total_loss = (1 - alpha) * reconstruction_dict["total"] + alpha * kl_dict["total"]

        # place in dict
        for key, value in reconstruction_dict.items():
            if key != "total":
                loss_dict[key] = value

        for key, value in kl_dict.items():
            if key != "total":
                loss_dict[key] = value

        loss_dict["total"] = total_loss

        # ------------------
        # Validation error (used for hyper-parameter search, not for training)
        # ------------------

        # api: https://ax.dev/docs/trial-evaluation.html
        # Dict[Tuple[mean: float, std: float]]
        validation_error = {}

        # reconstruction
        reconstrcution_validation_error = torch.sum(torch.abs(data_batch["wavelet"] - predicted_wavelet), dim=1)
        reconstrcution_validation_error_mean = torch.mean(reconstrcution_validation_error)
        reconstrcution_validation_error_std = torch.std(reconstrcution_validation_error)
        validation_error["reconstruction_error"] = (
            reconstrcution_validation_error_mean,
            reconstrcution_validation_error_std,
        )

        # variation
        batch_kld_loss = _batch_kl_div_with_unit_gaussian(mu, log_var)  # NaN problems?
        # batch_kld_loss = _dummy_mu_log_var_batch_error(mu, log_var)
        kld_loss_mean = torch.mean(batch_kld_loss)
        kld_loss_std = torch.std(batch_kld_loss)
        validation_error["variation_error"] = (kld_loss_mean, kld_loss_std)

        loss_dict["validation_error"] = validation_error

        return loss_dict

    @property
    def alpha(self):
        """float: 当前变分项权重。"""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        """设置变分项权重。

        Parameters
        ----------
        value : float
            新的 alpha 值；建议范围 [0, 1]，当前实现不做边界检查。
        """
        self._alpha = value


def _batch_kl_div_with_unit_gaussian(mu: Tensor, log_var: Tensor) -> Tensor:
    """计算每个样本相对单位高斯先验的 KL 散度。

    Parameters
    ----------
    mu : Tensor
        潜变量分布均值，shape 为 (batch, latent_dim)。
    log_var : Tensor
        潜变量分布方差对数，shape 为 (batch, latent_dim)。

    Returns
    -------
    Tensor
        批内逐样本 KL 散度，shape 为 (batch,)。
    """
    return -0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=1)


def OLD_tmp_mu_log_var_batch_error(mu: Tensor, log_var: Tensor) -> Tensor:
    """历史调试函数：基于 mu 与 log_var 的替代误差。

    Notes
    -----
    该函数名称与注释表明其为旧版临时实现，当前模块中未被主流程调用。

    Parameters
    ----------
    mu : Tensor
        潜变量均值，shape 为 (batch, latent_dim)。
    log_var : Tensor
        潜变量方差对数，shape 为 (batch, latent_dim)。

    Returns
    -------
    Tensor
        批内逐样本替代误差，shape 为 (batch,)。
    """
    # So far there are NaNs problems with kl_div...
    sigma = torch.exp(0.5 * log_var)  # positve

    # mu to 0 and sigma to 1

    # l1
    # batch_mu_error = torch.sum(torch.abs(mu), dim=1)
    # batch_sigma_error = torch.sum(torch.abs(sigma - torch.ones_like(sigma)), dim=1)

    # l2
    batch_mu_error = 100 * torch.sum(mu.pow(2), dim=1)
    batch_sigma_error = 100 * torch.sum((sigma - torch.ones_like(sigma)).pow(2), dim=1)

    return batch_mu_error + batch_sigma_error
