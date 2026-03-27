"""wtie.learning.network: 子波学习网络定义模块。

本模块提供确定性网络与变分网络两套 1D 架构，用于从地震与反射率输入中估计子波，
并提供轻量工具层以统一子模块调用。

边界说明
--------
- 本模块不负责训练循环、优化器调度与早停策略。
- 本模块不负责损失函数定义、数据导入导出与预处理。

核心公开对象
------------
1. Network / Net: 确定性子波估计网络。
2. VariationalNetwork: 变分子波估计网络。
3. Sequential: 对 `nn.Sequential` 的轻量包装。
4. ScalingOperator: 可学习幅值缩放算子。

Examples
--------
>>> import torch
>>> from wtie.learning.network import Net
>>> net = Net(wavelet_size=64)
>>> s = torch.randn(8, 1, 256)
>>> r = torch.randn(8, 1, 256)
>>> w = net(s, r)
>>> w.shape
torch.Size([8, 64])
"""

# See https://github.com/AntixK/PyTorch-VAE

import torch
import torch.nn as nn
import torch.nn.functional as F

from wtie.learning.blocks import DoubleConv1d, Down1d, LinearLrelu
from wtie.utils.types_ import List, Tensor, Tuple

##################################
# Constants
##################################
# _KERNEL_SIZE = 3
# _PADDING = _KERNEL_SIZE // 2
# _DOWN_SAMPLE_FACTOR = 2

# _ARGS = (_DOWN_SAMPLE_FACTOR, _KERNEL_SIZE, _PADDING)


##################################
# Abstarct Network
##################################

# TODO all nets must have same init arguments

##################################
# Regular Network
##################################


class Network(nn.Module):
    """确定性子波估计网络。

    网络将地震与反射率在通道维拼接后编码为潜在向量，再解码为子波并做幅值缩放。
    输入张量使用 NCL 约定：`(batch, channels, n_samples)`。

    Attributes
    ----------
    wavelet_size : int
        输出子波采样点数 `n`。
    encoder : Sequential
        编码器。
    wavelet_decoder : Sequential
        子波解码器。
    scaling : ScalingOperator
        可学习幅值缩放算子。
    """

    def __init__(self, wavelet_size: int, params: dict = None) -> None:  # type: ignore
        """初始化确定性网络。

        Parameters
        ----------
        wavelet_size : int
            输出子波长度（采样点数 `n`）。
        params : dict, optional
            网络超参数字典。支持键：

            - `dropout_rate`：Dropout 概率，默认 `0.25`。
            - `kernel_size`：卷积核长度，默认 `5`。
            - `downsampling_factor`：下采样倍率，默认 `2`。
        """

        super().__init__()

        self.wavelet_size = wavelet_size

        # Get extra paramters
        if params is None:
            params = {}

        p_drop = params.get("dropout_rate", 0.25)
        k_size = params.get("kernel_size", 5)
        padding = k_size // 2  # params.get('kernel_size', k_size//2)
        downsampling_factor = params.get("downsampling_factor", 2)

        n_in_channels = 2

        # ENCODE
        se_params = dict(factor=downsampling_factor, kernel_size=k_size, padding=padding)
        modules = []
        modules += [DoubleConv1d(in_channels=n_in_channels, out_channels=32, kernel_size=k_size, padding=padding)]
        modules += [nn.Dropout(p=p_drop, inplace=True)]
        modules += [Down1d(in_channels=32, out_channels=64, **se_params)]
        modules += [nn.Dropout(p=p_drop, inplace=True)]
        modules += [Down1d(in_channels=64, out_channels=128, **se_params)]
        modules += [nn.Dropout(p=p_drop, inplace=True)]
        modules += [Down1d(in_channels=128, out_channels=256, **se_params)]
        self.encoder = Sequential(modules)

        # DECODE
        modules = []
        modules += [LinearLrelu(in_features=256, out_features=512)]
        modules += [nn.Dropout(p=p_drop, inplace=True)]
        # modules += [LinearLrelu(in_features=n_hidden,out_features=n_hidden)]
        # modules += [nn.Dropout(p=p_drop, inplace = True)]
        modules += [nn.Linear(512, wavelet_size)]
        self.wavelet_decoder = Sequential(modules)

        # LEARN AMPLITUDE
        # modules = []
        # modules += [LinearLrelu(in_features=latent_dim,out_features=1)]
        # self.scaler = Sequential(modules)
        self.scaling = ScalingOperator()

    def encode(self, seismic: Tensor, reflecticvity: Tensor) -> Tensor:
        """编码地震与反射率为潜在向量。

        Parameters
        ----------
        seismic : Tensor
            输入地震张量，shape 为 `(batch, channels, n_samples)`。
        reflecticvity : Tensor
            输入反射率张量，shape 为 `(batch, channels, n_samples)`。

        Returns
        -------
        Tensor
            编码后的潜在表示，shape 为 `(batch, 256)`。
        """
        # cat
        x = torch.cat((seismic, reflecticvity), dim=1)

        # encode
        x = self.encoder(x)

        # gap
        x = torch.mean(x, 2, keepdim=False)

        return x

    def decode_wavelet(self, z: Tensor) -> Tensor:
        """将潜在向量解码为子波。

        Parameters
        ----------
        z : Tensor
            潜在向量，shape 为 `(batch, 256)`。

        Returns
        -------
        Tensor
            输出子波，shape 为 `(batch, wavelet_size)`。
        """
        w = self.wavelet_decoder(z)
        w = self.scaling(w)
        return w
        # scale = self.scaler(z)
        # return w*scale

    def forward(self, seismic: Tensor, reflectivity: Tensor) -> Tensor:
        """执行确定性前向推理。

        Parameters
        ----------
        seismic : Tensor
            输入地震张量，shape 为 `(batch, channels, n_samples)`。
        reflectivity : Tensor
            输入反射率张量，shape 为 `(batch, channels, n_samples)`。

        Returns
        -------
        Tensor
            预测子波，shape 为 `(batch, wavelet_size)`。
        """
        z = self.encode(seismic, reflectivity)
        wavelet = self.decode_wavelet(z)
        return wavelet


# alias
Net = Network


###################################
# Variational Network
###################################


class VariationalNetwork(nn.Module):
    """变分子波估计网络。

    该网络输出潜变量高斯分布参数 `mu` 与 `log_var`，通过重参数化采样后解码子波。
    输入张量使用 NCL 约定：`(batch, channels, n_samples)`。

    Attributes
    ----------
    wavelet_size : int
        输出子波采样点数 `n`。
    p_drop : float
        编码后特征的 Dropout 概率。
    encoder : Sequential
        编码器。
    fc_mu : torch.nn.Linear
        潜变量均值映射层。
    fc_logvar : torch.nn.Linear
        潜变量对数方差映射层。
    wavelet_decoder : Sequential
        子波解码器。
    scaling : ScalingOperator
        可学习幅值缩放算子。
    """

    def __init__(
        self,
        wavelet_size: int,
        params: dict = None,  # type: ignore
        one_ms: bool = False,
    ) -> None:
        """初始化变分网络。

        Parameters
        ----------
        wavelet_size : int
            输出子波长度（采样点数 `n`）。
        params : dict, optional
            网络超参数字典。支持键：

            - `dropout_rate`：Dropout 概率，默认 `0.05`。
            - `kernel_size`：卷积核长度，默认 `3`。
            - `downsampling_factor`：下采样倍率，默认 `3`。
        one_ms : bool, default=False
            是否启用 one_ms 分支结构。

        Notes
        -----
        当 `one_ms=True` 时，当前实现中的解码器线性层为 `Linear(1024, wavelet_size)`，
        与前一层输出维度是否严格匹配需结合外部配置进一步确认。
        """

        super().__init__()

        self.wavelet_size = wavelet_size

        # Get extra paramters
        if params is None:
            params = {}

        p_drop = params.get("dropout_rate", 0.05)
        k_size = params.get("kernel_size", 3)
        padding = k_size // 2
        downsampling_factor = params.get("downsampling_factor", 3)

        self.p_drop = p_drop

        n_in_channels = 2

        # ENCODE
        se_params = dict(factor=downsampling_factor, kernel_size=k_size, padding=padding)
        modules = []
        if one_ms:
            modules += [DoubleConv1d(in_channels=n_in_channels, out_channels=32, kernel_size=k_size, padding=padding)]
            modules += [nn.Dropout(p=p_drop, inplace=True)]
            modules += [Down1d(in_channels=32, out_channels=64, **se_params)]
            modules += [nn.Dropout(p=p_drop, inplace=True)]
            modules += [Down1d(in_channels=64, out_channels=128, **se_params)]
            modules += [nn.Dropout(p=p_drop, inplace=True)]
            modules += [Down1d(in_channels=128, out_channels=128, **se_params)]
            modules += [nn.Dropout(p=p_drop, inplace=True)]
            modules += [Down1d(in_channels=128, out_channels=256, **se_params)]
        else:
            modules += [DoubleConv1d(in_channels=n_in_channels, out_channels=32, kernel_size=k_size, padding=padding)]
            modules += [nn.Dropout(p=p_drop, inplace=True)]
            modules += [Down1d(in_channels=32, out_channels=64, **se_params)]
            modules += [nn.Dropout(p=p_drop, inplace=True)]
            modules += [Down1d(in_channels=64, out_channels=128, **se_params)]
            modules += [nn.Dropout(p=p_drop, inplace=True)]
            modules += [Down1d(in_channels=128, out_channels=256, **se_params)]
        self.encoder = Sequential(modules)

        # LATENT
        self.fc_mu = nn.Linear(256, 128)
        self.fc_logvar = nn.Linear(256, 128)

        # DECODE
        modules = []
        if one_ms:
            modules += [LinearLrelu(in_features=128, out_features=512)]
            modules += [nn.Dropout(p=p_drop, inplace=True)]
            modules += [nn.Linear(1024, wavelet_size)]
        else:
            modules += [LinearLrelu(in_features=128, out_features=512)]
            modules += [nn.Dropout(p=p_drop, inplace=True)]
            modules += [nn.Linear(512, wavelet_size)]
        self.wavelet_decoder = Sequential(modules)

        # LEARN AMPLITUDE
        self.scaling = ScalingOperator()

    def encode(self, seismic: Tensor, reflecticvity: Tensor) -> Tuple[Tensor]:
        """编码输入并输出潜变量分布参数。

        Parameters
        ----------
        seismic : Tensor
            输入地震张量，shape 为 `(batch, channels, n_samples)`。
        reflecticvity : Tensor
            输入反射率张量，shape 为 `(batch, channels, n_samples)`。

        Returns
        -------
        tuple of Tensor
            `(mu, log_var)`，两者 shape 均为 `(batch, 128)`。
        """
        # cat
        x = torch.cat((seismic, reflecticvity), dim=1)

        # encode
        x = self.encoder(x)

        # global average pooling (gap)
        x = torch.mean(x, 2, keepdim=False)

        # drop
        x = F.dropout(x, p=self.p_drop, training=self.training)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)

        return mu, log_var  # type: ignore

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """重参数化采样潜变量。

        Parameters
        ----------
        mu : Tensor
            潜变量高斯分布均值，shape 为 `(batch, latent_dim)`。
        log_var : Tensor
            潜变量高斯分布方差对数，shape 为 `(batch, latent_dim)`。

        Returns
        -------
        Tensor
            采样潜变量，shape 为 `(batch, latent_dim)`。
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: Tensor) -> Tensor:
        """将潜变量解码为子波。

        Parameters
        ----------
        z : Tensor
            潜变量张量，shape 为 `(batch, 128)`。

        Returns
        -------
        Tensor
            输出子波，shape 为 `(batch, wavelet_size)`。
        """
        w = self.wavelet_decoder(z)
        w = self.scaling(w)
        return w

    def sample(self, seismic: Tensor, reflectivity: Tensor) -> Tensor:
        """执行一次随机采样推理。

        Parameters
        ----------
        seismic : Tensor
            输入地震张量，shape 为 `(batch, channels, n_samples)`。
        reflectivity : Tensor
            输入反射率张量，shape 为 `(batch, channels, n_samples)`。

        Returns
        -------
        Tensor
            单次采样子波，shape 为 `(batch, wavelet_size)`。
        """
        wavelet, _, _ = self.forward(seismic, reflectivity)  # type: ignore
        return wavelet

    def sample_n_times(self, seismic: Tensor, reflectivity: Tensor, n: int) -> List[Tensor]:
        """重复采样子波分布。

        Parameters
        ----------
        seismic : Tensor
            输入地震张量，shape 为 `(batch, channels, n_samples)`。
        reflectivity : Tensor
            输入反射率张量，shape 为 `(batch, channels, n_samples)`。
        n : int
            采样次数，应为正整数。

        Returns
        -------
        list of Tensor
            采样结果列表，长度为 `n`，每个元素 shape 为 `(batch, wavelet_size)`。
        """
        wavelets_distribution = []
        for _ in range(n):
            wavelets_distribution.append(self.sample(seismic, reflectivity))
        return wavelets_distribution

    def compute_expected_wavelet(self, seismic: Tensor, reflectivity: Tensor) -> Tensor:
        """计算期望子波（使用 `mu` 直接解码）。

        Parameters
        ----------
        seismic : Tensor
            输入地震张量，shape 为 `(batch, channels, n_samples)`。
        reflectivity : Tensor
            输入反射率张量，shape 为 `(batch, channels, n_samples)`。

        Returns
        -------
        Tensor
            期望子波，shape 为 `(batch, wavelet_size)`。
        """
        mu, _ = self.encode(seismic, reflectivity)  # type: ignore
        wavelet = self.decode(mu)
        return wavelet

    def forward(self, seismic: Tensor, reflectivity: Tensor) -> Tuple[Tensor]:
        """执行变分前向推理。

        Parameters
        ----------
        seismic : Tensor
            输入地震张量，shape 为 `(batch, channels, n_samples)`。
        reflectivity : Tensor
            输入反射率张量，shape 为 `(batch, channels, n_samples)`。

        Returns
        -------
        tuple of Tensor
            `(wavelet, mu, log_var)`，其中：

            - `wavelet` 的 shape 为 `(batch, wavelet_size)`。
            - `mu` 与 `log_var` 的 shape 为 `(batch, 128)`。
        """
        mu, log_var = self.encode(seismic, reflectivity)  # type: ignore
        z = self.reparameterize(mu, log_var)
        wavelet = self.decode(z)
        return wavelet, mu, log_var  # type: ignore


###################################
# Utils classes
###################################


class Sequential(nn.Module):
    """`nn.Sequential` 的轻量包装器。"""

    def __init__(self, modules: List[nn.Module]) -> None:
        """初始化顺序容器。

        Parameters
        ----------
        modules : list of torch.nn.Module
            按执行顺序组织的子模块列表。
        """
        super().__init__()
        self.layers = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        """执行顺序前向传播。

        Parameters
        ----------
        x : Tensor
            输入张量。

        Returns
        -------
        Tensor
            依次通过内部模块后的输出张量。
        """
        x = self.layers(x)
        return x


class ScalingOperator(nn.Module):
    """可学习幅值缩放算子。"""

    def __init__(self):
        """初始化缩放算子。"""
        super().__init__()

        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """按可学习标量缩放输入。

        Parameters
        ----------
        x : Tensor
            输入张量。

        Returns
        -------
        Tensor
            乘以 `scale` 后的张量。
        """
        x *= self.scale
        return x
