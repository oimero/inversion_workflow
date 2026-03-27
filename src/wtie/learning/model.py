"""wtie.learning.model: 训练、轻量超参搜索与推理评估模型。

本模块提供确定性/变分网络训练器、超参数搜索场景下的轻量训练器，
以及基于已训练网络的波子波推理评估接口。

边界说明
--------
- 本模块不负责数据预处理、测井/地震导入导出与质量控制。
- 本模块不定义网络结构细节与损失函数实现，仅负责训练调度与推理封装。

核心公开对象
------------
1. Model: 标准重建网络训练器。
2. VariationalModel: 变分网络训练器。
3. LightModel: 超参数搜索用轻量重建训练器。
4. LightVariationalModel: 超参数搜索用轻量变分训练器。
5. Evaluator: 确定性网络推理器。
6. VariationalEvaluator: 变分网络推理器（期望与采样）。

Examples
--------
>>> from wtie.learning.model import Model, Evaluator
>>> model = Model(save_dir, train_ds, val_ds, parameters, logger)
>>> model.train()
>>> evaluator = Evaluator(model.net, expected_sampling=0.001)
>>> wavelet = evaluator(seismic, reflectivity, squeeze=True)
"""

import pickle
import time
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import torch
import torch.nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from wtie.dataset import BaseDataset, PytorchDataset
from wtie.learning.losses import ReconstructionLoss, VariationalLoss
from wtie.learning.network import Net, VariationalNetwork
from wtie.learning.utils import AlphaScheduler, EarlyStopping
from wtie.utils.logger import Logger
from wtie.utils.types_ import List, _path_t, tensor_or_ndarray


###############################################
# BASE
###############################################
class BaseModel:
    """训练器基类，定义完整训练生命周期与模型持久化接口。

    该类负责组织训练/验证主循环、调度器调用、早停判断、TensorBoard 记录与模型存储。
    子类需实现具体的 `train_one_epoch` 和 `validate_training`。

    Attributes
    ----------
    save_dir : pathlib.Path
        训练输出目录。
    logger : wtie.utils.logger.Logger
        日志记录器。
    tensorboard : torch.utils.tensorboard.SummaryWriter or None
        TensorBoard 记录器，`None` 表示禁用。
    params : dict
        训练参数字典。
    learning_rate : float
        优化器初始学习率。
    batch_size : int
        训练批大小。
    max_epochs : int
        最大训练轮数。
    train_loader : torch.utils.data.DataLoader
        训练数据迭代器。
    val_loader : torch.utils.data.DataLoader
        验证数据迭代器。
    device : torch.device
        计算设备。
    early_stopping : object or None
        早停调度器；未启用时为 `None`。
    schedulers : list
        每个 epoch 调用 `step()` 的调度器列表。
    start_epoch : int
        恢复训练时的起始 epoch。
    current_epoch : int
        当前训练 epoch 计数。
    history : dict
        训练历史记录，由子类初始化并维护。
    """

    # some names
    default_trained_net_state_dict_name = "trained_net_state_dict.pt"

    def __init__(
        self,
        save_dir: _path_t,
        base_train_dataset: BaseDataset,
        base_val_dataset: BaseDataset,
        parameters: dict,
        logger: Logger,
        device: torch.device = None,  # type: ignore
        tensorboard: SummaryWriter = None,  # type: ignore
        save_checkpoints: bool = False,
    ):
        """初始化训练器基类。

        Parameters
        ----------
        save_dir : str or pathlib.Path
            训练结果保存目录，必须已存在。
        base_train_dataset : wtie.dataset.BaseDataset
            训练数据集。
        base_val_dataset : wtie.dataset.BaseDataset
            验证数据集。
        parameters : dict
            训练参数，至少包含 `learning_rate`、`batch_size`、`max_epochs`。
        logger : wtie.utils.logger.Logger
            日志记录器。
        device : torch.device, optional
            计算设备；为 `None` 时自动选择 `cuda:0`（可用时）否则 `cpu`。
        tensorboard : torch.utils.tensorboard.SummaryWriter, optional
            TensorBoard 记录器；为 `None` 时不写入。
        save_checkpoints : bool, default=False
            是否按周期写出 checkpoint。

        Raises
        ------
        AssertionError
            当 `save_dir` 不是已存在目录时抛出。
        KeyError
            当 `parameters` 缺少必需键时抛出。
        """

        self.start_time = time.time()

        # work directory
        self.save_dir = Path(save_dir)
        assert self.save_dir.is_dir()
        self._save_ckpt = save_checkpoints

        # logger and tb
        self.logger = logger
        self.tensorboard = tensorboard

        # parameters
        self.params = parameters
        self.start_epoch = 0
        self.current_epoch = 0
        self.learning_rate = parameters["learning_rate"]
        self.batch_size = parameters["batch_size"]
        self.max_epochs = parameters["max_epochs"]

        # datasets No need to add to member attributes
        # self.base_train_dataset = base_train_dataset
        # self.base_val_dataset = base_val_dataset

        logger.write(("Start training in directory %s") % self.save_dir)

        # dataloaders
        pt_train_dataset = PytorchDataset(base_train_dataset)
        pt_val_dataset = PytorchDataset(base_val_dataset)

        self.num_training_samples = len(base_train_dataset)
        self.num_validation_samples = len(base_val_dataset)

        self.train_loader = torch.utils.data.DataLoader(
            pt_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=min(6, cpu_count()), pin_memory=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            pt_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=min(4, cpu_count()), pin_memory=True
        )

        # net and stuff
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        logger.write("Computing device: %s" % str(self.device))

        # to be orverwriten in child class
        self.early_stopping = None
        self.schedulers: list = []

    def train_one_epoch(self):
        """执行单个 epoch 的训练。

        Notes
        -----
        该方法由子类实现，通常应完成前向、损失计算、反向传播与参数更新。
        """
        raise NotImplementedError()

    def validate_training(self):
        """执行验证并返回当前验证损失。

        Returns
        -------
        float
            当前 epoch 的验证损失标量，供调度器与早停使用。

        Notes
        -----
        该方法由子类实现。
        """
        raise NotImplementedError()

    def train(self):
        """运行完整训练流程。

        训练流程包含：每轮训练、验证、调度器更新、早停判断、可选 checkpoint 写出、
        历史记录持久化与最终网络参数保存。

        Returns
        -------
        None
        """
        _div = self.num_training_samples // self.batch_size
        _remain = int(self.num_training_samples % self.batch_size > 0)
        num_iterations_per_epoch = _div + _remain
        self.logger.write(
            ("Training network for %d epochs (%d iterations per epoch)" % (self.max_epochs, num_iterations_per_epoch))
        )

        is_early_stop = False
        for epoch in tqdm(range(self.start_epoch, self.max_epochs)):
            # training / validation
            self.train_one_epoch()
            current_val_loss = self.validate_training()

            # schedulers
            if self.schedulers:
                for scheduler in self.schedulers:
                    scheduler.step()
            # self.scheduler.step()

            # increment count
            self.current_epoch += 1

            # monitor early stopping critera
            if self.early_stopping is not None:
                is_early_stop = self.early_stopping.step(current_val_loss)
                if is_early_stop:
                    break

            # tb
            if self.tensorboard is not None:
                self.tensorboard.add_scalar("lr", self.scheduler.get_last_lr()[0], self.current_epoch)

            # ckpt
            if self._save_ckpt:
                if (epoch % (self.max_epochs // 4) == 0) or (epoch == self.max_epochs - 1):
                    ckpt_path = self.save_dir / ("ckpt_epoch%s.tar" % str(epoch + 1).zfill(3))
                    self.save_model_ckpt(ckpt_path, epoch)

        if is_early_stop:
            self.logger.write(("Early stopping at epoch %d" % epoch))

        # save
        self.history["elapsed"] = time.time() - self.start_time
        self.save_history()
        self.save_network(self.save_dir / Model.default_trained_net_state_dict_name)

        # tb
        if self.tensorboard:
            self.tensorboard.flush()

    def save_history(self):
        """将训练历史保存为 `history.pkl`。

        Returns
        -------
        None
        """
        with open(self.save_dir / "history.pkl", "wb") as fp:
            pickle.dump(self.history, fp)

    def save_network(self, path):
        """保存网络 `state_dict` 到磁盘。

        Parameters
        ----------
        path : str or pathlib.Path
            输出文件路径，建议使用 `.pt` 后缀。

        Returns
        -------
        None
        """
        self.logger.write("Saving network's state_dict to %s" % path)
        torch.save(self.net.state_dict(), path)

    def restore_network_from_state_dict(self, path):
        """从 `state_dict` 文件恢复网络参数。

        Parameters
        ----------
        path : str or pathlib.Path
            `state_dict` 文件路径。

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            当文件不存在时可能抛出。
        RuntimeError
            当参数键或张量形状与当前网络不匹配时可能抛出。
        """
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def save_model_ckpt(self, path, epoch):
        """保存训练 checkpoint。

        Parameters
        ----------
        path : str or pathlib.Path
            checkpoint 文件路径，建议使用 `.tar` 后缀。
        epoch : int
            当前 epoch（零起始计数）。

        Returns
        -------
        None
        """
        self.logger.write("Saving model's checkpoint to %s" % path)
        torch.save(
            {
                "epoch": epoch,
                "net_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            path,
        )

    def restore_model_from_ckpt(self, ckpt_file):
        """从 checkpoint 恢复网络、优化器与训练状态。

        Parameters
        ----------
        ckpt_file : str or pathlib.Path
            checkpoint 文件路径。

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            当 checkpoint 文件不存在时可能抛出。
        KeyError
            当 checkpoint 缺少必要字段时可能抛出。
        RuntimeError
            当网络或优化器状态与当前实例不兼容时可能抛出。
        """
        checkpoint = torch.load(ckpt_file)
        self.net.load_state_dict(checkpoint["net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        self.current_epoch = checkpoint["epoch"]
        self.loss = checkpoint["loss"]


#########################################
# MODELS
#########################################


class Model(BaseModel):
    """确定性网络训练器。

    该类使用 `ReconstructionLoss` 训练 `Net`，并记录训练/验证阶段的各项损失。

    Attributes
    ----------
    net : wtie.learning.network.Net
        确定性网络实例。
    loss : wtie.learning.losses.ReconstructionLoss
        重建损失对象。
    optimizer : torch.optim.Optimizer
        参数优化器（Adam）。
    schedulers : list
        学习率调度器列表。
    early_stopping : wtie.learning.utils.EarlyStopping
        早停控制器。
    history : dict
        按损失项存储的训练/验证历史。
    """

    # some names
    default_trained_net_state_dict_name = BaseModel.default_trained_net_state_dict_name

    def __init__(
        self,
        save_dir,
        base_train_dataset,
        base_val_dataset,
        parameters,
        logger,
        device=None,
        tensorboard=None,
        save_checkpoints=False,
    ):
        """初始化确定性网络训练器。

        Parameters
        ----------
        save_dir : str or pathlib.Path
            训练输出目录。
        base_train_dataset : wtie.dataset.BaseDataset
            训练数据集。
        base_val_dataset : wtie.dataset.BaseDataset
            验证数据集。
        parameters : dict
            训练参数，需包含网络、优化器、调度与早停配置。
        logger : wtie.utils.logger.Logger
            日志记录器。
        device : torch.device, optional
            计算设备。
        tensorboard : torch.utils.tensorboard.SummaryWriter, optional
            TensorBoard 记录器。
        save_checkpoints : bool, default=False
            是否保存阶段性 checkpoint。
        """

        super().__init__(
            save_dir,
            base_train_dataset,
            base_val_dataset,
            parameters,
            logger,
            device=device,
            tensorboard=tensorboard,
            save_checkpoints=save_checkpoints,
        )

        if parameters["network_kwargs"] is None:
            network_kwargs = {}
        else:
            network_kwargs = parameters["network_kwargs"]

        self.net = Net(
            base_train_dataset.wavelet_size,  # type: ignore
            network_kwargs,
        )
        self.net.to(self.device)

        if self.tensorboard is not None:
            self.tensorboard.add_graph(self.net, next(iter(self.train_loader)).to(self.device))

        self.loss = ReconstructionLoss(parameters)

        self.optimizer = torch.optim.Adam(
            params=self.net.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=False,
            weight_decay=parameters["weight_decay"],
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, parameters["lr_decay_every_n_epoch"], gamma=parameters["lr_decay_rate"]
        )

        self.schedulers = [lr_scheduler]

        self.early_stopping = EarlyStopping(
            min_delta=parameters["min_delta_perc"],
            patience=parameters["patience"],
            min_epochs=int(0.8 * self.max_epochs),
        )

        self.history = {}
        for key in self.loss.key_names:
            self.history["train_loss_" + key] = []
            self.history["val_loss_" + key] = []

    def train_one_epoch(self):
        """执行一个 epoch 的确定性网络训练并写入训练损失历史。

        Returns
        -------
        None
        """
        self.net.train()

        loss_numerics = dict()
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0

        count_loop = 0
        for data_batch in self.train_loader:
            count_loop += 1
            self.optimizer.zero_grad()  # zero the parameter gradients
            data_batch = {k: v.to(self.device) for k, v in data_batch.items()}  # to gpu

            wavelet_output_batch = self.net(seismic=data_batch["seismic"], reflectivity=data_batch["reflectivity"])

            loss = self.loss(data_batch, wavelet_output_batch)
            loss["total"].backward()  # backprop
            self.optimizer.step()  # update params

            for key in self.loss.key_names:
                loss_numerics[key] += loss[key].item()

        for key in self.loss.key_names:
            _avg_numeric_loss = loss_numerics[key] / count_loop
            self.history["train_loss_" + key].append(_avg_numeric_loss)

            if self.tensorboard is not None:
                self.tensorboard.add_scalar("loss/train/" + key, _avg_numeric_loss, self.current_epoch)

    def validate_training(self):
        """在验证集上评估确定性网络损失。

        Returns
        -------
        float
            验证集上 `total` 损失的批均值。
        """
        loss_numerics = dict()
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0

        count_loop = 0
        with torch.no_grad():
            self.net.eval()
            for data_batch in self.val_loader:
                count_loop += 1
                data_batch = {k: v.to(self.device) for k, v in data_batch.items()}
                wavelet_output_batch = self.net(seismic=data_batch["seismic"], reflectivity=data_batch["reflectivity"])
                loss = self.loss(data_batch, wavelet_output_batch)

                for key in self.loss.key_names:
                    loss_numerics[key] += loss[key].item()

        for key in self.loss.key_names:
            _avg_numeric_loss = loss_numerics[key] / count_loop
            self.history["val_loss_" + key].append(_avg_numeric_loss)

            if self.tensorboard is not None:
                self.tensorboard.add_scalar("loss/val/" + key, _avg_numeric_loss, self.current_epoch)

        return loss_numerics["total"] / count_loop


class VariationalModel(BaseModel):
    """变分网络训练器。

    该类使用 `VariationalLoss` 训练 `VariationalNetwork`，并支持 `alpha` 动态调度。

    Attributes
    ----------
    net : wtie.learning.network.VariationalNetwork
        变分网络实例。
    loss : wtie.learning.losses.VariationalLoss
        变分损失对象。
    optimizer : torch.optim.Optimizer
        参数优化器（Adam）。
    alpha_scheduler : wtie.learning.utils.AlphaScheduler
        变分损失权重调度器。
    schedulers : list
        调度器列表，包含学习率与 alpha 调度器。
    history : dict
        按损失项存储的训练/验证历史。
    """

    # some names
    default_trained_net_state_dict_name = BaseModel.default_trained_net_state_dict_name

    def __init__(
        self,
        save_dir,
        base_train_dataset,
        base_val_dataset,
        parameters,
        logger,
        device=None,
        tensorboard=None,
        save_checkpoints=False,
    ):
        """初始化变分网络训练器。

        Parameters
        ----------
        save_dir : str or pathlib.Path
            训练输出目录。
        base_train_dataset : wtie.dataset.BaseDataset
            训练数据集。
        base_val_dataset : wtie.dataset.BaseDataset
            验证数据集。
        parameters : dict
            训练参数，需包含网络、优化器、学习率调度与 alpha 调度配置。
        logger : wtie.utils.logger.Logger
            日志记录器。
        device : torch.device, optional
            计算设备。
        tensorboard : torch.utils.tensorboard.SummaryWriter, optional
            TensorBoard 记录器。
        save_checkpoints : bool, default=False
            是否保存阶段性 checkpoint。
        """

        super().__init__(
            save_dir,
            base_train_dataset,
            base_val_dataset,
            parameters,
            logger,
            device=device,
            tensorboard=tensorboard,
            save_checkpoints=save_checkpoints,
        )

        if parameters["network_kwargs"] is None:
            network_kwargs = {}
        else:
            network_kwargs = parameters["network_kwargs"]

        self.net = VariationalNetwork(
            base_train_dataset.wavelet_size,  # type: ignore
            network_kwargs,
        )
        self.net.to(self.device)

        if self.tensorboard is not None:
            self.tensorboard.add_graph(self.net, next(iter(self.train_loader)).to(self.device))

        self.loss = VariationalLoss(parameters)

        self.optimizer = torch.optim.Adam(
            params=self.net.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=False,
            weight_decay=parameters["weight_decay"],
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, parameters["lr_decay_every_n_epoch"], gamma=parameters["lr_decay_rate"]
        )

        self.alpha_scheduler = AlphaScheduler(
            loss=self.loss,
            alpha_init=parameters["alpha_init"],
            alpha_max=parameters["alpha_max"],
            rate=parameters["alpha_scaling"],
            every_n_epoch=parameters["alpha_epoch_rate"],
        )

        self.schedulers = [lr_scheduler, self.alpha_scheduler]

        # self.early_stopping = EarlyStopping(min_delta=parameters['min_delta_perc'],
        #                                patience=parameters['patience'],
        #                                min_epochs=int(0.8*self.max_epochs))

        self.history = {}
        for key in self.loss.key_names:
            self.history["train_loss_" + key] = []
            self.history["val_loss_" + key] = []

    def train_one_epoch(self):
        """执行一个 epoch 的变分网络训练并写入训练损失历史。

        Returns
        -------
        None
        """
        self.net.train()

        loss_numerics = dict()
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0

        count_loop = 0
        for data_batch in self.train_loader:
            count_loop += 1
            self.optimizer.zero_grad()  # zero the parameter gradients
            data_batch = {k: v.to(self.device) for k, v in data_batch.items()}  # to gpu

            wavelet_batch, mu_batch, log_var_batch = self.net(
                seismic=data_batch["seismic"], reflectivity=data_batch["reflectivity"]
            )

            loss = self.loss(data_batch, wavelet_batch, mu_batch, log_var_batch)
            loss["total"].backward()  # backprop
            self.optimizer.step()  # update params

            for key in self.loss.key_names:
                loss_numerics[key] += loss[key].item()

        for key in self.loss.key_names:
            _avg_numeric_loss = loss_numerics[key] / count_loop
            self.history["train_loss_" + key].append(_avg_numeric_loss)

            if self.tensorboard is not None:
                self.tensorboard.add_scalar("loss/train/" + key, _avg_numeric_loss, self.current_epoch)

    def validate_training(self):
        """在验证集上评估变分网络损失。

        Returns
        -------
        float
            验证集上 `total` 损失的批均值。
        """
        loss_numerics = dict()
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0

        count_loop = 0
        with torch.no_grad():
            self.net.eval()
            for data_batch in self.val_loader:
                count_loop += 1
                data_batch = {k: v.to(self.device) for k, v in data_batch.items()}
                wavelet_batch, mu_batch, log_var_batch = self.net(
                    seismic=data_batch["seismic"], reflectivity=data_batch["reflectivity"]
                )
                loss = self.loss(data_batch, wavelet_batch, mu_batch, log_var_batch)

                for key in self.loss.key_names:
                    loss_numerics[key] += loss[key].item()

        for key in self.loss.key_names:
            _avg_numeric_loss = loss_numerics[key] / count_loop
            self.history["val_loss_" + key].append(_avg_numeric_loss)

            if self.tensorboard is not None:
                self.tensorboard.add_scalar("loss/val/" + key, _avg_numeric_loss, self.current_epoch)

        return loss_numerics["total"] / count_loop


#####################################
# LIGHT WEIGHT FOR HYPER-OTPIM
#####################################
class BaseLightModel:
    """超参数搜索用轻量训练器基类。

    该类保留训练与验证最小闭环，减少日志与持久化开销，适用于批量试验评估。

    Attributes
    ----------
    params : dict
        训练参数字典。
    current_epoch : int
        当前 epoch 计数。
    learning_rate : float
        优化器初始学习率。
    batch_size : int
        训练批大小。
    max_epochs : int
        最大训练轮数。
    train_loader : torch.utils.data.DataLoader
        训练数据迭代器。
    val_loader : torch.utils.data.DataLoader
        验证数据迭代器。
    device : torch.device
        计算设备。
    early_stopping : object or None
        早停控制器；未启用时为 `None`。
    schedulers : list
        每个 epoch 调用 `step()` 的调度器列表。
    """

    def __init__(self, base_train_dataset, base_val_dataset, parameters, device=None):
        """初始化轻量训练器基类。

        Parameters
        ----------
        base_train_dataset : wtie.dataset.BaseDataset
            训练数据集。
        base_val_dataset : wtie.dataset.BaseDataset
            验证数据集。
        parameters : dict
            训练参数，至少包含 `learning_rate`、`batch_size`、`max_epochs`。
        device : torch.device, optional
            计算设备；为 `None` 时自动选择 `cuda:0`（可用时）否则 `cpu`。
        """

        # parameters
        self.params = parameters
        self.current_epoch = 0
        self.learning_rate = parameters["learning_rate"]
        self.batch_size = parameters["batch_size"]
        self.max_epochs = parameters["max_epochs"]

        # dataloaders
        pt_train_dataset = PytorchDataset(base_train_dataset)
        pt_val_dataset = PytorchDataset(base_val_dataset)

        self.num_training_samples = len(base_train_dataset)
        self.num_validation_samples = len(base_val_dataset)

        self.train_loader = torch.utils.data.DataLoader(
            pt_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=min(4, cpu_count()), pin_memory=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            pt_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=min(2, cpu_count()), pin_memory=True
        )

        # net and stuff
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # to overwtite in children class
        self.early_stopping = None
        self.schedulers: list = []

    def train_and_validate(self):
        """执行轻量训练并返回最终验证指标。

        Returns
        -------
        Any
            `validate_training()` 的返回值。
            在 `LightModel` 中为 `(mean, std)`；在 `LightVariationalModel`
            中为指标字典。
        """
        count = 0
        # simple_progressbar(count, self.max_epochs)
        for epoch in range(0, self.max_epochs):
            self.train_one_epoch()
            current_val_loss, _ = self.validate_training()

            # self.lr_scheduler.step()
            if self.schedulers:
                for scheduler in self.schedulers:
                    scheduler.step()

            self.current_epoch += 1
            count += 1

            # simple_progressbar(count, self.max_epochs)
            if self.early_stopping is not None:
                if self.early_stopping.step(current_val_loss):
                    break

        # print("Trained for %d/%d epochs" % (epoch+1, self.max_epochs))
        return self.validate_training()


class LightModel(BaseLightModel):
    """用于超参数搜索的确定性轻量训练器。

    Attributes
    ----------
    net : wtie.learning.network.Net
        确定性网络实例。
    loss : wtie.learning.losses.ReconstructionLoss
        重建损失对象。
    optimizer : torch.optim.Optimizer
        参数优化器（Adam）。
    schedulers : list
        学习率调度器列表。
    early_stopping : wtie.learning.utils.EarlyStopping
        早停控制器。
    """

    def __init__(self, base_train_dataset, base_val_dataset, parameters, device=None):
        """初始化确定性轻量训练器。

        Parameters
        ----------
        base_train_dataset : wtie.dataset.BaseDataset
            训练数据集。
        base_val_dataset : wtie.dataset.BaseDataset
            验证数据集。
        parameters : dict
            训练参数。
        device : torch.device, optional
            计算设备。
        """

        super().__init__(base_train_dataset, base_val_dataset, parameters, device=device)

        if parameters["network_kwargs"] is None:
            network_kwargs = {}
        else:
            network_kwargs = parameters["network_kwargs"]

        self.net = Net(base_train_dataset.wavelet_size, network_kwargs)
        self.net.to(self.device)

        self.loss = ReconstructionLoss(parameters)

        self.optimizer = torch.optim.Adam(
            params=self.net.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=False,
            weight_decay=parameters["weight_decay"],
        )

        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        # optimizer=self.optimizer, mode='min', factor=0.5, patience=5, verbose=True, threshold=1e-4, cooldown=0, min_lr=1e-8)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, parameters["lr_decay_every_n_epoch"], gamma=parameters["lr_decay_rate"]
        )

        self.schedulers = [lr_scheduler]

        self.early_stopping = EarlyStopping(
            min_delta=parameters["min_delta_perc"],
            patience=parameters["patience"],
            min_epochs=int(0.8 * self.max_epochs),
        )

    def train_one_epoch(self):
        """执行一个 epoch 的轻量确定性训练。

        Returns
        -------
        None
        """
        self.net.train()

        for data_batch in self.train_loader:
            self.optimizer.zero_grad()  # zero the parameter gradients
            data_batch = {k: v.to(self.device) for k, v in data_batch.items()}  # to gpu

            wavelet_output_batch = self.net(seismic=data_batch["seismic"], reflectivity=data_batch["reflectivity"])

            loss = self.loss(data_batch, wavelet_output_batch)
            loss["total"].backward()  # backprop
            self.optimizer.step()  # update params

    def validate_training(self):
        """在验证集上计算重建误差均值与标准差。

        Returns
        -------
        tuple of float
            `(mean, std)`，分别对应 `validation_error_mean` 与
            `validation_error_std` 的批均值。
        """
        total_validation_error_mean = 0.0
        total_validation_error_std = 0.0
        count_loop = 0

        with torch.no_grad():
            self.net.eval()
            for data_batch in self.val_loader:
                count_loop += 1
                data_batch = {k: v.to(self.device) for k, v in data_batch.items()}
                wavelet_output_batch = self.net(seismic=data_batch["seismic"], reflectivity=data_batch["reflectivity"])
                loss = self.loss(data_batch, wavelet_output_batch)
                total_validation_error_mean += loss["validation_error_mean"].item()
                total_validation_error_std += loss["validation_error_std"].item()  # ~approximation

        mean = total_validation_error_mean / count_loop
        std = total_validation_error_std / count_loop
        return mean, std


class LightVariationalModel(BaseLightModel):
    """用于超参数搜索的变分轻量训练器。

    Attributes
    ----------
    net : wtie.learning.network.VariationalNetwork
        变分网络实例。
    loss : wtie.learning.losses.VariationalLoss
        变分损失对象。
    optimizer : torch.optim.Optimizer
        参数优化器（Adam）。
    schedulers : list
        学习率调度器列表。
    """

    def __init__(self, base_train_dataset, base_val_dataset, parameters, device=None):
        """初始化变分轻量训练器。

        Parameters
        ----------
        base_train_dataset : wtie.dataset.BaseDataset
            训练数据集。
        base_val_dataset : wtie.dataset.BaseDataset
            验证数据集。
        parameters : dict
            训练参数。
        device : torch.device, optional
            计算设备。
        """

        super().__init__(base_train_dataset, base_val_dataset, parameters, device=device)

        if parameters["network_kwargs"] is None:
            network_kwargs = {}
        else:
            network_kwargs = parameters["network_kwargs"]

        self.net = VariationalNetwork(base_train_dataset.wavelet_size, network_kwargs)
        self.net.to(self.device)

        self.loss = VariationalLoss(parameters)

        self.optimizer = torch.optim.Adam(
            params=self.net.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=False,
            weight_decay=parameters["weight_decay"],
        )

        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        # optimizer=self.optimizer, mode='min', factor=0.5, patience=5, verbose=True, threshold=1e-4, cooldown=0, min_lr=1e-8)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, parameters["lr_decay_every_n_epoch"], gamma=parameters["lr_decay_rate"]
        )
        self.schedulers = [lr_scheduler]
        # self.early_stopping = EarlyStopping(min_delta=parameters['min_delta_perc'],
        # patience=parameters['patience'],
        # min_epochs=int(0.8*self.max_epochs))

    def train_one_epoch(self):
        """执行一个 epoch 的轻量变分训练。

        Returns
        -------
        None
        """
        self.net.train()

        for data_batch in self.train_loader:
            self.optimizer.zero_grad()  # zero the parameter gradients
            data_batch = {k: v.to(self.device) for k, v in data_batch.items()}  # to gpu

            wavelet_batch, mu_batch, log_var_batch = self.net(
                seismic=data_batch["seismic"], reflectivity=data_batch["reflectivity"]
            )

            loss = self.loss(data_batch, wavelet_batch, mu_batch, log_var_batch)
            loss["total"].backward()  # backprop
            self.optimizer.step()  # update params

    def validate_training(self):
        """在验证集上计算变分模型的误差统计。

        Returns
        -------
        dict
            键包含：

            - `reconstruction_error`：`(mean, std)`。
            - `variation_error`：`(mean, std)`。

            两个元组元素均为对应指标在验证批次上的均值。
        """
        # TODO: ugly...

        recon_error_mean = 0.0
        recon_error_std = 0.0
        var_error_mean = 0.0
        var_error_std = 0.0

        count = 0

        return_validation_dict = {}

        with torch.no_grad():
            self.net.eval()
            for data_batch in self.val_loader:
                count += 1
                data_batch = {k: v.to(self.device) for k, v in data_batch.items()}
                wavelet_batch, mu_batch, log_var_batch = self.net(
                    seismic=data_batch["seismic"], reflectivity=data_batch["reflectivity"]
                )
                loss = self.loss(data_batch, wavelet_batch, mu_batch, log_var_batch)

                val_error_dict = loss["validation_error"]

                recon_error_mean += val_error_dict["reconstruction_error"][0].item()
                recon_error_std += val_error_dict["reconstruction_error"][1].item()  # ~approximation

                var_error_mean += val_error_dict["variation_error"][0].item()
                var_error_std += val_error_dict["variation_error"][1].item()  # ~approximation

        return_validation_dict["reconstruction_error"] = (recon_error_mean / count, recon_error_std / count)

        return_validation_dict["variation_error"] = (var_error_mean / count, var_error_std / count)

        ##############
        # TMP
        # print("Reconstruction error (mean, std): ", return_validation_dict['reconstruction_error'])
        # print("Variation error (mean, std): ", return_validation_dict['variation_error'])
        # print("\n")

        # api: https://ax.dev/docs/trial-evaluation.html
        # Dict[Tuple[mean: float, std: float]]
        return return_validation_dict


###############################
# EVALUATORS
###############################


class BaseEvaluator:
    """推理评估基类：根据地震与反射率计算波子波。

    该类管理推理设备与网络参数加载，不负责训练与数据预处理。

    Attributes
    ----------
    device : torch.device
        推理设备。
    expected_sampling : float
        输入地震与反射率的期望采样间隔 `dt`，单位为秒（s）。
    net : torch.nn.Module
        推理网络。
    state_dict : str or None
        网络参数文件路径；`None` 表示使用当前网络初始参数。
    """

    def __init__(
        self,
        network: torch.nn.Module,
        expected_sampling: float,
        state_dict: str = None,  # type: ignore
        device: torch.device = None,  # type: ignore
        verbose: bool = True,
    ):
        """初始化推理评估基类。

        Parameters
        ----------
        network : torch.nn.Module
            推理网络实例。
        expected_sampling : float
            输入反射率与地震道的采样间隔 `dt`，单位为秒（s）。
        state_dict : str, optional
            网络参数文件路径；为 `None` 时不从磁盘加载。
        device : torch.device, optional
            推理设备；为 `None` 时自动选择 `cuda:0`（可用时）否则 `cpu`。
        verbose : bool, default=True
            是否打印参数加载信息。

        Raises
        ------
        FileNotFoundError
            当 `state_dict` 指向的文件不存在时可能抛出。
        RuntimeError
            当参数与网络结构不匹配时可能抛出。
        """

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.expected_sampling = expected_sampling

        self.net = network
        self.net.to(self.device)

        self.state_dict = state_dict
        if state_dict is not None:
            if verbose:
                print("Loading network parameters from %s" % state_dict)
            self.net.load_state_dict(torch.load(state_dict, map_location=self.device))
        else:
            if verbose:
                print("Network initialized randomly.")


class VariationalEvaluator(BaseEvaluator):
    """变分网络推理器，支持期望波子波与随机采样。"""

    def __init__(self, network, expected_sampling, state_dict=None, device=None, verbose=True):
        """初始化变分推理器。

        Parameters
        ----------
        network : torch.nn.Module
            变分网络实例。
        expected_sampling : float
            输入采样间隔 `dt`，单位为秒（s）。
        state_dict : str, optional
            网络参数文件路径。
        device : torch.device, optional
            推理设备。
        verbose : bool, default=True
            是否打印加载信息。
        """

        super().__init__(
            network=network, expected_sampling=expected_sampling, state_dict=state_dict, device=device, verbose=verbose
        )

    def expected_wavelet(
        self,
        seismic: tensor_or_ndarray,
        reflectivity: tensor_or_ndarray,
        squeeze: bool = True,
    ) -> np.ndarray:
        """计算变分网络的期望波子波。

        Parameters
        ----------
        seismic : torch.Tensor or numpy.ndarray
            输入地震数据，shape 与网络输入约定一致（待确认）。
        reflectivity : torch.Tensor or numpy.ndarray
            输入反射率数据，shape 与网络输入约定一致（待确认）。
        squeeze : bool, default=True
            是否对输出执行 `numpy.squeeze`。

        Returns
        -------
        numpy.ndarray
            期望波子波数组；当 `squeeze=True` 时会移除长度为 1 的维度。
        """

        with torch.no_grad():
            self.net.eval()
            if type(seismic) is np.ndarray:
                seismic = torch.from_numpy(seismic)
                reflectivity = torch.from_numpy(reflectivity)

            seismic = seismic.to(self.device)
            reflectivity = reflectivity.to(self.device)
            wavelet = self.net.compute_expected_wavelet(seismic, reflectivity)

            wavelet = wavelet.cpu().data.numpy()

        if squeeze:
            wavelet = np.squeeze(wavelet)

        return wavelet

    def sample(
        self,
        seismic: tensor_or_ndarray,
        reflectivity: tensor_or_ndarray,
        squeeze: bool = True,
    ) -> np.ndarray:
        """从变分网络分布中采样一次波子波。

        Parameters
        ----------
        seismic : torch.Tensor or numpy.ndarray
            输入地震数据，shape 与网络输入约定一致（待确认）。
        reflectivity : torch.Tensor or numpy.ndarray
            输入反射率数据，shape 与网络输入约定一致（待确认）。
        squeeze : bool, default=True
            是否对输出执行 `numpy.squeeze`。

        Returns
        -------
        numpy.ndarray
            单次采样波子波。
        """

        with torch.no_grad():
            self.net.eval()
            if type(seismic) is np.ndarray:
                seismic = torch.from_numpy(seismic)
                reflectivity = torch.from_numpy(reflectivity)

            seismic = seismic.to(self.device)
            reflectivity = reflectivity.to(self.device)
            wavelet = self.net.sample(seismic, reflectivity)

            wavelet = wavelet.cpu().data.numpy()

        if squeeze:
            wavelet = np.squeeze(wavelet)

        return wavelet

    def sample_n_times(
        self,
        seismic: tensor_or_ndarray,
        reflectivity: tensor_or_ndarray,
        n: int,
        squeeze: bool = True,
    ) -> List[np.ndarray]:
        """从变分网络分布中重复采样波子波。

        Parameters
        ----------
        seismic : torch.Tensor or numpy.ndarray
            输入地震数据，shape 与网络输入约定一致（待确认）。
        reflectivity : torch.Tensor or numpy.ndarray
            输入反射率数据，shape 与网络输入约定一致（待确认）。
        n : int
            采样次数，应为正整数。
        squeeze : bool, default=True
            是否对每次采样结果执行 `numpy.squeeze`。

        Returns
        -------
        list of numpy.ndarray
            采样结果列表，长度为 `n`。
        """

        wavelets = []

        with torch.no_grad():
            self.net.eval()
            if type(seismic) is np.ndarray:
                seismic = torch.from_numpy(seismic)
                reflectivity = torch.from_numpy(reflectivity)

            seismic = seismic.to(self.device)
            reflectivity = reflectivity.to(self.device)

            for _ in range(n):
                wavelet_i = self.net.sample(seismic, reflectivity)
                wavelet_i = wavelet_i.cpu().data.numpy()

                if squeeze:
                    wavelet_i = np.squeeze(wavelet_i)

                    wavelets.append(wavelet_i)

        return wavelets


class Evaluator(BaseEvaluator):
    """确定性网络推理器：根据地震与反射率计算波子波。"""

    def __init__(self, network, expected_sampling, state_dict=None, device=None, verbose=True):
        """初始化确定性推理器。

        Parameters
        ----------
        network : torch.nn.Module
            确定性网络实例。
        expected_sampling : float
            输入采样间隔 `dt`，单位为秒（s）。
        state_dict : str, optional
            网络参数文件路径。
        device : torch.device, optional
            推理设备。
        verbose : bool, default=True
            是否打印加载信息。
        """

        super().__init__(
            network=network, expected_sampling=expected_sampling, state_dict=state_dict, device=device, verbose=verbose
        )

    def __call__(
        self,
        seismic: tensor_or_ndarray,
        reflectivity: tensor_or_ndarray,
        squeeze: bool = False,
        scale_factor: float = None,  # type: ignore
    ) -> np.ndarray:
        """执行一次确定性推理并返回波子波。

        Parameters
        ----------
        seismic : torch.Tensor or numpy.ndarray
            输入地震数据，shape 与网络输入约定一致（待确认）。
        reflectivity : torch.Tensor or numpy.ndarray
            输入反射率数据，shape 与网络输入约定一致（待确认）。
        squeeze : bool, default=False
            是否对输出执行 `numpy.squeeze`。
        scale_factor : float, optional
            输出缩放因子；为 `None` 时不缩放。

        Returns
        -------
        numpy.ndarray
            推理得到的波子波数组。
        """

        with torch.no_grad():
            self.net.eval()
            if type(seismic) is np.ndarray:
                seismic = torch.from_numpy(seismic)
                reflectivity = torch.from_numpy(reflectivity)

            seismic = seismic.to(self.device)
            reflectivity = reflectivity.to(self.device)
            wavelet = self.net(seismic, reflectivity)

            wavelet = wavelet.cpu().data.numpy()

        if squeeze:
            wavelet = np.squeeze(wavelet)

        if scale_factor is not None:
            wavelet *= scale_factor

        return wavelet
