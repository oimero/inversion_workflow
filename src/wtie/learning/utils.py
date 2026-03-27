"""wtie.learning.utils: 训练阶段调度与监控工具。

本模块提供早停控制、指数滑动平均以及变分损失权重 alpha 调度器，
用于训练循环中的状态管理与超参数动态更新。

边界说明
--------
- 本模块不负责网络前向计算与损失值定义。
- 本模块不负责优化器与学习率调度器实现。

核心公开对象
------------
1. EarlyStopping: 基于监控指标的早停判定器。
2. RunningAverage: 指数滑动平均器。
3. AlphaScheduler: 变分损失 alpha 权重调度器。

Examples
--------
>>> stopper = EarlyStopping(mode="min", patience=5, min_epochs=20)
>>> should_stop = stopper.step(0.123)
>>> should_stop
False
"""

import math
import warnings

import numpy as np

from wtie.learning.losses import VariationalLoss


class EarlyStopping:
    """早停控制器。

    根据监控指标是否持续恶化来决定是否终止训练，可选对指标先做指数滑动平均。
    实现基于公开 gist 改写。

    References
    ----------
    - https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d

    Attributes
    ----------
    mode : str
        优化方向，`min` 表示越小越好，`max` 表示越大越好。
    min_delta : float
        改善阈值；含义由 `percentage` 决定。
    patience : int
        允许连续未改善的最大 epoch 数。
    best : float or None
        历史最优指标。
    current_epoch : int
        当前累计 epoch 计数。
    min_epochs : int or None
        允许触发早停的最小 epoch。
    num_bad_epochs : int
        连续未改善计数。
    is_better : callable
        指标比较函数。
    averager : RunningAverage or None
        指标平滑器；当 `alpha is None` 时为 `None`。
    """

    def __init__(
        self,
        mode: str = "min",
        min_delta: float = 0,
        patience: int = 10,
        percentage: bool = True,
        min_epochs: int = None,  # type: ignore
        alpha: float = 0.1,
    ):
        """初始化早停控制器。

        Parameters
        ----------
        mode : str, default="min"
            指标优化方向，取值为 `min` 或 `max`。
        min_delta : float, default=0
            指标改善阈值。
        patience : int, default=10
            连续未改善容忍轮数。
        percentage : bool, default=True
            为 True 时，`min_delta` 按百分比解释；否则按绝对值解释。
        min_epochs : int, optional
            触发早停前的最小训练轮数。
        alpha : float, default=0.1
            指标平滑系数，范围应在 (0, 1)；为 None 时不平滑。

        Raises
        ------
        ValueError
            当 `mode` 非 `min`/`max`，或 `alpha` 超出 (0, 1)（由 RunningAverage 抛出）时。
        """
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.current_epoch = 0
        self.min_epochs = min_epochs
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if alpha is not None:
            self.averager = RunningAverage(alpha=alpha)
        else:
            self.averager = None

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics: float) -> bool:
        """输入当前指标并返回是否应停止训练。

        Parameters
        ----------
        metrics : float
            当前 epoch 的监控指标。

        Returns
        -------
        bool
            `True` 表示应触发早停，`False` 表示继续训练。

        Raises
        ------
        TypeError
            当 `min_epochs` 为 None 时，比较 `current_epoch < min_epochs` 可能抛出。
        """
        self.current_epoch += 1

        if self.averager is not None:
            metrics = self.averager(metrics)

        if self.best is None:
            self.best = metrics
            return False

        if self.current_epoch < self.min_epochs:
            return False

        if np.isnan(metrics):
            warnings.warn("Early stopping got NAN as metric.")
            return True

        if self.is_better(metrics, self.best):  # type: ignore
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        """初始化指标比较函数。

        Parameters
        ----------
        mode : str
            `min` 或 `max`。
        min_delta : float
            改善阈值。
        percentage : bool
            是否按百分比解释阈值。

        Returns
        -------
        None

        Raises
        ------
        ValueError
            当 `mode` 非 `min`/`max` 时抛出。
        """
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


class RunningAverage:
    """指数滑动平均器。"""

    def __init__(self, alpha: float):
        """初始化滑动平均器。

        Parameters
        ----------
        alpha : float
            平滑系数，取值范围为 (0, 1)。

        Raises
        ------
        ValueError
            当 `alpha` 不在 (0, 1) 范围内时抛出。
        """
        if not 0 < alpha < 1:
            raise ValueError("out of range, alpha=%f" % alpha)
        self.alpha = alpha
        self.x_old = None

    def __call__(self, x: float) -> float:
        """更新并返回滑动平均值。

        Parameters
        ----------
        x : float
            当前观测值。

        Returns
        -------
        float
            更新后的平滑值；首次调用返回输入值本身。
        """
        if self.x_old is None:
            self.x_old = x
            return x

        x_new = self.x_old + (x - self.x_old) * self.alpha
        self.x_old = x_new
        return x_new


class AlphaScheduler:
    """变分损失权重 alpha 调度器。

    调度规则为按固定 epoch 间隔进行指数缩放，并对上限 `alpha_max` 做截断。

    Attributes
    ----------
    loss : wtie.learning.losses.VariationalLoss
        需要被同步更新 `alpha` 的损失对象。
    alpha_init : float
        初始权重。
    alpha_max : float
        权重上限。
    rate : float
        每个调度周期的乘法倍率。
    every_n_epoch : int
        调度周期（单位：epoch）。
    current_epoch : int
        当前 epoch 计数（初始为 1）。
    """

    def __init__(self, loss: VariationalLoss, alpha_init: float, alpha_max: float, rate: float, every_n_epoch: int):
        """初始化 alpha 调度器。

        Parameters
        ----------
        loss : VariationalLoss
            目标变分损失对象。
        alpha_init : float
            初始 alpha。
        alpha_max : float
            alpha 上限。
        rate : float
            每个周期的倍率因子。
        every_n_epoch : int
            每隔多少个 epoch 更新一次倍率指数。
        """

        self.loss = loss

        self.alpha_init = alpha_init
        self.alpha_max = alpha_max
        self.rate = rate
        self.every_n_epoch = every_n_epoch

        self.current_epoch = 1

    @property
    def alpha(self):
        """float: 按当前 epoch 计算得到的 alpha 值（不超过 alpha_max）。"""
        current_rate = math.pow(self.rate, math.floor(self.current_epoch / self.every_n_epoch))
        return min(self.alpha_init * current_rate, self.alpha_max)

    def step(self):
        """推进一个 epoch 并同步更新损失对象的 alpha。

        Returns
        -------
        None
        """
        self.current_epoch = self.current_epoch + 1
        # update internal state of loss class
        # valid outside of this class as well
        self.loss.alpha = self.alpha
