"""wtie.modeling.reflectivity: 反射系数序列生成工具。

本模块提供多种 1D 反射系数（reflectivity）构造器，包括均匀随机、
simplex 噪声驱动、双分量均匀分布以及尖脉冲形式，
并提供随机选择不同构造策略的统一调用封装。

边界说明
--------
- 本模块不负责子波卷积、地震正演、叠加（stacking）或反演优化流程。
- 本模块仅生成反射系数序列本身，采样基准一致性与输入质量控制需在上游完成。

核心公开对象
------------
1. RandomUniformReflectivity: 均匀随机反射系数生成器。
2. RandomSimplexReflectivity: simplex 噪声驱动反射系数生成器。
3. RandomBiUniformReflectivity: 双分量均匀分布反射系数生成器。
4. RandomReflectivityCallable: 多生成器随机调度入口。

Examples
--------
>>> from wtie.modeling.reflectivity import RandomUniformReflectivity
>>> gen = RandomUniformReflectivity(num_samples=512, sparsity_rate=0.7)
>>> r = gen()
>>> r.shape
(512,)
"""

import random

import numpy as np

from wtie.modeling.noise import open_simplex_noise
from wtie.utils.types_ import List, Tuple


class RandomReflectivityCallable:
    """从多个反射系数生成器中随机选择一个并执行。

    该类是组合调度器：每次调用时在候选生成器中等概率随机抽取一个，
    并返回该生成器产生的 1D 反射系数序列。

    Attributes
    ----------
    random_reflectivity_choosers : List[RandomReflectivityChooser]
        可调用对象列表。每个对象应在调用后返回 shape 为 (n_samples,) 的
        numpy.ndarray。
    """

    def __init__(self, random_reflectivity_choosers: List["RandomReflectivityChooser"]):
        """初始化随机反射系数调度器。

        Parameters
        ----------
        random_reflectivity_choosers : List[RandomReflectivityChooser]
            候选反射系数生成器列表。
        """

        self.random_reflectivity_choosers = random_reflectivity_choosers

    def __call__(self):
        """随机选择一个生成器并返回其输出。

        Returns
        -------
        numpy.ndarray
            生成的反射系数序列，shape 为 (n_samples,)。
        """
        return random.choice(self.random_reflectivity_choosers)()  # type: ignore

    def __str__(self):
        """返回候选生成器的类名清单字符串。"""
        s = ""
        for f in self.random_reflectivity_choosers:
            s += f.__class__.__name__ + "\n"
        return s


############################
# Random parameters
############################
class RandomReflectivityChooser:
    """随机反射系数选择器的占位基类。

    Notes
    -----
    当前仅作为类型语义占位，不定义统一接口约束。
    """

    pass


class RandomSpikeReflectivity:
    """生成单个随机尖脉冲反射系数序列。

    每次调用在随机索引处放置一个正幅值尖脉冲，其余位置为 0。

    Attributes
    ----------
    num_samples : int
        输出序列采样点数 n。
    """

    def __init__(self, num_samples: int):
        """初始化随机尖脉冲生成器。

        Parameters
        ----------
        num_samples : int
            输出序列长度 n（采样点数）。
        """
        self.num_samples = num_samples

    def __call__(self):
        """生成随机尖脉冲序列。

        Returns
        -------
        numpy.ndarray
            1D 反射系数序列，shape 为 (n_samples,)。
            非零位置仅有一个样点，幅值范围为 [0.5, 1.0)。
        """
        spike = np.zeros((self.num_samples,))
        idx = np.random.randint(0, self.num_samples)
        amp = np.random.uniform(0.5, 1.0)
        spike[idx] = amp
        return spike


class RandomWeakUniformReflectivityChooser(RandomReflectivityChooser):
    """随机生成弱幅值均匀反射系数序列。

    先按随机稀疏度生成均匀分布反射系数，再乘以随机幅值缩放系数。

    Attributes
    ----------
    num_samples : int
        输出序列采样点数 n。
    sparsity_rate_range : Tuple[float, float]
        稀疏率采样区间，范围建议在 [0, 1]。
    max_amplitude_range : Tuple[float, float]
        幅值缩放系数采样区间。通常使用 (0, 1) 以获得弱反射系数。
    """

    def __init__(
        self, num_samples: int, sparsity_rate_range: Tuple[float, float], max_amplitude_range: Tuple[float, float]
    ):
        """初始化弱幅值均匀反射系数选择器。

        Parameters
        ----------
        num_samples : int
            输出序列长度 n（采样点数）。
        sparsity_rate_range : Tuple[float, float]
            稀疏率随机采样区间。
        max_amplitude_range : Tuple[float, float]
            幅值缩放随机采样区间。
        """

        self.sparsity_rate_range = sparsity_rate_range
        self.max_amplitude_range = max_amplitude_range
        self.num_samples = num_samples

    def __call__(self):
        """按随机参数生成弱幅值均匀反射系数序列。

        Returns
        -------
        numpy.ndarray
            反射系数序列，shape 为 (n_samples,)。
            在基于均匀反射系数结果的基础上做整体缩放。
        """
        # retunrs function, args, kwargs
        sr = np.random.uniform(self.sparsity_rate_range[0], self.sparsity_rate_range[1])
        max_ = np.random.uniform(self.max_amplitude_range[0], self.max_amplitude_range[1])
        ref = RandomUniformReflectivity(self.num_samples, sparsity_rate=sr)()
        ref *= max_  # assumes _max in ]0,1[ and ref is normalized to [-1,1]

        return ref


class RandomUniformReflectivityChooser(RandomReflectivityChooser):
    """按随机稀疏率生成均匀反射系数序列。

    Attributes
    ----------
    num_samples : int
        输出序列采样点数 n。
    sparsity_rate_range : Tuple[float, float]
        稀疏率采样区间，范围建议在 [0, 1]。
    """

    def __init__(
        self,
        num_samples: int,
        sparsity_rate_range: Tuple[float, float],
    ):
        """初始化均匀反射系数选择器。

        Parameters
        ----------
        num_samples : int
            输出序列长度 n（采样点数）。
        sparsity_rate_range : Tuple[float, float]
            稀疏率随机采样区间。
        """

        self.sparsity_rate_range = sparsity_rate_range
        self.num_samples = num_samples

    def __call__(self):
        """按随机稀疏率生成反射系数序列。

        Returns
        -------
        numpy.ndarray
            反射系数序列，shape 为 (n_samples,)。
        """
        # retunrs function, args, kwargs
        sr = np.random.uniform(self.sparsity_rate_range[0], self.sparsity_rate_range[1])
        return RandomUniformReflectivity(self.num_samples, sparsity_rate=sr)()


class RandomSimplexReflectivityChooser(RandomReflectivityChooser):
    """按随机稀疏率与变化尺度生成 simplex 反射系数序列。

    Attributes
    ----------
    num_samples : int
        输出序列采样点数 n。
    sparsity_rate_range : Tuple[float, float]
        稀疏率采样区间，范围建议在 [0, 1]。
    variation_scale_range : Tuple[int, int]
        simplex 噪声变化尺度采样区间。数值越大，序列变化通常越平缓。
    """

    def __init__(
        self, num_samples: int, sparsity_rate_range: Tuple[float, float], variation_scale_range: Tuple[int, int]
    ):
        """初始化 simplex 反射系数选择器。

        Parameters
        ----------
        num_samples : int
            输出序列长度 n（采样点数）。
        sparsity_rate_range : Tuple[float, float]
            稀疏率随机采样区间。
        variation_scale_range : Tuple[int, int]
            变化尺度随机采样区间。
        """

        self.sparsity_rate_range = sparsity_rate_range
        self.variation_scale_range = variation_scale_range
        self.num_samples = num_samples

    def __call__(self):
        """按随机参数生成 simplex 反射系数序列。

        Returns
        -------
        numpy.ndarray
            反射系数序列，shape 为 (n_samples,)。
        """
        # retunrs function, args, kwargs
        sr = np.random.uniform(self.sparsity_rate_range[0], self.sparsity_rate_range[1])
        vs = np.random.uniform(self.variation_scale_range[0], self.variation_scale_range[1])
        return RandomSimplexReflectivity(self.num_samples, sparsity_rate=sr, variation_scale=vs)()


class RandomBiUniformReflectivityChooser(RandomReflectivityChooser):
    """生成双分量均匀反射系数序列的选择器。

    Attributes
    ----------
    num_samples : int
        输出序列采样点数 n。
    """

    def __init__(self, num_samples: int):
        """初始化双分量均匀反射系数选择器。

        Parameters
        ----------
        num_samples : int
            输出序列长度 n（采样点数）。
        """
        self.num_samples = num_samples

    def __call__(self):
        """生成双分量均匀反射系数序列。

        Returns
        -------
        numpy.ndarray
            反射系数序列，shape 为 (n_samples,)。
        """
        return RandomBiUniformReflectivity(self.num_samples)()


############################
# Base reflectivity creation
#############################
class SpikeReflectivity:
    """生成固定位置单位尖脉冲反射系数序列。

    尖脉冲位于序列中心位置，其幅值为 1.0。

    Attributes
    ----------
    num_samples : int
        输出序列采样点数 n。
    spike : numpy.ndarray
        预构建的尖脉冲序列，shape 为 (n_samples,)。
    """

    def __init__(self, num_samples: int):
        """初始化中心尖脉冲反射系数。

        Parameters
        ----------
        num_samples : int
            输出序列长度 n（采样点数）。
        """
        self.num_samples = num_samples

        spike = np.zeros((self.num_samples,))
        spike[spike.size // 2] = 1.0

        self.spike = spike

    def __call__(self):
        """返回中心尖脉冲序列。

        Returns
        -------
        numpy.ndarray
            中心样点为 1.0 的反射系数序列，shape 为 (n_samples,)。
        """
        return self.spike


class RandomUniformReflectivity:
    """生成均匀分布的随机 1D 反射系数序列。

    先生成 [0, 1) 的随机幅值，再按稀疏率置零，最后随机赋予正负号。

    Attributes
    ----------
    n : int
        输出序列采样点数 n。
    sparsity_rate : float
        置零概率，范围为 [0, 1]。
    """

    def __init__(self, num_samples: int, sparsity_rate: float = 0.6):
        """初始化均匀随机反射系数生成器。

        Parameters
        ----------
        num_samples : int
            输出序列长度 n（采样点数）。
        sparsity_rate : float, optional
            稀疏率（置零概率），范围为 [0, 1]。
            值越大，输出中零值比例越高。默认值为 0.6。

        Raises
        ------
        AssertionError
            当 sparsity_rate 不在 [0, 1] 内时抛出。
        """
        assert sparsity_rate >= 0.0 and sparsity_rate <= 1.0
        self.n = num_samples
        self.sparsity_rate = sparsity_rate
        # self.power_stretch = power_stretch

    def __call__(self) -> np.ndarray:
        """生成均匀随机反射系数序列。

        Returns
        -------
        numpy.ndarray
            无量纲反射系数序列，shape 为 (n_samples,)。
            数值范围为 [-1.0, 1.0)；其中 0 的占比由 sparsity_rate 控制。

        Examples
        --------
        >>> gen = RandomUniformReflectivity(num_samples=256, sparsity_rate=0.5)
        >>> r = gen()
        >>> r.shape
        (256,)
        """
        # generate random response in [0, 1)
        reflectivity = np.random.rand(self.n)

        # zeroing
        zeros = np.random.rand(self.n)
        zeros = zeros < self.sparsity_rate
        reflectivity[zeros] = 0.0

        # strech
        # reflectivity **= self.power_stretch

        # adjust sign
        sign = np.random.rand(self.n)
        reflectivity[sign > 0.5] *= -1

        return reflectivity


class RandomSimplexReflectivity:
    """生成基于 simplex 噪声的随机 1D 反射系数序列。

    先构建 simplex 噪声序列并去均值，再按稀疏率置零。
    原始噪声幅值范围依赖 `open_simplex_noise` 实现，当前未在本类中归一化。

    Attributes
    ----------
    n : int
        输出序列采样点数 n。
    sparsity_rate : float
        置零概率，范围为 [0, 1]。
    variation_scale : float
        噪声变化尺度参数。数值越大，序列变化通常越平缓。
    """

    def __init__(self, num_samples: int, sparsity_rate: float = 0.6, variation_scale: float = 150.0):
        """初始化 simplex 随机反射系数生成器。

        Parameters
        ----------
        num_samples : int
            输出序列长度 n（采样点数）。
        sparsity_rate : float, optional
            稀疏率（置零概率），范围为 [0, 1]。默认值为 0.6。
        variation_scale : float, optional
            simplex 噪声变化尺度。默认值为 150.0。

        Raises
        ------
        AssertionError
            当 sparsity_rate 不在 [0, 1] 内时抛出。
        """
        assert sparsity_rate >= 0.0 and sparsity_rate <= 1.0

        self.n = num_samples
        self.sparsity_rate = sparsity_rate
        self.variation_scale = variation_scale

        # self.get_uniform_sparse_series = RandomUniformReflectivity(num_samples=num_samples,
        # sparsity_rate=sparsity_rate)

    def __call__(self) -> np.ndarray:
        """生成 simplex 随机反射系数序列。

        Returns
        -------
        numpy.ndarray
            无量纲反射系数序列，shape 为 (n_samples,)。
            输出已执行去均值和按 sparsity_rate 置零。
            数值范围待确认（取决于 `open_simplex_noise` 的实现细节）。

        Examples
        --------
        >>> gen = RandomSimplexReflectivity(num_samples=256, sparsity_rate=0.7, variation_scale=120.0)
        >>> r = gen()
        >>> r.shape
        (256,)
        """
        # random uniform sparse positive series
        reflectivity = open_simplex_noise(
            size=self.n,
            amplitude_scale=1,
            octaves=4,
            base=None,  # type: ignore
            variation_scale=self.variation_scale,  # type: ignore
        )
        # reflectivity = normalize(reflectivity, a=-1.,b=1.)
        reflectivity -= reflectivity.mean()

        zeros = np.random.rand(self.n)
        zeros = zeros < self.sparsity_rate
        reflectivity[zeros] = 0.0

        return reflectivity


class RandomBiUniformReflectivity:
    """生成双分量均匀分布的随机 1D 反射系数序列。

    分别生成“大反射系数分量”和“小反射系数分量”，按各自稀疏率置零后取逐点最大值，
    最终随机赋予正负号。

    Attributes
    ----------
    n : int
        输出序列采样点数 n。
    sparsity_rate_big : float
        大反射系数分量的置零概率，范围为 [0, 1]。
    sparsity_rate_small : float
        小反射系数分量的置零概率，范围为 [0, 1]。
    big_reflectivity_min : float
        大反射系数分量幅值下限。
    small_reflcetivity_max : float
        小反射系数分量幅值上限。
    """

    def __init__(
        self,
        num_samples: int,
        sparsity_rate_big: float = 0.95,
        sparsity_rate_small: float = 0.5,
        big_reflectivity_min: float = 0.6,
        small_reflcetivity_max: float = 0.2,
    ):
        """初始化双分量均匀反射系数生成器。

        Parameters
        ----------
        num_samples : int
            输出序列长度 n（采样点数）。
        sparsity_rate_big : float, optional
            大反射系数分量的稀疏率（置零概率），范围为 [0, 1]。
            默认值为 0.95。
        sparsity_rate_small : float, optional
            小反射系数分量的稀疏率（置零概率），范围为 [0, 1]。
            默认值为 0.5。
        big_reflectivity_min : float, optional
            大反射系数分量的最小正幅值。默认值为 0.6。
        small_reflcetivity_max : float, optional
            小反射系数分量的最大正幅值。默认值为 0.2。

        Raises
        ------
        AssertionError
            当 `sparsity_rate_big` 或 `sparsity_rate_small` 不在 [0, 1] 内时抛出。
        """

        assert sparsity_rate_big >= 0.0 and sparsity_rate_big <= 1.0
        assert sparsity_rate_small >= 0.0 and sparsity_rate_small <= 1.0

        self.n = num_samples
        self.sparsity_rate_big = sparsity_rate_big
        self.sparsity_rate_small = sparsity_rate_small
        self.big_reflectivity_min = big_reflectivity_min
        self.small_reflcetivity_max = small_reflcetivity_max

    def __call__(self) -> np.ndarray:
        """生成双分量均匀随机反射系数序列。

        Returns
        -------
        numpy.ndarray
            无量纲反射系数序列，shape 为 (n_samples,)。
            正负号随机，绝对值上界受初始化参数约束。

        Examples
        --------
        >>> gen = RandomBiUniformReflectivity(num_samples=128)
        >>> r = gen()
        >>> r.shape
        (128,)
        """
        # generate random response in [0, 1)
        reflectivity_big = np.random.uniform(low=self.big_reflectivity_min, high=1.0, size=(self.n,))
        reflectivity_small = np.random.uniform(low=0, high=self.small_reflcetivity_max, size=(self.n,))

        # zeroing
        zeros = np.random.rand(self.n)
        zeros = zeros < self.sparsity_rate_big
        reflectivity_big[zeros] = 0.0

        zeros = np.random.rand(self.n)
        zeros = zeros < self.sparsity_rate_small
        reflectivity_small[zeros] = 0.0

        # merge
        reflectivity = np.maximum(reflectivity_big, reflectivity_small)

        # adjust sign
        sign = np.random.rand(self.n)
        reflectivity[sign > 0.5] *= -1

        return reflectivity
