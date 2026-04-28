"""wtie.processing.grid: 井震对齐基础网格与轨迹数据结构。

本模块提供 1D 轨迹、叠前道集、测井集合、时深关系表与井轨迹对象，
以及常用的重采样、滤波和坐标域转换工具函数。

边界说明
--------
- 本模块不负责反演优化策略、训练流程或可视化渲染细节。
- 本模块仅处理数据组织与基础数值变换，输入质量控制需在上游完成。

核心公开对象
------------
1. BaseTrace / BasePrestackTrace: 统一轨迹与叠前道集抽象。
2. LogSet: 同井同采样基准测井集合。
3. TimeDepthTable: TWT-深度关系（支持 TVDSS 或 MD 域）。
4. WellPath: 井斜轨迹（MD 与 TVDSS 对应）。
5. integrate_twt_from_depth_velocity / build_local_tdt_from_vp: 从 Vp 构建局部时深关系。
6. convert_log_from_md_to_twt: MD 测井转换到 TWT 的入口函数。

Examples
--------
>>> import numpy as np
>>> from wtie.processing.grid import Log, TimeDepthTable
>>> vp = Log(np.array([2000., 2100., 2200.]), np.array([0., 10., 20.]), "md")
>>> td = TimeDepthTable(twt=np.array([0.0, 0.01, 0.02]), md=np.array([0., 10., 20.]))
>>> vp_twt = vp if vp.is_twt else vp
"""

import warnings
from collections import namedtuple

import numpy as np
import pandas as pd
from multimethod import multimethod
from scipy.interpolate import interp1d as _interp1d

from wtie.processing.sampling import downsample as _downsample
from wtie.processing.spectral import apply_butter_lowpass_filter
from wtie.utils.types_ import Dict, FunctionType, Tuple, Union, _sequence_t

##############################
# CONSTANTS
##############################
TWT_NAME: str = "TWT [s]"
MD_NAME: str = "MD (kb) [m]"
TVDSS_NAME: str = "TVDSS (MSL) [m]"
TVDKB_NAME: str = "TVDKB [m]"
TIMELAG_NAME: str = "Lag [s]"
ZLAG_NAME: str = "Lag [m]"
ANGLE_NAME: str = "Angle [°]"

_NAMES_DICT = {
    "twt": TWT_NAME,
    "md": MD_NAME,
    "tvdss": TVDSS_NAME,
    "tvdkb": TVDKB_NAME,
    "tlag": TIMELAG_NAME,
    "zlag": ZLAG_NAME,
    "angle": ANGLE_NAME,
}

EXISTING_BASIS_TYPES = _NAMES_DICT


def _inverted_name(name):
    return list(_NAMES_DICT.keys())[list(_NAMES_DICT.values()).index(name)]


def integrate_twt_from_depth_velocity(
    depth_m: _sequence_t,
    vp_mps: _sequence_t,
    *,
    origin_twt_s: float = 0.0,
    method: str = "slowness_trapezoid",
) -> np.ndarray:
    """Integrate two-way time from depth samples and interval velocity.

    Parameters
    ----------
    depth_m : _sequence_t
        Strictly increasing depth samples in metres.
    vp_mps : _sequence_t
        P-wave velocity samples in m/s, aligned with ``depth_m``.
    origin_twt_s : float, default=0.0
        TWT assigned to the first depth sample.
    method : {"slowness_trapezoid", "mean_velocity", "right"}, default="slowness_trapezoid"
        Numerical integration method. ``slowness_trapezoid`` integrates
        slowness (1/Vp), ``mean_velocity`` uses the interval average velocity,
        and ``right`` uses the right endpoint velocity.

    Returns
    -------
    np.ndarray
        TWT samples in seconds, same shape as ``depth_m``.
    """
    depth = np.asarray(depth_m, dtype=float)
    vp = np.asarray(vp_mps, dtype=float)
    origin = float(origin_twt_s)

    if depth.ndim != 1 or vp.ndim != 1:
        raise ValueError("depth_m and vp_mps must be one-dimensional arrays.")
    if depth.size != vp.size:
        raise ValueError(f"depth_m and vp_mps must have the same length, got {depth.size} and {vp.size}.")
    if depth.size < 2:
        raise ValueError("Need at least two depth samples to integrate a time-depth table.")
    if not np.isfinite(depth).all():
        raise ValueError("depth_m must contain only finite values.")
    if not np.isfinite(vp).all():
        raise ValueError("vp_mps must contain only finite values.")
    if not np.isfinite(origin):
        raise ValueError("origin_twt_s must be finite.")
    if np.any(vp <= 0.0):
        raise ValueError("vp_mps must be strictly positive.")

    dz = np.diff(depth)
    if np.any(dz <= 0.0):
        raise ValueError("depth_m samples must be strictly increasing.")

    if method == "slowness_trapezoid":
        interval_slowness = 0.5 * (1.0 / vp[:-1] + 1.0 / vp[1:])
        dtwt = 2.0 * dz * interval_slowness
    elif method == "mean_velocity":
        interval_vp = 0.5 * (vp[:-1] + vp[1:])
        dtwt = 2.0 * dz / interval_vp
    elif method == "right":
        dtwt = 2.0 * dz / vp[1:]
    else:
        raise ValueError(
            "method must be one of 'slowness_trapezoid', 'mean_velocity', or 'right', "
            f"got {method!r}."
        )

    return origin + np.concatenate(([0.0], np.cumsum(dtwt)))


def build_local_tdt_from_vp(
    tvdss_m: _sequence_t,
    vp_mps: _sequence_t,
    *,
    md_m: _sequence_t = None,  # type: ignore
    origin_twt_s: float = 0.0,
    method: str = "slowness_trapezoid",
) -> pd.DataFrame:
    """Build a local time-depth table DataFrame from TVDSS and Vp samples."""
    tvdss = np.asarray(tvdss_m, dtype=float)
    vp = np.asarray(vp_mps, dtype=float)

    if tvdss.ndim != 1 or vp.ndim != 1:
        raise ValueError("tvdss_m and vp_mps must be one-dimensional arrays.")
    if tvdss.size != vp.size:
        raise ValueError(f"tvdss_m and vp_mps must have the same length, got {tvdss.size} and {vp.size}.")

    if md_m is None:
        md = None
    else:
        md = np.asarray(md_m, dtype=float)
        if md.ndim != 1:
            raise ValueError("md_m must be a one-dimensional array.")
        if md.size != tvdss.size:
            raise ValueError(f"md_m and tvdss_m must have the same length, got {md.size} and {tvdss.size}.")
        if not np.isfinite(md).all():
            raise ValueError("md_m must contain only finite values.")

    order = np.argsort(tvdss)
    tvdss = tvdss[order]
    vp = vp[order]
    if md is not None:
        md = md[order]

    twt = integrate_twt_from_depth_velocity(tvdss, vp, origin_twt_s=origin_twt_s, method=method)

    data = {"tvdss_m": tvdss, "twt_s": twt, "vp_mps": vp}
    if md is not None:
        data["md_m"] = md
        return pd.DataFrame(data, columns=["md_m", "tvdss_m", "twt_s", "vp_mps"])
    return pd.DataFrame(data, columns=["tvdss_m", "twt_s", "vp_mps"])


##############################
# CLASSES
##############################
class BaseObject:
    """带有统一坐标基准标识的基础对象。

    Parameters
    ----------
    basis_type : str
        坐标基准类型键，取值需来自 EXISTING_BASIS_TYPES。

    Attributes
    ----------
    basis_type : str
        基准显示名，如 TWT [s]、MD (kb) [m]。
    is_twt, is_md, is_tvdss, is_tvdmsl, is_tvdkb, is_tlag, is_zlag : bool
        当前对象是否属于对应坐标域。
    """

    def __init__(self, basis_type):
        self.basis_type = _NAMES_DICT[basis_type]

        # basis boolean
        self.is_twt = self.basis_type == TWT_NAME
        self.is_md = self.basis_type == MD_NAME
        self.is_tvdss = self.basis_type == TVDSS_NAME
        self.is_tvdmsl = self.basis_type == TVDSS_NAME
        self.is_tvdkb = self.basis_type == TVDKB_NAME
        self.is_tlag = self.basis_type == TIMELAG_NAME
        self.is_zlag = self.basis_type == ZLAG_NAME


class BaseTrace(BaseObject):
    """一维轨迹对象，封装振幅序列及其采样坐标。

    该类约定基础 shape 为 (n_samples,)，基准坐标需等采样。
    通常通过子类 Log、Seismic、Wavelet、XCorr 使用，不建议直接实例化。

    Attributes
    ----------
    values : np.ndarray
        振幅值数组，shape 为 (n_samples,)。
    basis : np.ndarray
        采样坐标数组，shape 为 (n_samples,)。单位由 basis_type 决定。
    basis_type : str
        基准名称，例如 TWT [s]、MD (kb) [m]。
    sampling_rate : float
        采样间隔 dt（s 或 m）。
    size : int
        采样点数 n。
    shape : tuple
        轨迹 shape，固定为 (n_samples,)。
    duration : float
        坐标跨度，等于 basis[-1] - basis[0]。
    name : str or None
        轨迹名称。
    """

    def __init__(
        self,
        values: np.ndarray,
        basis: np.ndarray,
        basis_type: str,
        name: str = None,  # type: ignore
        unit: str = None,  # type: ignore
        allow_nan: bool = True,
    ):
        """初始化一维轨迹。

        Parameters
        ----------
        values : np.ndarray
            轨迹振幅值，shape 为 (n_samples,)。
        basis : np.ndarray
            采样坐标值，shape 为 (n_samples,)。
        basis_type : str
            基准类型键，例如 "twt"、"md"、"tvdss"。
            允许值见 EXISTING_BASIS_TYPES，单位需与类型约定一致。
        name : str, optional
            轨迹名称。
        unit : str, optional
            振幅物理单位，如 m/s、g/cc。
        allow_nan : bool, optional
            是否允许 NaN。False 时若检测到 NaN 将触发断言。

        Raises
        ------
        AssertionError
            当 values 或 basis 不是一维、长度不一致、采样不等间隔，
            或 allow_nan=False 且 values 含 NaN 时触发。

        Examples
        --------
        >>> values = np.random.normal(size=(101,))
        >>> basis = 1.2 + np.arange(101) * 0.004
        >>> tr = BaseTrace(values, basis, "twt", name="demo")

        """
        super().__init__(basis_type)

        self._name = name
        self.unit = unit
        self.allow_nan = allow_nan

        self.is_prestack = False

        # verify shape
        assert values.ndim == basis.ndim == 1
        assert values.size == basis.size

        self.series = pd.Series(data=values, name=name, index=pd.Index(data=basis, name=self.basis_type))
        # verify nans
        if not allow_nan:
            assert not np.isnan(values).any()

        # verify constant sampling
        sampling = self.basis[1:] - self.basis[:-1]
        assert np.allclose(sampling, sampling[0], atol=1e-3)

        # geom attributes
        self.sampling_rate = self.basis[1] - self.basis[0]
        self.size = self.basis.size
        self.shape = self.values.shape
        self.duration = self.basis[-1] - self.basis[0]

    def time_slice(self, tmin: float, tmax: float) -> "BaseTrace":
        """按坐标区间截取轨迹。

        Parameters
        ----------
        tmin : float
            截取起点（含邻近采样点）。
        tmax : float
            截取终点（含邻近采样点）。

        Returns
        -------
        BaseTrace
            与原对象同类型的新轨迹，shape 为 (n_samples_slice,)。

        Raises
        ------
        AssertionError
            当 tmin 或 tmax 超出当前 basis 可容忍边界时触发。
        """
        assert tmin >= self.basis[0] - self.sampling_rate / 2
        assert tmax <= self.basis[-1] + self.sampling_rate / 2
        idx_min = np.argmin(np.abs(self.basis - tmin))
        idx_max = np.argmin(np.abs(self.basis - tmax)) + 1
        new_basis = self.basis[idx_min:idx_max]
        new_values = self.values[idx_min:idx_max]
        return type(self)(
            new_values,
            new_basis,
            _inverted_name(self.basis_type),
            name=self.name,
            unit=self.unit,
            allow_nan=self.allow_nan,
        )

    @property
    def values(self) -> np.ndarray:
        """轨迹振幅值数组，shape 为 (n_samples,)。"""
        return self.series.values  # type: ignore

    @property
    def basis(self) -> np.ndarray:
        """轨迹采样坐标数组，shape 为 (n_samples,)。"""
        return self.series.index.values

    @basis.setter
    def basis(self, new_value):
        self.series = self.series.set_axis(new_value, axis="index", copy=False)

    def __len__(self):
        return self.size

    @property
    def name(self):
        """轨迹名称。"""
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name
        self.series.name = new_name

    def __str__(self):
        return str(self.series)

    @multimethod
    def __add__(self, other: "BaseTrace"):  # type: ignore
        assert self.basis_type == other.basis_type
        assert np.allclose(self.basis, other.basis)
        new_values = self.values + other.values
        return type(self)(new_values, self.basis, _inverted_name(self.basis_type))

    @multimethod
    def __add__(self, other: np.ndarray):  # noqa: F811
        new_values = self.values + other
        return type(self)(new_values, self.basis, _inverted_name(self.basis_type))

    def __mul__(self, scalar: float):
        new_values = scalar * self.values
        return type(self)(new_values, self.basis, _inverted_name(self.basis_type), name=self.name)

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    # @multimethod
    # def __radd__(self, other):
    # I can't make it work...


class Log(BaseTrace):
    """测井曲线轨迹对象。"""

    def __init__(self, values, basis, basis_type, **kwargs):
        super().__init__(values, basis, basis_type, **kwargs)


class Reflectivity(BaseTrace):
    """反射系数轨迹。

    Notes
    -----
    构造时 basis_type 参数仅用于接口兼容，内部固定按 TWT 域处理。
    """

    def __init__(self, values, basis, basis_type=None, theta: int = 0, **kwargs):
        # basis_type not used as always assumed twt, there for api compat
        super().__init__(values, basis, "twt", **kwargs)

        # incidence angle in degrees
        self.theta = theta


class Seismic(BaseTrace):
    """地震道轨迹对象，可携带单角度信息。"""

    def __init__(self, values, basis, basis_type, theta: int = 0, **kwargs):
        super().__init__(values, basis, basis_type, **kwargs)

        # incidence angle in degrees
        self.theta = theta

        # for compat with PreStackSeismic
        self.angle_range = None


WaveletUncertainties = namedtuple("WaveletUncertainties", ("ff", "ampl_mean", "ampl_std", "phase_mean", "phase_std"))


class Wavelet(BaseTrace):
    """子波轨迹对象，可携带频谱不确定性描述。"""

    def __init__(
        self,
        values,
        basis,
        basis_type=None,
        theta: int = 0,
        uncertainties: WaveletUncertainties = None,  # type: ignore
        **kwargs,  # type: ignore
    ):
        # basis_type not used as always assumed twt, there for api compat
        super().__init__(values, basis, "twt", **kwargs)

        # incidence angle in degrees
        self.theta = theta

        self.uncertainties_ = uncertainties

    @property
    def uncertainties(self):
        """子波频域不确定性，类型为 WaveletUncertainties 或 None。"""
        return self.uncertainties_

    @uncertainties.setter
    def uncertainties(self, values: WaveletUncertainties):
        self.uncertainties_ = values


class DynamicLag(BaseTrace):
    """动态时差或深差轨迹。"""

    def __init__(self, values, basis, basis_type, **kwargs):
        # basis_type not used as always assumed twt, there for api compat
        super().__init__(values, basis, basis_type, **kwargs)


class XCorr(BaseTrace):
    """归一化互相关曲线。

    Notes
    -----
    仅支持 lag 域基准：时间时差 tlag 或深度时差 zlag。
    """

    def __init__(self, values, basis, basis_type, **kwargs):
        assert (basis_type == _inverted_name(TIMELAG_NAME)) or (basis_type == _inverted_name(ZLAG_NAME))
        super().__init__(values, basis, basis_type, **kwargs)

    @property
    def lag(self) -> float:
        """最大相关系数对应的时差/深差（单位由 basis 决定）。"""
        lag_idx = np.argmax(self.values)
        return self.basis[lag_idx]

    @property
    def R(self) -> float:
        """互相关最大值，通常为无量纲量。"""
        # _max = self.values.max()
        # _min = self.values.min()
        # return _max if (abs(_max) >= abs(_min)) else _min
        return self.values.max()

    @property
    def Rc(self) -> float:
        """零时差（中心采样点）相关值，通常为无量纲量。"""
        return self.values[self.size // 2]


class DynamicXCorr(BaseObject):
    """动态互相关矩阵及其坐标描述。

    Attributes
    ----------
    values : np.ndarray
        动态互相关矩阵，shape 为 (n_traces, n_samples)。
    basis : np.ndarray
        第一维对应的主坐标。
    lags_basis : np.ndarray
        第二维 lag 轴坐标。
    lag_type : str
        lag 坐标名称，TWT 域对应 Lag [s]，深度域对应 Lag [m]。
    """

    def __init__(self, dxcorr, basis, basis_type, name=None):
        super().__init__(basis_type)
        self.name = name

        self.values = dxcorr
        self.basis = basis

        self.sampling_rate = basis[1] - basis[0]
        self.shape = dxcorr.shape

        mid_ = dxcorr.shape[1] // 2
        self.lags_basis = self.sampling_rate * np.arange(-mid_, mid_) + self.sampling_rate

        if self.is_twt:
            self.lag_type = TIMELAG_NAME
        elif self.is_tvdss or self.is_md:
            self.lag_type = ZLAG_NAME
        else:
            raise ValueError


class BasePrestackTrace:
    """叠前道集对象，按入射角组织多个一维轨迹。

    约定仅处理单方位角道集，二维数组 shape 为 (n_traces, n_samples)。

    Attributes
    ----------
    traces : tuple[BaseTrace]
        按角度排序的轨迹集合。
    angles : np.ndarray
        角度序列（度），等间隔递增。
    delta_theta : int
        角度采样间隔。
    angle_range : tuple[int, int, int]
        (起始角, 结束角, 角度间隔)。
    values : np.ndarray
        道集振幅矩阵，shape 为 (n_traces, n_samples)。
    basis : np.ndarray
        共享采样坐标，shape 为 (n_samples,)。
    """

    def __init__(self, traces: Tuple[BaseTrace], name: str = None):  # type: ignore
        """初始化叠前道集。

        Parameters
        ----------
        traces : Tuple[BaseTrace]
            轨迹元组。每条轨迹必须提供唯一的 theta 值，并共享同一 basis。
        name : str, optional
            道集名称。

        Raises
        ------
        AssertionError
            当 basis 不一致、角度非等间隔或非递增时触发。
        """
        self.name = name

        self.is_prestack = True

        # Short cuts
        self.basis_type = traces[0].basis_type
        self._basis = traces[0].basis
        self.sampling_rate = traces[0].sampling_rate
        self.is_md = traces[0].is_md
        self.is_twt = traces[0].is_twt
        self.is_tvdss = traces[0].is_tvdss
        self.is_tvdmsl = traces[0].is_tvdmsl
        self.is_tvdkb = traces[0].is_tvdkb

        # all same basis?
        for trace in traces:
            self._verify_trace(trace)

        # angles
        angles = np.array([trace.theta for trace in traces], dtype=int)
        self._verify_angles(angles)

        self.traces = traces
        self.angles = angles
        self.delta_theta = angles[1] - angles[0]
        self.angle_range = (angles[0], angles[-1], self.delta_theta)
        self.num_angles = angles.size

        self.shape = (len(self.traces), len(self.traces[0]))
        self.trace_shape = self.traces[0].shape
        self.trace_size = self.traces[0].size
        self.num_traces = len(traces)

    def __getitem__(self, theta: int) -> BaseTrace:
        """按角度值选择单条轨迹。

        Parameters
        ----------
        theta : int
            角度值（度）。

        Returns
        -------
        BaseTrace
            对应角度轨迹。

        Raises
        ------
        AssertionError
            当 theta 不在当前道集角度序列中时触发。
        """
        assert theta in self.angles, f"Angle {theta}° not in gather."
        idx = np.where(self.angles == theta)[0].item()
        return self.traces[idx]

    def _verify_trace(self, trace: BaseTrace):
        assert self.basis_type == trace.basis_type
        assert np.allclose(self.basis, trace.basis)

    def _verify_angles(self, angles: np.ndarray):
        dtheta = angles[1] - angles[0]
        assert dtheta > 0
        for i in range(len(angles) - 1):
            assert angles[i + 1] - angles[i] == dtheta, "angle sampling must be constant."

    @property
    def values(self) -> np.ndarray:
        """道集振幅矩阵，shape 为 (n_traces, n_samples)。"""
        # (angles, samples)
        values = np.empty(self.shape, dtype=float)

        for i, ref in enumerate(self.traces):
            values[i, :] = ref.values

        return values

    @property
    def basis(self) -> np.ndarray:  # type: ignore
        return self._basis

    @basis.setter
    def basis(self, new_basis):  # type: ignore
        self._basis = new_basis
        for trace in self.traces:
            trace.basis = new_basis

    @staticmethod
    def decimate_angles(
        trace: "BasePrestackTrace", start_angle: int, end_angle: int, delta_angle: int
    ) -> "BasePrestackTrace":
        """按角度步长抽取子道集。"""
        assert start_angle in trace.angles
        assert end_angle in trace.angles
        assert delta_angle >= trace.delta_theta

        new_angles = range(start_angle, end_angle + delta_angle, delta_angle)

        new_trace = [trace[theta] for theta in new_angles]
        return type(trace)(new_trace, name=trace.name)  # type: ignore

    @staticmethod
    def partial_stacking(ps_trace: "BasePrestackTrace", n: int) -> "BasePrestackTrace":
        """执行局部角度叠加。

        对每个角度道，将其左右各 n 条邻道与自身做平均，输出同角度数的新道集。

        Parameters
        ----------
        ps_trace : BasePrestackTrace
            输入叠前道集。
        n : int
            左右邻道数，必须满足 1 <= n < 角度道数。

        Returns
        -------
        BasePrestackTrace
            叠加后的叠前道集，shape 保持为 (n_traces, n_samples)。
        """
        assert n >= 1
        assert n < ps_trace.angles.size
        num_angles = ps_trace.shape[0]
        new_values = np.zeros_like(ps_trace.values)

        # new values
        for i in range(num_angles):
            count = 0
            new_value = np.zeros_like(ps_trace.traces[0].values)
            for j in range(max(0, i - n), min(num_angles - 1, i + n + 1)):
                new_value += ps_trace.values[j, :]
                count += 1
            new_values[i, :] = new_value / count

        # traces objects
        new_traces = []
        trace_type = type(ps_trace.traces[0])
        trace_basis = ps_trace.traces[0].basis
        trace_basis_type = ps_trace.traces[0].basis_type
        for i, theta in enumerate(ps_trace.angles):
            new_trace = trace_type(
                new_values[i, :],
                trace_basis,
                _inverted_name(trace_basis_type),
                theta=theta,
                name=ps_trace.traces[i].name,
            )

            new_traces.append(new_trace)

        return type(ps_trace)(new_traces, name=ps_trace.name)  # type: ignore

    def __str__(self):
        txt = "Prestack %s traces from %d to %d degrees" % (self.name, self.angles[0], self.angles[-1])
        return txt


class PreStackReflectivity(BasePrestackTrace):
    """叠前反射系数道集。"""

    def __init__(self, reflectivities: Tuple[Reflectivity], name: str = "P-P reflectivity"):
        super().__init__(reflectivities, name=name)


class PreStackSeismic(BasePrestackTrace):
    """叠前地震道集。"""

    def __init__(self, seismics: Tuple[Seismic], name: str = "Angle gather"):
        super().__init__(seismics, name=name)


class PreStackWavelet(BasePrestackTrace):
    """叠前子波道集。"""

    def __init__(self, wavelets: Tuple[Wavelet], name: str = "Prestack wavelet"):
        super().__init__(wavelets, name=name)


class PreStackXCorr(BasePrestackTrace):
    """叠前互相关道集。"""

    def __init__(self, traces: Tuple[XCorr], name: str = "Prestack normalized x-correlation"):
        super().__init__(traces, name=name)
        self.is_tlag = traces[0].is_tlag
        self.is_zlag = traces[0].is_zlag

    @property
    def lag(self) -> np.ndarray:
        """每个角度道的最大相关 lag，shape 为 (n_traces,)。"""
        lag_indices = np.argmax(self.values, axis=-1)
        return np.array([self.basis[idx] for idx in lag_indices])

    @property
    def R(self) -> np.ndarray:
        """每个角度道的最大相关幅值，shape 为 (n_traces,)。"""
        return np.abs(self.values).max(axis=-1)

    @property
    def Rc(self) -> np.ndarray:
        """每个角度道中心 lag 处相关值，shape 为 (n_traces,)。"""
        return self.values[:, self.trace_size // 2]


# Union types
seismic_t = Union[Seismic, PreStackSeismic]
ref_t = Union[Reflectivity, PreStackReflectivity]
wlt_t = Union[Wavelet, PreStackWavelet]
xcorr_t = Union[XCorr, PreStackXCorr]
trace_t = Union[BaseTrace, BasePrestackTrace]


# @dataclass
class LogSet:
    """同井同基准测井集合。

    该对象以字典方式管理多条 Log，要求至少包含 Vp 与 Rho，
    并保证所有曲线共用同一采样基准。Vs 在叠前流程中通常必需。

    Attributes
    ----------
    Vp, Rho, Vs : Log or None
        核心测井对象。
    vp, rho, vs : np.ndarray or None
        对应振幅数组，shape 为 (n_samples,)。
    basis : np.ndarray
        共享采样坐标。
    sampling_rate : float
        采样间隔 dt。
    """

    # The follwoing key convention must be followed
    mandatory_keys = ["Vp", "Rho"]
    optional_keys = ["Vs", "GR", "Cali"]  # , no in use so far

    def __init__(self, logs: Dict[str, Log]):
        """初始化测井集合。

        Parameters
        ----------
        logs : Dict[str, Log]
            键为测井类型名（如 Vp、Rho、Vs），值为对应 Log 对象。
            必须至少包含 mandatory_keys 中定义的键。

        Raises
        ------
        AssertionError
            缺少必需键，或任意 Log 的 basis 与基准类型不一致时触发。
        """
        # logs dict must at least contain the keys 'Vp' and 'Rho'
        for key in LogSet.mandatory_keys:
            assert key in logs.keys()

        # Short cut
        self.basis_type = logs["Vp"].basis_type
        self.basis = logs["Vp"].basis
        self.sampling_rate = logs["Vp"].sampling_rate
        self.is_md = logs["Vp"].is_md
        self.is_twt = logs["Vp"].is_twt
        self.is_tvdss = logs["Vp"].is_tvdss
        self.is_tvdmsl = logs["Vp"].is_tvdmsl
        self.is_tvdkb = logs["Vp"].is_tvdkb

        # Verify basis
        for log in logs.values():
            self._verify_log(log)

        self.Logs = logs

        # Dataframe
        self.df = None
        self._create_or_update_df()

        # Short cuts
        # captial letters for Log object
        self.Vp = logs["Vp"]
        self.Rho = logs["Rho"]
        self.Vs = logs["Vs"] if "Vs" in logs.keys() else None

        # small letter for numpy values
        self.vp = self.Vp.values
        self.rho = self.Rho.values
        self.vs = None if self.Vs is None else self.Vs.values

    def _verify_log(self, log: Log):
        assert self.basis_type == log.basis_type
        assert np.allclose(self.basis, log.basis)

    def _create_or_update_df(self):
        _log_dict = {}
        for name, log in self.Logs.items():
            _log_dict[name] = log.values

        df = pd.DataFrame.from_dict(dict(_log_dict, **{self.basis_type: self.basis}))
        df.set_index(self.basis_type, inplace=True)
        self.df = df

    def __getitem__(self, key: str):
        return self.Logs[key]

    def add_log(self, key: str, log: Log):
        """显式向 LogSet 新增一条曲线。

        Parameters
        ----------
        key : str
            曲线键名，例如 'Vs'、'GR'、'Cali'。
        log : Log
            待插入的曲线对象，需与当前 LogSet 共享同一 basis 与 basis_type。

        Raises
        ------
        AssertionError
            当 key 已存在，或输入曲线与当前 LogSet 基准不一致时触发。
        """
        self._verify_log(log)
        assert key not in self.Logs.keys()
        self.Logs[key] = log
        self._create_or_update_df()
        if key == "Vs":
            self.Vs = log
            self.vs = log.values

    def __setitem__(self, key: str, log: Log):
        """按映射协议插入曲线，等价于 add_log(key, log)。"""
        self.add_log(key, log)

    @property
    def AI(self) -> Log:
        """声阻抗曲线（AI = Vp * Rho）。"""
        ai = self.vp * self.rho
        return Log(ai, self.basis, _inverted_name(self.basis_type), name="AI")

    @property
    def ai(self) -> np.ndarray:
        """声阻抗数组，shape 为 (n_samples,)。"""
        return self.AI.values

    @property
    def Vp_Vs_ratio(self):
        """Vp/Vs 比值曲线。"""
        assert self.vs is not None, "You did not provide a Vs log."
        assert (self.vs > 1e-8).all(), "There are null/negative values in the Vs log."
        ratio = self.vp / self.vs
        return Log(ratio, self.basis, _inverted_name(self.basis_type), name="Vp/Vs")

    @property
    def vp_vs_ratio(self):
        """Vp/Vs 比值数组，shape 为 (n_samples,)。"""
        return self.Vp_Vs_ratio.values

    def __str__(self):
        s_log = "%d logs" % len(self.Logs)
        s_basis = " in %s" % self.basis_type
        s_shape = " of length %d." % self.Vp.size

        return s_log + s_basis + s_shape


class TimeDepthTable:
    """时深关系表，支持 TWT-TVDSS 或 TWT-MD 两种域。

    该类用于保存并插值两程时与深度映射，不负责 checkshot 质量控制。

    Attributes
    ----------
    twt : np.ndarray
        两程时序列，单位 s，shape 为 (n_samples,)。
    depth : np.ndarray
        当前域深度序列（TVDSS 或 MD），单位 m，shape 为 (n_samples,)。
    is_md_domain : bool
        是否为 MD 域。
    size : int
        采样点数 n。
    """

    def __init__(self, twt: _sequence_t, tvdss: _sequence_t = None, md: _sequence_t = None):  # type: ignore
        """初始化时深关系。

        Parameters
        ----------
        twt : _sequence_t
            两程时序列，单位 s。
        tvdss : _sequence_t, optional
            真垂深（海平面基准）序列，单位 m。
        md : _sequence_t, optional
            井深 MD 序列，单位 m。

        Raises
        ------
        ValueError
            同时未提供 tvdss 与 md，或二者同时提供时触发。
        AssertionError
            当 twt 非严格递增，或深度非非降序时触发。
        """
        dtype = float

        # Validate input
        if tvdss is None and md is None:
            raise ValueError("Must provide either 'tvdss' or 'md' parameter.")
        if tvdss is not None and md is not None:
            raise ValueError("Cannot provide both 'tvdss' and 'md' parameters.")

        twt = np.array(twt, dtype=dtype)

        if tvdss is not None:
            depth = np.array(tvdss, dtype=dtype)
            self._depth_type = TVDSS_NAME
            self._is_md_domain = False
            self.table = _create_dataframe((twt, depth), (TWT_NAME, TVDSS_NAME))
        else:
            depth = np.array(md, dtype=dtype)
            self._depth_type = MD_NAME
            self._is_md_domain = True
            self.table = _create_dataframe((twt, depth), (TWT_NAME, MD_NAME))

        # verify series are always (non-strictly) increasing
        assert ((depth[1:] - depth[:-1]) >= 0).all()
        assert ((twt[1:] - twt[:-1]) > 0).all()

    @classmethod
    def from_vp(
        cls,
        vp: Log,
        *,
        wellpath: "WellPath" = None,  # type: ignore
        origin_twt_s: float = 0.0,
        method: str = "slowness_trapezoid",
    ) -> "TimeDepthTable":
        """Construct a TVDSS-domain time-depth table from a Vp log.

        ``vp`` must be sampled in TVDSS or MD. MD logs require ``wellpath`` so
        the velocity samples can be converted to TVDSS before integration.
        """
        if vp.is_tvdss:
            vp_tvdss = vp
        elif vp.is_md:
            if wellpath is None:
                raise ValueError("wellpath is required when vp is in MD domain.")
            vp_tvdss = _convert_log_from_md_to_tvdss(vp, wellpath)
        elif vp.is_twt:
            raise ValueError("TimeDepthTable.from_vp only supports TVDSS or MD Vp logs.")
        else:
            raise NotImplementedError("%s basis type not implemented." % vp.basis_type)

        twt = integrate_twt_from_depth_velocity(
            vp_tvdss.basis,
            vp_tvdss.values,
            origin_twt_s=origin_twt_s,
            method=method,
        )
        return cls(twt=twt, tvdss=vp_tvdss.basis)

    @property
    def twt(self) -> np.ndarray:
        """两程时数组，单位 s，shape 为 (n_samples,)。"""
        return self.table.loc[:, TWT_NAME].values  # type: ignore

    @property
    def tvdss(self) -> np.ndarray:
        """TVDSS 深度数组，单位 m，shape 为 (n_samples,)。"""
        if self._is_md_domain:
            raise ValueError("This TimeDepthTable is in MD domain. Use '.md' property instead.")
        return self.table.loc[:, TVDSS_NAME].values  # type: ignore

    @property
    def md(self) -> np.ndarray:
        """MD 深度数组，单位 m，shape 为 (n_samples,)。"""
        if not self._is_md_domain:
            raise ValueError("This TimeDepthTable is in TVDSS domain. Use '.tvdss' property instead.")
        return self.table.loc[:, MD_NAME].values  # type: ignore

    @property
    def depth(self) -> np.ndarray:
        """通用深度访问器，返回当前域深度数组（TVDSS 或 MD）。"""
        # The depth column is always the second column (index 1)
        return self.table.iloc[:, 1].values  # type: ignore

    @property
    def is_md_domain(self) -> bool:
        """是否使用 MD 域（True 表示 depth 对应 MD）。"""
        return self._is_md_domain

    @property
    def size(self):
        return self.twt.size

    def __len__(self):
        return self.size

    def slope_velocity_twt(self, dt: float = 0.004) -> Log:
        """在 TWT 等采样网格上计算区间速度。

        Parameters
        ----------
        dt : float, default=0.004
            时间采样间隔 dt，单位 s。

        Returns
        -------
        Log
            区间速度曲线，单位 m/s，basis 为 twt[1:]，shape 为 (n_samples-1,)。
        """
        table = self.temporal_interpolation(dt)
        slope = self._compute_slope_from_table(table)
        return Log(slope, table.twt[1:], "twt", name="Slope velocity")

    def slope_velocity_depth(self, dz: float = 5) -> Log:
        """在深度等采样网格上计算区间速度。

        Parameters
        ----------
        dz : float, default=5
            深度采样间隔 dz，单位 m。

        Returns
        -------
        Log
            区间速度曲线，单位 m/s；若当前域为 MD 则 basis_type 为 md，否则为 tvdss。
        """
        table = self.depth_interpolation(dz)
        slope = self._compute_slope_from_table(table)
        domain_name = "md" if self._is_md_domain else "tvdss"
        return Log(slope, table.depth[1:], domain_name, name="Slope velocity")

    def slope_velocity_tvdss(self, dz: float = 5) -> Log:
        """兼容接口：调用 slope_velocity_depth。

        Notes
        -----
        当对象处于 MD 域时会发出警告，但仍返回 depth 域区间速度。
        """
        if self._is_md_domain:
            warnings.warn("Table is in MD domain. Consider using slope_velocity_depth() instead.")
        return self.slope_velocity_depth(dz)

    def _compute_slope_from_table(self, table: "TimeDepthTable") -> np.ndarray:
        depth_seg = table.depth[1:] - table.depth[:-1]
        twt_sampling = table.twt[1:] - table.twt[:-1]
        slope = 2.0 * depth_seg / twt_sampling  # 2 accounts for two-way-time
        return slope

    @staticmethod
    def z_bulk_shift(table: "TimeDepthTable", z: float) -> "TimeDepthTable":
        if table.is_md_domain:
            return TimeDepthTable(table.twt, md=table.md + z)
        return TimeDepthTable(table.twt, tvdss=table.tvdss + z)

    @staticmethod
    def t_bulk_shift(table: "TimeDepthTable", t: float) -> "TimeDepthTable":
        if table.is_md_domain:
            return TimeDepthTable(table.twt + t, md=table.md)
        return TimeDepthTable(table.twt + t, tvdss=table.tvdss)

    def temporal_interpolation(self, dt: float, mode: str = "linear") -> "TimeDepthTable":
        """按给定时间采样间隔插值时深关系。

        Parameters
        ----------
        dt : float
            目标时间采样间隔 dt，单位 s。
        mode : str, default="linear"
            插值模式，透传至 scipy.interpolate.interp1d。

        Returns
        -------
        TimeDepthTable
            新的时深表，twt 为等采样。
        """
        current_twt = self.twt
        current_depth = self.depth

        new_twt = np.arange(current_twt[0], current_twt[-1] + dt, dt)
        # interp = _interp1d(current_twt, current_tvd, kind=mode,
        # bounds_error=False, fill_value=current_tvd[-1])

        interp = _interp1d(current_twt, current_depth, kind=mode, bounds_error=False, fill_value="extrapolate")  # type: ignore

        new_depth = interp(new_twt)

        if self._is_md_domain:
            return TimeDepthTable(new_twt, md=new_depth)
        return TimeDepthTable(new_twt, tvdss=new_depth)

    def extend_to_twt_range(
        self,
        twt_min: float,
        twt_max: float,
        dt: float,
        fit_points: int = 3,
    ) -> "TimeDepthTable":
        """按局部线性趋势将时深表外推到指定 TWT 范围。

        Parameters
        ----------
        twt_min, twt_max : float
            目标覆盖的 TWT 范围，单位 s。
        dt : float
            目标时间采样间隔，单位 s。
        fit_points : int, default=3
            首尾局部线性拟合使用的样点数，至少为 2。

        Returns
        -------
        TimeDepthTable
            覆盖目标时间范围的新时深表。

        Raises
        ------
        ValueError
            当输入范围无效、``dt`` 非正，或样点数不足以拟合趋势时抛出。
        """
        twt_min = float(twt_min)
        twt_max = float(twt_max)
        dt = float(dt)
        fit_points = int(fit_points)

        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        if twt_max < twt_min:
            raise ValueError(f"twt_max must be >= twt_min, got twt_min={twt_min}, twt_max={twt_max}.")
        if fit_points < 2:
            raise ValueError(f"fit_points must be >= 2, got {fit_points}.")
        if self.size < fit_points:
            raise ValueError(
                f"TimeDepthTable must contain at least {fit_points} samples for trend fitting, got {self.size}."
            )

        current_twt = self.twt.astype(float, copy=False)
        current_depth = self.depth.astype(float, copy=False)
        if current_twt[0] <= twt_min and current_twt[-1] >= twt_max:
            if self._is_md_domain:
                return TimeDepthTable(current_twt.copy(), md=current_depth.copy())
            return TimeDepthTable(current_twt.copy(), tvdss=current_depth.copy())

        head_count = 0
        if twt_min < current_twt[0]:
            head_count = int(np.ceil((current_twt[0] - twt_min) / dt))
        tail_count = 0
        if twt_max > current_twt[-1]:
            tail_count = int(np.ceil((twt_max - current_twt[-1]) / dt))

        pieces_twt = []
        pieces_depth = []

        if head_count > 0:
            fit_twt = current_twt[:fit_points]
            fit_depth = current_depth[:fit_points]
            head_slope, head_intercept = np.polyfit(fit_twt, fit_depth, 1)
            head_twt = current_twt[0] - np.arange(head_count, 0, -1, dtype=float) * dt
            head_depth = head_slope * head_twt + head_intercept
            pieces_twt.append(head_twt)
            pieces_depth.append(head_depth)

        pieces_twt.append(current_twt)
        pieces_depth.append(current_depth)

        if tail_count > 0:
            fit_twt = current_twt[-fit_points:]
            fit_depth = current_depth[-fit_points:]
            tail_slope, tail_intercept = np.polyfit(fit_twt, fit_depth, 1)
            tail_twt = current_twt[-1] + np.arange(1, tail_count + 1, dtype=float) * dt
            tail_depth = tail_slope * tail_twt + tail_intercept
            pieces_twt.append(tail_twt)
            pieces_depth.append(tail_depth)

        new_twt = np.concatenate(pieces_twt)
        new_depth = np.concatenate(pieces_depth)
        new_depth = np.maximum.accumulate(new_depth)

        if self._is_md_domain:
            return TimeDepthTable(new_twt, md=new_depth)
        return TimeDepthTable(new_twt, tvdss=new_depth)

    def depth_interpolation(self, dz: float, mode: str = "linear") -> "TimeDepthTable":
        """按给定深度采样间隔插值时深关系。

        Parameters
        ----------
        dz : float
            目标深度采样间隔 dz，单位 m。
        mode : str, default="linear"
            插值模式，透传至 scipy.interpolate.interp1d。

        Returns
        -------
        TimeDepthTable
            新的时深表，depth 为等采样。
        """
        current_twt = self.twt
        current_depth = self.depth

        new_depth = np.arange(current_depth[0], current_depth[-1] + dz, dz)
        # interp = _interp1d(current_tvd, current_twt, kind=mode,
        # bounds_error=False, fill_value=current_twt[-1])

        interp = _interp1d(current_depth, current_twt, kind=mode, bounds_error=False, fill_value="extrapolate")  # type: ignore

        new_twt = interp(new_depth)

        if self._is_md_domain:
            return TimeDepthTable(new_twt, md=new_depth)
        return TimeDepthTable(new_twt, tvdss=new_depth)

    def __str__(self):
        table = self.table
        return "Time-Depth table (%s vs %s) with %d entries." % (table.columns[0], table.columns[1], table.shape[0])

    @staticmethod
    def get_twt_start_from_checkshots(Vp: Log, wp: "WellPath", checkshots: "TimeDepthTable", return_error: bool = True):
        """由 checkshot 估计声波积分曲线的起始 TWT。

        Parameters
        ----------
        Vp : Log
            速度曲线，要求 MD 域。
        wp : WellPath
            井轨迹，用于 MD 到 TVDSS 转换。
        checkshots : TimeDepthTable
            参考时深关系。
        return_error : bool, default=True
            True 时返回 (t_start, z_error)，否则仅返回 t_start。

        Returns
        -------
        float or tuple[float, float]
            起始时间 t_start（s），以及首样点深度匹配误差 z_error（m，可选）。
        """

        # md to tvdss
        assert Vp.is_md
        Vp = _convert_log_from_md_to_tvdss(Vp, wp)

        # resample checkshots at log dz
        checkshots = checkshots.depth_interpolation(Vp.sampling_rate)

        # verify sampling
        z_error = np.abs(checkshots.tvdss - Vp.basis[0]).min()
        assert z_error < Vp.sampling_rate

        # t_start
        idx = np.argmin(np.abs(checkshots.tvdss - Vp.basis[0]))
        t_start = checkshots.twt[idx]

        return (t_start, z_error) if return_error else t_start

    @staticmethod
    def get_tvdss_start_from_checkshots(Vp: Log, checkshots: "TimeDepthTable", return_error: bool = True):
        """由 checkshot 估计声波积分曲线的起始 TVDSS。

        Parameters
        ----------
        Vp : Log
            速度曲线，要求 TWT 域。
        checkshots : TimeDepthTable
            参考时深关系。
        return_error : bool, default=True
            True 时返回 (tvdss_start, t_error)，否则仅返回 tvdss_start。

        Returns
        -------
        float or tuple[float, float]
            起始深度 tvdss_start（m），以及首样点时间匹配误差 t_error（s，可选）。
        """

        # md to tvdss
        assert Vp.is_twt

        # resample checkshots at log dz
        checkshots = checkshots.temporal_interpolation(Vp.sampling_rate)

        # verify sampling
        t_error = np.abs(checkshots.twt - Vp.basis[0]).min()
        assert t_error < Vp.sampling_rate

        # t_start
        idx = np.argmin(np.abs(checkshots.twt - Vp.basis[0]))
        tvdss_start = checkshots.tvdss[idx]

        return (tvdss_start, t_error) if return_error else tvdss_start


class WellPath:
    """井轨迹对象，用于建立 MD 与 TVDSS 的对应关系。

    Attributes
    ----------
    md : np.ndarray
        井深序列，单位 m，shape 为 (n_samples,)。
    tvdss : np.ndarray
        真垂深序列，单位 m，shape 为 (n_samples,)。
    tvdkb : np.ndarray
        井口基准真垂深序列，单位 m，shape 为 (n_samples,)。
    kb : float
        Kelly Bushing 高程，单位 m。
    """

    def __init__(self, md: _sequence_t, kb: float, tvdss: _sequence_t = None):  # type: ignore
        """初始化井轨迹。

        Parameters
        ----------
        md : np.ndarray
            井深序列，单位 m。首元素需接近 0。
        kb : float
            Kelly Bushing 高程，单位 m。
        tvdss : np.ndarray , optional
            真垂深序列，单位 m。未提供时按垂直井处理并使用 tvdss=md-kb。

        Raises
        ------
        AssertionError
            当 md 非严格递增，或起点不接近 0 时触发。
        """

        dtype = float

        md = np.array(md, dtype=dtype)

        # kelly bushing
        self.kb = float(kb)

        if tvdss is None:
            warnings.warn(
                "You did not provide a true vertical depth (SS) series,\
                          the well is therefore assumed to be vertical."
            )
            tvdss = md - self.kb

        # 将该检查滞后到 get_tvdkb_from_inclination
        # assert np.allclose(md[0], 0.0, rtol=1e-3)
        # md is strictly increasing
        assert ((md[1:] - md[:-1]) > 0).all()

        is_going_upward = not ((tvdss[1:] - tvdss[:-1]) >= 0).all()  # type: ignore
        if is_going_upward:
            warnings.warn(
                "Decreasing tvd detected,\
                          this means the well is going upward at some point."
            )

        self.table = _create_dataframe((md, tvdss), (MD_NAME, TVDSS_NAME))  # type: ignore

    def __str__(self):
        return "Well path (MD [m] vs TVDSS [m]) with %d samples." % self.size

    @property
    def size(self):
        """轨迹采样点数 n。"""
        return self.md.size

    def __len__(self):
        return self.size

    @property
    def tvdss(self) -> np.ndarray:
        """TVDSS 序列，单位 m，shape 为 (n_samples,)。"""
        return self.table.loc[:, TVDSS_NAME].values  # type: ignore

    @property
    def tvdkb(self) -> np.ndarray:
        """TVDKB 序列，单位 m，等于 tvdss + kb。"""
        return self.tvdss + self.kb

    @property
    def md(self) -> np.ndarray:
        """MD 序列，单位 m，shape 为 (n_samples,)。"""
        return self.table.loc[:, MD_NAME].values  # type: ignore

    @staticmethod
    def get_tvdkb_from_inclination(md: _sequence_t, inclination: _sequence_t) -> np.ndarray:
        """由井斜角计算 TVDKB 轨迹。

        Parameters
        ----------
        md : _sequence_t
            井深序列，单位 m，长度为 n，且首值需为 0。
        inclination : _sequence_t
            井斜角序列（相对竖直方向），单位度，长度为 n-1。

        Returns
        -------
        np.ndarray
            TVDKB 序列，单位 m，shape 为 (n_samples,)。
        """
        assert md[0] == 0.0, "Deviation survey should start at 0 meters [MD]"
        assert len(inclination) == len(md) - 1
        md = np.array(md, dtype=float)
        alpha = np.deg2rad(np.array(inclination, dtype=float))

        md_segments = md[1:] - md[:-1]
        tvd_segments = md_segments * np.cos(alpha)

        tvd = np.concatenate((np.zeros((1,), dtype=float), np.cumsum(tvd_segments)))

        return tvd

    @staticmethod
    def tvdss_to_tvdkb(tvdss: np.ndarray, kb: float) -> np.ndarray:
        """将 TVDSS 转换为 TVDKB。

        Parameters
        ----------
        tvdss : np.ndarray
            TVDSS 序列，单位 m。
        kb : float
            Kelly Bushing 高程，单位 m。

        Returns
        -------
        np.ndarray
            TVDKB 序列，单位 m。
        """
        return tvdss + kb

    @staticmethod
    def tvdkb_to_tvdss(tvdkb: np.ndarray, kb: float) -> np.ndarray:
        """将 TVDKB 转换为 TVDSS。

        Parameters
        ----------
        tvdkb : np.ndarray
            TVDKB 序列，单位 m。
        kb : float
            Kelly Bushing 高程，单位 m。

        Returns
        -------
        np.ndarray
            TVDSS 序列，单位 m。
        """

        return tvdkb - kb

    def md_interpolation(self, dz: float, mode: str = "linear"):
        """在 MD 轴上对井轨迹重采样。

        Parameters
        ----------
        dz : float
            目标井深采样间隔，单位 m。
        mode : str, default="linear"
            插值模式，透传至 scipy.interpolate.interp1d。

        Returns
        -------
        WellPath
            重采样后的井轨迹对象。
        """
        # new md
        md_start = self.md[0]
        md_end = self.md[-1]

        md_linear_dz = np.arange(md_start, md_end + dz, dz)

        # interpolate tvd
        interp = _interp1d(self.md, self.tvdss, bounds_error=False, fill_value=self.tvdss[-1], kind=mode)

        new_tvd = interp(md_linear_dz)

        return WellPath(md=md_linear_dz, tvdss=new_tvd, kb=self.kb)


##############################
# FUNCTIONS
##############################
def update_trace_values(new_values: np.ndarray, original_trace: BaseTrace) -> BaseTrace:
    """用新振幅数组更新轨迹并保留原坐标与元信息。

    Parameters
    ----------
    new_values : np.ndarray
        新振幅数组，期望 shape 与 original_trace.values 一致。
    original_trace : BaseTrace
        参考轨迹。

    Returns
    -------
    BaseTrace
        与 original_trace 同类型的新对象。
    """
    return type(original_trace)(
        new_values, original_trace.basis, _inverted_name(original_trace.basis_type), name=original_trace.name
    )


def get_matching_traces(
    trace1: Union[BaseTrace, BasePrestackTrace], trace2: Union[BaseTrace, BasePrestackTrace]
) -> Union[Tuple[BaseTrace], Tuple[BasePrestackTrace]]:
    """对齐两条轨迹或两组叠前道集的公共坐标区间。

    Parameters
    ----------
    trace1, trace2 : BaseTrace or BasePrestackTrace
        待对齐对象。二者必须同基准类型、同采样间隔；若为叠前对象，角度序列需一致。

    Returns
    -------
    tuple
        对齐后的 (trace1_match, trace2_match)。

    Raises
    ------
    AssertionError
        当基准类型、采样间隔、叠前角度或对齐后长度不满足要求时触发。
    NotImplementedError
        输入类型组合不受支持时触发。
    """
    # assert same basis
    assert trace1.basis_type == trace2.basis_type
    assert np.allclose(trace1.sampling_rate, trace2.sampling_rate, rtol=1e-4)

    # basis bound
    b_start = max(trace1.basis[0], trace2.basis[0])
    b_end = min(trace1.basis[-1], trace2.basis[-1])

    # indices bound
    idx_start_t1 = np.argmin(np.abs(trace1.basis - b_start)).item()
    idx_start_t2 = np.argmin(np.abs(trace2.basis - b_start)).item()

    idx_end_t1 = np.argmin(np.abs(trace1.basis - b_end)).item() + 1
    idx_end_t2 = np.argmin(np.abs(trace2.basis - b_end)).item() + 1

    # should be okay with small sampling rate rounding errors
    assert (idx_end_t1 - idx_start_t1) == (idx_end_t2 - idx_start_t2)

    # fork depending on type
    if issubclass(type(trace1), BaseTrace) and issubclass(type(trace2), BaseTrace):
        trace1_match = type(trace1)(
            trace1.values[idx_start_t1:idx_end_t1],
            trace1.basis[idx_start_t1:idx_end_t1],
            _inverted_name(trace1.basis_type),  # type: ignore
            name=trace1.name,
        )

        trace2_match = type(trace2)(
            trace2.values[idx_start_t2:idx_end_t2],
            trace2.basis[idx_start_t2:idx_end_t2],
            _inverted_name(trace2.basis_type),  # type: ignore
            name=trace2.name,
        )

    elif issubclass(type(trace1), BasePrestackTrace) and issubclass(type(trace2), BasePrestackTrace):
        assert (trace1.angles == trace2.angles).all()  # type: ignore
        trace1_match = []
        trace2_match = []

        for theta in trace1.angles:  # type: ignore
            trace1_match.append(
                type(trace1[theta])(  # type: ignore
                    trace1[theta].values[idx_start_t1:idx_end_t1],  # type: ignore
                    trace1.basis[idx_start_t1:idx_end_t1],
                    _inverted_name(trace1.basis_type),
                    name=trace1[theta].name,  # type: ignore
                    theta=theta,
                )
            )

            trace2_match.append(
                type(trace2[theta])(  # type: ignore
                    trace2[theta].values[idx_start_t2:idx_end_t2],  # type: ignore
                    trace2.basis[idx_start_t2:idx_end_t2],
                    _inverted_name(trace2.basis_type),
                    name=trace2[theta].name,  # type: ignore
                    theta=theta,
                )
            )

        trace1_match = type(trace1)(trace1_match)  # type: ignore
        trace2_match = type(trace2)(trace2_match)  # type: ignore

    else:
        raise NotImplementedError

    # assert trace1_match.size == trace2_match.size
    # print(trace1_match.basis)
    # print(trace2_match.basis)
    assert np.allclose(trace1_match.basis, trace2_match.basis, rtol=1e-2)
    # DOUBLECHECK IF IT MAKES SENSE IN ALL CASES
    trace2_match.basis = trace1_match.basis
    assert np.allclose(trace1_match.basis, trace2_match.basis, rtol=1e-3)

    return trace1_match, trace2_match  # type: ignore


def _lowpass_filter_trace(trace: BaseTrace, highcut: float, order: int = 6) -> BaseTrace:
    """对单条轨迹执行零相位巴特沃斯低通滤波。"""
    fs = 1 / trace.sampling_rate
    fN = fs / 2.0
    assert highcut < fN

    low_signal = apply_butter_lowpass_filter(trace.values, highcut, fs, order=order, zero_phase=True)
    return type(trace)(
        values=low_signal, basis=trace.basis, basis_type=_inverted_name(trace.basis_type), name=trace.name
    )


def lowpass_filter_trace(
    trace: Union[BaseTrace, BasePrestackTrace], highcut: float, order: int = 6
) -> Union[BaseTrace, BasePrestackTrace]:
    """对单道或叠前道集执行低通滤波。

    Parameters
    ----------
    trace : BaseTrace or BasePrestackTrace
        输入轨迹对象。
    highcut : float
        截止频率，单位 Hz，要求小于 Nyquist 频率。
    order : int, default=5
        巴特沃斯滤波器阶数。

    Returns
    -------
    BaseTrace or BasePrestackTrace
        滤波后的同类型对象。
    """
    if issubclass(type(trace), BaseTrace):
        return _lowpass_filter_trace(trace, highcut, order=order)  # type: ignore
    elif issubclass(type(trace), BasePrestackTrace):
        return _apply_trace_process_to_prestack_trace(_lowpass_filter_trace, trace, highcut, order=order)  # type: ignore
    else:
        raise NotImplementedError


def _apply_trace_process_to_prestack_trace(
    process: FunctionType, trace: BasePrestackTrace, *args, **kwargs
) -> BasePrestackTrace:
    separate_traces = []
    for theta in trace.angles:
        that_tace = process(trace[theta], *args, **kwargs)
        that_tace.theta = theta
        separate_traces.append(that_tace)

    return type(trace)(separate_traces)  # type: ignore


def _apply_trace_process_to_logset(process: FunctionType, logset: LogSet, *args, **kwargs) -> LogSet:
    new_logs = {}
    for name, log in logset.Logs.items():
        _log = process(log, *args, **kwargs)
        new_logs[name] = _log

    return LogSet(new_logs)


def lowpass_filter_logset(logset: LogSet, highcut: float, order: int = 6) -> LogSet:
    """对 LogSet 中每条曲线执行低通滤波。"""
    return _apply_trace_process_to_logset(lowpass_filter_trace, logset, highcut, order=order)


def downsample_trace(trace: BaseTrace, new_sampling: float) -> BaseTrace:
    """对单条轨迹降采样。

    Parameters
    ----------
    trace : BaseTrace
        输入轨迹。
    new_sampling : float
        新采样间隔，必须大于当前采样间隔。

    Returns
    -------
    BaseTrace
        降采样后的同类型轨迹。
    """
    assert new_sampling > trace.sampling_rate
    # lowpass and decimate
    div_factor = int(round(new_sampling / trace.sampling_rate))
    # signal_resamp = _decimate(trace.values, div_factor)
    # correct for DC bias
    # signal_resamp = signal_resamp - signal_resamp.mean() + trace.values.mean()
    signal_resamp = _downsample(trace.values, div_factor)

    basis_resamp = trace.basis[::div_factor]

    return type(trace)(
        values=signal_resamp, basis=basis_resamp, basis_type=_inverted_name(trace.basis_type), name=trace.name
    )


def downsample_logset(logset: LogSet, new_sampling: float) -> LogSet:
    """对 LogSet 中每条曲线执行降采样。"""
    return _apply_trace_process_to_logset(downsample_trace, logset, new_sampling)


def resample_logset(logset: LogSet, new_sampling: float) -> LogSet:
    """对 LogSet 中每条曲线执行重采样。"""
    return _apply_trace_process_to_logset(resample_trace, logset, new_sampling)


def resample_trace(
    trace: Union[BaseTrace, BasePrestackTrace], new_sampling: float
) -> Union[BaseTrace, BasePrestackTrace]:
    """按目标采样间隔重采样轨迹或叠前道集。"""
    if issubclass(type(trace), BaseTrace):
        return _resample_trace(trace, new_sampling)  # type: ignore
    elif issubclass(type(trace), BasePrestackTrace):
        return _apply_trace_process_to_prestack_trace(_resample_trace, trace, new_sampling)  # type: ignore
    else:
        raise NotImplementedError


def _resample_trace(trace: BaseTrace, new_sampling: float) -> BaseTrace:
    sr = trace.sampling_rate

    if new_sampling < sr:
        return upsample_trace(trace, new_sampling)
    elif new_sampling > sr:
        return downsample_trace(trace, new_sampling)
    else:
        return trace


def upsample_trace(trace: BaseTrace, new_sampling: float) -> BaseTrace:
    """使用 sinc 插值对轨迹升采样。

    Parameters
    ----------
    trace : BaseTrace
        输入轨迹。
    new_sampling : float
        新采样间隔，必须小于当前采样间隔。

    Returns
    -------
    BaseTrace
        升采样后的同类型轨迹。
    """
    # Find the period
    assert new_sampling < trace.sampling_rate
    current_sr = trace.sampling_rate

    new_basis = np.arange(trace.basis[0], trace.basis[-1], step=new_sampling)
    # new_length = int(len(trace) * new_sampling / current_sr)

    sincM = np.tile(new_basis, (len(trace), 1)) - np.tile(trace.basis[:, np.newaxis], (1, len(new_basis)))
    new_signal = np.dot(trace.values, np.sinc(sincM / current_sr))
    return type(trace)(values=new_signal, basis=new_basis, basis_type=_inverted_name(trace.basis_type), name=trace.name)


def _convert_log_from_md_to_tvdss(
    log: Log,
    trajectory: WellPath,
    dz: float = None,  # type: ignore
    interpolation: str = "linear",  # type: ignore
) -> Log:
    """将 MD 域测井转换到 TVDSS 域，并按线性深度采样重采样。

    Parameters
    ----------
    log : Log
        输入测井，必须在 MD 域。
    trajectory : WellPath
        井轨迹对象。
    dz : float, optional
        目标 TVDSS 采样间隔，单位 m。None 时沿用 log.sampling_rate。
    interpolation : str, default="linear"
        插值模式，透传至 scipy.interpolate.interp1d。

    Returns
    -------
    Log
        TVDSS 域测井。
    """
    # input log
    assert log.is_md

    # interpolate trajectory at same log sampling
    trajectory_at_log_dz = trajectory.md_interpolation(log.sampling_rate)

    # current tvd
    idx_start = np.argmin(np.abs(trajectory_at_log_dz.md - log.basis[0])).item()
    if idx_start + len(log) >= len(trajectory_at_log_dz):
        warnings.warn(
            "Truncating log as the well path information does not reach\
                      the maximum depth."
        )
        max_idx = idx_start + (len(trajectory_at_log_dz) - idx_start)
        log = Log(
            log.values[: (len(trajectory_at_log_dz) - idx_start)],
            log.basis[: (len(trajectory_at_log_dz) - idx_start)],
            basis_type=_inverted_name(log.basis_type),
            name=log.name,
        )

    else:
        max_idx = idx_start + len(log)

    # assert idx_start + len(log) < len(trajectory_at_log_dz)
    current_tvd = trajectory_at_log_dz.tvdss[idx_start:max_idx]

    # interpolate to linear tvd
    if dz is None:
        # keep the same sampling rate
        dz = log.sampling_rate
    linear_tvd = np.arange(current_tvd[0], current_tvd[-1] + dz, dz)

    interp = _interp1d(current_tvd, log.values, bounds_error=False, fill_value=log.values[-1], kind=interpolation)

    values_at_tvd_dz = interp(linear_tvd)

    return Log(values_at_tvd_dz, linear_tvd, "tvdss", name=log.name, allow_nan=log.allow_nan)


def convert_log_from_md_to_tvdss_to_twt(
    log: Log, table: TimeDepthTable, trajectory: WellPath, dt: float, interpolation: str = "linear"
) -> Log:
    """通过 MD->TVDSS->TWT 路径将测井转换到 TWT 域。

    Parameters
    ----------
    log : Log
        输入测井，必须在 MD 域。
    table : TimeDepthTable
        时深关系表，当前实现按 TVDSS 访问。
    trajectory : WellPath
        井轨迹，用于 MD 到 TVDSS 映射。
    dt : float
        输出时间采样间隔 dt，单位 s。
    interpolation : str, default="linear"
        插值模式，透传至 scipy.interpolate.interp1d。

    Returns
    -------
    Log
        TWT 域测井。
    """
    # input log
    assert log.is_md
    dz = log.sampling_rate

    # md to tvd
    log = _convert_log_from_md_to_tvdss(log, trajectory, dz=dz, interpolation=interpolation)
    assert log.is_tvdss
    start_z = log.basis[0]  # tvd

    # interpolate t-d table at dz
    table_at_dz = table.depth_interpolation(dz)
    # max_table_tvdss = table_at_dz.tvdss[-1]

    # find equivalent twt
    idx_start = np.argmin(np.abs(table_at_dz.tvdss - start_z)).item()

    # truncate log if longer than table relationship
    if idx_start + len(log) >= len(table_at_dz):
        warnings.warn(
            "Truncating log as the time-depth table does not reach\
                      the maximum depth."
        )
        max_idx = len(table_at_dz) - idx_start
        log = Log(log.values[:max_idx], log.basis[:max_idx], _inverted_name(log.basis_type), name=log.name)

    log_twt = table_at_dz.twt[idx_start : idx_start + len(log)]

    # interpolate to regular dt
    linear_twt = np.arange(log_twt[0], log_twt[-1] + dt, dt)

    interp = _interp1d(log_twt, log.values, bounds_error=False, fill_value=log.values[-1], kind=interpolation)
    values_at_dt = interp(linear_twt)

    return Log(values_at_dt, linear_twt, "twt", name=log.name, allow_nan=log.allow_nan)


def convert_log_from_md_to_twt(
    log: Log, table: TimeDepthTable, trajectory: WellPath, dt: float, interpolation: str = "linear"
) -> Log:
    """将 MD 域测井转换到 TWT 域。

    当 trajectory 不为 None 时，函数会发出警告并自动切换为
    MD->TVDSS->TWT 路径；否则使用 MD 域时深表直接转换。

    Parameters
    ----------
    log : Log
        输入测井，必须为 MD 域。
    table : TimeDepthTable
        时深关系表。trajectory 为 None 时必须是 MD 域。
    trajectory : WellPath
        井轨迹。提供后将触发间接转换路径。
    dt : float
        输出时间采样间隔 dt，单位 s。
    interpolation : str, default="linear"
        插值模式，透传至 scipy.interpolate.interp1d。

    Returns
    -------
    Log
        TWT 域测井。

    Raises
    ------
    AssertionError
        当输入 log 不是 MD 域，或无 trajectory 时 table 不是 MD 域。
    """
    # Redirect if trajectory is present
    if trajectory is not None:
        warnings.warn("Well trajectory provided. Switching to MD -> TVDSS -> TWT conversion mode.")
        return convert_log_from_md_to_tvdss_to_twt(log, table, trajectory, dt, interpolation)

    # Direct MD -> TWT conversion logic
    assert log.is_md, "Input log must be in Measured Depth (MD)."
    assert table.is_md_domain, "TimeDepthTable must be in MD domain when no trajectory is provided."

    dz = log.sampling_rate

    # Interpolate table to log sampling rate (MD domain)
    table_at_dz = table.depth_interpolation(dz)

    log_md = np.asarray(log.basis, dtype=float)
    table_md = np.asarray(table_at_dz.depth, dtype=float)
    overlap_min = max(float(log_md[0]), float(table_md[0]))
    overlap_max = min(float(log_md[-1]), float(table_md[-1]))
    if overlap_max < overlap_min:
        raise ValueError("Log MD range and time-depth table MD range do not overlap.")

    overlap_mask = (table_md >= overlap_min) & (table_md <= overlap_max)
    if np.count_nonzero(overlap_mask) < 2:
        raise ValueError("Insufficient overlapping MD samples to convert log from MD to TWT.")

    if overlap_min > float(log_md[0]):
        warnings.warn("Clipping log start to match the time-depth table (MD) minimum depth.")
    if overlap_max < float(log_md[-1]):
        warnings.warn("Truncating log as the time-depth table (MD) does not reach the maximum depth.")

    overlap_md = table_md[overlap_mask]
    overlap_twt = np.asarray(table_at_dz.twt, dtype=float)[overlap_mask]
    overlap_values = np.interp(overlap_md, log_md, np.asarray(log.values, dtype=float))

    # Interpolate to regular TWT sampling (dt)
    linear_twt = np.arange(overlap_twt[0], overlap_twt[-1] + dt, dt)

    interp = _interp1d(
        overlap_twt,
        overlap_values,
        bounds_error=False,
        fill_value=(overlap_values[0], overlap_values[-1]),
        kind=interpolation,
    )
    values_at_dt = interp(linear_twt)

    return Log(values_at_dt, linear_twt, "twt", name=log.name, allow_nan=log.allow_nan)


################################
# UTILS
################################
def _create_dataframe(arrays: Tuple[np.ndarray], names: Tuple[str]) -> pd.DataFrame:
    _table = np.stack(arrays, axis=-1)
    return pd.DataFrame(_table, columns=names)
