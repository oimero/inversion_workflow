# Synthoseis-lite 科学核心与 Benchmark 外壳重构规格

> 实施状态（2026-07-16）：阶段 0–5 已完成。科学 truth、投影、base builder、LFM、
> base/variant Writer 已进入共享 Module；时间与深度通过各自 ForwardAdapter 接入；五个旧
> Module 已删除。删除后的完整 pytest 为 36 passed，`compileall`、深度 CUDA debug
> generation 和严格 depth reader 均通过。

## 1. 目的

Synthoseis-lite 同时包含地下真值生成和 Benchmark 数据产品构建。当前目录区分
`core`、`time`、`depth`、`readers` 与 `reporting`：共享 Module 负责科学模型、模型网格
投影、LFM、Benchmark builder 和 Writer，域 Module 负责轴准备、正演与运行编排。

本重构建立三个明确的 Seam：

1. 共享 field-conditioned 科学核心生成 `SyntheticTruth`；
2. 共享 Benchmark builder 调用冻结的投影 policy 和域 ForwardAdapter，把真值物化为
   `BenchmarkSample`；
3. 共享 Writer 把已经完成数值构建的 base sample 和 seismic variant 写成固定 v4
   artifact。

时间域和深度域是同一 field-conditioned 科学模型与同一 Benchmark 合同的两个真实
Adapter。共享 Module 因此具有实际 Leverage。投影边缘规则、正演、子波、采样轴来源和
工区输入仍按当前数值行为保持 Locality。

## 2. 固定决策

- `synthoseis_lite_v4` 的 HDF5 字段树、manifest、sample index 和 reader 合同保持不变。
- 第一轮只改变 Module 结构，不主动改变背景拟合、对象序列、三参数 profile、横向几何、
  投影边缘、正演、LFM 或科学拒绝规则。
- 迁移采用“旁路新建 → 用户测试 → 切换 → 删除”，不保留长期双实现。
- 当前完整可执行的真实工区生成链是深度域，因此深度域是首个集成门禁。
- 时间域是长期工作流主线，在深度域验证共享 Module 后接入同一 Seam。
- 只复用当前语义和数值实现相同的部分。时间与深度投影先使用不同的冻结 policy；时间
  正演与深度正演始终由不同 ForwardAdapter 实现。
- field-conditioned 与 canonical suite 是两个 truth producer，共享 `SyntheticTruth` 数据
  合同，不共享万能请求或生成函数。
- 不提供 legacy、compat、字段猜测或错误兜底。
- 仓库外没有对 `generate_field_section`、`GeneratedSection` 或 `GenerationScenario` 的
  Python Interface 兼容要求。
- RGT 条件化 gain 不属于本轮重构。

## 3. 目标数据流

```text
Pipeline 加载工区输入
  - geometry / horizons
  - wavelet
  - survey axis
  - depth AI–Vp inputs
        │
        ▼
ForwardAdapter.prepare()
  - model SampleAxis
  - required truth context
  - frozen ProjectionPolicy
  - forward configuration
        │
        ▼
TruthGenerationRequest 或 CanonicalTruthRequest
        │
        ├── generate_field_conditioned_truth
        └── generate_canonical_truth
        │
        ▼
SyntheticTruth
        │
        ▼
BenchmarkBuilder.build()
  1. project_truth_to_model_grid
  2. ForwardAdapter.forward
  3. canonical decomposition
  4. LFM construction/degradation
  5. mask/support validation
  6. BenchmarkSample validation
        │
        ▼
BenchmarkSample
        │
        ├── shared/domain variant builders
        │           ▼
        │    BenchmarkVariant[]
        │
        ▼
shared v4 Writer
  - write_benchmark_sample
  - write_benchmark_variant
  - return artifact references
        │
        ▼
Pipeline
  - acceptance
  - sample index
  - manifest
  - progress/retry
```

Pipeline 负责 I/O、attempt、acceptance、index 和 manifest；ForwardAdapter 负责真实域物理；
Builder 负责数值产品顺序；Writer 只负责 v4 布局。truth producer 不知道 Benchmark，
projection 不知道 forward，forward 不知道 HDF5，Writer 不计算任何数组。

## 4. 科学真值 Interface

### 4.1 `RandomNamespace`

随机命名空间使用不可变类型：

```python
@dataclass(frozen=True)
class RandomNamespace:
    benchmark_version: str
    generator_family: str
```

`TruthGenerationRequest` 必须验证 namespace 的 generator family 与 calibration 一致。随机
key 中的 stream purpose、realization、zone、object、coefficient 和 geometry variant 仍由
科学核心在固定位置补充，调用方不拼接任意 key。

### 4.2 `TruthGenerationRequest`

该请求只服务 field-conditioned truth，至少包含：

- realization 与 `GenerationScenario`；
- global seed 与 `RandomNamespace`；
- sampling domain 与 axis unit；
- lateral、inline/xline、XY 与解释层位几何；
- model sample interval、纵向过采样倍数、轴原点和 context extent；
- minimum high-resolution cells；
- `sequence_minimum_duration_reference`，只接受 `median` 或 `minimum`；
- reversal、clipping 与 profile 科学 QC 阈值。

请求类型负责形状、有限性、层位次序、正采样间隔和 domain/unit 一致性校验。当前时间与
深度 Pipeline 都显式使用 `minimum`；`median` 保留为科学请求可选值，不得通过默认值隐式选择。

`GenerationScenario` 内部使用 `geometry_variant_id` 表示 pinchout 的 `035`、`065` 等几何
变体，避免与 LFM/seismic variant 混淆。Writer 在冻结 artifact 需要旧字段名时显式映射。

### 4.3 多个 truth producer

共享 field-conditioned Interface 为：

```python
generate_field_conditioned_truth(
    calibration: ImpedanceCalibration,
    request: TruthGenerationRequest,
) -> SyntheticTruth
```

时间 canonical suite 使用独立 Interface：

```python
generate_canonical_truth(
    calibration: ImpedanceCalibration,
    request: CanonicalTruthRequest,
) -> SyntheticTruth
```

canonical request 保存受控几何、主频相关尺度和 suite 参数，不进入 field-conditioned 请求。

### 4.4 `SyntheticTruth`

`SyntheticTruth` 只保存高分辨率地下真值及科学记录：

- realization、场景、采样域、轴单位和 high-resolution axis；
- lateral、inline/xline 与 XY 坐标；
- high-resolution log AI 与 RGT；
- state、object、object-xi、zone、geometry-event 和 boundary 网格；
- object catalog 与 lateral coefficient catalog；
- reversal、clipping、profile 和相关尺度诊断。

它不保存 model-grid target、reflectivity、seismic、forward support、canonical background、
target increment、LFM、HDF5 路径、split、manifest 或 schema version。

## 5. 投影 Interface

### 5.1 `SampleAxis`

数值轴只保存：

```python
@dataclass(frozen=True)
class SampleAxis:
    sample_domain: str
    unit: str
    coordinates: np.ndarray
    sample_interval: float
    positive_direction: str
    depth_basis: str | None
```

`SampleAxis` 不保存 HDF5 dataset 名称、artifact path 或 axis order。Writer 内部使用冻结的
`AxisArtifactLayout` 映射 time/depth v4 轴布局。

### 5.2 `ProjectionPolicy`

共享投影 Module 通过显式 policy 冻结两个域当前的数值行为：

```python
@dataclass(frozen=True)
class ProjectionPolicy:
    continuous_method: str
    edge_mode: str
    support_mode: str
    antialias_taps_per_factor: int
    cutoff_output_nyquist_fraction: float
    kaiser_beta: float
    categorical_window_mode: str
    geometric_valid_mode: str
```

第一轮 policy 固定为：

```text
time:
  continuous_method = scipy_resample_poly
  edge_mode = line
  geometric_valid_mode = categorical_window_any

depth:
  continuous_method = valid_fir_decimate
  edge_mode = finite_support
  geometric_valid_mode = point_sample_highres
```

时间 geometric valid 使用 categorical window 内存在有效状态；深度 geometric valid 使用
高分辨率状态网格在模型轴上的点采样。characterization 若证明两个 policy 在完整支持、边缘、
NaN、mask、长度和相位上等价，后续才允许合并。本轮 parity 不以共享为理由替换任一域的算法。

### 5.3 `ProjectedTruth`

`project_truth_to_model_grid(truth, axis, policy)` 由 Builder 调用，也可以独立测试。结果
保存：

- model target log AI 与 model RGT；
- state fraction、dominant object、zone、boundary fraction/mask；
- geometric valid mask；
- projection support high-resolution/model mask。

它不保存 forward-support mask。投影 Module 不知道子波长度、reflectivity 接口、AI–Vp 或
forward executor。

## 6. ForwardAdapter 与准备阶段

### 6.1 `DomainPreparation`

ForwardAdapter 在 truth 生成前执行准备：

```python
@dataclass(frozen=True)
class DomainPreparation:
    model_axis: SampleAxis
    required_context_extent: float
    projection_policy: ProjectionPolicy
    forward_configuration: object
```

Pipeline 先加载子波、工区轴和 AI–Vp 等输入，再调用 `prepare()`。返回的 context extent、
model axis 和 projection policy 用于构造 truth request，保证 high-resolution/model axis、
halo 和 forward support 与当前实现一致。

### 6.2 最小 Adapter Interface

```python
class ForwardAdapter(Protocol):
    def prepare(...) -> DomainPreparation: ...
    def forward(
        self,
        truth: SyntheticTruth,
        projected: ProjectedTruth,
        preparation: DomainPreparation,
    ) -> ForwardResult: ...
```

canonical 与 LFM 是独立的 `CanonicalIncrementPolicy` 和 `LfmPolicy`，由配置/Pipeline 构造
并传给 Builder。Adapter 不拥有第二套 canonical 或 LFM 数值实现。

### 6.3 `ForwardResult` 与 domain extras

`ForwardResult` 保存：

- observed seismic；
- model-consistent seismic；
- subgrid forward residual；
- `ForwardSupport`：high-resolution、model、observed 与 physics support；
- forward QC 与 metadata；
- 严格类型的 domain extras。

```python
@dataclass(frozen=True)
class TimeForwardExtras:
    reflectivity_highres: np.ndarray
    reflectivity_model: np.ndarray
    forward_valid_mask_highres: np.ndarray
    forward_valid_mask_model: np.ndarray

@dataclass(frozen=True)
class DepthForwardExtras:
    vp_highres_mps: np.ndarray
    vp_model_mps: np.ndarray
```

`ForwardResult.extras` 只接受这两个类型的 union，不接受开放的 domain array 字典。共享 Writer
根据 sample domain 校验 extras 类型并写入冻结的 v4 字段。

Builder 验证公共 valid mask 是 geometric valid、projection support 和 forward support 的
子集。ROI 内任一必需数组缺少完整支持时显式拒绝。

## 7. Benchmark 产品 Interface

### 7.1 `BenchmarkBuilder`

Builder 真正拥有 projection 和后续数值顺序：

```python
class BenchmarkBuilder:
    def build(
        self,
        *,
        truth: SyntheticTruth,
        preparation: DomainPreparation,
        forward_adapter: ForwardAdapter,
        canonical_policy: CanonicalIncrementPolicy,
        lfm_policy: LfmPolicy,
        build_policy: BenchmarkBuildPolicy,
    ) -> BenchmarkSample: ...
```

调用方不传入已经完成的 `ProjectedTruth`。Builder 内部依次执行 projection、forward、
canonical decomposition、LFM degradation、support validation 和 sample validation。

### 7.2 `BenchmarkSample`

`BenchmarkSample` 保存：

- `SyntheticTruth`、`ProjectedTruth` 与 `ForwardResult`；
- canonical background 与 target increment；
- canonical 和 controlled-degraded input LFM；
- residual、public valid mask、QC 和 domain metadata。

记录只引用现有数组，不在构造时无意义复制整幅数据。

### 7.3 `BenchmarkVariant`

Seismic variant 是独立内存产品：

```python
@dataclass(frozen=True)
class BenchmarkVariant:
    owner_realization_id: str
    variant_id: str
    sample_kind: str
    sample_domain: str
    seismic_observed: np.ndarray
    positive_gain: np.ndarray
    additive_noise: np.ndarray
    metadata: Mapping[str, object]
    qc: Mapping[str, object]
```

LFM variants 是同一个 base realization 内的 input-prior variant；seismic variants 是具有
独立 sample index row 的 Benchmark 样本变体。model-consistent seismic、valid mask、target
和 residual 由 owner base realization 提供，variant record 不重复持有。二者不使用同一层级记录。

## 8. Rejection Interface

内部按失败阶段区分：

- `TruthGenerationRejected`；
- `ProjectionRejected`；
- `ForwardRejected`；
- `BenchmarkBuildRejected`。

每类异常携带不可变 `RejectionDetail`：stage、reason code、message 和 diagnostics。Pipeline
把分层异常映射回当前冻结的外部 rejection reason 字符串，使 generation QC、catalog 和
summary 保持 v4 parity。

## 9. Writer Interface

共享 Writer 提供：

```python
write_benchmark_sample(...)
write_benchmark_variant(...)
```

Writer 只创建 v4 group/dataset、应用 `AxisArtifactLayout`、写数组与 metadata，并返回
Pipeline 构建 index 所需的 artifact reference。Writer 不允许导入：

```text
cup.impedance
scipy
cup.synthetic.time.forward
cup.synthetic.depth
core.lfm 的数值构建函数
```

该约束由 AST/import 静态测试执行。Writer 可以导入 HDF5、schema/layout、records 和
metadata validator。

## 10. 最终目录结构

```text
src/cup/synthetic/
├── __init__.py
├── benchmark.py
├── schemas.py
├── workflow.py
├── core/
│   ├── __init__.py
│   ├── artifacts.py
│   ├── calibration.py
│   ├── config.py
│   ├── contracts.py
│   ├── lfm.py
│   ├── progress.py
│   ├── projection.py
│   ├── protocols.py
│   ├── random.py
│   ├── records.py
│   ├── rejections.py
│   ├── sample_builder.py
│   ├── scenarios.py
│   ├── seismic_variants.py
│   ├── truth.py
│   └── writer.py
├── time/
│   ├── __init__.py
│   ├── calibration_pipeline.py
│   ├── canonical.py
│   ├── config.py
│   ├── forward.py
│   ├── forward_adapter.py
│   ├── geometry.py
│   ├── model.py
│   ├── pipeline.py
│   ├── sample_builder.py
│   └── seismic_variants.py
├── depth/
│   ├── __init__.py
│   ├── calibration.py
│   ├── calibration_adapter.py
│   ├── config.py
│   ├── forward_adapter.py
│   ├── generation.py
│   └── model.py
├── readers/
│   ├── __init__.py
│   ├── depth.py
│   └── time.py
└── reporting/
    ├── __init__.py
    ├── figures.py
    └── metrics.py
```

`core/artifacts.py` 第一轮保持现有职责，但不承接新 Writer、projection、rejection 或 builder
逻辑。后续是否拆分 planning、artifact contracts 和 QC 单独评估，不纳入本轮删除门禁。

## 11. 文件处置

### 11.1 直接新写

| 文件 | 责任 |
| --- | --- |
| `core/truth.py` | field-conditioned 请求、`SyntheticTruth` 和对象真值生成 |
| `core/scenarios.py` | 场景记录与 geometry variant |
| `core/records.py` | 轴、准备、投影、forward、base sample、variant 的严格记录 |
| `core/rejections.py` | 分层 rejection 类型及 detail |
| `core/projection.py` | 共享框架与冻结的 time/depth projection policy |
| `core/sample_builder.py` | projection、forward、canonical、LFM 和支持校验顺序 |
| `core/writer.py` | base/variant 的共享 v4 Writer |
| `time/forward_adapter.py` | context preparation、时间 forward 与 extras |
| `time/model.py` | 时间域对共享 `BenchmarkSample` 的只读视图与 catalog 字段映射 |
| `time/sample_builder.py` | 时间输入和 canonical producer 到共享 truth/builder policy 的集中转换 |
| `depth/calibration_adapter.py` | 深度 calibration 与共享 calibration 结构翻译 |
| `depth/forward_adapter.py` | context preparation、TVDSS、AI–Vp、深度 forward 与 extras |

### 11.2 保留并小步修改

| 文件 | 修改原则 |
| --- | --- |
| `core/lfm.py` | 深化为共享 LFM policy、degradation 和 metadata Module |
| `core/seismic_variants.py` | 两个域相同的 noise/gain 操作与 metadata |
| `time/forward.py` | 保留时间正演数值实现 |
| `time/canonical.py` | 作为独立 canonical truth producer 返回 `SyntheticTruth` |
| `time/pipeline.py` | 保留时间输入、attempt、acceptance、index 和 manifest 编排 |
| `time/config.py` | 构造时间 preparation、projection/canonical/LFM/build policy |
| `time/seismic_variants.py` | 只保留时间相位和 time-shift Adapter |
| `depth/generation.py` | 保留深度输入、attempt、acceptance、index 和 manifest 编排 |
| `depth/model.py` | 保留深度 section geometry 和 domain-only 记录 |
| `core/calibration.py` | 保持 calibration schema、provenance 和内部时间命名 |
| `workflow.py` | 切换到新的明确 Interface |
| `cup.synthetic.__init__` | 只导出稳定 Benchmark 消费 Interface |
| `core.__init__` | 使用显式导出，移除 wildcard export |

### 11.3 测试通过后删除

- `src/cup/synthetic/time/generation.py`
- `src/cup/synthetic/core/generation.py`
- `src/cup/synthetic/depth/object_core_adapter.py`
- `src/cup/synthetic/time/lfm.py`
- `src/cup/synthetic/time/writer.py`

删除后不保留同名 wrapper。正式代码通过 AST 或 `rg` 负向门禁验证不存在残余 import。

## 12. 实施阶段

### 阶段 0：冻结当前行为

建立紧凑内存 fixture 和当前深度配置的 debug-attempt-limit smoke，不复制现有约 7 GB 的
完整 HDF5。characterization 至少包括：

1. field-conditioned `none`、双向 wedge、双向 pinchout、固定/不同 seed、科学拒绝和
   catalog；
2. time/depth 投影的内部、两端、NaN、输出长度、support mask 和 categorical window；
3. truth/model 轴起止、长度、oversampling nesting 和 wavelet/forward halo；
4. 小 fixture 的 dataset path、shape、dtype、dataset/group attrs、sample index 列和
   manifest keys；
5. truth、projection support、forward support、canonical/LFM rejection 阶段及冻结的外部
   reason。

用户确认旧实现 characterization 通过后进入阶段 1。

### 阶段 1：旁路建立 truth、records 与 projection

新写 `core/truth.py`、`core/scenarios.py`、`core/records.py`、`core/rejections.py` 和
`core/projection.py`。第一版保留 time/depth 两个冻结 ProjectionPolicy，不统一边缘算法。

新旧实现对同一 request 和 random namespace 运行 parity。CPU NumPy/SciPy float64 truth、
projection、canonical、LFM 和 CPU forward 使用 `rtol=1e-10`、`atol=1e-12`。离散网格、
对象顺序、catalog 和 rejection 精确比较。用户确认后进入阶段 2。

### 阶段 2：共享 base builder/Writer，深度域先切换

新写 `core/sample_builder.py`、`core/writer.py`、`depth/calibration_adapter.py` 和
`depth/forward_adapter.py`，深化 `core/lfm.py`。本阶段只切 base realization：

```text
depth inputs
→ prepare
→ shared truth
→ shared projection
→ depth forward
→ shared canonical/LFM builder
→ shared base Writer
→ acceptance/index/manifest
```

深度门禁：

- 固定 seed truth、投影、catalog 和 rejection parity；
- TVDSS、米制采样间隔、axis positive direction 和 depth basis；
- Vp extras、observed/model-consistent/subgrid、support 与 public valid mask；
- canonical closure、LFM variants 和 residual；
- v4 base HDF5、manifest、index 和 depth reader；
- NumPy forward parity；
- CUDA 只做 backend-specific tolerance 与真实 smoke，不以 `1e-10` 证明等价。

CUDA tolerance 由阶段 0 在旧实现的重复运行和 CPU/CUDA 对照中冻结；shape、finite、mask、
support 和 QC 合同必须精确一致。artifact layout/dtype/attrs 精确比较，float32 数值按冻结的
dtype-aware tolerance 比较。

用户确认当前深度工区 debug generation 通过后进入阶段 3。

### 阶段 3：时间域接入共享 base Module

新写 `time/forward_adapter.py`、`time/model.py` 和 `time/sample_builder.py`。field-conditioned 与 canonical truth producer 分别验证，
共同进入共享 projection、builder、LFM 和 base Writer。

时间门禁覆盖时间轴/context、reflectivity/support extras、observed/model-consistent forward、
canonical closure、field-conditioned/canonical suite、v4 base Writer/reader、manifest/index 和
最小 generation smoke。用户确认后进入阶段 4。

### 阶段 4：迁移 seismic variants

共享层统一 `BenchmarkVariant` record、metadata 合同和 variant Writer。time/depth 的噪声、
gain、相位与位移数值实现保留在域 Module：两域的轴单位、AR(1) 边界、随机流和 forward
operator 不同，不建立表面相同但数值合同不同的公共实现。

每个 `BenchmarkVariant` 必须拥有独立 sample index row；LFM variants 继续属于 base sample
内部。共享 Writer 分别验证 base 与 variant group。用户确认两域 variant 门禁后进入阶段 5。

### 阶段 5：删除旧 Module 并收口 Interface

- 删除第 11.3 节五个旧 Module；
- 用 AST/`rg` 验证正式代码没有残余 import；
- 根包只暴露 Benchmark reader、sample protocol 和稳定消费 Interface；
- `core.__init__` 使用显式导出；
- `workflow.py` 与脚本入口切换到新 Pipeline Interface；
- 清理两个 Pipeline 中迁入共享 Module 的数值和字段树逻辑；
- readers、reporting 和 GINN v2 保持现有消费逻辑。

`core/calibration.py` 的内部 `truth_dt_s`、`twt_s` 和 duration 命名本轮不重命名。冻结的
calibration artifact 由深度 calibration Adapter 集中翻译，不为命名纯度扩大 schema 变更。

## 13. 人工门禁

Agent 在每阶段负责写测试、运行精确命令并记录结果。阶段专用门禁通过后进入下一阶段；旧
Module 的删除只在对应迁移门禁完整通过后执行。测试放在被忽略的 `tests/`，不提交大型 golden artifact。新旧实现并存时在同一
测试进程比较；旧实现删除后保留 Interface 不变量、确定性、artifact 和 reader 测试。

推荐基础命令：

```powershell
$env:PYTHONPATH = "src"
& C:\Users\WangQinZhuo\miniconda3\envs\pinn_inversion\python.exe -m compileall -q src/cup
& C:\Users\WangQinZhuo\miniconda3\envs\pinn_inversion\python.exe -m pytest -q -p no:cacheprovider <当前阶段测试文件>
```

深度阶段额外运行当前配置的 debug-attempt-limit 命令。全仓测试不能代替阶段专用门禁。

## 14. 完成定义

重构在以下条件全部满足时完成：

1. time/depth field-conditioned 通过同一 truth Interface；
2. canonical producer 返回相同 `SyntheticTruth` 数据合同，但保持独立生成 Interface；
3. time/depth 通过同一 projection Interface 和各自冻结 policy；
4. 两个 ForwardAdapter 在 truth 前准备 context/axis，并在 truth 后执行域正演；
5. projection support 与 forward support 分属正确记录，public valid mask 通过交集校验；
6. 共享 Builder 独占 projection 到 `BenchmarkSample` 的数值顺序；
7. `BenchmarkVariant` 与 LFM input variants 的层级明确；
8. 共享 Writer 只写 base/variant，不导入或执行数值算法；
9. v4 artifact、reader、index 和 manifest 合同保持稳定；
10. 五个旧 Module 已删除且没有兼容 wrapper 或残余 import；
11. 当前深度工区门禁和时间域 fixture 门禁均由用户确认通过。

达到该状态后，对象科学模型、投影 policy、域 forward、canonical/LFM 构建和 artifact 写出
各自具有单一 Locality。共享 Module 为两个域提供实际 Leverage，同时保持现有数值行为。
