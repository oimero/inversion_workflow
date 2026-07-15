# Synthoseis-lite 科学核心与 Benchmark 外壳重构规格

## 1. 目的

Synthoseis-lite 同时包含地下真值生成和 Benchmark 数据产品构建。当前目录已经区分
`core`、`time`、`depth`、`readers` 与 `reporting`，但科学模型、模型网格投影、正演、
LFM、variant、artifact 和运行编排仍有交叉。时间域与深度域也分别持有一部分语义相同的
Benchmark 构建逻辑。

本重构建立三个明确的 Seam：

1. 共享科学核心从校准模型生成 `SyntheticTruth`；
2. 共享 Benchmark builder 把真值、公共投影和域 Adapter 的正演结果组装为
   `BenchmarkSample`；
3. 共享 Writer 把已经物化的样本写成固定 v4 artifact。

时间域和深度域是同一科学模型与同一 Benchmark 合同的两个真实 Adapter。共享 Module
因此具有实际 Leverage，不是为单一调用方预留的假设抽象。正演、子波、采样轴来源和工区
输入仍按物理域保持 Locality。

## 2. 固定决策

- `synthoseis_lite_v4` 的 HDF5 字段树、manifest、sample index 和 reader 合同保持不变。
- 第一轮只改变 Module 结构，不主动改变背景拟合、对象序列、三参数 profile、横向几何、
  正演合同、LFM 合同或科学拒绝规则。
- 固定 seed 的连续数组使用 `rtol=1e-10`、`atol=1e-12` 比较；离散网格、对象序列、
  catalog 行和拒绝原因精确一致。
- 迁移采用“旁路新建 → 用户测试 → 切换 → 删除”。不保留长期双实现。
- 当前工区与当前可执行的完整生成链是深度域，因此深度域是首个真实集成门禁。
- 时间域是长期工作流主线，在深度域验证共享 Module 后接入同一 Seam。
- 只复用语义和数值实现真正相同的部分；时间正演与深度正演不强行合并。
- 不提供 legacy、compat、字段猜测或错误兜底。
- 仓库外没有对 `generate_field_section`、`GeneratedSection` 或 `GenerationScenario` 的
  Python Interface 兼容要求。
- RGT 条件化 gain 不属于本轮重构。

## 3. 目标数据流

```text
ImpedanceCalibration + TruthGenerationRequest
                    │
                    ▼
        generate_synthetic_truth
                    │
                    ▼
             SyntheticTruth
                    │
                    ▼
       shared model-grid projection
                    │
                    ▼
             ProjectedTruth
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 time DomainAdapter     depth DomainAdapter
          │                   │
          └─────────┬─────────┘
                    ▼
       shared build_benchmark_sample
                    │
                    ▼
             BenchmarkSample
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   variants    shared Writer   reporting
                    │
                    ▼
          v4 HDF5 / index / manifest
```

共享 Module 不读取地震体、解释文件或子波文件。这些外部输入由分域 Pipeline 加载，再由
DomainAdapter 转换为共享 builder 所需的正演结果和轴描述。

## 4. 共享 Interface

### 4.1 `TruthGenerationRequest`

`TruthGenerationRequest` 汇总生成一个地下真值所需的调用知识：

- realization 与场景；
- global seed、随机命名空间和 generator family；
- lateral、inline/xline、XY 与解释层位几何；
- sampling domain、axis unit、模型采样间隔和纵向过采样倍数；
- 轴原点、上下文范围和 minimum high-resolution cells；
- reversal、clipping 与 profile 科学 QC 阈值。

请求类型负责形状、有限性、层位次序、正采样间隔和 domain/unit 一致性校验。科学核心不
导入 Benchmark schema；调用方把 schema version 作为随机命名空间传入，以维持现有随机
stream。

### 4.2 `SyntheticTruth`

`SyntheticTruth` 只保存高分辨率地下真值及解释它所需的科学记录：

- realization、场景、采样域、轴单位和高分辨率采样轴；
- lateral、inline/xline 与 XY 坐标；
- high-resolution `log_ai` 与 RGT；
- state、object、object-xi、zone、geometry-event 和 boundary 网格；
- object catalog 与 lateral coefficient catalog；
- reversal、clipping、profile 和相关尺度诊断。

它不保存 model-grid target、reflectivity、seismic、forward mask、canonical background、
target increment、LFM、HDF5 路径、split、manifest 或 schema version。

科学核心的外部 Interface 为：

```python
generate_synthetic_truth(
    calibration: ImpedanceCalibration,
    request: TruthGenerationRequest,
) -> SyntheticTruth
```

### 4.3 `SampleAxis` 与 `ProjectedTruth`

`SampleAxis` 显式保存：

- `sample_domain`；
- `unit`；
- 均匀采样坐标；
- sample interval；
- depth basis 或 TWT 语义；
- v4 artifact 使用的轴 dataset 名称和 axis order。

`project_truth_to_model_grid()` 对时间域和深度域使用同一套规则完成：

- 连续属性抗混叠降采样；
- categorical fraction 与 dominant ID；
- boundary fraction；
- model-grid valid mask；
- high-resolution 与 model-grid forward-support mask。

结果保存为 `ProjectedTruth`。该 Module 只使用规则采样轴和数值数组，不解释秒或米的物理
含义。

### 4.4 `DomainAdapter`

时间域与深度域分别提供一个真实 Adapter，满足同一个最小 Interface：

```python
class DomainAdapter(Protocol):
    def forward(self, truth: SyntheticTruth, projected: ProjectedTruth, ...) -> ForwardResult: ...
    def canonical_contract(self, axis: SampleAxis) -> CanonicalIncrementContract: ...
    def lfm_policy(self, ...) -> LfmPolicy: ...
```

`ForwardResult` 统一保存：

- observed seismic；
- model-consistent seismic；
- subgrid forward residual；
- observed/physics support 与 QC；
- domain-specific forward metadata。

时间 Adapter 拥有 TWT、时间子波和时间正演。深度 Adapter 拥有 TVDSS、深度子波、AI–Vp
关系和深度正演 executor。Adapter 不重新实现对象模型、投影、canonical decomposition、
LFM degradation 编排或 artifact 字段树。

### 4.5 `BenchmarkSample`

`BenchmarkSample` 是两个域共享的完整内存产品，保存：

- `SyntheticTruth` 与 `ProjectedTruth`；
- `ForwardResult`；
- canonical background 与 target increment；
- canonical 和 controlled-degraded input LFM；
- residual、mask、QC 和 domain metadata。

共享构建 Interface 为：

```python
build_benchmark_sample(
    truth: SyntheticTruth,
    projected: ProjectedTruth,
    adapter: DomainAdapter,
    policy: BenchmarkBuildPolicy,
) -> BenchmarkSample
```

该 Module 固定以下顺序：

```text
truth
→ model-grid projection
→ domain forward
→ canonical decomposition
→ LFM degradation
→ masks and QC
→ BenchmarkSample
```

### 4.6 共享 Writer

Writer 接收已经完成数值构建的 `BenchmarkSample`，只执行：

- 创建固定 v4 group 和 dataset；
- 根据 `SampleAxis` 写入时间域或深度域轴；
- 写入数组、dtype、unit、domain、axis order 和 metadata；
- 返回 Pipeline 构建 index 所需的 HDF5 group path。

Writer 不调用低通、正演、重采样、decomposition、LFM 或 variant 生成。两个域通过同一个
Writer 写出公共字段，DomainAdapter 只提供轴和 forward metadata。

## 5. 最终目录结构

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
│   ├── model.py
│   ├── progress.py
│   ├── projection.py
│   ├── protocols.py
│   ├── random.py
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
│   ├── domain_adapter.py
│   ├── forward.py
│   ├── geometry.py
│   ├── pipeline.py
│   └── seismic_variants.py
├── depth/
│   ├── __init__.py
│   ├── calibration.py
│   ├── calibration_adapter.py
│   ├── config.py
│   ├── domain_adapter.py
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

`depth/generation.py` 保留工区输入加载、attempt、acceptance、variant 和 manifest 编排，
但不再包含对象科学模型、公共投影、LFM degradation 数值实现或 HDF5 字段树写出。

## 6. 文件处置

### 6.1 直接新写

| 文件 | 责任 |
| --- | --- |
| `core/truth.py` | 请求类型、`SyntheticTruth` 和完整对象真值生成 |
| `core/scenarios.py` | 场景记录及显式 controls 到场景目录的转换 |
| `core/model.py` | `SampleAxis`、`ProjectedTruth`、`ForwardResult`、`BenchmarkSample` |
| `core/projection.py` | 两个域共享的连续与 categorical model-grid projection |
| `core/sample_builder.py` | truth、domain forward、canonical 与 LFM 的公共构建顺序 |
| `core/writer.py` | 两个域共享的 v4 HDF5 Writer |
| `time/domain_adapter.py` | 时间轴、时间 forward 与时间域 metadata |
| `depth/calibration_adapter.py` | 深度 calibration 与共享 calibration 结构的翻译 |
| `depth/domain_adapter.py` | TVDSS、AI–Vp、深度 forward 与深度域 metadata |

新 Module 先旁路建立。parity 测试直接调用新旧 Interface，不建立长期 import wrapper。

### 6.2 保留并小步修改

| 文件 | 修改原则 |
| --- | --- |
| `core/lfm.py` | 从 metadata helper 深化为共享 LFM policy、degradation 和 metadata Module |
| `core/seismic_variants.py` | 收纳两个域相同的 noise/gain 操作与 metadata，域 shift 留给 Adapter |
| `time/forward.py` | 保留时间域正演数值实现 |
| `time/canonical.py` | 生成 truth 后进入共享 projection 和 builder |
| `time/pipeline.py` | 保留时间工区输入、attempt、acceptance、variant 和 manifest 编排 |
| `time/config.py` | 保留严格解析，集中构造共享 policy 与时间 Adapter 配置 |
| `time/seismic_variants.py` | 只保留时间相位和时间 shift Adapter |
| `depth/generation.py` | 保留深度工区编排，调用共享 truth、projection、builder 和 Writer |
| `depth/model.py` | 保留深度 section geometry 与 domain-only 记录 |
| `core/calibration.py` | 保持现有 calibration schema、provenance 和内部时间命名 |
| `workflow.py` | 切换到新的明确 Interface |
| `cup.synthetic.__init__` | 只导出稳定 Benchmark 消费 Interface |
| `core.__init__` | 使用显式导出，移除 wildcard export |

### 6.3 测试通过后删除

- `src/cup/synthetic/time/generation.py`
- `src/cup/synthetic/core/generation.py`
- `src/cup/synthetic/depth/object_core_adapter.py`
- `src/cup/synthetic/time/lfm.py`
- `src/cup/synthetic/time/writer.py`

`time/lfm.py` 的数值实现迁入共享 `core/lfm.py`；`time/writer.py` 由共享 Writer 取代。
删除后不保留同名 wrapper。

## 7. 实施阶段

### 阶段 0：冻结深度域当前行为

当前深度工区是第一验收来源。先建立两类 fixture：

1. 紧凑内存 fixture，用于对象科学模型、投影、canonical、LFM、Writer 和拒绝 parity；
2. 当前深度配置的 debug-attempt-limit smoke，用于真实地震几何、TVDSS、AI–Vp、CUDA/NumPy
   forward 和 v4 reader。

不复制现有约 7 GB 的完整 HDF5。内存 fixture 至少覆盖 `none`、双向 wedge、双向
pinchout、固定/不同 seed、科学拒绝、catalog、model projection 和 canonical closure。

用户确认旧实现 characterization 测试通过后进入阶段 1。

### 阶段 1：旁路建立共享 truth 与 projection

新写 `core/truth.py`、`core/scenarios.py`、`core/model.py` 和 `core/projection.py`。

`core/truth.py` 迁移：

- 背景采样；
- Semi-Markov 对象序列；
- 三参数 profile 采样与约束；
- 横向相关、wedge、pinchout 与 minimum-thickness；
- high-resolution log AI、RGT、categorical truth；
- 科学拒绝与诊断。

`core/projection.py` 迁移两个域相同的抗混叠降采样、categorical fraction、dominant ID、
boundary fraction 和 support mask。

新旧实现对同一 request 和随机命名空间运行 parity。用户确认后进入阶段 2。

### 阶段 2：共享 builder 与 Writer，深度域先切换

新写 `core/sample_builder.py`、`core/writer.py` 和 `depth/domain_adapter.py`；深化
`core/lfm.py`。

深度 Pipeline 改为：

```text
load depth inputs
→ shared truth
→ shared projection
→ depth forward Adapter
→ shared Benchmark builder
→ domain variants
→ shared Writer
→ acceptance / manifest
```

深度门禁：

- 固定 seed 连续数组近似，离散对象、catalog 和拒绝原因一致；
- TVDSS、米制采样间隔、axis positive direction 和 depth basis 正确；
- observed、model-consistent 与 subgrid residual 角色一致；
- canonical closure、LFM variants 和 valid mask 合同一致；
- v4 HDF5 字段、dtype、attrs、manifest 和 sample index 一致；
- depth reader、NumPy forward 和可用时的 CUDA forward smoke 通过；
- 当前深度工区 debug-attempt-limit generation 由用户确认通过。

深度域通过真实门禁后，才允许时间域接入共享 builder。

### 阶段 3：时间域接入共享 Module

新写 `time/domain_adapter.py`。时间 Pipeline、canonical suite 和 seismic variants 切换到共享
truth、projection、builder、LFM 和 Writer。

时间域没有当前真实工区门禁时，使用紧凑内存 fixture 验证：

- truth 与当前时间实现 parity；
- 时间轴、时间子波、high-resolution observed forward 和 model-consistent forward；
- canonical 与 field-conditioned suite；
- phase、time shift、noise、gain 和 combined variants；
- v4 Writer/reader 和 manifest/index 结构；
- import 与最小 generation smoke。

用户确认后进入阶段 4。

### 阶段 4：共享 variant 实现并删除旧 Module

把两个域相同的 white/colored noise、global gain、lateral gain、axis-lateral gain 和公共 QC
迁入 `core/seismic_variants.py`。时间相位/time shift 与深度 static shift 继续由各自 Adapter
提供。

时间、深度全套门禁通过后删除第 6.3 节文件，并检查正式路径不存在残余 import。

### 阶段 5：收口公共 Interface

- 根包只暴露 Benchmark reader、sample protocol 和稳定消费 Interface；
- 科学生成、projection 和 builder 通过明确的 `core` 路径导入；
- `core.__init__` 显式列出导出项；
- `workflow.py` 与脚本入口使用新 Pipeline Interface；
- 清理两个 Pipeline 中已迁入共享 Module 的数值和字段树逻辑；
- readers、reporting 和 GINN v2 保持现有消费逻辑。

`core/calibration.py` 的内部 `truth_dt_s`、`twt_s` 和 duration 命名本轮不重命名。冻结的
calibration artifact 由 `depth/calibration_adapter.py` 集中翻译，不为命名纯度扩大 schema
变更。

## 8. 人工门禁

Agent 在每个阶段负责写测试和给出精确命令，用户亲自运行。未收到用户通过确认，不进入
下一阶段删除。协作顺序固定为：

1. Agent 完成当前阶段及测试；
2. Agent 汇总修改、预期产物和命令；
3. 用户运行并返回完整失败信息或通过确认；
4. Agent 修复当前阶段，或进入下一阶段；
5. 两个域全部切换并通过后才删除旧 Module。

测试放在被忽略的 `tests/`，不提交大型 golden artifact。新旧实现并存时在同一测试进程
比较；旧实现删除后保留 Interface 不变量、确定性、artifact 和 reader 测试。

## 9. 验收矩阵

| 类别 | 验收内容 |
| --- | --- |
| 科学 parity | 连续数组近似；离散网格、对象序列、catalog 和拒绝原因精确 |
| 投影 parity | 连续降采样、categorical fraction、dominant ID、boundary 和 mask 一致 |
| 随机性 | 同 seed 可复现；不同 seed 有变化；随机命名空间维持现有 stream |
| 深度集成 | 当前工区几何、TVDSS、AI–Vp、forward、artifact 和 reader smoke |
| 时间集成 | 时间轴、子波、forward、canonical suite、artifact 和 reader fixture |
| Artifact | 两域 v4 字段树、metadata、sample IDs、manifest 和 reader 合同一致 |
| 数值合同 | canonical closure、forward roles、valid mask 和 LFM variants 一致 |
| Adapter | 共享 Module 无 TWT/TVDSS 分支；物理差异集中在两个 DomainAdapter |
| 依赖方向 | truth/projection/builder 不导入 HDF5、Pipeline、reporting 或工区 loader |
| 清理 | 五个旧 Module 删除，正式路径不存在 compat wrapper |
| 静态检查 | `compileall`、import smoke 和阶段专用 pytest 通过 |

推荐基础命令：

```powershell
$env:PYTHONPATH = "src"
& C:\Users\WangQinZhuo\miniconda3\envs\pinn_inversion\python.exe -m compileall -q src/cup
& C:\Users\WangQinZhuo\miniconda3\envs\pinn_inversion\python.exe -m pytest -q -p no:cacheprovider <当前阶段测试文件>
```

每阶段用实际测试文件替换占位符。深度域阶段额外运行当前配置的 debug-attempt-limit 命令，
全仓测试不能代替阶段专用门禁。

## 10. 完成定义

重构在以下条件全部满足时完成：

1. 时间域与深度域通过同一 truth 和 projection Seam；
2. 两个 DomainAdapter 满足同一 forward/canonical/LFM Interface；
3. `SyntheticTruth` 不携带 Benchmark 数值产品或 artifact 语义；
4. 共享 builder 独占 truth 到 `BenchmarkSample` 的构建顺序；
5. 共享 Writer 独占 v4 公共字段树写出；
6. 时间与深度 Pipeline 只保留工区输入、attempt、variant、acceptance 和 manifest 编排；
7. v4 artifact 与 reader 合同保持稳定；
8. 五个旧 Module 已删除且没有兼容 wrapper；
9. 当前深度工区门禁和时间域 fixture 门禁均由用户确认通过。

达到该状态后，对象科学模型、公共投影、canonical/LFM 构建和 artifact 写出各自具有单一
Locality。时间域与深度域只在真实物理差异处使用不同 Adapter，共享 Module 为两个域提供
实际 Leverage。
