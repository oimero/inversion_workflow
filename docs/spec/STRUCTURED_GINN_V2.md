# Structured GINN V2 实施规格

> 状态：路线规划。本文定义从结构化 truth 生产到物理排序的最小可验证路径。
> 当前优先级是先统一 `cup.synthetic` 的 producer 合同，再让 Oracle 闭环成立。

## 1. 目标与路线决策

Structured GINN V2 首先验证两件事：

1. 网络能否从 seismic + LFM 恢复三状态半马尔科夫对象的状态、持续时间、zone background 和对象内部三参数 profile；
2. 深度域物理正演能否在多个结构化候选之间提供有效排序。

目标变量是对象级 structured latent，而不是逐采样自由 increment：

```text
seismic + LFM
    -> state / boundary / duration
    -> zone a/b
    -> object c0/c1/c2
    -> deterministic decoder
    -> high-resolution log AI
    -> projection
    -> depth-domain forward seismic
```

结构化残差相对于生成器的 zone background 定义；LFM 是输入和诊断信息，不替代生成器内部的 `a/b`。真实工区输出解释为由 seismic、LFM 和标定地质先验共同约束的候选集合，不解释为唯一地下真值。

首轮模型固定为深度域、单 zone、单 trace。`cup.synthetic` 的生产合同和 forward seam 必须同时支持 time/depth；模型首轮只在 depth 上训练不改变这个公共合同。

### 1.1 increment 的位置

increment 语义属于历史实验路线，不属于当前 `cup.synthetic`、`ginn_v2` 或新 artifact 的接口。当前代码中仍出现的以下对象都属于阶段 0 的迁移范围：

- `CanonicalIncrementPolicy`；
- `target_increment_log_ai`、`canonical_background_log_ai`；
- 以 `decompose_log_ai()` 和 `validate_increment_contract()` 为核心的 LFM builder；
- `BenchmarkSample` 中的 increment 字段；
- 依赖这些字段的旧 writer、reader、index 和 manifest。

新路线不为历史 artifact 增加兼容 adapter，也不把最终 `target - canonical` 反推成 structured truth。历史文件可以保留作考古和数值对照，但不参与新 producer、reader、训练或 Oracle 合同。

### 1.2 canonical artifact 与重生成

`cup.synthetic` 的 canonical 输出改为 `structured_truth_v1`。它不是 increment artifact 的 sidecar，而是一次 time/depth 生成运行的正式主产物，包含：

- 观测 seismic、LFM、valid mask 和完整轴信息；
- high-resolution/model-grid truth 与投影 support；
- zone 表、segment 表和 state/object 高分辨率网格；
- realization-zone `a/b`；
- raw/projected/effective 三套 `c0/c1/c2`；
- forward identity、calibration、projection、wavelet 和几何 identity。

因此，旧产物不需要先全量重跑，但新 `ginn_v2` 数据集必须在 producer 合同改好后重新生成。执行顺序固定为：

```text
修改 cup.synthetic producer
    -> 生成少量 time/depth dev artifact
    -> strict reader / decoder / Oracle
    -> Oracle 通过后生成完整结构化数据集
```

### 1.3 阶段与代码量

| 阶段 | 目标 | 预计新增或迁移生产代码 |
|---|---|---:|
| 0 | canonical structured producer、time/depth 对称 seam 与 forward 骨架 | 900–1400 行 |
| 1 | structured reader、decoder 与 Oracle 闭环 | 800–1300 行 |
| 2 | 单 zone 监督模型与 exact HSMM | 1500–2500 行 |
| 3 | 候选生成与合成物理排序 | 500–900 行 |
| 4 | 真实工区冻结候选排序 | 400–700 行 |

代码量不含测试、配置、数据生成运行时间和文档。阶段保持五个顶层垂直切片；阶段内部按实现顺序推进，不再拆出 1A、2A 等子阶段。

## 2. time/depth 对称性合同

### 2.1 共享层必须相同

以下内容由 `cup.synthetic.core` 提供一套实现，time/depth 使用同一接口和同一错误语义：

- `SampleAxis`、`SectionGeometry` 和 lateral geometry identity；
- field-conditioned truth 生成、zone/object/state 表和 raw/projected/effective 参数捕获；
- finite-support projection、nested-axis 校验和 support mask；
- structured in-memory record、artifact writer、artifact reader 和 manifest；
- mask、主键、segment endpoint、duration 和 split 语义；
- 生成 attempt plan、接受/拒绝、QC 汇总和发布事务；
- LFM observation 的字段、axis、source provenance 和 valid mask。

### 2.2 允许存在的 domain-specific 层

差异集中在 domain adapter，不扩散到 structured truth 或公共 artifact：

- time 使用秒和 TWT axis，depth 使用米和 TVDSS axis；
- time forward 直接使用 AI/reflectivity，depth forward 使用 AI–Vp relation 和 depth executor；
- time/depth 的 wavelet resampling、forward halo 和额外 forward arrays 可以不同；
- real geometry、horizon 输入、survey 读取和 calibration source 可以不同。

这些差异必须通过显式 `sample_domain`、`unit`、`depth_basis` 和 `ForwardExtras` 表达。公共实现不通过 CSV 列名、`hasattr` 或数组形状猜测域语义。

### 2.3 坐标、采样和几何

- time axis 单位是 `s`，positive direction 是 `increasing_time`，`depth_basis` 为空；
- depth axis 单位是 `m`，positive direction 是 `down`，`depth_basis` 是 `tvdss`；
- high-resolution axis 与 model-grid axis 严格嵌套，抽取因子为正整数；
- `xline_step=4` 是 survey line number 的采样属性，不是物理距离，也不是数组 stride；
- inline/xline 只记录几何身份，横向物理距离使用 `lateral_m` 或 survey geometry；
- 单 trace 的垂向 forward 不因 xline step 改变；
- 所有观测、latent、segment 表和 forward 数组都必须带明确的 axis/domain/unit identity。

## 3. 阶段 0：canonical producer、对称 seam 与 forward 骨架

阶段 0 是承上启下的重构阶段，生产代码改动主要发生在 `src/cup/synthetic`，同时建立独立的 `src/ginn_v2/`。阶段 0 完成后，可以用同一套 reader 和 artifact schema 读取一个 time dev 样本和一个 depth dev 样本；不训练网络。

### 3.1 producer-owned structured truth

在 `src/cup/synthetic/core/truth.py` 的生成 seam 上直接保存中间值：

```text
raw       = _object_lateral_parameters 产生的 c0/c1/c2
projected = _project_profile_coefficients 之后、c0 conditioning 之前
effective = c0 conditioning 之后、用于生成 log AI 的 c0/c1/c2
```

同时保存：

- realization-zone 级别的 `background_a/background_b`；
- `realization_id + lateral_index + zone_id + object_id` 主键；
- 当前 trace 的 zone top/bottom、object top/bottom、实际 duration fraction；
- high-resolution 网格上 segment 的 `duration_samples`；
- state、state id、event target、projection scale、c0 conditioning adjustment；
- high-resolution/model axis、seismic、LFM、valid mask 和 forward identity。

producer 通过 domain-neutral `StructuredSampleRecord` 发布这组字段；结构化 truth 的中间字段保留在 `SyntheticTruth`，producer writer 位于 `cup.synthetic.core`。producer 不依赖 `ginn_v2` 的模型代码。

payload 的表语义固定如下：

```text
zone row:
    realization_id, lateral_index, zone_id
    top, bottom, background_a, background_b, zone_valid

segment row:
    realization_id, lateral_index, zone_id, object_id
    state, state_id, top, bottom
    duration_fraction, duration_samples
    raw_c0/raw_c1/raw_c2
    projected_c0/projected_c1/projected_c2
    effective_c0/effective_c1/effective_c2
    segment_supervision_valid
```

`a/b` 在 realization × zone 上共享；writer 可以在 lateral zone row 中重复其值，但必须验证同一 realization-zone 的一致性。`top/bottom`、duration 和三套 coefficient 是 trace-specific。

生成器最后的 AI bounds clipping 也属于 truth contract。若发生 clipping，payload 记录 clipping mask 或等价的确定性规则；decoder 按同一规则复现，Oracle 不通过忽略 clipping 放宽。

LFM 以 `LfmObservation` 发布，至少包含值、axis、unit、source identity 和 valid mask。LFM producer 与 structured latent 的关系必须显式记录；target-dependent 的 canonical decomposition 不能伪装成独立外部 LFM。阶段 0 先保留真实工作流需要的 LFM 输入 seam，再决定合成训练使用外部 LFM、独立先验 LFM 或受控的 target-derived 对照。

### 3.2 替换旧 shared record

公共生成 seam 使用一个不含 increment 字段的 `StructuredSampleRecord`，包含 truth payload、observed seismic、`LfmObservation`、projected structured truth、`ForwardResult`、domain metadata、QC 和 support masks。

旧 `BenchmarkSample` 仍可作为现有 HDF5 发布的兼容来源，但 pipeline 会立即转换为 `StructuredSampleRecord`；`CanonicalIncrementPolicy`、`build_lfm_products()` 的 increment 返回值和旧字段不进入新的 structured artifact。后续若移除兼容 HDF5，再共同清理 `core/sample_builder.py`、`core/records.py`、`core/lfm.py`、`core/writer.py`、`benchmark.py`、`readers/v5.py` 和相关 schema/index，避免只改某个 domain adapter 后留下半套合同。

### 3.3 对称性重构清单

阶段 0 必须处理当前扫描发现的结构差异：

1. 使用 `src/cup/synthetic/core/geometry.py` 的通用 `SectionGeometry`，统一 time 的 `horizon_twt_s` 和 depth 的 `horizon_tvdss_m` 为带 axis identity 的 `horizon_coordinates`；
2. 把 `time/geometry.py` 和 `depth/model.py` 中重复的 section record 收敛到同一 shared record，domain-specific QC 放入显式 extension；
3. 让 `geometry_feasibility_rows()` 接受通用 section/axis，不再通过 `hasattr(section, "horizon_twt_s")` 猜测字段；
4. 让 time/depth 都从同一个 structured sample seam 返回，不再分别包装成 `TimeBenchmarkSample` 和 `DepthGeneratedSection`；
5. writer/reader 采用一个共同的 structured field layout，time/depth 只在 forward extras 和单位上扩展；
6. axis metadata 校验基于显式 axis identity，而不是只根据 `twt`/`tvdss` 字符串匹配。

time 的 TargetZone support/fill QC 与 depth 的 survey support QC 可以保留不同字段，但二者都必须落在同一个 section QC interface 中，并明确标记 domain-specific extension。

### 3.4 canonical artifact

producer writer 发布一个自包含的 `structured_truth_v1` artifact，而不是向历史 artifact 附加 sidecar：

- artifact 包含 reader 所需的观测、LFM、truth、projection、forward support 和 zone/segment 数据；
- manifest 包含 artifact type/version、realization/scenario、producer、calibration、projection、forward、wavelet、axis 和 geometry identity；
- time/depth 使用同一字段名、同一主键、同一 mask 语义和同一 writer/reader；
- 历史 artifact 只作为外部 provenance 或人工数值对照，不承担字段补全；
- producer 内存 payload 与写出再读回的 payload 必须逐字段一致。
- `structured_truth_v1/sample_index.csv` 是结构化样本索引，包含 realization、lateral、zone 和 geometry 字段；legacy v5 的 `realization_index.csv` 继续服务于 HDF5 兼容发布，两个索引有明确边界。
- `structured_truth_v1/publication_report.json` 记录 accepted parent、trace、主键和 orphan 审计；发布前该报告必须通过。
- trace artifact 同时保存观测 seismic、model-consistent seismic 和 forward context，供磁盘 round-trip Oracle 重建同一垂向 forward。

阶段 0 的小型运行至少生成一个 depth dev artifact，并执行一个 time artifact 的 schema/axis symmetry smoke check。完整数据生成放在阶段 1 Oracle 通过之后。

### 3.5 `src/ginn_v2/` 与 forward seam

新包保持扁平：

```text
src/ginn_v2/
├── __init__.py
├── runtime.py       # device、日志等运行时工具
└── forward.py       # NumPy/Torch structured forward seam
```

新包直接依赖 `cup.physics.numpy_backend`、`cup.physics.torch_backend`、`cup.physics.adapters`、`cup.physics.execution.DepthForwardExecutor`、`cup.synthetic.core.projection`、`cup.synthetic.core.records.SampleAxis` 和 `cup.synthetic.core` 的 structured producer/artifact 合同。

`forward.py` 对 NumPy/Torch 使用相同输入顺序和轴语义，至少支持 sample axis、sample domain/unit、depth basis、wavelet、AI–Vp relation、log AI 和 chunk size。Torch 路径保持对 log AI、a/b 和连续 coefficient 的有限梯度。

### 3.6 阶段 0 门禁

1. 同一 structured schema 可以写入并读回 time/depth dev artifact；
2. producer 显式发布 `a/b` 和 raw/projected/effective 三套系数；
3. 新 shared record 不含 increment 训练字段，LFM source provenance 完整；
4. time/depth 共有字段、主键、mask、axis 和 support 校验一致；
5. depth 的 TVDSS、xline step=4、`lateral_m` 和 forward seam 通过检查；
6. NumPy/Torch forward 输出一致且梯度有限；
7. 缺字段、轴错位、非有限值、主键歧义或域语义猜测直接失败。

结构化发布审计从磁盘读取 `structured_truth_v1`，检查 accepted parent 集合、trace 文件、数组合同、axis identity、raw/projected/effective segment 字段和 primary key；sample index 仅包含 accepted parent。

阶段 0 未通过时不进入阶段 1；不在旧 artifact 上继续补 adapter。

## 4. 阶段 1：structured reader、decoder 与 Oracle

阶段 1 是第一道科学门禁，也是本轮实现最详细的阶段。输入是阶段 0 生成的 canonical `structured_truth_v1`；主 Oracle 运行使用 depth，time 运行验证相同 reader/decoder/schema 的对称性。

### 4.1 端到端 seam

Oracle 使用写出再读回的 artifact：

```text
producer in-memory payload
    -> structured writer
    -> strict reader / StructuredTruthAdapter
    -> StructuredSample
    -> decoder
    -> finite-support projection
    -> domain forward
    -> Oracle report
```

允许保留一个 in-memory adapter 作为调试工具，但科学门禁必须跨过 writer/reader seam，才能同时发现生成逻辑和持久化错误。

### 4.2 最小结构化 interface

```text
StructuredSample
├── ObservedTrace
│   ├── sample_axis, sample_domain, sample_unit, depth_basis
│   ├── seismic, model_consistent_seismic, lfm, observed_valid
│   ├── lateral_m, inline, xline, xline_step
│   └── forward_context
├── LatentTrace
│   ├── latent_axis, latent_valid, log_ai_highres_truth
│   └── state/object/xi/zone high-resolution grids
├── ZoneTruth[]
│   ├── zone_id, top, bottom, zone_valid
│   └── background_a, background_b
└── SegmentTruth[]
    ├── zone_id, object_id, state, state_id
    ├── top, bottom, duration_fraction, duration_samples
    ├── c0/c1/c2 raw
    ├── c0/c1/c2 projected
    ├── c0/c1/c2 effective
    └── segment_supervision_valid
```

采样轴优先使用带轴对象；公共 interface 不长期传递没有 domain、unit 和轴身份的裸 `np.ndarray`。

mask 分开记录：

- `observed_valid`：观测、projection、forward 和观测 loss 支持；
- `latent_valid`：高分辨率 structured truth 合法位置；
- `segment_supervision_valid`：整个 segment 可用于参数监督；
- padding mask：只用于 batch 对齐；
- mask 不进入 encoder channel，也不与 zone mask 合并。

高分辨率 axis 必须覆盖 decoder 和 forward 所需的完整 support。单 zone 可以裁成 zone 加明确的 forward context；保留 zone 外 context 时，producer 发布 context extension 和 mask。未知区域不使用边缘填充值冒充 truth。

### 4.3 strict reader / adapter

reader 读取版本化 artifact，adapter 映射 producer contract 到 `StructuredSample`。校验至少包括：

1. artifact type/version、producer/calibration/projection/forward identity 完整；
2. high-resolution/model axis 同域、同单位、同 basis，并按整数因子嵌套；
3. realization/lateral/zone/object 主键、zone/segment endpoint 和 state/object/zone/xi 网格一致；
4. 三类 mask、有限性、单位、几何、LFM provenance 和 forward support 完整。

缺字段或合同不一致时直接失败。历史 target/canonical/increment 字段、最终系数、随机 seed、CSV 列名和未声明字段不作为结构化 truth 的推断入口。

### 4.4 decoder

decoder 接受完整 zone 和 segment 描述，不接受逐点自由 increment。zone 内坐标为 ζ，object 内坐标为 ξ：

```text
background = a + b * (2ζ - 1)
profile    = c0 + c1 * (2ξ - 1) + c2 * sin(πξ)
```

顺序固定为：

```text
raw
    -> profile projection
    -> c0 conditioning
    -> effective
    -> unclipped log AI
    -> producer-declared AI bounds handling
    -> high-resolution log AI
```

NumPy/Torch 共享字段、参数顺序、endpoint 规则、输出形状和 clipping 语义。Torch 对 a/b/c0/c1/c2 保持有限梯度；离散 state、object id 和 mask 只控制选择。

### 4.5 Oracle 检查

Oracle 从发布 truth 复现 high-resolution log AI、model-grid log AI 和 forward seismic，检查：

1. decoder 与 producer truth 在 `latent_valid` 上一致，raw/projected/effective 逐字段一致；
2. axis nesting、factor、endpoint、support mask 和 `cup.synthetic.core.projection` 一致；
3. time/depth 使用各自 adapter，但共享同一 structured 输入/输出合同；
4. NumPy/Torch decoder parity、Torch decoder/forward 有限梯度；
5. `xline_step=4`、`lateral_m`、zone/object endpoint、三类 mask 和 forward waveform 误差/相关性。

初始 decoder 门禁为 `rtol=1e-4`、`atol=1e-5`；projection 沿用现有 float32 容差；forward 同时报告误差与相关性。非有限值、轴/主键/endpoint 错位、clipping 语义不一致或梯度异常均失败。

### 4.6 阶段 1 产物与停止条件

```text
structured_truth_v1 dev artifact
strict reader / StructuredTruthAdapter
NumPy/Torch decoder parity report
time/depth schema symmetry report
Oracle contract round-trip report
forward identity and support report
```

生成命令提供 `--run-structured-oracle` 开关。打开后，producer 写出 artifact，重新从磁盘读取每条 trace，执行 decoder、projection、model-consistent forward 和 Torch gradient checks；Oracle report 未通过时，run 不发布为成功数据集。

Oracle 未通过时，修复 producer、writer/reader、decoder 或 forward seam，并重新生成 dev artifact；不进入模型训练。Oracle 通过后，按同一 producer 合同生成完整 depth 训练集，并保留 time 结构化样本作为公共 schema 回归样本。

## 5. 阶段 2：单 zone 结构化监督模型

阶段 2 只实现 depth、单 zone、单 trace、clean seismic/LFM 双通道、三状态 emission、固定 calibration transition/duration prior、exact HSMM、zone `a/b` head 和 segment `c0/c1/c2` head。

训练顺序是 teacher-forced segmentation，小范围 boundary jitter，再验证 predicted MAP segmentation 下的参数 head。首轮使用 deterministic MAP；posterior sampling、backward sampling、多样本候选、多 zone、lateral pooling 和 time 模型留到后续讨论。

科学对照固定使用同一 parent split：no-seismic control、seismic+LFM 主模型、parent-shuffled seismic 和 LFM-only baseline。进入阶段 3 的条件是主模型相对 no-seismic 在结构化指标上有配对改善，parent shuffle 后改善明显减弱，且改善不是只体现在 forward closure。

## 6. 阶段 3：候选生成与合成物理排序

阶段 3 让模型输出多个 structured candidate。每个候选包含 state sequence、object boundary、duration、zone `a/b`、object `c0/c1/c2`、decoded high-resolution log AI、projected log AI 和 forward seismic。

物理排序先作为独立 selector，不作为训练 loss。在合成 holdout 上比较 MAP candidate、physics-best candidate、oracle-best-of-K、Calibration-prior 和 LFM-only baseline。

oracle-best 不能覆盖 truth 时检查 posterior coverage/表示能力；oracle 可行但 physics-best 错时检查 forward、wavelet、nuisance 和评分函数；只有 physics-best 稳定优于 MAP，才进入真实工区。

## 7. 阶段 4：真实工区冻结候选排序

阶段 4 使用冻结 Structured 模型生成真实剖面候选，固定 wavelet 和 depth forward，允许受限的全局 gain、小范围 shift/phase nuisance，输出候选排序、分数分解和井震诊断。

真实阶段保持模型和候选生成规则冻结。候选共享受限 nuisance 参数；低质量井只作诊断，井震相关性只作为一致性证据。合成排序门禁通过后，才研究固定 segmentation 下的连续参数小范围 refinement；边界搜索、soft HSMM 和真实 posterior adaptation 属于更晚的研究方向。

## 8. 当前代码审计结论与实施边界

### 8.1 已经对称、可以保留的核心能力

- `SampleAxis` 对 time/depth 的 domain、unit、basis 和 regular sampling 做统一校验；
- `generate_field_conditioned_truth()`、`project_truth_to_model_grid()`、finite-support signal 和 lateral `m` 逻辑是公共能力；
- `core.pipeline` 已经提供共享的 attempt plan、发布事务和 seismic view seam；
- `core.amplitude_calibration` 在 aligned seismic/RGT/valid-mask seam 之后是公共实现；
- time/depth forward 的物理差异已经集中在各自 forward adapter，属于合理差异。

### 8.2 当前需要在阶段 0 修复的非对称或历史耦合

1. `core.geometry.SectionGeometry` 是通用 record，但当前 time/depth 各自定义 section record，通用 record 没有成为实际 seam；
2. `core.artifacts.geometry_feasibility_rows()` 通过 `hasattr` 在 TWT/TVDSS 字段之间选择，公共实现缺少显式 axis interface；
3. time 使用 `TimeBenchmarkSample`，depth 使用 `DepthGeneratedSection`，同一生成生命周期的返回类型不对称；
4. `core.writer.py` 虽然共享部分计划，但公共字段仍围绕 increment，time/depth 的 structured field layout 尚未统一；
5. `core.artifacts.validate_dataset_metadata()` 依赖 `twt`/`tvdss` 名称推断 axis，应该改成 manifest/axis identity 校验；
6. `core.pipeline.py`、`core/sample_builder.py`、`core/lfm.py`、`core/records.py`、`benchmark.py`、`readers/v5.py` 和 v5 index/schema 共同绑定历史 increment 合同，阶段 0 需要成组迁移；
7. time/depth 的 section QC 字段集合不同，保留差异时必须放入显式 domain extension，不能让共享报告通过字段存在性猜语义。

这些问题是架构重构事项，不是要求把 time/depth 的 forward 物理强行写成同一实现。当前代码没有发现 `xline_step=4` 在 `core.geometry.resample_section_path()` 或 depth survey index 中被当作单位步长；新 structured artifact 仍必须持久化该几何属性并用 `lateral_m` 做物理距离。

### 8.3 版本与停止条件

建议产物版本：

```text
structured_truth_v1
structured_ginn_v2_experiment_v1
structured_ginn_v2_checkpoint_v1
structured_ginn_v2_prediction_v1
structured_ginn_v2_report_v1
```

每个产物记录 sample axis、domain、calibration、projection、forward、parent split、normalization 和 source identity。

全局停止条件：阶段 0 的 producer/artifact/对称 seam 未通过时不进入阶段 1；阶段 1 Oracle 未闭合时不训练网络；阶段 2 的结构化监督和 parent-shuffle 门禁失败时不做物理排序；阶段 3 的 physics-best 没有稳定优于 MAP 时真实阶段只保留合成诊断。

本文不展开 posterior schema、32 samples、复杂 mixture/diffusion、真实联合微调或完整报告字段；这些内容等到阶段 1/2 的实际结果出现后再决定。
