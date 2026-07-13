# Synthoseis-lite 阶段 2.5：边界重构与检阅建议修复规格

## 1. 目的与范围

阶段 2.5 位于 canonical increment/v4 基础实现之后、微纹理阶段之前。本阶段冻结
Synthoseis-lite 的语义合同和模块 seam，使后续微纹理实验只改变生成器，不同时改变
标签、reader 或 benchmark 入口。

本阶段包含：

- canonical 增量合同的公共位置、数值校验和 LFM producer metadata；
- Synthoseis-lite v4 的 probe 关闭和明确拒绝；
- time/depth 按实际依赖的包划分；
- v4 writer、reader、manifest 的合同一致性检查；
- 阶段 2 代码的本地可复现验证和迁移清单。

本阶段不实现微纹理、GINN v2、physics、R0/R1 或旧 full-correction 产物迁移。
GINN v2 只同步移除已关闭 probe 的下游输出与旧 synthetic import；不新增或改写其
reader、loss、checkpoint 或训练语义。

固定取舍：v4 不生成、读取或报告 frequency probe；`cup.seismic.observability` 和
已有 forward-observability 报告仍是独立分析旁路。继续使用 `synthoseis_lite_v4`，
阶段 2.5 之前写出的 v4 与 v3 一样冻结，需要训练时重新生成，不做字段猜测或自动
迁移。测试继续放在仓库已经忽略的 `tests/`，不新增逐文件 SHA-256、递归 provenance
或其他防御性指纹。

## 2. Canonical 公共包

公共 seam 为：

```text
src/cup/impedance/
├── __init__.py
├── contracts.py
└── canonical.py
```

`cup.impedance` 的公共接口为：

```python
CanonicalIncrementContract
validate_increment_contract
validate_sample_axis
canonical_lowpass
decompose_log_ai
generation_contract
```

LFM metadata 另有 `validate_lfm_producer_contract` 和
`validate_contract_compatibility`。`build_lfm_producer_contract` 只负责生成
manifest 字段，不改变 canonical 数学定义。

### 2.1 Canonical contract

合同版本为 `canonical_increment_v1`，并要求以下固定语义：

```yaml
contract_version: canonical_increment_v1
semantics: canonical_complement_log_ai
value_domain: log(AI)
log_base: natural
ai_unit_convention: m/s*g/cm3
sample_axis_uniform: true
sample_axis_dtype: float64
sample_interval_relative_tolerance: 1.0e-6
sample_interval_absolute_tolerance: 1.0e-9
lowpass:
  implementation: scipy_butter_sosfiltfilt
  design_order: 6
  effective_zero_phase_order: 12
  cutoff_definition: single_pass_minus_3db_final_minus_6db
  buffer_mode: reflect
```

时间域固定 `sample_unit=s`、`cutoff_hz=15.0`、`buffer_axis_units=0.4`；深度域固定
`sample_unit=m`、`depth_basis=tvdss`、`cutoff_wavelength_m=400.0`、
`buffer_axis_units=400.0`。错误的 domain、unit、basis、cutoff、buffer、implementation、
order 或语义字段直接失败。

入口轴可以是任意数值 dtype，校验时转换为 float64；不接受 NaN、Inf、非递增、重复
或不等间隔轴。等间隔比较使用合同中的相对和绝对容差，所有物化轴写出为 float64。
采样间隔是合同权威值，期望轴为 `axis[0] + arange(n_sample) * sample_interval`。

低通固定使用 `scipy.signal.butter(..., output="sos")`、reflect buffer 和
`scipy.signal.sosfiltfilt(..., padtype=None)`。有限连续段逐段处理，短于
`max(21, pad_samples + 1)` 时失败。21 是项目 structural minimum，不是
`padtype=None` 的 SciPy 内部必要 padlen。

## 3. LFM producer contract

benchmark manifest 和真实工区 LFM manifest 继续使用公共字段名 `lfm_contract`。
reader 只读取并校验生产者写入的 metadata，不重复低通，也不重新生成标签。

`lfm_contract` 至少包含：

```yaml
producer_kind: synthoseis_lite
producer_schema: synthoseis_lite_v4
sample_domain: depth
sample_unit: m
sample_interval: 5.0
sample_axis_uniform: true
sample_axis_dtype: float64
sample_interval_relative_tolerance: 1.0e-6
sample_interval_absolute_tolerance: 1.0e-9
depth_basis: tvdss
value_domain: log(AI)
log_base: natural
ai_unit_convention: m/s*g/cm3
canonical_lowpass_applied_to: target_log_ai
canonical_lowpass_application_count: 1
well_control_lowpass_application_count: 0
final_volume_lowpass_application_count: 0
post_lowpass_vertical_warp_applied: false
implementation: scipy_butter_sosfiltfilt
design_order: 6
effective_zero_phase_order: 12
cutoff_definition: single_pass_minus_3db_final_minus_6db
cutoff_wavelength_m: 400.0
buffer_mode: reflect
buffer_axis_units: 400.0
variant_selection:
  selected: controlled_default
  available: [canonical, controlled_default]
final_background_complement_response_rms: 0.0
final_background_complement_response_ratio: 0.0
final_background_max_trace_response_ratio: 0.0
```

时间域使用 `sample_unit=s` 和 `cutoff_hz=15.0`，并省略 `depth_basis`。
`well_control_lowpass_application_count` 与 `final_volume_lowpass_application_count`
分开记录；空间建模后的最终体不被“井控曾低通过”自动证明为严格的 `P(m)`。
最终体 complement-response QC 只用于诊断和方法比较，不是消费端重新滤波的理由。

## 4. v4 probe 关闭

v4 配置不包含 `probe_selection`。time/depth parser 一旦看到该字段，或任何
probe-enabled 配置，直接报错并指出 probe 不属于 v4 canonical benchmark。

正式 v4 生成链不再创建：

- `frequency_probe` 或 `frequency_probe_seismic_variant` sample row；
- probe HDF5 group、frequency catalog 和 probe metrics/report-card；
- `src/cup/synthetic/probes.py` 或等价的 domain wrapper。

time reader 对历史 probe row 明确失败，而不是把它解释为普通 base。深度 reader
只接受 `base` 和 `seismic_variant`。`cup.seismic.observability`、
`scripts/forward_observability.py` 及已有分析结果保留在独立旁路；它们不再成为
v4 benchmark 的训练样本选择器。

## 5. Synthetic 包边界

当前目标拓扑为：

```text
src/cup/synthetic/
├── __init__.py
├── benchmark.py
├── schemas.py
├── workflow.py
├── core/
│   ├── calibration.py
│   ├── generation.py
│   ├── random.py
│   ├── artifacts.py
│   ├── config.py
│   └── progress.py
├── time/
│   ├── __init__.py
│   ├── config.py
│   ├── geometry.py
│   ├── canonical.py
│   ├── calibration_pipeline.py
│   ├── generation.py
│   ├── pipeline.py
│   ├── forward.py
│   ├── lfm.py
│   ├── writer.py
│   └── seismic_variants.py
├── depth/
│   ├── __init__.py
│   ├── config.py
│   ├── calibration.py
│   ├── generation.py
│   ├── model.py
│   ├── lfm.py
│   ├── writer.py
│   ├── pipeline.py
│   └── object_core_adapter.py
├── readers/
│   ├── __init__.py
│   ├── time.py
│   └── depth.py
└── reporting/
    ├── __init__.py
    ├── figures.py
    └── metrics.py
```

`core` 不导入 `time` 或 `depth`。它只保存共享校准、对象记录、场景目录、随机流、
artifact schema 和进度工具。time/depth 通过明确 seam 使用 core、`cup.impedance` 和
物理后端。深度对象生成通过 `object_core_adapter` 复用共享对象记录，TWT 命名只停留
在 adapter/实现内部，不泄漏到 depth reader 的公共字段。

`readers` 只读取物化数组、manifest、index 和 dataset metadata，并执行合同校验；不
重新低通、不在 patch 内生成标签、不猜字段含义。`reporting` 只做图和指标，不参与
生成或训练。不存在 `old`、`legacy`、`compat` wrapper。

阶段 2.5 已完成 time、core、readers、reporting 的实际迁移，并将深度域的数据记录
抽到 `depth/model.py`；深度域的数值生成仍由 `depth/generation.py` 统一编排，
`depth/lfm.py`、`depth/writer.py` 和 `depth/pipeline.py` 是后续按 seam 拆分的目标
文件，不在本阶段复制或包一层兼容 wrapper。这样目录边界先固定，深度域逐步抽取时
不改变已有数值语义。

阶段 2.5 删除提前落地的 `core/microtexture.py` 和 `none` emitter。微纹理接口只
保留在微纹理设计规格中，阶段 3 重新设计并实现。

## 6. v4 物化字段与迁移

`synthoseis_lite_v4` 的 canonical 字段仍为：

```text
truth/model_target_log_ai
priors/canonical_background_log_ai
targets/target_increment_log_ai
priors/input_lfm_variants/<variant>/log_ai
```

完整道先计算 `P(target_log_ai)` 与 `target_increment_log_ai`，patch 或 reader 只
切片已有结果。v4 重新生成时写入新的 `increment_contract` 和 `lfm_contract`；旧 v4
manifest 缺字段时直接失败，不做自动补全。

历史 v3、阶段 2 之前生成的 v4、forward-observability 结果以及旧报告保持原目录
不变。它们由不可变产物和 Git 历史保证可复现，不要求正式生产包永久保留旧 reader。

## 7. 验收矩阵

本地 `tests/`（被 `.gitignore` 忽略）至少覆盖：

| 类别 | 验收 |
| --- | --- |
| contract | 错 semantics、unit、log base、cutoff、buffer、implementation、版本直接失败 |
| axis | NaN/Inf、反向、重复、非等间隔失败；相对/绝对容差按合同工作 |
| operator | time/depth 独立复算；连续有限段不跨 NaN；短段失败；输出轴 float64 |
| LFM | producer contract 完整字段、domain compatibility 和错误计数失败 |
| writer/reader | time/depth v4 字段树、manifest contract 和 shared target increment |
| probe | 配置、index、reader、report 遇 probe 明确失败或不生成 |
| package | `cup.impedance`、`cup.synthetic.benchmark`、time/depth reader import smoke |
| cleanup | 正式 synthetic 包不存在旧 root import、microtexture emitter 或 probe writer |

建议验证命令：

```powershell
$env:PYTHONPATH = "src"
& C:\Users\WangQinZhuo\miniconda3\envs\pinn_inversion\python.exe -m compileall -q src/cup src/ginn_v2
& C:\Users\WangQinZhuo\miniconda3\envs\pinn_inversion\python.exe -m pytest -q -p no:cacheprovider tests/test_canonical_increment.py tests/test_synthoseis_v4_canonical.py tests/test_synthoseis_stage_2_5.py
```

测试文件不随本阶段提交；Handoff 只记录本地运行结果和失败原因。

## 8. 阶段门禁与后续

阶段 2.5 完成门禁：

1. canonical contract、axis、LFM compatibility validator 通过本地 fixture；
2. time/depth v4 writer/reader 能读新 manifest，并拒绝缺少新合同的旧 v4；
3. probe 配置、row、writer、reader 和 report 正式路径全部关闭；
4. 新包 import smoke 和 `compileall` 通过；
5. Handoff 写明重生成范围、测试命令和未覆盖项。

只有这些门禁全部满足后，阶段 3 才能重新设计并实现 `thin_bed_cluster`、
`canonical_well_patch` 和 paired A/B/C benchmark。阶段 3 继续复用同一
`target_increment_log_ai` 合同，不在微纹理发射器中引入第二套 residual 语义。

## 9. 检阅建议的采纳边界

本阶段采纳：canonical 包位置、显式 contract_version、双轴容差、float64 写出轴、
LFM producer metadata、井控/最终体低通计数分离、probe v4 拒绝、按依赖拆分包结构、
阶段 2 之前 v4 重新生成。

本阶段不采纳：测试文件纳入 Git、强行 time/depth 对称目录、重新引入微纹理 emitter、
把 forward-observability 的 whole-target 结果转成 v4 probe 样本、逐文件 SHA-256、
旧产物自动迁移或双语义兼容层。冻结历史 GINN v2 summarize 可以继续读取旧 report-card
字段，但这些字段不再由新 v4 生产和评估路径写出。
