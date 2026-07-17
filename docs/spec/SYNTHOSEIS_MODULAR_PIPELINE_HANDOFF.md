# Synthoseis-lite 积木化与共享 Pipeline Handoff

## 1. 目标

本 Handoff 交接下一轮 Synthoseis-lite 与 GINN v2 重构。目标不是增加更多失配类型，而是把当前隐式、成套、由产物行数决定训练分布的行为改成可选择、可组合、可审计的积木。

重构完成后应满足：

- 域无关逻辑尽可能由同一个共享 Pipeline Module 实现；
- 时间域和深度域通过两个真实 Adapter 接入同一个 Seam；
- 一个父实现只拥有一份 truth、target、canonical background 和 mask；
- 地震失配是父实现之上的显式视图，不复制父实现语义；
- 增加或删除地震视图不会隐式改变父实现的训练概率；
- 生成哪些视图、训练使用哪些视图、各视图权重是多少，均由显式配置决定；
- LFM 只保留 canonical background，不生成研究价值已经不足的退化版本；
- 旧 schema、旧配置字段和旧产物直接失败，不提供兼容兜底。

这次重构追求两个架构结果：

- **Leverage**：共享 Pipeline 的一次修改同时覆盖时间域和深度域；
- **Locality**：配置校验、视图编排、索引、采样和验证聚合分别集中在单一 Module 内。

## 2. 已有证据与解释

### 2.1 时间域失配训练总体有效

已有时间域基线门禁得到：

| 模型 | 训练数据 | 干净 RMSE | 失配 RMSE |
|---|---|---:|---:|
| 一维，4 层 | 仅干净地震 | 0.0550 | 0.0609 |
| 一维，4 层 + 失配训练 | 干净 + 失配各半 | 0.0419 | 0.0491 |
| 一维，膨胀卷积 + 失配训练 | 干净 + 失配各半 | **0.0399** | **0.0488** |

这证明聚合后的失配增强对鲁棒性和正则化有价值。它不证明每一种失配都有效，也不证明当前数值已经由真实工区标定。

因此下一轮不再重复“失配训练整体是否有用”的问题，而是回答：

1. 哪一类失配贡献了收益；
2. 某类失配被加入或移除时，训练分布是否保持可比；
3. 当前积木是否值得留在活动配置中。

### 2.2 当前扁平索引改变了训练分布

当前深度域结果包含 641 个父实现。每个父实现被展开为 1 个 base 和 12 个 seismic variant，最终形成 8,333 行。

现有 GINN sampler 只在 `base` 与 `seismic_variant` 两组之间做 50:50 平衡；variant 组内部由索引行数隐式决定概率。因此：

- 新增一个 variant 会降低其他 variant 的概率；
- 同一父实现对应多行 patch，父实现语义和视图语义耦合；
- 配置看不出训练的精确视图分布；
- 训练、checkpoint 选择和逐视图评估无法共享一个明确的权重合同。

新版必须先选择父实现，再选择地震视图。

### 2.3 LFM degradation 不进入下一轮实验

已有 LFM 消融表明，退化 LFM 对网络预测结果几乎没有影响。LFM 的主要作用是作为预测增量最终进行对数加法的背景项：

\[
\widehat m_{\mathrm{logAI}}
=
L_{\mathrm{external}}
+
\widehat u_{\mathrm{canonical}}
\]

下一轮不再安排 LFM degradation 门禁。整套 `controlled_default`、退化参数、退化残差、退化 QC 和相关 manifest 语义均删除。

## 3. 已冻结决策

### 3.1 包含范围

- Synthoseis-lite calibrate 与 generate 的域无关编排统一；
- 时间域与深度域入口同时切换到新版，不保留两套并行编排；
- 地震视图在生成阶段按配置物化；
- GINN v2 使用父实现均衡、显式视图权重的训练和验证；
- 深度域运行 clean、三类失配和全失配归因；
- 实施完成后更新 Synthoseis-lite、GINN v2 教程并新增 ADR。

### 3.2 不包含范围

- 三参数半马尔科夫先验是否有效的归因；
- 训练时在线生成地震失配；
- LFM degradation 的重新设计或重新验证；
- 新增空间变增益或其他失配物理模型；
- 自动效果阈值；
- v4 reader、旧训练字段或旧产物的兼容路径。

### 3.3 用词

- **父实现（realization）**：一套固定的 truth、target、canonical background、base seismic、mask、轴和地质元数据。
- **base**：未施加失配的理想正演地震。base 不是空算子视图，也没有 `view_id`。
- **地震视图（seismic view）**：对某个父实现的 base 地震按有序算子生成的物化输入。
- **原子算子（operator）**：一种可独立配置和执行的失配 Implementation。
- **活动视图**：当前配置显式声明并物化的视图。

`canonical` 只用于 canonical background / canonical increment 语义，不作为 base 地震的别名。

## 4. 目标架构

### 4.1 共享 Pipeline Module

建立深的 `SyntheticBenchmarkPipeline` Module。它的 Interface 只暴露两个工作流动作：

```text
calibrate(config, domain_adapter)
generate(config, calibration, domain_adapter)
```

Interface 同时包含以下不变量和错误模式：

- 配置在执行前完整校验；
- calibration、scenario、attempt、split 和随机流由共享 Implementation 编排；
- truth 生成、投影、正演、canonical decomposition、视图物化按固定顺序执行；
- realization index、view index、HDF5、QC、接受率和 manifest 原子发布；
- 任一必需产物失败时整次发布失败；
- 未声明的视图不计算、不写出；
- 时间域与深度域输出使用同一 schema 和同一字段语义。

共享 Pipeline 应接管当前分别散落在时间域和深度域中的：

- 公共配置解析与未知字段检查；
- calibration run 发布；
- scenario 和 attempt 计划；
- split 分配；
- 父实现循环；
- 视图循环；
- 索引、manifest 和 provenance；
- 接受率聚合；
- 公共图件与 QC 调度。

时间域和深度域入口只负责选择 Adapter 并调用共享 Pipeline。删除入口后若编排复杂度会重新出现在两个调用方，说明共享 Pipeline 正在提供足够的 Depth。

### 4.2 域 Adapter Seam

`SyntheticDomainAdapter` Interface 只承载确实随域变化的行为：

- `sample_domain`、采样单位和深度基准；
- 工作流上游数据到 calibration 输入的转换；
- section、horizon、轨迹和采样轴输入适配；
- 高分辨率轴与模型轴的域参数；
- 正演准备与执行；
- 改变子波相位或时移后重新正演；
- 域专属静校正和域元数据；
- 域专属轴与单位校验。

必须提供两个 Adapter：

- `TimeSyntheticDomainAdapter`；
- `DepthSyntheticDomainAdapter`。

公共配置、随机流、scenario、truth、视图索引、writer、接受率和报告不进入 Adapter。Adapter 不得成为把整个旧 pipeline 包起来的浅 pass-through Module。

### 4.3 地震视图 Pipeline

`SeismicViewPipeline` 是共享 Pipeline 内部的 Module。其 Interface 接收：

```text
父实现 base seismic
父实现正演上下文
有序 operator_ids
父实现随机命名空间
domain_adapter
```

返回：

```text
seismic_observed
operator trace
物化参数
随机流身份
逐算子和最终 QC
```

算子按 `operator_ids` 声明顺序执行。相同 seed、父实现和视图配置必须逐值确定。算子顺序属于视图身份的一部分。

改变 wavelet 的算子通过 domain Adapter 重新正演；可直接作用于采样地震的 gain、noise 和 axis static 算子不重建 truth。不得由调用方根据 operator kind 重写分支编排。

## 5. Synthoseis-lite v5 合同

### 5.1 schema

新版使用：

```text
benchmark schema: synthoseis_lite_v5
seismic view contract: seismic_views_v1
```

v5 reader 只接受 v5。遇到 v4、v3、旧 `sample_index.csv`、旧 LFM degradation metadata 或旧 seismic variant contract 时直接报错。

旧产物只作为历史实验依据，不进入新版训练、验证或 checkpoint provenance。

### 5.2 父实现 HDF5

每个父实现只物化一次公共数据：

```text
/realizations/<realization_id>/
├── axes/
├── truth/
├── priors/
│   └── canonical_background_log_ai
├── targets/
│   └── target_increment_log_ai
├── seismic/
│   ├── seismic_observed          # base
│   ├── seismic_model_consistent
│   └── subgrid_forward_residual
├── seismic_views/
│   └── <view_id>/
│       ├── seismic_observed
│       ├── operator_trace_json
│       └── qc/
├── masks/
│   └── valid_mask
└── qc/
```

必须保持：

```text
target_log_ai
  = canonical_background_log_ai
  + target_increment_log_ai
```

删除以下数据集及同义副本：

```text
priors/lfm_ideal
priors/lfm_controlled_degraded
priors/input_lfm_variants/*
residuals/residual_vs_lfm_ideal
residuals/residual_vs_lfm_controlled_degraded
```

训练 reader 的 `input_lfm_log_ai` 直接读取 `priors/canonical_background_log_ai`，不再通过 `lfm_variant_id` 选择路径。

### 5.3 `realization_index.csv`

一行一个父实现。至少包含：

```text
realization_id
sample_domain
sample_unit
depth_basis
suite
section_id
scenario_id
geometry_family
duration_mode
split
evaluation_role
status
hdf5_group
base_seismic_dataset
model_consistent_seismic_dataset
canonical_background_dataset
target_increment_dataset
valid_mask_dataset
n_valid
```

`realization_id` 唯一。split 的分配单位固定为父实现。

### 5.4 `seismic_view_index.csv`

一行表示一个父实现上的一个物化视图。至少包含：

```text
realization_id
view_id
status
hdf5_group
seismic_observed_dataset
operator_ids_json
operator_kinds_json
operator_parameters_json
random_stream_identity_json
n_valid
```

联合键 `(realization_id, view_id)` 唯一。每一行必须引用状态为 `ok` 的父实现，轴、mask 和有效样点数必须与父实现一致。

base 不写入该索引。空索引表示当前 benchmark 只有 base。

### 5.5 reader Interface

benchmark reader 提供父实现和视图两个层次的读取：

```text
realization_ids(...)
load_realization(realization_id)
available_view_ids(realization_id)
load_seismic_view(realization_id, view_id)
```

`load_realization` 返回 target、canonical background、increment、base seismic、model-consistent seismic、mask 和轴。`load_seismic_view` 只返回指定视图地震及其视图 metadata。

训练 dataset 在内存中组合父实现 patch 与所选地震输入，不把每个 view 伪装成一份新的 target sample。

## 6. 地震视图配置

### 6.1 配置形态

新版使用原子算子目录和有序视图：

```yaml
seismic_views:
  operators:
    white_noise:
      kind: additive_white_noise
      rms_fraction: 0.05

    colored_noise:
      kind: additive_colored_noise
      rms_fraction: 0.05
      axis_correlation:
        value: 25.0
        unit: m

    global_gain:
      kind: global_gain
      log_sigma: 0.15

    tracewise_gain:
      kind: tracewise_gain
      log_sigma: 0.15

    axis_lateral_gain:
      kind: axis_lateral_gain
      log_sigma: 0.15
      lateral_correlation_fraction: 0.30
      axis_correlation_fraction: 0.30

    phase_rotation_m10deg:
      kind: wavelet_phase_rotation
      degrees: -10.0

    phase_rotation_p10deg:
      kind: wavelet_phase_rotation
      degrees: 10.0

    wavelet_time_shift_m0p001s:
      kind: wavelet_time_shift
      seconds: -0.001

    wavelet_time_shift_p0p001s:
      kind: wavelet_time_shift
      seconds: 0.001

    axis_static_m2p5:
      kind: axis_static
      shift:
        value: -2.5
        unit: m

    axis_static_p2p5:
      kind: axis_static
      shift:
        value: 2.5
        unit: m

  views:
    - view_id: white_noise
      operator_ids: [white_noise]
    - view_id: colored_noise
      operator_ids: [colored_noise]
    - view_id: global_gain
      operator_ids: [global_gain]
    - view_id: tracewise_gain
      operator_ids: [tracewise_gain]
    - view_id: axis_lateral_gain
      operator_ids: [axis_lateral_gain]
    - view_id: phase_rotation_m10deg
      operator_ids: [phase_rotation_m10deg]
    - view_id: phase_rotation_p10deg
      operator_ids: [phase_rotation_p10deg]
    - view_id: wavelet_time_shift_m0p001s
      operator_ids: [wavelet_time_shift_m0p001s]
    - view_id: wavelet_time_shift_p0p001s
      operator_ids: [wavelet_time_shift_p0p001s]
    - view_id: axis_static_m2p5
      operator_ids: [axis_static_m2p5]
    - view_id: axis_static_p2p5
      operator_ids: [axis_static_p2p5]
```

上例是深度域活动默认。时间域使用同一 Interface，但：

- 不声明两个深度静校正算子和视图；
- colored noise 的当前等价轴相关长度为 5 个 2 ms 样点，即 `0.01 s`；
- axis-lateral gain 的当前时间轴相关比例为 `0.25`；
- 其余当前数值保持一致。

现有数值继续作为活动默认工程参数。文档和报告不得把这些数值称为真实工区反演得到的物理分布。

### 6.2 组合

复合视图只能通过显式有序列表声明：

```yaml
- view_id: example_gain_then_noise
  operator_ids: [axis_lateral_gain, colored_noise]
```

首轮活动配置不包含任何复合视图。固定 `combined_moderate` 及其专属 Implementation、配置和 metadata 全部删除。

### 6.3 配置失败条件

以下情况直接失败：

- operator ID 重复；
- view ID 重复；
- view 引用未知 operator；
- 同一 view 重复引用同一个 operator ID；
- operator kind 未注册；
- 参数缺失或出现未知参数；
- 非有限、非正或越界参数；
- 轴单位与 Adapter 的采样域不匹配；
- 时间域声明深度米制 static；
- 深度域 colored-noise 轴相关长度使用秒；
- wavelet 变换缺少可重新正演的上下文。

以下配置合法：

```yaml
seismic_views:
  operators: {}
  views: []
```

其语义是只生成 base。

## 7. GINN v2 父实现均衡训练

### 7.1 配置升级

GINN v2 实验 schema 升版。删除：

```text
sources.<synthetic>.input_seismic_variant
sampling.kind = balanced_sample_kind
```

新增 sampling kind：

```text
parent_balanced_seismic_view
```

训练 sampling 示例：

```yaml
sampling:
  kind: parent_balanced_seismic_view
  base_weight: 0.5
  variant_weight: 0.5
  view_weights:
    global_gain: 0.3333333333333333
    tracewise_gain: 0.3333333333333333
    axis_lateral_gain: 0.3333333333333333
```

权重合同：

- `base_weight` 与 `variant_weight` 必填、非负且和为 1；
- `variant_weight > 0` 时 `view_weights` 非空；
- `variant_weight = 0` 时 `view_weights` 必须为空；
- 每个 view weight 必须为正，全部 view weight 的和为 1；
- view ID 必须存在于 benchmark 的每个可训练父实现；
- 重复、未知、缺失或不可用 view 直接失败；
- 不按行数推导任何权重。

clean-only 使用：

```yaml
sampling:
  kind: parent_balanced_seismic_view
  base_weight: 1.0
  variant_weight: 0.0
  view_weights: {}
```

### 7.2 抽样顺序

每个训练 item 按以下顺序确定：

1. 从当前 split 的父实现中均衡选择一个父实现；
2. 按 `base_weight` / `variant_weight` 选择 base 或 variant；
3. 若选择 variant，按 `view_weights` 选择 `view_id`；
4. 从该父实现的有效 patch 中选择 patch；
5. 读取一次父实现 target/background/mask，并组合所选 seismic 输入。

随机种子必须由实验 seed、stage、epoch、step、父实现和视图选择用途稳定派生。相同配置和 seed 的抽样序列逐项一致。

新增或删除未被 sampling 引用的视图，不得改变：

- 父实现顺序；
- 父实现概率；
- base/variant 比例；
- 已引用视图之间的相对概率；
- patch split。

### 7.3 patch catalog 与 normalization

patch catalog 一行表示父实现上的一个空间窗口，不按 view 复制。

至少包含：

```text
patch_id
realization_id
split
lateral_start / lateral_stop
vertical_start / vertical_stop
valid_samples
valid_fraction
geometry/evaluation metadata
```

normalization 只使用 train split，并按训练 sampling 的显式分布计算或确定性加权。不得因为某个父实现拥有更多物化视图而被重复计权。

### 7.4 validation 与 checkpoint

validation 使用与训练分开的显式权重配置。所有归因实验共享完全相同的 validation 配置：

```yaml
validation:
  selection_metric: synthetic_increment.weighted_mse
  seismic_views:
    base_weight: 0.5
    variant_weight: 0.5
    view_weights:
      white_noise: 0.09090909090909091
      colored_noise: 0.09090909090909091
      global_gain: 0.09090909090909091
      tracewise_gain: 0.09090909090909091
      axis_lateral_gain: 0.09090909090909091
      phase_rotation_m10deg: 0.09090909090909091
      phase_rotation_p10deg: 0.09090909090909091
      wavelet_time_shift_m0p001s: 0.09090909090909091
      wavelet_time_shift_p0p001s: 0.09090909090909091
      axis_static_m2p5: 0.09090909090909091
      axis_static_p2p5: 0.09090909090909091
```

validation 必须在相同父实现、相同 patch 上评估 base 和每个 view，输出：

- base 指标；
- 每个 view 指标；
- amplitude/noise/operator 分组指标；
- 固定权重聚合指标；
- 每个 view 相对同 patch base 的 paired 指标变化；
- 父实现级和 patch 级样本计数。

不得用索引行平均代替显式权重聚合。

### 7.5 checkpoint provenance

checkpoint 和 resolved sources 至少记录：

```text
benchmark schema 与 run directory
benchmark manifest identity
realization index 与 seismic view index
训练 base/variant 权重
训练 view 权重
validation base/variant 权重
validation view 权重
每个 epoch 的父实现、base 和逐 view 实际计数
patch split identity
normalization identity
```

checkpoint 恢复时重新校验这些身份。benchmark 视图清单、权重或 split 不一致时直接失败。

## 8. 三阶段实施清单

### 阶段 1：共享 Pipeline 与 v5 产物

- [ ] 将 benchmark schema 升为 `synthoseis_lite_v5`；
- [ ] 建立共享公共配置 parser；
- [ ] 建立 `SyntheticBenchmarkPipeline`；
- [ ] 建立并接入时间域、深度域两个 `SyntheticDomainAdapter`；
- [ ] 统一 calibrate 发布；
- [ ] 统一 generate、scenario、attempt、split 和接受率编排；
- [ ] 建立原子 operator registry 和 `SeismicViewPipeline`；
- [ ] 只物化配置声明的 views；
- [ ] 建立 `realization_index.csv` 与 `seismic_view_index.csv`；
- [ ] 重写共享 writer 和 reader Interface；
- [ ] 删除 `controlled_default` 及全部 LFM degradation Implementation；
- [ ] 删除 `combined_moderate` 及其专属 Implementation；
- [ ] 同时切换时间域、深度域入口；
- [ ] 删除旧时间域和深度域重复编排；
- [ ] 拒绝全部旧 schema 和旧配置字段。

阶段完成条件：同一共享 Pipeline 能通过两个 Adapter 生成符合 v5 的时间域和深度域最小 benchmark；入口中不存在各自实现的公共视图、索引、manifest 或接受率编排。

### 阶段 2：GINN 父实现均衡训练

- [ ] 升级 GINN v2 实验 schema；
- [ ] 删除 `input_seismic_variant`；
- [ ] 删除 `balanced_sample_kind`；
- [ ] patch catalog 改为每个父实现 patch 一行；
- [ ] 实现 `parent_balanced_seismic_view`；
- [ ] 实现训练和 validation 的完整权重校验；
- [ ] normalization 消除视图行数重复计权；
- [ ] validation 使用固定显式权重；
- [ ] 输出 base、逐视图、分组和 paired 指标；
- [ ] checkpoint 保存视图合同与实际采样计数；
- [ ] 拒绝 v4 benchmark 和旧 checkpoint 来源语义。

阶段完成条件：在相同 seed 下，向 benchmark 增加一个未引用视图不会改变任何已存在训练 item；所有采样与验证聚合值均能从保存的计数和权重独立复算。

### 阶段 3：深度域归因与人工裁决

固定：

```text
architecture = trace_dilated_tcn
父实现集合、split、patch catalog
训练 seed、模型初始权重
训练预算、优化器、batch size
normalization
validation views 与权重
```

运行五组：

| 组 | base weight | variant weight | 训练视图 |
|---|---:|---:|---|
| clean-only | 1.0 | 0.0 | 无 |
| amplitude | 0.5 | 0.5 | global、tracewise、axis-lateral gain，组内等权 |
| noise | 0.5 | 0.5 | white、colored noise，组内等权 |
| operator | 0.5 | 0.5 | ±phase、±wavelet shift、±axis static，组内等权 |
| all | 0.5 | 0.5 | 全部 11 个原子视图，组内等权 |

所有模型使用第 7.4 节的统一全视图 validation 选择 checkpoint。

报告至少比较：

- clean weighted/paired RMSE；
- 全视图 fixed-weight RMSE；
- 每个视图 RMSE；
- 三个失配组的聚合 RMSE；
- clean 与 mismatch 的预测范围、相关性和偏差；
- 各训练组相对 clean-only 和 all 的变化；
- 实际抽样计数是否符合合同。

不设置自动效果阈值。由用户结合指标和物理合理性决定活动视图。无收益算子从活动配置移除，但保留其通用 Implementation；报告记录人工决定和依据。

## 9. 测试合同

测试写入被忽略的 `tests/`，由用户运行。

### 9.1 配置测试

- 空 operators/views 只生成 base；
- 重复 operator ID 失败；
- 重复 view ID 失败；
- 未知 operator kind 失败；
- view 引用未知或重复 operator 失败；
- 缺失、未知、非有限和非法参数失败；
- 域/单位不匹配失败；
- 旧 `lfm.controlled_degraded`、`combined`、`input_seismic_variant` 和 `balanced_sample_kind` 字段失败；
- sampling 和 validation 权重缺失或不归一失败。

### 9.2 视图与产物测试

- 只计算、写出配置声明的 views；
- 同 seed、父实现和配置逐值确定；
- 有序组合严格按声明顺序执行；
- 改变算子顺序会改变视图身份和结果；
- base 不出现在 view index；
- 双索引唯一键和引用关系正确；
- HDF5 中 canonical background 只物化一次；
- HDF5、CSV 和 manifest 不含 LFM degradation 字段；
- HDF5、CSV 和 manifest 不含 `combined_moderate`；
- v5 reader 拒绝 v4；
- view 轴、mask 和有效样点与父实现一致。

### 9.3 共享 Pipeline 测试

- 使用测试 Adapter 从同一共享 Pipeline 跑通 calibrate/generate；
- 时间域和深度域公共配置产生一致的公共 resolved 结构；
- 两个 Adapter 的域轴和单位分别保持正确；
- 时间域与深度域走同一索引、writer、manifest、接受率和报告 Implementation；
- 入口不包含公共父实现循环或 view 循环；
- 任一发布步骤失败时不产生成功 run summary。

### 9.4 GINN 采样测试

- patch catalog 不按视图复制；
- 父实现选择均衡且确定；
- base/variant 实际计数符合显式权重；
- 逐 view 实际计数符合显式权重；
- 新增未引用 view 不改变既有抽样序列；
- view index 行顺序改变不改变抽样结果；
- clean-only 从不读取 view；
- normalization 不受物化视图数量影响；
- checkpoint 保存并恢复完整视图合同；
- benchmark、split 或权重身份变化导致 checkpoint 恢复失败。

### 9.5 validation 与坐标测试

- base 和每个 view 使用同一父实现、同一 patch；
- fixed-weight 指标可由逐视图指标独立复算；
- paired 指标不跨父实现或 patch 配对；
- 诊断视图不会重复计入聚合指标；
- 深度域 TVDSS、5 m 模型轴和 xline 物化坐标保持不变；
- xline 步长 4 保持真实坐标差，不按数组下标解释；
- 时间域轴、单位和 wavelet shift 秒语义保持正确。

## 10. 文档与 ADR 收口

实现完成后：

- 更新 Synthoseis-lite 教程，只描述 v5 当前行为；
- 更新 GINN v2 教程，说明父实现均衡视图抽样和显式权重；
- 教程使用占位符，不出现真实井名、真实数据路径或用户绝对路径；
- 新增 ADR，记录“域无关逻辑尽可能进入共享 Pipeline，时间域和深度域通过 Adapter 接入同一 Seam”；
- ADR 说明两个 Adapter 使该 Seam 成为真实变化点；
- ADR 记录 v5 不兼容迁移和双索引选择；
- 将最终命令、用户测试结果、实验路径和人工视图裁决写回本 Handoff。

## 11. 完成定义

只有同时满足以下条件才算本 Handoff 完成：

- 时间域与深度域使用同一共享 Pipeline；
- v5 只保存一份 canonical background；
- base 与 views 使用双索引表达；
- 未配置视图不被计算或物化；
- GINN 先选父实现再选视图；
- 训练和 validation 权重均为显式合同；
- variant 数量不会隐式改变父实现概率；
- 旧 schema 和旧配置字段直接失败；
- 五组深度域归因使用同一验证分布完成；
- 活动视图由用户人工裁决；
- 用户运行全部测试并将结果写回；
- 教程与 ADR 已按当前 Implementation 更新。

允许最终实验结论是某一类或多类失配没有独立收益。无收益不构成代码失败；隐式采样、不可复算指标和域间重复编排才构成合同失败。
