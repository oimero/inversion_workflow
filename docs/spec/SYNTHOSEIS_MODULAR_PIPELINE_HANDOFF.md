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
- benchmark 只拥有地质评估角色，GINN 实验套件单独拥有 train/validation/test split；
- 五组归因共享从 train 父实现 base 地震计算的一份 normalization；
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
SyntheticBenchmarkPipeline(adapter).calibrate(
  config, output_dir=..., **runtime
)
SyntheticBenchmarkPipeline(adapter).generate(
  config, calibration, output_dir=..., **runtime
)
```

Interface 同时包含以下不变量和错误模式：

- 配置在执行前完整校验；
- calibration、scenario、attempt、evaluation role 和随机流由共享 Implementation 编排；
- truth 生成、投影、正演、canonical decomposition、视图物化按固定顺序执行；
- realization index、view index、HDF5、QC、接受率和 manifest 原子发布；
- 父实现级 view 失败按第 5.6 节拒绝该 attempt；run 级 contract、程序或最终 artifact 校验失败才终止整次发布；
- 未声明的视图不计算、不写出；
- 时间域与深度域输出使用同一 schema 和同一字段语义。

共享 Pipeline 应接管当前分别散落在时间域和深度域中的：

- 公共配置解析与未知字段检查；
- calibration run 发布；
- scenario 和 attempt 计划；
- evaluation role 计算；
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

算子分为两类：

```text
forward_parameter_operator
  wavelet_phase_rotation
  wavelet_time_shift

sampled_seismic_operator
  axis_static
  global_gain
  tracewise_gain
  axis_lateral_gain
  additive_white_noise
  additive_colored_noise
```

执行 grammar 固定为：

1. forward-parameter operators 必须连续出现在视图最前面；
2. 同一种 forward operator kind 在一个视图中最多出现一次；
3. Pipeline 汇总所有 forward 参数，通过 domain Adapter 只重新正演一次；
4. sampled-seismic operators 随后按声明顺序作用于该次正演结果；
5. sampled-seismic operator 后再出现 forward operator 时配置直接失败。

这样可以保证重新正演不会覆盖前序 gain 或 noise。不得由调用方根据 operator kind 重写分支编排。

## 5. Synthoseis-lite v5 合同

### 5.1 schema

新版使用：

```text
benchmark schema: synthoseis_lite_v5
science revision: synthoseis_lite_science_v3
projection contract: finite_support_projection_v1
seismic view contract: seismic_views_v1
seismic operator contract: seismic_operators_v1
random stream contract: synthoseis_random_v3
```

science-v3 manifest 必须完整记录上述版本链。删除 science-v2 中的：

```text
lfm_degradation_contract_version
seismic_variant_contract_version
```

缺少任一 science-v3 字段或字段值不一致时直接失败。

v5 reader 只接受 v5。遇到 v4、v3、旧 `sample_index.csv`、旧 LFM degradation metadata 或旧 seismic variant contract 时直接报错。

旧产物只作为历史实验依据，不进入新版训练、验证或 checkpoint provenance。

### 5.2 父实现 HDF5

每个父实现只物化一次公共数据：

```text
/realizations/<realization_id>/
├── axes/
├── truth/
│   └── model_target_log_ai
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

`target_log_ai` 的唯一物化路径固定为：

```text
/realizations/<realization_id>/truth/model_target_log_ai
```

不得新增 `targets/target_log_ai` 同义副本，也不得由 reader 只根据 background 与 increment 临时构造后冒充已物化 target。reader 必须读取该固定路径并复算上式进行校验。

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
section_id
scenario_id
geometry_family
duration_mode
evaluation_role
hdf5_group
base_seismic_dataset
model_consistent_seismic_dataset
target_log_ai_dataset
canonical_background_dataset
target_increment_dataset
valid_mask_dataset
n_valid
```

`realization_id` 唯一。该索引只包含完整发布、可被 consumer 使用的父实现，不保存失败 attempt，因此不含 `status`。逐 attempt 进度进入 `attempt_progress.csv`，拒绝明细进入 `generation_rejection_details.csv`，原因汇总进入 `rejection_reason_summary.csv`。

`evaluation_role` 只允许 `development_pool` 或 `geometry_holdout`。它属于 benchmark 的地质评估语义，不等于 GINN 的 train/validation/test split。

### 5.4 `seismic_view_index.csv`

一行表示一个父实现上的一个物化视图。至少包含：

```text
realization_id
parent_realization_id
view_id
sample_domain
sample_unit
hdf5_group
seismic_observed_dataset
operator_ids_json
operator_kinds_json
operator_parameters_json
operator_contract_versions_json
view_spec_canonical_json
view_spec_sha256
random_stream_identity_json
n_valid
```

联合键 `(realization_id, view_id)` 唯一。每一行必须引用状态为 `ok` 的父实现，轴、mask 和有效样点数必须与父实现一致。

base 不写入该索引。空索引表示当前 benchmark 只有 base。该索引同样只包含完整发布的视图，不保存失败行。

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

### 5.6 attempt 事务与发布

一个父实现 attempt 的 base 和全部声明视图构成同一事务：

```text
truth → projection → base forward → canonical decomposition
      → all declared views → parent/view validation → commit
```

任一声明视图失败时：

- 当前父实现 attempt 整体拒绝；
- HDF5 不保留该父实现或其部分视图；
- realization/view index 不写入任何对应行；
- attempt 进度写入 `attempt_progress.csv`，拒绝明细写入 `generation_rejection_details.csv`，原因汇总写入 `rejection_reason_summary.csv`；
- 其他 attempt 继续执行。

整个 run 仅在配置或程序错误、acceptance contract 不满足、最终 artifact 校验失败时失败。run 在临时目录构建，所有 HDF5、双索引、manifest 和 acceptance 校验通过后才发布成功 summary。

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

复合视图仍受第 4.3 节的 operator grammar 约束。例如以下顺序非法：

```yaml
- view_id: invalid_gain_then_phase
  operator_ids: [axis_lateral_gain, phase_rotation_p10deg]
```

以下顺序合法，并且只执行一次带相位旋转的正演：

```yaml
- view_id: phase_then_gain_then_noise
  operator_ids:
    - phase_rotation_p10deg
    - axis_lateral_gain
    - colored_noise
```

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
- forward operator 出现在 sampled-seismic operator 之后；
- 一个视图含有多个同 kind forward operator。

以下配置合法：

```yaml
seismic_views:
  operators: {}
  views: []
```

其语义是只生成 base。

### 6.4 视图身份与随机流

`view_id` 是可读名称，不单独构成科学身份。每个 view 必须生成规范化 spec：

```text
view_spec_canonical_json
view_spec_sha256
```

规范化 spec 至少覆盖：

- 有序 operator IDs；
- 每个 operator 的 kind；
- 规范化后的完整参数和单位；
- 每个 operator contract version；
- science revision；
- seismic-view contract version；
- random-stream contract version。

JSON 使用固定键顺序、固定数值表示和 UTF-8 编码计算 SHA-256。`view_spec_sha256` 写入 view index、HDF5 view attributes、manifest 和 GINN checkpoint。相同 `view_id` 但 spec hash 不同视为不同科学内容，consumer 不得只比较名称。

随机系数的身份固定由以下元组派生：

```text
science_revision
random_stream_contract_version
realization_id
operator_id
operator_spec_sha256
coefficient_name
```

随机身份不得依赖 view ID、view 配置行号、view 列表长度或其他视图是否存在。同一个 operator ID 和 spec 被多个视图复用时，其随机系数相同；若需要独立 realization，必须声明新的 operator ID。operator 的最终输出仍可因前序 sampled-seismic operator 改变输入尺度而不同。

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

该 sampling 配置属于具体 synthetic loss block，而不属于 source。不同 synthetic block 可以引用同一 benchmark，但必须各自显式声明地震输入分布。real-well 和 real-field loss block 不接受该 sampling kind。

训练 sampling 示例：

```yaml
sampling:
  kind: parent_balanced_seismic_view
  parent_weights: {base: 0.5, variant: 0.5}
  view_weights:
    global_gain: 0.3333333333333333
    tracewise_gain: 0.3333333333333333
    axis_lateral_gain: 0.3333333333333333
```

权重合同：

- `parent_weights.base` 与 `parent_weights.variant` 必填、非负且和为 1；
- `parent_weights.variant > 0` 时 `view_weights` 非空；
- `parent_weights.variant = 0` 时 `view_weights` 必须为空；
- 每个 view weight 必须为正，全部 view weight 的和为 1；
- view ID 必须存在于 benchmark 的每个可训练父实现；
- 重复、未知、缺失或不可用 view 直接失败；
- 不按行数推导任何权重。

clean-only 使用：

```yaml
sampling:
  kind: parent_balanced_seismic_view
  parent_weights: {base: 1.0, variant: 0.0}
  view_weights: {}
```

对 block 的语义固定为：

- synthetic supervised block 使用所选 base/view 作为模型输入，target 始终是父实现的 canonical increment；
- synthetic physics block 若需要失配输入，也必须在自身 sampling 中声明；其 physics target 始终是父实现的 nominal `seismic_model_consistent`，不随输入 view 改写；
- validation 的 weighted metric 必须进入 checkpoint selection、deployment eligibility 和 checkpoint manifest，不能只作为日志指标。

### 7.2 抽样顺序

每个训练 item 按以下顺序确定：

1. 从当前 split 的父实现中均衡选择一个父实现；
2. 按 `parent_weights.base` / `parent_weights.variant` 选择 base 或 variant；
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

### 7.3 split 所有权

benchmark 只发布 `evaluation_role`，不发布 train/validation/test split。GINN 实验套件是 split 的唯一所有者，并在建立 patch catalog 前物化：

```text
split_assignment_<source_id>.csv
```

split contract 固定为：

```yaml
split_contract:
  version: parent_hash_split_v1
  owner: ginn_v2_experiment_suite
  seed: 20260714
  hash_algorithm: sha256
  validation_fraction: 0.15
  test_fraction: 0.15
  geometry_holdout_role: test
```

算法固定为：

1. `evaluation_role=geometry_holdout` 的父实现直接分配到 test；
2. 其余父实现对 UTF-8 字符串 `parent_hash_split_v1\0<seed>\0<realization_id>` 计算 SHA-256；
3. 取 digest 前 8 字节按 unsigned big-endian 整数解释并除以 `2**64`，得到 `[0, 1)` 值；
4. 值小于 `0.15` 分配到 test；
5. 值位于 `[0.15, 0.30)` 分配到 validation；
6. 其余分配到 train。

每个 synthetic source 生成一份 `split_assignment_<source_id>.csv`，至少记录 `realization_id`、`evaluation_role`、hash value、split 和 split contract identity。同一 source 的各阶段与各归因组共用该文件及其 SHA-256；实验 manifest 通过 `split_assignments` 映射保存所有 source 文件。

patch 构建只允许 strict 读取物化 split，不再根据父 ID derive。父实现缺失、重复、多余或 split contract 不一致时直接失败。

### 7.4 patch catalog 与 normalization

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

normalization 只使用 train split，不得因为某个父实现拥有更多物化视图而被重复计权。五组实验统一冻结以下 normalization：

```yaml
normalization:
  seismic_reference: base_only
  lfm_reference: canonical_background
  parent_weighting: uniform
  split: train
```

计算时每个 train 父实现总权重相同，每个父实现内部的有效样点等权；直接读取完整父实现数组一次，不按重叠 patch 重复计数。seismic 统计只来自 base，LFM 统计只来自 canonical background，任何 view 都不参与 normalization。

normalization 物化为单独的可读产物并记录 canonical JSON 与 SHA-256。五组实验必须引用同一 normalization identity；variant 使用该固定统计量变换，因此 gain/noise 的影响不会被各组各自重算的均值和标准差吸收。

### 7.5 validation 与 checkpoint

validation 使用与训练分开的显式权重配置。所有归因实验共享完全相同的 validation 配置：

```yaml
validation:
  selection_metric: synthetic_increment.weighted_mse
  seismic_views:
    parent_weights: {base: 0.5, variant: 0.5}
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

该 checkpoint objective 明确定义为 **view-uniform objective**：variant 总权重中的 11 个视图逐 view 等权，因此 amplitude/noise/operator 三个家族的权重分别为 `3/11`、`2/11`、`6/11`。这不是 family-uniform objective。三个家族另行报告等家族权重的诊断指标，但不参与 checkpoint 选择。

validation 必须在相同父实现、相同 patch 上评估 base 和每个 view，输出：

- base 指标；
- 每个 view 指标；
- amplitude/noise/operator 分组指标；
- 固定权重聚合指标；
- 每个 view 相对同 patch base 的 paired 指标变化；
- 父实现级和 patch 级样本计数。

amplitude/noise/operator 的分组由 view index 的 `operator_kinds_json` 解析；`view_id` 只用于标识 view，不参与分组猜测。

不得用索引行平均代替显式权重聚合。

### 7.6 checkpoint provenance

checkpoint 和 resolved sources 至少记录：

```text
benchmark schema 与 run directory
benchmark manifest identity
science-v3 contract identity
realization index 与 seismic view index
split_assignments identity
view spec hash set
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

- [x] 将 benchmark schema 升为 `synthoseis_lite_v5`；
- [x] 将 science contract 升为 science-v3，并删除旧 LFM/variant 版本字段；
- [x] 建立共享公共配置 parser；
- [x] 建立 `SyntheticBenchmarkPipeline`；
- [x] 建立并接入时间域、深度域两个 `SyntheticDomainAdapter`；
- [x] 统一 calibrate 发布；
- [x] 统一 generate、scenario、attempt、evaluation role 和接受率编排；
- [x] 建立原子 operator registry 和 `SeismicViewPipeline`；
- [x] 只物化配置声明的 views；
- [x] 建立 `realization_index.csv` 与 `seismic_view_index.csv`；
- [x] 重写共享 writer 和 reader Interface；
- [x] 删除 `controlled_default` 及全部 LFM degradation Implementation；
- [x] 删除 `combined_moderate` 及其专属 Implementation；
- [x] 同时切换时间域、深度域入口；
- [x] 删除旧时间域和深度域重复编排；
- [x] 拒绝全部旧 schema 和旧配置字段。

阶段完成条件：同一共享 Pipeline 能通过两个 Adapter 生成符合 v5 的时间域和深度域最小 benchmark；入口中不存在各自实现的公共视图、索引、manifest 或接受率编排。

内部按以下门禁推进，但最终只发布一次 v5，不保留长期双实现：

```text
1A  [x] v5 records、science contract、view spec、双索引和内存 Reader/Writer
1B  [x] SeismicViewPipeline、operator registry、grammar 和随机身份
1C  [x] 共享 generate Pipeline 与测试 Adapter
1D  [x] Depth Adapter 接入和深度域 smoke
1E  [x] Time Adapter 接入和时间域 smoke
1F  [x] 共享 calibrate Pipeline
1G  [x] 删除旧路径、完成最终 artifact 校验并正式发布 v5
```

阶段 1 的关闭证据：

| 门禁 | Implementation | 验证 | 结果 |
|---|---|---|---|
| 1A | `core/records.py`、`core/writer.py`、`core/v5_artifacts.py`、`readers/v5.py` | `test_synthoseis_stage0_artifact_characterization.py`、`test_synthoseis_v5_contracts.py` | 14 passed，2 个需要临时目录的测试未纳入本次命令 |
| 1B | `core/views.py`、`core/view_runner.py`、`core/pipeline.py` | v5 view grammar、随机流和顺序测试 | 通过 |
| 1C | `core/pipeline.py`、`tests/test_synthoseis_stage1_shared_pipeline.py` | shared lifecycle/transaction smoke 与入口 AST 检查 | 通过 |
| 1D | `adapters.py`、`depth/generation.py` | 深度域 v5 `debug_attempt_limit=1` smoke | 5 个父实现、55 个视图，reader 可读 |
| 1E | `adapters.py`、`time/pipeline.py` | Time Adapter 最小 shared-Pipeline smoke | 通过 |
| 1F | `time/calibration_pipeline.py`、`depth/calibration.py`、`core/pipeline.py` | calibrate smoke | `status=success` |
| 1G | `core/pipeline.py`、`core/v5_artifacts.py`、`readers/v5.py`、`core/protocols.py` 删除、`core/reader_contract.py` 删除 | compile/import、`git diff --check`、发布前严格 v5 Reader 门禁、原子发布检查 | core 文件数 23→21；深度 smoke 在 rename 前通过 strict Reader，全部通过 |

本表中的“通过”只表示阶段 1 的代码合同和 smoke 门禁；GINN 采样扩展与五组深度域归因仍按阶段 2、3 处理。

### 阶段 2：GINN 父实现均衡训练

- [x] 升级 GINN v2 实验 schema；
- [x] 删除 `input_seismic_variant`；
- [x] 删除 `balanced_sample_kind`；
- [x] 为每个 synthetic source 物化 `split_assignment_<source_id>.csv`，patch 构建只允许 strict split；
- [x] patch catalog 改为每个父实现 patch 一行；
- [x] 实现 `parent_balanced_seismic_view`；
- [x] 将 view sampling 限定在各 synthetic loss block 内；
- [x] 实现训练和 validation 的完整权重校验；
- [x] 从 train 父实现 base 地震和 canonical background 生成共享 normalization；
- [x] validation 使用固定显式权重；
- [x] 输出 base、逐视图、分组和 paired 指标；
- [x] weighted metric 接入 checkpoint selection 与 deployment eligibility；
- [x] checkpoint 保存视图合同与实际父实现/类型/视图采样计数；
- [x] 拒绝 v4 benchmark 和旧 checkpoint 来源语义；
- [x] 保存 split assignment 内容哈希、view spec hash set 和 science-v3 identity。

阶段完成条件：在相同 seed 下，向 benchmark 增加一个未引用视图不会改变任何已存在训练 item；所有采样与验证聚合值均能从保存的计数和权重独立复算。

阶段 2 的代码收口证据：

| 合同 | Implementation | 验证 |
|---|---|---|
| 父实现切分与 catalog | `ginn_v2/composable.py`、`ginn_v2/data.py` | split contract/hash、parent-owned catalog 测试；一轮 1 epoch 深度 smoke 生成独立 train/validation parent |
| 父实现/视图采样 | `DeterministicParentBalancedViewCycler` | 相同 seed 下 mapping 顺序和未引用 view 不改变既有序列；checkpoint 保存 parent/kind/view 计数 |
| validation 与 checkpoint | `ginn_v2/composable.py`、`ginn_v2/checkpoint.py` | 深度 smoke 输出 base、11 view、3 组 family 和 paired 指标；`load_checkpoint` 严格读取 v5 provenance |
| 发布消费 | `scripts/ginn_v2.py`、`src/cup/synthetic/readers/v5.py` | 深度 v5 smoke 通过严格 reader；predict smoke 输出 99 个 validation patch |

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

所有模型使用第 7.5 节的统一全视图 validation 选择 checkpoint。

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
- 已声明 view 的 `operator_ids` 为空时失败；
- `seismic_views` 或 view 条目包含未知字段时失败；
- 重复 operator ID 失败；
- 重复 view ID 失败；
- 未知 operator kind 失败；
- view 引用未知或重复 operator 失败；
- 缺失、未知、非有限和非法参数失败；
- 域/单位不匹配失败；
- 旧 `lfm.controlled_degraded`、`combined`、`input_seismic_variant` 和 `balanced_sample_kind` 字段失败；
- sampling 和 validation 权重缺失或不归一失败。
- forward/sample operator grammar 违反时失败；
- science-v3 版本链缺失或不匹配时失败。

### 9.2 视图与产物测试

- 只计算、写出配置声明的 views；
- 同 seed、父实现和配置逐值确定；
- 有序组合严格按声明顺序执行；
- 改变算子顺序会改变视图身份和结果；
- forward 参数只触发一次重新正演，且不会覆盖后续 sampled operators；
- canonical view spec 与 SHA-256 可独立复算；
- 同 operator ID/spec 跨视图复用相同随机系数；
- 新增或重排其他视图不改变既有 operator 随机系数；
- base 不出现在 view index；
- 双索引唯一键和引用关系正确；
- realization/view index 只包含成功发布的行；
- 任一 view 失败时父实现事务整体拒绝且不留半成品；
- HDF5 中 canonical background 只物化一次；
- target 唯一路径为 `truth/model_target_log_ai`，reader 独立校验 canonical decomposition；
- HDF5、CSV 和 manifest 不含 LFM degradation 字段；
- HDF5、CSV 和 manifest 不含 `combined_moderate`；
- v5 reader 拒绝 v4；
- view 轴、mask 和有效样点与父实现一致。
- `axis_static` 使用独立的 operator source support，非零位移不因公开 mask 边界丢失支撑；

### 9.3 共享 Pipeline 测试

- 使用测试 Adapter 从同一共享 Pipeline 跑通 calibrate/generate；
- 时间域和深度域公共配置产生一致的公共 resolved 结构；
- 两个 Adapter 的域轴和单位分别保持正确；
- 时间域与深度域走同一索引、writer、manifest、接受率和报告 Implementation；
- 入口不包含公共父实现循环或 view 循环；
- 任一发布步骤失败时不产生成功 run summary。
- acceptance 允许拒绝单个父实现 attempt，同时禁止发布不完整父实现；
- 临时目录通过最终 artifact 校验后才发布成功 summary。

### 9.4 GINN 采样测试

- patch catalog 不按视图复制；
- split assignment 算法、seed、比例和 geometry holdout 规则逐值符合合同；
- patch 构建拒绝 derive split 和不完整 split assignment；
- 父实现选择均衡且确定；
- base/variant 实际计数符合显式权重；
- 逐 view 实际计数符合显式权重；
- 新增未引用 view 不改变既有抽样序列；
- view index 行顺序改变不改变抽样结果；
- `view_weights` mapping 键顺序改变不改变抽样结果；
- 多个 synthetic source 各自保存并消费自己的 split assignment，不互相覆盖；
- clean-only 从不读取 view；
- normalization 不受物化视图数量影响；
- 五组实验的 normalization identity 完全相同且只来自 base/canonical background；
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
- science-v3、projection、view、operator 和 random contract identity 可独立校验；
- realization index 只表达成功父实现，失败 attempt 只进入 rejection records；
- 每个 synthetic source 的 split 只由带固定 contract 的 `split_assignment_<source_id>.csv` 所有；
- GINN 先选父实现再选视图；
- 训练和 validation 权重均为显式合同；
- 五组归因共享 base-only normalization identity；
- forward/sample operator grammar 与 view fingerprint 可独立复算；
- variant 数量不会隐式改变父实现概率；
- 旧 schema 和旧配置字段直接失败；
- 五组深度域归因使用同一验证分布完成；
- 活动视图由用户人工裁决；
- 用户运行全部测试并将结果写回；
- 教程与 ADR 已按当前 Implementation 更新。

允许最终实验结论是某一类或多类失配没有独立收益。无收益不构成代码失败；隐式采样、不可复算指标和域间重复编排才构成合同失败。

## 12. 当前实现收口

本轮已落地的 Implementation 包括：

- Synthoseis-lite v5/science-v3 schema、canonical-only LFM、原子 operator grammar、规范化 view fingerprint 和命名随机流；
- 共享 `SeismicViewPipeline`、真正拥有完整父实现事务的 `SyntheticBenchmarkPipeline`、时间/深度 domain Adapter、成功父实现/视图双索引和严格 v5 reader；入口只负责加载域输入、构造 Adapter 并调用共享 Pipeline；
- 共享 Pipeline 统一拥有 calibrate/generate 的临时目录、preflight、父实现循环、视图物化、HDF5、双索引、QC、接受率、manifest、summary 和最终目录发布；任一父实现视图失败时整组事务回滚；
- 本轮没有新增 `core` 过渡模块，并将样本协议与 reader header 合并到既有公共模块，`src/cup/synthetic/core` 文件数由 23 个收敛为 21 个；
- 父实现事务提交、HDF5 视图路径、base-only normalization、GINN-owned per-source split assignments、父实现均衡视图 sampler 和 weighted validation diagnostics；
- GINN checkpoint 的 benchmark、view、split、normalization 和 sampling provenance；
- 合成基准和 GINN 教程，以及共享 Pipeline ADR。

阶段 1 验收证据：

```powershell
$env:PYTHONPATH = "src"
& <python> -m compileall -q src scripts tests
& <python> -c "import cup.synthetic.pipeline, cup.synthetic.time.pipeline, cup.synthetic.depth.generation, ginn_v2.composable"
```

上述命令已通过。随后使用当前 v5 深度配置执行了一个 `debug_attempt_limit=1`、仅保留 `none` 几何族的真实 smoke：输出目录发布成功，`run_summary.json` 为 `development_limited`，`realization_index.csv` 为 5 个父实现，`seismic_view_index.csv` 为 55 个视图行（5 × 11），manifest 含 contract fingerprint，HDF5 reader 成功读取 5 个父实现及其视图，失败 attempt 只出现在拒绝记录。另一次 `calibrate` smoke 输出 `run_summary.json.status=success`，校准产物完整发布。最小测试 Adapter 也已从同一共享 Pipeline 生成完整 v5 双索引和 summary。

被忽略的 `tests/` 已写入 v5 配置失败、算子随机身份、共享视图 Pipeline、阶段 1 共享生命周期/事务、父实现 sampler 不变性、split 合同和 GINN 验证权重测试，由用户在本机运行。正式五组深度域归因和活动视图人工裁决仍属于下一步实验工作，不在代码 smoke 中替代完成。
