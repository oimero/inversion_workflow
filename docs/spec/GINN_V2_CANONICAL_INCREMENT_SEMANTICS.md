# GINN v2 Canonical Increment 语义重构规格

## 1. 目标

GINN v2 的公共输出统一定义为**规范阻抗增量**（canonical impedance increment）。地质生成器内部的对象扰动、正演观测误差、监督目标和真实工区网络输出使用各自准确的名称。

本文描述目标实现合同。20260706 合成 benchmark、当前四模型 checkpoint 和已有报告作为 full-correction 基线冻结。

## 2. 数学定义与职责边界

令 `m` 表示物理单位下的 `log(AI)`，`P` 表示本合同规定的零相位 Butterworth 低通算子：

```text
canonical_background_log_ai = b = P(m)
canonical_increment_log_ai  = u = m - P(m) = (I - P)m
predicted_log_ai             = external_lfm_log_ai + predicted_increment_log_ai
```

因此恒有：

```text
m = canonical_background_log_ai + canonical_increment_log_ai
```

`P` 是低通算子，不是数学投影。Butterworth 一般满足 `P² != P`，所以 `P(u) == 0` 不是本合同的正确性条件。`I-P` 只表示按上述代数定义构造的互补算子。

各模块职责如下：

| 对象 | 定义 | 所属模块 |
| --- | --- | --- |
| 地质扰动 | 合成地质模型内部相对于背景或状态均值的变化 | 合成生成器 |
| 规范背景 | `P(m)` | 共享阻抗分解模块 |
| 规范阻抗增量 | `m-P(m)` | 合成标签、监督训练、网络输出 |
| 外部 LFM | 由真实井和空间建模得到、符合规范低通合同的背景模型 | 第七步 LFM |
| 正演波形误差 | observed seismic 与 modeled seismic 的差 | physics loss、R1 |
| 阻抗预测误差 | predicted logAI 与已知 target logAI 的差 | 合成评估、井点评估 |

公共字段和产物不得使用无修饰的 `residual` 或 `delta`。局部数值实现可以使用误差变量，但输出名称必须说明被减数、减数或物理对象。

## 3. Canonical increment 合同

### 3.1 固定算子

| 采样域 | 低通实现 | 截止 | 阶数 | 边界 | buffer |
| --- | --- | --- | ---: | --- | ---: |
| 时间域 | zero-phase Butterworth | `15 Hz` | 6 | reflect | `0.4 s` |
| 深度域 | zero-phase Butterworth | `1/400 cycles/m`，即最小波长 `400 m` | 6 | reflect | `400 m` |

低通沿采样轴逐道执行。每个连续有限区段独立滤波，不跨越 NaN 或 invalid 间隙。有限区段长度不足以满足滤波器要求时明确失败。

共享实现位于工作流基础层，合成 benchmark、真实井目标和真实 LFM 构建共同调用。合成标签必须先在完整连续道上计算，再由训练数据集裁剪 patch；patch 内不得重新计算低通或增量。

### 3.2 实验配置

新实验 schema 在 `ginn_v2` 根节点中要求显式的 `increment_contract`。深度域示例：

```yaml
ginn_v2:
  schema_version: ginn_v2_experiment_v2
  experiment_id: depth_tcn_canonical_increment

  increment_contract:
    semantics: canonical_complement_log_ai
    sample_domain: depth
    sample_unit: m
    depth_basis: tvdss
    lowpass:
      kind: butterworth_zero_phase
      order: 6
      cutoff_wavelength_m: 400.0
      buffer_mode: reflect
      buffer_axis_units: 400.0
    low_frequency_qc:
      kind: butterworth_zero_phase
      order: 6
      cutoff_wavelength_m: 800.0
      buffer_mode: reflect
      buffer_axis_units: 400.0
```

时间域使用：

```yaml
  increment_contract:
    semantics: canonical_complement_log_ai
    sample_domain: time
    sample_unit: s
    lowpass:
      kind: butterworth_zero_phase
      order: 6
      cutoff_hz: 15.0
      buffer_mode: reflect
      buffer_axis_units: 0.4
    low_frequency_qc:
      kind: butterworth_zero_phase
      order: 6
      cutoff_hz: 7.5
      buffer_mode: reflect
      buffer_axis_units: 0.4
```

配置必须使用与采样域匹配的 cutoff 字段。深度域要求 `depth_basis: tvdss`；时间域拒绝 `depth_basis`。所有数值显式必填。

benchmark、真实 LFM、实验 manifest 和 checkpoint 记录解析后的完整合同。训练和 R0 校验以下内容一致：

- sample domain、sample unit 和 depth basis；
- filter kind、order、cutoff；
- buffer mode 和 buffer axis units；
- `log(AI)` 值域；
- 输入和输出字段语义。

外部 LFM 在构建阶段满足该合同。GINN v2 消费 LFM 时只做合同校验，不重复低通。

### 3.3 完整训练示例

下面示例给出深度域 synthetic supervised 后接 synthetic physics 的完整配置形状：

```yaml
ginn_v2:
  schema_version: ginn_v2_experiment_v2
  experiment_id: depth_tcn_canonical_increment_then_physics
  seed: 20260712
  device: auto

  increment_contract:
    semantics: canonical_complement_log_ai
    sample_domain: depth
    sample_unit: m
    depth_basis: tvdss
    lowpass:
      kind: butterworth_zero_phase
      order: 6
      cutoff_wavelength_m: 400.0
      buffer_mode: reflect
      buffer_axis_units: 400.0
    low_frequency_qc:
      kind: butterworth_zero_phase
      order: 6
      cutoff_wavelength_m: 800.0
      buffer_mode: reflect
      buffer_axis_units: 400.0

  architecture:
    id: trace_dilated_tcn
    hidden_channels: 32
    depth: 5

  sources:
    synthetic:
      kind: synthoseis_lite
      benchmark_dir: experiments/synthoseis_lite/results/REPLACE_V4_RUN/generate_field_conditioned
      input_seismic_variant: observed_mismatch
      physics_target_variant: model_consistent

  normalization_reference:
    source: synthetic

  patching:
    lateral_samples: 32
    vertical_samples: 128
    lateral_stride: 16
    vertical_stride: 64

  stages:
    - stage_id: synthetic_supervised
      epochs: 10
      steps_per_epoch: 300
      optimizer:
        kind: adamw
        learning_rate: 0.001
        weight_decay: 0.0001
      loss_blocks:
        - block_id: synthetic_increment
          kind: synthetic_supervised
          source: synthetic
          weight: 1.0
          update_interval: 1
          batch_size: 8
          min_valid_samples: 128
          sampling:
            kind: balanced_sample_kind
      validation:
        selection_metric: synthetic_increment.mse
        mode: full

    - stage_id: synthetic_physics
      initialize_from: synthetic_supervised.best
      epochs: 3
      steps_per_epoch: 150
      optimizer:
        kind: adamw
        learning_rate: 0.0001
        weight_decay: 0.0001
      loss_blocks:
        - block_id: synthetic_waveform
          kind: physics
          source: synthetic
          weight: 1.0
          update_interval: 1
          batch_size: 8
          min_valid_samples: 128
          sampling:
            kind: balanced_sample_kind
          increment_l2_weight: 0.01
      validation:
        selection_metric: synthetic_waveform.total
        mode: full

  deployment_checkpoint: synthetic_physics.best
```

时间域配置只替换 `increment_contract` 中的采样合同和 cutoff 字段；architecture、source、stage 和 loss block 组合规则相同。

## 4. 合成 benchmark

### 4.1 数据流

对于每个完整合成真值：

```text
target_log_ai
    |
    +--> P --------------------------------> canonical_background_log_ai
    |                                              |
    +---------------- subtract -------------------> target_increment_log_ai

canonical_background_log_ai --> controlled degradation --> input_lfm_log_ai
```

受控退化可以包含常数偏置、纵向趋势、分区偏置、横向平滑偏置、振幅缩放、过度平滑和局部缺失井控偏置。它只改变 `input_lfm_log_ai`，不改变 `target_increment_log_ai`。

HDF5 基础 realization 至少保存：

```text
/truth/model_target_log_ai
/priors/canonical_background_log_ai
/priors/input_lfm_log_ai
/targets/target_increment_log_ai
```

`target_increment_log_ai` 与 `model_target_log_ai`、`canonical_background_log_ai`、模型采样轴和 `valid_mask_model` 同形。invalid 点保持 NaN。seismic variant 复用父 realization 的三项阻抗数组。

生成器内部的对象级 profile 使用 `geologic_perturbation` 或更具体的地质名称。`seismic_observed - seismic_model_consistent` 使用 `subgrid_forward_error`。这些对象不进入网络增量合同。

### 4.2 版本

新 benchmark schema 为：

```text
synthoseis_lite_v4
```

`synthoseis_lite_v3` benchmark 文件和报告只作为冻结历史基线保留，不可作为 `ginn_v2_experiment_v2` 的训练 source。新 benchmark 需要重新生成，不提供运行时派生标签或 sidecar 转换。

## 5. 网络与训练

### 5.1 网络接口

四类架构保持相同输入：

```text
channel 0: normalized seismic
channel 1: normalized input_lfm_log_ai
channel 2: valid mask
```

网络直接输出物理 `log(AI)` 单位的：

```text
predicted_increment_log_ai
```

最后一层零初始化。零初始化模型必须逐点满足：

```text
predicted_increment_log_ai == 0
predicted_log_ai == input_lfm_log_ai
```

首版不在网络输出或 patch 内执行 `(I-P)` 硬滤波。监督标签定义输出语义，训练与评估使用保守低频 QC 观察偏离。这样避免在短 patch 上引入 Butterworth 边界差异和 stitching 伪影。

### 5.2 合成监督

`synthetic_supervised` 直接读取 benchmark 保存的标签：

```text
L_supervised = masked_mse(
    predicted_increment_log_ai,
    target_increment_log_ai
)
```

最终监督 mask 要求 seismic、input LFM、target increment 和 target logAI 有限，并继续使用 block 的 `min_valid_samples`。

### 5.3 真实井监督

每口井先在完整连续测井区段上应用相同 `P`：

```text
well_canonical_background_log_ai = P(well_log_ai)
well_target_increment_log_ai = well_log_ai - well_canonical_background_log_ai
```

`real_well_supervised` 在井样点监督 `well_target_increment_log_ai`。输入 LFM 仍来自对应真实工区 source。井上的外部 LFM 背景误差由 LFM 工作流和最终 AI 指标单独衡量，不并入网络增量标签。

### 5.4 Physics

合成 physics 的正演阻抗为：

```text
physics_pred_log_ai = canonical_background_log_ai
                      + predicted_increment_log_ai
```

真实 physics 的正演阻抗为：

```text
physics_pred_log_ai = external_lfm_log_ai
                      + predicted_increment_log_ai
```

波形损失、时间/深度正演分派、连续有效段和 forward-support-safe mask 延续当前合同。增量正则项命名为：

```text
increment_l2_weight
L_increment = masked_mean(predicted_increment_log_ai ** 2)
L_physics = L_waveform + increment_l2_weight * L_increment
```

任一包含 `physics` block 的阶段必须从一个已经完成的监督 checkpoint 初始化。合法监督祖先是：

- `synthetic_supervised`；
- `real_well_supervised`。

解析器沿 `initialize_from` 引用链检查祖先。以下情况明确失败：

- 第一阶段只包含 physics；
- 第一阶段同时包含 supervised 和 physics；
- physics 阶段从 `zero` 初始化；
- physics 阶段引用的 checkpoint 祖先链中没有已完成监督阶段。

监督阶段之后可以有一个或多个 physics 阶段，也可以再接监督阶段。deployment checkpoint 可以指向任一合法阶段的 best/final。

## 6. R0、R1 与反事实语义

### 6.1 R0

R0 对每个 patch 预测 `predicted_increment_log_ai`，按现有 uniform 策略拼接增量，再与同一点的 external LFM 相加：

```text
predicted_log_ai = external_lfm_log_ai + predicted_increment_log_ai
```

生产推理继续满足：

```text
(prediction_support_count > 0) == valid_mask_model
```

valid 点的增量和最终阻抗均有限；invalid 点保持 NaN。xline 步长 4 只由显式坐标轴决定，不参与数组步长猜测。

R0 模型产物至少包含：

```text
predicted_increment_log_ai
predicted_log_ai
input_lfm_log_ai
prediction_support_count
max_context_valid_fraction
valid_mask_model
ilines
xlines
samples
```

### 6.2 LFM 反事实

对相同 seismic 和两个满足合同的 LFM 输入：

```text
u1 = f(seismic, lfm_1)
u2 = f(seismic, lfm_2)
```

新目标的期望关系为：

```text
u2 - u1 ~= 0
predicted_log_ai_2 - predicted_log_ai_1
    ~= lfm_2 - lfm_1
```

报告至少包括：

```text
increment_input_sensitivity_rms = RMS(u2 - u1)
lfm_replacement_rms = RMS(lfm_2 - lfm_1)
conditional_to_lfm_ratio = increment_input_sensitivity_rms
                           / lfm_replacement_rms
```

该比值只用于模型比较，不设固定通过阈值。

### 6.3 R1

R1 使用 `predicted_log_ai` 正演，并将 observed-minus-modeled 明确命名为 waveform error。R1 同时记录 experiment ID、increment contract、external LFM variant 和 deployment checkpoint。

## 7. 低频 QC

定义独立分析低通算子 `P_low`：

- 时间域截止 `7.5 Hz`；
- 深度域截止 `1/800 cycles/m`；
- 六阶、零相位、与 canonical lowpass 相同的边界和 buffer 合同。

对任一增量 `u` 报告：

```text
low_frequency_rms(u) = RMS(P_low(u))

low_frequency_energy_ratio(u) =
    sum(P_low(u) ** 2) / (sum(u ** 2) + epsilon)
```

合成监督报告 target 和 prediction 两组指标及其差值。physics 阶段另报告：

```text
physics_low_frequency_drift_rms =
    RMS(P_low(increment_after - increment_before))
```

真实工区没有 target increment，只报告 prediction 指标和 physics 前后漂移。所有指标按完整连续道计算，不在单个训练 patch 内计算。QC 是诊断产物，不删除预测样点，也不作为默认 checkpoint gate。

## 8. Schema 与迁移

| 产物 | 新 schema |
| --- | --- |
| Synthoseis-lite benchmark | `synthoseis_lite_v4` |
| 实验配置和 manifest | `ginn_v2_experiment_v2` |
| checkpoint | `ginn_v2_checkpoint_v5` |
| 合成 prediction | `ginn_v2_prediction_v3` |
| R0 单模型产物 | `real_field_zero_shot_model_v5` |
| R0 汇总 | `real_field_zero_shot_summary_v5` |
| R1 汇总 | `real_field_forward_diagnostic_summary_v6` |

checkpoint 必须记录完整 increment contract、输入通道合同、输出语义、训练 source 和阶段祖先。新训练、predict、R0 和 R1 明确拒绝以下输入：

- `synthoseis_lite_v3` benchmark；
- `ginn_v2_experiment_v1` 配置或 manifest；
- `ginn_v2_checkpoint_v4`；
- 输出语义为 `physical_delta_log_ai` 或 `truth-input_lfm` 的产物；
- `delta_l2_weight` 字段。

错误信息包含实际 schema、期望 schema 和本文路径。不提供字段猜测、自动迁移、checkpoint 权重转换或双语义运行模式。

本重构沿用项目已有的直接上游合同记录，不新增逐文件 SHA-256、递归 provenance、数组摘要或图件指纹。

## 9. 实施顺序

1. 在工作流基础层建立共享 canonical decomposition，实现完整连续道的时间/深度 Butterworth 背景和增量计算。
2. 升级 Synthoseis-lite schema，生成新 benchmark，并让 controlled degradation 只作用于输入 LFM。
3. 升级 GINN v2 配置、reader、训练 batch、损失字段、checkpoint 和评估命名。
4. 将真实井监督目标切换为井的 canonical increment，并加入 physics 监督祖先校验。
5. 升级 R0/R1 产物和反事实报告，保持全覆盖与显式坐标合同。
6. 运行新旧目标对照实验；旧四模型及 20260706 结果只作为冻结比较输入。

## 10. 测试与验收矩阵

### 10.1 算子和数据

- 时间域与深度域均满足 `m == P(m) + (m-P(m))`，误差只来自浮点舍入。
- 连续有限段独立滤波，NaN 间隙两侧互不影响；短段明确失败。
- shared operator 与真实 LFM 构建调用同一实现和参数解释。
- benchmark 中 target increment 与 controlled degradation 的常数偏置、趋势、分区偏置和过度平滑参数无关。
- patch 标签等于完整 realization 标签的直接切片。

### 10.2 网络和训练

- 四类架构均输出与输入空间同形的 `predicted_increment_log_ai`。
- 零初始化输出严格为零，最终阻抗严格等于输入 LFM。
- synthetic supervised 和 real-well supervised 前向、反向及梯度有限。
- 时间域和深度域 supervised 后 physics 前向、反向及梯度有限。
- physics-first、zero-initialized physics 和无监督祖先的 physics 配置明确失败。
- `increment_l2_weight` 生效，`delta_l2_weight` 明确失败。

### 10.3 推理与评估

- R0 valid 点全部得到有限增量和最终阻抗，invalid 点保持 NaN。
- inline 步长 1、xline 步长 4、多剖面和完整体均保持显式坐标。
- 低频 RMS、能量占比和 physics 漂移可从保存数组复算。
- 成对 LFM 反事实分别报告增量条件效应和 LFM 直接替换效应。
- 新 CLI 对旧 benchmark、配置、checkpoint 和预测 schema 给出明确错误。

### 10.4 新旧目标对照

对照使用相同架构、seismic 输入、split、seed、训练预算和 normalization，分别报告：

- canonical increment MSE、RMSE 和相关系数；
- `canonical_background + predicted_increment` 的合成 AI 指标；
- `input/external LFM + predicted_increment` 的最终 AI 指标；
- geometry holdout 和 pinchout；
- 低频 QC；
- LFM 反事实增量不变性；
- 真实井最终 AI；
- R1 waveform error。

代码验收以合同、数值正确性和错误边界为硬门禁。模型质量作为消融结果报告，不要求新语义在所有效果指标上全面超过 full-correction 基线。

## 11. 首版边界

首版保持单输出头和三通道输入。以下内容由后续独立消融决定：

- 删除或弱化 LFM 输入通道；
- seismic-band 与 prior-supported high-frequency 双输出头；
- 输出端可微 Butterworth 互补滤波；
- canonical cutoff 扫描；
- physics-first 诊断模式；
- 独立低频背景适配器。
