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

规范阻抗增量是阻抗分解和接口语义，不是可观测性声明。`u` 包含 canonical cutoff 以上的全部阻抗成分，其中只有一部分受地震有效频带约束，更高频成分由合成训练分布和地质先验约束。产物和报告不得把 `predicted_increment_log_ai` 描述为全部由地震唯一恢复的高频阻抗。

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
| 时间域 | SciPy Butterworth SOS + forward/backward | `15 Hz` | 单程设计 6 阶，最终等效 12 阶 | reflect | `0.4 s` |
| 深度域 | SciPy Butterworth SOS + forward/backward | `1/400 cycles/m`，即最小波长 `400 m` | 单程设计 6 阶，最终等效 12 阶 | reflect | `400 m` |

低通沿采样轴逐道执行。每个连续有限区段独立滤波，不跨越 NaN 或 invalid 间隙。有限区段长度不足以满足滤波器要求时明确失败。

共享实现位于工作流基础层，合成 benchmark、真实井目标和真实 LFM 构建共同调用。合成标签必须先在完整连续道上计算，再由训练数据集裁剪 patch；patch 内不得重新计算低通或增量。

数值实现固定为：

```text
scipy.signal.butter(
    N=6,
    Wn=declared_cutoff,
    btype="lowpass",
    fs=1/sample_step,
    output="sos",
)

np.pad(segment, pad_samples, mode="reflect")
scipy.signal.sosfiltfilt(sos, padded, padtype=None)
crop original segment
```

这里的 `15 Hz` 或 `1/400 cycles/m` 是单程六阶 Butterworth 的设计截止点：单程幅度为 `1/sqrt(2)`（约 −3.0103 dB），前后向最终幅度为 `1/2`（约 −6.0206 dB）。不做最终 −3 dB 截止补偿。

滤波输入和持久化采样轴统一使用 float64。入口数值轴转换为 float64；NaN、Inf、非递增、重复和非等间隔轴失败。`increment_contract.sample_interval` 是权威采样间隔，规则轴由它重建并验证：

```text
expected_axis = sample_axis[0] + arange(n_sample) * sample_interval
allclose(sample_axis, expected_axis, rtol=sample_interval_relative_tolerance,
        atol=sample_interval_absolute_tolerance)
```

设计阶数 6 产生 3 个 SOS section。`21` 是本项目保留的 structural minimum，不是 `padtype=None` 对 SciPy 的必要 padlen。实际最小连续有限段长度为：

```text
structural_minimum_segment_samples = 3 * (2 * n_sos + 1) = 21
pad_samples = ceil(buffer_axis_units / sample_step)
minimum_segment_samples = max(21, pad_samples + 1)
```

这等价于同时要求 `pad_samples < segment_samples`。任一条件不满足时明确失败，不切换滤波实现或边界模式。

### 3.2 实验配置

新实验 schema 在 `ginn_v2` 根节点中要求显式的 `increment_contract`。深度域示例：

```yaml
ginn_v2:
  schema_version: ginn_v2_experiment_v2
  experiment_id: depth_tcn_canonical_increment

  increment_contract:
    contract_version: canonical_increment_v1
    semantics: canonical_complement_log_ai
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
    lowpass:
      implementation: scipy_butter_sosfiltfilt
      design_order: 6
      effective_zero_phase_order: 12
      cutoff_definition: single_pass_minus_3db_final_minus_6db
      cutoff_wavelength_m: 400.0
      buffer_mode: reflect
      buffer_axis_units: 400.0
    low_frequency_qc:
      implementation: scipy_butter_sosfiltfilt
      design_order: 6
      effective_zero_phase_order: 12
      cutoff_definition: single_pass_minus_3db_final_minus_6db
      cutoff_wavelength_m: 800.0
      buffer_mode: reflect
      buffer_axis_units: 400.0
```

时间域使用：

```yaml
  increment_contract:
    contract_version: canonical_increment_v1
    semantics: canonical_complement_log_ai
    sample_domain: time
    sample_unit: s
    sample_interval: 0.002
    sample_axis_uniform: true
    sample_axis_dtype: float64
    sample_interval_relative_tolerance: 1.0e-6
    sample_interval_absolute_tolerance: 1.0e-9
    value_domain: log(AI)
    log_base: natural
    ai_unit_convention: m/s*g/cm3
    lowpass:
      implementation: scipy_butter_sosfiltfilt
      design_order: 6
      effective_zero_phase_order: 12
      cutoff_definition: single_pass_minus_3db_final_minus_6db
      cutoff_hz: 15.0
      buffer_mode: reflect
      buffer_axis_units: 0.4
    low_frequency_qc:
      implementation: scipy_butter_sosfiltfilt
      design_order: 6
      effective_zero_phase_order: 12
      cutoff_definition: single_pass_minus_3db_final_minus_6db
      cutoff_hz: 7.5
      buffer_mode: reflect
      buffer_axis_units: 0.4
```

配置必须使用与采样域匹配的 cutoff 字段。深度域要求 `depth_basis: tvdss`；时间域拒绝 `depth_basis`。所有数值显式必填。

benchmark、真实 LFM、实验 manifest 和 checkpoint 记录解析后的完整合同。训练和 R0 校验以下内容一致：

- sample domain、sample unit 和 depth basis；
- filter implementation、design/effective order、cutoff definition 和 cutoff；
- buffer mode 和 buffer axis units；
- sample interval、axis dtype 与等间隔容差；
- `log(AI)` 值域；
- 输入和输出字段语义。

外部 LFM 在构建阶段满足该合同。GINN v2 消费 LFM 时只做合同校验，不重复低通。

### 3.3 外部 LFM 的生产与校验

消费端不通过 `P(external_lfm) ~= external_lfm` 判断合同是否成立。硬校验只读取 LFM 生产者写入的元数据：

```text
producer_schema
sample_domain
sample_unit
sample_interval
sample_axis_uniform
sample_axis_dtype = float64
sample_interval_relative_tolerance
sample_interval_absolute_tolerance
depth_basis
value_domain = log(AI)
log_base = natural
ai_unit_convention = m/s*g/cm3
canonical_lowpass_applied_to
canonical_lowpass_application_count
well_control_lowpass_application_count = 1
final_volume_lowpass_application_count = 0
post_lowpass_vertical_warp_applied
filter_implementation
design_order
effective_zero_phase_order
cutoff_definition
cutoff_hz | cutoff_wavelength_m
buffer_mode
buffer_axis_units
```

两个 application count 分别描述井控进入空间建模前的低通和最终规则体上的低通。首版 LFM 合同固定为井控阶段 1 次、最终体阶段 0 次。`post_lowpass_vertical_warp_applied` 记录低通之后是否发生比例分层、纵向拉伸压缩或其他垂向坐标变换。空间趋势拟合或 kriging 本身不增加低通计数。

井控经过低通并不能证明空间建模后的最终规则体严格位于 canonical background 频带内。外部 LFM 的严格语义是 **contract-compatible deployment background**，不保证等于未知真实阻抗的 `P(m)`；它与 canonical increment 可能存在频带重叠。

LFM 生产者在具备最终体 QC 的阶段应在最终规则模型体上逐道计算并汇总：

```text
final_background_complement_response_rms = RMS(LFM - P(LFM))
final_background_complement_response_ratio =
    sum((LFM - P(LFM)) ** 2) / (sum(LFM ** 2) + epsilon)
final_background_max_trace_response_ratio
post_lowpass_vertical_warp_applied
```

合同同时包含 `final_background_complement_response_status`。状态为
`measured` 时上述三个值必须是有限非负数；阶段 2.5 的 Synthoseis v4 writer
尚未计算最终体 QC，因此写入 `not_computed` 和三个 `null`，不把未测量结果写成
零。上述指标只用于对比 LFM 方法和发现空间建模引入的纵向频带变化，不设通过阈值，
也不能证明数组由指定生产流程生成。GINN v2 记录并透传该 QC，不在消费端重复生成
或修改 LFM。

### 3.4 完整训练示例

下面示例给出深度域 synthetic supervised 后接 synthetic physics 的完整配置形状：

```yaml
ginn_v2:
  schema_version: ginn_v2_experiment_v2
  experiment_id: depth_tcn_canonical_increment_then_physics
  seed: 20260712
  device: auto

  increment_contract:
    contract_version: canonical_increment_v1
    semantics: canonical_complement_log_ai
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
    lowpass:
      implementation: scipy_butter_sosfiltfilt
      design_order: 6
      effective_zero_phase_order: 12
      cutoff_definition: single_pass_minus_3db_final_minus_6db
      cutoff_wavelength_m: 400.0
      buffer_mode: reflect
      buffer_axis_units: 400.0
    low_frequency_qc:
      implementation: scipy_butter_sosfiltfilt
      design_order: 6
      effective_zero_phase_order: 12
      cutoff_definition: single_pass_minus_3db_final_minus_6db
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

  deployment_checkpoint: synthetic_supervised.best
```

时间域配置只替换 `increment_contract` 中的采样合同和 cutoff 字段；architecture、source、stage 和 loss block 组合规则相同。示例中的 physics checkpoint 是诊断产物，默认部署模型是 `synthetic_supervised.best`。

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

每个加性退化场先沿完整采样轴通过退化低通 `P_d`：

```text
d_lfm = P_d(eta)
input_lfm_log_ai = canonical_background_log_ai + d_lfm
```

`P_d` 使用与 canonical `P` 相同的 SciPy SOS 数值合同，其截止频率不得高于 canonical cutoff：时间域 `cutoff_hz <= 15`，深度域 `cutoff_wavelength_m >= 400`。分区和局部缺失井控扰动的边界也必须在相加前通过 `P_d`，避免阶跃将中高频内容写入 LFM。过度平滑可以对 canonical background 使用更低 cutoff。常数振幅缩放产生的背景差值天然沿用 canonical background 的带宽。

HDF5 基础 realization 至少保存：

```text
/truth/model_target_log_ai
/priors/canonical_background_log_ai
/priors/input_lfm_variants/<lfm_variant_id>/log_ai
/targets/target_increment_log_ai
```

reader 根据训练采样器选定的 `lfm_variant_id` 将对应数组暴露为 `input_lfm_log_ai`。`target_increment_log_ai` 与 `model_target_log_ai`、`canonical_background_log_ai`、模型采样轴和单一 `valid_mask` 同形。invalid 点保持 NaN。seismic variant 复用父 realization 的三项阻抗数组。

每个父 realization 至少生成两个不同的 `input_lfm_log_ai` variant，共享相同 seismic、canonical background 和 target increment。benchmark 在父 realization 下保存 LFM variant，而不是把同一 patch 按 variant 复制成多行。训练采样器先按现有父 realization/seismic sample 规则选择样本，再从该父 realization 的 LFM variant 中确定性均匀选择一个，避免 LFM variant 数量改变父样本权重。首版不增加 paired consistency loss。

生成器内部的对象级 profile 使用 `geologic_perturbation` 或更具体的地质名称。`seismic_observed - seismic_model_consistent` 使用 `subgrid_forward_error`。这些对象不进入网络增量合同。

### 4.2 有效区域与正演支持

Synthoseis v4 manifest 通过 `mask_contract=single_valid_mask_v1` 固定公共掩码：
`masks/valid_mask` 表示完整目标 ROI。合成生成器使用上下文和 halo 保证 ROI 内的
target、canonical background、increment、input LFM、observed seismic 与
model-consistent seismic 有限；支持不足时拒绝当前 attempt，而不是改变 ROI。

高分辨率正演支持数组可以作为生成 QC 保存，但不作为训练 mask。GINN v2 的三个
输入通道仍为 normalized seismic、input LFM 和 `valid_mask`。physics 的固定 halo 与
central-crop 规则属于后续训练阶段的 patch 合同，本阶段不在生成器中实现。
patch 外的 seismic/LFM/physics 数值仅作有限的零填充，训练和正演 loss 始终由
`valid_mask` 决定，不把数值填充当作新的有效区域。

### 4.3 版本

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

合成监督直接使用 benchmark 的 `valid_mask`；reader 已在该区域完成 seismic、input LFM、target increment 和 target logAI 的有限性断言，block 只使用 `min_valid_samples` 判断 patch 是否可训练。

### 5.3 真实井监督

真实井从第六步输出的预处理后 canonical `log(AI)` 和井轨迹开始。低通只在网络最终输出所对应的规则模型轴上执行。

深度域顺序固定为：

```text
预处理后井 logAI（MD）
-> 使用井轨迹映射到 TVDSS
-> 按有效连续段重采样到目标等间隔 TVDSS 模型轴
-> 在每个重采样后的连续有限段应用 depth canonical P
-> 计算 well_target_increment_log_ai
```

时间域顺序固定为：

```text
预处理后井 logAI（MD）
-> 使用最终井震标定时深关系映射到 TWT
-> 按有效连续段重采样到目标等间隔 TWT 模型轴
-> 在每个重采样后的连续有限段应用 time canonical P
-> 计算 well_target_increment_log_ai
```

重采样不得跨越原始无效间隙。深时转换、轨迹映射和重采样之前不应用 canonical `P`，也不复用其他采样轴上计算的背景或增量。完成上述顺序后：

```text
well_canonical_background_log_ai = P(well_log_ai)
well_target_increment_log_ai = well_log_ai - well_canonical_background_log_ai
```

`real_well_supervised` 在井样点监督 `well_target_increment_log_ai`。输入 LFM 仍来自对应真实工区 source。井上的外部 LFM 背景误差由 LFM 工作流和最终 AI 指标单独衡量，不并入网络增量标签。

### 5.4 Physics

合成数据明确区分两种闭环。canonical closure 用于合成 physics loss，检验网络是否恢复 canonical increment：

```text
canonical_closure_pred_log_ai = canonical_background_log_ai
                                + predicted_increment_log_ai
```

deployment closure 使用受控退化 LFM，模拟真实部署组合，只作为合成诊断：

```text
deployment_closure_pred_log_ai = input_lfm_log_ai
                                 + predicted_increment_log_ai
```

真实 physics 和 R1 只有 deployment closure：

```text
deployment_closure_pred_log_ai = external_lfm_log_ai
                                 + predicted_increment_log_ai
```

合成 physics block 的 waveform loss 固定使用 `canonical_closure_pred_log_ai`。合成报告同时输出 canonical closure 与 deployment closure 的 AI 和 waveform 指标，字段名必须包含 closure 类型。

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

监督阶段之后可以有一个或多个 physics 阶段，也可以再接监督阶段。physics-only 阶段按 waveform metric 选出的 best/final 是诊断 checkpoint，不能被 `deployment_checkpoint` 引用。

可部署 checkpoint 仅有两类：

1. 只包含监督 block 的阶段 best/final；
2. 同时包含 physics 和 `synthetic_supervised` block、从监督 checkpoint 初始化，并使用该 synthetic block 的 `<block_id>.mse` 作为 selection metric 的阶段 best/final。

仅由 `real_well_supervised` 提供监督的联合 physics 阶段标记为 experimental，即使按井点 MSE 选优也不自动具备部署资格。稀疏井指标不能约束井间区域沿 physics 多解方向的漂移。

独立 AI 或井上 gate 可以决定后续实验是否采用 physics 配方，但首版不实现人工 promotion、相对于前一 checkpoint 的 increment anchor 或轻量参数适配器。waveform-best 和 real-well-plus-physics checkpoint 都不能直接提升为部署模型。标准示例部署 `synthetic_supervised.best`。

## 6. R0、R1 与反事实语义

### 6.1 R0

R0 对每个 patch 预测 `predicted_increment_log_ai`，按现有 uniform 策略拼接增量，再与同一点的 external LFM 相加：

```text
predicted_log_ai = external_lfm_log_ai + predicted_increment_log_ai
```

生产推理继续满足：

```text
(prediction_support_count > 0) == valid_mask
```

valid 点的增量和最终阻抗均有限；invalid 点保持 NaN。xline 步长 4 只由显式坐标轴决定，不参与数组步长猜测。

R0 模型产物至少包含：

```text
predicted_increment_log_ai
predicted_log_ai
input_lfm_log_ai
prediction_support_count
max_context_valid_fraction
valid_mask
ilines
xlines
samples
```

R0 summary 同时记录 external LFM 的两个 application count、垂向 warp 标志和最终体 complement-response QC。它们描述 deployment background 的来源和可能的频带重叠，不改变预测数组。

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
- 单程设计六阶、最终等效十二阶、与 canonical lowpass 相同的 SciPy SOS、截止定义、边界和 buffer 合同。

对任一增量 `u` 报告：

```text
low_frequency_rms(u) = RMS(P_low(u))

conservative_low_frequency_response_ratio(u) =
    sum(P_low(u) ** 2) / (sum(u ** 2) + epsilon)
```

由于 `P_low` 不是正交频带投影，`conservative_low_frequency_response_ratio` 不是严格的低频能量百分比，只用于相同采样合同下不同模型和阶段之间的相对比较。

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
| 真实工区 LFM run | `real_field_lfm_run_v4` |
| 真实工区 LFM variant | `real_field_lfm_variant_v4` |
| 实验配置和 manifest | `ginn_v2_experiment_v2` |
| checkpoint | `ginn_v2_checkpoint_v5` |
| 合成 prediction | `ginn_v2_prediction_v3` |
| R0 单模型产物 | `real_field_zero_shot_model_v5` |
| R0 汇总 | `real_field_zero_shot_summary_v5` |
| R1 汇总 | `real_field_forward_diagnostic_summary_v6` |

checkpoint 必须记录完整 increment contract、输入通道合同、输出语义、训练 source 和阶段祖先。新训练、predict、R0 和 R1 明确拒绝以下输入：

- `synthoseis_lite_v3` benchmark；
- `real_field_lfm_variant_v3` 及缺少 canonical lowpass 生产元数据的 LFM；
- `ginn_v2_experiment_v1` 配置或 manifest；
- `ginn_v2_checkpoint_v4`；
- 输出语义为 `physical_delta_log_ai` 或 `truth-input_lfm` 的产物；
- `delta_l2_weight` 字段。

错误信息包含实际 schema、期望 schema 和本文路径。不提供字段猜测、自动迁移、checkpoint 权重转换或双语义运行模式。

本重构沿用项目已有的直接上游合同记录，不新增逐文件 SHA-256、递归 provenance、数组摘要或图件指纹。

实验 manifest 为每个 stage 记录 `deployment_eligible` 及判定原因。配置解析时即校验 `deployment_checkpoint` 指向可部署 stage，不能在 R0 阶段临时绕过。

## 9. 实施顺序

实现拆成 11 个可独立合并的阶段。阶段 1–9 保持当前正式入口和冻结产物可用；新链路使用独立实验入口，并且该入口只接受本规格的新 schema。阶段 10 切换正式默认入口，阶段 11 删除被替代的生产实现。不存在同一配置、reader 或 checkpoint 的双语义解释。

### 9.1 共享数学基础层

只实现：

- 时间域和深度域 canonical lowpass；
- `background=P(m)` 与 `increment=m-P(m)`；
- float64 规则轴验证、连续有限段和短段失败；
- increment contract 的值对象、序列化和精确比较。

这一阶段不修改 benchmark、训练语义、checkpoint 或 CLI。完成条件是第 10.1 节中与算子有关的测试全部通过。回滚只删除未被生产路径调用的基础模块。

### 9.2 Synthoseis-lite v4 生产端

建立只写 `synthoseis_lite_v4` 的 versioned writer，生成：

- canonical background；
- canonical increment；
- 每个父 realization 的多个带限 LFM variant；
- variant 共享标签和确定性选择元数据。

先生成一个小型 time fixture 和一个小型 depth fixture，不运行完整 D1 benchmark。正式训练入口仍只消费现有 schema。完成条件是共享标签、带限退化、完整道标签和 patch 直接切片测试通过。

fixture 和后续完整 benchmark 均保存在新的不可变 run 目录；通过显式路径复用，不增加内容哈希或额外缓存系统。

### 9.3 最小 synthetic supervised 垂直切片

新增只接受以下组合的实验入口：

```text
synthoseis_lite_v4
-> ginn_v2_experiment_v2
-> trace_conv1d
-> synthetic_supervised
-> ginn_v2_checkpoint_v5
-> ginn_v2_prediction_v3
```

只实现 normalization、reader、batch、masked increment MSE、best/final checkpoint 和最小 synthetic prediction。physics、真实井、R0、R1 和反事实报告均不进入此阶段。

完成条件：一次 1 epoch smoke 可以从 v4 fixture 训练、恢复 checkpoint 并输出 `predicted_increment_log_ai`；零初始化、标签语义和 checkpoint manifest 可复算。当前正式训练入口保持不变。

### 9.4 扩展四类架构

将阶段 9.3 的同一输入、输出和 checkpoint 接口扩展到：

- `trace_conv1d`；
- `trace_dilated_tcn`；
- `trace_lateral_mixer`；
- `patch_conv2d`。

本阶段只运行 synthetic supervised smoke，不改变 loss 或报告语义。四类架构全部满足同形输出、有限梯度和零初始化后，synthetic supervised 主干才视为完成。

### 9.5 Synthetic closure 与报告

先以纯诊断方式实现：

- canonical closure；
- deployment closure；
- increment fidelity；
- LFM-only、两类 closure AI 和 waveform 指标；
- 保守低频响应 QC。

本阶段不让 waveform loss 参与反向传播。完成条件是 `canonical_background + target_increment == truth`，所有保存数组和指标可以独立复算，两类 closure 的字段不会混用。

### 9.6 Synthetic physics 与部署资格

在阶段 9.5 的 closure 已验证后加入：

- synthetic physics loss；
- supervised ancestor 校验；
- `deployment_eligible`；
- physics 前后低频漂移；
- physics-only 诊断 checkpoint；
- dense synthetic supervised + physics 的部署资格规则。

完成条件包括有限梯度、非法 physics-first 明确失败、waveform-best 无法部署，以及标准示例继续部署 supervised best。该阶段不接真实工区 source。

### 9.7 真实 LFM v4 与真实井监督

独立实现：

- `real_field_lfm_run_v4` 和 `real_field_lfm_variant_v4` 元数据；
- 最终规则体 complement-response QC；
- MD 到 TVDSS/TWT 的固定转换顺序；
- 最终规则轴重采样和连续有效段；
- `well_target_increment_log_ai`；
- real-well supervised 的采样与指标。

使用固定的小型时间域井 fixture 和深度域井 fixture 做 golden test，确认低通发生在最终规则模型轴。此阶段不启用 real physics，也不运行完整工区 R0。

### 9.8 Real physics 实验路径

在真实 LFM 和井监督合同稳定后加入 real physics：

- deployment closure forward；
- waveform loss 和 increment L2；
- physics 前后漂移 QC；
- real-well-plus-physics 的 experimental 标记。

所有 real physics checkpoint 均不能自动成为 deployment checkpoint。完成条件是时间域和深度域前向、反向、连续段 mask 和 experimental 资格测试通过。

### 9.9 R0、R1 与反事实报告

新 checkpoint 稳定后再升级生产消费端：

- R0 的 increment stitching、最终阻抗和全覆盖；
- 多剖面、完整体和 xline 步长 4；
- external LFM 元数据与最终体 QC 透传；
- R1 deployment closure；
- paired LFM 反事实；
- v5/v6 summary 和完整 report 字段。

完成条件是第 10.3 节全部通过，并且 R0/R1 只接受新 checkpoint 与新 LFM schema。当前正式 R0/R1 默认入口仍未切换。

### 9.10 完整对照与默认入口切换

前九个阶段通过后才生成完整 D1 v4 benchmark，并运行第 10.4 节的新旧目标对照。固定架构、split、seed、训练预算和 normalization，分别判断：

1. 新链路合同和数值实现是否正确；
2. canonical increment 模型效果如何；
3. 是否将 canonical increment 设为正式训练和部署入口。

只有此阶段修改默认 CLI 和教程入口。切换提交同时让正式新入口明确拒绝旧 schema；旧 benchmark、checkpoint 和报告继续保留在原目录，不改写也不迁移。若切换后发现阻断问题，回滚默认入口提交即可，前九阶段的 versioned 模块和新产物无需删除。

### 9.11 旧实现退役与代码清理

阶段 9.10 切换完成并通过一次完整 regression smoke 后，删除旧 full-correction 生产实现。历史可复现性由不可变产物、Git 历史和冻结报告保证，不由在当前生产代码中永久保留旧实现保证。

本阶段从正式代码树删除：

- `synthoseis_lite_v3` 写入路径和只服务该 schema 的生成分支；
- `ginn_v2_experiment_v1` parser、默认值和配置分派；
- `ginn_v2_checkpoint_v4` 加载、恢复和转换路径；
- `physical_delta_log_ai`、`truth-input_lfm`、`target_delta`、`pred_delta` 和 `pred_delta_vs_lfm` 的公共字段、数据类成员和写出路径；
- `delta_l2_weight` alias、fallback 和兼容解析；
- patch 内计算 `truth-input_lfm` 标签的实现；
- 被新 reader、writer、loss、prediction、R0 或 R1 替代的函数、类和模块；
- 被 v4 reader 取代的 `src/cup/synthetic/readers/time_v2.py`、`depth_v2.py`，以及被新真实井监督模块取代的 `src/ginn_v2/real_delta.py`；
- 只验证退役生产行为的测试、fixture、示例配置、教程和 notebook；
- 迁移期实验入口、adapter、wrapper、feature flag 和双路径分派。

被替代的模块必须真正删除或由当前语义实现原位取代，不能仅改名为 `legacy`、`old`、`compat`、`deprecated` 或退休版本后缀后继续留在 `src/`、`scripts/` 或正式实验目录。

当前树不建立 executable legacy baseline 工具目录。确需重新执行旧基线时，使用冻结报告记录的 Git commit、环境和原始配置 checkout 对应版本运行。任何未来例外都必须先修改本文，逐项写明保留原因、唯一调用者和删除条件；首版 executable legacy allowlist 为空。

旧 schema 和字段名称只允许出现在：

- 本规格的迁移与清理说明；
- 明确断言失败的 migration tests；
- 不参与当前代码扫描的冻结 benchmark、checkpoint 和报告目录。

清理提交必须附带删除清单，按模块、公共符号、CLI、配置、测试和文档分类。清理完成后运行第 10.1–10.5 节的全部测试和 smoke。阶段 9.11 之前发现问题回滚阶段 9.10 的默认入口提交；阶段 9.11 之后发现阻断问题，按顺序 revert 清理和默认切换提交，不恢复兼容 fallback。

### 9.12 合并纪律

每个阶段对应一个或少量紧邻提交，并满足：

- 合并前本阶段测试和 smoke 完成；
- 主分支现有默认工作流保持可运行，直到阶段 9.10；阶段 9.11 后当前树只保留新生产链路；
- 后续阶段只能消费前一阶段已冻结的公共合同；
- schema、字段或数值定义变化回到本文修订，不在调用端猜测；
- 完整 benchmark 和长训练只在需要它们的阶段运行。

## 10. 测试与验收矩阵

### 10.1 算子和数据

- 时间域与深度域均满足 `m == P(m) + (m-P(m))`，误差只来自浮点舍入。
- 连续有限段独立滤波，NaN 间隙两侧互不影响；短段明确失败。
- 采样轴按 float64 保存，并按权威 `sample_interval` 重建验证；非等间隔轴和超出绝对容差的坐标明确失败。
- 单程设计 cutoff 响应约为 −3.0103 dB，前后向最终响应约为 −6.0206 dB；设计 6 阶对应最终等效 12 阶。
- structural minimum 固定为 21 个样点，实际 minimum 为 `max(21, pad_samples+1)`。
- shared operator 与真实 LFM 构建调用同一实现和参数解释。
- benchmark 中 target increment 与 controlled degradation 的常数偏置、趋势、分区偏置和过度平滑参数无关。
- controlled degradation 的每个加性场均满足 `P_d` cutoff 不高于 canonical cutoff。
- 同一父 realization 的多个 LFM variant 共享 seismic 和 target increment，variant 选择均匀且不改变父样本权重。
- patch 标签等于完整 realization 标签的直接切片。
- LFM 消费端分别校验井控和最终体 application count，不使用 `P(LFM) ~= LFM` 作为硬门禁。
- 空间建模后的最终 LFM 保存 complement-response QC 和垂向 warp 标志，GINN v2 原样记录。
- 时间域和深度域真实井均在转换、重采样到最终规则模型轴后计算背景和增量。

### 10.2 网络和训练

- 四类架构均输出与输入空间同形的 `predicted_increment_log_ai`。
- 零初始化输出严格为零，最终阻抗严格等于输入 LFM。
- synthetic supervised 和 real-well supervised 前向、反向及梯度有限。
- 时间域和深度域 supervised 后 physics 前向、反向及梯度有限。
- physics-first、zero-initialized physics 和无监督祖先的 physics 配置明确失败。
- physics-only checkpoint 标记为不可部署，`deployment_checkpoint` 引用时明确失败。
- 含 synthetic supervised 与 physics block 的后续阶段只有按 dense synthetic MSE 选优时才可部署。
- real-well supervised 与 physics 的联合阶段保持 experimental，不能自动部署。
- `increment_l2_weight` 生效，`delta_l2_weight` 明确失败。

### 10.3 推理与评估

- R0 valid 点全部得到有限增量和最终阻抗，invalid 点保持 NaN。
- inline 步长 1、xline 步长 4、多剖面和完整体均保持显式坐标。
- 低频 RMS、保守低频响应比和 physics 漂移可从保存数组复算。
- 合成 canonical closure、合成 deployment closure 和真实 deployment closure 使用明确字段并可分别复算。
- 成对 LFM 反事实分别报告增量条件效应和 LFM 直接替换效应。
- 新 CLI 对旧 benchmark、配置、checkpoint 和预测 schema 给出明确错误。

### 10.4 新旧目标对照

对照使用相同架构、seismic 输入、split、seed、训练预算和 normalization，分别报告：

- canonical increment MSE、RMSE 和相关系数；
- `canonical_background + predicted_increment` 的合成 AI 指标；
- `input/external LFM + predicted_increment` 的最终 AI 指标；
- LFM-only 的井上 RMSE、bias、相关系数和 R1 waveform error；
- 最终模型相对于同一 external LFM 的 AI 与 waveform 净增益；
- geometry holdout 和 pinchout；
- 低频 QC；
- LFM 反事实增量不变性；
- 真实井最终 AI；
- R1 waveform error。

### 10.5 旧实现清理

- 正式生产包的依赖图中不存在 executable legacy baseline 工具或退役模块。
- 正式 CLI、reader、writer 和 checkpoint loader 不存在旧 schema 的成功执行路径。
- 旧 schema 输入只触发通用 schema mismatch，不委托给旧 parser、reader 或 loader。
- 正式路径不存在旧字段 alias、fallback、转换器、feature flag 或双路径分派。
- 被替代的模块已删除，不存在以 `legacy`、`old`、`compat`、`deprecated` 或退休版本后缀命名且仍被正式入口调用的实现。
- 当前语义的 reader、真实井监督和 checkpoint 实现具有唯一生产入口。
- 删除旧实现后，第 10.1–10.4 节测试和全部 smoke 继续通过。

对 tracked current-code 范围执行静态搜索。扫描范围至少包括 `src/`、`scripts/`、正式实验配置和 `docs/guide/`，排除 `results/`、`scripts/output/` 和冻结审计目录。除本文及 migration failure tests 外，不得出现：

```text
synthoseis_lite_v3
real_field_lfm_run_v3
real_field_lfm_variant_v3
ginn_v2_experiment_v1
ginn_v2_checkpoint_v4
ginn_v2_prediction_v2
real_field_zero_shot_model_v4
real_field_zero_shot_summary_v4
real_field_forward_diagnostic_summary_v5
physical_delta_log_ai
truth-input_lfm
target_delta
pred_delta
pred_delta_vs_lfm
delta_l2_weight
lfm_controlled_degraded
residual_vs_lfm_ideal
residual_vs_lfm_controlled_degraded
real_delta
```

无修饰 `residual` 仍可作为明确的局部数值误差变量，但公共字段和产物必须说明具体误差对象。静态搜索结果及 allowlist 必须随清理提交报告；首版 executable legacy allowlist 为空。

代码验收以合同、数值正确性和错误边界为硬门禁。模型质量作为消融结果报告，不要求新语义在所有效果指标上全面超过 full-correction 基线。

## 11. 首版边界

首版保持单输出头和三通道输入。以下内容由后续独立消融决定：

- 删除或弱化 LFM 输入通道；
- seismic-band 与 prior-supported high-frequency 双输出头；
- 输出端可微 Butterworth 互补滤波；
- canonical cutoff 扫描；
- physics-first 诊断模式；
- 独立低频背景适配器。
