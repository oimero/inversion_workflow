# GINN v2 积木式训练设计

## 1. 目标

GINN v2 采用“实验、架构、数据源、损失块、训练阶段”五层组合模型，支持：

- 使用 Synthoseis-lite 标签进行监督学习；
- 在合成或真实地震上进行无标签物理约束学习；
- 使用真实井控制进行稀疏监督；
- 按任意顺序组合和重复上述训练阶段；
- 在真实工区推理时覆盖全部输入有效样点。

架构标识只描述网络拓扑，不编码训练样本、损失函数或下游用途。实验标识负责稳定标识一套完整训练配方及其产物。

本设计直接启用新配置和产物 schema。旧 GINN v2 配置、模型标识和 checkpoint 不自动迁移。

## 2. 非目标

- 不提供逐层声明任意神经网络的通用图配置；
- 不在真实物理损失中引入可训练 gain、taper、相位扫描或 TV 正则；
- 不自动选择多阶段顺序、损失权重，或跨阶段猜测更优的部署 checkpoint；
- 不用插值、LFM 回填或其他数值替代掩盖 R0 的预测覆盖缺口；
- 不将经过某个真实工区物理适配的模型强制绑定到该工区；
- 不读取旧模型后猜测其架构、训练样本或损失配方。

## 3. 设计原则

### 3.1 单一模型语义

所有架构固定使用三个输入通道：

1. 地震；
2. 低频模型；
3. 有效掩码。

网络直接输出无量纲物理量 `delta_log_ai`：

```text
pred_log_ai = lfm_log_ai + pred_delta_log_ai
```

模型输出不使用合成标签的均值和标准差。所有架构的最后一个输出层使用零权重和零偏置初始化，因此未训练模型严格输出低频模型。

### 3.2 能力由组合决定

任何已注册架构都可以使用任何首版损失块。代码不得根据架构标识限制监督学习、物理损失或真实井监督。

数据可用性和契约决定损失块是否可用。例如，深度物理损失必须存在冻结的 AI–Vp 关系和秒制子波；这属于数据源契约，不属于架构能力。

### 3.3 阶段边界显式

一个实验包含有序训练阶段。每个阶段显式声明数据、损失、优化器、训练步数和选优指标。阶段之间只继承模型权重，不继承优化器或调度器状态。

## 4. 配置 schema

配置根节点为 `ginn_v2`。下面是一个监督预训练后进行真实工区物理适配的完整示例。

```yaml
ginn_v2:
  experiment_id: supervised_then_field_physics
  seed: 20260617

  architecture:
    id: trace_dilated_tcn
    hidden_channels: 32
    depth: 5

  sources:
    synthetic:
      kind: synthoseis_lite
      benchmark_dir: auto
      input_seismic_variant: observed_mismatch
      physics_target_variant: model_consistent

    field:
      kind: real_field
      lfm_run_dir: scripts/output/real_field_lfm_<run>
      variant_id: trend_baseline
      well_control_run_dir: scripts/output/real_field_well_controls_<run>
      model_input_seismic_transform: p99_abs_matched
      physics_target_seismic_transform: identity
      validation_split:
        kind: spatial_block
        fraction: 0.10
        gap_m: 250.0
        anchor: high_inline_tail

    wells:
      kind: real_wells
      field_source: field
      well_control_run_dir: scripts/output/real_field_well_controls_<run>
      held_out_well: PH5
      exclude_same_cluster: true
      cluster_radius_m: 500.0

  normalization_reference:
    source: synthetic

  patching:
    lateral_samples: 32
    vertical_samples: 128
    lateral_stride: 16
    vertical_stride: 64

  stages:
    - stage_id: synthetic_pretrain
      epochs: 20
      steps_per_epoch: 500
      optimizer:
        kind: adamw
        learning_rate: 0.001
        weight_decay: 0.0001
      loss_blocks:
        - block_id: synthetic_ai
          kind: synthetic_supervised
          source: synthetic
          weight: 1.0
          update_interval: 1
          batch_size: 8
          min_valid_samples: 128
      validation:
        selection_metric: synthetic_ai.mse
        mode: full

    - stage_id: field_physics
      epochs: 10
      steps_per_epoch: 300
      optimizer:
        kind: adamw
        learning_rate: 0.0001
        weight_decay: 0.0001
      loss_blocks:
        - block_id: field_waveform
          kind: physics
          source: field
          weight: 1.0
          update_interval: 1
          batch_size: 8
          min_valid_samples: 128
          delta_l2_weight: 0.01
          waveform_standardization: masked_centered_rms
          centered_rms_epsilon: 1.0e-12
          min_centered_rms: 1.0e-6
        - block_id: well_anchor
          kind: real_well_supervised
          source: wells
          weight: 0.1
          update_interval: 4
          batch_size: 4
          min_valid_samples: 8
      validation:
        selection_metric: field_waveform.total
        mode: fixed_steps
        steps: 100

  deployment_checkpoint: last_stage.best
```

### 4.1 实验标识

`experiment_id` 必填，必须是非空、可用于目录名的稳定标识。同一个 R0/R1 配置中不得出现重复实验标识。

`experiment_id` 决定：

- 实验产物目录名；
- R0/R1 输出子目录名；
- manifest 和比较配置中的模型引用。

不再定义 `model_role`，也不从架构名称推断 `no_lateral` 或 `lateral`。

### 4.2 架构注册表

首版只注册四个纯架构标识：

| 架构标识 | 网络 | 横向行为 |
|---|---|---|
| `trace_conv1d` | 逐道普通一维卷积 | 1 |
| `trace_dilated_tcn` | 逐道膨胀一维 TCN | 1 |
| `trace_lateral_mixer` | 膨胀 TCN 加浅层横向混合 | 由 mixer 合同计算 |
| `patch_conv2d` | 二维卷积 patch 网络 | 由二维卷积合同计算 |

公共参数：

- `hidden_channels`：正整数；
- `depth`：满足对应架构最小层数要求的正整数；
- `lateral_kernel`：仅 `trace_lateral_mixer` 接受，必须是正奇数。

不属于当前架构的参数出现时明确报错。

架构合同还必须冻结实际 kernel、dilation schedule、block 数、mixer 层数和 padding 规则。架构实例根据这些字段计算并导出 `vertical_receptive_field` 与 `lateral_receptive_field`；manifest 不用简化公式或架构名称猜测感受野。

### 4.3 数据源

`sources` 是以用户自定义 source ID 为键的映射。loss block 只通过 source ID 引用数据。

首版 source kind：

| kind | 主要内容 | 可用损失块 |
|---|---|---|
| `synthoseis_lite` | 合成地震、目标 logAI、LFM、一致正演地震、掩码 | `synthetic_supervised`、`physics` |
| `real_field` | 真实地震、真实工区 LFM、有效掩码、采样轴 | `physics` |
| `real_wells` | 井 logAI、LFM、空间簇和井位 | `real_well_supervised` |

每个 source 在实验开始时解析一次，冻结路径、schema、合同指纹、采样域、单位和 depth basis。manifest 记录全部 source provenance。

`real_wells.field_source` 必须引用一个 `real_field` source。井监督在该 source 的 LFM、采样轴和几何上构造井旁 delta 标签，禁止按字段名猜测或另行发现 LFM。

`real_field` 和引用它的 `real_wells` 都显式记录 well-control 路径。两者解析出的合同指纹必须相同；不一致时明确失败。保留两处路径是为了让真实体 source 和井标签 source 各自拥有完整、可审计的输入合同。

#### 4.3.1 模型输入与物理目标张量

每个可用于 physics 的 source 必须显式提供四个不同语义的张量：

| 张量 | 语义 |
|---|---|
| `seismic_model_input` | 经过 source 模型输入变换和实验归一化后的网络输入 |
| `seismic_physics_target` | 只经过冻结 physics-target 预处理的波形目标，不应用模型 mean/std |
| `lfm_model_input` | 经过实验归一化后的网络输入 LFM |
| `lfm_log_ai_physical` | 未标准化的物理 logAI，用于组装预测 logAI 和物理正演 |

数据流固定为：

```text
seismic_raw
  -> model_input_seismic_transform
  -> experiment seismic mean/std
  -> seismic_model_input

seismic_raw
  -> physics_target_seismic_transform
  -> seismic_physics_target

lfm_log_ai_physical
  -> experiment LFM mean/std
  -> lfm_model_input
```

physics loss 禁止把 `seismic_model_input` 当成波形目标，也禁止把 `lfm_model_input` 当成物理 logAI。

Synthoseis source 必须显式声明 `input_seismic_variant`。首版允许 `nominal` 或 `observed_mismatch`。`physics_target_variant` 固定且必须为 `model_consistent`；其他值明确失败。这样网络可以消费 mismatch 输入，但物理目标始终来自同一 AI 的一致正演。

真实 source 的两类 seismic transform 分别冻结并写入 manifest。physics target 不应用实验级模型输入 mean/std；真实 MaskedRMS 在 loss 内处理其尺度。

经过真实工区物理训练的 checkpoint 允许用于其他工区。跨工区 R0/R1 必须继续校验：

- 时间域或深度域一致；
- 秒或米单位一致；
- 深度域使用 TVDSS；
- 模型输入通道和归一化合同一致。

真实输入合同指纹不要求相同。若 deployment checkpoint 的任一训练阶段消费过 `real_field` physics source，而部署工区合同指纹不同，R0/R1 默认明确拒绝；用户必须配置 `allow_cross_field_adapted_checkpoint: true` 才能继续。允许后仍须在运行摘要醒目标记 adaptation source 与 deployment source，并执行输入分布 OOD QC。

### 4.4 实验级归一化

`normalization_reference.source` 必须引用一个能够提供地震和 LFM 的 source。只使用该 source 的训练分区计算：

- seismic mean/std；
- LFM logAI mean/std。

统计量生成后冻结，所有阶段、checkpoint 和 R0 使用同一份统计量。mask 不归一化。

网络输出是直接 `delta_log_ai`，因此 normalization 中不存在 delta target mean/std。real-only physics-first 可以选择真实工区 source 作为 reference，不依赖 Synthoseis 标签。

数据源自身的显式值域变换先执行，再应用实验级归一化。不得因阶段变化重新拟合统计量。

### 4.5 Patch 几何

`patching` 只定义窗口大小和滑动步长：

- `lateral_samples`；
- `vertical_samples`；
- `lateral_stride`；
- `vertical_stride`。

不再定义全局 `min_valid_fraction`。训练资格由每个 loss block 的最终监督 mask 和 `min_valid_samples` 决定；R0 推理使用独立的全覆盖规则。

## 5. 多阶段状态机

### 5.1 阶段顺序

`stages` 是非空有序列表，`stage_id` 在实验内唯一。支持：

- 监督后物理约束；
- 物理约束后监督；
- 多次交替；
- 同类阶段重复；
- 一个阶段组合多个数据源和损失块。

第一阶段使用零初始化模型，除非显式配置外部新 schema checkpoint。后续阶段默认从上一阶段 best checkpoint 初始化，也可显式引用任一先前阶段的 `best` 或 `final`。

阶段引用只允许指向列表中更早的阶段，禁止循环或前向引用。

### 5.2 优化器

每个阶段独立声明 optimizer。首版只要求实现 AdamW：

- `learning_rate` 必须为有限正数；
- `weight_decay` 必须为有限非负数。

阶段切换时加载模型权重并重新创建 optimizer。Adam 动量和 scheduler 状态不跨阶段继承。

### 5.3 Step 与 epoch

每个阶段显式声明正整数 `epochs` 和 `steps_per_epoch`，因此 epoch 不依赖任一数据源长度。

每个 loss block 有独立的确定性循环采样器。随机种子使用 `numpy.random.SeedSequence([experiment_seed, stage_index, block_index, epoch])` 派生，其中两个 index 是 stage 和 block 在冻结配置中的位置。禁止使用 Python 内置 `hash()`。

在全局 stage step `s` 上，满足下面条件的 block 到期：

```text
s % update_interval == 0
```

到期 block 分别读取自己的 batch、计算标量损失，再按 block weight 求和，进行一次反向传播和 optimizer step。至少一个 loss block 的 `update_interval` 必须为 1，保证每一步都有优化目标。

block 的有效频率由 `update_interval` 本身表达，不对低频 block 的权重做自动补偿。

### 5.4 Checkpoint 与阶段交接

每个阶段至少写出：

- `checkpoint_best.pt`；
- `checkpoint_final.pt`；
- 每 epoch 训练和验证指标；
- 实际执行 step 数和各 block 的 batch 数；
- best epoch 和 selection metric 值。

`validation.selection_metric` 必须引用当前阶段某个 loss block 导出的验证指标。selection metric 必须有限且按最小化选择。

配置中的 `deployment_checkpoint` 可以省略。配置解析时立即将缺省值解析为最后阶段 best；也可以显式引用任意阶段的 best/final。实验 manifest 必须写入完整的已解析 stage ID、checkpoint kind 和路径。R0/R1 只读取 manifest 中的显式记录，不再次应用默认值。

## 6. 损失块

### 6.1 公共字段

所有 loss block 必须声明：

- `block_id`：阶段内唯一；
- `kind`；
- `source`；
- `weight`：有限非负数；
- `update_interval`：正整数；
- `batch_size`：正整数；
- `min_valid_samples`：正整数。

block 先构造自身最终监督 mask。只有监督 mask 中有效样点数不少于 `min_valid_samples` 的 patch 才进入该 block 的训练或验证索引。

同一个原始窗口可以进入一个 block 而不进入另一个 block。

`min_valid_samples` 的计数单位由 block kind 固定：

- `synthetic_supervised`：label-valid 网格样点；
- `physics`：观测、LFM 和物理目标均有限且不属于 padding 的 waveform 样点；
- `real_well_supervised`：有效井监督点。

不得用 input-valid 数量代替上述 block 专属计数。

### 6.2 合成监督损失

`synthetic_supervised` 只接受 `synthoseis_lite` source。

目标：

```text
target_delta_log_ai = target_log_ai - lfm_log_ai
```

损失：

```text
L_supervised = sum(mask * (pred_delta_log_ai - target_delta_log_ai)^2)
               / sum(mask)
```

mask 为 label-valid mask，要求目标、LFM 和输入均有限。训练和验证按父实现分组切分，禁止同一父实现的重叠 patch 跨 split。

### 6.3 物理损失

`physics` 接受 `synthoseis_lite` 或 `real_field` source。

公共计算：

```text
pred_log_ai = lfm_log_ai + pred_delta_log_ai
pred_seismic = forward(pred_log_ai)
L_delta = sum(input_valid * pred_delta_log_ai^2) / sum(input_valid)
L_physics = L_waveform + delta_l2_weight * L_delta
```

`delta_l2_weight` 必须为有限正数。它锚定反射率无法识别的绝对 logAI 尺度。

时间域使用公共 PyTorch `forward_time`。深度域使用冻结 AI–Vp 关系从预测 AI 派生 Vp，再调用公共 PyTorch `forward_depth`。核心正演输入全部来自 source 合同。

physics 使用完整、连续且有限的 LFM patch 作为背景。模型在整个 patch 输出 delta，正演一次性处理完整 patch，不按目标层切段，不在目标 mask 外强制 delta 为零，也不对子波支撑构造额外腐蚀 mask。

waveform loss 的 mask 只表示观测、LFM 和合成目标是否有限，以及样点是否属于真实数据而非 padding。第三个网络输入通道仍表示主要解释区域，但它是模型提示，不是正演边界，也不决定 waveform loss 的范围。合成标签的 label-valid mask 只用于监督损失。

#### 合成 source

- 观测目标固定为 `seismic_model_consistent`；
- 不把含噪声、gain、相位或静差的 seismic variant 当作物理真值；
- 在观测目标、LFM 均有限且不属于 padding 的样点上计算原始振幅 masked MSE。

```text
L_waveform_synthetic = sum(mask * (pred_seismic - seismic_model_consistent)^2)
                       / sum(mask)
```

#### 真实 source

对完整 patch 执行一次连续正演。真实地震、LFM 或合成波形非有限以及 patch padding 的样点不参与 waveform loss；目标层边界不裁剪正演输入和损失。

标准化粒度固定为单个 batch item/patch，不是单道，也不跨 batch item 共享统计量。这样保留一个 patch 内不同 trace 的相对振幅关系。

对于 patch 张量 `x` 和 finite/padding mask `m`：

```text
masked_mean(x, m) = sum(m * x) / sum(m)
centered_rms(x, m) = sqrt(sum(m * (x - masked_mean(x, m))^2) / sum(m))
denominator(x, m) = max(centered_rms(x, m), sqrt(centered_rms_epsilon))
standardize(x, m) = where(
    m,
    (x - masked_mean(x, m)) / denominator(x, m),
    0
)
```

观测和合成分别计算自己的 mean 与 centered RMS，但使用同一个 finite/padding mask。合成侧统计量不 detach，梯度必须穿过 mean、centered RMS 和标准化运算。

`centered_rms_epsilon` 必须为有限正数，默认 `1e-12`；`min_centered_rms` 必须为有限正数，默认 `1e-6`。任一侧 centered RMS 小于 `min_centered_rms` 时，该 patch 的 waveform item 无效：训练器跳过该 item、增加带原因的计数并写入 epoch 指标。若一个 batch 或冻结数据索引没有任何有效 waveform item，则明确失败，禁止产生 NaN loss。

```text
L_waveform_real = masked_mse(
    standardize(pred_seismic, finite_padding_mask),
    standardize(seismic_physics_target, finite_padding_mask)
)
```

这里没有可训练 gain、显式振幅拟合、taper、相位扫描或 TV 正则。

##### 振幅可识别性限制

观测和合成独立做 patch-centered RMS 标准化后，真实 waveform loss 对两者之间的正整体振幅缩放不敏感。因此 real-only physics-first 不声称仅依靠 waveform loss 恢复绝对反射强度。它主要约束事件时序、极性、相位结构、patch 内横向相对振幅和波形形状；阻抗对比度幅度由 LFM 锚定、零 delta 初始化、delta L2、可选井监督或先前监督阶段共同决定。

真实 physics block 无论是否作为 selection metric，都必须记录：

- `observed_centered_rms`；
- `predicted_centered_rms`；
- `predicted_to_observed_rms_ratio`；
- `delta_log_ai_rms`；
- 因样点不足、低 RMS 或非有限值跳过的 item 数量和比例。

真实 source 的 validation split 使用确定性 XY 空间块和物理距离 gap。训练窗口与验证窗口的实际空间支撑不得重叠或穿过 gap。

### 6.4 真实井监督

`real_well_supervised` 只接受 `real_wells` source。

损失目标为井样点上的直接 delta logAI：

```text
target_delta_log_ai = well_log_ai - sampled_lfm_log_ai
L_well = mean((pred_delta_log_ai_at_well - target_delta_log_ai)^2)
```

保留以下现有语义：

- 按空间半径构建井簇；
- 显式 held-out 井；
- 可排除 held-out 井同簇的其他井；
- cluster 和 cluster 内 well 均衡采样；
- 记录每井实际采样次数。

所有架构均可使用此 block。逐道网络可以使用稀疏支持优化；具有横向感受野的架构必须走可微完整 patch 预测，不能把横向上下文替换为单道快捷路径。

## 7. 验证与选优

每个 block 每 epoch 都计算验证指标，不受训练时 `update_interval` 影响。每个 stage 的 validation 必须显式选择以下模式之一：

- `mode: full`：遍历冻结验证索引的全部 batch，禁止同时填写 `steps`；
- `mode: fixed_steps`：必须填写正整数 `steps`，从冻结验证索引循环读取固定数量 batch。

验证采样器与训练采样器相互独立，不随 epoch 重洗。解析后的验证索引保存为显式索引文件并由该 stage 的所有 epoch 复用；manifest 记录索引文件路径和采样 seed。

| block | 验证隔离 | 可选主指标 |
|---|---|---|
| `synthetic_supervised` | 父实现级 split | `<block_id>.mse` |
| 合成 `physics` | 父实现级 split | `<block_id>.waveform_mse`、`<block_id>.total` |
| 真实 `physics` | XY 空间块 + gap | `<block_id>.waveform_mse`、`<block_id>.total` |
| `real_well_supervised` | held-out 空间簇 | `<block_id>.mse` |

混合阶段必须从当前 block 实际导出的指标中显式选择一个 selection metric。训练器不自动构造加权验证总分。

## 8. R0 全覆盖推理

### 8.1 已确认的问题

历史运行 `real_field_zero_shot_20260706_163351` 使用 `32 × 128` patch 和 `min_valid_fraction=0.5`。inline 1661、xline 4599–5079 每道只有约 54–67 个有效深度样点，约占完整 551 点轴的 10%。有效带随 xline 倾斜，整块有效比例无法达到 0.5，因此包含真实有效输入的窗口被整体丢弃。

该运行输入有效率为 18.72%，预测覆盖率为 18.22%，证明部分输入有效样点没有进入任何预测 patch。拼接权重为零的位置最终成为 NaN。

### 8.2 推理窗口

R0 只读取模型 patch 几何，不读取任何训练 block 的 `min_valid_samples`。

每个 inline/section 使用覆盖完整横向和采样轴的规则窗口，包括轴末端补齐窗口。窗口内只要存在一个 `valid_mask_model=True` 样点就必须执行推理。

窗口构造和填充值固定如下：

- 轴长度大于或等于 patch 长度时，规则 stride 后的最后一个窗口起点固定为 `axis_length - patch_length`，避免不必要 padding；
- 轴长度小于 patch 长度时，从索引 0 开始并只在轴末端做右侧 padding；横向轴和采样轴采用相同规则；
- 原始 invalid 位置和 padding 位置的 normalized seismic 固定为 0；
- 原始 invalid 位置和 padding 位置的 normalized LFM 固定为 0；
- 原始 invalid 位置和 padding 位置的 valid mask 固定为 0；
- 这些零表示实验 normalization 下的参考中心，不表示物理地震或物理 LFM 为零；
- 只有原始轴内的 valid 点参与 stitching，padding 和原始 invalid 点永不参与累计。

生产拼接固定为 uniform：

```text
pred_sum[p] += prediction[p]  if valid[p]
weight[p] += 1                if valid[p]
```

invalid 点不参与累计。推理完成后：

```text
stitched[p] = pred_sum[p] / weight[p]  if weight[p] > 0
stitched[p] = NaN                       otherwise
```

### 8.3 覆盖硬门禁

必须逐点满足：

```text
(stitching_weight > 0) == valid_mask_model
```

并同时满足：

- 所有 valid 点预测有限；
- 所有 invalid 点保持 NaN；
- 不使用 LFM、零值或插值填补未预测 valid 点。

任一条件失败时整次模型推理失败。错误产物列出每个缺口的 inline、xline、sample 坐标和对应数组索引。

### 8.4 支持质量 QC

预测 NPZ 新增与输出体同形状的：

- `prediction_support_count`：该点累计了多少个 patch；
- `max_context_valid_fraction`：所有贡献 patch 中最大的上下文有效比例。

模型摘要记录 valid 点上两者的最小值及 p01、p05、p50、p95、p99。另输出 `support_count == 1` 的连续低支持区域清单。

支持质量仅用于 QC，不据此删除预测或重新产生 NaN。

## 9. R0/R1 身份和比较

R0 配置按模型实验目录加载，输出以 `experiment_id` 为键。多个模型之间的差值、横向能力或其他对比必须显式声明：

```yaml
real_field_zero_shot:
  allow_cross_field_adapted_checkpoint: false
  models:
    - experiment_dir: experiments/ginn_v2/results/trace_baseline
    - experiment_dir: experiments/ginn_v2/results/lateral_model

  comparisons:
    - comparison_id: lateral_vs_trace
      left: lateral_model
      right: trace_baseline
```

`left` 和 `right` 必须引用已加载且采样轴、输出形状完全一致的 experiment ID。不存在隐式 `no_lateral`/`lateral` 推断。

R1 使用同一 `experiment_id` 和 comparison ID。单模型诊断不要求 comparison；模型间诊断只消费显式 comparison。

## 10. 产物 schema

### 10.1 实验 manifest

新 schema：`ginn_v2_experiment_v1`。

至少记录：

- `experiment_id`；
- architecture ID、kernel/dilation/mixer 合同、实例计算的双轴感受野和参数量；
- 固定输入/输出语义；
- normalization reference、统计量和输入合同；
- 全部 source 路径、schema 和已有的直接上游合同指纹；
- 有序 stage 配置；
- 每阶段实际 loss blocks、update interval、采样 seed 和 optimizer；
- 每阶段 best/final checkpoint、selection metric、验证模式和验证索引文件；
- deployment checkpoint；
- sample domain、sample unit、depth basis；
- 时间/深度正演算子、有效支撑阈值和 forward inputs；
- 所有真实 physics adaptation source；
- 代码版本和单个实验级合同指纹。

指纹的边界保持克制：实验 manifest 可以记录现有直接上游合同的指纹，并为解析后的完整实验合同生成一个实验级指纹。checkpoint、patch/验证索引、batch 序列、指标、图件、数组和其他输出文件不单独计算摘要，也不建立递归的祖先指纹链。

### 10.2 Checkpoint

新 schema：`ginn_v2_checkpoint_v4`。

checkpoint 记录纯 architecture ID、完整 architecture 合同、直接 delta logAI 输出契约、固定 normalization、sample-axis 合同、所属 experiment/stage、checkpoint kind 和模型权重。

deployment checkpoint 还必须记录 patch deployment contract：

- `lateral_samples`、`vertical_samples`；
- `lateral_stride`、`vertical_stride`；
- 轴末端窗口起点规则；
- 短轴右侧 padding 规则；
- invalid/padding 三通道填充值；
- uniform stitching 和全覆盖后置条件。

optimizer 状态可以存在于阶段恢复 checkpoint，但 deployment checkpoint 不依赖它。

### 10.3 R0/R1

- `real_field_zero_shot_model_v4`；
- `real_field_zero_shot_summary_v4`；
- `real_field_forward_diagnostic_summary_v5`，记录 experiment ID、deployment checkpoint、显式 comparison 和预测覆盖合同。

R0 模型产物不得再出现 `model_role`。所有模型引用使用 `experiment_id`。

## 11. 兼容与失败规则

以下输入明确拒绝：

- 根节点仍为旧 `train` 的配置；
- 旧的十个带训练语义 model ID；
- 包含 `model_role`、全局 `min_valid_fraction` 或 normalized delta 输出语义的配置；
- `ginn_v2_checkpoint_v3` 及更早 checkpoint；
- 缺少 architecture、normalization reference、stages 或已解析 deployment checkpoint 的 manifest；
- R0/R1 中按 `no_lateral`/`lateral` 猜测模型身份的旧配置。

错误信息必须包含实际 schema、期望 schema 和新配置文档入口。不得自动翻译旧 ID、注入默认 stage、从 checkpoint 文件名推断训练配方或补造归一化统计量。

## 12. 实施分解

### 阶段 A：配置、架构和产物骨架

- 建立新配置解析和值对象；
- 将十个旧 ID 收口为四个架构 ID；
- 统一直接 delta logAI 输出和零初始化；
- 写入新 experiment/checkpoint schema；
- 对旧配置和 checkpoint 建立失败门禁。

### 阶段 B：数据源和损失块

- 建立 source registry 和实验级 normalization；
- 实现三类 loss block 及独立 patch index；
- 将 real-delta 迁为 real-well-supervised；
- 建立合成/真实、时间/深度物理分派。

### 阶段 C：多阶段训练器

- 实现 stage 状态机、独立 sampler、update interval 和显式 steps-per-epoch；
- 实现阶段 best/final、权重承接和 optimizer 重建；
- 实现各 block 的验证隔离与 selection metric。

### 阶段 D：R0/R1

- 用 experiment ID 替换 model role；
- 实现显式 comparison；
- 实现 R0 uniform 全覆盖、硬门禁和支持质量 QC；
- 升级 R0/R1 schema 和导出合同。

## 13. 测试矩阵

### 13.1 架构与损失组合

- 四个架构分别运行三类 loss block 的前向和反向；
- 横向 mixer 的不同合法 kernel；
- 不属于架构的参数明确失败；
- 所有架构零初始化时 `pred_delta_log_ai == 0` 且 `pred_log_ai == LFM`。

### 13.2 多阶段

- supervised → physics；
- physics → supervised；
- supervised → physics → supervised；
- 同阶段多 block、不同 update interval；
- batch 序列在相同 seed 下完全一致；
- `SeedSequence` 派生结果跨进程一致；
- full/fixed-steps 验证量和冻结索引顺序可复现；
- 下一阶段默认加载上一阶段 best；
- 显式加载先前 final；
- optimizer 状态未跨阶段继承。

### 13.3 物理损失

- 时间域和深度域的合成 physics 前向、反向和有限梯度；
- 时间域和深度域的真实 physics 完整连续 patch；
- 模型输入 seismic 与 physics target 使用不同冻结数据流，互换时测试失败；
- 目标层 mask 不裁剪正演、不限制 waveform loss，也不强制 mask 外 delta 为零；
- patch 级标准化保留 patch 内 trace 间相对振幅，且不跨 batch item 共享统计量；
- centered mean/RMS 的合成侧梯度有限且非零；
- mask 外标准化值严格为零；
- real-only physics-first 不读取 synthetic delta 统计量；
- masked RMS 对整体振幅缩放不敏感；
- 低于 min RMS、单样点区段和非有限输入按原因计数；全 batch/索引无有效 item 时失败；
- delta L2 对常量漂移产生非零约束。
- 四项振幅可识别性诊断及跳过计数可复算。

### 13.4 真实井

- 空间簇均衡与 held-out 同簇排除；
- 逐道架构稀疏支持预测；
- 横向架构可微完整 patch 预测；
- 所有架构均可使用真实井 block。

### 13.5 R0 覆盖

- 任意形状有效 mask 上 `(weight > 0) == valid`；
- invalid 点保持 NaN，valid 点全部有限；
- inline 步长 1、xline 步长 4；
- 轴首尾和不足 stride 的末端窗口；
- 短轴右侧 padding 和三通道零填充值；
- center-crop 配置进入生产 R0 时明确拒绝；
- 支持度数组和摘要可复算；
- inline 1661、xline 4599–5079 薄有效带 fixture 全覆盖，即使窗口有效率约 10%。

### 13.6 下游和兼容

- 任意合法 experiment ID 的 R0/R1；
- 显式 comparison 的左右引用和轴一致性；
- 跨工区使用时记录 provenance 并执行 OOD QC；
- field-adapted checkpoint 跨工区默认拒绝，显式 override 后才允许；
- 所有旧配置、旧模型 ID、旧 checkpoint 和旧 R0/R1 schema 明确失败。

## 14. 验收条件

- 网络架构标识不包含数据或损失语义；
- 任意架构可以组合任意首版 loss block；
- 监督、物理、真实井训练可以任意排序和重复；
- real-only physics-first 不依赖合成标签统计量；
- 时间域和深度域均使用公共可微正演内核；
- 阶段选优、交接和部署 checkpoint 可审计、可复现；
- R0 对全部输入有效点产生有限预测，不再因 patch 有效率阈值制造 NaN；
- R0/R1 只使用 experiment ID 和显式 comparison；
- 新产物无法与旧 schema 静默混用。
