# GINN v2 模型训练与消融

`ginn_v2.py` 是工作流的研究旁路。它把网络架构、数据源、损失函数和训练阶段分别配置，通过受控消融实验揭示不同架构选择和训练策略的实际增益。一个实验由稳定的实验标识管理，产物写入实验目录，可直接用于后续真实工区的零样本推理。

---

## 快速开始

```powershell
# 训练
python scripts/ginn_v2.py train --config experiments/ginn_v2/train.yaml

# 可选：指定输出目录
python scripts/ginn_v2.py --output-dir scripts/output/ginn_v2_smoke `
  train --config experiments/ginn_v2/train.yaml

# 在合成基准上预测
python scripts/ginn_v2.py --output-dir scripts/output/ginn_v2_predict `
  predict --model-run-dir experiments/ginn_v2/results/<experiment_id> `
  --split validation

# 生成合成基准评估报告
python scripts/ginn_v2.py --output-dir scripts/output/ginn_v2_report `
  report --prediction-dir scripts/output/ginn_v2_predict
```

训练配置使用 `ginn_v2_experiment_v1` schema。旧的训练配置和旧 checkpoint 会明确报错，不会静默兼容。

---

## 运行前需要什么

| 来源 | 文件 | 用途 |
|------|------|------|
| 合成基准旁路 | `synthetic_benchmark.h5`、`sample_index.csv`、`benchmark_manifest.json` | 冻结的训练数据和真值 |
| 第七步 | `lfm_run_summary.json`、`variants/<id>/lfm.npz` | 真实工区低频模型（仅使用真实数据源时） |
| 第六步 | `run_summary.json`、`well_control_manifest.csv` | 井控事实（仅使用真实井监督时） |
| 第五步（时间域） | `selected_wavelet.csv` | 全局子波（仅时间域物理损失） |
| 岩石物理旁路（深度域） | `forward_model_inputs.json` | 冻结子波和 AI–Vp 关系（仅深度域物理损失） |
| 用户配置 | `experiments/ginn_v2/train.yaml` | 实验配方 |

合成基准目录设为 `auto` 时，脚本自动从合成基准的结果目录下搜索最新且完整的场条件基准。显式填路径则固定使用该目录。

---

## 配置参考

GINN v2 的训练配方由五层组件自由组合而成：

```
实验 identity
  └─ 架构           ← 网络长什么样（拓扑、容量）
  └─ 数据源         ← 喂什么数据（合成 / 真实工区 / 井）
  └─ 损失块         ← 学什么目标（监督 / 物理 / 井约束）
  └─ 阶段           ← 以什么顺序学（单阶段 / 多阶段序列）
  └─ 归一化参考     ← 输入标准化从哪算
```

每一层只管自己的事。架构不知道训练数据从哪来；损失块不知道网络有没有横向混合。**你做消融实验时，只要换其中一层，其他层不用动。**

下面逐一说明每层组件有哪些选项、各自什么意思、怎么选。

---

### 架构：网络长什么样

四个架构的核心差异是**有没有横向视野**——即是否能看到相邻道的信息：

| 架构 | 横向行为 | 纵向行为 | 适用场景 |
|------|------|------|------|
| `trace_conv1d` | 逐道独立，完全看不到邻居 | 5 层普通 Conv1d | 最简单的基线——如果没有横向信息，模型能做到什么程度？ |
| `trace_dilated_tcn` | 逐道独立 | 膨胀 Conv1d，depth=5 时纵向感受野约 125 点 | 单道基线但纵向视野更大——比 trace_conv1d 看得更深 |
| `trace_lateral_mixer` | 浅层横向混合，视野 = `lateral_kernel`（默认 3） | 同 dilated TCN | **关键消融点**：加了一点点横向信息，效果变好还是变差？ |
| `patch_conv2d` | 全横向，视野 = `1 + 2*depth` | 同横向视野 | 完全二维——横向信息最多但也最重 |

三个公共参数：

| 参数 | 含义 | 怎么调 |
|------|------|------|
| `hidden_channels` | 隐藏层通道数，默认 32 | 越大模型容量越高，但也更容易过拟合合成基准 |
| `depth` | 网络层数，默认 5，须 ≥ 2 | 越大纵向感受野越大；dilated TCN 族随 depth 指数增长 |
| `lateral_kernel` | 仅 `trace_lateral_mixer`，须为正奇数 | 默认 3，增大可让横向混合看到更远的道 |

所有架构的输出层权重和偏置初始化为零，所以**训练前的预测严格等于低频模型**——这保证你看到的所有改善都来自训练，不是初始化的偶然。

### 数据源：喂什么数据

用自定义 ID 声明一个或多个数据源（如 `synthetic`、`field`、`wells`）。ID 可以随便取，后面的损失块通过 ID 引用。

**`synthoseis_lite` — 合成基准**

从合成基准旁路产出的冻结数据。这是消融实验的主战场——所有条件完全受控，可以干净地比较架构和训练策略。

| 字段 | 含义 |
|------|------|
| `benchmark_dir` | 合成基准目录。填 `auto` 自动发现最新场条件基准 |
| `input_seismic_variant` | 网络输入用哪种地震。`nominal` 用理想正演；`observed_mismatch` 用加了噪声和子波失配的地震——训练出来的模型更鲁棒 |

`input_seismic_variant` 的选择影响训练样本种类：`nominal` 只用基础样本，`observed_mismatch` 同时用基础样本和失配变体样本。

**`real_field` — 真实工区**

真实工区的地震体和低频模型。它不提供真值标签，只能用于物理约束——让正演合成尽量接近观测地震。

| 字段 | 含义 |
|------|------|
| `lfm_run_dir` / `variant_id` | 指向第七步产出的低频模型变体 |
| `model_input_seismic_transform` | 网络输入前对地震做什么变换。`identity` 不处理；`p99_abs_matched` 按 p99 绝对值归一化 |
| `physics_target_seismic_transform` | 物理损失的目标地震经过什么变换。可以和模型输入不同——比如网络看 p99 归一化后的地震，但物理损失用原始振幅做比较 |
| `validation_split` | 训练/验证的空间切分方式。`kind: spatial_block` 按 inline 尾段划出验证区，`gap_m` 控制训练和验证之间的隔离带宽度 |
| `wavelet_generation_dir` | 仅时间域，指向第五步子波结果目录 |
| `forward_model_inputs_path` | 仅深度域，指向岩石物理旁路产出的冻结子波和 AI–Vp 关系 |

`model_input_seismic_transform` 和 `physics_target_seismic_transform` 是两条独立管线：输入变换过的是为了让网络好训练，物理目标保持原始振幅是为了让物理约束有意义。两者不能互换。

**`real_wells` — 真实井**

从第六步井控集读取每口井的波阻抗，提供稀疏的绝对阻抗监督。它不独立存在——必须通过 `field_source` 引用一个 `real_field` source，共享该 source 的低频模型和采样轴。

| 字段 | 含义 |
|------|------|
| `field_source` | 引用哪个 `real_field` source |
| `well_control_run_dir` | 第六步井控集目录 |
| `held_out_well` | 始终不参与训练的井名，用作验证 |
| `exclude_same_cluster` | 是否同时排除与 held-out 井同空间簇的井 |
| `cluster_radius_m` | 空间聚类半径，用于均衡采样 |

井按空间位置聚类，每个训练 step 随机选择若干簇、每簇随机选一口井。这样避免了空间相邻井在同一个 batch 里造成梯度重复。

### 损失块：学什么目标

在阶段中声明一个或多个损失块。每个块独立管理自己的 batch、采样器和有效资格判断。

**公共字段（所有 block 都有）：**

| 字段 | 含义 |
|------|------|
| `block_id` | 阶段内唯一标识 |
| `source` | 引用哪个数据源 |
| `weight` | 该 block 损失在总损失中的权重。多个 block 时先各自算 loss，再按 weight 加权求和 |
| `update_interval` | 每多少个 step 执行一次该 block。设为 2 表示隔步执行——低频 block 不占每步的计算量 |
| `batch_size` | 该 block 每个 step 取多少个 patch |
| `min_valid_samples` | 该 block 对 patch 有效性的最低要求。patch 内有效样点不够就丢弃 |

关键设计：**同一个原始窗口可以进入 block A 而不进入 block B**——每个 block 按自己的 mask 和 `min_valid_samples` 独立判断资格。所以各 block 的训练样本不完全重叠是正常的。

**`synthetic_supervised` — 你有答案**

计算网络输出的 `delta_log_ai` 与合成基准真值 delta 的 MSE。只在有效标签点上计算。

最直接的监督信号。优点是干净、梯度好；缺点是只在合成数据上有——真实工区没有这份答案。

**`physics` — 正演闭环**

把预测波阻抗做正演合成地震记录，和观测地震比波形相似度。这是唯一能在真实工区上使用的无标签约束。

| 特有字段 | 含义 |
|------|------|
| `delta_l2_weight` | delta L2 正则的权重。防止模型仅靠正演拟合波形却产生不合理的绝对阻抗值 |
| `waveform_standardization` | 合成数据固定 `raw`，真实工区固定 `masked_centered_rms`（在 patch 内独立做零均值 RMS 归一化后再比较） |

合成数据上：`总损失 = waveform MSE + delta_l2_weight × (pred_delta² 的均值)`。

真实工区上：观测和合成各自在 patch 内独立做 centered RMS 标准化，比较标准化后的波形。这意味着**正演约束对整体振幅不敏感**——它管的是事件时序、极性、相位——绝对阻抗幅度由 LFM 锚定、delta L2 约束、或来自其他阶段的监督提供。

**`real_well_supervised` — 井告诉你的**

在井位处计算预测 delta 与井控 delta 的 MSE。稀疏但绝对——井上的波阻抗是对的。

只有 training 井参与损失；held-out 井和同簇排除井只做验证。

### 阶段：按什么顺序学

一个实验由一个或多个阶段顺序组成。每个阶段独立配置优化器、训练步数、包含哪些损失块、以及用什么指标选最优 checkpoint。

```
阶段 1：synthetic_pretrain
  优化器: AdamW lr=0.001
  损失: synthetic_supervised (weight=1.0)
  验证: 全量，选优指标 = synthetic_ai.mse

阶段 2：field_physics
  优化器: AdamW lr=0.0001   ← 学习率独立设置
  损失: physics (weight=1.0) + real_well_supervised (weight=0.1, update_interval=4)
  验证: 固定 100 步，选优指标 = waveform.total
```

阶段之间只传模型权重。上一阶段的 optimizer 状态（动量、学习率衰减等）全部清零重建。默认从上一阶段 best checkpoint 继承，也可以显式指定 `initialize_from: <stage_id>.final`。

### 如何组合：典型的实验配方

以下是一个完整的两阶段实验配置——先在合成基准上学阻抗反演，再在真实工区上做物理适配：

```yaml
ginn_v2:
  experiment_id: tcn_synthetic_then_field
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

    field:
      kind: real_field
      lfm_run_dir: scripts/output/real_field_lfm_<run>
      variant_id: trend_baseline
      model_input_seismic_transform: p99_abs_matched
      physics_target_seismic_transform: identity
      validation_split: {kind: spatial_block, fraction: 0.10, gap_m: 250.0}
      wavelet_generation_dir: scripts/output/wavelet_generation_<run>

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
      optimizer: {kind: adamw, learning_rate: 0.001, weight_decay: 0.0001}
      loss_blocks:
        - {block_id: synthetic_ai, kind: synthetic_supervised, source: synthetic, weight: 1.0, update_interval: 1, batch_size: 8, min_valid_samples: 128}
      validation: {selection_metric: synthetic_ai.mse, mode: full}

    - stage_id: field_physics
      epochs: 10
      steps_per_epoch: 300
      optimizer: {kind: adamw, learning_rate: 0.0001, weight_decay: 0.0001}
      loss_blocks:
        - {block_id: waveform, kind: physics, source: field, weight: 1.0, update_interval: 1, batch_size: 8, min_valid_samples: 128, delta_l2_weight: 0.01}
      validation: {selection_metric: waveform.total, mode: fixed_steps, steps: 100}

  deployment_checkpoint: last_stage.best
```

**这个配方在测什么？** 第一阶段在合成数据上学会"从地震推断阻抗"的基本能力。第二阶段把这能力迁移到真实工区——没有任何标签，只靠"正演合成应该像观测地震"这一条物理约束来适应真实数据的分布。通过比较两个阶段的 checkpoint，你可以量化物理适配到底带来了多大改善。

**常见的组合思路：**

| 你想知道什么 | 怎么组合 |
|------|------|
| 架构 A 比架构 B 强吗？ | 两个实验，唯一差异是 `architecture.id`，其余全部相同 |
| mismatch 训练有用吗？ | `input_seismic_variant: nominal` vs `observed_mismatch`，在 mismatch 验证集上比指标 |
| 物理约束能改善真实工区迁移吗？ | 阶段 1 纯监督 → 阶段 2 纯物理，对比只用阶段 1 checkpoint 和阶段 2 checkpoint 的 R0 结果 |
| 井监督比物理约束信号更强吗？ | 阶段 2 加一个 `real_well_supervised` block（小 weight），对比纯物理 |
| 低频 block 有效吗？ | 对比 `update_interval: 1` vs `update_interval: 4`（同样的总 step 数），看低频 block 是否浪费了计算 |
| 纯真实工区训练可行吗？ | `normalization_reference.source` 指 `real_field`，只用 `physics` 损失块，不依赖合成标签 |

---

## 脚本在做什么

脚本的运行逻辑分为训练、合成评估和真实工区推理三个独立路径。

### 训练阶段

训练由 `train` 子命令或 `run_experiment` 函数执行。核心流程：

**1) 解析与校验。** 加载 YAML 配置，严格校验每个字段的类型、取值范围和引用完整性。架构 ID、source ID、stage ID、block ID 全部显式声明且交叉引用，不存在隐式推断。旧配置和旧模型标识直接失败并给出明确信息。

**2) 解析数据源。** 对每个声明的 source 做一次解析：合成基准校验 benchmark manifest 和直接上游契约；真实工区体校验低频模型变体和井控集的契约身份一致性。深度域 source 还需要冻结的 AI–Vp 关系和米制子波。

**3) 计算归一化。** 从 `normalization_reference` 指定的 source 的训练分区计算地震和低频模型的均值与标准差。统计量冻结后，所有阶段和后续 R0 推理共用同一份。

**4) 构建块索引。** 按 patching 参数和每个损失块的 `min_valid_samples`，为每个 block 独立构建训练和验证的 patch 索引。同一个窗口可以进入一个 block 而不进入另一个——每个 block 根据自己的监督 mask 独立判断资格。合成数据的训练和验证按父实现分组，同一剖面的不同块不会同时出现在训练集和验证集中，避免空间泄漏。

**5) 逐阶段训练。** 按 stages 列表顺序执行。每个阶段：

- 加载初始权重（零初始化或从指定阶段承接）
- 创建新的 AdamW 优化器
- 按 steps_per_epoch 和 epochs 训练，每个 step 中到期的 loss block 各自取 batch、计算标量损失，加权求和后一次反向传播
- 每 epoch 后在冻结的验证索引上计算选优指标，更新 best checkpoint
- 阶段结束时写出 best 和 final 两个 checkpoint 及完整训练历史

**6) 写入实验 manifest。** 记录完整的架构合同、归一化统计量、全部 source 路径和契约指纹、有序 stage 配置、每阶段的 checkpoints 和训练历史、部署 checkpoint 的显式引用，以及实验级唯一契约指纹。

### 合成评估

训练完成后，可以在合成基准上做评估。流程分两步：

**预测（`predict` 子命令）：** 加载指定 checkpoint，在合成基准的指定 split 上做前向推理。对每个 patch 输出预测 logAI、真值、低频模型和 mask，写入 `predictions.npz`。

**报告（`report` 子命令）：** 消费预测结果，计算回归指标（RMSE、NRMSE、相关系数）、几何分解指标（边界误差、事件误差、横向梯度误差）、拼接指标和频率探针指标。同时计算低频模型基线和理想低频基线的对照指标。输出指标 CSV 和多面板对比图。

### 真实工区推理

训练完成的模型可以直接用于真实工区的零样本推理（R0）。详见 [08 R0 真实工区零样本预测](8-r0-real-field-zero-shot.md)。新架构下的关键变化：

- 通过 `experiment_id` 引用模型，不再使用 `model_role`
- 部署 checkpoint 的 patch 几何、归一化和覆盖合同全部冻结在 manifest 中
- R0 推理读取这些合同，确保任何包含有效样点的窗口都参与推理，不再因全局有效比例阈值丢弃窗口

---

## 核心输出文件

所有文件在 `experiments/ginn_v2/results/<experiment_id>/` 下。

### 实验级文件

| 文件 | 内容 |
|------|------|
| `experiment_manifest.json` | 完整实验身份：架构合同、source 路径和契约、归一化、有序 stage 摘要、部署 checkpoint 引用、实验级契约指纹 |
| `model_run_manifest.json` | 与 experiment manifest 内容相同，用于下游模型消费 |
| `normalization.json` | 冻结的地震和 LFM 均值/标准差 |
| `input_reference_stats.json` | 地震数据值域变换的参考统计量 |

### 阶段产物

每个阶段在 `stages/<stage_id>/` 下：

| 文件 | 内容 |
|------|------|
| `checkpoint_best.pt` | 选优指标最优的权重，使用 `ginn_v2_checkpoint_v4` schema |
| `checkpoint_final.pt` | 最终 epoch 的权重 |
| `training_history.csv` | 每个 epoch 的指标、loss 分量、batch 计数和耗时 |
| `<block_id>_patch_index.csv` | 该 block 的全部 patch 索引、划分和有效性计数 |

### 预测和报告输出

在指定的输出目录下：

| 文件 | 内容 |
|------|------|
| `predictions.npz` | 预测 logAI、真值、低频模型和 mask |
| `prediction_index.csv` | 每个 patch 的元数据 |
| `model_patch_metrics.csv` | 每个 patch 的回归指标 |
| `model_geometry_patch_metrics.csv` | 每个 patch 的几何分解指标 |
| `model_report_card.json` | 汇总报告卡 |

---

## 如何阅读结果

### 第一步：看终端输出

确认训练正常完成、阶段数和 epoch 数与配置一致。关键信息包括最终选优指标值和部署 checkpoint 的引用。

### 第二步：看 experiment manifest

打开 `experiment_manifest.json`，关注：

- `status`：应为 `ok`
- `architecture`：确认网络拓扑符合实验意图
- `deployment_checkpoint`：确认部署使用的 stage 和 checkpoint kind
- `stages` 中每阶段的 `best_selection_metric_value`：确认训练过程中指标在改善

### 第三步：看训练历史

打开 `stages/<stage_id>/training_history.csv`：

- 选优指标是否随 epoch 单调下降。如果某个 epoch 后不再改善，说明该阶段可能已经收敛。
- 多个 block 的 loss 分量是否都在合理范围内下降。某个 block 的 loss 不下降可能意味着其权重过小或数据源存在问题。
- 物理损失中 `observed_centered_rms` 和 `predicted_centered_rms` 的比值——这个比值反映预测和观测的整体振幅关系，偏离 1 过远时说明模型在振幅拟合上存在系统偏差。

### 第四步：看合成评估指标

打开 `model_report_card.json`（或汇总后的 `ablation_report_card.json`）：

- `oracle_aggregate`：oracle 的 RMSE 必须接近零。不为零说明基准数据存在泄漏或评估逻辑异常。
- `rmse_improvement_pct_vs_lfm`：模型相对于退化低频模型基线的提升百分比。如果接近零，说明模型几乎没有学到超越先验的信息。
- `geometry_aggregate`：`mean_boundary_rmse` 远高于 `mean_event_rmse` 说明模型在阻抗突变处有模糊化倾向。`mean_lateral_gradient_rmse` 衡量横向梯度的预测精度。

### 第五步：对比消融维度

如果同时训练了多个实验（如不同的架构或不同的阶段组合），对比它们的 report card：

- 逐道架构 vs 横向架构：横向模型的 lateral_gradient_rmse 应当更低。差异集中在哪些频带可以通过频带拆分图判断。
- 仅监督 vs 监督加物理约束：物理约束后的模型在 mismatch 样本上的指标应当更稳定。
- 仅合成训练 vs 合成加真实井监督：井监督后的模型在井位的 delta 应当在物理合理范围内进一步改善。

---

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| 配置缺少 `ginn_v2` 根节点 | 使用了旧版训练配置格式 | 按当前模板重写配置 |
| 旧模型标识或旧 checkpoint | 使用了旧 schema 的产物 | 用当前版脚本重新训练 |
| 损失块 source 类型不匹配 | 损失块引用了不兼容的数据源 | 检查每个 block 的 kind 和 source 是否匹配 |
| 验证指标未引用当前块 | `selection_metric` 指向了不存在的 block | 修改为当前阶段的某个 `block_id.metric` |
| 真实工区没有训练或验证块 | 空间切分或有效样点下限过严 | 检查 validation split 参数和 `min_valid_samples` |
| 阶段没有正权重且 `update_interval: 1` 的 block | 每个 step 必须有至少一个 block 提供梯度 | 确保至少一个 loss block 的 update_interval 为 1 且 weight 为正 |
| 第一阶段未从零初始化 | `initialize_from` 引用了不存在的阶段 | 第一阶段使用 `initialize_from: zero` 或省略此字段 |
| 部署 checkpoint 引用无效 | `deployment_checkpoint` 指向不存在的阶段 | 检查 stage ID 和 best/final 后缀 |
| 合成基准 `auto` 发现失败 | `experiments/synthoseis_lite/results/` 下没有完整基准 | 先生成合成基准 |
| 真实井源缺少 `field_source` | `real_wells` 必须引用一个 `real_field` source | 添加 `field_source` 字段并确保引用的 source 存在 |

---

## 留到第二轮

- 训练超参自动搜索：`hidden_channels`、`depth` 和损失块权重等关键超参目前靠手调，后续应支持网格搜索或贝叶斯优化。
- 架构快速扩展：当前添加新架构需要改模型文件和注册表，后续应支持通过配置注册自定义网络。
- 与真实工区 R1 诊断结果的双向反馈：真实工区正演闭环诊断的结论反向校准合成基准的退化参数，使两个闸门的难度对齐。
- 早期停止策略：当前固定 epoch 数训练，后续应支持基于验证指标的 patience-based 早期停止。
- 跨工区模型迁移的 OOD 检测自动化：当前跨工区使用 adapted checkpoint 需要手动确认，后续应自动量化输入分布偏移程度。
- 合成探针增量评估的完整覆盖：当前频率探针指标依赖独立的 predict + report 路径，后续应纳入训练后的默认评估管线。
