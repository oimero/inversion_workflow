# GINN v2 模型训练与消融

`ginn_v2.py` 是工作流的研究旁路。它把网络架构、数据源、损失函数和训练阶段分别配置，通过受控消融实验揭示不同架构选择和训练策略的实际增益。一个实验由稳定的实验标识管理，产物写入实验目录，可直接用于后续真实工区的零样本推理。

完整字段契约见 [GINN v2 积木式训练设计](../spec/GINN_V2_COMPOSABLE_TRAINING_DESIGN.md)。本文档聚焦实用工作流，不重复规范中的全部字段细节。

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

配置根节点为 `ginn_v2`，包含以下顶层段：

```yaml
ginn_v2:
  experiment_id: <descriptive-id>      # 必填，决定产物目录名
  seed: 20260617

  architecture:                         # 网络拓扑
    id: trace_dilated_tcn
    hidden_channels: 32
    depth: 5

  sources:                              # 数据源声明
    synthetic:
      kind: synthoseis_lite
      benchmark_dir: auto
      input_seismic_variant: observed_mismatch

  normalization_reference:              # 归一化统计量的来源
    source: synthetic

  patching:                             # 滑动窗口参数
    lateral_samples: 32
    vertical_samples: 128
    lateral_stride: 16
    vertical_stride: 64

  stages:                               # 有序训练阶段列表
    - stage_id: synthetic_pretrain
      ...

  deployment_checkpoint: last_stage.best  # 部署时使用的 checkpoint
```

### 架构

首版提供四类网络，按横向感受野从小到大排列：

| 架构 | 网络 | 横向行为 |
|------|------|------|
| `trace_conv1d` | 逐道一维卷积 | 道间独立 |
| `trace_dilated_tcn` | 逐道膨胀一维 TCN | 道间独立 |
| `trace_lateral_mixer` | 膨胀 TCN 加浅层横向混合 | 由 mixer 核大小决定 |
| `patch_conv2d` | 二维卷积 | 由卷积层数决定 |

所有网络固定接收三个输入通道（地震、低频模型、有效掩码），直接输出物理量 `delta_log_ai`。输出层使用零权重和零偏置初始化，因此未训练模型的预测严格等于低频模型。

架构标识只描述网络拓扑，不编码训练样本类型或损失函数。任何架构都可以使用任何首版损失块——能力由组合决定，不由架构限制。

### 数据源

实验可以声明三类数据源，以用户自定义 ID 为键：

| source kind | 提供什么 | 可用损失块 |
|------|------|------|
| `synthoseis_lite` | 合成地震、真值 logAI、LFM、一致正演 | 合成监督、物理约束 |
| `real_field` | 真实地震、工区 LFM、有效掩码 | 物理约束 |
| `real_wells` | 井 logAI、LFM、空间簇信息 | 真实井监督 |

每个 source 在实验开始时解析一次并冻结。`real_wells` 必须引用一个 `real_field` source——井监督使用该 source 的低频模型和采样轴，不自行发现。

深度域真实物理约束的数据源需要显式填写正演输入合同文件路径；时间域则填写子波结果目录。配置示例见规范文档。

### 损失块

三类损失块与数据源一一对应：

| loss kind | 接受 source | 核心计算 |
|------|------|------|
| `synthetic_supervised` | `synthoseis_lite` | 预测 delta logAI 与真值 delta 的 masked MSE |
| `physics` | `synthoseis_lite` 或 `real_field` | 预测 logAI 正演合成与观测地震的 waveform MSE + delta L2 |
| `real_well_supervised` | `real_wells` | 井旁预测 delta 与井控 delta 的 MSE |

每个损失块独立声明自己的 batch size、update interval、权重和有效样点下限。一个阶段可以组合多个不同数据源的损失块——到期损失按权重求和后只执行一次参数更新。

物理损失在合成和真实工区上有不同的标准化策略。合成数据使用原始振幅的 masked MSE；真实工区数据在 patch 内独立做 centered RMS 标准化后再比较波形，不引入可训练增益。

### 多阶段训练

阶段按配置顺序执行。每个阶段独立声明：

- 优化器（首版仅支持 AdamW）
- 训练 epoch 数和每 epoch 的 step 数
- 包含哪些损失块
- 验证模式（全量或固定步数）和选优指标

阶段之间只继承模型权重，不继承优化器或调度器状态。默认从上一阶段的最佳 checkpoint 初始化，也可以显式引用更早阶段的 best 或 final。第一阶段使用零初始化模型。

支持的阶段组合包括：监督后物理约束、物理约束后监督、多次交替，以及同阶段多数据源联合训练。

### 旧配置兼容

以下配置会被明确拒绝，错误信息会指明期望的 schema 和文档入口：

- 根节点为旧 `train` 的配置
- 包含 `model_id`、`model_role`、`min_valid_fraction` 的配置
- 旧的十个带训练语义的模型标识

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
| 阶段没有任何 `update_interval: 1` 的 block | 每个 step 必须有至少一个 block 提供梯度 | 确保至少一个 loss block 的 update_interval 为 1 |
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
