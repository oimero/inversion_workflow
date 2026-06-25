# 模型消融训练与评估

`ginn_v2.py` 是第三个研究闸门，也是合成数据链的消费端。它做一件事：**在冻结的合成基准上训练并评估轻量反演模型，通过受控消融实验揭示不同架构选择的实际增益。** 训练完成后，自动在合成基准的三个验证范围上做推理评估，最终汇总成一份消融报告卡，为后续真实工区的模型选择提供定量依据。

---

## 快速开始

```bash
# 方式一（推荐）：PowerShell 一键运行
cd experiments/ginn_v2
.\train_network.ps1 <实验名称>                    # 训练 + 三组预测 + 三组报告
.\run_synthetic_gate.ps1 <闸门名> <实验名1> ...     # 汇总 + 盖章

# 方式二：手动分步运行
# 第一步：训练
python scripts/ginn_v2.py train --config experiments/ginn_v2/train.yaml

# 第二步：预测（三组）
python scripts/ginn_v2.py predict --model-run-dir <训练输出目录> \
  --sample-kind base --split validation
python scripts/ginn_v2.py predict --model-run-dir <训练输出目录> \
  --sample-kind base --split test
python scripts/ginn_v2.py predict --model-run-dir <训练输出目录> \
  --sample-kind base frequency_probe seismic_variant --split validation

# 第三步：报告（三组，每组预测各跑一次）
python scripts/ginn_v2.py report --prediction-dir <预测输出目录>

# 第四步：汇总
python scripts/ginn_v2.py summarize \
  --report <模型角色>:<范围>:<报告目录> \
  --report ...

# 第五步：盖章
python scripts/ginn_v2.py stamp-gate \
  --model-run-dir <训练输出目录> \
  --synthetic-gate-report-dir <闸门目录> \
  --synthetic-gate-report-card <闸门目录>/ablation_report_card.json
```

`train_network.ps1` 封装了第一到第三步，`run_synthetic_gate.ps1` 封装了第四到第五步。日常迭代用 runner，调试或自定义评估范围时用手动命令。

---

## 运行前需要什么

| 输入 | 来源 | 用途 |
|------|------|------|
| `synthetic_benchmark.h5` | 合成基准生成 | 冻结的训练数据和真值 |
| `sample_index.csv` | 合成基准生成 | 样本切分索引 |
| `benchmark_manifest.json` | 合成基准生成 | 数据完整性校验 |
| `experiments/ginn_v2/train.yaml` | 用户配置 | 模型架构、训练超参 |

`train.yaml` 中 `benchmark_dir` 设为 `auto` 时，脚本自动从合成基准的实验结果目录中找最新合格的场条件基准。

---

## 配置参考

```yaml
# 配置文件：experiments/ginn_v2/train.yaml

train:
  benchmark_dir: auto          # auto = 自动找最新合成基准，也可填显式路径
  model_id: trace_1d_dilated_tcn_mismatch_training
  model_role: no_lateral       # 留空则从 model_id 自动推断

  # 切块参数
  patch_lateral: 32            # 横向样本数
  patch_twt: 128               # 时间方向样本数
  lateral_stride: 16           # 横向滑动步长
  twt_stride: 64               # 时间方向滑动步长
  min_valid_fraction: 0.50     # 块内最低有效像素比例

  # 数据划分
  split_policy: derive         # derive = 按剖面分组划分；strict = 随机划分
  validation_fraction: 0.15    # 校验集比例
  test_fraction: 0.15          # 测试集比例
  max_patches:                 # 限制最大块数，留空则使用全量

  # 训练超参
  epochs: 5
  batch_size: 8
  learning_rate: 0.001
  hidden_channels: 32
  depth: 5

  # 设备与种子
  device: auto
  seed: 0

  # 物理正演辅助损失。设为 >0 时仅对 base 样本生效。
  # 支持模型：patch_2d_with_physics_loss、trace_1d_dilated_tcn_mismatch_training
  lambda_physics: 0.0
```

### `benchmark_dir`

设为 `auto` 时，脚本从 `experiments/synthoseis_lite/results/` 下搜索含 `generate_field_conditioned` 子目录且关键文件齐全的最新结果。也可以直接填写合成基准目录的路径来锁定特定版本。

### `model_id`

可选模型架构（共 10 个注册 ID，按族分组）：

| 族 | 架构 ID | 说明 |
|------|------|------|
| 一维基础 | `trace_1d` | 单道 TCN，仅用 base 样本训练 |
| | `trace_1d_mismatch_training` | 单道 TCN，加入失配样本训练 |
| 一维膨胀 | `trace_1d_dilated_tcn` | 膨胀时间卷积，更长感受野，仅 base |
| | `trace_1d_dilated_tcn_mismatch_training` | 膨胀时间卷积 + 失配训练，也可开启物理损失 |
| 横向混合 | `trace_1d_tcn_lateral_mixer_k1_mismatch_training` | 膨胀 TCN + 邻道横向混合（核宽=1） |
| | `trace_1d_tcn_lateral_mixer_mismatch_training` | 膨胀 TCN + 邻道横向混合（核宽=3，默认） |
| | `trace_1d_tcn_lateral_mixer_k5_mismatch_training` | 膨胀 TCN + 邻道横向混合（核宽=5） |
| 二维 | `patch_2d_supervised` | 二维块监督学习基线，仅 base |
| | `patch_2d_mismatch_training` | 二维块 + 失配对抗训练 |
| | `patch_2d_with_physics_loss` | 二维块 + 物理正演辅助损失 |

### `model_role`

模型在消融实验中的角色标签。留空时从 `model_id` 自动推断：含 `lateral_mixer` 的推断为 `lateral`，含 `trace_1d` 的推断为 `no_lateral`。这个标签决定了后续真实工区预测时结果存放的子目录名。

### 切块参数

合成基准的剖面被滑动窗口切成小块，每个块包含 `patch_lateral` 条道和 `patch_twt` 个时间采样点。步长越小块数越多，训练越慢但覆盖越密集。`min_valid_fraction` 过滤掉有效像素不足的块。

---

## 脚本在做什么

### 训练阶段

1. **加载基准**：校验合成基准的文件完整性（SHA-256），读取样本索引。
2. **构建块索引**：按切块参数和划分策略，将所有可用剖面切分为训练块、校验块和测试块。`split_policy=derive` 时，同一剖面的块全部分到同一集合，避免训练集和校验集中出现同一剖面的不同块。
3. **计算归一化与参考统计量**：在训练集上计算地震和低频模型的均值、标准差。
4. **训练循环**：每个 epoch 后在校验集上评估损失并记录最佳值。

### 预测与报告阶段

训练完成后，`train_network.ps1` 自动跑三组评估：

| 范围 | 数据划分 | 样本类别 | 测什么 |
|------|---------|---------|--------|
| `validation_base` | 校验集 | 只有基础样本 | 核心精度——干净数据上学进去没有 |
| `test_base` | 测试集 | 只有基础样本 | 泛化能力——未见剖面上的真实水平 |
| `validation_mismatch` | 校验集 | 基础 + 频率探针 + 地震变体 | 鲁棒性——噪声、子波失配等干扰下扛不扛得住 |

每组评估先运行预测（加载 checkpoint 做前向推理），再运行报告（与真值对比计算指标）。

### 汇总与盖章阶段

`run_synthetic_gate.ps1` 消费所有模型的全部报告：

1. **汇总**：把分散的模型报告合并成一份消融报告卡，并排对比各模型在各范围下的指标，自动选出各维度上的最优模型。
2. **盖章**：将消融报告卡的路径和内容指纹写回每个模型的运行清单。这一步是后续真实工区入口的准入门槛——运行清单中缺少合成闸门证据的模型不会被真实工区接受。

---

## 核心输出文件

### 训练输出

所有文件在 `experiments/ginn_v2/results/<实验名称>/` 下：

| 文件 | 内容 |
|------|------|
| `checkpoint.pt` | 模型权重 |
| `training_history.json` | 每个 epoch 的训练和校验损失 |
| `patch_index.csv` | 全部块的索引、划分和元数据 |
| `normalization.json` | 输入数据的归一化参数 |
| `input_reference_stats.json` | 地震和低频模型的参考统计量 |
| `model_run_manifest.json` | 运行清单：架构、超参、基准哈希、切块参数、训练结果摘要 |

### 预测与报告输出

在训练输出目录下：

| 子目录 | 内容 |
|--------|------|
| `predict_validation_base/` | 校验基础集的预测张量 |
| `report_validation_base/` | 校验基础集的指标报告卡 |
| `predict_test_base/` | 测试基础集的预测张量 |
| `report_test_base/` | 测试基础集的指标报告卡 |
| `predict_validation_mismatch/` | 校验不匹配集的预测张量 |
| `report_validation_mismatch/` | 校验不匹配集的指标报告卡 |

每份报告卡包含：回归指标（RMSE、NRMSE、相关系数）、低频模型基线和理想低频基线的对照指标、几何指标（边界误差、事件误差、横向梯度误差）、块级缝合评估指标。

### 闸门输出

所有文件在 `experiments/ginn_v2/results/synthetic_gate_<闸门名称>/` 下：

| 文件 | 内容 |
|------|------|
| `ablation_report_card.json` | 消融报告卡：各模型各范围指标并排对比，最优模型推荐 |
| `synthetic_gate_manifest.json` | 闸门清单：参与模型列表、报告来源、盖章记录 |

---

## 如何阅读结果

### 第一步：看训练输出

```
Patches: 192390
Best validation loss: 0.39
```

块数太少说明基准规模小或 `min_valid_fraction` 设得太高。训练损失和校验损失差距过大说明过拟合——增加 `max_patches` 或减小 `batch_size`。

### 第二步：看消融报告卡

打开 `ablation_report_card.json`，关注三个关键字段：

- **`best.test_base_by_rmse`**：在未见剖面上的最小 RMSE 模型。这是消融实验的核心结论——谁泛化最好。
- **`best.mismatch_by_rmse`**：在不匹配数据上的最小 RMSE 模型。反映谁的鲁棒性最强。
- **`conclusion`**：自动生成的推荐文字。如果 recommend 为 `continue_ablation`，说明横向混合模型仍未显著超过单道模型，可以继续增加消融维度。

### 第三步：对比低频模型基线

每次报告都包含 `lfm_rmse`（退化低频模型基线）和 `lfm_ideal_rmse`（理想低频基线）。关键技术指标是 `rmse_improvement_pct_vs_lfm`——模型相对于退化低频基线提升了多少。如果这个值接近零，说明模型几乎没有学到超越先验的信息。

### 第四步：检查自检值

`oracle_rmse` 必须为 0（或机器精度级别接近 0）。不为 0 说明基准数据存在泄漏或加载路径错误。

### 第五步：看几何分解指标

`geometry_boundary_rmse`、`geometry_event_rmse`、`geometry_lateral_gradient_rmse` 分别衡量模型在层位边界、地质事件和横向梯度上的表现。如果边界误差远高于事件误差，说明模型在阻抗突变处有模糊化倾向。

---

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| `No synthoseis_lite result found` | `benchmark_dir: auto` 找不到合格合成基准 | 确认 `experiments/synthoseis_lite/results/` 下有含 `generate_field_conditioned` 的完整结果 |
| 训练损失不下降 | 学习率或架构不适配当前基准 | 调低学习率、增加 `epochs`、或换模型架构 |
| 校验损失远高于训练损失 | 过拟合 | 增加 `max_patches` 限制数据量、减小模型容量（`hidden_channels`、`depth`） |
| `best_validation_loss` 极大 | 模型几乎没有学到有效映射 | 检查归一化是否正确、基准数据是否合理 |
| `oracle_rmse` 不为 0 | 基准数据存在泄漏 | 重新生成合成基准，检查 SHA-256 一致性 |
| stamp-gate 报 `missing` | 运行清单缺少某些报告 | 确认 `train_network.ps1` 的三组评估全部完成 |
| 预测时显存溢出 | 评估块数或块尺寸过大 | 增大 `lateral_stride` 和 `twt_stride`，或设置 `max_patches` |
| `model_role` 自动推断错误 | `model_id` 命名不规范 | 在 `train.yaml` 中显式填写 `model_role` |

---

## 留到第二轮

- 模型架构快速扩展机制：当前添加新架构需要改多处代码和注册表，后续应支持通过配置文件注册自定义模型。
- 探针增量评估与频率选择性诊断：当前探针相关指标在核心评估中未完全覆盖，后续应将探针对比和频率假能量检查纳入默认评估范围。
- 训练超参自动搜索：`hidden_channels` 和 `depth` 等关键超参目前靠手调，后续应支持网格搜索或贝叶斯优化。
- 与真实工区诊断结果的双向反馈：真实工区 R1 的正演诊断结论反向校准合成基准的退化参数，使两个闸门的难度对齐。
