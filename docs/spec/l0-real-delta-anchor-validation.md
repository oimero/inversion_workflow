# L0 Real-Delta Anchor Validation

## 文档地位

L0 是一项独立的真实井监督研究验证：在 GINN-v2 synthetic truth-first 训练中，从首个
epoch 起加入稀疏真实井 `delta` anchor，检验这种监督能否改善一口预注册、未参与 anchor
训练的真实井。

阶段关系为：

```text
reference synthetic training recipe + Step 7 real-field LFM
  -> L0 paired control + configured single-well holdout
  -> positive only -> L1 all-well application and full-field evaluation
```

L0 固定只训练两个模型：

1. `control`：完全复现 synthetic 配方，`lambda_anchor=0`。
2. `holdout`：相同初始化和 synthetic batch 序列，加入真实井 anchor，但排除配置的
   `held_out_well`。

L0 不训练 all-well 应用模型，不生成完整工区预测体。

## 1. L0 回答的问题

```text
在 synthetic truth 仍是主监督的条件下，
真实井 delta anchor 能否改善一口预注册 holdout 井，
同时不造成 delta 塌缩或 synthetic 灾难性退化？
```

这是 `anchor-label holdout`，不是端到端井盲测。当前 LFM、子波和 synthetic 分布校准仍可能
使用过 holdout 井。

若默认 `exclude_same_cluster=false`，holdout 井同空间簇的其他井仍可进入训练。此时结果必须
带 `same_cluster_training_leakage_risk=true`，只能解释为局部密井条件下的标签迁移。

## 2. 冻结实验

### 2.1 模型与训练

- 只运行 `no_lateral` GINN-v2。
- 从随机初始化重新训练完整网络，不加载参考 checkpoint 权重。
- checkpoint 只用于读取并校验架构元数据。
- control 与 holdout 从同一初始 state dict 开始，optimizer state 相互独立。
- epoch、batch size、学习率、normalization、patch spec 和 seed 继承参考
  `model_run_manifest.json`。
- 真实井 anchor 从第一步开始，固定 `lambda_anchor=0.1`。
- 正式评估只使用 final epoch checkpoint，不按 holdout 指标挑 epoch。

训练目标：

```text
well_delta   = filtered_log_ai - lfm_log_ai
well_delta_n = (well_delta - synthetic_delta_mean) / synthetic_delta_std

L = L_synthetic_normalized_delta_MSE
  + 0.1 * L_real_normalized_delta_MSE
```

不加入 corr、gradient、energy 或真实波形 loss。

### 2.2 手工 holdout

配置必须提供：

```yaml
held_out_well: PH5
exclude_same_cluster: false
```

井名只是配置值，代码、run ID 和目录不得硬编码具体井名。

行为定义：

- `exclude_same_cluster=false`：只把 `held_out_well` 从 anchor sampler 排除。
- `exclude_same_cluster=true`：把该井所在空间簇的所有井从 anchor sampler 排除。
- 无论开关如何，正式通过判定只使用预注册的 `held_out_well`。
- 开启同簇排除时，其他被排除井可以输出辅助诊断，但不得参与正式判定。
- 所有被排除井必须在采样审计中显式记录 `selected_count=0`。

更换 holdout 井只修改 YAML，不创建新脚本或模型分支。

## 3. 规模化计算契约

L0 的主体训练开销不得随井数线性增长。

### 3.1 固定训练次数

无论有 11 井还是 50 井，L0 都只训练：

```text
1 control + 1 holdout = 2 complete runs
```

不得按井或空间簇重新训练额外模型。

### 3.2 固定单步 anchor 规模

每个 synthetic optimization step：

```text
k = min(4, n_training_clusters)
```

无放回选择最多 4 个空间簇，每簇均匀循环选择 1 口井。被选井使用其完整有效窗。

```text
L_anchor = (1 / k) * sum_over_selected_clusters(
    mean_over_selected_well_valid_samples(
        (pred_delta_n - well_delta_n)^2
    )
)
```

因此优化 step 数和每步井数都与总井数无关。井数增加只改变 shuffled-cycle 的覆盖周期。

### 3.3 一次性预计算

模型 forward 前的静态工作必须在训练开始前完成一次：

- 每口井轨迹样点的空间/TWT 插值几何。
- 每个支撑网格节点对应的 canonical patch 列表。
- real seismic/LFM/mask 的归一化支撑张量。
- 支撑节点和 trace-patch 数量审计。

缓存只保存静态输入，不保存模型输出或 detach 后隐藏特征。模型 forward、stitch 和轨迹插值
必须保留计算图。

同一步选中的多口井必须合并支撑节点并执行批量 forward；不得为每口井重复启动完整的模型
调用。密井共享的支撑节点应在该 step 内去重。

一次性标签读取和支撑预计算仍是 `O(井数)`；冻结的是主训练的 run 数、optimization step 数
和每步 anchor 规模。

## 4. 输入与来源

L0 显式消费：

1. `reference_training_run_dir`：参考 synthetic 训练 manifest、normalization、patch index 和
   benchmark provenance。
2. `real_field_lfm_dir`：第七步 `real_field_lfm_summary.json` 与 `real_field_lfm.npz`。
3. 第七步 summary 冻结的井震标定目录和井清单。
4. 顶层 `seismic` 配置。

禁止猜测 `latest`。必须校验 manifest、benchmark 三文件、patch index、normalization、输入
统计、LFM、seismic 和派生井来源的路径及 SHA-256。

真实输入统一调用 `ginn_v2.real_field.load_real_field_volume`，不得在 L0 中另写 seismic/LFM
变换。

## 5. 井标签契约

模型无关公共构建器输出 `l0_well_anchor_samples.csv`，至少包含：

```text
well_name, sample_index, twt_s, inline, xline, x_m, y_m,
spatial_cluster_id, spatial_cluster_size,
filtered_log_ai, lfm_log_ai,
valid_for_fit, valid_reason,
sampling_mode, sample_method, wellbore_class
```

训练只使用 `valid_for_fit=true` 且 `sampling_mode=volume` 的样点。空间簇复用
`spatial_debias.cluster_radius_m` 的半径连通定义。

斜井不得拼成弯曲地震道。必须运行周围垂直网格道的 canonical patch/stitch，再对输出执行
可微三维轨迹插值。

训练前必须比较稀疏可微路径与完整 canonical patch 路径：

```text
max_abs(pred_sparse - pred_canonical) * synthetic_delta_std <= 1e-5 logAI
```

超限时整次运行失败。

## 6. 配置与 CLI

```text
python scripts/l0_real_delta_anchor.py
python scripts/l0_real_delta_anchor.py --config experiments/common/common.yaml
python scripts/l0_real_delta_anchor.py --output-dir scripts/output/l0_real_delta_anchor_test
```

配置：

```yaml
l0_real_delta_anchor:
  reference_training_run_dir: <explicit-no-lateral-synthetic-run>
  real_field_lfm_dir: <explicit-step-7-run>
  model_role: no_lateral
  lambda_anchor: 0.1
  anchor_clusters_per_step: 4
  held_out_well: PH5
  exclude_same_cluster: false
  device: cuda
  diagnostic_max_hz: 80.0
  reconstruction_tolerance_log_ai: 1.0e-5
  real_field_inputs:
    seismic_value_transform: p99_abs_matched
    lfm_value_transform: identity
  thresholds:
    minimum_held_out_delta_corr_gain: 0.0
    minimum_held_out_full_ai_corr_gain: 0.0
    maximum_held_out_full_ai_rmse_delta: 0.0
    maximum_good_well_corr_drop: 0.02
    maximum_good_well_rmse_relative_increase: 0.05
    maximum_good_well_delta_corr_drop: 0.02
    synthetic_warning_error_relative_increase: 0.05
    synthetic_warning_corr_drop: 0.02
    maximum_synthetic_error_relative_increase: 0.20
    maximum_synthetic_corr_drop: 0.05
```

CLI 不提供临时 holdout 井或权重覆盖，避免运行命令脱离 YAML 审计。

## 7. 评估与判定

### 7.1 Holdout 指标

对配置的 holdout 井分别计算 control 和 anchor：

- delta corr/RMSE/bias。
- full-AI corr/RMSE/bias。
- delta RMS、连续相邻样点 gradient RMS。
- target-relative delta/gradient energy error。
- 冻结频带指标。
- 真实井 waveform raw/positive-scale 指标。

```text
energy_error = abs(log(pred_rms / target_rms))
```

零或无效 target energy 产生 `invalid_held_out_metric`，不得用 epsilon 伪造通过。

### 7.2 正式通过规则

正式判定只读取 `held_out_well` 的一行 summary：

1. holdout 指标完整有限。
2. `delta_corr_gain > 0`。
3. `full_ai_corr_gain >= 0`。
4. `full_ai_rmse_delta <= 0`。
5. delta-energy error change `<= 0`。
6. gradient-energy error change `<= 0`。
7. 动态 good-well protection 通过。
8. synthetic 无 catastrophic regression。

Synthetic warning 阈值为 error >5% 或 corr drop >0.02；hard failure 为 error >20% 或 corr
drop >0.05。

通过状态：

```text
l0_positive
l0_positive_with_synthetic_warning
```

两者均可进入 L1，但后者必须携带 warning。

单井正结果不能估计跨井改善比例或总体泛化分布；它只说明预注册井上存在/不存在迁移信号。

## 8. 输出

```text
l0_well_anchor_samples.csv
l0_holdout_metrics.csv
l0_holdout_summary.csv
l0_training_history.csv
l0_anchor_sampling_qc.csv
l0_synthetic_preservation.csv
l0_decision_table.csv
l0_real_delta_anchor_summary.json
control/final_checkpoint.pt
holdout/final_checkpoint.pt
figures/wells/<cluster_id>/<well_name>_ai_delta_qc.png
figures/wells/<cluster_id>/<well_name>_control_forward_qc.png
figures/wells/<cluster_id>/<well_name>_anchor_forward_qc.png
```

`l0_holdout_metrics.csv` 每个被排除井 × control/anchor 一行，并用
`is_primary_holdout` 区分正式井和辅助诊断井。

`l0_holdout_summary.csv` 每个被排除井一行。`l0_decision_table.csv` 只能读取
`is_primary_holdout=true` 的行。

Summary 必须记录 holdout 配置、实际排除井、同簇泄漏风险、预计算井/节点/trace-patch 数、
初始权重 hash、两次训练的 synthetic sequence hash、checkpoint/hash 和所有判定规则。

## 9. 井 QC

每个被排除井生成：

1. filtered logAI、LFM、control、anchor 及对应 delta 的同轴图。
2. control 正演 QC。
3. anchor 正演 QC。

正演图复用 `cup.seismic.viz.plot_well_waveform_qc`。连续支撑不足时记录
`insufficient_forward_qc_support`，不静默缺图。

## 10. 测试要求

- control/holdout 初始权重完全一致，optimizer 独立。
- synthetic patch ID 序列完全一致。
- holdout 井及可选同簇井抽样计数严格为零。
- 50 井 fixture 下每步最多 4 井，训练 run 数固定为 2。
- sampler 簇无放回、簇内井循环覆盖且确定性。
- 预计算后静态支撑 tensor 对象不被每步重建。
- 多井批量 forward 与逐井 forward 数值、loss 和梯度等价。
- 稀疏与完整 canonical patch 路径在 `1e-5 logAI` 内闭合。
- 斜井插值可微且梯度有限非零。
- primary holdout 决策不混入辅助被排除井。
- `exclude_same_cluster=false/true` 两种排除集合正确。
- collapse/explosion、good-well、synthetic warning/hard gate 正确。
- L0 只生成 control 和 holdout 两个 checkpoint，不生成 all-well 模型或全工区体。

测试文件由实现阶段写入 `tests/`，用户在 `pinn_inversion` 环境中运行。

## 11. L1 边界

只有 `l0_positive` 或 `l0_positive_with_synthetic_warning` 才允许规划 L1。L1 才使用全部有效
井训练应用模型并生成完整工区体。L0 的 holdout checkpoint 不能作为 all-well 应用模型。
