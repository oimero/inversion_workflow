# GINN v2 深度域精简消融 Handoff

## 0. 执行定位

本 handoff 用于在当前积木式 GINN v2 代码上开展**最小但足以作决策的深度域消融**。

目标不是复制时间域全部历史实验，也不是穷举网络和损失权重，而是回答三个主要矛盾：

1. **Mismatch training 是否必要？**
2. **横向上下文是否有稳定增益？**
3. **Synthetic physics fine-tuning 是否在监督学习之外提供增量？**

正式训练最少 4 个 run。只有结果接近、无法作决策时，才追加确认 seed。

---

## 1. 硬性原则

### 1.1 本轮不做

- 不运行 `trace_conv1d`；
- 不运行 `patch_conv2d`；
- 不扫描 lateral kernel；
- 不扫描 hidden channels、depth 或 patch 大小；
- 不扫描 physics 权重；
- 不做 real-field physics 或 real-well training；
- 不修改网络、loss、mask 或数据生成代码；
- 不重新生成 20260706 benchmark；
- 不以训练 waveform loss 单独判断模型优劣。

### 1.2 为什么只比较 TCN 与 k3 lateral mixer

时间域已有强证据表明 dilated TCN 是可靠的逐道基线，k3 lateral mixer 是当前横向主候选。

本轮只需回答：

```text
深度域中，横向上下文是否比强逐道 TCN 更好？
```

`patch_conv2d` 会同时改变横向行为、垂向结构和参数组织，使变量不够干净；时间不足时不纳入主矩阵。

---

## 2. 数据与固定合同

### 2.1 Benchmark

统一使用：

```text
experiments/synthoseis_lite/results/20260706/generate_field_conditioned
```

预期合同：

```text
sample_domain: depth
sample_unit: m
depth_basis: tvdss
vertical sample step: 5 m
```

开始前记录：

```powershell
git rev-parse HEAD
```

并将 commit 写入最终报告。

### 2.2 固定随机种子

主 seed：

```text
20260712
```

可选确认 seed：

```text
20260713
```

除非进入“结果接近”分支，否则不要运行第二 seed。

### 2.3 固定架构参数

```yaml
hidden_channels: 32
depth: 5
```

lateral 模型固定：

```yaml
lateral_kernel: 3
```

### 2.4 固定 patch

```yaml
patching:
  lateral_samples: 32
  vertical_samples: 128
  lateral_stride: 16
  vertical_stride: 64
```

### 2.5 固定监督训练预算

```yaml
epochs: 10
steps_per_epoch: 300

optimizer:
  kind: adamw
  learning_rate: 0.001
  weight_decay: 0.0001

loss:
  batch_size: 8
  min_valid_samples: 128
```

本轮所有监督模型使用同样预算。

### 2.6 固定归一化

```yaml
normalization_reference:
  source: synthetic
```

不得在模型之间改变 normalization reference。

---

## 3. 正式实验矩阵

## Run A：逐道 TCN，nominal supervision

实验 ID：

```text
depth_tcn_nominal_s20260712
```

目的：

```text
建立不含 mismatch augmentation 的深度域逐道基线。
```

关键配置：

```yaml
architecture:
  id: trace_dilated_tcn
  hidden_channels: 32
  depth: 5

sources:
  synthetic:
    kind: synthoseis_lite
    benchmark_dir: experiments/synthoseis_lite/results/20260706/generate_field_conditioned
    input_seismic_variant: nominal
    physics_target_variant: model_consistent

loss_blocks:
  - block_id: synthetic_ai
    kind: synthetic_supervised
    source: synthetic
    weight: 1.0
    update_interval: 1
    batch_size: 8
    min_valid_samples: 128
    sampling:
      kind: uniform_patch
```

---

## Run B：逐道 TCN，balanced mismatch supervision

实验 ID：

```text
depth_tcn_mismatch_s20260712
```

目的：

```text
隔离 mismatch training 的增量。
```

与 Run A 唯一的实质变化：

```yaml
sources:
  synthetic:
    input_seismic_variant: observed_mismatch
```

以及：

```yaml
sampling:
  kind: balanced_sample_kind
```

必须确认每个 epoch 记录的：

```text
sampled_base_count
sampled_seismic_variant_count
```

两者差值不超过实现合同允许的 1 个样本级误差。

比较：

```text
Run B vs Run A
```

回答 mismatch 是否必要。

---

## Run C：k3 lateral mixer，balanced mismatch supervision

实验 ID：

```text
depth_lateral_k3_mismatch_s20260712
```

目的：

```text
在相同 mismatch、预算和数据下，隔离横向上下文的增量。
```

与 Run B 唯一的实质变化：

```yaml
architecture:
  id: trace_lateral_mixer
  hidden_channels: 32
  depth: 5
  lateral_kernel: 3
```

比较：

```text
Run C vs Run B
```

回答 lateral context 是否必要。

---

## Run D：监督优胜架构 → synthetic physics

实验 ID 在 Run B/C 决策后确定：

```text
depth_<winner>_mismatch_then_physics_s20260712
```

其中 `<winner>` 为：

```text
tcn
或
lateral_k3
```

目的：

```text
检验短 synthetic physics fine-tuning 是否在监督模型之上提供增量。
```

第一阶段完整复制优胜模型的 mismatch supervised 配方：

```yaml
- stage_id: synthetic_pretrain
  epochs: 10
  steps_per_epoch: 300
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
      sampling:
        kind: balanced_sample_kind
  validation:
    selection_metric: synthetic_ai.mse
    mode: full
```

第二阶段：

```yaml
- stage_id: synthetic_physics
  initialize_from: synthetic_pretrain.best
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
      delta_l2_weight: 0.01
      waveform_standardization: raw
      centered_rms_epsilon: 1.0e-12
      min_centered_rms: 1.0e-6
  validation:
    selection_metric: synthetic_waveform.total
    mode: full
```

部署：

```yaml
deployment_checkpoint: synthetic_physics.best
```

不扫描 `delta_l2_weight`。首版固定为：

```text
0.01
```

比较：

```text
Run D vs 对应的 Run B 或 Run C
```

---

## 4. 标准 YAML 模板

DeepSeek 应在下列目录创建配置：

```text
experiments/ginn_v2/depth_ablation_20260712/configs
```

基础模板：

```yaml
ginn_v2:
  experiment_id: REPLACE_EXPERIMENT_ID
  seed: 20260712
  device: auto

  architecture:
    id: REPLACE_ARCHITECTURE
    hidden_channels: 32
    depth: 5
    # lateral_kernel: 3  # 仅 lateral mixer

  sources:
    synthetic:
      kind: synthoseis_lite
      benchmark_dir: experiments/synthoseis_lite/results/20260706/generate_field_conditioned
      input_seismic_variant: REPLACE_INPUT_VARIANT
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
        - block_id: synthetic_ai
          kind: synthetic_supervised
          source: synthetic
          weight: 1.0
          update_interval: 1
          batch_size: 8
          min_valid_samples: 128
          sampling:
            kind: REPLACE_SAMPLING
      validation:
        selection_metric: synthetic_ai.mse
        mode: full

  deployment_checkpoint: last_stage.best
```

替换表：

| Run | `REPLACE_ARCHITECTURE` | `REPLACE_INPUT_VARIANT` | `REPLACE_SAMPLING` |
|---|---|---|---|
| A | `trace_dilated_tcn` | `nominal` | `uniform_patch` |
| B | `trace_dilated_tcn` | `observed_mismatch` | `balanced_sample_kind` |
| C | `trace_lateral_mixer` + `lateral_kernel: 3` | `observed_mismatch` | `balanced_sample_kind` |

Run D 单独创建双阶段 YAML。

---

## 5. 执行顺序

### 5.1 一次最小 smoke

只对 Run A 做一次短 smoke。

创建临时副本：

```text
depth_tcn_nominal_smoke.yaml
```

把训练量改为：

```yaml
epochs: 1
steps_per_epoch: 10
```

运行并确认：

- 配置解析成功；
- sample domain 为 depth/TVDSS；
- normalization 输出有限；
- checkpoint best/final 均生成；
- manifest 记录 5 m 采样合同；
- prediction/report 命令可运行。

smoke 通过后删除或明确标记临时产物，不纳入比较。

### 5.2 正式训练

依次运行：

```powershell
python scripts/ginn_v2.py train --config experiments/ginn_v2/depth_ablation_20260712/configs/depth_tcn_nominal_s20260712.yaml

python scripts/ginn_v2.py train --config experiments/ginn_v2/depth_ablation_20260712/configs/depth_tcn_mismatch_s20260712.yaml

python scripts/ginn_v2.py train --config experiments/ginn_v2/depth_ablation_20260712/configs/depth_lateral_k3_mismatch_s20260712.yaml
```

默认输出应位于：

```text
experiments/ginn_v2/results/<experiment_id>
```

Run A/B/C 全部评估后再决定 Run D 架构。

---

## 6. 每个模型的统一 test 评估

对每个正式模型，用 deployment checkpoint 在重新生成的 eval index 上跑完整 test：

```powershell
python scripts/ginn_v2.py `
  --output-dir experiments/ginn_v2/depth_ablation_20260712/predictions/<experiment_id>_test `
  predict `
  --model-run-dir experiments/ginn_v2/results/<experiment_id> `
  --benchmark-dir experiments/synthoseis_lite/results/20260706/generate_field_conditioned `
  --index-source eval `
  --split test `
  --checkpoint primary
```

然后：

```powershell
python scripts/ginn_v2.py `
  --output-dir experiments/ginn_v2/depth_ablation_20260712/reports/<experiment_id>_test `
  report `
  --prediction-dir experiments/ginn_v2/depth_ablation_20260712/predictions/<experiment_id>_test
```

不要只读取训练 validation history。正式比较使用同一套 test report。

---

## 7. 汇总命令

A/B/C 完成后：

```powershell
python scripts/ginn_v2.py `
  --output-dir experiments/ginn_v2/depth_ablation_20260712/summary_screen `
  summarize `
  --report "depth_tcn_nominal:test:experiments/ginn_v2/depth_ablation_20260712/reports/depth_tcn_nominal_s20260712_test" `
  --report "depth_tcn_mismatch:test:experiments/ginn_v2/depth_ablation_20260712/reports/depth_tcn_mismatch_s20260712_test" `
  --report "depth_lateral_k3_mismatch:test:experiments/ginn_v2/depth_ablation_20260712/reports/depth_lateral_k3_mismatch_s20260712_test"
```

Run D 完成后重新汇总四个模型。

---

## 8. 决策指标

不要构造复杂加权总分。

按下列优先级判断。

### 8.1 第一优先级：AI truth

从 `model_report_card.json` 和 `ablation_summary.csv` 读取：

```text
model_rmse
model_nrmse
model_corr
rmse_improvement_pct_vs_lfm
```

### 8.2 第二优先级：geometry holdout

读取：

```text
geometry_holdout_rmse
model_patch_metrics_geometry_holdout_by_family.csv
```

pinchout 必须单列，不得只看整个 test 平均。

### 8.3 第三优先级：几何结构

读取：

```text
geometry_boundary_rmse
geometry_event_rmse
geometry_lateral_gradient_rmse
```

lateral mixer 是否有价值，重点看：

```text
geometry_holdout_rmse
geometry_lateral_gradient_rmse
```

而不是只看全局相关系数。

### 8.4 第四优先级：frequency probe

读取：

```text
probe_nrmse
probe_corr
probe_mean_abs_amplitude_error
probe_median_abs_phase_error_deg
unsupported_false_frequency_rms
```

用于排除通过过度平滑获得较低平均 RMSE 的模型。

---

## 9. 明确决策规则

这些阈值是项目管理门槛，不是普适统计定律。

## 9.1 Mismatch：Run B vs Run A

保留 mismatch training，如果满足：

1. test `model_rmse` 改善至少 1%；并且
2. geometry holdout 不恶化超过 1%；并且
3. probe/false-frequency 指标没有明显恶化。

若差异小于 1%，视为实际平局；考虑真实工区存在 domain gap，仍可保留 mismatch，但报告必须写明 synthetic 增益不显著。

若 mismatch 明显恶化 AI truth 或 holdout，则 nominal 获胜，不因“更贴近真实”而强行保留 mismatch。

## 9.2 Lateral：Run C vs Run B

lateral mixer 只有满足以下条件才成为主候选：

1. geometry holdout RMSE 改善至少 2%；或 lateral-gradient 指标有清楚改善；
2. 全局 test RMSE 不恶化超过 1%；
3. frequency probe 不出现明显退化。

若差异小于 1%，视为平局并选择更简单的：

```text
trace_dilated_tcn
```

若 lateral 只让图像更平滑、但 holdout 和结构指标没有改善，选择 TCN。

## 9.3 Physics：Run D vs supervised winner

保留 physics fine-tuning，必须满足：

1. geometry holdout 或 probe 至少一项有可见改善；
2. 全局 AI RMSE 不恶化超过 1%；
3. `delta_log_ai` 没有明显 collapse；
4. 改善不只是 `synthetic_waveform.total` 下降。

若 physics 只改善 waveform loss，而 AI truth、holdout 或 probe 变差，则拒绝 physics。

---

## 10. 可选第二 seed

仅在以下情况追加：

```text
关键比较差异落在 1%–2%；
或全局指标和 holdout 指标方向冲突。
```

追加 seed：

```text
20260713
```

只重跑发生争议的两个模型，不重跑全部矩阵。

例如 lateral 结果接近，只重跑：

```text
depth_tcn_mismatch_s20260713
depth_lateral_k3_mismatch_s20260713
```

若第二 seed 仍不一致，判定没有稳定 lateral 增益，选择简单 TCN。

---

## 11. 真实工区阶段

本 handoff 不安排新的 real-field physics 或井监督训练。

synthetic gate 完成后，只将以下两个模型送入现有 R0/R1：

1. 最终 synthetic 主候选；
2. `depth_tcn_mismatch_s20260712` 作为逐道参考；如果它本身就是主候选，则额外保留 nominal TCN 作参考。

真实工区只回答：

- zero-shot 正演相关性；
- 相对 LFM 的 residual energy reduction；
- lateral 模型是否在真实体上提供额外价值；
- 井旁 AI 是否有正向信号。

不要因为真实正演相关性高就跳过 synthetic truth gate。

---

## 12. DeepSeek 最终交付物

DeepSeek 应提交：

```text
experiments/ginn_v2/depth_ablation_20260712/
├── configs/
│   ├── depth_tcn_nominal_s20260712.yaml
│   ├── depth_tcn_mismatch_s20260712.yaml
│   ├── depth_lateral_k3_mismatch_s20260712.yaml
│   └── depth_<winner>_mismatch_then_physics_s20260712.yaml
├── predictions/
├── reports/
├── summary_screen/
├── summary_final/
├── commands.ps1
└── DEPTH_ABLATION_DECISION.md
```

`DEPTH_ABLATION_DECISION.md` 至少包含：

1. 运行 commit；
2. 四个实验的实际配置摘要；
3. 训练是否完成及 best epoch；
4. base/variant 实际采样数；
5. test aggregate；
6. pinchout holdout；
7. geometry metrics；
8. frequency probe；
9. 三个问题的结论：
   - mismatch 是否保留；
   - lateral 是否保留；
   - physics 是否保留；
10. 进入真实 R0/R1 的模型列表。

---

## 13. 停止条件

满足以下任一项时停止当前分支，不追加实验：

- mismatch 明确失败；
- lateral 改善低于 1%；
- physics 导致 AI truth 或 holdout 恶化超过 1%；
- 第二 seed 仍无法证明复杂模型稳定优于简单模型。

默认原则：

```text
证据不足时，选择更简单的模型和更少的训练阶段。
```

---

## 14. 给 DeepSeek 的简化执行指令

```text
请按本 handoff 执行 GINN v2 深度域精简消融。

只回答三个问题：
1. mismatch training 是否必要；
2. k3 lateral mixer 是否稳定优于 dilated TCN；
3. 短 synthetic physics fine-tuning 是否提供监督之外的增量。

正式训练最少 4 个 run：
A. depth_tcn_nominal_s20260712
B. depth_tcn_mismatch_s20260712
C. depth_lateral_k3_mismatch_s20260712
D. depth_<winner>_mismatch_then_physics_s20260712

所有监督 run 固定相同 benchmark、seed、patch、hidden/depth、optimizer、steps 和 normalization。Mismatch run 必须显式使用 balanced_sample_kind。

先做一次 1 epoch × 10 steps smoke。随后训练 A/B/C，统一对 test split 运行 predict/report，先决定 winner，再运行 D。只有关键差异处于 1%–2% 或指标方向冲突时，才追加 seed 20260713。

不要修改核心代码，不增加架构，不扫描权重，不做 real-field physics 或井训练。最终输出 commands.ps1、四份配置、统一 summary 和 DEPTH_ABLATION_DECISION.md。
```
