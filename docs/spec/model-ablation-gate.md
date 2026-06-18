# 模型消融闸门

## 文档地位

本文定义第五步之后第三个研究闸门：在冻结的 `synthoseis-lite` 基准上做模型消融，
判断一维、二维 patch、物理一致性损失和失配训练是否真正提升 2 ms 模型网格上的
`log(AI)` 真值恢复。

本文档不定义“GINN-v2 闸门”。GINN-v2 是首个实现对象，但不是唯一候选架构。第三闸门的
目标是比较模型族，避免在证据不足时把某个网络名字直接固化为正式生产架构。

稳定生产链仍终止于第五步。本研究入口不是“第六步”，也不恢复旧第八步或第九步语义。
旧 `src/ginn/`、旧 GINN 配置、旧 checkpoint 和旧推理契约不得迁移或兼容；需要的功能按
`synthoseis-lite` 新数据契约重写。

未来实现暂定使用不带步骤编号的入口：

- 训练：`scripts/ginn_v2.py train`
- 预测：`scripts/ginn_v2.py predict`
- 报告：`scripts/ginn_v2.py report`
- 核心包：`src/ginn_v2/`

本文只冻结模型消融规范。具体网络结构、训练超参数和正式通过阈值必须由本闸门的实验结果
再决定。

## 1. 闸门回答的问题

本闸门回答以下问题：

1. 一维逐道模型是否已经足够，还是二维横向上下文能稳定改善阻抗恢复。
2. 二维 patch 模型是否能同时改善全频误差、分频误差、薄层、楔状体和尖灭指标。
3. 物理一致性损失是否提供真实泛化收益，还是重新诱发“只拟合地震、不恢复阻抗”的伪解。
4. 子波、gain、噪声、相位和时移失配训练是否提升鲁棒性，还是损害可恢复细节。
5. 模型在 `0x` 和 unsupported probe 上是否产生虚假高频恢复。

模型选择不得只依据 training loss、waveform loss 或单一相关系数。首轮不设总分，但报告
顺序和指标语义必须固定，便于后续版本建立相对门槛。

## 2. 数据与监督契约

### 2.1 主监督目标

首版监督目标固定为 `synthetic_benchmark.h5` 中的：

```text
truth/model_target_log_ai
```

这是 2 ms 模型网格上的抗混叠阻抗目标。首版不得使用 high-resolution truth 作为模型监督
目标，不得把任务隐式改成超分辨率反演。

训练和评分 mask 固定为：

```text
truth/valid_mask_model
```

`valid_mask_model` 是唯一阻抗 loss 和阻抗指标 mask。`forward_valid_mask_*` 只表示正演上下文
支持，不能把上下文延拓区纳入训练目标或阻抗评分。

### 2.2 样本类型语义

训练、预测和报告必须保持 `SynthoseisBenchmark` 的样本语义：

| `sample_kind` | 地震输入 | 阻抗目标 |
| --- | --- | --- |
| `base` | `seismic/seismic_model_consistent` | `truth/model_target_log_ai` |
| `frequency_probe` | probe 后的 model-consistent seismic | base `model_target_log_ai + probe_log_ai_model_grid` |
| `seismic_variant` | gain/noise/wavelet/phase/shift variant seismic | source base 的 `model_target_log_ai` |
| `frequency_probe_seismic_variant` | probe + mismatch seismic | source base target + probe increment |

`seismic_from_highres_truth_model_grid` 只用于 forward QC 或专门消融，不作为首轮主输入。

### 2.3 首轮输入通道

首轮模型输入通道固定为：

- `seismic`
- `lfm_controlled_degraded`
- `valid_mask_model`

`valid_mask_model` 可作为模型输入通道，也必须作为 loss 和评估 mask。`rgt_model`、
`zone_id_model`、`boundary_mask_model`、gain/noise/wavelet scenario metadata 等地质或场景通道
暂不进入首轮主模型，只能作为后续消融项。

### 2.4 归一化与输出参数化

所有输入、目标和输出归一化参数只能由 train split 统计拟合，并冻结到 model run manifest。
Validation、test 和 benchmark split 不得重新估计归一化参数。至少记录：

- `normalization_scope=train_only`
- `normalization_mask=valid_mask_model`
- seismic mean/std
- `lfm_controlled_degraded` mean/std
- target `log(AI)` mean/std

首轮模型推荐内部预测低频先验残差：

```text
pred_delta_log_ai = model(seismic, lfm_controlled_degraded, valid_mask_model)
pred_log_ai = lfm_controlled_degraded + pred_delta_log_ai
```

对外预测契约固定为 `pred_log_ai`。Evaluator 只消费反归一化后的 `pred_log_ai`，不消费
normalized prediction，也不依赖模型内部是否预测 delta。这样 `lfm_controlled_degraded` 自然
对应零残差输出基线，但报告仍统一比较绝对 `log(AI)` 预测。

## 3. Patch 数据集

### 3.1 主训练形态

首轮主路径是二维 patch 训练，不使用整剖面训练作为主路线。整剖面可以用于可视化、报告和
后续拼接预测验证。

Patch sampler 从冻结的 `synthoseis-lite` realization 裁剪样本，输出独立 patch index。每行至少
记录：

- patch id、source sample id、sample kind、parent realization id。
- split、HDF5 group、lateral 范围、TWT 范围。
- patch shape、axis order、有效样点数。

Probe patch 还必须记录：

- `paired_zero_sample_id`
- `probe_group_id`
- `probe_frequency_hz`
- `probe_phase`
- `probe_amplitude_multiplier`
- `probe_lateral_shape`

这些字段用于保证 probe 配对差分评估可复现。

### 3.2 Split 规则

Split 必须按 parent realization 隔离：

- 同一 parent realization 的 base、probe、seismic variants 和 frequency-probe seismic variants
  必须位于同一个 split。
- 同一 probe group 的 `0x` 与非零 probe 必须位于同一个 split。
- 相邻 patch、重叠 patch、LFM 变体、噪声/gain/wavelet 变体不得跨 split。

任何 split 泄漏都使训练数据集无效。当前 `synthoseis-lite` 生成端若仍标记
`unassigned`，正式训练前必须先生成或派生满足上述规则的训练 index。

### 3.3 一维对照

`trace_1d` 使用与二维 patch 相同的 TWT 范围和输入通道，只移除横向上下文。实现可取中心道
或显式抽样道，但必须记录所选 trace 的 lateral 坐标。它是“无横向上下文”的公平对照，不是
另一套窗口定义。

## 4. 模型矩阵

首批消融按以下顺序报告：

1. `oracle_target`：管线自检，误差应接近零，不参与模型排名。
2. `lfm_ideal`：理想低频先验参考，不代表真实输入条件。
3. `lfm_controlled_degraded`：真实输入先验基线，也即 `lfm_only`。
4. `trace_1d`：逐道监督模型。
5. `trace_1d_dilated_tcn`：逐道膨胀时间卷积模型，检验更长时间感受野是否提升恢复。
6. `trace_1d_mismatch_training`：在逐道模型上加入 seismic variants，检验鲁棒性收益是否不依赖二维上下文。
7. `trace_1d_dilated_tcn_mismatch_training`：逐道膨胀时间卷积加 seismic variants 的交叉消融。
8. `patch_2d_supervised`：GINN-v2 首个二维 patch 闭环。
9. `patch_2d_with_physics_loss`：在二维监督模型上加入物理一致性损失。
10. `patch_2d_mismatch_training`：加入 seismic variants 的鲁棒训练消融。

模型矩阵使用中性名称。GINN-v2 可以实现 `patch_2d_supervised`，但报告中必须首先按消融项
解释结果，而不是按网络名字预设胜负。

首轮训练样本策略固定为：

| 模型 | 训练样本 | 评估样本 |
| --- | --- | --- |
| `trace_1d` | `base` | 全部 sample kind |
| `trace_1d_dilated_tcn` | `base` | 全部 sample kind |
| `trace_1d_mismatch_training` | `base` + `seismic_variant` | 全部 sample kind |
| `trace_1d_dilated_tcn_mismatch_training` | `base` + `seismic_variant` | 全部 sample kind |
| `patch_2d_supervised` | `base` | 全部 sample kind |
| `patch_2d_with_physics_loss` | `base` | 全部 sample kind |
| `patch_2d_mismatch_training` | `base` + `seismic_variant` | 全部 sample kind |

`frequency_probe` 和 `frequency_probe_seismic_variant` 首轮主要作为评测样本，不默认进入前三个
训练配置。若后续训练 probe-aware 模型，必须单独命名并与首轮模型分开比较。训练 sampler
必须在 model run manifest 中记录 `sample_kind` sampling weights。

所有模型报告必须记录容量与感受野信息，至少包括 `parameter_count`、
`trainable_parameter_count`、lateral/TWT receptive field、input patch size 和 output patch size。
这些字段用于区分“二维上下文收益”和“单纯参数量收益”。

## 5. Loss 规则

Synthetic supervised 阶段以 AI truth loss 为主损失：

```text
L_ai = loss(pred_log_ai, model_target_log_ai, valid_mask_model)
```

Physics loss 只能作为辅助正则和消融项，不得替代 AI truth supervision，也不得用 waveform
loss 选择首批模型。

`base` 与 `frequency_probe` 可在 model-consistent seismic 上使用 nominal forward physics loss。
`seismic_variant` 与 `frequency_probe_seismic_variant` 默认关闭 nominal physics loss；若要启用，
必须使用对应 variant 的已知扰动算子，作为单独消融记录。不得让预测阻抗吸收未知 gain、
噪声、相位或子波失配来降低 waveform loss。

Physics loss 权重、适用 sample kind 和 forward operator 必须写入 model run manifest。至少记录：

- `lambda_ai`
- `lambda_physics`
- `physics_loss_applied_sample_kinds`
- `physics_forward_operator_id`
- `wavelet_scenario_id`
- `gain_scenario_id`（如适用）

Synthetic supervised 模型不得只用 physics loss 训练。

## 6. 预测与报告

首版优先支持 patch-level prediction 和 patch-level metrics。预测输出必须能被统一 evaluator
消费，并校验 sample id、patch id、axis order、shape 和 mask。

预测产物至少记录：

- `prediction_id`
- `model_run_id`
- benchmark manifest SHA-256
- sample index SHA-256
- patch id 与 sample id
- `pred_log_ai`
- `valid_mask_model`
- axis order
- normalization manifest

不得只输出 normalized prediction，也不得只输出 delta prediction。

若生成 realization-level prediction，重叠 patch 只能使用以下策略之一：

- uniform averaging。
- 中心有效区裁剪后拼接。

实现必须记录 blend mask 或裁剪 mask。不同模型不得使用不可追踪的拼接策略。

模型选择基于 validation split 的 `synthoseis_lite_report_v1` 指标。报告顺序固定覆盖：

- base fullband NRMSE、RMSE、bias 和 correlation。
- 分频 `log(AI)` 幅度与相位误差。
- probe paired increment error。
- wedge、pinchout、thin-bed 几何指标。
- mismatch degradation ratio。
- `0x` 和 unsupported probe 的 false high-frequency energy。

分频指标的 band 定义必须来自 benchmark manifest 或统一 report 配置，不得由训练脚本按模型
自行定义。

首版不设单一总分。正式架构选择必须等一维、二维、physics loss 和 mismatch training 消融
结果齐全后再定。

## 7. 实现前置检查

未来实现必须覆盖以下检查：

- Patch sampler 不跨 parent realization、probe group 和 paired `0x` split。
- 四类 `sample_kind` 都能被训练、预测和评估消费。
- Probe target 使用 base target 加 probe increment。
- `valid_mask_model` 是唯一阻抗 loss 和 evaluation mask。
- `trace_1d` 与二维 patch 只差横向上下文。
- Physics loss 不在 mismatch variants 上错误使用 nominal forward。
- Prediction 输出可被统一 evaluator 消费。
- `oracle_target` 接近零误差，作为 evaluator 自检。
- `overfit_tiny` smoke：只使用 `base` sample，1 个 realization / 少量 patch 能拟合到很低误差；
  验证 train loss 可下降、`pred_log_ai` 与 target shape/mask 对齐、evaluator 能读预测。

这些检查属于模型消融入口的最低工程门槛。通过 smoke 只说明训练、预测和报告链路连通，
不能作为架构有效性的证据。

训练结果异常时按以下顺序诊断：

1. `oracle_target` 是否接近零误差；否则优先检查 evaluator、axis 和 mask。
2. `overfit_tiny` 是否通过；否则优先检查 sampler、model output、loss 和归一化。
3. `lfm_controlled_degraded` 基线是否正常；否则优先检查 benchmark target 与 mask。
4. `patch_2d_supervised` 是否优于 `lfm_controlled_degraded`；否则模型没有学到可恢复细节。
5. Physics loss 是否改善 probe、geometry 或 mismatch 指标，而不是只改善 waveform loss。

## 8. 约束与默认选择

- 不使用 “GINN v0” 名称；首个实现称为 “GINN-v2 首个二维 patch 闭环” 或
  `patch_2d_supervised`。
- 首轮不引入 RGT、zone、boundary 等额外地质通道。
- 旧 GINN 代码不迁移、不兼容。
- 所有训练、预测、报告契约按 `synthoseis-lite` 新数据接口重写。
- 第三闸门的结论只来自冻结 benchmark 和统一报告卡，不来自单次井旁相关性或旧 workflow
  经验。
