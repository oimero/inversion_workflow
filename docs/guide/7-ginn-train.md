# 07 GINN 训练

`ginn_train.py` 是工作流的第七步。它读取地震体、第六步产出的 AI 低频模型和第五步选出的全局子波，训练一个物理信息神经网络（GINN），在目标层内预测 AI 的高频扰动分量。

网络不直接输出 AI，而是输出一个乘性扰动——最终 AI 由 `LFM × exp(residual)` 合成。物理正演（阻抗→反射率→子波褶积→合成地震记录）全程可微分，损失函数同时在观测地震匹配和扰动正则化两个维度上约束网络。

第一版主线使用固定振幅增益，不依赖 dynamic gain、well constraints 或 enhance。

---

## 快速开始

```bash
python scripts/ginn_train.py
python scripts/ginn_train.py --config experiments/ginn/train.yaml
```

训练结果写入 `checkpoint_dir`。默认配置会校验 LFM 的采样域和 TWT 轴与地震体完全对齐，校验不通过则直接失败。

---

## 运行前需要什么

第七步不重新选择子波，不重建 LFM，不扫描 residual shift。事实链全部来自前置步骤。

| 来源 | 内容 | 用途 |
|------|------|------|
| 地震数据 | 时间域地震体 | 观测地震道和 inline/xline/time 几何 |
| 第五步 | 全局子波 | 物理正演的褶积子波 |
| 第六步 | AI 低频模型 NPZ | LFM 体、目标层 metadata、层位路径 |

第六步的 NPZ 承载了目标层的层位选择和 QC 口径。第七步从 NPZ 的 `metadata_json.horizons` 中重新读取层位文件并重建训练 mask，因此 train config 不再单独配置顶底层位。LFM 的采样域必须是 `time`，单位必须是秒，`samples` 轴必须与地震采样轴完全对齐——三者有一个不满足就会在数据加载阶段失败。

---

## 配置参考

```yaml
seismic_file: null
segy_iline: 189
segy_xline: 193
segy_istep: 1
segy_xstep: 1

ai_lfm_file: null

wavelet_source: precomputed_wavelet
wavelet_file: null
wavelet_type: ricker
wavelet_freq: 25.0
wavelet_dt: 0.001
wavelet_length: 201

gain_source: fixed_gain
fixed_gain: null
fixed_gain_num_traces: 256
dynamic_gain_model: null

include_lfm_input: true
include_mask_input: true
include_dynamic_gain_input: false
in_channels: 3
hidden_channels: 64
out_channels: 1
num_res_blocks: 8
dilations: [1, 2, 4, 8, 16, 32, 64, 128]
kernel_size: 3

batch_size: 16
epochs: 30
lr: 0.001
weight_decay: 0.0001
grad_clip: 1.0

lambda_l2: 0.03
lambda_tv: 0.0
log_ai_anchor_file: null
lambda_log_ai_anchor: 0.0
log_ai_anchor_radius_xy_m: 0.0
well_control_enabled: false
zero_residual_outside_mask: true
boundary_effect_samples: null

validation_split_mode: spatial_block
validation_fraction: 0.10
validation_gap_traces: 8
validation_block_anchor: maxmin
early_stopping_patience: 8
early_stopping_min_delta: 0.0001
early_stopping_warmup: 5

device: cuda
num_workers: 0
pin_memory: true
checkpoint_dir: null
log_interval: 50
save_every: 5
```

### 网络结构

网络架构为 1D 膨胀卷积残差网络。输入通道依次为：归一化地震道、归一化 LFM、目标层 mask。输出为单通道高频扰动。

- `in_channels` 必须等于启用的输入通道数：1（地震） + 是否含 LFM + 是否含 mask + 是否含 dynamic gain。V1 主线使用 `in_channels: 3`。
- `num_res_blocks` 和 `dilations` 必须长度一致。`dilations` 序列决定了纵向感受野：更大的 dilation 让深层能感知更宽的地震上下文，但也增加计算量。
- `hidden_channels` 是残差块内部的特征通道数，`out_channels` 固定为 1。

### 子波

`wavelet_source` 支持两种模式：

- `precomputed_wavelet`：从第五步的 `selected_wavelet.csv` 读取。训练端会校验子波采样间隔与地震采样间隔一致。
- `ricker_wavelet`：按配置的主频、采样间隔和长度生成立即用。仅适合快速实验，不作为正式主线默认。

子波在物理正演中作为不可训练参数使用。子波绝对最大值位置决定褶积的非对称 padding，确保合成记录与观测地震等长。

### 振幅补偿

V1 使用 `fixed_gain`：一个全局标量，让单位增益下的合成记录和 RMS 归一化后的观测地震处于同一量级。

当 `fixed_gain: null` 时，训练端从目标层内抽样若干道，用单位子波对 LFM 做正演，计算合成记录的 RMS，再与同批观测地震归一化后的 RMS 比较，自动估计增益。抽样道数由 `fixed_gain_num_traces` 控制。

`dynamic_gain_model` 是后续支线的随样点变化增益体，主线不启用。

### 损失函数

损失由三项组成，均在归一化地震振幅域内计算：

| 项 | 作用 | 权重 |
|----|------|------|
| Waveform MAE | 合成地震与观测地震在侵蚀掩码内的平均绝对误差 | 固定为 1 |
| Residual L2 | 约束高频扰动幅度，防止 AI 偏离 LFM 太远 | `lambda_l2` |
| Residual TV | 沿时间轴的一阶差分惩罚，抑制层内高频振铃 | `lambda_tv` |

Waveform MAE 只在目标层内部的有效区域（core mask 向内侵蚀掉子波边界影响区后的 loss mask）计算。Residual L2 和 TV 在 core mask 加上向外的 halo 过渡带（taper mask）上计算，halo 宽度由 `boundary_effect_samples` 控制。

`boundary_effect_samples: null` 时，训练端根据子波的有效半支撑（绝对值超过峰值 5% 的时间范围）自动估计。自动估计是推荐的默认行为。

`zero_residual_outside_mask: true` 将目标层外的网络输出用 taper 权重平滑压回 0，避免层位边界处形成硬切——此时层外 AI 保持 LFM。

### 井约束

V1 主线不启用：

```yaml
log_ai_anchor_file: null
lambda_log_ai_anchor: 0.0
well_control_enabled: false
```

后续若从第六步点级控制生成 anchor bundle（一个 NPZ 文件，记录每口控制井在哪些 trace、哪些 sample 上的 log-AI 目标值和权重），才打开这些配置。不要在第七步里临时从 LAS 或时深表拼约束。

### 验证集切分

地震道空间相关性很强，随机道验证会因空间泄漏高估泛化能力。默认使用 `spatial_block`：在工区一角（由 `validation_block_anchor` 指定）划出一块连续区域作为验证集，训练集与验证集之间保留 `validation_gap_traces` 道缓冲带。

早停监控指标为验证集的 waveform MAE。`early_stopping_min_delta` 控制什么算"有效改善"，`early_stopping_patience` 控制连续多少个 epoch 无改善后停止。warmup 阶段不做早停判断。

---

## 脚本在做什么

训练流程由 `src.ginn.data.build_dataset()` 和 `src.ginn.trainer.Trainer` 串联：

### 数据准备

1. 读取地震体，获取 3D 几何（n_inline, n_xline, n_sample）和 inline/xline 网格的 XY 坐标。
2. 读取第六步的 LFM NPZ，校验三项：采样域为 `time`、采样单位为秒、`samples` 轴与地震采样轴对齐。
3. 从 LFM metadata 中读取顶底解释层位路径，用第六步同套目标层 QC 参数重建 TargetZone，生成训练 mask 和推理 mask。
4. 根据子波自动估计边界影响宽度（`boundary_effect_samples`），用它对 mask 侵蚀得到 loss mask、向外扩展得到 residual taper。
5. 将所有 3D 数据展平为 `(n_trace, n_sample)` 的道集。
6. 若配置了空间块验证，按比例划分训练集和验证集。

### 模型与正演

7. 初始化膨胀卷积残差网络。网络接收多通道输入（地震、LFM、mask），输出单通道高频扰动。
8. 初始化物理正演模块：将子波的绝对最大值定位为中心，翻转子波构造卷积核。
9. 网络输出高频扰动后，通过 `AI = LFM × exp(residual)` 合成阻抗。若启用 `zero_residual_outside_mask`，扰动先与 taper 权重相乘，使目标层外扰动平滑归零。

### 正演与损失

10. 物理正演分三步：阻抗→反射率（`(AI[t+1] - AI[t]) / (AI[t+1] + AI[t])`）→反射率与子波做 1D 卷积→合成地震记录。
11. 合成地震与归一化观测地震在 loss mask 内计算 MAE。
12. 高频扰动在 taper mask 内计算 L2 和 TV 正则化项。
13. 三项加权求和为总损失。

### 训练循环

14. Adam 优化器 + 余弦退火学习率调度。
15. 每个 epoch 依次运行训练和验证。
16. 记录逐 epoch 指标到 `metrics.csv`，按 `save_every` 保存中间 checkpoint。
17. 早停监控验证 MAE：连续 `early_stopping_patience` 个 epoch 无显著改善则停止。
18. 训练结束保存 `best.pt`（验证指标最优）和 `final.pt`（最后一轮）。

---

## 核心输出文件

训练输出在 `checkpoint_dir` 下：

| 文件 | 内容 |
|------|------|
| `best.pt` | 验证集指标最优的 checkpoint |
| `final.pt` | 最后一轮 checkpoint |
| `epoch_*.pt` | 按 `save_every` 保存的中间 checkpoint |
| `metrics.csv` | 每个 epoch 的训练/验证损失和分项指标 |
| `run_summary.json` | 完整配置、数据统计、归一化参数、模型摘要 |

### checkpoint 结构

每个 `.pt` 文件保存模型权重、优化器状态、学习率调度器状态和完整训练配置。第八步反演应从 checkpoint 中读取配置来重建数据加载和模型的口径——不应重新手抄一份 train config。

---

## 如何阅读结果

### 第一步：看终端输出

训练过程中每个 epoch 都会输出一行聚合指标。关注：

- `train_loss` 和 `val_loss`（或 `val_mae`）是否在下降。如果训练 loss 持续下降但验证 loss 不降甚至上升，说明过拟合——尝试增大 `lambda_tv` 或减小 `epochs`。
- `train_mae` 和 `val_mae` 的差距。差距过大（超过 2 倍）通常是验证集划分问题或训练集过小。
- `train_res`（高频扰动绝对均值）。如果接近 0，说明模型几乎没学到扰动，可能是学习率太低或 LFM 已经与真实 AI 非常接近。

### 第二步：看 `metrics.csv`

按 epoch 排列，观察趋势：

- `train_waveform_mae` 和 `val_waveform_mae` 应该同步下降并趋于平稳。平稳后的 val_mae 绝对值反映合成记录与观测地震的匹配程度。
- `train_l2_term` 和 `train_tv_term` 乘了各自的 lambda 权重。如果 L2/TV term 明显大于 waveform MAE，说明正则化主导了优化——增大 lambda 要谨慎。
- `best_epoch` 和对应的 `best_loss` 告诉你最优模型来自哪一轮。

### 第三步：看 `run_summary.json`

确认 `data.normalization.seis_rms` 和 `lfm_scale` 的值合理（地震 RMS 不为 0，LFM 缩放因子在 AI 量级范围内）。同时确认 `split_metadata.actual_validation_fraction` 接近预期的 `validation_fraction`——如果实际占比远小于预期，说明验证块落入了无效区，可尝试更换 `validation_block_anchor`。

---

### 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| LFM NPZ 无 `metadata_json.horizons` | 第六步 NPZ 缺少层位信息 | 回到第六步确认 `metadata_json` 写入了 `horizons` 字段 |
| LFM `sample_domain` 不是 `time` 或 `sample_unit` 不是秒 | LFM 不在时间域 | 确认第六步使用时间域地震几何 |
| LFM `samples` 轴与地震采样轴不对齐 | 最大差异超过 1e-6 秒 | 确认第六步和第七步使用同一套地震体和采样轴 |
| LFM shape 与地震 shape 不匹配 | 第六步和第七步的 inline/xline/sample 维度不一致 | 检查地震体路径和 SEG-Y 头字节配置 |
| 子波采样间隔与地震不一致 | 第五步子波来源和训练地震的 dt 不同 | 确认子波导出时使用的是正确的采样间隔 |
| `in_channels` 与启用通道数不匹配 | 配置冲突 | V1 使用 `include_lfm_input=true`、`include_mask_input=true`、`include_dynamic_gain_input=false`、`in_channels=3` |
| `dilations` 长度不等于 `num_res_blocks` | 只改了其中一个 | 两者必须一起维护 |
| Non-finite loss | NaN 或 Inf 出现在损失中 | 检查梯度裁剪是否生效；尝试降低学习率或增大 `lambda_l2` |
| 早停在 warmup 期就触发 | warmup 设置不合理 | `early_stopping_warmup` 应至少比预期收敛 epoch 数少 5-10 |

---

## 留到第二轮

- 接入点级 `log_ai_anchor_file`，启用在训练 batch 中混入井控道的 in-batch well control 机制。
- `gain_source: dynamic_gain_model`，将随样点变化的增益体作为输入通道和正演增益。
- 按井或平台做专用验证集切分，替代单纯的空间块切分。
- enhance stage-2 训练的契约对接。




