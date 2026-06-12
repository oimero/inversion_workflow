# 08 GINN 训练

`ginn_train.py` 是工作流的第八步。它读取地震体、第七步产出的波阻抗低频模型和第五步选出的全局子波，训练一个物理信息神经网络（GINN），在目标层内预测波阻抗的残差分量。

网络不直接重画一套波阻抗体，而是在第七步低频模型的基础上学习一个相对残差。这样做的直觉是：LFM 负责大尺度趋势，神经网络只补目标层内能被地震波形约束的高频细节。

---

## 快速开始

```bash
python scripts/ginn_train.py
python scripts/ginn_train.py --config experiments/ginn/train.yaml
```

训练结果写入 `checkpoint_dir`。默认配置会校验 LFM 的采样域和 TWT 轴与地震体完全对齐，校验不通过则直接失败。

---

## 运行前需要什么

| 来源 | 内容 | 用途 |
|------|------|------|
| 地震数据 | 时间域地震体 | 观测地震道和 inline/xline/time 几何 |
| 第五步 | 全局子波 | 物理正演的褶积子波 |
| 第六步 | `log_ai_anchor_time.npz` | 可选井上 GINN target log-AI anchor 监督 |
| 第七步 | 波阻抗低频模型 NPZ | LFM 体、目标层 metadata、层位路径 |

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
fixed_gain: <recommended_fixed_gain>
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

lambda_l2: 0.05
lambda_tv: 0.05
log_ai_anchor_file: null
lambda_log_ai_anchor: 0.0
log_ai_anchor_radius_xy_m: 0.0
well_control_enabled: true
well_waveform_min_weight: 0.3
well_anchor_batch_fraction: 0.25
well_anchor_distance_decay: gaussian
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

网络按道处理，每次看一条地震道上的时间序列。默认输入是三类信息：地震道、低频波阻抗和目标层 mask；输出是这条道上的残差。

- `in_channels` 要和启用的输入信息数量一致。V1 主线使用地震、LFM、mask 三个通道。
- `num_res_blocks` 和 `dilations` 要成套调整。更大的 dilation 让网络看到更宽的纵向上下文，但也会增加计算量。
- `hidden_channels` 是残差块内部的特征通道数，`out_channels` 固定为 1。

### 子波

`wavelet_source` 支持两种模式：

- `precomputed_wavelet`：从第五步的 `selected_wavelet.csv` 读取。训练端会校验子波采样间隔与地震采样间隔一致。
- `ricker_wavelet`：按配置的主频、采样间隔和长度生成立即用。仅适合快速实验，不作为正式主线默认。

子波在物理正演中作为不可训练参数使用。子波绝对最大值位置决定褶积的非对称 padding，确保合成记录与观测地震等长。

### 振幅补偿

固定增益让正演出的合成记录和观测地震处在可比较的量级。它只处理整体振幅尺度，不解决随空间或时间变化的增益问题。

`gain_source: fixed_gain` 时必须显式填写 `fixed_gain`，推荐使用 dynamic gain 旁路输出的 `recommended_fixed_gain.json`。训练端不再根据 LFM 自动估计 gain，因为 LFM 与实际 GINN target 的响应不同；用 LFM 合成记录作分母会把目标差异误算成振幅增益。

`dynamic_gain_model` 是旁路生成的随样点变化增益体，主线不默认启用。时间域旁路的实现规格见 [dynamic-gain.md](dynamic-gain.md)；它必须使用和本训练端一致的 `seismic_raw / train_mask_rms` 归一化口径，并会同时给出一个人工可选的 recommended fixed gain。

### 损失函数

训练目标的主线是：先让合成记录贴近观测地震，再约束残差不要离 LFM 太远，最后抑制不合理的层内振铃。启用第六步 anchor 后，再额外加入井上低频 log-AI 监督项。

| 项 | 作用 | 权重 |
|----|------|------|
| Waveform MAE | 合成地震与观测地震在侵蚀掩码内的平均绝对误差 | 固定为 1 |
| Residual L2 | 约束残差幅度，防止波阻抗偏离 LFM 太远 | `lambda_l2` |
| Residual TV | 沿时间轴的一阶差分惩罚，抑制层内高频振铃 | `lambda_tv` |
| Log-AI Anchor | 井控样点上的预测 log-AI 与第六步 GINN target anchor 对齐 | `lambda_log_ai_anchor` |

Waveform MAE 只在目标层内部的有效区域（core mask 向内侵蚀掉子波边界影响区后的 loss mask）计算。Residual L2 和 TV 在 core mask 加上向外的 halo 过渡带（taper mask）上计算，halo 宽度由 `boundary_effect_samples` 控制。

`boundary_effect_samples: null` 时，训练端根据子波的有效半支撑（绝对值超过峰值 5% 的时间范围）自动估计。自动估计是推荐的默认行为。

`zero_residual_outside_mask: true` 将目标层外的网络输出用 taper 权重平滑压回 0，避免层位边界处形成硬切——此时层外波阻抗保持 LFM。

### 井约束

第六步已经输出 `log_ai_anchor_time.npz`，它记录每条井控 trace 的实际 GINN target log-AI、有效样点 mask、来源语义和训练权重。目标可以是诊断 cutoff 低通，也可以是第四步 filtered LAS 的直接 TWT 投影。井约束是可选项，关闭时只需要让 anchor loss 权重为 0：

```yaml
log_ai_anchor_file: null
lambda_log_ai_anchor: 0.0
well_control_enabled: true
```

正式启用时，`log_ai_anchor_file` 指向第六步的 `log_ai_anchor_time.npz`，`lambda_log_ai_anchor` 设为正值。`log_ai_anchor_radius_xy_m: 0.0` 表示只约束井所在中心道；大于 0 时会在井附近地震道上按距离衰减扩散井控影响。`well_anchor_batch_fraction` 控制训练 batch 中井影响区样本的目标占比，避免密井监督被普通地震道完全稀释。

井约束只惩罚预测 AI 的 log 值与第六步 GINN target anchor 的差异；它不替代 waveform loss。`well_waveform_min_weight` 保留井中心附近的最低波形损失权重，避免井点附近只顾井曲线而放弃真实地震匹配。

### 验证集切分

地震道空间相关性很强，随机道验证会因空间泄漏高估泛化能力。默认使用 `spatial_block`：在工区一角（由 `validation_block_anchor` 指定）划出一块连续区域作为验证集，训练集与验证集之间保留 `validation_gap_traces` 道缓冲带。

早停监控指标为验证集的 waveform MAE。`early_stopping_min_delta` 控制什么算"有效改善"，`early_stopping_patience` 控制连续多少个 epoch 无改善后停止。warmup 阶段不做早停判断。

---

## 脚本在做什么

训练流程可以分成四段：准备数据、初始化模型、做可微正演、循环优化。

### 数据准备

1. 读取地震体，获取 3D 几何（n_inline, n_xline, n_sample）和 inline/xline 网格的 XY 坐标。
2. 读取第七步的 LFM NPZ，校验三项：采样域为 `time`、采样单位为秒、`samples` 轴与地震采样轴对齐。
3. 从 LFM metadata 中读取顶底解释层位路径，用第七步同套目标层 QC 参数重建 TargetZone，生成训练 mask 和推理 mask。
4. 根据子波自动估计边界影响宽度（`boundary_effect_samples`），用它对 mask 侵蚀得到 loss mask、向外扩展得到 residual taper。
5. 将所有 3D 数据展平为 `(n_trace, n_sample)` 的道集。
6. 若启用井约束，读取第六步 `log_ai_anchor_time.npz`，校验 schema、采样轴、trace 编号和样点数，并构建 in-batch well-control 索引。
7. 若配置了空间块验证，按比例划分训练集和验证集。

### 模型与正演

8. 初始化膨胀卷积残差网络。网络接收多通道输入（地震、LFM、mask），输出单通道残差。
9. 初始化物理正演模块：将子波的绝对最大值定位为中心，翻转子波构造卷积核。
10. 网络输出残差后，通过 `AI = LFM × exp(residual)` 合成阻抗。若启用 `zero_residual_outside_mask`，残差先与 taper 权重相乘，使目标层外残差平滑归零。

### 正演与损失

11. 物理正演分三步：波阻抗转反射率，反射率与子波褶积，得到合成地震记录。
12. 合成地震与归一化观测地震在 loss mask 内计算 MAE。
13. 残差在 taper mask 内计算 L2 和 TV 正则化项。
14. 若启用井约束，在井控样点上计算预测 `log(AI)` 与 anchor `target_log_ai` 的 Smooth L1 损失。
15. 各项加权求和为总损失。

### 训练循环

16. Adam 优化器 + 余弦退火学习率调度。
17. 每个 epoch 依次运行训练和验证。
18. 记录逐 epoch 指标到 `metrics.csv`，按 `save_every` 保存中间 checkpoint。
19. 早停监控验证 MAE：连续 `early_stopping_patience` 个 epoch 无显著改善则停止。
20. 训练结束保存 `best.pt`（验证指标最优）和 `final.pt`（最后一轮）。

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

每个 `.pt` 文件保存模型权重、优化器状态、学习率调度器状态和完整训练配置。第九步反演应从 checkpoint 中读取配置来重建数据加载和模型的口径——不应重新手抄一份 train config。

---

## 如何阅读结果

### 第一步：看终端输出

训练过程中每个 epoch 都会输出一行聚合指标。关注：

- `train_loss` 和 `val_loss`（或 `val_mae`）是否在下降。如果训练 loss 持续下降但验证 loss 不降甚至上升，说明过拟合——尝试增大 `lambda_tv` 或减小 `epochs`。
- `train_mae` 和 `val_mae` 的差距。差距过大（超过 2 倍）通常是验证集划分问题或训练集过小。
- `train_res`（残差绝对均值）。如果接近 0，说明模型几乎没学到残差，可能是学习率太低或 LFM 已经与真实波阻抗非常接近。

### 第二步：看 `metrics.csv`

按 epoch 排列，观察趋势：

- `train_waveform_mae` 和 `val_waveform_mae` 应该同步下降并趋于平稳。平稳后的 val_mae 绝对值反映合成记录与观测地震的匹配程度。
- `train_l2_term` 和 `train_tv_term` 乘了各自的 lambda 权重。如果 L2/TV term 明显大于 waveform MAE，说明正则化主导了优化——增大 lambda 要谨慎。
- `best_epoch` 和对应的 `best_loss` 告诉你最优模型来自哪一轮。

### 第三步：看 `run_summary.json`

确认地震归一化和 LFM 缩放值是否合理：地震 RMS 不能为 0，LFM 缩放应落在正常波阻抗量级附近。再看实际验证集比例是否接近配置预期；如果偏差很大，说明验证块可能落在无效区，需要换一个验证块位置。

---

### 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| LFM NPZ 无 `metadata_json.horizons` | 第七步 NPZ 缺少层位信息 | 回到第七步确认 `metadata_json` 写入了 `horizons` 字段 |
| LFM `sample_domain` 不是 `time` 或 `sample_unit` 不是秒 | LFM 不在时间域 | 确认第七步使用时间域地震几何 |
| LFM `samples` 轴与地震采样轴不对齐 | 最大差异超过 1e-6 秒 | 确认第七步和第八步使用同一套地震体和采样轴 |
| LFM shape 与地震 shape 不匹配 | 第七步和第八步的 inline/xline/sample 维度不一致 | 检查地震体路径和 SEG-Y 头字节配置 |
| 子波采样间隔与地震不一致 | 第五步子波来源和训练地震的 dt 不同 | 确认子波导出时使用的是正确的采样间隔 |
| anchor NPZ 采样轴或样点数不匹配 | 第六步和第八步使用的地震体或采样轴不一致 | 重跑第六步，或确认第六、七、八步使用同一套时间域地震几何 |
| `in_channels` 与启用通道数不匹配 | 配置冲突 | V1 使用 `include_lfm_input=true`、`include_mask_input=true`、`include_dynamic_gain_input=false`、`in_channels=3` |
| `dilations` 长度不等于 `num_res_blocks` | 只改了其中一个 | 两者必须一起维护 |
| Non-finite loss | NaN 或 Inf 出现在损失中 | 检查梯度裁剪是否生效；尝试降低学习率或增大 `lambda_l2` |
| 早停在 warmup 期就触发 | warmup 设置不合理 | `early_stopping_warmup` 应至少比预期收敛 epoch 数少 5-10 |

---

## 留到第二轮

- enhance stage-2 训练的契约对接。
