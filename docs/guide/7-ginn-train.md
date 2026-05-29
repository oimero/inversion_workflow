# 07 时间域 GINN 训练

`ginn_train.py` 是时间域工作流的第七步。它读取时间域地震、AI 低频模型和第五步选出的全局子波，训练一个在目标层内预测 AI 高频扰动的 GINN。

时间域主线沿用 `src.ginn` 和 `experiments/ginn/train.yaml`。第一版使用 `fixed_gain`，不依赖 dynamic gain、well constraints 或 enhance。

---

## 快速开始

```bash
python scripts/ginn_train.py
python scripts/ginn_train.py --config experiments/ginn/train.yaml
```

训练结果写入 `checkpoint_dir`，默认类似 `scripts/output/ginn_train/checkpoints/`。

---

## 运行前需要什么

| 来源 | 文件 | 用途 |
|------|------|------|
| 地震数据 | 时间域 SEG-Y 或 ZGY | 观测地震和 inline/xline/time 几何 |
| 第五步 | `selected_wavelet.csv` | 物理正演使用的全局子波 |
| 第六步 | `ai_lfm_time.npz` | AI 低频模型、目标层 metadata 和 mask 重建信息 |

训练脚本不重新选择子波，不重建 LFM，也不扫描 residual shift。事实链来自前置步骤。

---

## 推荐配置

```yaml
seismic_file: data/raw/obn-clipped-240-912-872-1544.sgy
segy_iline: 5
segy_xline: 21
segy_istep: 1
segy_xstep: 4

ai_lfm_file: scripts/output/lfm_precomputed_<timestamp>/ai_lfm_time.npz

wavelet_source: precomputed_wavelet
wavelet_file: scripts/output/global_wavelet_generation_<timestamp>/selected_wavelet.csv
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
num_res_blocks: 7
dilations: [1, 2, 4, 8, 16, 32, 64]
kernel_size: 3

batch_size: 16
epochs: 50
lr: 0.001
weight_decay: 0.0001
grad_clip: 1.0

lambda_l2: 0.1
lambda_tv: 0.05
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
checkpoint_dir: scripts/output/ginn_train/checkpoints
log_interval: 50
save_every: 5
```

### LFM 与 mask

`ai_lfm_file` 必须指向第六步的 `ai_lfm_time.npz`。训练端会读取其中的 `metadata_json.horizons` 和 `metadata_json.target_layer`，重新构建 target mask，因此 train config 不再单独配置顶底层位。

LFM 的 `volume` shape 必须与地震体 shape 一致：`(n_inline, n_xline, n_sample)`。`geometry_json.sample_domain` 必须是 `time`，`geometry_json.sample_unit` 必须是秒，`samples` 轴必须和训练地震体的 `sample_min/sample_step/n_sample` 对齐。

### 子波

主线使用第五步输出的 `selected_wavelet.csv`：

```yaml
wavelet_source: precomputed_wavelet
wavelet_file: scripts/output/global_wavelet_generation_<timestamp>/selected_wavelet.csv
```

训练端会校验子波采样间隔和地震时间采样间隔一致。Ricker 子波只适合快速实验，不作为正式主线默认。

### 振幅补偿

V1 使用固定增益：

```yaml
gain_source: fixed_gain
include_dynamic_gain_input: false
in_channels: 3
```

`fixed_gain` 为空时，训练端从目标层内抽样估计一个全局标量，使单位增益合成记录和归一化观测地震处于相近量级。dynamic gain 是后续支线，不作为第七步依赖。

`num_res_blocks` 和 `dilations` 必须一起维护：`dilations` 的长度必须等于 `num_res_blocks`。如果只改其中一个，配置加载会直接失败。

`lambda_tv` 建议先贴近现有时间域配置使用较温和的 `0.05`。后续若发现预测体 ringing 明显，再把它作为参数扫描项提高。

### 井约束

第一版主线不要求 `log_ai_anchor_file`：

```yaml
log_ai_anchor_file: null
lambda_log_ai_anchor: 0.0
well_control_enabled: false
```

后续如果从第六步点级控制或专门的 well constraints 脚本生成 anchor bundle，再打开这部分配置。不要在第七步里临时从 LAS、TDT 或轨迹拼约束。

---

## 脚本在做什么

训练流程由 `src.ginn.data.build_dataset()` 和 `src.ginn.trainer.Trainer` 串起来：

1. 读取时间域地震体和工区几何。
2. 读取 `ai_lfm_time.npz`，校验 LFM shape、采样域和 TWT 轴与地震一致。
3. 从 LFM metadata 读取顶底解释层位，构建训练 mask 和推理 mask。
4. 读取全局子波，自动估计或使用配置的 `boundary_effect_samples`。
5. 展平为 trace dataset，按空间块划分训练/验证集。
6. 训练网络输出 log-AI residual，最终 AI 为 `AI = LFM * exp(residual)`。
7. 写出 checkpoint、metrics 和 run summary。

---

## 输出文件

训练输出在 `checkpoint_dir` 下：

| 文件 | 内容 |
|------|------|
| `best.pt` | 验证集指标最好的 checkpoint |
| `last.pt` | 最后一轮 checkpoint |
| `epoch_*.pt` | 按 `save_every` 保存的中间 checkpoint |
| `metrics.csv` | 每个 epoch 的训练/验证损失和分项指标 |
| `run_summary.json` | 配置、数据统计、mask、normalization 和模型摘要 |

`best.pt` 会保存训练配置，后续第八步反演应优先使用 checkpoint 内的配置，而不是重新手抄一份 train config。

---

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| `AI LFM NPZ metadata must contain at least two sorted horizons` | 第六步 NPZ 没写 `metadata_json.horizons` | 回到第六步检查 `ai_lfm_time.npz` metadata |
| `LFM shape does not match seismic shape` | LFM 和训练地震几何不一致 | 确认第六步和第七步使用同一套地震体与几何参数 |
| `AI LFM geometry_json.sample_domain must be 'time'` | LFM 不是时间域产物 | 检查第六步是否误用了深度域几何或旧 NPZ |
| `AI LFM samples axis does not match the seismic time axis` | LFM 的 TWT 轴和训练地震轴错位 | 确认第六、七步使用同一套地震采样轴，没有重采样或裁剪不一致 |
| `wavelet dt does not match seismic trace dt` | 子波采样间隔不一致 | 检查第五步子波来源和训练地震采样间隔 |
| `in_channels does not match enabled input channels` | 输入通道配置不一致 | V1 使用 `include_lfm_input=true`、`include_mask_input=true`、`include_dynamic_gain_input=false`、`in_channels=3` |

---

## 留到第二轮

- 接入点级 `log_ai_anchor_file`。
- dynamic gain 作为输入通道或增益模型。
- 按井/平台做专用验证集切分。
- enhance stage-2 训练。
