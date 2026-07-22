# Enhance v2 深度域微纹理增强 Handoff

> **状态：暂时冻结路线。**

## 1. 目标与定位

Enhance v2 是位于 ablation R0 之后的独立微纹理补全阶段。ablation 负责恢复由当前合成先验和地震有效频带支持的中频规范阻抗增量；Enhance v2 在固定的 R0 结果上补充由井统计约束的薄层纹理。

定义：

\[
m_{\mathrm{base}}
=
L_{\mathrm{external}}+\widehat u_{\mathrm{canonical}}
\]

\[
m_{\mathrm{enhanced}}
=
m_{\mathrm{base}}+\widehat h_{\mathrm{microtexture}}
\]

其中：

- `base_log_ai`：纯合成监督 ablation 在真实工区的 `predicted_log_ai`；
- `predicted_microtexture_log_ai`：Enhance v2 输出的井统计条件微纹理；
- `enhanced_log_ai`：二者之和；
- `enhanced_ai`：`exp(enhanced_log_ai)`。

`predicted_microtexture_log_ai` 是条件先验补全，不是 canonical increment，也不是地震唯一确定的地下真值。公共字段、配置和报告不使用无对象限定的 `residual` 或 `delta`。

首轮只回答两个问题：

1. 受控薄层训练目标是否能让网络产生高于 base 的有效分辨率；
2. 分频与 RMS loss 相比普通 MSE 是否提供额外收益。

## 2. 首版边界

首版固定：

- 仅支持深度域 TVDSS，纵向单位为米；
- 使用单道 1D 网络；
- base 来自纯合成监督 ablation 的真实工区 R0；
- 真实井只用于全目标层纹理统计标定；
- 训练不包含真实井监督和 physics 反向传播；
- 训练数据由 Enhance v2 在线生成；
- Synthoseis-lite 和 ablation 不生成或学习同一套薄层纹理；
- A/B/C 先跑覆盖全部井的剖面，剖面审查后才允许完整体；
- B/C 首轮各跑一个 seed，C 通过机制审查后再跑三个 seed。

首版不包含：

- 时间域；
- 横向网络或横向 loss；
- 井纹理 patch 复制；
- 真实井训练 anchor；
- latent ensemble、扩散模型或不确定性输出；
- 自动进入完整体的效果阈值；
- 逐文件或递归 SHA-256 provenance。

## 3. 代码边界

目标结构：

```text
src/enhance_v2/
├── __init__.py
├── contracts.py        # schema、公共字段和轴合同
├── calibration.py      # 全目标层井纹理统计
├── data.py             # R0 substrate、空间 split 和训练窗口
├── generation.py       # none/thin-bed 在线生成
├── model.py            # 单一 1D dilated TCN
├── losses.py           # MSE 与 detail/RMS loss
├── training.py         # B/C 训练和固定验证
├── prediction.py       # 剖面与体推理
└── reporting.py        # 合成、井、频谱和侧向 QC

scripts/enhance_v2.py
experiments/enhance_v2/train.yaml
experiments/enhance_v2/run_enhance_v2.ps1
```

边界规则：

- `enhance_v2` 可依赖 `cup.physics`、R0 物化产物和井控 reader；
- `enhance_v2` 不依赖 `cup.synthetic`、`ginn_depth` 或旧 `enhance`；
- ablation、R0 和 R1 不导入 `enhance_v2`；
- A 是物化的 base 对照，不创建伪 checkpoint；
- B/C 共享 generator 和验证 catalog，只替换 loss；
- v2 最小垂直切片通过后删除 `src/enhance`，历史实现由 Git 保存。

## 4. 输入合同

### 4.1 R0 substrate

输入目录必须是由纯合成监督 checkpoint 生成的深度域 R0。Enhance v2 读取模型子目录中的 `predictions.npz`，要求：

```text
predicted_log_ai       float32 [inline, xline, sample]
seismic_input          float32 [inline, xline, sample]
valid_mask             bool    [inline, xline, sample]
ilines                 float64 [inline]
xlines                 float64 [xline]
samples                float64 [sample]
sample_domain          depth
sample_unit            m
depth_basis            tvdss
```

同时读取：

- R0 summary；
- ablation model manifest；
- `forward_model_inputs.json`；
- 该文件声明的时间子波和 AI–Vp 关系。

输入校验：

- checkpoint 的部署阶段只包含 `synthetic_supervised`；
- `predicted_log_ai` 是 deployment closure；
- 三个体数组 shape 完全相同；
- `valid_mask=True` 内 base 和 seismic 均有限；
- TVDSS 轴严格递增且与 R0 sample-axis contract 一致；
- xline 使用物化坐标，步长 4 保持为真实坐标差。

### 4.2 井控制

井统计读取 `real_field_well_controls_v3` 中状态为 `ok` 的垂直井 NPZ。首版使用全部可用井和完整目标层有效样点，不划分训练井和留出井。

井只决定纹理生成分布。所有井上的最终 AI 指标属于商业校准拟合度，不表述为井间泛化验证。

### 4.3 可读 provenance

manifest 只记录以下可读路径和 schema：

- R0 summary；
- R0 predictions；
- ablation model manifest；
- well-control summary；
- forward-model inputs；
- 配置快照。

consumer 校验路径指向的 schema、采样域、单位和轴，不递归计算文件摘要。

## 5. 井纹理统计标定

标定产物版本为 `enhance_v2_calibration_v1`。

处理顺序：

1. 读取每口井在最终 5 m TVDSS 规则轴上的 AI 与有效 mask；
2. 转换为 natural-log AI；
3. 在每个连续有效段计算相邻样点差分；
4. 用每口井差分的 MAD 估计噪声尺度；
5. 将绝对差分不小于 `3 × 1.4826 × MAD` 的位置识别为候选界面；
6. 将相邻且符号相反的候选界面配成一个薄层；
7. 只保留厚度在 10–60 m 的界面对；
8. 记录厚度和两侧 logAI 跳变绝对值的均值；
9. 汇总所有井，不按井、zone 或 state 分组；
10. 对比度采样范围使用汇总分布的 p05–p95，厚度从保留的经验分布直接采样。

若没有得到任何合格界面对，标定直接失败，不使用手工默认分布。

产物至少记录：

```text
schema_version
sample_domain / sample_unit / depth_basis
sample_interval_m
source_well_names
n_candidate_interfaces
n_paired_beds
bed_thickness_m samples and quantiles
contrast_log_ai samples and quantiles
log_ai p01 / p50 / p99
```

`log_ai` 分位数用于生成结果 QC，不在训练时裁剪网络输出。

## 6. 在线生成合同

### 6.1 窗口与 split

训练窗口固定为 128 个深度样点，步长 64；窗口只沿单道纵向采样。每个窗口在形成最终 mask 后至少包含 64 个有效样点。

R0 父道按实际 inline/xline 坐标建立互斥空间块：

- train：80%；
- validation：10%；
- test：10%。

同一父道只属于一个 split。validation/test catalog 固定物化，训练样本由确定性计数器在线生成。

随机种子由以下元组唯一决定：

```text
(generator_seed, split, epoch, sample_ordinal, parent_trace_id)
```

B/C 使用相同 `generator_seed`、父道顺序和 sample ordinal，因此目标与输入逐样本一致。

### 6.2 `none`

训练和固定验证中 25% 样本使用 `none`：

```text
target_microtexture_log_ai == 0
target_log_ai == base_log_ai
```

该模式用于教网络在缺少纹理证据时保持 base，并直接测量 false texture。

### 6.3 `thin_bed_cluster`

其余 75% 样本使用 `thin_bed_cluster`：

- 每个窗口生成一个簇；
- 每簇包含 2–5 个薄层；
- 厚度从 calibration 的经验厚度样本中有放回抽样；
- 对比度从 calibration 的 p05–p95 经验样本中有放回抽样；
- 相邻薄层符号严格交替；
- 整个簇随机放置在窗口有效连续段内；
- 簇内微纹理按物理厚度加权去均值；
- 簇外微纹理严格为零。

生成在 6 倍高分辨率深度轴完成。组合真值为：

\[
m_{\mathrm{target,hi}}
=
m_{\mathrm{base,hi}}+h_{\mathrm{microtexture,hi}}
\]

随后执行抗混叠并降采样到 5 m 模型轴。物化 target 重新按降采样后的数组计算：

```text
target_microtexture_log_ai = target_log_ai - base_log_ai
```

### 6.4 正演与输入脏化

目标阻抗通过 `DepthForwardExecutor` 正演：

- backend：`auto | numpy | torch_cuda`；
- dtype：固定 float64；
- `auto` 在 CUDA 可用时选择 Torch CUDA；
- wavelet 和 AI–Vp 关系来自 R0 的 `forward_model_inputs`。

正演结果只对网络输入施加确定性 mismatch：

- 常相位：均匀分布 `[-12°, 12°]`；
- 振幅倍率：`[0.9, 1.1]`；
- 高斯噪声 RMS：信号 RMS 的 `[0.02, 0.06]`；
- 频谱倾斜最大绝对值：0.12。

每个窗口的 synthetic/real seismic 均在有效点内独立去均值并除以 centered RMS。base channel 使用 train 分区 `base_log_ai` 的冻结均值和标准差。

网络输入固定为：

```text
channel 0: standardized_seismic
channel 1: normalized_base_log_ai
```

invalid/padding 位置在标准化后填零，mask 不作为第三输入通道。

## 7. 模型合同

首版只有一个架构 ID：`trace_dilated_tcn`。

固定结构：

- `in_channels=2`；
- `hidden_channels=32`；
- `depth=5`；
- kernel size 5；
- dilation 依次为 1、2、4、8；
- GELU 激活；
- 输出 1 个 `predicted_microtexture_log_ai` 通道；
- 最后一层权重和 bias 零初始化；
- 不使用 BatchNorm；
- 推理无随机算子。

checkpoint 版本为 `enhance_v2_checkpoint_v1`，必须保存：

- architecture 和参数；
- normalization；
- calibration 路径；
- R0 substrate 路径；
- generator 配置与 seed；
- loss 配置；
- validation catalog；
- best epoch、selection score 和训练历史。

## 8. A/B/C 消融

### A：Base

A 不训练网络：

```text
predicted_microtexture_log_ai = 0
enhanced_log_ai = base_log_ai
```

### B：普通 MSE

```text
loss_B = masked_mse(
    predicted_microtexture_log_ai,
    target_microtexture_log_ai
)
```

### C：分频/RMS loss

定义移动平均算子 `M_w`。窗口长度以米配置，再按 5 m 采样间隔转换为奇数样点。它仅用于增强 loss，不属于 canonical decomposition。

```text
local_mean_w85(h) = M_85m(h)
detail_w35(h)     = h - M_35m(h)
```

```text
loss_C =
    0.05 * smooth_l1(local_mean_w85(pred), local_mean_w85(target))
  + 1.50 * smooth_l1(detail_w35(pred), detail_w35(target))
  + 0.25 * smooth_l1(rms(pred), rms(target))
  + 0.50 * smooth_l1(relu(0.9 * rms(target) - rms(pred)), 0)
```

所有项只在最终有效 mask 内计算。

B/C 使用共同的 checkpoint selection metric：

```text
selection_score
  = thin_bed_microtexture_mse
  + none_false_texture_mse
```

两个分量分别按各自样本数聚合后等权相加，不按训练集 75:25 比例加权。真实井和人工剖面观感不参与选模。

## 9. 推理与覆盖

推理读取同一纯监督 R0 的 base、seismic 和 valid mask。

- 使用 128 样点窗口和 64 样点步长；
- 窗口只要包含至少一个有效点就参与推理；
- 仅在有效点累计；
- stitching 固定 uniform；
- invalid 点保持 NaN；
- valid 点必须同时获得 base、microtexture 和 enhanced 预测；
- 任一有效点没有 stitching support 时整次运行失败并报告坐标。

物化字段：

```text
base_log_ai
predicted_microtexture_log_ai
enhanced_log_ai
enhanced_ai
valid_mask
prediction_support_count
ilines / xlines / samples
```

必须逐点满足：

```text
enhanced_log_ai
  == base_log_ai + predicted_microtexture_log_ai
```

输出版本：

- `enhance_v2_prediction_v1`：单模型预测；
- `enhance_v2_summary_v1`：A/B/C 或多 seed 汇总。

完整体额外导出 `enhanced_ai` SEG-Y；inline/xline header 和 xline 步长沿用输入工区。

## 10. 报告合同

### 10.1 固定合成验证

分别报告 none 和 thin-bed：

- microtexture MSE、MAE、相关性；
- predicted/target microtexture RMS ratio；
- 35 m detail RMS ratio；
- none false-texture RMS；
- 界面 precision、recall 和 F1；
- 按目标薄层厚度分箱的恢复指标；
- base、target、enhanced 的频谱；
- selection score。

界面检测对 target 和 prediction 使用同一梯度阈值与同一容许深度误差，不为 prediction 单独调参。

### 10.2 真实剖面

在覆盖全部井的批量剖面上同时展示 A/B/C：

- seismic、base、microtexture、enhanced 四联图；
- 每口井 base/enhanced AI 相关性和 RMSE；
- 全部井聚合指标；
- 井旁曲线叠合；
- base/enhanced 频谱；
- 微纹理 RMS 和 p95；
- 相邻实际道之间的微纹理差异；
- R1 normalized waveform 指标变化。

一维模型没有横向连续性约束。侧向不连续性必须作为显式 QC 报告，不能通过后处理平滑隐藏。

### 10.3 多 seed

C 的首个 seed 通过机制审查后，使用相同数据合同再运行两个 seed，形成三个 seed：

- 每个 seed 独立训练；
- validation catalog 完全相同；
- 按 selection score 选择单一部署 seed；
- 不按井指标或人工观感选择 seed；
- 报告三个 seed 的微纹理 RMS、相关性、剖面差异和逐点标准差；
- 首版不输出 ensemble 平均体。

## 11. 实施清单

### 阶段 0：冻结中频 base

- [ ] 建立只包含 `synthetic_supervised` 的专用 ablation 实验；
- [ ] 完成覆盖全部井的 section R0；
- [ ] 完成 volume R0；
- [ ] 确认 R0 manifest、checkpoint 和 closure 均为纯监督；
- [ ] 将 R0 路径写入 Enhance v2 配置。

### 阶段 1：合同、calibration 与 reader

- [ ] 建立 `enhance_v2` schema 和公共字段；
- [ ] 实现 R0 substrate reader；
- [ ] 实现井纹理统计 calibration；
- [ ] 写出 `enhance_v2_calibration_v1`；
- [ ] 验证 5 m TVDSS 和 xline 步长 4；
- [ ] 建立 train/validation/test 父道 split。

### 阶段 2：确定性生成器

- [ ] 实现 25% none；
- [ ] 实现 75% thin-bed cluster；
- [ ] 实现 6 倍高分辨率生成和抗混叠降采样；
- [ ] 接入共享 `DepthForwardExecutor`；
- [ ] 实现输入 mismatch 和窗口标准化；
- [ ] 物化固定 validation/test catalog；
- [ ] 验证 B/C 逐样本完全一致。

### 阶段 3：最小训练垂直切片

- [ ] 实现固定 1D dilated TCN；
- [ ] 验证零初始化返回 base；
- [ ] 实现 B 的 masked MSE；
- [ ] 实现 C 的 local-mean/detail/RMS loss；
- [ ] 实现共同 selection score；
- [ ] 写出 checkpoint、history 和合成报告；
- [ ] 用小预算 smoke 跑通 B/C。

### 阶段 4：单 seed A/B/C

- [ ] 固定完整训练预算和 generator seed；
- [ ] 训练 B；
- [ ] 训练 C；
- [ ] 生成 A/B/C 合成验证报告；
- [ ] 生成覆盖全部井的批量剖面；
- [ ] 完成井指标、频谱、false texture、R1 和侧向 QC；
- [ ] 记录是否进入多 seed 阶段的人工决定。

### 阶段 5：C 多 seed

- [ ] 补跑 C 的另外两个 seed；
- [ ] 生成统一多 seed 报告；
- [ ] 按固定合成 selection score 选择部署 seed；
- [ ] 在同一批剖面上比较三个 seed；
- [ ] 记录是否进入完整体的人工决定。

### 阶段 6：完整体与收口

- [ ] 对选定 C seed 运行完整体；
- [ ] 导出 microtexture NPZ 和 enhanced AI SEG-Y；
- [ ] 完成 valid 点全覆盖检查；
- [ ] 完成完整体频谱和侧向 QC；
- [ ] 删除旧 `src/enhance`；
- [ ] 全仓清理旧 enhance public 字段、配置和 import；
- [ ] 将最终命令、测试结果和产物路径写回本 Handoff。

## 12. 活动配置草案

```yaml
enhance_v2:
  schema_version: enhance_v2_experiment_v1
  experiment_id: depth_microtexture_detail_loss_s202607XX
  seed: 202607XX
  device: auto

  sources:
    r0_run_dir: scripts/output/real_field_zero_shot_<pure_supervised_run>
    experiment_id: <pure_supervised_experiment_id>
    well_control_run_dir: scripts/output/real_field_well_controls_<run>

  calibration:
    interface_threshold_mad: 3.0
    bed_thickness_m: [10.0, 60.0]
    contrast_quantiles: [0.05, 0.95]

  windows:
    vertical_samples: 128
    vertical_stride: 64
    min_valid_samples: 64

  generator:
    seed: 202607XX
    none_fraction: 0.25
    highres_factor: 6
    cluster_bed_count: [2, 5]
    phase_deg: [-12.0, 12.0]
    amplitude_multiplier: [0.9, 1.1]
    noise_rms_fraction: [0.02, 0.06]
    spectral_tilt_max: 0.12

  architecture:
    id: trace_dilated_tcn
    hidden_channels: 32
    depth: 5

  loss:
    kind: detail_rms  # masked_mse | detail_rms
    local_mean_window_m: 85.0
    detail_window_m: 35.0
    local_mean_weight: 0.05
    detail_weight: 1.5
    rms_weight: 0.25
    rms_underfit_weight: 0.5
    rms_floor: 0.9

  training:
    epochs: 6
    samples_per_epoch: 2048
    batch_size: 16
    learning_rate: 0.0003
    weight_decay: 0.00001

  prediction:
    mode: section
    sections_file: experiments/common/real_field_sections.yaml
    stitch_strategy: uniform
```

B/C 只允许修改：

```text
experiment_id
loss.kind
```

首轮 B/C 的训练 seed、generator seed、calibration、R0 source、架构和训练预算必须相同。C 的多 seed 阶段只修改顶层训练 `seed`，固定 generator seed 和 validation catalog。

## 13. 测试与硬验收

测试继续放在被 `.gitignore` 忽略的 `tests/`。

必须覆盖：

- R0 不是纯监督 deployment 时明确失败；
- sample domain、unit、depth basis 或轴不匹配时明确失败；
- 固定 seed 的 calibration sampling 和生成完全确定；
- none target 严格为零；
- thin-bed 数量、厚度、对比度和交替符号符合 calibration；
- 高分辨率生成和降采样 shape 正确；
- B/C 相同 sample ordinal 产生逐值相同输入与 target；
- seismic 逐窗口 centered RMS 标准化有限；
- base normalization 只使用 train 分区；
- zero-init 网络输出严格为零；
- B/C loss 在全零 target、稀疏 mask 和普通 thin-bed target 上有限；
- selection score 对 none/thin-bed 等权；
- CPU/CUDA depth forward 数值一致；
- `enhanced_log_ai = base + microtexture` 可独立复算；
- invalid 点保持 NaN、valid 点无 NaN；
- section 和 volume 所有有效点都有 stitching support；
- xline 步长 4 的坐标、split 和 SEG-Y header 保持正确；
- 正式入口不导入旧 `enhance` 或 `ginn_depth`。

效果指标不作为代码正确性的硬门禁。A/B/C、井指标和多 seed 结果必须如实报告，允许实验结论是分频 loss 无收益或微纹理不适合进入完整体。

## 14. 解释边界

对外表述固定区分：

- base：地震有效频带和合成地质先验支持的中频反演；
- microtexture：井统计条件下的一种高频补全；
- enhanced AI：base 与该补全的组合结果；
- 所有井相关性：使用同一批井统计后的商业校准拟合度；
- 多 seed 差异：带限地震到宽带薄层一对多性的内部风险指标。

禁止将 Enhance v2 的新增薄层统一描述为“地震恢复出的真实薄层”。
