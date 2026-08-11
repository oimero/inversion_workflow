# 带限证据网络规格

## 目标

`evidence` 模块从地震、完整低频模型和物理坐标中提取模型网格上的带限证据：

```text
seismic + full LFM + validity + lateral geometry
→ projected log-AI increment mean/scale
→ signed reflectivity mean/scale
→ three-state log potential
```

这些输出描述当前地震频带能够支持的阻抗变化，不承担高分辨率地质体生成、微边界定位或唯一反射系数恢复的语义。

## 公开 interface

包根暴露：

- `EvidenceInput`：带采样轴、有效性和米制横向几何的输入；
- `EvidenceModel.predict()`：生成一份 `BandlimitedEvidence`；
- `train_evidence_model()`：训练三个证据 head；
- `evaluate_evidence_model()`：评价证据、基线和地震干预；
- `EvidenceNetworkConfig` 与 `EvidenceLearningConfig`。

`BandlimitedEvidence` 包含：

- zone-linear LFM anchor；
- projected log-AI increment mean/scale；
- signed reflectivity mean/scale；
- 三状态归一化 log potential；
- local tuning scale；
- sample support；
- `lateral_m` 和可选 XY 坐标。

## 输入合同

垂向语义来自 `SampleAxis`：

```text
depth: sample_domain=depth, unit=m, depth_basis=tvdss
time:  sample_domain=time,  unit=s, depth_basis=null
```

深度域输入携带模型网格 `Vp`，用于计算 `Vp / (4 f_dom)` 调谐尺度；时间域调谐尺度为 `1 / (2 f_dom)`。

`inline/xline` 只承担寻址。横向混合使用 `lateral_m`，因此工区 `xline_step=4` 不会被解释成物理单位距离。无效 patch 边缘通过 `lateral_valid` 和 `observed_valid` 显式屏蔽。

每道已知 zone 内将完整 LFM 拟合为线性 anchor：

```text
background_lfm_linear = a + b * (2ζ - 1)
```

网络输入是缩放后的 seismic、`full_lfm - background_lfm_linear` 和有效性。尺度由版本化 target contract 提供。

## 监督目标

canonical synthetic corpus 为每个 parent-zone 生成三个模型网格目标：

1. `projected_log_ai_increment`：模型网格 log-AI 相对线性 anchor 的增量；
2. `signed_reflectivity`：与 `cup.physics` 一致的下界面反射率；
3. `state_emission`：高分辨率状态在嵌套模型网格上的取值。

三个目标共享严格 support。反射率首样点以及缺少上一个有效样点的位置不参与监督。

## 训练与 checkpoint

活动配置：

```text
experiments/evidence/evidence.yaml
experiments/evidence/target_contract.json
experiments/evidence/checkpoints/evidence_full_v1.pt
```

其中 `evidence_full_v1.pt` 是当前 8-epoch full-input 权重按本规格 checkpoint schema 发布的版本。

训练命令：

```powershell
python scripts/evidence.py `
  --config experiments/evidence/evidence.yaml `
  train `
  --corpus <canonical-corpus> `
  --output-dir <output-directory>
```

`--smoke` 在 training/tuning 各读取 3 个 parent 并运行 1 个 epoch。每个完整 epoch 发布 `epoch_XXXX.pt`、`last.pt` 和当前 `best.pt`；`--resume` 只接受当前 checkpoint schema。

`input_mode=no_seismic` 使用同一网络和尺度合同，将 seismic 通道置零，作为独立训练的地震贡献对照。

## 评价

```powershell
python scripts/evidence.py `
  --config experiments/evidence/evidence.yaml `
  evaluate `
  --corpus <canonical-corpus> `
  --checkpoint <full-checkpoint> `
  --output-dir <output-directory> `
  --split calibration
```

评价固定报告：

- increment RMSE、MAE、相关性和区间 coverage；
- reflectivity RMSE、相关性、极性准确率和区间 coverage；
- state cross-entropy、Brier、accuracy 和 balanced accuracy；
- zone-linear anchor 与 full-LFM baseline；
- matched within-parent seismic shuffle；
- 可选的独立 no-seismic checkpoint；
- 以 parent 为统计单位的配对 bootstrap 区间。

## 模块边界

`src/evidence` 只拥有证据提取所需的合同、特征构造、网络、语料 adapter、训练、评价、augmentation 和 checkpoint。高分辨率结构生成属于后续独立研究问题，不能通过本模块的重建相关性或视觉连续性得到证明。
