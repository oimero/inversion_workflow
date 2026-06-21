# 真实工区 Zero-Shot 与正演诊断

## 文档地位

本文定义模型消融闸门之后、进入真实工区适配之前的两阶段研究接口：

- **R0 zero-shot**：用冻结合成基准上最强的两条模型主线直接预测真实工区，只做 QC，
  不训练、不校准、不宣称生产结果。
- **R1 forward diagnostic**：对 R0 预测的 `log(AI)` 做固定正演诊断，拆解真实地震
  残差来源，为后续 calibration-only 或 adapter 判断依据。

本文是 R0/R1 的实施级规格和验收入口；相关脚本即使落地，也不注册为正式第六步，不恢复旧 `src/ginn/`、旧 GINN
配置、旧 checkpoint 或旧推理契约。需要借鉴旧实现时，只借鉴输入契约、mask 边界、
正演诊断和 QC 记录方式，不继承真实地震 waveform loss 训练主模型的路线。

当前实现入口为：

- R0：`scripts/real_field_zero_shot.py`
- R1：`scripts/real_field_forward_diagnostic.py`
- 核心包：`src/ginn_v2/real_field.py` 或后续等价模块

R0 使用的真实工区 LFM 必须来自当前分支可复现的输入准备契约，见
[`real-field-lfm-input.md`](real-field-lfm-input.md)。历史遗留
`scripts/output/lfm_precomputed_*/ai_lfm_time.npz` 不得作为默认 LFM；如需对照，只能显式标记为
`legacy_lfm_negative_control`。

稳定生产链仍终止于第五步。R0/R1 输出均为 research output，在完成 R2/R3/R4 前不得替代
前五步生产链。

## 1. R0 回答的问题

R0 只回答：

1. 冻结合成基准上表现最强的模型，直接外推到真实工区时是否发生明显崩坏。
2. 无 lateral 与有 lateral 两条主线在真实剖面上的差异，是稳定收益、视觉平滑，还是潜在
   横向污染。
3. 模型预测的 `pred_log_ai`、输入 LFM 和 `pred_delta_vs_lfm` 在井旁、剖面和频谱上是否
   具有可解释的形态。

R0 不回答：

- 是否应该进入生产。
- 是否需要修改 wavelet、gain、phase 或 LFM。
- 是否需要真实井 adapter。
- 是否某个模型已经通过真实工区验证。

R0 运行状态只允许使用：

- `qc_ready`
- `input_contract_failed`
- `prediction_failed`
- `needs_forward_diagnostic`

任何 R0 结果都不能写成“通过/失败”的架构结论。R0 完成后默认进入 R1 正演诊断。

## 2. R0 模型集合

首批 R0 只比较两条主线：

| 角色 | 模型 |
| --- | --- |
| 无 lateral 对照 | `trace1d_tcn_mismatch` |
| 有 lateral 主候选 | `trace1d_tcn_lateral_mixer_mismatch` |

k5 mixer、post-hoc smoothing、physics loss、tiny physics 和其他临时消融不进入首批 R0
主表，只能作为后续附录候选或单独复核实验。这样做是为了把真实工区体检保持为一个清晰
问题：横向上下文相对强 1D 时间主干是否提供真实增益。

两条模型必须使用同一真实工区输入、同一 LFM、同一 mask、同一采样轴和同一输出格式。
不得为了某个模型单独调输入、mask、归一化或后处理。

## 3. R0 输入契约

每次 R0 必须显式指定全部来源，禁止搜索 `latest`、按修改时间猜目录、扫描 checkpoint
目录或缺失时自动回退：

```yaml
real_field_zero_shot:
  source_runs:
    wavelet_generation_dir: <step-5-run>
  real_field_inputs:
    seismic_file: <real seismic volume>
    seismic_type: segy|zgy
    seismic_value_transform: p99_abs_matched
    lfm_file: <real_field_lfm_v1/real_field_lfm.npz>
    lfm_value_transform: identity
    target_mask_file: <target-window mask or horizon-derived mask source>
  models:
    - model_role: no_lateral
      model_id: trace1d_tcn_mismatch
      model_run_dir: <frozen model run>
      checkpoint_file: <checkpoint>
      normalization_file: <normalization manifest>
      synthetic_gate_summary: <frozen synthetic gate summary>
      synthetic_gate_report_sha256: <sha256>
    - model_role: lateral
      model_id: trace1d_tcn_lateral_mixer_mismatch
      model_run_dir: <frozen model run>
      checkpoint_file: <checkpoint>
      normalization_file: <normalization manifest>
      synthetic_gate_summary: <frozen synthetic gate summary>
      synthetic_gate_report_sha256: <sha256>
```

实现必须校验：

- 真实地震、LFM、mask 使用同一 inline/xline/TWT 轴或可审计的显式重采样结果。
- LFM 是 `log(AI)` 语义，且单位、采样间隔和时间轴写入 manifest。
- LFM 必须包含 R0 可直接消费的 `log_ai` 字段；不得把旧 AI 体通过 `lfm_value_transform: log`
  伪装成当前分支 LFM。
- LFM 必须提供 `valid_mask_model` 或等价目标层有效区；mask 是 R0/R1 的权威边界，时间窗只是
  裁剪和图件配置。
- 模型 checkpoint、model manifest、normalization manifest 和 `model_id` 一致。
- 模型候选必须能追溯到冻结 synthetic gate summary 和 report SHA-256；不得把其他 seed、
  其他 checkpoint、k5 变体或临时平滑后处理混作同一候选。
- 归一化参数来自合成基准 train split，R0 不重新估计 normalization。
- 第五步 selected wavelet 只用于 R1 正演诊断和边界建议，不改变 R0 模型预测。

若真实工区数据需要从 3D 体裁剪剖面或小体，裁剪范围必须进入 `prediction_index.csv`，
不能只保存在命令行日志中。

R0 不强制只使用目标层时间窗。若配置使用更宽时窗，patch 采样、stitching、输入 QC、井旁图件和
R1 正演诊断都必须严格使用 LFM 的 `valid_mask_model`。不得通过目标层外常数外推制造 finite LFM，
也不得把 mask 外平坦 LFM 解释为模型真实输入条件。推荐使用目标窗加小上下文，只是为了改善图件
和 patch 覆盖，不是地质边界。

R0 不得将 NaN 送入模型张量。送入模型前，mask 外 LFM 和 seismic 在各自 normalization 之后填 0，
并保留 `valid_mask_model=false` 作为输入通道和评价边界。若 `lfm_file` 指向
`lfm_precomputed_*`，primary R0 必须失败；只有显式 `legacy_lfm_negative_control: true` 时才允许
作为历史负对照运行。

若 LFM summary 或 `zone_boundary_jump_qc.csv` 显示相邻 zone 边界跳变主导正演响应，R1 必须标记
`lfm_boundary_jump_dominated`。该反射只能解释为 LFM 构造风险，不能解释为模型预测能力。

### 3.1 固定输入值域变换

R0 允许对真实地震输入做一个显式、可追溯的值域变换，使其进入合成训练集的 seismic 输入
尺度。该变换属于输入契约，不是 gain 校准、wavelet 校准、模型后处理或 R1 正演调参。变换
必须在 `real_field_zero_shot_summary.json` 中记录真实输入统计、synthetic train 参考统计、缩放
系数、中心值和极性。

当前工区冻结候选为：

```yaml
real_field_inputs:
  seismic_value_transform: p99_abs_matched
```

其语义是：以真实地震窗内中位数为中心，用 synthetic train seismic 的 `abs_p99` 匹配真实地震
窗内 `abs_p99`。R0.6 输入变换闸门中，原始地震归一化后几乎全窗越界；`p99_abs_matched`
将 `fraction_abs_normalized_seismic_gt_5` 降为 0，并使两条 zero-shot 模型输出回到约
`log(AI)≈9` 的合理范围。因此，后续 R0/R1 默认使用该变换；`raw` 只保留为负对照。

可选候选包括 `robust_rms_matched`、`p95_abs_matched` 及极性翻转版本，但不能在同一次 R0
主表中与 `p99_abs_matched` 混用。若未来换工区或换合成基准，必须重新运行输入变换闸门，
不能继承当前工区的缩放系数。

### 3.2 输入分布 QC

真实地震和真实 LFM 是模型输入，不只是图件背景。R0 必须在模型预测前输出输入分布 QC，
并与 synthetic train normalization manifest 对照。至少记录：

- real seismic 的 mean、RMS、robust RMS、P01/P50/P99。
- real seismic 经 synthetic train normalization 后的 mean、std、P01/P50/P99。
- `fraction_abs_normalized_seismic_gt_3` 和 `fraction_abs_normalized_seismic_gt_5`。
- real LFM `log(AI)` 的 mean、std、P01/P50/P99。
- real LFM 经 synthetic train normalization 后的 outlier fraction。
- mask 内有限样点比例、每个剖面或小体的有效覆盖率。

若真实输入明显超出 synthetic train 分布，应记录 `input_distribution_warning`。该状态不自动
阻断 R0，但 R0 结论必须明确说明：异常可能来自输入域偏移，而不是模型结构本身。

## 4. R0 输出语义

模型对外只输出：

```text
pred_log_ai
```

内部是否预测 `delta_log_ai` 不属于真实工区契约。R0 同时输出：

```text
pred_delta_vs_lfm = pred_log_ai - lfm_input
```

但它只用于 QC，不作为下游正式模型接口。任何 evaluator、图件和 R1 正演诊断都应优先消费
反归一化后的 `pred_log_ai`。

首版至少输出：

- patch-level `predictions.npz` 或等价剖面预测文件。
- stitched prediction：整剖面或整小体的 `stitched_pred_log_ai`。
- `stitching_weight` 或 `blend_mask`，记录每个样点被多少 patch 覆盖。
- `prediction_index.csv`：记录模型、输入文件、inline/xline/TWT 范围、采样轴和有效 mask。
- `model_input_qc.csv`：记录真实工区输入的 finite 比例、RMS、范围、mask 覆盖率和拒绝原因。
- `real_field_zero_shot_summary.json`：记录来源、模型、归一化、状态和文件校验值。
- 图件：
  - 剖面级 `LFM / pred_delta_vs_lfm / pred_log_ai` 三联图。
  - 井旁抽样曲线图。
  - 局部频谱或分频能量 QC 图。
  - 无 lateral 与 lateral 的差值图。

图件必须能区分三件事：视觉平滑、频谱平滑和正演一致性。若 lateral 模型剖面更连续，
不能只凭视觉连续性判断其更好。

Patch 到剖面的拼接策略只能使用：

- uniform averaging。
- center-crop stitching。

两条模型必须使用同一种拼接策略，并记录 `patch_id`、inline/xline/TWT 范围、overlap count、
blend method 和 prediction coverage mask。真实工区无 lateral 与 lateral 的差异不得来自
不同拼接策略。

`post-hoc smoothing` 只能作为后处理诊断附录，不得覆盖 R0 主输出，也不得与 zero-shot
主模型混列。k5 mixer 只有在完成 full/multi-seed synthetic gate 复核后，才允许进入 R0
附录候选。

## 5. 边界与 mask 策略

所有边界业务参数使用秒，不使用裸样点数：

```yaml
real_field_boundary:
  loss_or_eval_erosion_s: <seconds>
  prediction_taper_halo_s: <seconds>
  forward_diagnostic_crop_s: <seconds>
  suggested_from_nominal_wavelet: true
```

实现可用第五步 nominal 子波 active half-support 生成建议值，但必须分别记录建议值和最终
使用值。样点数只由 `dt_s` 派生，并写入 manifest：

```text
loss_or_eval_erosion_samples = ceil(loss_or_eval_erosion_s / dt_s)
prediction_taper_halo_samples = ceil(prediction_taper_halo_s / dt_s)
forward_diagnostic_crop_samples = ceil(forward_diagnostic_crop_s / dt_s)
```

`loss_or_eval_erosion_s`、`prediction_taper_halo_s` 和 `forward_diagnostic_crop_s` 不得共用
同一个隐藏参数。R0 预测不得被 R1 的 forward crop 偷偷改写；crop 只用于 R1 的正演诊断
统计和图件。若 R0 需要对层外预测做 taper，必须只作用于 `pred_delta_vs_lfm`，并在
summary 中记录。

本文借鉴旧 GINN 的 wavelet half-support 思路，只是为了避免正演诊断统计被卷积边界污染；
它不是 physics loss，也不是训练或优化模型的依据。

## 6. R1 正演诊断

R1 消费 R0 的 `pred_log_ai`，固定正演成 synthetic seismic，与真实地震比较。R1 不反向传播、
不更新 checkpoint、不写 adapter 权重、不校准主模型。

至少比较三类阻抗输入：

| 输入 | 语义 |
| --- | --- |
| `lfm_only` | 只用真实工区 LFM/低频先验 |
| `zero_shot_no_lateral` | 无 lateral 主线的 `pred_log_ai` |
| `zero_shot_lateral` | lateral 主候选的 `pred_log_ai` |

正演契约固定为：

```text
x = log(AI)
r[j] = tanh((x[j] - x[j-1]) / 2), j = 1,...,N-1
```

实现必须明确记录：

- 反射系数挂点。
- 子波来源和 SHA-256。
- 卷积 mode、子波中心和输出时间轴。
- `synthetic_twt_axis` 与裁剪后的 `observed_twt_axis_after_alignment`。
- 因 N 到 N-1 正演轴转换而丢弃、裁剪或对齐的样点数。
- `time_alignment_mode`，必须与第五步和正演可观测性闸门的 forward core 一致。
- 是否使用 R0 mask erosion 与 R1 forward crop。

首版使用第五步 nominal selected wavelet。候选子波、相位和时移只进入诊断扫描，不作为
R0 预测输入，也不覆盖 nominal 主结果。

R1 不得在脚本内重新发明一套反射系数挂点和卷积对齐。若实现无法复用第五步或第一闸门
公开的 forward core，必须在 summary 中记录新的 forward operator id，并将其视为不同
诊断 schema。

## 7. R1 残差拆解

R1 至少输出以下诊断：

- positive scale / gain：带截距或去均值后的正约束最小二乘缩放。
- scalar bias / intercept：记录整体均值差异，不解释为模型地质能力。
- phase perturbation：小范围常相位扫描。
- fractional shift：小范围亚采样时移扫描。
- wavelet bandwidth / candidate sensitivity：nominal 与第五步合格候选子波的结果对照。
- spatial residual pattern：沿 inline/xline 或剖面距离的残差能量、条带和局部异常。

`scale <= 0`、synthetic 能量过低、真实地震能量无效、时间轴错位或无有效 mask 时，必须写入
明确状态。不得通过反极性让负 scale 结果变成有效。

scale/intercept 只是诊断项，不得改写 R0 prediction。每个模型、窗口和井旁诊断至少输出三套
指标：

- `raw_residual_metrics`
- `positive_scale_only_metrics`
- `positive_scale_plus_intercept_metrics`

字段至少包括 `scale_positive`、`intercept`、`scale_status`、`synthetic_rms_before_scale`、
`observed_rms`、`residual_rms_raw`、`residual_rms_scaled`、`residual_corr_raw` 和
`residual_corr_scaled`。

phase/shift 扫描范围必须显式配置并写入 manifest，例如：

```yaml
real_field_forward_diagnostic:
  diagnostic_scan:
    phase_deg: [-20, -10, 0, 10, 20]
    fractional_shift_samples: [-1.0, -0.5, 0.0, 0.5, 1.0]
```

phase/shift 扫描只用于解释 residual sensitivity，不得自动选择最优 phase/shift 覆盖 nominal
主诊断，也不得用最优扫描结果替换 R0/R1 主表。

R1 至少输出：

- `forward_diagnostic_metrics.csv`
- `well_forward_diagnostic.csv`
- `residual_decomposition.csv`
- `spatial_residual_qc.csv`
- `real_field_forward_diagnostic_summary.json`
- 图件：
  - 井旁 observed/synthetic/residual 对比。
  - LFM-only、no-lateral、lateral 三者正演对比。
  - phase/shift/gain 扫描图。
  - 空间残差图。

井旁 QC 是局部钉子和异常检测，不是 R0/R1 全区通过阈值。`well_forward_diagnostic.csv`
应记录井旁 `log(AI)` 对比、井旁 synthetic 与 observed 对比、局部频谱和局部 bias，但不得
用个位数井直接给整区模型打分。

## 8. 关于锯齿与增量参数化

R0/R1 不改变已训练模型的内部参数化，也不把“预测增量导致锯齿”写成预设结论。

报告必须同时展示：

- `lfm_input`
- `pred_log_ai`
- `pred_delta_vs_lfm`
- 局部频谱或分频能量
- 正演 synthetic 与 residual

若 `pred_delta_vs_lfm` 呈锯齿状，但 `pred_log_ai` 和正演残差稳定，需要把它记录为
参数化/频谱现象，而不是直接判定模型失败。若 lateral 模型只是视觉上平滑，但井旁曲线、
频谱或正演残差变差，也不得把视觉连续性解释为真实收益。

无 lateral 与 lateral 的差值图必须拆分为：

- `lateral_minus_no_lateral_fullband`
- `lateral_minus_no_lateral_lowfreq`
- `lateral_minus_no_lateral_observable_band`
- `lateral_minus_no_lateral_highfreq_or_nullspace`

频带边界不得硬写 35/70 Hz；应来自第一闸门证据、当前 nominal 子波响应或 R1 显式配置。

R0/R1 必须正式报告不可观测频带异常，而不是只画“局部频谱”。至少记录：

- `nullspace_energy_ratio`
- `observable_band_energy_ratio`
- `pred_delta_spectrum_vs_synthetic_train`
- `pred_delta_spectrum_vs_well`
- `lateral_minus_no_lateral_nullspace_energy`

这些指标优先作用于 `pred_delta_vs_lfm`，因为 `pred_log_ai` 会被 LFM 低频主导，容易掩盖模型
增量的异常。

## 9. 阶段判定

R1 完成后，才允许给出下一步建议：

- 进入 **R2 calibration-only**：当主要问题可由 gain、bias、低频校正、小相位/时移解释，
  且模型预测本身没有明显异常。
- 继续 synthetic 消融：当真实工区异常对应到合成基准未覆盖的场景，例如横向几何、LFM
  误差或失配类型不足。
- 暂缓真实工区适配：当 zero-shot 预测存在无法由 R1 解释的异常振幅、条带、锯齿或井旁
  不合理偏移。

R1 不能直接批准 R3 adapter。Adapter 必须等 R2 calibration-only 证明仅靠输入域校准不足，
且留井/留簇验证设计清楚之后再进入。

R1 red flags 出现时，不得进入 R2 calibration-only：

- `scale <= 0`、synthetic energy invalid 或 observed energy invalid。
- 明显时间轴错位尚未解释或修复。
- `pred_delta_vs_lfm` 在不可观测频带异常增能。
- lateral 与 no-lateral 的差异主要集中在 null-space。
- 井旁 AI 明显偏移，但 waveform residual 因 scale/phase/shift 扫描而变好。
- R0 输入分布存在严重 OOD warning，且无法由 normalization 或数据准备问题解释。

出现 red flag 时，应优先回到 synthetic 消融、输入准备或正演契约检查，而不是继续调 gain。

## 10. Manifest 最小字段

R0/R1 manifest 至少记录：

- `source_file_sha256`
- `model_checkpoint_sha256`
- `normalization_sha256`
- `synthetic_gate_report_sha256`
- `wavelet_sha256`
- `axis_contract`
- `mask_contract`
- `input_distribution_qc`
- `output_prediction_sha256`
- `code_version_or_git_commit`

真实工区输入文件、模型 checkpoint 或 normalization 发生变化时，必须产生新的运行目录和新的
manifest。不得在原目录中覆盖旧结果。

## 11. 未来实现测试

未来实现至少覆盖：

1. 显式路径输入，缺失或来源不一致时失败，不搜索 `latest`。
2. 真实工区 seismic、LFM、mask、TWT 轴与模型 normalization manifest 一致。
3. 两个 R0 模型使用同一真实工区输入、同一 mask 和同一输出格式。
4. `pred_log_ai` 是对外唯一预测目标，`pred_delta_vs_lfm` 只作 QC。
5. 边界参数以秒配置，样点数派生正确；erosion、halo、forward crop 不共用字段。
6. R0 预测不会被 R1 forward crop 改写。
7. R1 使用固定 `tanh(ΔlogAI/2)`、第五步 nominal wavelet 和明确卷积约定。
8. `scale <= 0`、低 synthetic energy、时间轴错位、无有效 mask 都产生明确状态。
9. R1 不反向传播、不更新 checkpoint、不写 adapter 权重。
10. R0/R1 图件能区分视觉平滑、频谱平滑和正演一致性。
11. 两条主线可横向对照，且不会把 k5、post-hoc smoothing 或 physics/tiny physics 误混进主表。
12. 运行 summary 记录输入文件、checkpoint、normalization、wavelet 和输出文件校验值。
13. 输入分布 QC 能识别 seismic/LFM normalized outlier，并写入 warning。
14. patch 拼接输出 overlap count 和 blend mask；两条模型使用同一拼接策略。
15. R1 正演 N 到 N-1 的时间轴对齐与第五步/第一闸门 forward core 一致。
16. phase/shift 扫描不会覆盖 nominal 主诊断。
17. null-space energy red flag 能阻止进入 R2 calibration-only。

## 12. 首版边界

- 本文只定义 R0/R1 research output 的实施边界；它们仍不是正式第六步。
- R0 首批只比较 `trace1d_tcn_mismatch` 与 `trace1d_tcn_lateral_mixer_mismatch`。
- 边界默认建议可以来自 nominal 子波 active half-support，但所有业务配置和报告都使用秒。
- R1 的 wavelet/gain/phase/shift 诊断是解释工具，不是 physics loss，也不是训练目标。
- 真实工区输出在完成 R2/R3/R4 前均标记为 research output，不替代前五步稳定生产链。
