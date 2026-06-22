# 真实工区 R0-LFM 输入准备

## 文档地位

本文定义模型消融闸门之后、R0 zero-shot 之前的真实工区 LFM 输入准备规格。
它只服务于 R0/R1 research output，不恢复旧第六步井约束，也不恢复旧第七步
`lfm_precomputed` 生产语义。

旧 `scripts/lfm_precomputed.py` 只能作为历史参考。新脚本必须独立实现，不能依赖旧
`well_constraints/lfm_control_points.csv`，不能搜索 `latest`，不能在缺少输入时自动回退到旧
`lfm_precomputed_*` 产物。

未来入口暂定为：

- CLI：`scripts/real_field_lfm.py`
- 核心逻辑：`src/cup/seismic/real_field_lfm.py`
- schema：`real_field_lfm_v1`

首版目标很窄：为真实工区 R0 生成一个当前分支可复现、可审计、缓慢时变的 `log(AI)` LFM
先验，替代历史遗留的 `lfm_precomputed_*/ai_lfm_time.npz`。

## 1. 输入契约

所有来源必须显式配置：

```yaml
real_field_lfm:
  source_runs:
    well_auto_tie_dir: <step-4-run>
  well_inventory_file: <step-1-well-inventory.csv>
  seismic:
    file: <real seismic volume>
    type: segy|zgy
  target_interval:
    horizons:
      - {name: <top>, file: <horizon-file>}
      - {name: <bottom>, file: <horizon-file>}
  trend_fit:
    min_valid_samples_per_well: 32
    huber_f_scale_log_ai: 0.05
    log_ai_min: null
    log_ai_max: null
    max_abs_b_log_ai: 0.35
  parameter_modeling:
    min_wells: 3
    allow_constant_fallback: false
  output_geometry:
    mode: volume
    target_context_s: 0.05
  lfm_qc:
    min_time_diff_rms: 1.0e-4
    min_trace_time_std_median: 1.0e-4
```

要求：

- `target_interval.horizons` 接受任意 `N >= 2` 个有序层位。首层位到末层位定义完整目标窗；
  中间层位只用于 QC 和分层统计，不作为 LFM 拟合边界。
- 层位值必须为 TWT 秒，或携带可审计的单位转换；层位按浅到深有序，交叉 trace 不进入
  `valid_mask_model`。
- 真实地震体只用于输出体几何、采样轴和 R0 对齐，不参与 LFM 数值拟合。
- 控制井来自第四步 `well_tie_metrics.csv` 中 `tie_status=success` 的全部井。
- 每口控制井必须有 `filtered_las_file` 和 `optimized_tdt_file`。
- 第五步 `wavelet_generation` 不参与 LFM 生成；R1 正演诊断仍单独读取第五步子波。

`filtered LAS` 不是低频曲线。它只是当前井震标定体系下的井上阻抗来源，必须经过本文定义的
低阶参数化后，才能成为 LFM 控制信号。

## 2. 控制趋势

每口井在完整目标窗内生成一个低阶控制趋势。首版不按内部层位拆分拟合，避免在中间层位处
人为制造 LFM 台阶。

1. 从第四步 `filtered_las_file` 读取线性 `AI`，剔除非正值和 LAS null，再显式转换为 `log(AI)`。
2. 用第四步 `optimized_tdt_file` 将 MD 域 filtered LAS 投影到 TWT。
3. 在井对应位置采样首层位和末层位，计算完整目标窗比例坐标 `u_total`。中间层位只用于 QC。
4. 对每口井拟合：

```text
logAI(u_total) = a + b(2u_total - 1)
```

拟合规则：

- 主域为 `log(AI)`，不在 AI 线性值上拟合空间模型。
- TWT 化后的拟合输入使用 `well_log_ai_twt_cell_mean`：对每个输出 TWT cell 内的 LAS `log(AI)`
  做长度加权平均。不跨 LAS null、不跨 TDT 长缺口、不跨目标窗边界。
- 使用 `scipy.optimize.least_squares(loss="huber")` 或等价的 Huber 损失拟合；默认
  `huber_f_scale_log_ai = 0.05`。
- 每口井默认至少需要 `min_valid_samples_per_well = 32` 个目标窗内有效样点；不足时拒绝该井。
- 拟合后检查 `a`、`a-b`、`a+b` 和 `b` 的合理范围。`log_ai_min/log_ai_max` 可显式配置；
  若未配置，则用全部有效井上 `log(AI)` 的 robust P01/P99 作为本次运行阈值。默认
  `abs(b) <= 0.35`，超限只记录 `qc_warning`，不直接拒绝控制井。中心、顶底超限分别记录
  `center_out_of_range`、`top_or_bottom_out_of_range`。若未来要把超限升级为拒绝，必须新增显式配置。
- 不跨长缺口插值，不把目标窗外样点纳入拟合。
- 输出记录拟合样点数、`a`、`b`、残差 RMS、状态和拒绝原因。

直井位置使用 `well_inventory.csv` 中的 `inline_float/xline_float`。

斜井使用第四步 `optimized_trace_sample_plan_file` 沿 TWT 插值轨迹坐标，但每口井只贡献
一组 `a/b` 和一个代表位置。代表位置固定为该井目标窗内有效拟合样点的 TWT-cell-length
加权平均 `inline/xline/twt/u`。斜井不得把每个轨迹样点作为独立空间控制点，否则会重新引入
密集轨迹重复加权问题。

斜井控制记录必须额外输出 `trajectory_sample_count`、`representative_inline`、
`representative_xline`、`representative_twt`、`representative_u`、`weighted_twt_min/max` 和
`weighted_u_min/max`，用于审计该井代表了哪一段轨迹。

## 3. 参数场建模

对每口井拟合得到的 `a` 和 `b` 分别做空间参数场建模，再重建三维 LFM：

```text
lfm_log_ai(inline, xline, twt)
  = a_field(inline, xline)
  + b_field(inline, xline) * (2u_total - 1)
```

要求：

- 新 LFM 禁止调用旧比例切片建模入口，包括 `build_point_constrained_model`、
  `build_lfm_time_model_from_points` 以及旧 `scripts/lfm_precomputed.py` 主流程。那些入口按 `u`
  切片逐层克里金采样值，与本文的 `a/b` 参数场方案不是同一件事。
- 空间建模首版采用 ordinary kriging 参数场。距离坐标必须使用真实 XY 米制坐标，不得直接用
  inline/xline 线号做克里金距离，因为 `istep/xstep` 和道间物理距离未必相同。若复用现有
  kriging 代码，只能抽出或新建“点到二维参数场”的 helper，不能走 slice-control pipeline。
- `a_field` 和 `b_field` 共享同一个空间 range hint，由有效控制井坐标的
  真实 XY 最近邻距离中位数估计，避免两个参数场使用不同平滑尺度。
- 默认至少需要 `min_wells = 3` 口有效控制井；不足时显式标记状态。默认
  `allow_constant_fallback = false`，控制井不足时本次运行不产出可消费 `real_field_lfm.npz`。
- 内部层位不生成独立参数场，也不在内部层位处制造 LFM 不连续。
- 不恢复旧 `post_slice_smoothing`、切片后滤波或额外频率后处理。
- LFM 的低频性来自完整目标窗一阶趋势和空间参数场，而不是对输出体做后处理平滑。

参数场 QC 必须显式记录空间外推风险，包括控制井数、控制点凸包面积、网格在凸包外的比例、
到最近控制井距离的 P50/P95、kriging variance 的 P50/P95、range hint、nugget 和模型状态。
可选输出 `distance_to_control` 与 `parameter_uncertainty` 数组；R0 不直接消费这些数组，但图件和
报告必须能看到远离井控的区域。

内部层位处不允许出现由 LFM 参数化制造的跳变。首版不输出 `zone_boundary_jump_qc.csv`，而是输出
`internal_horizon_continuity_qc.csv`，检查每条中间层位上下相邻样点的 LFM 差异是否只来自一阶趋势
的正常采样变化。若发现内部层位处存在异常跃变，应标记 `lfm_internal_horizon_discontinuity`。

## 4. 有效区与 mask

新 LFM 可以输出完整地震体采样轴，但 mask 是唯一权威边界。`log_ai` 只在目标层位之间有限；
目标层外默认写为 `NaN`，对应 `valid_mask_model = false`。首版不做目标层外常数外推。

`valid_mask_model` 与 `lfm_support_mask` 不表达同一件事：

- `valid_mask_model` 是目标层内、轴有效、R0 可参与预测/评价的权威边界。
- `lfm_support_mask` 是 LFM 空间支撑 QC，表示该位置离控制井和参数场证据是否足够近。

R0 主链只用 `valid_mask_model` 作为有效边界；`lfm_support_mask`、`distance_to_control` 和
`parameter_uncertainty` 只用于报告与风险解释。

R0 不强制只能切目标层时间窗。如果 `output_geometry.sample_start_s/sample_end_s` 未显式配置，
新脚本默认按当前输出几何上的首/末层位包络自动裁剪，并添加 `target_context_s` 上下文。这样避免
每个工区手工填写绝对时间窗，同时不把全地震时间轴的大量无效样点交给 patch sampler。

如果 R0 使用更宽时窗，必须严格消费 `valid_mask_model`，并保证：

- patch 采样、prediction stitching、输入 QC、R1 正演诊断和图件统计都不把 mask 外样点当作有效 LFM。
- mask 边界处的正演诊断必须使用显式 crop 或连续有效段，避免 `NaN`/常数区边界产生假反射。
- 时间窗只是工程配置；推荐使用目标窗加小上下文以改善图件和 patch 覆盖，但不是地质边界。

若未来确需目标层外上下文，必须新增显式 `context_s` 语义，并在 summary 中标记外推区。不得用
目标层外常数填充来制造 finite mask。

## 5. 输出契约

首版至少输出：

```text
real_field_lfm.npz
well_trend_controls.csv
parameter_field_qc.csv
internal_horizon_continuity_qc.csv
horizon_qc.csv
real_field_lfm_summary.json
figures/
```

`real_field_lfm.npz` 至少包含：

- `log_ai`：R0 可直接消费的主 LFM，shape 为 `(n_inline, n_xline, n_twt)`。
- `valid_mask_model`：与 `log_ai` 同 shape，目标层内且 LFM 有效处为 true。
- `lfm_support_mask`：可选 QC mask，不作为 R0 主有效边界。
- `distance_to_control`：可选 QC 数组。
- `ilines`
- `xlines`
- `samples`
- `metadata_json`

可选另存 `ai` 仅用于人工 QC，不作为 R0 主接口。R0 配置消费本产物时应使用：

```yaml
real_field_inputs:
  lfm_file: <real_field_lfm>/real_field_lfm.npz
  lfm_value_transform: identity
```

`well_trend_controls.csv` 每口井只输出一条完整目标窗趋势控制，记录控制状态、位置、趋势参数和
拒绝原因。
`parameter_field_qc.csv` 记录控制井数、`a/b` 参数范围、空间建模状态和方差摘要。
`horizon_qc.csv` 记录目标窗与中间层位的层位有效率、厚度 P01/P50/P99、交叉 trace 数、过薄比例和
超出 TDT 支持的井计数。

`real_field_lfm_summary.json` 必须记录：

- `schema_version = real_field_lfm_v1`
- 第四步目录、well inventory、地震体和层位文件路径。
- 使用井、拒绝井、趋势控制数量和拒绝原因。
- 输出字段名：`log_ai`，其值域已经是 `log(AI)`。
- 有效区字段名：`valid_mask_model`。
- LFM 统计：全局 RMS、每道时间方向 std、时间差分 RMS、横向 std。
- 与 synthetic train LFM 的输入域对照统计：mean/std/P01/P99、time diff RMS、trace time std
  median、normalization 后 OOD fraction。
- 对旧 `lfm_precomputed_*` 的禁止说明：本次运行没有从历史 LFM 产物读取任何数值。

图件至少包括：

- 代表 inline/xline 剖面和时间切片。
- `a_field`、`b_field` QC。
- internal horizon continuity QC。
- distance-to-control / parameter uncertainty QC。
- 井旁 `filtered LAS logAI`、拟合趋势和采样 LFM 对照。

输出几何支持 `volume` 与 `section` 两种模式。两者使用同一 schema；`section` 模式只是输出较小的
inline/xline 轴用于 R0 诊断，不能改变 `log_ai`、`valid_mask_model` 和 metadata 语义。

## 6. R0 关系

R0/R1 不得默认使用旧 `scripts/output/lfm_precomputed_*/ai_lfm_time.npz`。当前分支不再要求按路径名
硬拦截旧目录，但必须通过 schema、字段和值域自检确认 LFM 是当前 `real_field_lfm_v1` 的
`log_ai` 语义。若为了对照而使用历史产物，必须在配置和报告中标记为
`legacy_lfm_negative_control`，不得写成当前分支的真实工区 LFM。

R0 应同时读取 `log_ai` 和 `valid_mask_model`。即使 R0 配置使用宽时窗，mask 仍是唯一有效边界；
不得通过目标层外常数外推让 mask 外样点进入模型输入解释或 R1 正演统计。

R0 不得将 NaN 送入模型张量。送入模型前，mask 外的 LFM 和 seismic 在各自 normalization 之后
填 0，`valid_mask_model=false` 作为第三通道保留。不同模型、图件和 R1 诊断必须使用同一 zero-fill
规则。

R0 输入 QC 必须检查 LFM 是否具备基本时间结构，至少报告：

- `lfm_trace_time_std_median`
- `lfm_trace_time_std_max`
- `lfm_time_diff_rms`
- `lfm_lateral_std_median`

默认阈值为 `min_time_diff_rms = 1.0e-4`、`min_trace_time_std_median = 1.0e-4`。低于阈值时标记
`lfm_time_flat_or_invalid` 或 `lfm_time_structure_weak`，并阻止把 R0 结果解释为模型真实外推表现。

## 7. 未来测试

未来实现必须覆盖：

- 所有来源路径显式，不搜索 `latest`。
- 新 LFM 不调用旧切片建模入口。
- 第四步 success 井正确加载 `filtered_las_file` 和 `optimized_tdt_file`。
- filtered LAS 线性 `AI` 被显式转换为 `log(AI)`，R0 不二次取 log。
- filtered LAS 到 TWT 使用 cell mean，不做简单点采样。
- filtered LAS 投影到 TWT 后 `log(AI)` 有效，缺口不静默跨越。
- 任意数量、任意名称的有序层位生成一个完整目标窗，中间层位只做 QC。
- 层位乱序、交叉、无支撑或超出 TDT 支持范围时显式失败或拒绝对应井。
- 直井和斜井控制位置来源正确，斜井每口井只产生一个代表控制。
- 井级趋势拟合可复现，低样点井有状态和拒绝原因。
- `a`、`a-b`、`a+b` 或 `b` 超限井默认只记录 `qc_warning`，不因端点轻微越界被拒绝。
- 少于 3 口有效井不产出可消费 LFM，除非配置显式允许常数退化。
- 克里金、range hint、distance-to-control 和凸包外推比例均使用真实 XY 米制坐标，而非
  inline/xline 线号。
- `a/b` 参数场重建的 `log_ai` 与 R0 loader 轴契约一致。
- 内部层位连续性 QC 输出完整；首版不允许内部层位出现人工跳变。
- kriging 外推风险 QC、distance-to-control 和 support mask 语义正确。
- 目标层外输出为 `NaN` 且 `valid_mask_model=false`；R0 不能把 mask 外样点纳入有效统计。
- R0 mask 外输入 normalization 后 zero-fill，不把 NaN 送入模型。
- 线性 AI 或其他非 `log(AI)` LFM 值域必须被拒绝。
- 输出 LFM 在有效 mask 内不再出现全道时间方向完全平坦，除非所有输入趋势本身为平坦且 summary 明确说明。
- `compileall` 通过；完整测试仍由用户运行。

## 8. Assumptions

- 首版只解决 R0 真实工区 LFM 输入可复现问题，不恢复旧生产第七步。
- 控制井使用第四步 `tie_status=success` 全部井，不引入旧第六步井约束筛选。
- 第五步 selected wavelet 不参与 LFM 生成。
- 首版输出完整工区采样轴，而不是仅输出当前 R0 剖面；目标层外由 mask 排除，不做常数外推。
- 不为坏井、缺层位、缺 TDT 或缺 filtered LAS 做自动回退；只记录拒绝原因。
