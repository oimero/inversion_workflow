# 核心 CSV 契约

这些 CSV 是当前工作流的跨脚本契约，覆盖主链第一至八步及三个旁路。

下游读取上游结果时必须按本文解释字段语义，不得凭字段名猜测。
标题中标记“诊断”的 CSV 只供人工审阅。

---

## 01 · well_inventory.py

### `well_inventory.csv`

消费者：`well_screen.py`、`well_auto_tie.py`、`well_trajectory.py`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `has_well_head` / `has_las` / `has_well_trace` / `has_time_depth` / `has_well_tops` | 资产存在性 |
| `surface_x` / `surface_y` | 井口坐标 |
| `bottom_x` / `bottom_y` | 底孔坐标 |
| `survey_position` | `inside` / `near_outside` / `outside` / `invalid_xy` |
| `wellbore_class` | `vertical` / `deviated` / `unknown` |
| `inventory_status` | 资产盘点状态，不等同于最终候选状态 |

## 02 · well_screen.py

### `well_screen.csv`

消费者：`well_preprocess.py`、`well_auto_tie.py`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `las_file` | 原始 LAS 路径 |
| `screen_status` | `passed` / `partial` / `failed` |
| `has_p_sonic` / `has_density` / `has_caliper` | 曲线类别可用性 |
| `primary_p_sonic` / `primary_density` / `primary_caliper` | 代表曲线 mnemonic |
| `selected_curve_count` | 选中曲线数量 |
| `exported_las` | 第二步瘦身 LAS；仅 passed 井应有值 |

### `las_curve_inventory.csv`

消费者：`well_preprocess.py`

一条 LAS 曲线一行。第三步用它恢复每口井的曲线类别、primary 选择和原始单位。

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `mnemonic` | 原始 LAS 曲线名，必须能在对应 LAS 中精确回查 |
| `unit` | LAS 头中的原始单位 |
| `category` | 标准类别或未分类状态 |
| `is_primary` | 是否为该类别进入下游的代表曲线 |
| `classification_source` | `override` / `mnemonic_rule` / `unclassified` |
| `confidence` | 分类置信度 |
| `notes` | override、歧义或禁用原因 |

## 03 · well_preprocess.py

### `well_preprocess_status.csv`

消费者：`well_auto_tie.py`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `preprocess_status` | `passed` / `failed` |
| `usable_p_sonic` / `usable_density` / `usable_caliper` | 清洗后的曲线可用性 |
| `final_p_sonic` / `final_density` / `final_caliper` | 最终标准 mnemonic |
| `preprocessed_las` | 第三步 LAS；固定包含 `DT_USM`、`RHO_GCC` 和缺失值严格传播的全频 `AI` |

## 旁路 · well_trajectory.py

### `well_trajectory.csv`

消费者：`well_auto_tie.py`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `trajectory_status` | `passed` / `warning` / `failed` / `missing` |
| `wellbore_class_initial` | 第一阶段井头初分 |
| `wellbore_class_qc` | 轨迹复核后的井型 |
| `class_changed` | 初分与复核是否不一致 |
| `surface_survey_position` / `bottom_survey_position` | 井口和井底相对工区的位置 |
| `trajectory_inside_fraction` | 轨迹点位于工区内的比例 |

## 04 · well_auto_tie.py

### `well_tie_plan.csv`

消费者：`wavelet_generation.py`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `route` | 标定路由 |
| `route_status` | `planned` / `skipped_disabled` / `rejected` |
| `wellbore_class_qc` | 轨迹复核后的井型 |
| `has_time_depth` / `has_well_trace` / `has_well_tops` | 路由相关资产存在性 |
| `usable_p_sonic` / `usable_density` | 标定所需曲线是否可用 |
| `input_las` | 第三步预处理 LAS |
| `time_depth_file` / `well_trace_file` | 时深表和井轨迹输入 |
| `surface_x` / `surface_y` | 井口平面坐标 |
| `kb_m` | 井口补心高程，单位 m |

### `well_tie_metrics.csv`

消费者：`wavelet_generation.py`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `route` | 实际执行的标定路由 |
| `tie_status` | `success` / `failed` |
| `optimized_tdt_file` | 细标定后的内部 TDT CSV |
| `filtered_las_file` | 最优滤波参数导出的 LAS，包含重新计算的 `DT_USM`、`RHO_GCC` 和 `AI` |
| `seismic_trace_file` | 保存的井旁或轨迹地震道 |
| `optimized_trace_sample_plan_file` | 斜井 optimized TDT 对应的落道计划；直井为空 |
| `joint_observed_fraction` | 原目标窗内 DT/RHO 联合观测比例 |
| `short_gap_filled_samples` / `long_gap_count` / `longest_long_gap_s` | 缺口处理 QC |
| `continuous_tie_window_start_s` / `continuous_tie_window_end_s` | 实际标定连续窗 |
| `continuous_tie_sample_count` | 连续窗的地震样点数 |
| `continuous_tie_log_sample_count` | 连续窗的原始 MD 样点数，仅作 QC |
| `tie_window_clipped_for_log_gap` | 标定窗是否因长缺口裁剪 |

`optimized_trace_sample_plan_file` 中的 `trace_plan_index` 是当前计划内的局部行号。
裁剪或重建计划后会重新编号，不是地震体全局索引。解释轨迹位置时应读取同一行的
`twt_s`、`inline_float` 和 `xline_float`。

### `wavelet_inventory.csv`

消费者：`wavelet_generation.py`

第四步写出的候选子波清单。第五步只从此表选择候选，不扫描目录中的其他子波文件。

| 关键字段 | 含义 |
|----------|------|
| `source_well` | 子波来源井 |
| `route` | 来源井标定路由 |
| `wavelet_file` | 候选子波 CSV 的 repo-relative 路径 |
| `dt_s` | 子波采样间隔，单位 s |
| `n_samples` | 子波样点数 |
| `tie_corr` / `tie_nmae` | 第四步最终匹配指标 |
| `usable_as_candidate` | 是否允许作为第五步候选 |
| `reasons` | 不可用或需审计的原因 |

## 05 · wavelet_generation.py

第五步产物由第六步 `forward_observability.py` 消费。以下表均由第六步读取。

### `selected_wavelet.csv`

第五步选出的全局子波。

| 关键字段 | 含义 |
|----------|------|
| `time_s` | 子波时间轴，单位 s，中心应在 0 附近 |
| `amplitude` | 已归一化的子波振幅 |

### `batch_synthetic_metrics.csv`

第五步用全局子波在评测井上重新正演后的逐井指标。

| 关键字段 | 含义 |
|----------|------|
| `candidate_wavelet` | 评测使用的全局子波名称 |
| `source_well` | 子波来源；共识子波可为 `optimized_consensus` |
| `eval_well` | 被评测井名 |
| `route` | 被评测井的第四步路由 |
| `corr` / `nmae` | 合成记录与保存地震道的匹配指标 |
| `scale` | 单井最小二乘振幅缩放，仅用于评价和 QC |
| `n_eval_samples` | 参与评价的样点数 |
| `spatial_cluster_id` / `spatial_cluster_size` | 密井平台去偏用的空间簇信息 |
| `status` | `ok` / `failed` |
| `reasons` | 失败原因 |

### `wavelet_candidate_aggregate.csv`

消费者：`forward_observability.py`

闸门只用 `source_well` 与第四步 `wavelet_inventory.csv.source_well` 做规范化井名后的
1:1 联接。`candidate_wavelet` 是场景名，不是文件路径联接键。

### `evaluation_well_spatial_clusters.csv`

消费者：`forward_observability.py`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 第五步固定的评测井 |
| `spatial_cluster_id` | 跨井证据去偏使用的空间簇 |
| `spatial_cluster_size` | 当前簇井数，仅作 QC |

## 06 · forward_observability.py

所有表使用 schema `forward_observability_v1`。上游来源可从配置显式指定，也可自动发现最新合格产物。

### `operator_transfer.csv`

逐子波场景、逐频率记录子波、离散差分和联合算子的幅值/相位、归一化联合幅值、
`operator_support_class` 以及 FFT、差分和卷积约定。

### `well_frequency_sensitivity.csv`

一口井、一个窗口、一个频率、一个子波场景一行。

| 关键字段 | 含义 |
|----------|------|
| `window_id` / `window_type` | 通用全目标窗或相邻层段窗 |
| `frequency_hz` | 本行扰动频率 |
| `wavelet_scenario` / `wavelet_scenario_kind` | nominal、候选或人工小失配场景 |
| `baseline_scale` | standardized observed units 下的 nuisance amplitude slope，不是物理增益 |
| `mismatch_rms` | 同窗加权现实失配底 |
| `sensitivity_scale_marginalized` | 边缘化整体 scale 后的保守灵敏度 |
| `noise_equivalent_log_ai` | 产生现实失配底所需的等效 `log(AI)` 幅度 |
| `detectability_ratio` | 第三步井上窄带幅度与 noise-equivalent 幅度之比 |
| `operator_support_class` | 解析算子支持等级，与经验状态分开 |
| `status` / `reasons` | 本场景有效性或拒绝原因 |

### `well_frequency_aggregate.csv`

在同井、同窗、同频率内对有效子波场景做 lower empirical P25 聚合。记录 nominal、
候选和人工扰动有效数量；场景不足时为 `insufficient_wavelet_scenarios`，不允许
nominal-only 进入空间聚合。

### `cluster_frequency_aggregate.csv`

先在同一 `spatial_cluster_id` 内对井的 detectability ratio 取中位数。层位 TWT 因井而异，
不属于跨井分组键；表中仅将窗口起止时间中位数作为 QC。

### `frequency_evidence_bands.csv`

逐通用窗口和频率记录有效井数、有效空间簇数、跨簇中位数与 lower empirical P25、
经验状态、nominal/conservative 解析支持和 zone warnings。全目标窗至少需要 5 井、
3 簇，否则为 `insufficient_evidence`。

### 其他正式输出

- `well_status.csv`：来源核验和逐井运行状态，并保留第五步 batch 指标作为 QC。
- `well_window_status.csv`：逐井窗口的实际连续分析范围与拒绝原因。
- `wavelet_scenario_qc.csv`：第五步汇总与第四步 inventory 的候选联接结果。
- `recommended_experiment_ranges.json`：仅用于 `synthoseis-lite` 的实验区间。
- `run_summary.json`：完整来源、参数、拒绝统计、warning 和建议区间。

## 旁路 · synthoseis_lite.py (calibrate)

校准阶段冻结一份可从井数据复现的阻抗统计模型。所有输出使用 schema
`synthoseis_lite_impedance_calibration_v1`。

### `impedance_calibration.json`

消费者：`synthoseis_lite.py generate`

| 关键字段 | 含义 |
|----------|------|
| `schema_version` | 固定 `synthoseis_lite_impedance_calibration_v1` |
| `generator_family` | 固定 `object_coefficients_v1` |
| `truth_dt_s` | 高分辨率真值网格的时间采样间隔 |
| `state_threshold_sigma` | 三态划分的 log(AI) 残差 σ 倍数 |
| `ordered_horizons` | 从浅到深的层位名列表 |
| `zones` | 每个区域的顶层位、底层位和区域级统计 |
| `parent` | 父先验：每种态的参数均值、协方差和权重 |
| `zone_models` | 每个区域的三态高斯模型参数 |
| `source_runs` | 校准所用的第三、四、五步来源目录 |
| `source_hashes` | 各来源文件的 SHA-256，锁定输入版本 |

### `zone_models.csv`

| 关键字段 | 含义 |
|----------|------|
| `zone_id` | 区域标识 |
| `state` | `low_impedance` / `background` / `high_impedance` |
| `mean` / `std` | 该态 log(AI) 残差的均值与标准差 |
| `n_samples` | 该态有效样本数 |
| `weight` | 层级化证据权重 |

### `object_catalog.csv`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 来源井 |
| `zone_id` | 所在区域 |
| `state` | 对象阻抗态 |
| `c0` / `c1` / `c2` | 轮廓拟合系数（均值、线性趋势、曲率） |
| `duration_s` / `thickness_m` | 对象持续时间和厚度 |
| `profile_mean` / `endpoint_difference` / `peak_to_peak` | 轮廓诊断指标 |

### `transfer_matrix.csv`

| 关键字段 | 含义 |
|----------|------|
| `from_state` / `to_state` | 转移方向 |
| `probability` | 层级化估计的转移概率 |
| `support` | `direct` / `mixed` / `forbidden` |

## 旁路 · synthoseis_lite.py (generate)

生成阶段冻结一个可复现的二维声阻抗合成基准。master schema 为 `synthoseis_lite_v1`。

### `synthetic_benchmark.h5`

消费者：`evaluate_synthoseis_lite.py` 及所有模型训练/推理

HDF5 文件，每个 sample 为一个 group，包含 `model_target_log_ai`、
`seismic_input`、`valid_mask` 及 priors。所有数组带有 `sha256`、`unit`、
`domain` 和 `axis_order` 属性。

### `sample_index.csv`

消费者：`evaluate_synthoseis_lite.py`

| 关键字段 | 含义 |
|----------|------|
| `sample_id` | 样本唯一标识 |
| `sample_kind` | `base` / `frequency_probe` / `seismic_variant` / `frequency_probe_seismic_variant` |
| `suite` | `canonical` / `field_conditioned` |
| `section_id` | 所属剖面 |
| `geometry_family` | `none` / `wedge` / `pinchout` 等 |
| `status` | `ok` / `rejected` |
| `hdf5_group` | HDF5 内 group 路径 |

### `benchmark_manifest.json`

| 关键字段 | 含义 |
|----------|------|
| `schema_version` | 固定 `synthoseis_lite_v1` |
| `global_seed` | 全局随机种子 |
| `config_summary` | 关键的生成配置参数快照 |
| `file_hashes` | 所有输出文件的 SHA-256 |
| `accepted_realizations` / `rejected_realizations` | 接受/拒绝计数 |

### 其他生成输出

- `frequency_probe_results.csv`：每个探针变体的频率、振幅、相位、横向形状和 RMS。
- `probe_frequency_catalog.csv`：探针频率的选择理由和噪声等效参考。
- `seismic_variant_results.csv`：每个地震变体（噪声、增益、相移等）的参数。
- `scenario_catalog.csv`：所有场景的定义和场条件接受状态。
- `generation_qc.csv`：每次生成尝试的 QC 指标和拒绝原因。
- `section_geometry_qc.csv`：场条件截面的横向层位支撑状态。

## 旁路 · evaluate_synthoseis_lite.py

评估阶段产出模型无关的基线报告卡。schema 为 `synthoseis_lite_report_v1`。

### `model_sample_metrics.csv`

| 关键字段 | 含义 |
|----------|------|
| `sample_id` | 样本标识 |
| `baseline_id` | `lfm_ideal` / `lfm_controlled_degraded` / `oracle_target` |
| `bias` / `mae` / `rmse` / `nrmse` / `corr` | 预测 vs 真值的回归指标 |
| `target_rms` / `prediction_rms` | 真值和预测的 RMS |
| `status` | `ok` / `failed` |

### `model_probe_metrics.csv`

| 关键字段 | 含义 |
|----------|------|
| `sample_id` | 探针样本标识 |
| `baseline_id` | 基线标识 |
| `probe_frequency_hz` / `probe_phase` / `probe_amplitude_multiplier` | 探针参数 |
| `paired_zero_sample_id` | 配对的零振幅探针 |
| `probe_metric_semantics` | `absolute_zero_or_unpaired_probe_error` / `paired_probe_increment_error` |
| `bias` / `mae` / `rmse` / `nrmse` / `corr` | 绝对或增量误差指标 |

### `model_geometry_metrics.csv`

按 `(baseline_id, suite, geometry_family)` 聚合 base 样本的
`n_samples` / `mean_rmse` / `mean_nrmse` / `median_corr`。

### `model_report_card.json`

包含 `baseline_aggregate` 和 `probe_aggregate` 两个聚合段，以及各基线语义说明。
`oracle_target` 的 RMSE 应为 0（pipeline 自检）。

### `evaluation_summary.json`

包含 benchmark 文件 SHA-256、输出文件列表及各文件 SHA-256，确保评估可复现。

## 旁路 · ginn_v2.py

模型消融训练与评估。训练产物被第八步 R0 消费，汇总产物供人工审阅。

### `model_run_manifest.json`

消费者：第八步 R0 `real_field_zero_shot.py`

训练完成时写出。R0 从此文件读取模型元数据和标准化参数，不扫描目录猜测。

| 关键字段 | 含义 |
|----------|------|
| `schema_version` | 固定 `ginn_v2_model_run_v1` |
| `model_id` | 模型架构标识，来自注册的 10 个 ID 之一 |
| `model_role` | `lateral` 或 `no_lateral`，决定 R0 输出子目录名。训练时自动从 `model_id` 推断，也可显式指定 |
| `benchmark_dir` | 训练使用的合成基准目录 |
| `benchmark_hashes` | 合成基准三文件（`.h5` / `sample_index.csv` / `benchmark_manifest.json`）的 SHA-256 |
| `patch_spec` | 切块规格：`lateral_samples` / `twt_samples` / `lateral_stride` / `twt_stride` / `min_valid_fraction` |
| `normalization` | 训练集统计：seismic/LFM/target/delta 的 mean/std |
| `input_channels` | 固定 `["seismic", "lfm_controlled_degraded", "valid_mask_model"]` |
| `train_sample_kinds` | 训练使用的样本类别列表 |
| `loss` | 损失配置：`lambda_ai` / `lambda_physics` / `physics_loss_applied_sample_kinds` |
| `model_info` | 参数量、感受野（lateral/twt）、输入/输出通道数 |
| `checkpoint` | 模型权重文件路径 |
| `best_validation_loss` | 最佳校验损失 |
| `synthetic_gate_evidence` | 由 `stamp-gate` 盖章写入；R0 必检字段，缺失则拒绝推理 |

### `input_reference_stats.json`

消费者：第八步 R0

当 R0 的 `seismic_value_transform` 非 identity 时，从此文件读取合成训练集地震的参考统计量，用于将真实地震变换到训练时的值域。

| 关键字段 | 含义 |
|----------|------|
| `stats` | 合成训练集地震的统计量（mean/std/P99 等） |
| `sampling` | 统计所基于的样本信息 |
| `file` | 本文件路径 |
| `sha256` | 本文件 SHA-256 |

## 07 · real_field_lfm.py

第七步产物供第八步 R0 消费。schema 为 `real_field_lfm_v1`。

### `real_field_lfm.npz`

消费者：`real_field_zero_shot.py`

| 数组 | 含义 |
|------|------|
| `log_ai` | 三维 `log(AI)` 低频模型，shape `[n_inline, n_xline, n_twt]` |
| `valid_mask_model` | 目标层内且有效处为 true，R0 的权威有效边界 |
| `lfm_support_mask` | 离控制井足够近处为 true，仅用于空间外推风险 QC |
| `distance_to_control` | 每个网格点到最近控制井的距离，仅作 QC |
| `a_field` | 趋势截距参数场，shape `[n_inline, n_xline]` |
| `b_field` | 趋势斜率参数场，shape `[n_inline, n_xline]` |
| `ilines` / `xlines` / `samples` | 坐标轴 |
| `metadata_json` | schema 版本、值域、层位名、来源路径、地震体 SHA-256 |

重建公式：`log_ai = a_field + b_field * (2*u - 1)`，其中 `u` 是顶底层位间的归一化坐标。

### `well_trend_controls.csv`

每口井的趋势拟合结果，一行一井。

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `a` / `b` | Huber 回归拟合的趋势系数 |
| `representative_x_m` / `representative_y_m` | 代表位置的真实 XY 坐标 |
| `representative_inline` / `representative_xline` | 代表位置的 inline/xline |
| `n_fit_samples` | 目标窗内有效 TWT 样点数 |
| `residual_rms` | 趋势拟合的残差 RMS |
| `status` | `ok` 或拒绝原因 |

### `parameter_field_qc.csv`

a/b 参数场的空间建模 QC，两行（一行 a、一行 b）。

| 关键字段 | 含义 |
|----------|------|
| `parameter` | `a` 或 `b` |
| `n_controls` | 参与克里金的控制井数 |
| `range_hint_m` | 从最近邻距离中位数估计的 range hint |
| `variance_p50` / `variance_p95` | 克里金方差的分位数 |
| `distance_to_control_p50_m` / `distance_to_control_p95_m` | 离最近控制井距离的分位数 |
| `outside_control_hull_fraction` | 网格在控制点凸包外的比例 |
| `variogram` / `nugget` | 变差函数模型和块金值 |

### `internal_horizon_continuity_qc.csv`

逐中间层位检查 LFM 在层位上下样点间是否存在非物理跃变。

### `horizon_qc.csv`

层位有效率、厚度统计、交叉道数、超出 TDT 支持的井数。

### `real_field_lfm_summary.json`

| 关键字段 | 含义 |
|----------|------|
| `schema_version` | 固定 `real_field_lfm_v1` |
| `status` | `ok` / `warning` / `insufficient_control_wells` |
| `control_wells` | 接受/拒绝/总数 |
| `lfm_stats` | 时间差分 RMS、每道时间标准差、横向标准差等 |
| `source_runs` | 上游第四步和第一步来源路径 |
| `outputs` | 所有输出文件路径 |

## 08 R0 · real_field_zero_shot.py

R0 产物供 R1 消费。schema 为 `real_field_zero_shot_summary_v1`。

### `predictions.npz`（每个模型子目录）

消费者：`real_field_forward_diagnostic.py`

| 数组 | 含义 |
|------|------|
| `stitched_pred_log_ai` | 拼接后的最终预测 `log(AI)` |
| `pred_delta_vs_lfm` | 预测与低频模型的差值 |
| `lfm_input` | 输入的低频模型 |
| `seismic_input` | 变换后的地震输入 |
| `valid_mask_model` | 有效掩码 |
| `stitching_weight` | patch 拼接权重 |
| `ilines` / `xlines` / `twt_s` | 坐标轴 |

### `real_field_zero_shot_summary.json`

消费者：`real_field_forward_diagnostic.py`

| 关键字段 | 含义 |
|----------|------|
| `schema_version` | 固定 `real_field_zero_shot_summary_v1` |
| `status` | 固定 `needs_forward_diagnostic`，表示尚未正演验证 |
| `mode` | `volume` 或 `section` |
| `source_runs` | 上游第五步、第七步来源路径 |
| `axis_contract` | 坐标轴范围、采样间隔 |
| `mask_contract` | 有效掩码比例和总数 |
| `boundary_contract` | 侵蚀和锥度参数 |
| `source_file_sha256` | 输入文件（地震体、LFM、子波等）的 SHA-256 |
| `models` | 逐模型的推理元数据、标准化参数、输出路径 |
| `outputs` | 所有输出文件路径 |
| `wavelet_sha256` | 使用的全局子波 SHA-256 |

### `model_input_qc.csv`

真实工区输入在送入模型前的标准化分布检查。

| 关键字段 | 含义 |
|----------|------|
| `input` | 输入通道名（seismic / lfm） |
| `fraction_abs_normalized_gt_3` / `fraction_abs_normalized_gt_5` | 标准化后超过 3σ / 5σ 的样本比例 |

### `real_field_spectral_qc.csv`

每个模型 × 每个频带的预测差值能量和可观测性证据联表。每行带 `observability_evidence_status` / `dominant_evidence_status` / `operator_support_summary` / `detectability_ratio_p25_range` 等第六步证据字段。

### `lateral_difference_band_qc.csv`

两模型预测差值的逐频带能量分析。若高频零空间带能量占比超阈值，标记 `lateral_difference_concentrated_in_nullspace`。

## 08 R1 · real_field_forward_diagnostic.py

R1 是当前流程的最终闭环。schema 为 `real_field_forward_diagnostic_summary_v1`。

### `forward_diagnostic_metrics.csv`

每个阻抗输入 × 子波场景一行。

| 关键字段 | 含义 |
|----------|------|
| `model_role` | `lfm_only` / `zero_shot_no_lateral` / `zero_shot_lateral` |
| `source_role` | 阻抗来源标识 |
| `forward_operator_id` | 正演算子标识，约定文档字符串 |
| `reflectivity_hang_point` | 反射系数挂点约定 |
| `residual_corr_raw` / `residual_corr_scaled` | 无缩放 / 带正尺度优化的相关系数 |
| `residual_rms_raw` / `residual_rms_scaled` | 标准化后的残差 RMS |
| `scale_positive` / `scale_status` | 正约束最小二乘缩放因子和状态 |

### `well_forward_diagnostic.csv`

逐井 × 逐角色（filtered_las / lfm_input / lateral / no_lateral）一行。包含四组指标：

| 指标组 | 关键字段 |
|--------|---------|
| 波阻抗闭环 | `well_ai_rmse` / `well_ai_bias` / `well_ai_corr` / `well_ai_n_valid` |
| 频带拆分 | `well_ai_<band>_rmse` / `well_ai_<band>_corr` / `pred_delta_<band>_rms` |
| 波形匹配 | `waveform_residual_corr_scaled` / `waveform_residual_rms_scaled` / `waveform_scale_status` |
| 井分类 | `status` / `classification` / `classification_explanation` |

`filtered_las` 行是参考基准（自比 RMSE=0、corr=1），`lfm_input` 行是模型必须击败的基线。

### `residual_decomposition.csv`

每个阻抗输入的相位和分数偏移扫描结果。`scan_type` 为 `phase` 或 `fractional_shift`，记录各扫描点的 `residual_rms_scaled`。若最优相位远离 0° 或最优偏移远离 0，残差可通过调整子波来部分消除。

### `wavelet_sensitivity.csv`

候选子波场景下的正演诊断指标，评估子波不确定性对结论的影响。

### `spatial_residual_qc.csv`

逐 inline / xline 的残差模式，检查是否存在系统性空间偏差。

### `forward_band_residual_qc.csv`

各频带的观测 RMS、合成 RMS、残差 RMS 和残差/观测比值。

### `ai_plausibility_qc.csv`

预测波阻抗和预测差值的全局统计分布、频带能量、与训练分布的对比。含 `real_to_synthetic_std_ratio` 字段用于判断真实工区预测的幅度是否在训练集见过的范围内。

### `well_ai_comparison_summary.csv`

逐井 × 逐角色的 LFM vs 模型对比和分类。

| 关键字段 | 含义 |
|----------|------|
| `rmse_delta_model_minus_lfm` | 模型 RMSE - LFM RMSE，负值表示改善 |
| `corr_delta_model_minus_lfm` | 模型 corr - LFM corr，正值表示改善 |
| `classification` | `model_improves_ai` / `shape_improves_bias_worse` / `bias_improves_shape_worse` / `waveform_good_ai_worse` / `filtered_las_weak_reference` / `mixed_or_insufficient` |

### `well_ai_band_comparison.csv`

逐井 × 逐角色 × 逐频带的 RMSE 和相关系数，用于定位模型在哪个频带改善或退化了井曲线匹配。

### `real_field_forward_diagnostic_summary.json`

| 关键字段 | 含义 |
|----------|------|
| `schema_version` | 固定 `real_field_forward_diagnostic_summary_v1` |
| `status` | `ok` |
| `forward_contract` | 正演约定：反射率公式、卷积约定、对齐方式、丢弃样点数 |
| `red_flags` | 自动红色告警列表，空列表表示无致命问题 |
| `recommended_next_state` | `return_to_input_preparation_or_synthetic_diagnostic`（有告警）或 `future_sparse_well_adapter_candidate`（无告警） |
| `wavelet_sha256` / `zero_shot_summary_sha256` | 输入文件校验值 |

## 第八步 L0 · l0_real_delta_anchor.py

L0 是独立的 synthetic + 真实井 delta-anchor 单井 holdout 研究验证，summary schema 为
`l0_real_delta_anchor_v1`。完整实验约束见
[`docs/spec/l0-real-delta-anchor-validation.md`](../spec/l0-real-delta-anchor-validation.md)。

### `l0_well_anchor_samples.csv`

模型无关的逐井逐 TWT 样点标签。每行一个井轨迹样点；`well_delta` 不落冗余列，由
`filtered_log_ai - lfm_log_ai` 唯一派生。

| 关键字段 | 含义 |
|----------|------|
| `well_name` / `sample_index` / `twt_s` | 井名、井内样点序号和 TWT 秒轴 |
| `inline` / `xline` / `x_m` / `y_m` | 三维轨迹采样位置 |
| `spatial_cluster_id` / `spatial_cluster_size` | 600 m 半径连通空间簇 |
| `filtered_log_ai` / `lfm_log_ai` | 真实井目标与第七步 LFM，均为 log(AI) |
| `valid_for_fit` / `valid_reason` | 是否进入 anchor loss 及唯一原因码 |
| `sampling_mode` / `sample_method` / `wellbore_class` | 三维采样与井型审计字段 |

### `l0_holdout_metrics.csv`

每个 held-out well × `control`/`anchor` 一行。保存 full-AI、delta、能量、梯度、分频和
waveform 指标，以及相对 paired control 的 gain、good-well 守门状态和三类井 QC 图路径。

### `l0_holdout_summary.csv`

每个被排除井一行，`is_primary_holdout=true` 标记配置的正式 holdout 井。正式判定只允许
读取该行；同簇自动排除井只作辅助诊断。

### `l0_training_history.csv` / `l0_anchor_sampling_qc.csv`

前者逐 run × epoch 保存 synthetic/anchor/total loss、梯度尺度和覆盖计数；后者逐
run × epoch × cluster × well 保存抽样次数。Held-out 井必须显式记录且
`selected_count == 0`。

### `l0_synthetic_preservation.csv` / `l0_decision_table.csv`

前者逐 run × frozen synthetic scope 保存 RMSE/NRMSE/corr 及 warning/catastrophic 守门结果；
后者逐条保存正式判定规则、观测值、阈值和布尔结果。
