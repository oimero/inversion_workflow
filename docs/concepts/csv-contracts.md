# 核心 CSV 契约

这些 CSV 是当前稳定工作流的跨脚本契约。稳定生产链截止第五步；
第五步产物暂时没有正式下游消费者。

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

第五步是当前稳定工作流终点。以下产物由研究入口
`forward_observability.py` 显式消费；该入口不是编号生产步骤。

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

## Research Gate · forward_observability.py

所有表使用 schema `forward_observability_v1`。运行必须显式指定第三、四、五步目录，
不搜索 latest，也不自动回退。

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

## Research Gate · synthoseis_lite.py (calibrate)

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

## Research Gate · synthoseis_lite.py (generate)

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

## Research Gate · evaluate_synthoseis_lite.py

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
