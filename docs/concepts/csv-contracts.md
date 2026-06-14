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

第五步是当前稳定工作流终点。以下产物可作为后续 research gate 的输入，
但在新接口确定前没有正式脚本消费者。

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

任何后续研究脚本若要消费第五步产物，必须先在本文定义新契约，再实现读取逻辑。
