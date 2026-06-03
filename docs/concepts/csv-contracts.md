# 核心 CSV 契约

这些 CSV 是脚本之间的数据契约。脚本读取上游结果时，应按这里的字段语义判断，不要凭字段名猜测。

## `well_inventory.csv`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `has_well_head` / `has_las` / `has_well_trace` / `has_time_depth` / `has_well_tops` | 资产存在性 |
| `surface_x` / `surface_y` | 井口坐标 |
| `bottom_x` / `bottom_y` | 底孔坐标 |
| `survey_position` | 工区位置：inside / near_outside / outside / invalid_xy |
| `wellbore_class` | 井型初分：vertical / deviated / unknown |
| `inventory_status` | 资产状态，不等同于最终筛选候选状态 |

## `well_screen.csv`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `las_file` | 原始 LAS 路径 |
| `screen_status` | passed / partial / failed |
| `has_p_sonic` / `has_density` / `has_caliper` | 第二步识别到的类别可用性 |
| `primary_p_sonic` / `primary_density` / `primary_caliper` | 主曲线 mnemonic |
| `selected_curve_count` | 选中曲线数量，partial 井也保留真实数量 |
| `exported_las` | 第二步瘦身 LAS；仅 passed 井应有值 |

## `well_preprocess_status.csv`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `preprocess_status` | passed / failed |
| `usable_p_sonic` / `usable_density` / `usable_caliper` | 清洗和复核后的曲线可用性 |
| `final_p_sonic` / `final_density` / `final_caliper` | 最终标准 mnemonic |
| `preprocessed_las` | 第三步预处理 LAS；仅 passed 井应有值 |

## `well_trajectory.csv`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `trajectory_status` | passed / warning / failed / missing |
| `wellbore_class_initial` | 第一阶段井头初分 |
| `wellbore_class_qc` | 轨迹复核后井型 |
| `class_changed` | 初分与复核是否不一致 |
| `surface_survey_position` / `bottom_survey_position` | 井口/井底相对工区位置 |
| `trajectory_inside_fraction` | 轨迹点位于工区内的比例 |

## `well_tie_plan.csv`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `route` | 路由类型 |
| `route_status` | planned / skipped_disabled / rejected |
| `wellbore_class_qc` | 供路由决策的井型 |
| `has_time_depth` / `has_well_trace` / `has_well_tops` | 路由相关资产存在性 |
| `usable_p_sonic` / `usable_density` | 路由所需曲线是否可用 |
| `input_las` | 第三步预处理 LAS 路径 |
| `time_depth_file` / `well_trace_file` | 时深表与井轨迹输入路径 |
| `surface_x` / `surface_y` | 井口平面坐标；第六步井约束和第七步 LFM 的直井控制点使用这两个字段定位井口 trace |
| `kb_m` | 井口补心高程，单位 m；供需要井口高程的后续步骤复用 |

## `well_tie_metrics.csv`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `route` | 实际执行的第四步标定路由 |
| `tie_status` | success / failed |
| `optimized_tdt_file` | 第四步细标定后的内部 TDT CSV |
| `filtered_las_file` | 第四步用最优滤波参数导出的 LAS；第五步从这里读取 `DT_USM`/`RHO_GCC` |
| `seismic_trace_file` | 第四步保存的井旁或轨迹地震道 |
| `optimized_trace_sample_plan_file` | 斜井细标定后按 optimized TDT 重新生成的样点级落道计划；直井为空 |

## `well_constraint_points.csv`

第六步写出的点级井约束事实表。它是低频 anchor、高频井监督、高频统计和 LFM 控制点的共同来源。

| 关键字段 | 含义 |
|----------|------|
| `well_name` / `route` | 来源井和第四步标定路径 |
| `source` | 空间来源：vertical_trace / deviated_trajectory |
| `anchor_eligible` | 是否允许进入 GINN 低频 anchor |
| `twt_s` / `md_m` | 点级样本所在 TWT 和 MD |
| `x_m` / `y_m` | 点级样本平面坐标 |
| `inline_float` / `xline_float` | 投影到工区后的浮点线号 |
| `flat_idx` / `sample_index` | 依赖当前地震几何的派生索引，仅用于 bundle 构建和 QC |
| `zone_name` / `u_in_zone` | 所属层段和层内比例位置 |
| `ai_full` / `log_ai_full` | 井上全频 AI 与 log-AI |
| `well_low_ai` / `well_low_log_ai` | 第六步分频后的低频井曲线 |
| `well_high_log_ai` | 全频 log-AI 减低频 log-AI 后的高频 residual |
| `weight` | 由第五步批量合成质量等因素得到的约束权重 |

`inline_float`、`xline_float`、`twt_s` 是空间事实的规范坐标；`flat_idx` / `sample_index` 只能在同一地震几何和采样轴内解释。

## `well_anchor_points.csv` / `well_high_supervision_conflicts.csv`

`well_anchor_points.csv` 是 `well_constraint_points.csv` 中允许进入 GINN 低频 anchor 的子集。`well_anchor_conflicts.csv` 和 `well_high_supervision_conflicts.csv` 记录同一 `(flat_idx, sample_index)` 上有多条井约束被聚合前的差异。

| 关键字段 | 含义 |
|----------|------|
| `flat_idx` / `sample_index` | 发生冲突的地震道和采样点 |
| `n_points` | 冲突点数量 |
| `well_names` / `sources` | 参与冲突的井和空间来源 |
| `min_value` / `max_value` / `range_value` | 被审计目标值的范围 |
| `strategy` | 当前聚合策略 |
| `point_rows_json` | 冲突点的原始井名、位置、目标值和权重 |

## `lfm_layer_control_points.csv`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 控制点来源井 |
| `route` | 第四步标定路径 |
| `source` | 控制点空间来源：vertical_trace / deviated_trajectory |
| `twt_s` / `md_m` | 控制点所在 TWT 和 MD |
| `x_m` / `y_m` | 控制点平面坐标 |
| `inline_float` / `xline_float` | 控制点投影到工区后的浮点线号 |
| `zone_name` / `u_in_zone` | 所属层段和层内比例位置 |
| `ai` | 第六步分频后的低频 AI 控制值 |
| `weight` | 第六步按井震匹配质量等因素计算的控制点权重 |

`inline_float`、`xline_float`、`twt_s` 是规范坐标。当前第六步输出的是按单井、层段和切片聚合后的代表控制点，第七步只消费它做 LFM 建模。`flat_idx` / `sample_index` 可以作为派生字段写出，便于 QC 和调试，但它们依赖当前地震几何与采样轴，不能作为跨步骤主键。

## `lfm_control_qc.csv`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `status` | selected / rejected / failed |
| `route` | 第四步标定路径 |
| `batch_corr` / `batch_nmae` | 第五步全局子波批量合成指标 |
| `control_point_count` | 进入 LFM 的有效控制点数量 |
| `invalid_point_count` / `invalid_point_fraction` | 因目标层、轨迹、TDT 或曲线问题被丢弃的点 |
| `unique_trace_count` | 控制点覆盖的唯一 trace 数；斜井通常大于 1 |
| `reasons` | 拒绝或失败原因 |
