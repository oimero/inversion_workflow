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

第六步写出的点级井约束事实表。它是低频 anchor、高频井监督、高频统计和 LFM 点级控制点的共同来源。默认训练材料只使用直井点；斜井点保留在事实表中用于审计和 LFM 控制点候选，不默认进入 GINN 或 enhance 训练。

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

## `well_anchor_points.csv` / `well_anchor_conflicts.csv`

`well_anchor_points.csv` 是 `well_constraint_points.csv` 中允许进入 GINN 低频 anchor 的子集（第一版默认只含直井）。`well_anchor_conflicts.csv` 记录同一 `(flat_idx, sample_index)` 上有多条井约束被聚合前的差异。

| 关键字段 | 含义 |
|----------|------|
| `flat_idx` / `sample_index` | 发生冲突的地震道和采样点 |
| `n_points` | 冲突点数量 |
| `well_names` / `sources` | 参与冲突的井和空间来源 |
| `min_value` / `max_value` / `range_value` | 被审计目标值的范围 |
| `strategy` | 当前聚合策略 |
| `point_rows_json` | 冲突点的原始井名、位置、目标值和权重 |

## `well_anchor_trace_summary.csv`

每条进入 GINN 低频 anchor 的受控道一行，用于审计道级覆盖。

| 关键字段 | 含义 |
|----------|------|
| `flat_idx` | 受控地震道编号 |
| `well_names` | 约束该道的井名，多井用分号连接 |
| `sources` | 空间来源类型 |
| `sample_count` | 该道上有效锚点样点数 |
| `weight_min` / `weight_mean` / `weight_max` | 该道锚点权重的最小、均值和最大值 |
| `inline` / `xline` | 该道的线号 |

## `well_high_supervision_qc.csv`

逐井入选结果和质量控制摘要，每口候选井一行。`status=selected` 表示该井进入点级事实表和 LFM 候选；若该井为斜井，在默认配置下仍不会进入 `well_high_supervision_time.npz` 和 `well_high_stats_*`。

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `status` | `selected` / `rejected` / `failed` |
| `route` | 第四步标定路径 |
| `batch_corr` / `batch_nmae` | 第五步批量合成指标 |
| `control_point_count` | 有效样点数 |
| `high_supervision_eligible` | 是否实际进入 enhance 高频监督和高频统计；默认斜井为 false |
| `high_supervision_point_count` | 实际进入 enhance 高频监督和高频统计的样点数 |
| `invalid_point_count` / `invalid_point_fraction` | 无效样点数及比例 |
| `unique_trace_count` | 样点覆盖的唯一道数；斜井应大于 1 |
| `reasons` | 分号分隔的拒绝或失败原因 |
| `frequency_split_qc_trace_path` | 分频前后曲线数值文件路径 |
| `frequency_split_qc_figure_path` | 分频 QC 图路径 |

## `well_high_stats_by_layer.csv`

每层一行，记录该层的高频统计特征和可靠度。后续 enhance 合成器用这些统计驱动分层样本生成。

| 关键字段 | 含义 |
|----------|------|
| `zone_name` | 层段名 |
| `well_count` / `sample_count` / `event_count` | 该层的有效井数、样点数和事件数 |
| `reliability` | 0–1 可靠度，由井数、样点数和事件数综合决定 |
| `alpha_to_layer` | 收缩因子 α，越高越相信本层自身统计 |
| `event_density_per_sample` / `event_density_per_second` | 事件密度（每样点 / 每秒） |
| `amplitude_rms` / `amplitude_p10` / `amplitude_p50` / `amplitude_p90` / `amplitude_abs_p95` | 高频残余振幅分布 |
| `run_length_p50` / `run_length_p90` | 正负状态持续样点数的分位数 |
| `transition_matrix_json` | 三状态转移矩阵（正/静/负）的计数和概率 |

## `frequency_split_diagnostics.csv`

分频诊断时每口井、每个候选截止频率一行。脚本用该 cutoff 下的低通井 AI 正演合成记录，并与第四步保存的井旁地震道比较。

| 关键字段 | 含义 |
|----------|------|
| `well_name` / `route` | 参与诊断的井和第四步标定路径 |
| `cutoff_hz` | 候选截止频率 |
| `status` | ok / failed / manual |
| `corr` | 低通 AI 正演合成记录与井旁地震道的相关系数 |
| `nmae` | 低通 AI 正演合成记录与井旁地震道的归一化绝对误差 |
| `scale` | 最小二乘缩放系数 |
| `n_eval_samples` | 参与正演匹配评价的样点数 |
| `wavelet_file` | 第五步最终全局子波路径 |
| `reason` | 失败原因，仅 failed 行有值 |

## `frequency_split_aggregate.csv`

分频诊断的多井聚合表，每个候选截止频率一行。

| 关键字段 | 含义 |
|----------|------|
| `cutoff_hz` | 候选截止频率 |
| `n_wells` | 该 cutoff 下成功参与评价的井数 |
| `median_corr` / `mean_corr` | 多井相关系数聚合 |
| `p25_corr` / `p75_corr` | 相关系数四分位范围 |
| `median_nmae` / `mean_nmae` | 多井误差聚合 |
| `p25_nmae` / `p75_nmae` | 误差四分位范围 |
| `median_scale` | 最小二乘缩放系数中位数 |
| `median_n_eval_samples` | 参与评价样点数中位数 |

## `well_high_motif_manifest.csv`

可选真实高频 motif patch 的索引表。第一版只写表头（占位），不生成对应的 motif 数据包。

| 关键字段 | 含义 |
|----------|------|
| `motif_id` | motif 编号 |
| `well_name` / `zone_name` | 来源井和层段 |
| `start_twt_s` / `end_twt_s` | motif 片段的时间范围 |
| `quality_tag` | 质量标签 |
| `reason` | 选择原因 |

## `lfm_control_points.csv`

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

`inline_float`、`xline_float`、`twt_s` 是规范坐标。第六步输出的是点级低频控制事实，不按单井、层段或顺层切片聚合；第七步 LFM 根据自己的 `modeling.n_slices` 决定如何分配切片、聚合重复控制点和插值建模。`flat_idx` / `sample_index` 作为派生字段写出，便于 QC 和调试，但它们依赖当前地震几何与采样轴，不能作为跨步骤主键。

## `lfm_log_control_points.csv`

字段与 `lfm_control_points.csv` 基本一致，但控制值列为：

| 关键字段 | 含义 |
|----------|------|
| `log_ai` | 第六步分频后的低频 log-AI 控制值 |

这张表用于 log 域建模、审计或后续方法扩展。当前第七步 AI-LFM 主路径读取 `lfm_control_points.csv`。

## `lfm_control_qc.csv`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `status` | selected / rejected / failed |
| `route` | 第四步标定路径 |
| `batch_corr` / `batch_nmae` | 第五步全局子波批量合成指标 |
| `control_point_count` | 第六步点级事实中的有效控制样点数量 |
| `lfm_control_point_count` | 写入 `lfm_control_points.csv` 的 AI 控制点数量 |
| `lfm_log_control_point_count` | 写入 `lfm_log_control_points.csv` 的 log-AI 控制点数量 |
| `invalid_point_count` / `invalid_point_fraction` | 因目标层、轨迹、TDT 或曲线问题被丢弃的点 |
| `unique_trace_count` | 控制点覆盖的唯一 trace 数；斜井通常大于 1 |
| `reasons` | 拒绝或失败原因 |
