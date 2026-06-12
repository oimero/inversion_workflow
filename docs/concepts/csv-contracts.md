# 核心 CSV 契约

这些 CSV 是脚本之间的数据契约。下游脚本读取上游结果时，应按这里的字段语义判断，不要凭字段名猜测。

每个 CSV 列在其生产者脚本的章节下，`→` 后标注下游消费者。标题中标记「诊断」的 CSV 不被任何脚本自动读取，仅供人工审阅。

---

## 01 · well_inventory.py

### `well_inventory.csv` — 核心契约

→ well_screen.py、well_auto_tie.py、well_trajectory.py

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `has_well_head` / `has_las` / `has_well_trace` / `has_time_depth` / `has_well_tops` | 资产存在性 |
| `surface_x` / `surface_y` | 井口坐标 |
| `bottom_x` / `bottom_y` | 底孔坐标 |
| `survey_position` | 工区位置：inside / near_outside / outside / invalid_xy |
| `wellbore_class` | 井型初分：vertical / deviated / unknown |
| `inventory_status` | 资产状态，不等同于最终筛选候选状态 |

---

## 02 · well_screen.py

### `well_screen.csv` — 核心契约

→ well_preprocess.py、well_auto_tie.py

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `las_file` | 原始 LAS 路径 |
| `screen_status` | passed / partial / failed |
| `has_p_sonic` / `has_density` / `has_caliper` | 第二步识别到的类别可用性 |
| `primary_p_sonic` / `primary_density` / `primary_caliper` | 主曲线 mnemonic |
| `selected_curve_count` | 选中曲线数量，partial 井也保留真实数量 |
| `exported_las` | 第二步瘦身 LAS；仅 passed 井应有值 |

### `las_curve_inventory.csv` — 核心契约

→ well_preprocess.py

一条 LAS 曲线一行，不是一口井一行。第三步用它复原每口井的曲线类别、primary 选择和原始单位。

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `mnemonic` | 原始 LAS 曲线名，必须能在对应 LAS 中精确回查 |
| `unit` | LAS 头中的原始单位 |
| `category` | 工作流标准类别，如 `p_sonic` / `density` / `caliper`，或 `unclassified` / `ambiguous` / `disabled` |
| `is_primary` | 是否为该类别进入下游的代表曲线 |
| `classification_source` | 分类来源：`override` / `mnemonic_rule` / `unclassified` |
| `confidence` | 分类置信度；规则命中通常为 1.0，歧义为 0.0 |
| `notes` | override、歧义或禁用原因 |

---

## 03 · well_preprocess.py

### `well_preprocess_status.csv` — 核心契约

→ well_auto_tie.py

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `preprocess_status` | passed / failed |
| `usable_p_sonic` / `usable_density` / `usable_caliper` | 清洗和复核后的曲线可用性 |
| `final_p_sonic` / `final_density` / `final_caliper` | 最终标准 mnemonic |
| `preprocessed_las` | 第三步预处理 LAS；仅 passed 井应有值，固定包含 `DT_USM`、`RHO_GCC` 和缺失值严格传播的全频 `AI` |

---

## 旁路 · well_trajectory.py

### `well_trajectory.csv` — 核心契约

→ well_auto_tie.py

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `trajectory_status` | passed / warning / failed / missing |
| `wellbore_class_initial` | 第一阶段井头初分 |
| `wellbore_class_qc` | 轨迹复核后井型 |
| `class_changed` | 初分与复核是否不一致 |
| `surface_survey_position` / `bottom_survey_position` | 井口/井底相对工区位置 |
| `trajectory_inside_fraction` | 轨迹点位于工区内的比例 |

---

## 04 · well_auto_tie.py

### `well_tie_plan.csv` — 核心契约

→ wavelet_generation.py、well_constraints.py

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

### `well_tie_metrics.csv` — 核心契约

→ wavelet_generation.py、well_constraints.py、dynamic_gain.py

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `route` | 实际执行的第四步标定路由 |
| `tie_status` | success / failed |
| `optimized_tdt_file` | 第四步细标定后的内部 TDT CSV |
| `filtered_las_file` | 第四步用最优滤波参数导出的 LAS，固定包含重新计算的 `DT_USM`、`RHO_GCC`、`AI`；第五步仍以基础曲线构造自己的 `LogSet` |
| `seismic_trace_file` | 第四步保存的井旁或轨迹地震道 |
| `optimized_trace_sample_plan_file` | 斜井细标定后按 optimized TDT 重新生成的样点级落道计划；直井为空 |
| `joint_observed_fraction` | 原目标窗内 DT/RHO 原始联合观测比例 |
| `short_gap_filled_samples` / `long_gap_count` / `longest_long_gap_s` | `10 ms` 缺口规则的 QC |
| `continuous_tie_window_start_s` / `continuous_tie_window_end_s` | 实际用于 auto-tie 的最长连续联合有效段 |
| `continuous_tie_sample_count` | 该连续段按地震时间采样间隔计算的样点数，用于 `min_tie_samples` 判据 |
| `continuous_tie_log_sample_count` | 同一连续段在原始 MD 井曲线轴上的样点数，仅作 QC |
| `tie_window_clipped_for_log_gap` | 标定窗是否因长缺口被裁剪 |

`optimized_trace_sample_plan_file` 指向的 CSV 使用 `trace_plan_index` 表示当前计划内从 0 开始的局部行号。它在裁剪或重建计划后会重新编号，不能解释为地震体全局样点索引；下游必须使用同一行的 `twt_s`、`inline_float`、`xline_float`。

### `wavelet_inventory.csv` — 核心契约

→ wavelet_generation.py

第四步写出的候选子波清单，第五步只从这里挑选候选子波，不再扫描第四步目录下的所有子波文件。

| 关键字段 | 含义 |
|----------|------|
| `source_well` | 子波来源井 |
| `route` | 来源井第四步标定路由 |
| `wavelet_file` | 候选子波 CSV 路径，repo-relative |
| `dt_s` | 子波采样间隔，单位 s |
| `n_samples` | 子波样点数 |
| `tie_corr` / `tie_nmae` | 第四步该井 auto-tie 的最终匹配指标 |
| `usable_as_candidate` | 第五步是否允许作为候选；第五步还会再做长度、中心、能量和采样间隔 QC |
| `reasons` | 不可用或需要审计的原因 |

---

## 05 · wavelet_generation.py

### `selected_wavelet.csv` — 核心契约

→ well_constraints.py、deterministic_inversion.py、dynamic_gain.py、ginn_train.py

第五步选出的全局子波。第六步分频诊断和第八步 GINN 正演都应读取这条子波，而不是回退到单井候选子波。

| 关键字段 | 含义 |
|----------|------|
| `time_s` | 子波时间轴，单位 s，中心应在 0 附近 |
| `amplitude` | 已归一化的子波振幅 |

### `batch_synthetic_metrics.csv` — 核心契约

→ well_constraints.py

第五步用全局子波在所有评测井上重新正演后的逐井指标。第六步用它筛选控制井和计算井约束权重。

| 关键字段 | 含义 |
|----------|------|
| `candidate_wavelet` | 评测使用的全局子波名称 |
| `source_well` | 全局子波来源；共识子波可为 `optimized_consensus` |
| `eval_well` | 被评测井名 |
| `route` | 被评测井第四步标定路由 |
| `corr` / `nmae` | 全局子波合成记录与第四步保存地震道的匹配指标 |
| `scale` | 单井最小二乘振幅缩放系数，仅用于评价和 QC |
| `n_eval_samples` | 参与评价的样点数 |
| `spatial_cluster_id` / `spatial_cluster_size` | 密井平台去偏用的空间簇信息 |
| `status` | `ok` / `failed` |
| `reasons` | 失败原因 |

---

## 06 · well_constraints.py

第六步是 CSV 输出最多的步骤。它同时产出点级事实表、分频诊断、冲突报告和高频统计。

### `well_constraint_points.csv` — 核心契约

→ ginn_inversion.py（井 QC）、deterministic_inversion.py（井 QC）

点级井约束事实表。它是低频 anchor、高频井监督、高频统计和 LFM 点级控制点的共同来源。默认训练材料只使用直井点；斜井点保留在事实表中用于审计和 LFM 控制点候选，不默认进入 GINN 或 enhance 训练。

| 关键字段 | 含义 |
|----------|------|
| `well_name` / `route` | 来源井和第四步标定路径 |
| `source` | 空间来源：vertical_trace / deviated_trajectory |
| `anchor_eligible` | 是否允许进入 GINN 低频 anchor |
| `twt_s` / `md_m` | 点级样本所在 TWT 和 MD |
| `x_m` / `y_m` | 点级样本平面坐标 |
| `inline_float` / `xline_float` | 投影到工区后的浮点线号 |
| `flat_idx` / `seismic_sample_index` | 依赖当前地震几何和全局采样轴的派生索引，仅用于 bundle 构建和 QC |
| `zone_name` / `u_in_zone` | 所属层段和层内比例位置 |
| `reference_ai` / `reference_log_ai` | 轻量清洗并低通到 `f_reference` 的参考井曲线 |
| `lfm_ai` / `lfm_log_ai` | 低通到 `f_lfm` 的 LFM 控制 |
| `ginn_target_ai` / `ginn_target_log_ai` | 低通到 `f_ginn` 的 GINN 井 anchor 目标 |
| `ginn_band_log_ai` | `ginn_target_log_ai - lfm_log_ai` |
| `enhance_residual_log_ai` | `reference_log_ai - ginn_target_log_ai` |
| `observed_well_sample` | 是否为可进入监督、统计和控制点的真实井样点 |
| `short_gap_interpolated` / `hampel_conditioned` | 是否仅作为滤波支撑的插值或异常替换样点 |
| `frequency_band_valid` | 三条低通曲线在该点是否都有效 |
| `weight` | 由第五步批量合成质量等因素得到的约束权重 |

`inline_float`、`xline_float`、`twt_s` 是空间事实的规范坐标；`flat_idx` / `seismic_sample_index` 只能在同一地震几何和全局采样轴内解释。任何体采样入口必须由 `twt_s` 在当前采样轴上重新求最近索引，并用 `seismic_sample_index` 做交叉校验，不能直接信任派生索引。

### `well_constraint_qc.csv` — 核心契约

→ lfm_precomputed.py

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
| `frequency_band_qc_trace_path` | reference/LFM/GINN/enhance 三频带数值文件路径 |
| `frequency_band_qc_figure_path` | 三频带 QC 图路径 |
| `frequency_band_qc_plot_start_s` / `frequency_band_qc_plot_end_s` | 三频带 QC 图实际显示的该井目的层 TWT 范围；完整数值 CSV 仍保留全地震时间轴 |

### `lfm_control_points.csv` — 核心契约

→ lfm_precomputed.py

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

`inline_float`、`xline_float`、`twt_s` 是规范坐标。第六步输出的是点级低频控制事实，不按单井、层段或顺层切片聚合；第七步 LFM 根据自己的 `modeling.n_slices` 决定如何分配切片、聚合重复控制点和插值建模。`flat_idx` / `seismic_sample_index` 作为派生字段写出，便于 QC 和调试，但它们依赖当前地震几何与全局采样轴，不能作为跨步骤主键。

### `ginn_cutoff_diagnostics.csv` — 诊断

分频诊断时每口井、每个候选截止频率一行。脚本用该 cutoff 下的低通井 AI 正演合成记录，并与第四步保存的井旁地震道比较。**仅供人工审阅**，不被任何下游脚本自动读取。

| 关键字段 | 含义 |
|----------|------|
| `well_name` / `route` | 参与诊断的井和第四步标定路径 |
| `cutoff_hz` | 候选 GINN 截止频率 |
| `status` | ok / failed / manual |
| `corr` | 低通 AI 正演合成记录与井旁地震道的相关系数 |
| `nmae` | 低通 AI 正演合成记录与井旁地震道的归一化绝对误差 |
| `scale` | 最小二乘缩放系数 |
| `n_eval_samples` | 参与正演匹配评价的样点数 |
| `wavelet_file` | 第五步最终全局子波路径 |
| `reason` | 失败原因，仅 failed 行有值 |

### `ginn_cutoff_cluster_aggregate.csv` / `ginn_cutoff_aggregate.csv` — 诊断

前者先在空间簇内取井指标中位数；后者再跨空间簇取中位数，避免密井平台按井数重复投票。**仅供人工审阅**。

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

### `well_anchor_conflicts.csv` — 诊断（可选）

只有 GINN 低频 anchor 中存在同一 `(flat_idx, seismic_sample_index)` 上多条井约束时才写出，记录被聚合前的差异。**仅供人工审阅**。

| 关键字段 | 含义 |
|----------|------|
| `flat_idx` / `seismic_sample_index` | 发生冲突的地震道和全局采样点 |
| `n_points` | 冲突点数量 |
| `well_names` / `sources` | 参与冲突的井和空间来源 |
| `min_value` / `max_value` / `range_value` | 被审计目标值的范围 |
| `strategy` | 当前聚合策略 |
| `point_rows_json` | 冲突点的原始井名、位置、目标值和权重 |

### `well_high_supervision_conflicts.csv` — 诊断（可选）

只有 enhance 监督点中存在同一 `(flat_idx, seismic_sample_index)` 上多条井约束时才写出。字段与 `well_anchor_conflicts.csv` 一致，但审计目标值为 `enhance_residual_log_ai`。**仅供人工审阅**。

### `well_high_stats_by_layer.csv` — 诊断 / enhance 输入

→ enhance 合成器（深度域）

每层一行，记录该层的高频统计特征和可靠度。后续 enhance 合成器用这些统计驱动分层样本生成。时间域工作流当前不自动消费此文件。

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

---

## 06 → 07/08/enhance：NPZ 与 JSON 数据包

第六步除了 CSV 外，还产出以下 NPZ 和 JSON 文件，是第七、八步和 enhance 的正式输入。详细 schema 见[数据与单位约定](data-and-coordinate-conventions.md)。

```
第六步 well_constraints.py
  ├─ lfm_control_points.csv         → 第七步 lfm_precomputed.py（点级 AI 控制）
  ├─ log_ai_anchor_time.npz         → 第八步 ginn_train.py（ginn_target_log_ai 井约束）
  ├─ well_high_supervision_time.npz → enhance 训练（enhance_residual_log_ai，schema v2）
  └─ well_high_stats_*.json/csv     → enhance 合成器（统计驱动生成，默认只含直井）
```

每个下游步骤只读取第六步的输出，不应再直接访问第四步的 LAS、时深表或井轨迹。

---

## 07 · lfm_precomputed.py

第七步不产出 CSV 契约。它读取第六步的 `lfm_control_points.csv` 和 `well_constraint_qc.csv`，产出 `ai_lfm_time.npz` 供第八步 GINN 训练使用。

---

## 旁路 · dynamic_gain.py

Dynamic gain 旁路，置于第七步之后、第八步之前。读取第四步 `well_tie_metrics.csv` 和第五步 `selected_wavelet.csv`，产出 `dynamic_gain.npz` 和若干诊断 CSV（`dynamic_gain_samples.csv`、`dynamic_gain_well_medians.csv` 等）。诊断 CSV 仅供人工审阅，`dynamic_gain.npz` 可由第八步按配置读取。

---

## 08 · ginn_train.py

第八步不读写 CSV。它通过 YAML 配置读取第七步的 `ai_lfm_time.npz`、第五步的 `selected_wavelet.csv`（经 `ginn.config` 解析）和可选的第六步 anchor NPZ，产出 GINN checkpoint。

---

## 09 · ginn_inversion.py

第九步读取第八步 checkpoint 执行反演，产出 `well_qc_metrics.csv` 和逐井 `well_qc_*.csv` 用于井 QC 评估。这些 QC CSV 是脚本内部产物，不进入其他主链步骤的输入。

---

## 旁路 · deterministic_inversion.py

确定性反演旁路，置于第七步之后、第八步之前。读取第五步 `selected_wavelet.csv` 和第六步 `well_constraint_points.csv`（用于井 QC），产出 `well_qc_metrics.csv` 和确定性反演体。不产出新的跨步骤 CSV 契约。
