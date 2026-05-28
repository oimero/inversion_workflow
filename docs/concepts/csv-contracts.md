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

## `well_curve_screen.csv`

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

## `well_trajectory_qc.csv`

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
