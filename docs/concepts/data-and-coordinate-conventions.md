# 数据与坐标约定

## TWT（两程时）

- 时间域主链内部统一使用**正秒**。
- Petrel 导出可能是负毫秒，读取时通过 `abs(twt_ms) / 1000.0` 归一化。
- `cup.well.td.load_petrel_time_depth_table()` 是 Petrel 时深表进入项目内 TDT 的统一入口。

## TVDSS 与 Petrel `Z`

- Petrel checkshot adapter 使用 `abs(Z)` 表示向下为正的 TVDSS/depth below MSL。
- `WellTrajectory.tvdss_m = tvd_kb_m - kb_m`，更接近 signed TVDSS（井口附近可能为负）。
- `export_vertical_tdt_to_petrel_checkshots` 写 `-abs(tdt.tvdss)`。
- **斜井路径混用前必须显式统一口径**，不要在第四步隐式混用两种约定。

## 地震工区几何

- `geometry["inline_step"]` / `geometry["xline_step"]` 是线号步长，**不是 XY 米制间距**，也不保证为 1。
- 最近道吸附使用轴吸附公式：`line_min + round((line_float - line_min) / line_step) * line_step`。
- 物理距离必须通过 `SurveyContext.line_to_coord()` 或 `cup.seismic.spatial` 的 XY 网格计算。

## 井名规范化

- 项目内统一使用 `normalize_well_name(name)` 作为匹配键：`str(name).strip().casefold()`。
- 所有文件查找、DataFrame join、lookup dict 均使用规范化键。

## 核心 CSV 契约

以下 CSV 文件是脚本之间的数据契约接口：

### `well_inventory.csv`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `has_well_head` / `has_las` / `has_well_trace` / `has_time_depth` / `has_well_tops` | 资产存在性 |
| `surface_x` / `surface_y` | 井口坐标 |
| `bottom_x` / `bottom_y` | 底孔坐标 |
| `survey_position` | 工区位置（inside / near_outside / outside） |
| `wellbore_class` | 井型初分（vertical / deviated / unknown） |
| `inventory_status` | 可用状态 |

### `well_curve_screen.csv`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `screen_status` | passed / partial / failed |
| `has_p_sonic` / `has_density` / `has_caliper` | 可用性 |
| `primary_p_sonic` / `primary_density` | 主曲线 mnemonic |
| `exported_las` | 筛选后 LAS 路径 |

### `well_preprocess_status.csv`

| 关键字段 | 含义 |
|----------|------|
| `well_name` | 井名 |
| `preprocess_status` | passed / failed |
| `usable_p_sonic` / `usable_density` / `usable_caliper` | 预处理后可用性 |
| `preprocessed_las` | 预处理后 LAS 路径 |

### `well_trajectory_qc.csv`

| 关键字段 | 含义 |
|----------|------|
| `trajectory_status` | passed / warning / failed / missing |
| `wellbore_class_initial` | 井头初分 |
| `wellbore_class_qc` | 轨迹复核后井型 |
| `class_changed` | 初分与复核是否不一致 |

### `well_tie_plan.csv`

| 关键字段 | 含义 |
|----------|------|
| `route` | 分配的路由类型 |
| `route_status` | planned / skipped_disabled / rejected |
| `wellbore_class_qc` | 供路由决策的井型 |

## 单位约定

| 物理量 | 单位 | 说明 |
|--------|------|------|
| TWT | s | 正秒，内部统一 |
| 深度 MD / TVDKB | m | 向下为正 |
| Petrel TDT 的 TVDSS / `Z` 适配值 | m | 当前使用 `abs(Z)`，向下为正 |
| `WellTrajectory.tvdss_m` | m | `tvd_kb_m - kb_m`，可为负；斜井路径混用前必须显式转换 |
| 速度 (Vp / Vs) | m/s | 内部统一 |
| 密度 (Rho) | g/cm³ | 内部统一 |
| KB 高程 | m | 正值 |
| 频率 | Hz | |
| 采样间隔 (dt / dz) | s / m | |
