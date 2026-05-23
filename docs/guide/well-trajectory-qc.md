# 井轨迹 QC

`well_trajectory_qc.py` 是一个不带序号的旁路前置脚本。它不属于 LAS 曲线筛选和预处理主链，也不强制所有工区运行；它的任务是读取井轨迹文件，生成可信的井几何事实，供 `well_auto_tie.py`、后续 LFM 和井约束流程使用。

推荐依赖关系是：

```text
well_inventory
  ├── las_curve_screen ── log_preprocess ──┐
  └── well_trajectory_qc ──────────────────┤
                                            ↓
                                      well_auto_tie
```

也就是说，轨迹 QC 可以在第一步之后任意时间运行，只要在需要斜井路由、轨迹采样或井型复核之前完成即可。对于明确全直井、且不需要轨迹复核的工区，这一步可以跳过。

---

## 目标

`well_trajectory_qc.py` 回答四件事：

1. 哪些井的轨迹文件可以被正确解析。
2. Petrel 轨迹文件中的 `MD`、`TVD`、`Z`、`KB` 口径是否自洽。
3. 真实轨迹复核后的井型是什么：`vertical`、`deviated` 或 `unknown`。
4. 每口井是否具备进入第四步斜井路径、后续 LFM 或井约束的几何条件。

第一步 `well_inventory.py` 已经基于井头文件里的井口/底孔坐标给出 `wellbore_class`，但那只是初分。`well_trajectory_qc.py` 读取真实轨迹后产出的 `wellbore_class_qc` 才应该作为后续路由优先依据。

---

## 输入

- 第一阶段清单：`well_inventory.csv`。
- 井轨迹目录：`data/all_well_trace`。
- 地震体或工区：用于把轨迹点投影到 inline/xline，并检查轨迹点是否落在工区内。
- 可选：井头文件或第一步输出中的 `kb_m`、井口坐标，用于和轨迹文件头互相校验。

建议配置片段：

```yaml
well_trajectory_qc:
  source_runs:
    mode: latest
    well_inventory_dir: null

  inventory_file: null
  well_trace_dir: all_well_trace

  seismic:
    file: raw/obn-clipped-240-912-872-1544.zgy
    type: zgy

  classification:
    vertical_max_offset_m: 30.0
    min_deviated_max_offset_m: 30.0
    surface_xy_tolerance_m: 2.0
    kb_tolerance_m: 0.5
    z_tvd_tolerance_m: 0.1

  survey_qc:
    enabled: true
    allow_partial_outside: true

  output:
    write_trajectory_points: true
    sampled_trajectory_dir: trajectory_points
```

`source_runs.mode: latest` 表示默认自动寻找最新的 `well_inventory_<timestamp>` 输出。复现实验时可以显式填写 `inventory_file`。

---

## Petrel 井轨迹格式

当前数据目录是 `data/all_well_trace`。典型 Petrel 导出文件类似：

```text
# WELL TRACE FROM PETREL
# WELL NAME:              A1
# DEFINITIVE SURVEY:      MD Incl Azim survey 1
# WELL HEAD X-COORDINATE: 686352.08000000 (m)
# WELL HEAD Y-COORDINATE: 3217437.84000000 (m)
# WELL DATUM (KB, Kelly bushing, from MSL): 23.00000000 (m)
# MD AND TVD ARE REFERENCED (=0) AT WELL DATUM AND INCREASE DOWNWARDS
# DEPTH (Z, tvd_z) GIVEN IN m-UNITS
#================================================================================================================================
      MD            X            Y            Z           TVD           DX          DY          AZIM         INCL         DLS
#================================================================================================================================
 0.0000000000 686352.08000 3217437.8400 23.000000000 0.0000000000 ...
```

有效数据列为空白分隔：

| 列 | 含义 |
| --- | --- |
| `MD` | 测深，从 KB 起算，向下为正，单位 m |
| `X`, `Y` | 轨迹点平面坐标 |
| `Z` | Petrel 导出的高程/深度坐标；通常 `Z ~= KB - TVD` |
| `TVD` | 从 KB 起算的真垂深，向下为正，单位 m |
| `DX`, `DY` | 相对井口偏移，单位 m |
| `AZIM` | 方位角，单位度 |
| `INCL` | 井斜角，单位度 |
| `DLS` | 狗腿严重度 |

第一版不需要根据 `INCL/AZIM` 重新积分轨迹，应优先使用文件中已经给出的 `MD/X/Y/TVD`。`INCL/AZIM/DLS` 只作为 QC 字段保留。

---

## 核心数据模型

建议在 `src/cup/well/trajectory.py` 中新增项目内主模型：

```text
WellTrajectory
```

建议字段：

| 字段 | 含义 |
| --- | --- |
| `well_name` | 井名 |
| `md_m` | 测深 MD，单位 m |
| `tvd_kb_m` | 从 KB 起算的 TVD，向下为正，单位 m |
| `tvdss_m` | 项目内部 TVDSS 口径，单位 m |
| `z_m` | 原始 Petrel `Z` 列 |
| `x_m`, `y_m` | 轨迹点 XY 坐标 |
| `dx_m`, `dy_m` | 相对井口偏移 |
| `azim_deg`, `incl_deg`, `dls` | 轨迹 QC 字段 |
| `kb_m` | KB 高程 |
| `metadata` | 来源文件、文件头、QC 信息 |

建议方法：

| 方法 | 功能 |
| --- | --- |
| `from_petrel_trace(path)` | 读取 Petrel well trace txt |
| `to_wtie_wellpath()` | 转成 `wtie.processing.grid.WellPath`，仅作为 wtie Adapter |
| `with_inline_xline(survey)` | 用 `SurveyContext` 补充轨迹点浮点线号和最近道 |
| `position_at_md(md)` | 按 MD 插值得到 `x/y/tvdss` 等位置 |
| `representative_position(policy)` | 返回井口、井底、最大偏移点或目标层代表点 |

`WellTrajectory` 是项目内完整井轨迹模型；`wtie.grid.WellPath` 只表达 `MD/TVDSS/KB`，不能反过来主导项目设计。

### TVDSS 口径

项目内第一版沿用现有 checkshot/tdt 处理习惯：

```text
tvdss_m = tvd_kb_m - kb_m
```

读取 Petrel 轨迹时同时检查：

```text
z_m ~= kb_m - tvd_kb_m
```

这个换算必须集中在 `cup.well.trajectory` 或后续 `cup.well.depth_time` Adapter 中实现，脚本层不要临时散写 `tvdss = tvd - kb` 或 `tvdss = kb - z`。

注意：现有 Petrel 时深表导入/导出 Adapter 里仍有历史口径，第四步把已有时深表和 `WellTrajectory.to_wtie_wellpath()` 拼接前，必须在 `cup.well.depth_time` 中显式统一 TDT 的 TVDSS 口径。本脚本只产出轨迹侧的 `tvd_kb_m/tvdss_m/z_m` 事实和一致性 QC，不在这里静默修正旧 TDT。

---

## QC 规则

### 文件级 QC

| 检查 | 处理 |
| --- | --- |
| 文件无法解析 | `trajectory_status = failed` |
| 缺少 `MD/X/Y/TVD` 必要列 | `trajectory_status = failed` |
| 有效轨迹点少于 2 个 | `trajectory_status = failed` |
| `MD` 非单调递增 | `trajectory_status = failed` 或排序后写警告，第一版建议失败 |
| `X/Y` 全为空或非有限值 | `trajectory_status = failed` |

### 口径一致性 QC

| 检查 | 处理 |
| --- | --- |
| 文件头井名与文件名 stem 不一致 | `qc_flags += name_mismatch` |
| 文件头井口 XY 与第一步井头 XY 超过阈值 | `qc_flags += surface_xy_mismatch` |
| 文件头 KB 与第一步 `kb_m` 超过阈值 | `qc_flags += kb_mismatch` |
| `Z` 与 `KB - TVD` 偏差超过阈值 | `qc_flags += z_tvd_inconsistent` |
| `TVD` 或 `MD` 出现明显负值 | `qc_flags += invalid_depth_values` |

这些 QC flag 不一定导致失败，但第四步路由时应能读取并按配置决定是否拒绝。

### 井型复核

建议计算：

```text
surface_to_bottom_offset_m = hypot(x_last - x_first, y_last - y_first)
max_horizontal_offset_m = max(hypot(x_i - x_first, y_i - y_first))
```

井型判定第一版可以简单使用：

| 条件 | `wellbore_class_qc` |
| --- | --- |
| 轨迹解析失败 | `unknown` |
| `max_horizontal_offset_m <= vertical_max_offset_m` | `vertical` |
| `max_horizontal_offset_m > min_deviated_max_offset_m` | `deviated` |
| 其他 | `unknown` |

注意这里使用 `max_horizontal_offset_m`，不是只看井口到底孔偏移。某些轨迹可能中段偏移明显、底孔又回到井口附近，只看底孔会误判。

### 工区内外 QC

如果配置了 `survey_qc.enabled`，脚本应使用 `SurveyContext.coord_to_line()` 或后续 trace/index Adapter 计算每个轨迹点的：

- `inline_float`
- `xline_float`
- `nearest_inline`
- `nearest_xline`
- `survey_position`

这里必须遵守 `AGENTS.md` 的线号步长约定：`geometry["inline_step"]` / `xline_step` 是线号步长，不是 XY 米制距离；最近线号必须按轴吸附，不能直接 `round(inline_float)`。

对于斜井，井口可能在工区外，但目标层轨迹段进入工区；也可能井口在工区内，但深部轨迹出界。因此 `well_trajectory_qc.py` 不应该只输出一个简单的井口 `survey_position`，还应该统计：

| 字段 | 含义 |
| --- | --- |
| `trajectory_inside_fraction` | 全轨迹点中位于工区内的比例 |
| `trajectory_inside_sample_count` | 位于工区内的轨迹点数 |
| `trajectory_outside_sample_count` | 位于工区外的轨迹点数 |
| `surface_survey_position` | 井口位置相对工区 |
| `bottom_survey_position` | 井底位置相对工区 |

当 `allow_partial_outside: true` 时，部分轨迹在工区外的井会保留为非致命 warning，并在 `qc_flags` 中写入 `partial_outside_survey`。当该开关为 false 时，这类井会被本脚本标记为 failed。第四步仍应读取 inside/outside 统计，并按 route 决定是否接受。

---

## 输出

默认输出目录建议为：

```text
scripts/output/well_trajectory_qc_<timestamp>/
```

核心文件：

- `well_trajectory_qc.csv`：一井一行的轨迹 QC 和井型复核结果。
- `trajectory_points/*.csv`：可选，一口井一个轨迹点表，供人工抽查或后续复用。
- `failed_trajectories.csv`：解析失败或关键 QC 失败的井。
- `run_summary.json`：输入、配置、井型统计、失败统计。

### `well_trajectory_qc.csv`

建议字段：

| 字段 | 含义 |
| --- | --- |
| `well_name` | 统一井名 |
| `trajectory_file` | 轨迹文件路径 |
| `trajectory_status` | `passed`、`warning`、`failed`、`missing` |
| `wellbore_class_initial` | 第一阶段井头底孔坐标初分 |
| `wellbore_class_qc` | 轨迹复核井型 |
| `class_changed` | 初分和复核是否不同 |
| `point_count` | 轨迹点数量 |
| `md_min_m`, `md_max_m` | MD 范围 |
| `tvd_kb_min_m`, `tvd_kb_max_m` | TVD 范围 |
| `tvdss_min_m`, `tvdss_max_m` | TVDSS 范围 |
| `surface_x_m`, `surface_y_m` | 轨迹井口 XY |
| `bottom_x_m`, `bottom_y_m` | 轨迹末点 XY |
| `surface_to_bottom_offset_m` | 井口到轨迹末点水平偏移 |
| `max_horizontal_offset_m` | 相对井口最大水平偏移 |
| `max_incl_deg` | 最大井斜角 |
| `max_dls` | 最大狗腿严重度 |
| `surface_survey_position` | 井口相对工区位置 |
| `bottom_survey_position` | 井底相对工区位置 |
| `trajectory_inside_fraction` | 轨迹点在工区内比例 |
| `qc_flags` | 分号分隔的警告 |
| `reasons` | 失败或拒绝原因 |

### `trajectory_points/<well>.csv`

建议字段：

| 字段 | 含义 |
| --- | --- |
| `well_name` | 井名 |
| `sample_index` | 轨迹点序号 |
| `md_m` | MD |
| `tvd_kb_m` | TVD from KB |
| `tvdss_m` | 项目内部 TVDSS |
| `z_m` | 原始 Petrel Z |
| `x_m`, `y_m` | XY 坐标 |
| `dx_m`, `dy_m` | 相对井口偏移 |
| `azim_deg`, `incl_deg`, `dls` | 原始井斜字段 |
| `inline_float`, `xline_float` | 浮点线号 |
| `nearest_inline`, `nearest_xline` | 最近线号 |
| `survey_position` | 该轨迹点相对工区位置 |

点表可能较大，因此建议配置为可选输出；但第一轮开发时建议开启，方便人工抽查。

---

## 与其他步骤的关系

### 与 `well_inventory.py`

第一步只检查轨迹文件是否存在，并基于井头文件给出井型初分。它不解析轨迹文件。

`well_trajectory_qc.py` 读取第一步的主清单，主要复用：

- `well_name`
- `has_well_trace`
- `surface_x/surface_y`
- `bottom_x/bottom_y`
- `kb_m`
- `wellbore_class`
- `survey_position`

输出中的 `wellbore_class_initial` 应来自第一步，`wellbore_class_qc` 来自真实轨迹。

### 与 `las_curve_screen.py` / `log_preprocess.py`

LAS 筛选和曲线预处理不消费轨迹 QC 产物。它们可以和轨迹 QC 并行运行。

因此不要因为新增轨迹 QC 就重排已有脚本编号。`well_trajectory_qc.py` 是第四步之前的条件依赖，而不是第二、第三步的前置条件。

### 与 `well_auto_tie.py`

第四步路由应优先读取 `well_trajectory_qc.csv`：

- 对 `vertical_with_tdt`，如果没有轨迹 QC，可退回第一步 `wellbore_class_initial`。
- 对 `deviated_with_tdt`，必须要求 `trajectory_status in {passed, warning}` 且 `wellbore_class_qc = deviated`。
- 对轨迹 QC 失败的斜井，应拒绝进入斜井路径。
- 对 `class_changed = true` 的井，应写入 `well_tie_plan.csv`，方便人工复核。

第四步不应该重新实现 Petrel 轨迹解析；如果需要轨迹点，直接读取本步骤产物或调用 `cup.well.trajectory`。

### 与 `deviated-well-src-cup-refactor.md`

`deviated-well-src-cup-refactor.md` 保持跨步骤架构文档定位，不作为脚本操作指南。本文只落地其中的第一块能力：

```text
Petrel trace -> WellTrajectory -> trajectory QC
```

后续的 `WellSpatialSampleSet`、LFM 斜井控制点、GINN anchor 点聚合仍放在重构规划文档和对应步骤文档里展开。

---

## 第一版实现范围

第一版建议只做这些：

1. 解析 `data/all_well_trace` 下的 Petrel 轨迹文件。
2. 新增 `WellTrajectory` 数据模型。
3. 输出 `well_trajectory_qc.csv`。
4. 可选输出 `trajectory_points/*.csv`。
5. 用真实轨迹复核直井/斜井。
6. 做 `Z ~= KB - TVD`、井口 XY、KB、MD 单调等基础 QC。
7. 如果给定地震工区，计算轨迹点 inline/xline 和工区内外统计。

第一版暂不做：

- 根据 `INCL/AZIM` 重新积分轨迹。
- 把轨迹采样到 TWT 轴。
- 沿斜井轨迹读取地震道。
- 生成 LFM 或 GINN 约束点。
- 自动修正错误轨迹。

这些能力属于第四步、后续斜井重构或 `cup.seismic.trace_sampling` 的范围。
