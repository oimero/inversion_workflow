# 旁路 井轨迹 QC

`well_trajectory.py` 读取井轨迹文件，生成可信的井几何事实，供井震标定、低频建模和井约束流程使用。

---

## 快速开始

```bash
python scripts/well_trajectory.py
python scripts/well_trajectory.py --config experiments/my_project.yaml
python scripts/well_trajectory.py --output-dir /tmp/traj_test
```

不带参数时，脚本自动发现最新的井资产盘点产物，在 `<output_root>/well_trajectory_<timestamp>/` 下写出结果。

## 运行前需要什么

| 输入 | 用途 |
|------|------|
| `well_inventory.csv` | 提供井名、井头坐标、KB 和井型初分 |
| Petrel 井轨迹目录 | 读取每口井的 MD/XY/Z/TVD 轨迹点 |
| 时间域地震体 | 可选；启用 survey QC 时用于判断轨迹点是否在工区内 |

轨迹 QC 是第四步井震标定的前置条件：只有知道井的真实几何，才能决定它走直井路径还是斜井路径。但它不依赖第二步和第三步的 LAS 处理，所以可以在第一步之后任意时间运行。

如果工区全部是直井、且不打算复核轨迹，这一步可以跳过。但建议至少跑一次，因为井头文件里的底孔坐标可能不准。

---

## 配置参考

```yaml
well_trajectory:
  source_runs:
    mode: latest
    well_inventory_dir: null

  well_trace_dir: all_well_trace

  seismic:
    file: raw/your-seismic.zgy
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

### `source_runs`

默认接上最新一次井资产盘点结果。复现实验时，在 `well_inventory_dir` 填入某次第一步输出目录即可固定输入；`mode` 目前只支持 `latest`。

### `well_trace_dir`

井轨迹文件目录。文件按 stem 匹配井名（不要求特定扩展名）。

### `classification`

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `vertical_max_offset_m` | 30.0 | 轨迹整体偏移很小时，复核为直井 |
| `min_deviated_max_offset_m` | 30.0 | 轨迹整体偏移明显时，复核为斜井 |
| `surface_xy_tolerance_m` | 2.0 | 井口 XY 最大允许偏差 |
| `kb_tolerance_m` | 0.5 | KB 基准面最大允许偏差 |
| `z_tvd_tolerance_m` | 0.1 | Z 与 KB-TVD 残差最大允许值 |

`vertical_max_offset_m` 和 `min_deviated_max_offset_m` 可以设成不同值，中间留出“不确定”灰色区间。两个值相等时，所有有效轨迹都会被明确分成直井或斜井。

### `survey_qc`

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `enabled` | true | 是否计算轨迹点的 inline/xline 和工区内外 |
| `allow_partial_outside` | true | 轨迹部分在工区外时，true=警告，false=硬失败 |

### `output`

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `write_trajectory_points` | true | 是否写出逐点 CSV |
| `sampled_trajectory_dir` | trajectory_points | 逐点 CSV 的子目录名 |

---

## 脚本在做什么

对每口井，依次做三件事：

### 1. 解析轨迹文件

读取 Petrel 导出的井轨迹文本，提取 `MD`、`X`、`Y`、`Z`、`TVD` 五列必要数据，以及可选的 `DX`、`DY`、`AZIM`、`INCL`、`DLS`。

解析失败的硬条件：

- 文件缺少 `MD/X/Y/Z/TVD` 任一列
- 有效轨迹点少于 2 个
- MD 不单调递增
- XY 全为空或非有限值
- 文件头缺少 KB 基准面

以上任一触发，该井 `trajectory_status = failed`，不进入后续检查。

### 2. 口径一致性 QC

用轨迹文件头数据和第一步井头数据互相校验：

| 检查项 | 不合格时 |
|--------|---------|
| 文件头井名与文件名 stem 不一致 | 硬失败 |
| 文件头缺少井名 | 警告 |
| 井口 XY 与第一步井头偏差超过阈值 | 警告 |
| KB 与第一步井头偏差超过阈值 | 警告 |
| `Z` 与 `KB - TVD` 偏差超过阈值 | 警告 |
| MD 或 TVD 出现负值 | 警告 |
| 必要列中存在非有限值的行被丢弃 | 警告 |

警告不阻塞流程，但会写入 `qc_flags` 列供第四步路由时判断。

### 3. 井型复核 + 工区落点

第一步用井头底孔坐标初分直井/斜井，这里用真实轨迹复核：

- **用最大水平偏移判断，不用井口-底孔偏移。** 有些井中段偏斜明显但底孔又回到井口附近，井口-底孔偏移会漏判。
- 轨迹整体偏移很小 → 复核为直井
- 轨迹整体偏移明显 → 复核为斜井
- 落在两个阈值之间 → 暂时标记为不确定

如果配置了地震工区，还会把每个轨迹点投影到 inline/xline，统计：

- 井口和井底在工区内还是工区外
- 全部轨迹点中有多大比例在工区内
- 部分轨迹出界的井，按 `allow_partial_outside` 决定是警告还是硬失败

---

### 井轨迹文件格式

脚本期望 Petrel 导出的空白分隔文本，文件头包含 `#` 注释行，数据部分首列为 `MD`。必需列：

| 列 | 含义 |
|---|------|
| `MD` | 测深，从 KB 起算，向下为正 |
| `X`, `Y` | 轨迹点平面坐标 |
| `Z` | 高程/深度坐标 |
| `TVD` | 真垂深，从 KB 起算，向下为正 |

可选列（缺失时填 NaN，不影响解析）：`DX`、`DY`、`AZIM`、`INCL`、`DLS`。

轨迹里的 `Z` 和 `TVD` 应满足 `Z ≈ KB - TVD`。脚本计算残差 `Z - (KB - TVD)`，超过 `z_tvd_tolerance_m` 时发出警告。

### TVDSS 口径

脚本内部按 `tvdss_m = tvd_kb_m - kb_m` 计算。这个换算只在本模块和后续时深转换模块中实现，后续脚本不要自己散写这份逻辑。

---

## 核心输出文件

所有文件在 `<output_root>/well_trajectory_<timestamp>/` 下：

### `well_trajectory.csv` — 每井一行

| 字段 | 含义 |
|------|------|
| `well_name` | 井名 |
| `trajectory_file` | 轨迹文件路径 |
| `trajectory_status` | `passed` / `warning` / `failed` / `missing` |
| `wellbore_class_initial` | 第一步井头底孔坐标初分 |
| `wellbore_class_qc` | 轨迹复核后的井型 |
| `class_changed` | 初分和复核是否不同 |
| `point_count` | 轨迹点数量 |
| `md_min_m` / `md_max_m` | MD 范围 |
| `tvd_kb_min_m` / `tvd_kb_max_m` | TVD 范围 |
| `tvdss_min_m` / `tvdss_max_m` | TVDSS 范围 |
| `surface_x_m` / `surface_y_m` | 轨迹井口 XY |
| `bottom_x_m` / `bottom_y_m` | 轨迹末点 XY |
| `surface_to_bottom_offset_m` | 井口到末点水平偏移 |
| `max_horizontal_offset_m` | 相对井口最大水平偏移 |
| `max_incl_deg` | 最大井斜角 |
| `max_dls` | 最大狗腿严重度 |
| `surface_survey_position` | 井口相对工区位置 |
| `bottom_survey_position` | 井底相对工区位置 |
| `trajectory_inside_fraction` | 轨迹点在工区内的比例 |
| `trajectory_inside_sample_count` | 工区内轨迹点数 |
| `trajectory_outside_sample_count` | 工区外轨迹点数 |
| `qc_flags` | 分号分隔的警告标签 |
| `reasons` | 失败或拒绝原因 |

### `trajectory_points/<well>.csv` — 每口井逐轨迹点

仅当 `output.write_trajectory_points: true` 时写出。每行包含该轨迹点的 MD、TVD、TVDSS、Z、XY、DX/DY、井斜角、方位角、DLS、浮点线号、最近线号、工区内外。

### `failed_trajectories.csv`

`trajectory_status` 为 `failed` 或 `missing` 的井子集，方便快速排查。

### `run_summary.json`

输入路径、配置阈值、各状态计数、井型分布、初分与复核不一致的井数。

---

## 如何阅读结果

### 第一步：看终端输出

```
Wrote trajectory QC for 103 wells to ... ({'passed': 80, 'warning': 18, 'failed': 3, 'missing': 2}).
```

`failed` + `missing` 越少越好。`warning` 井需要检查 `qc_flags` 判断是否影响后续路由。

### 第二步：看 class_changed

在 `well_trajectory.csv` 中筛选 `class_changed == True`：

- 初分为直井、复核为斜井 → 井头底孔坐标低估了实际偏移，第四步应走斜井路径。
- 初分为斜井、复核为直井 → 可能是井头底孔坐标有误，也可能该井可以按直井处理。

这些变更直接影响第四步的路由决策。

### 第三步：看 qc_flags

警告标签的含义：

| flag | 含义 |
|------|------|
| `surface_xy_mismatch` | 轨迹文件头井口 XY 与井头不一致 |
| `kb_mismatch` | 轨迹文件头 KB 与井头不一致 |
| `z_tvd_inconsistent` | Z 与 KB-TVD 残差超限 |
| `invalid_depth_values` | MD 或 TVD 出现负值 |
| `invalid_required_rows_dropped` | 部分行因必要列为空被丢弃 |
| `missing_header_well_name` | 文件头没有井名 |
| `partial_outside_survey` | 轨迹部分在工区外 |

大多数警告不影响使用，但 `z_tvd_inconsistent` 值得优先排查——它意味着 Z 和 TVD 至少有一列不可信。

### 第四步：看部分出界的井

筛选 `trajectory_inside_fraction` 在 0.3-0.7 之间的井。井口可能在工区外但目标层段进入了工区，或反之。第四步是否接受这类井，取决于 auto-tie 配置。

### 第五步：抽查一口井的轨迹点

打开 `trajectory_points/<well>.csv`，看 `incl_deg` 列的最大值、`x_m`/`y_m` 随 MD 的变化趋势，对斜井形成直观印象。

---

## 留到第二轮

- 对同平台密井生成轨迹交叉/近距离诊断。
