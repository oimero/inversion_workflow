# 井轨迹 QC

`well_trajectory.py` 读取井轨迹文件，生成可信的井几何事实，供井震标定、低频建模和井约束流程使用。

---

## 快速开始

```bash
python scripts/well_trajectory.py
python scripts/well_trajectory.py --config experiments/my_project.yaml
python scripts/well_trajectory.py --output-dir scripts/output/<trajectory-qc-run>
```

不带参数时，脚本自动发现最新的井资产盘点产物，在 `<output_root>/well_trajectory_<timestamp>/` 下写出结果。

## 运行前需要什么

| 输入 | 用途 |
|------|------|
| `well_inventory.csv` | 提供井名、井头坐标、KB 和井型初分 |
| Petrel 井轨迹目录 | 读取每口井的 MD/XY/Z/TVD 轨迹点 |
| 地震体及其线号/道号几何 | 判断轨迹点是否在工区内 |

如果工区全部是直井、且不打算复核轨迹，这一步可以跳过。但建议至少跑一次，因为井头文件里的底孔坐标可能不准。

---

## 配置参考

```yaml
assets:
  well_trace_dir: <well-trajectory-dir>

seismic:
  file: <seismic-volume-file>
  type: zgy
  domain: time
  zgy_inline_chunk_size: 16

well_trajectory:
  classification:
    vertical_max_offset_m: 30.0
    min_deviated_max_offset_m: 30.0
    surface_xy_tolerance_m: 2.0
    kb_tolerance_m: 0.5
    z_tvd_tolerance_m: 0.1

  survey_qc:
    allow_partial_outside: true
```

### `source_runs`

默认自动接上最新一次井资产盘点结果。复现实验时可按需加入 `source_runs.well_inventory_dir` 固定输入。

### `well_trace_dir`

井轨迹文件目录来自顶层 `assets.well_trace_dir`。文件按 stem 匹配井名（不要求特定扩展名）。

### `classification`

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `vertical_max_offset_m` | 30.0 | 轨迹整体偏移很小时，复核为直井 |
| `min_deviated_max_offset_m` | 30.0 | 轨迹整体偏移明显时，复核为斜井 |
| `surface_xy_tolerance_m` | 2.0 | 井口 XY 最大允许偏差 |
| `kb_tolerance_m` | 0.5 | KB 基准面最大允许偏差 |
| `z_tvd_tolerance_m` | 0.1 | Z 与 KB-TVD 残差最大允许值 |

`vertical_max_offset_m` 和 `min_deviated_max_offset_m` 可以设成不同值，中间形成未判定区间。两个值相等时，所有有效轨迹都会被明确分成直井或斜井。

### `survey_qc`

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `allow_partial_outside` | true | 轨迹部分在工区外时，true=警告，false=硬失败 |

工区几何 QC 固定启用；逐点 CSV 默认写入 `trajectory_points` 子目录。`output.write_trajectory_points` 默认值为 `true`，设为 `false` 时省略逐点文件。

---

## 输入格式

### 井轨迹文件格式

脚本期望 Petrel 导出的空白分隔文本，文件头包含 `#` 注释行，数据部分首列为 `MD`。必需列：

| 列 | 含义 |
|---|------|
| `MD` | 测深，从 KB 起算，向下为正 |
| `X`, `Y` | 轨迹点平面坐标 |
| `Z` | 高程/深度坐标 |
| `TVD` | 真垂深，从 KB 起算，向下为正 |

可选列（缺失时填空值，不影响解析）：`DX`、`DY`、`AZIM`、`INCL`、`DLS`。

轨迹里的 `Z` 和 `TVD` 应满足 `Z ≈ KB - TVD`。脚本计算残差 `Z - (KB - TVD)`，超过 `z_tvd_tolerance_m` 时发出警告。

### TVDSS 口径

脚本内部按 `TVDSS = TVD(KB) - KB` 计算，后续时深转换沿用同一米制定义。

---

## 脚本在做什么

脚本把井资产清单、Petrel 轨迹文本和时间域地震体几何合并为逐井、逐点的质量控制结果。

1. **解析轨迹并建立深度坐标。** 读取测深、平面坐标、高程坐标和真垂深，筛除必要数据中的非有限行，检查有效点数量和测深递增性；依据 KB 建立相对海平面垂深：

   \[
   TVDSS = TVD(KB) - KB
   \]

2. **校验井名与几何口径。** 将轨迹文件头与井资产清单的井名、井口坐标和 KB 进行核对，并按高程与真垂深残差识别异常：

   \[
   r_Z = Z - (KB - TVD)
   \]

   井名不一致形成失败原因；坐标、基准面、深度值或被丢弃数据行的异常形成质量警告。

3. **复核井型并映射工区。** 以轨迹首点为参考，计算最大水平偏移

   \[
   d_{\max} = \max_i\sqrt{(X_i-X_0)^2+(Y_i-Y_0)^2}
   \]

   再依据阈值划分直井、斜井或待判定状态。对每个轨迹点转换时间域工区的浮点线号和道号，统计井口、井底及全轨迹的工区内外状态，并依据配置的部分出界策略形成警告或失败状态。

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

`output.write_trajectory_points` 为 `true` 时写出。每行包含该轨迹点的测深、TVD、TVDSS、Z、XY、DX/DY、井斜角、方位角、DLS、浮点线号、最近线号、工区内外。

### `failed_trajectories.csv`

`trajectory_status` 为 `failed` 或 `missing` 的井子集，方便快速排查。

### `run_summary.json`

输入路径、配置阈值、各状态计数、井型分布、初分与复核不一致的井数。

---

## 如何阅读结果

### 第一步：看终端输出

```
Wrote trajectory QC for <N> wells to <output-dir> ({'passed': <N>, 'warning': <N>, 'failed': <N>, 'missing': <N>}).
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

打开 `trajectory_points/<well>.csv`，查看 `incl_deg` 列的最大值和 `x_m`/`y_m` 随测深的变化趋势，核对斜井几何。
