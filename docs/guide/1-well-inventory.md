# 01 井资产盘点

`well_inventory.py` 是工作流的第一步，只做一件事：**盘点你手头有哪些井，各自具备哪些进入后续流程的资产条件。**

---

## 快速开始

```bash
python scripts/well_inventory.py
python scripts/well_inventory.py --config experiments/my_project.yaml
python scripts/well_inventory.py --output-dir /tmp/inventory_test
```

不带参数运行时，脚本读取 `experiments/common.yaml`，在 `scripts/output/well_inventory_<timestamp>/` 下写出四份文件。

---

## 配置参考

脚本从共享配置的 `well_inventory` 段读取参数。所有路径均相对于 `data_root`（顶层配置，默认 `data`）。

```yaml
well_inventory:
  well_heads_file: raw/well_heads          # Petrel 井头导出文件
  las_dir: all_well_las                     # LAS 文件目录
  well_trace_dir: all_well_trace            # 井轨迹文件目录（只查存在，不解析）
  well_tops_file: raw/well_tops             # Petrel 井分层导出文件
  time_depth_dir: time_depth_table          # 时深表目录（只查存在，不解析列名）
  seismic:
    file: <path-to-seismic>                 # SEG-Y 或 ZGY 文件
    type: zgy                               # "segy" 或 "zgy"
    # 以下仅 SEG-Y 需要
    iline: <byte-location>
    xline: <byte-location>
    istep: <step>
    xstep: <step>
  near_survey_threshold_m: 500.0
  vertical_bottom_offset_threshold_m: 30.0
  platform_cluster_threshold_m: 12.5
  dense_well_neighbor_threshold_m: 150.0
```

**数据资产的预期格式：**

| 输入 | 格式要求 |
|------|----------|
| 井头文件 | Petrel `BEGIN HEADER ... END HEADER` 文本，必须包含 `Name`、`Surface X`、`Surface Y`、`Bottom hole X`、`Bottom hole Y`、`Well datum value` 列 |
| LAS 目录 | 文件名 stem 即为井名，扩展名 `.las` |
| 井轨迹目录 | 文件名 stem 即为井名，不检查扩展名；本脚本仅检查文件是否存在 |
| 井分层文件 | Petrel 格式，必须包含 `Well`、`Surface`、`MD` 列 |
| 时深表目录 | 文件名 stem 即为井名；本脚本仅检查文件是否存在 |

**井名匹配规则：** 所有资产通过文件名 stem 或记录中的 `Name`/`Well` 字段做大小写不敏感匹配。`A1`、`a1`、`A1.las`、`A1.petrel_dev` 被视为同一口井；井轨迹文件也可以没有扩展名。大小写冲突会直接报错。名为 `nan`、`none`、`null` 或空白的记录会被跳过。

---

## 阈值设置指南

### `near_survey_threshold_m`（默认 500.0）

工区外的井，如果井口到工区边界的最近距离 ≤ 此值，标记为 `near_outside`；否则为 `outside`。

这个阈值取决于你的工区边缘地质情况。如果工区边界附近有可靠的地震数据覆盖，可以放宽；如果边界处地震质量差，保持默认即可。

### `vertical_bottom_offset_threshold_m`（默认 30.0）

用井头底孔坐标做直井/斜井初分的阈值。井口到底孔水平距离 ≤ 此值为 `vertical`，否则为 `deviated`。

**注意：这是初分，仅依据 Petrel 井头导出中的 `Bottom hole X/Y` 字段。** Petrel 导出的底孔坐标精度有限（可能四舍五入到整米）。如果后续发现大量假阳性直井或假阴性斜井，调整此值；最终应以第四步轨迹文件复核后的井型为准。

### `platform_cluster_threshold_m`（默认 10.0）

同平台井的判定半径。井口距离 ≤ 此值的井会通过连通分量算法聚合成 `well_clusters.csv`。同一平台的多口斜井通常是有意设计的钻井方案，不属于"冲突"。

同时，同平台井对 **不会** 作为两两近邻警告导出到 `well_neighbor_pairs.csv`，这避免了每次看到上百对虚假冲突时，还要手动分辨哪些来自同一个平台。

### `dense_well_neighbor_threshold_m`（默认 150.0）

密井网近邻的统计半径。仅用于统计计数（写入 `run_summary.json`），不会全量导出为 CSV。

`nominal_bin_spacing_m` 是脚本根据真实 XY 道中心坐标估算出的名义道间距，写在 `run_summary.json` 的 `bin_spacing_m.nominal_bin_spacing_m` 中。它不是配置里的线号步长，而是从相邻 inline、xline 道中心距离中取出的稳健代表值，单位为米。

建议将默认值理解为 `max(3 × nominal_bin_spacing_m, 150)`。实际道间距会由脚本写入 `run_summary.json`，方便交叉验证。如果此阈值过大，`dense_neighbor_pair_count` 会膨胀到几百甚至上千，失去可读性。

---

## 输出文件

脚本在 `<output_root>/well_inventory_<timestamp>/` 下生成四份文件：

### 1. `well_inventory.csv` — 主清单，一井一行

| 字段 | 含义 |
|------|------|
| `well_name` | 统一井名 |
| `has_well_head` | 井头文件是否包含 |
| `has_las` | LAS 文件是否存在 |
| `has_well_trace` | 井轨迹文件是否存在 |
| `has_time_depth` | 时深表文件是否存在 |
| `has_well_tops` | 井分层是否包含该井 |
| `surface_x`, `surface_y` | 井口 XY 坐标（无井头时为空） |
| `bottom_x`, `bottom_y` | 底孔 XY 坐标（无井头时为空） |
| `kb_m` | Kelly Bushing 高程（无井头时为空） |
| `inline_float`, `xline_float` | 井口投影到工区的浮点线号；工区外为空 |
| `nearest_inline`, `nearest_xline` | 井口吸附到的最近线号；工区外为空 |
| `survey_position` | `inside`、`near_outside`、`outside`、`invalid_xy` |
| `distance_to_survey_m` | 井口到工区边界最近 XY 距离；工区内也为正数，计算失败时为 null |
| `bottom_offset_m` | 井口到底孔水平距离；坐标缺失时为 null |
| `wellbore_class` | `vertical`、`deviated`、`unknown`（基于井头底孔坐标的初分） |
| `inventory_status` | `usable_for_las_screen`、`head_only`、`las_only`、`unknown` |
| `reasons` | 分号分隔的警告/失败原因 |

### 2. `well_neighbor_pairs.csv` — 高风险井口同道冲突

**不输出全部近邻对。** 只导出井口落在同一最近地震道、但**不是**同平台的井对。同平台同道被视为正常情况，不在此输出。

| 字段 | 含义 |
|------|------|
| `well_a`, `well_b` | 井名 |
| `distance_m` | 井口 XY 距离 |
| `same_surface_nearest_trace` | 始终为 true（因为只导出同道对） |
| `same_surface_platform` | 始终为 false（同平台对已过滤） |
| `class_pair` | 例如 `vertical/deviated` |
| `risk` | 当前固定为 `same_trace_conflict` |

### 3. `well_clusters.csv` — 同平台井分组

井口距离 ≤ `platform_cluster_threshold_m` 的井通过连通分量算法聚合。

| 字段 | 含义 |
|------|------|
| `cluster_id` | 平台编号，格式 `platform_001` |
| `well_name` | 井名 |
| `surface_x`, `surface_y` | 井口 XY 坐标 |
| `wellbore_class` | 初分井型 |
| `survey_position` | 井口工区位置 |
| `nearest_inline`, `nearest_xline` | 井口吸附最近线号 |
| `cluster_size` | 该平台井数 |

### 4. `run_summary.json` — 输入、阈值、统计摘要

包含：脚本名、配置路径、所有输入文件路径（相对于仓库根目录）、四项阈值、工区几何信息、道间距（inline/xline/nominal，单位米）、工区 footprint 四角 XY，以及全部统计计数和名单。

关键的 `neighbor_summary` 段：

| 字段 | 含义 |
|------|------|
| `valid_surface_well_count` | 有效井口坐标井数（参与近邻计算） |
| `dense_neighbor_pair_count` | ≤ `dense_well_neighbor_threshold_m` 的井对总数 |
| `same_surface_nearest_trace_pair_count` | 井口吸附到同一最近道的井对数 |
| `same_platform_pair_count` | 井口 ≤ `platform_cluster_threshold_m` 的井对数 |
| `same_trace_platform_pair_count` | 同时满足同道和同平台的井对数 |
| `exported_neighbor_pair_count` | 写入 `well_neighbor_pairs.csv` 的硬冲突数 |
| `platform_cluster_count` | 平台分组数 |
| `platform_cluster_well_count` | 参与平台分组的井数 |

---

## 如何阅读结果

### 第一步：看 `run_summary.json` 的顶层计数

```
well_count: 103
asset_counts: {well_heads: 103, las: 102, well_trace: 103, time_depth: 10, ...}
survey_position_counts: {inside: 61, outside: 42}
wellbore_class_counts: {deviated: 85, vertical: 18}
```

这几行直接回答：有多少井？缺哪些资产？多少在工区内？多少看起来是斜井？

### 第二步：如果有 `las_only` 井 → 补井头

`las_only` 表示有 LAS 曲线但没有井头记录，缺 XY 坐标。这类井无法进入任何后续井震流程。检查是否漏导了井头，或者 LAS 文件名与井头 Name 是否存在拼写差异。

### 第三步：如果有 `head_only` 井 → 确定是否需要

`head_only` 表示有井头但没有 LAS 文件。这类井保留了空间信息，但不能进入第二步的曲线筛选。检查是 LAS 文件缺失，还是文件名不匹配。

### 第四步：看 `neighbor_summary`

- `dense_neighbor_pair_count` 很大（>300）→ 说明井网很密，但这不直接意味着有问题。
- `exported_neighbor_pair_count` > 0 → 存在井口落在同一地震道但不在同一平台的井对。查看 `well_neighbor_pairs.csv` 了解详情。这一对井在后续 auto-tie 和 well constraints 中可能需要特殊处理。
- `platform_cluster_count` 告诉你工区内有多少个集中钻井平台。每个 cluster 的 `cluster_size` 可以帮助判断后续是否需要对同平台井做代表井选择或加权处理。

### 第五步：看 `well_inventory.csv` 的具体列

- 按 `survey_position` 筛选 `inside`，按 `inventory_status` 筛选 `usable_for_las_screen`——这是进入第二步的候选井。
- 关注 `wellbore_class == deviated` 且 `has_well_trace == false` 的井——斜井但没有轨迹文件，第四步无法走斜井路径。
- 关注 `wellbore_class == unknown` 的井——井头坐标缺失或无效。
- `reasons` 列汇总了每口井的所有警告标签（`no_time_depth`、`outside_survey`、`invalid_surface_xy` 等），方便快速筛出有问题的井。
