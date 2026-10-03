# 01 井资产盘点

`well_inventory.py` 是工作流的第一步，只做一件事：**盘点你手头有哪些井，各自具备哪些进入后续流程的资产条件。**

---

## 快速开始

```bash
python scripts/well_inventory.py
python scripts/well_inventory.py --config experiments/<project>.yaml
python scripts/well_inventory.py --output-dir <OUTPUT_DIR>
```

不带参数运行时，脚本读取 `experiments/common/common.yaml`，在配置中的 `output_root/well_inventory_<timestamp>/` 下写出四份文件；未配置 `output_root` 时使用 `scripts/output`。

---

## 运行前需要什么

| 输入 | 用途 |
|------|------|
| Petrel 井头导出 | 井名、井口/底孔坐标、KB 高程 |
| LAS 目录 | 判断每口井是否有可进入第二步的曲线文件 |
| 井轨迹目录 | 判断是否存在轨迹文件；本步只查存在性 |
| 井分层文件 | 判断是否有后续标定/建模可用的井分层 |
| 时深表目录 | 判断每口井是否有 Petrel TDT |
| 地震体 | 解析工区几何，判断井口是否在工区内 |

**数据资产的预期格式：**

| 输入 | 格式要求 |
|------|----------|
| 井头文件 | Petrel `BEGIN HEADER ... END HEADER` 文本，必须包含 `Name`、`Surface X`、`Surface Y`、`Well datum name`、`Well datum value`、`Bottom hole X`、`Bottom hole Y` 列 |
| LAS 目录 | 文件名 stem 即为井名，扩展名 `.las` |
| 井轨迹目录 | 文件名 stem 即为井名，不检查扩展名；本脚本仅检查文件是否存在 |
| 井分层文件 | Petrel 格式，必须包含 `Well`、`Surface`、`X`、`Y`、`Z`、`MD`、`PVD auto` 列 |
| 时深表目录 | 文件名 stem 即为井名；本脚本仅检查文件是否存在 |

**井名匹配规则：** 所有资产通过文件名 stem 或记录中的 `Name`/`Well` 字段做大小写不敏感匹配。相同井名的不同大小写形式和同名不同扩展名会归为同一口井；井轨迹文件也可以没有扩展名。大小写冲突会直接报错。名为 `nan`、`none`、`null` 或空白的记录会被跳过。

---

## 配置参考

资产路径和地震体是整个工区的共享事实，放在顶层；第一步只保留自己的空间 QC 阈值。所有资产路径均相对于 `data_root`。

```yaml
data_root: <DATA_ROOT>
output_root: <OUTPUT_ROOT>

assets:
  well_heads_file: <WELL_HEADS_FILE>
  las_dir: <LAS_DIRECTORY>
  well_trace_dir: <WELL_TRACE_DIRECTORY>
  well_tops_file: <WELL_TOPS_FILE>
  time_depth_dir: <TIME_DEPTH_DIRECTORY>

seismic:
  file: <SEISMIC_FILE>
  type: segy
  domain: time
  iline_byte: 189
  xline_byte: 193
  istep: 1
  xstep: <XLINE_STEP>


well_inventory:
  spatial_qc:
    near_survey_threshold_m: 500.0
    vertical_bottom_offset_threshold_m: 30.0
    platform_cluster_threshold_m: 12.5
    dense_well_neighbor_threshold_m: 150.0
```

### `seismic`

`seismic` 块描述工区地震体。第一步只从中读取工区几何（线号范围、道间距、footprint 四角）。

#### 通用字段

| 字段 | 必填 | 说明 |
|------|------|------|
| `file` | 是 | 地震体路径，相对于 `data_root` |
| `type` | 是 | 地震体格式：`segy` 或 `zgy` |
| `domain` | 是 | 时间域地震使用 `time`，需显式填写 |

#### 格式相关参数

这些参数并非第一步的逻辑依赖，但会随 `seismic` 块一起加载到配置模型中。配不配取决于你的数据格式：

| 字段 | 适用 `type` | 说明 |
|------|-------------|------|
| `zgy_inline_chunk_size` | `zgy` | inline 分块读取大小（正整数），影响后续步骤的 trace 读取性能。第一步不读 trace，**对第一步无影响** |
| `iline`、`xline` | `segy` | 覆盖 SEG-Y 道头中 inline/xline 号的字节位置 |
| `istep`、`xstep` | `segy` | 覆盖 inline/xline 步长 |
| `iline_byte`、`xline_byte` | `segy` | 与 `iline`/`xline` 等效，映射到同一底层读取参数 |

如果你的 SEG-Y 使用标准道头位置，可以省略所有 SEG-Y 字段。ZGY `zgy_inline_chunk_size` 默认 16。

示例中的 `<XLINE_STEP>` 填写输入地震体的横线号步长。线号步长表示相邻道的线号间隔，米制道间距由工区几何计算。

### `spatial_qc`

#### `near_survey_threshold_m`

用于区分“刚好在工区边缘外”和“离工区很远”的井。脚本会计算井口到地震工区边界的最近距离；距离在这个范围内的井记为 `near_outside`，更远的井记为 `outside`。这个阈值取决于你的工区边缘地质情况。如果工区边界附近有可靠的地震数据覆盖，可以放宽；如果边界处地震质量差，保持默认即可。

#### `vertical_bottom_offset_threshold_m`

用于在还没有解析完整轨迹之前，先给每口井一个粗略井型。脚本会用 Petrel 井头导出中的 `Surface X/Y` 和 `Bottom hole X/Y` 计算井口到底孔的水平偏移；偏移很小的井先视为直井，偏移明显的井先视为斜井。注意：**这是初分，不是最终轨迹解释，后面的井轨迹 QC 会用完整轨迹重新复核井型**。如果初分和复核经常不一致，再回头调整这个阈值。

#### `dense_well_neighbor_threshold_m`

用于统计井口水平距离不超过阈值的近邻井对。该阈值控制运行摘要中的近邻计数，便于了解井口的空间分布。

#### `platform_cluster_threshold_m`

用于按井口距离建立连通平台簇。脚本先识别同平台井，再从同道冲突清单中排除同平台井对。

`dense_well_neighbor_threshold_m` 和 `platform_cluster_threshold_m` 这两个阈值的大小关系也因此应该不同：平台阈值通常很小，只识别井口几乎贴在一起的井；近邻阈值更大，用来观察密井网中可能互相影响的井对。此外，`dense_well_neighbor_threshold_m` 只影响 `run_summary.json` 中的近邻统计计数；`well_neighbor_pairs.csv` 的导出更克制：只保留落到同一最近地震道且非同平台的井对。

---

## 脚本在做什么

1. **建立资产索引。** 读取井头、测井文件、井轨迹、井分层和时深表的井名，按大小写不敏感的匹配规则合并为井清单，并检查同名冲突。
2. **计算井口位置。** 从地震体建立工区几何，将井口坐标换算为线号，计算最近道和到工区边界的距离，区分工区内、近边界外和远离工区的井。
3. **初分井型。** 使用井口到底孔的水平距离判断直井和斜井；坐标不完整时保留未知状态。
4. **统计空间关系。** 按井口米制距离形成连通平台簇，并统计近邻井对。落到同一最近地震道且属于不同平台的井对进入冲突清单。
5. **写出盘点结果。** 输出资产主表、冲突井对、平台分组和运行摘要，供后续筛选与轨迹复核使用。

---

## 核心输出文件

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

**只导出高风险同道冲突井对。** 筛选条件：井口落在同一最近地震道、且**非同平台**。同平台同道被视为正常情况，不在本文件中输出。

| 字段 | 含义 |
|------|------|
| `well_a`, `well_b` | 井名 |
| `distance_m` | 井口 XY 距离 |
| `same_surface_nearest_trace` | 始终为 true（因为只导出同道对） |
| `same_surface_platform` | 始终为 false（同平台对已过滤） |
| `class_pair` | 例如 `vertical/deviated` |
| `risk` | 当前固定为 `same_trace_conflict` |

### 3. `well_clusters.csv` — 同平台井分组

把井口距离很近、很可能属于同一平台的井放在一起。这个文件适合用来检查平台井规模，以及后续是否需要按平台加权或选代表井。

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

包含：脚本名、配置路径、所有输入文件路径（相对于仓库根目录）、四项阈值、工区几何信息、道间距（线号/道号/nominal，单位米）、工区 footprint 四角 XY，以及全部统计计数和名单。

其中，地震体几何：

| JSON 路径 | 含义 |
|-----------|------|
| `geometry.sample_domain` / `geometry.sample_unit` | 时间采样轴类型和单位，分别为 `time` 和 `s` |
| `geometry.sample_min` / `geometry.sample_max` / `geometry.sample_step` | 采样轴起止值和采样间隔；查时间采样间隔就看 `geometry.sample_step` |
| `geometry.n_sample` | 时间采样点数 |
| `geometry.inline_min` / `geometry.inline_max` / `geometry.inline_step` | inline 线号范围和线号步长 |
| `geometry.xline_min` / `geometry.xline_max` / `geometry.xline_step` | xline 线号范围和线号步长 |
| `geometry.n_il` / `geometry.n_xl` | inline / xline 数量 |
| `bin_spacing_m.nominal` | 近似道间距，单位米 |
| `footprint_xy` | 工区 footprint 四角 XY |

关键的 `neighbor_summary` 段：

| 字段 | 含义 |
|------|------|
| `valid_surface_well_count` | 有效井口坐标井数（参与近邻计算） |
| `dense_neighbor_pair_count` | 落入 `spatial_qc.dense_well_neighbor_threshold_m` 统计半径内的井对总数 |
| `same_surface_nearest_trace_pair_count` | 井口吸附到同一最近道的井对数 |
| `same_platform_pair_count` | 被 `spatial_qc.platform_cluster_threshold_m` 识别为同平台的井对数 |
| `same_trace_platform_pair_count` | 同时满足同道和同平台的井对数 |
| `exported_neighbor_pair_count` | 写入 `well_neighbor_pairs.csv` 的硬冲突数 |
| `platform_cluster_count` | 平台分组数 |
| `platform_cluster_well_count` | 参与平台分组的井数 |

---

## 如何阅读结果

### 第一步：看 `run_summary.json` 的顶层计数

```
well_count: <WELL_COUNT>
asset_counts: {well_heads: <WELL_HEAD_COUNT>, las: <LAS_COUNT>, well_trace: <TRACE_COUNT>, time_depth: <TDT_COUNT>, ...}
survey_position_counts: {inside: <INSIDE_COUNT>, outside: <OUTSIDE_COUNT>}
wellbore_class_counts: {deviated: <DEVIATED_COUNT>, vertical: <VERTICAL_COUNT>}
```

这几行直接回答：有多少井？缺哪些资产？多少在工区内？多少看起来是斜井？

地震几何也记录在同一份 `run_summary.json` 中。时间采样间隔看 `geometry.sample_step`，单位为秒；线号范围看 `geometry.inline_*` 和 `geometry.xline_*`，近似物理道间距看 `bin_spacing_m.nominal`。

### 第二步：如果有 `las_only` 井 → 补井头

`las_only` 表示有 LAS 曲线但没有井头记录，缺 XY 坐标。这类井无法进入任何后续井震流程。检查是否漏导了井头，或者 LAS 文件名与井头 Name 是否存在拼写差异。

### 第三步：如果有 `head_only` 井 → 确定是否需要

`head_only` 表示有井头但没有 LAS 文件。这类井保留了空间信息，但不能进入第二步的曲线筛选。检查是 LAS 文件缺失，还是文件名不匹配。

### 第四步：看 `neighbor_summary`

- `dense_neighbor_pair_count` 很大（>300）→ 说明井网很密，但不等于数据有问题。
- `exported_neighbor_pair_count` > 0 → 存在井口落在同一地震道、但不属于同一平台的井对。查看 `well_neighbor_pairs.csv` 了解详情；这类井在后续 auto-tie 和井约束中可能需要特殊处理。
- `platform_cluster_count` 告诉你工区内有多少个集中钻井平台。每个 cluster 的 `cluster_size` 可以帮助判断后续是否需要对同平台井做代表井选择或加权处理。

### 第五步：看 `well_inventory.csv` 的具体列

- 按 `survey_position` 筛选 `inside`，按 `inventory_status` 筛选 `usable_for_las_screen`——这是进入第二步的候选井。
- 关注 `wellbore_class == deviated` 且 `has_well_trace == false` 的井——斜井但没有轨迹文件，第四步无法走斜井路径。
- 关注 `wellbore_class == unknown` 的井——井头坐标缺失或无效。
- `reasons` 列汇总了每口井的所有警告标签。`no_time_depth` 表示缺少时深表；`no_well_trace` 只对斜井或井型未知的井记录。
