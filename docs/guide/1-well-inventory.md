# 01 井资产盘点

本文只讨论第一个规划脚本：`well_inventory.py`。

反演工作流默认按时间域推进，因此脚本名、配置 key、输出目录都不加 `_time` 后缀。当前阶段先不落地代码，先把脚本边界、输入输出和应拆出的模块能力讨论清楚。

## 目标

`well_inventory.py` 是一个非破坏性盘点脚本，只回答五件事：

1. 原始 LAS 文件和井头文件里有哪些井，交集和差集分别是什么。
2. 每口井的井口位置在地震工区内、工区外但靠近、还是明确在工区外。
3. 每口井按井口到底孔的水平偏移，初分为直井、斜井或未知，并标注密井网近邻风险。
4. 每口井是否有时深表、是否有井分层。
5. 每口井是否有井轨迹/井斜文件，可支撑后续斜井路径。

这个脚本不读取 LAS 曲线数据、不做完整曲线筛选、不替换异常值、不做井震标定。时深表只做“同名文件是否存在”的资产标记，列名兼容和 `TimeDepthTable` 构造放到井震标定阶段讨论。

## 输入

- 配置文件：建议默认 `experiments/common.yaml`。
- 井头文件：`data/raw/well_heads`。
- LAS 目录：`data/all_well_las`。
- 井轨迹目录：`data/all_well_trace`，本脚本只检查同名文件是否存在。
- 井分层文件：`data/raw/well_tops`。
- 时深表目录：`data/time_depth_table`。
- 地震体：`data/raw/obn-clipped-240-912-872-1544.zgy`。

建议配置片段：

```yaml
well_inventory:
  well_heads_file: raw/well_heads
  las_dir: all_well_las
  well_trace_dir: all_well_trace
  well_tops_file: raw/well_tops
  time_depth_dir: time_depth_table
  seismic:
    file: raw/obn-clipped-240-912-872-1544.zgy
    type: zgy
  near_survey_threshold_m: 500.0
  vertical_bottom_offset_threshold_m: 30.0
  dense_well_neighbor_threshold_m: 150.0
```

前三个阈值先按上面默认值走：

- `near_survey_threshold_m = 500.0`
- `vertical_bottom_offset_threshold_m = 30.0`
- `dense_well_neighbor_threshold_m = 150.0`

`dense_well_neighbor_threshold_m` 是早期 QC 阈值，不应长期脱离地震 bin 尺寸硬编码。落地时应同时报告 nominal bin spacing，并建议把默认值理解为 `max(3 * nominal_bin_spacing_m, 150 m)` 或按工区配置覆盖；否则密井网中会产生过多近邻对，降低报告可读性。

如果后续发现井头坐标精度较差，再把直井阈值放宽到 50 m。

## 输出

默认输出目录建议为：

```text
scripts/output/well_inventory_<timestamp>/
```

核心文件：

- `well_inventory.csv`：一井一行的主清单。
- `well_neighbor_pairs.csv`：近邻井对 QC。
- `run_summary.json`：输入、阈值、统计摘要和失败原因汇总。

`well_inventory.csv` 建议字段：

| 字段 | 含义 |
| --- | --- |
| `well_name` | 统一井名 |
| `has_well_head` | 井头文件是否包含 |
| `has_las` | LAS 文件是否存在 |
| `has_well_trace` | 井轨迹/井斜文件是否存在 |
| `has_time_depth` | `data/time_depth_table` 下是否存在同名时深表 |
| `has_well_tops` | 井分层文件是否包含该井 |
| `surface_x`, `surface_y` | 井口坐标 |
| `bottom_x`, `bottom_y` | 底孔坐标 |
| `kb_m` | `Well datum value` |
| `inline_float`, `xline_float` | 井口投影到工区后的浮点道号；工区外可为空 |
| `survey_position` | `inside`、`near_outside`、`outside`、`invalid_xy` |
| `distance_to_survey_m` | 井口到真实 XY 工区边界的最短距离；工区内为正数，边界上约为 0 |
| `bottom_offset_m` | 井口到底孔水平距离 |
| `wellbore_class` | `vertical`、`deviated`、`unknown` |
| `inventory_status` | `usable_for_las_screen`、`head_only`、`las_only`、`outside_survey` 等 |
| `reasons` | 分号分隔的失败或警告原因 |

`well_neighbor_pairs.csv` 建议字段：

| 字段 | 含义 |
| --- | --- |
| `well_a`, `well_b` | 近邻井名 |
| `distance_m` | 井口 XY 距离 |
| `same_nearest_trace` | 是否落在同一最近地震道 |
| `class_pair` | 例如 `vertical/deviated` |
| `risk` | `same_bin`、`close_wells`、`overlap_likely` |

近邻风险只标注，不自动删除井。怎么合并、降权或选择代表井，留到井约束文档里讨论。

## 处理逻辑

### 井名集合

井名来源有两个：

- 井头：`import_well_heads_petrel(well_heads_file)["Name"]`
- LAS：`Path(las_dir).glob("*.las")` 的 stem

主清单使用二者并集。每口井记录：

- 同时有井头和 LAS：进入后续 LAS header 筛选候选。
- 只有井头：保留空间信息，但不进入曲线筛选。
- 只有 LAS：保留文件信息，但没有空间状态，不能进入井震相关流程。

井轨迹文件只作为存在性标记，例如 `data/all_well_trace/<well>.txt`。本脚本不解析轨迹内容，避免把斜井地震道读取逻辑提前混进来。

### 路由资产核查

第四步井震标定需要按井的数据条件分流，因此第一步应补充几个轻量资产标记：

- `has_time_depth`：检查 `data/time_depth_table/<well>` 或同名扩展文件是否存在。
- `has_well_tops`：检查 `data/raw/well_tops` 中是否有该井记录。
- `has_well_trace`：检查 `data/all_well_trace/<well>.txt` 或同名轨迹文件是否存在。

这里不解析时深表列名，不构造 `TimeDepthTable`，也不解析井轨迹的 MD/XYZ 内容。`has_well_trace` 只是路由提示，第四步处理斜井前还需要正式读取轨迹并做范围 QC。

这些字段主要服务后续井震标定的路径规划：

| 条件 | 后续含义 |
| --- | --- |
| 有时深、直井 | 可走现有时深基础上的直井 auto-tie。 |
| 无时深、有井分层、直井 | 可走层位锚点 + 纵波积分的直井路径。 |
| 有时深、有井轨迹、斜井 | 可走斜井地震道读取 + 现有时深基础上的 auto-tie。 |
| 缺少关键资产 | 后续井震标定拒绝或等待人工补数。 |

第一步只标记条件，不决定最终路径。最终路径应由第四步结合第二、第三步的曲线可用性重新判定。

### 工区位置

工区内判断：

1. 用 `open_survey(seismic_file, seismic_type="zgy")` 打开工区。
2. 对井口坐标调用 `survey.coord_to_line(x, y)`。
3. 如果转换成功，并且浮点 inline/xline 落在 `survey.query_geometry("time")` 的线号范围内，则为 `inside`。

实现时必须捕获 `coord_to_line()` 或底层 `coord_to_index()` 抛出的 `ValueError`。对 ZGY 工区而言，工区外点不会返回一个越界浮点线号，而是直接报错。捕获后再进入工区外距离计算，不应让单口工区外井中断整个盘点。

工区外但靠近：

1. 用 `survey.line_to_coord()` 取工区四角或边界点，形成真实 XY footprint。
2. 计算井口点到 footprint 的最短欧氏距离。
3. 如果距离小于等于 `near_survey_threshold_m`，标记为 `near_outside`；否则为 `outside`。

`distance_to_survey_m` 始终表示到工区边界的真实 XY 距离。工区内也可以计算到边界的最近距离，不用 0 伪装“已在工区内”。如果第一版暂不计算工区内到边界距离，则应写 `null`，不要写 0。

易错点：不能用 `hypot(dil * inline_step, dxl * xline_step)` 估算物理距离。`inline_step` 和 `xline_step` 是线号步长，不是米制间距。只要参数名涉及 `_m`、`radius_xy_m`、物理半径或距离衰减，就必须通过 `line_to_coord()` 或 `cup.seismic.spatial` 的真实 XY 工具计算。

### 直井/斜井初分

第一版直接使用井头文件里的底孔坐标：

```text
bottom_offset_m = hypot(Bottom hole X - Surface X, Bottom hole Y - Surface Y)
```

分类规则：

- `bottom_offset_m <= 30.0`：`vertical`
- `bottom_offset_m > 30.0`：`deviated`
- 任一坐标缺失或非有限：`unknown`

这个分类只用于早期分流，不代表最终井轨迹 QC。后续资产准备脚本解析 `data/all_well_trace` 后，需要用完整井轨迹复核。

如果存在井轨迹文件，第四步路由前应以轨迹复核后的井型为准。井头底孔坐标只做第一轮粗分，因为 Petrel 导出的底孔坐标可能存在四舍五入或缺失。

### 密井网 QC

只在有有效井口坐标的井之间计算近邻：

- 距离小于等于 `dense_well_neighbor_threshold_m` 的井对写入 `well_neighbor_pairs.csv`。
- 如果两口井 round 到同一最近 inline/xline，`same_nearest_trace = true`。
- `same_nearest_trace` 的风险等级高于普通近邻，因为后续井震标定、井约束和训练采样可能读到同一地震道。

这里仍然只做标注，不做自动筛井。

## 模块边界

这一轮可以同步明确首脚本需要的模块边界，但先不写实现。

### 保留并复用

- `cup.petrel.load.import_well_heads_petrel()`：继续负责 Petrel 井头文本解析。
- `cup.seismic.survey.open_survey()`：继续作为 SEG-Y/ZGY 工区入口。
- `cup.seismic.spatial.build_trace_xy_grids()`、`xy_circle_mask()`：后续需要半径/影响区时使用；首脚本暂不必生成全量 XY grid。

### 建议新增

`cup.well.assets`

- `WellHead`：井头记录的结构化对象，包含井名、井口、底孔、KB。
- `WellInventoryRecord`：`well_inventory.csv` 的一行，集中表达空间状态、文件存在性和分流结果。
- `WellInventory`：清单集合，负责过滤候选井、导出 CSV/JSON。

`cup.seismic.survey`

- `footprint_xy()`：返回地震工区真实 XY footprint。
- `distance_to_footprint(x, y)`：返回点到工区 footprint 的最短米制距离。

这两个 survey 能力应放在工区上下文里，不要在脚本中散写几何算法。这样后续所有脚本都能复用同一个“工区距离”定义。

### 暂不处理

这些内容与第一个脚本无关，先不放进本文档：

- LAS header AI 判读和曲线筛选。
- `TimeDepthTable` 的列名兼容。
- 井轨迹文件解析和 XY/MD/TVD 轨迹 dataclass。
- 斜井沿轨迹提取地震道。
- 井约束阶段的近井冲突消解策略。

## 下游契约

后续脚本不再自己扫描 `data/all_well_las` 决定井列表，而是读取 `well_inventory.csv`：

- `inventory_status == usable_for_las_screen` 的井进入 LAS header 筛选。
- `survey_position == outside` 的井默认不进入井震相关流程。
- `near_outside` 的井保留，但需要后续脚本显式决定是否使用。
- `wellbore_class` 只作为初筛标签，不能替代后续完整轨迹 QC。
- `has_time_depth`、`has_well_tops`、`has_well_trace` 只作为第四步分流的早期提示，不替代后续正式读取和 QC。
