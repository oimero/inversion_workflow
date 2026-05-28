# 04 井震自动标定

`well_auto_tie.py` 是时间域工作流的第四步。它读取前三步的井资产、曲线筛选和测井预处理结果，再结合时深表、井轨迹、解释层位和地震体，为每口井生成井震标定结果和候选子波。

读完这篇你会知道：脚本怎么跑、三条已落地路径分别做什么、斜井路径怎么沿轨迹取地震道，以及应该从哪些输出文件里判断结果是否可信。

---

## 快速开始

```bash
python scripts/well_auto_tie.py
python scripts/well_auto_tie.py --config experiments/common.yaml
python scripts/well_auto_tie.py --well <well-name>
python scripts/well_auto_tie.py --output-dir scripts/output/well_auto_tie_test
```

不带参数运行时，脚本读取 `experiments/common.yaml`，自动发现最新的前置产物，在 `scripts/output/well_auto_tie_<timestamp>/` 下写出结果。

建议在正式批量跑之前先用 `--well <name>` 抽查一口直井和一口斜井。斜井路径已经落地，但它会真实读取井轨迹并沿轨迹取道，数据问题比直井更容易暴露。

---

## 运行前检查

第四步需要这些上游结果：

| 来源 | 必要文件 | 用途 |
|------|----------|------|
| 第一步 | `well_inventory.csv` | 井口/底孔坐标、资产存在性、井型初分、工区位置 |
| 第二步 | `well_curve_screen.csv` | 保留曲线筛选审计信息 |
| 第三步 | `well_preprocess_status.csv`、`preprocessed_las/*.las` | 判断曲线是否可用，并读取标准 `DT_USM`、`RHO_GCC` |
| 轨迹 QC | `well_trajectory_qc.csv` | 优先使用轨迹复核后的井型 |
| 数据目录 | 时深表目录、井轨迹目录、井分层文件 | 时深表、Petrel 井轨迹、井分层 |
| 地震数据 | ZGY 或 SEG-Y 体、解释层位 | 读取井旁/轨迹地震道和目标时间窗 |

如果没有找到最新的 `well_trajectory_qc_*` 输出，脚本仍可运行，但斜井/直井判定会更多依赖第一步的井头底孔初分。

---

## 配置参考

核心配置位于 `well_auto_tie` 段。路径默认相对于顶层 `data_root`，前置产物路径默认从 `scripts/output` 自动发现。

```yaml
well_auto_tie:
  source_runs:
    mode: latest
    well_inventory_dir: null
    las_curve_screen_dir: null
    log_preprocess_dir: null
    well_trajectory_qc_dir: null

  inventory_file: null
  curve_screen_file: null
  preprocess_status_file: null
  preprocessed_las_dir: null
  trajectory_qc_file: null

  time_depth_dir: <time-depth-dir>
  well_trace_dir: <well-trajectory-dir>
  well_tops_file: <well-tops-file>

  interpretation:
    top_horizon: <top-horizon-file>
    bottom_horizon: <bottom-horizon-file>

  target_interval:
    top: <top-marker-name>
    bottom: <bottom-marker-name>
    margin_top_ms: 100.0
    margin_bottom_ms: 100.0
    twt_unit: auto

  seismic:
    file: <seismic-volume-file>
    type: zgy

  enabled_routes:
    - vertical_with_tdt
    - vertical_anchor_from_tops
    - deviated_with_tdt

  coarse_anchor:
    enabled: true
    apply_to_routes:
      - vertical_anchor_from_tops
    config_file: experiments/well_auto_tie_anchors.yaml

  reject:
    allow_near_outside: false
    min_valid_log_fraction: 0.7
    min_tie_samples: 64
    max_trajectory_outside_fraction: 0.05
```

### `enabled_routes`

`experiments/common.yaml` 当前启用三条已落地路径：

| 路径 | 状态 | 说明 |
|------|------|------|
| `vertical_with_tdt` | 已实现 | 直井，有 Petrel MD 域时深表 |
| `vertical_anchor_from_tops` | 已实现 | 直井，无时深表，用井分层和解释层位锚点积分建初始 TDT |
| `deviated_with_tdt` | 已实现 | 斜井，有 Petrel MD 域时深表和井轨迹，沿轨迹取地震道 |

脚本内部保守默认只启用前两条；主配置 `experiments/common.yaml` 显式启用了 `deviated_with_tdt`。如果要先做计划审计，可以暂时从配置里删掉 `deviated_with_tdt`，对应井会在 `well_tie_plan.csv` 中显示为 `skipped_disabled`。

`deviated_anchor_from_tops` 仍未落地。它是“斜井、无时深、有井轨迹、有井分层”的第四条路径，后续设计放在 `docs/guide/deviated-well-src-cup-refactor.md`。

### `target_interval`

脚本用顶底解释层位构造标定目标窗。当前三条已落地路径都在井口 XY 处读取 `top_horizon` 和 `bottom_horizon` 的 TWT，再按 `margin_top_ms`、`margin_bottom_ms` 向上下拓展。

`target_crop_ms` 只控制成功标定后导出的候选子波长度，不是 auto-tie 输入窗口长度。

### `coarse_anchor`

`vertical_anchor_from_tops` 必须使用锚点配置。通常做法是选择一个目标层位或可靠标志层，用该井分层 MD 对齐对应地震解释层位 TWT，作为声波积分的绝对时间基准。

有时深表路径默认不使用锚点粗校正。若要对 `vertical_with_tdt` 或 `deviated_with_tdt` 做整体 TWT shift，必须显式把对应 route 加入 `coarse_anchor.apply_to_routes`，并在跑批前单独验证。

### `max_trajectory_outside_fraction`

只影响 `deviated_with_tdt`。脚本先按目标窗口生成每个 TWT 样点的轨迹落道计划：

- 出界比例 `> max_trajectory_outside_fraction`：整井失败，原因写 `trajectory_outside_fraction_exceeded`。
- 出界比例 `<= max_trajectory_outside_fraction`：裁剪到最长连续工区内 TWT 段，再继续 auto-tie。
- 裁剪后样点数 `< min_tie_samples`：整井失败，原因写 `trajectory_inside_tie_samples_too_few`。

---

## 脚本在做什么

### 第一步：构建路由计划

脚本 join `well_inventory.csv`、`well_preprocess_status.csv` 和可选的 `well_trajectory_qc.csv`，生成 `well_tie_plan.csv`。

路由规则：

| route | 条件 | 动作 |
|------|------|------|
| `vertical_with_tdt` | 工区内或允许边缘井；直井；有时深；`DT_USM` 和 `RHO_GCC` 可用 | 执行 |
| `vertical_anchor_from_tops` | 工区内或允许边缘井；直井；无时深；有井分层；曲线可用 | 执行 |
| `deviated_with_tdt` | 工区内或允许边缘井；斜井；有时深；有井轨迹；曲线可用 | 执行 |
| `deviated_anchor_from_tops` | 斜井；无时深；有井轨迹；有井分层；曲线可用 | 仅识别，未实现 |
| `rejected` | 其他情况 | 拒绝 |

`route_status` 有三种：`planned`、`skipped_disabled`、`rejected`。只有 `planned` 且 route 已实现的井会进入实际 auto-tie。

### 第二步：准备初始时深表和测井窗口

三条已落地路径都使用 MD 域 `grid.TimeDepthTable`：

- `vertical_with_tdt`：读取 Petrel 时深表；若目标窗只被部分覆盖，就用 `DT_USM` 从 TDT 端点向上或向下补齐。
- `vertical_anchor_from_tops`：用井分层 MD 和解释层位 TWT 建锚点，再沿 `DT_USM` 向上、向下积分出初始 TDT。
- `deviated_with_tdt`：读取 Petrel MD 域时深表；轨迹只负责 `TWT -> MD -> XY -> trace` 的空间定位，不替代时深表。

如果有时深表路径完全不接触目标窗，直井会尝试改走锚点路径；斜井当前直接失败，原因是 `tdt_no_target_window_overlap`。

### 第三步：读取地震道

直井路径调用：

```text
survey.read_trace_at_xy(surface_x, surface_y, domain="time")
```

斜井 `deviated_with_tdt` 路径调用的是沿轨迹取道：

1. 从地震采样轴取出目标窗口内的 TWT 样点。
2. `sample_trajectory_on_twt(trajectory, table, twt_axis)` 用 MD 域 TDT 做 `TWT -> MD`，再用 `WellTrajectory.position_at_md()` 得到每个样点的 `x_m/y_m/tvdss_m`。
3. `build_nearest_trace_sample_plan(samples, survey)` 把每个 XY 样点吸附到最近 inline/xline，并记录 `flat_idx` 与 `survey_position`。
4. 允许少量出界后，裁剪到最长连续 inside TWT 段。
5. `assemble_nearest_trace_from_plan(...)` 对唯一 `(inline_index, xline_index)` 去重批读，再按 TWT 样点拼成一条 `grid.Seismic`。

第一版采用最近道，不做双线性或多道加权。所有样点级落道信息都会写入 `trace_sample_plan_<well>.csv`。

### 第四步：调用 `wtie` 细标定

脚本用窗口内的 `logset_md`、`seismic` 和 MD 域 `table` 构造：

```text
InputSet(logset_md=logset_md, seismic=seismic, table=table, wellpath=None)
```

这里 `wellpath=None` 是有意的：斜井路径已经在进入 `wtie` 之前把地震道沿轨迹拼好了，`wtie` 只处理 MD 域曲线和 TWT 表之间的细调。

---

## 输出文件

所有文件写到 `<output_root>/well_auto_tie_<timestamp>/`：

| 文件 | 内容 |
|------|------|
| `well_tie_plan.csv` | 一井一行的路由计划 |
| `well_tie_metrics.csv` | 已执行井的标定指标和输出路径 |
| `rejected_wells.csv` | 路由阶段被拒绝的井 |
| `tie_window_report.csv` | 目标窗口、实际窗口、TDT 支持类型和裁剪原因 |
| `anchor_report.csv` | 锚点建表路径的锚点 MD/TWT、层位采样信息 |
| `wavelet_inventory.csv` | 成功井导出的候选子波清单，供第五步使用 |
| `wavelets/wavelet_201ms_<well>.csv` | 裁剪并能量归一化后的子波 |
| `time_depth/initial_tdt_<well>.csv` | 进入 `wtie` 前的初始 TDT，含 `source` 列 |
| `time_depth/optimized_tdt_<well>.csv` | `wtie` 优化后的 TDT |
| `synthetic_qc/tie_qc_<well>.csv` | 地震、反射系数、合成记录和残差 |
| `seismic_trace/seismic_trace_<well>.csv` | 实际用于标定的井旁或轨迹地震道 |
| `trace_sample_plan/trace_sample_plan_<well>.csv` | 斜井样点级轨迹落道计划；直井通常没有 |
| `figures/<well>/*.png` | 优化目标、TDT、合成匹配、子波 QC 图 |
| `run_summary.json` | 输入路径、路由统计、失败统计、逐井补充信息 |

### `trace_sample_plan_<well>.csv`

这是斜井路径最重要的审计文件。

| 字段 | 含义 |
|------|------|
| `twt_s` / `md_m` | 当前地震时间样点及其由 TDT 反查得到的 MD |
| `x_m` / `y_m` | 该 MD 在井轨迹上的 XY |
| `inline_float` / `xline_float` | XY 投影到工区后的浮点线号 |
| `nearest_inline` / `nearest_xline` | 最近道线号 |
| `inline_index` / `xline_index` | 最近道数组索引 |
| `flat_idx` | 地震体内部 trace index |
| `survey_position` | `inside` 或 `outside` |
| `used_for_tie` | 裁剪后是否实际进入 auto-tie |

如果一口斜井失败，也可能已经写出这份文件。它可以帮助判断失败是轨迹本身出界、几何转换失败，还是最长 inside 窗口太短。

---

## 如何阅读结果

### 第一步：看 `run_summary.json`

先检查：

```text
route_counts
route_status_counts
tie_status_counts
planned_run_count
successful_tie_count
```

如果 `skipped_disabled` 很多，说明配置没有启用对应 route；如果 `failed` 集中在斜井，优先看 `result_extras.<well>.trace_sampling` 和该井的 `trace_sample_plan`。

### 第二步：看 `well_tie_plan.csv`

确认每口井为什么进入某条路径。常见原因标签：

- `route_disabled_deviated_with_tdt`：斜井有时深路径被配置禁用。
- `survey_position_outside`：井口不在允许工区位置。
- `unusable_p_sonic` / `unusable_density`：第三步曲线不可用。
- `no_time_depth` / `no_well_trace` / `no_well_tops`：缺少路线所需资产。

### 第三步：看 `well_tie_metrics.csv`

重点看：

- `initial_corr`：初始 TDT 与地震的相关性。
- `optimized_corr`：优化后的相关性。
- `optimized_nmae`：归一化误差，越低越好。
- `best_table_shift_ms`：`wtie` 找到的整体时移。
- `tie_window_start_s` / `tie_window_end_s`：实际标定窗口。
- `tdt_support_class`：窗口来自原始 TDT、声波拓延，还是锚点积分。

`optimized_corr` 提高不等于结果一定可靠；必须结合 TDT 图、合成记录图和子波形态一起看。

### 第四步：看 `tie_window_report.csv`

这个文件回答“脚本到底拿哪段曲线和地震做标定”。如果 `window_clip_reason` 包含 `trajectory_inside_crop`，说明斜井目标窗因为出界样点被裁剪到最长连续 inside 段。

### 第五步：看图和 QC CSV

抽查每口成功井的：

- `figures/<well>/time_depth_table.png`
- `figures/<well>/synthetic_match.png`
- `figures/<well>/wavelet.png`
- `synthetic_qc/tie_qc_<well>.csv`

重点看优化后的 TDT 是否出现不合理扭曲，合成记录是否只在局部强行对齐，以及子波是否过窄、偏相或振铃异常。

---

## 常见失败原因

| 原因 | 含义 | 处理建议 |
|------|------|----------|
| `tdt_no_target_window_overlap` | 时深表完全不覆盖目标窗口 | 检查 Petrel TDT、解释层位单位和目标窗配置 |
| `trajectory_outside_fraction_exceeded` | 斜井目标窗内轨迹样点出界比例超过阈值 | 查看 `trace_sample_plan`，确认轨迹或工区几何 |
| `trajectory_inside_tie_samples_too_few` | 裁剪后的连续 inside 样点太少 | 放宽窗口、检查轨迹，或暂时跳过该井 |
| `TWT axis ... outside table range` | 地震窗口超出准备后的 TDT 范围 | 检查声波拓延是否有足够曲线覆盖 |
| `Seismic trace has zero standard deviation` | 读取到的地震道窗口无有效振幅变化 | 检查地震体、窗口和道索引 |

---

## 留到第二轮

- `deviated_anchor_from_tops`：斜井无时深、有井分层和轨迹的第四条路径。
- 斜井沿轨迹地震采样从最近道升级到双线性或多道加权。
- 斜井轨迹 inline/xline 随 TWT 的专门 QC 图。
- 多口密集井共用同一 trace/time 样点时的冲突报告和权重策略。
