# 04 井震自动标定

`well_auto_tie.py` 是时间域工作流的第四步。它把前三步的产出（井资产、曲线筛选、预处理 LAS）与时深表、井轨迹、地震体组合起来，对每口井做井震标定，输出优化后的时深关系、候选子波和合成记录 QC。

---

## 快速开始

```bash
python scripts/well_auto_tie.py
python scripts/well_auto_tie.py --config experiments/common.yaml
python scripts/well_auto_tie.py --well <well-name>
python scripts/well_auto_tie.py --output-dir scripts/output/well_auto_tie_test
```

不带参数时，脚本自动发现最新的前三步产物，在 `scripts/output/well_auto_tie_<timestamp>/` 下写出结果。

建议批量跑之前先用 `--well` 各抽一口直井和一口斜井试跑。斜井路径会真实加载井轨迹并沿轨迹取道，数据问题比直井更容易暴露。

---

## 运行前需要什么

| 来源 | 文件 | 用途 |
|------|------|------|
| 第一步 | `well_inventory.csv` | 井口坐标、资产清单、井型初分、工区位置 |
| 第二步 | `well_curve_screen.csv` | 曲线筛选审计、每口井有哪些可用曲线 |
| 第三步 | `well_preprocess_status.csv`、`preprocessed_las/*.las` | 判断 `DT_USM` 和 `RHO_GCC` 是否可用，加载标准 LAS |
| 轨迹 QC | `well_trajectory_qc.csv` | 优先用复核后的井型替代第一步的初分 |
| 数据目录 | 时深表目录、井轨迹目录、井分层文件 | 时深表、Petrel 井轨迹、井分层 |
| 地震数据 | ZGY 或 SEG-Y 体、解释层位 | 读取井旁地震道或沿轨迹道集，确定目标时间窗 |

如果找不到最新的 `well_trajectory_qc_*` 目录，脚本退回到第一步的井口/底孔初分来判定直井斜井——对直井没影响，但斜井判定可能不准。

---

## 配置参考

```yaml
well_auto_tie:
  source_runs:
    mode: latest                  # 自动发现最新前置产物
    well_inventory_dir: null
    las_curve_screen_dir: null
    log_preprocess_dir: null
    well_trajectory_qc_dir: null

  time_depth_dir: <time-depth-dir>
  well_trace_dir: <well-trajectory-dir>
  well_tops_file: <well-tops-file>

  target_interval:
    top_horizon: <top-horizon-file>
    bottom_horizon: <bottom-horizon-file>
    margin_top_ms: 100.0          # 层位之上冗余时间宽度
    margin_bottom_ms: 100.0       # 层位之下冗余时间宽度
    twt_unit: auto

  seismic:
    file: <seismic-volume-file>
    type: zgy

  enabled_routes:                 # 当前启用的路径
    - vertical_with_tdt
    - vertical_anchor_from_tops
    - deviated_with_tdt

  coarse_correction:
    anchor:
      enabled: true
      apply_to_routes:
        - vertical_anchor_from_tops
      config_file: experiments/well_auto_tie_anchors.yaml
    manual_shift:
      default_ms: 0.0
      config_file: experiments/well_auto_tie_manual_shifts.yaml

  reject:
    allow_near_outside: false
    min_tie_samples: 64
    max_trajectory_outside_fraction: 0.05
```

### `target_interval`

脚本在井口 XY 处读取顶底解释层位的 TWT，加上 `margin_top_ms` 和 `margin_bottom_ms` 的冗余，构成标定目标窗。报告里的层位名从文件名自动推断，便于追溯输入；图件里的目标层位线只标注 `top` / `bottom`。

### `enabled_routes`

目前有三条可用的路径：

| 路径 | 适合的井 | 关键资产 |
|------|---------|---------|
| `vertical_with_tdt` | 直井，有 Petrel 时深表 | 时深表 + 预处理 LAS |
| `vertical_anchor_from_tops` | 直井，无时深表 | 井分层 + 解释层位 + 预处理 LAS |
| `deviated_with_tdt` | 斜井，有 Petrel 时深表和井轨迹 | 时深表 + 井轨迹 + 预处理 LAS |

脚本自身的保守默认只启用前两条。主配置 `experiments/common.yaml` 额外启用了 `deviated_with_tdt`。如果你暂时不想跑斜井，从配置里删掉它即可——对应井会在 `well_tie_plan.csv` 里显示为 `skipped_disabled`，不会报错。

`deviated_anchor_from_tops`（斜井、无时深、有轨迹和分层）还没有落地，设计文档在 `docs/guide/deviated-well-src-cup-refactor.md`。

### `coarse_correction`

粗标定发生在 `wtie` 细标定之前，当前支持两种来源：

| 来源 | 作用 |
|------|------|
| `anchor` | 用井分层 MD 和解释层位 TWT 计算逐井锚点。路径 2 用它建初始 TDT；路径 1/3 如果显式加入 `apply_to_routes`，则用它对已有 Petrel TDT 做逐井整体 TWT shift。 |
| `manual_shift` | 用户手动给 TDT 加一个常数 TWT 偏移，单位 ms。三条已实现路径都支持，默认 0。 |

对有 Petrel TDT 的路径，锚点 shift 的计算方式是逐井的：

```text
anchor_shift_s = horizon_twt_at_this_well_anchor_xy - interp(petrel_tdt, anchor_md)
shifted_tdt.twt = petrel_tdt.twt + anchor_shift_s + manual_shift_s
```

这里要求 `anchor_md` 落在 Petrel TDT 的 MD 范围内；否则脚本会失败，不会把锚点夹到 TDT 端点去估算 shift。

对 `vertical_anchor_from_tops`，锚点不是“平移已有 TDT”，而是用于从声波曲线积分建立初始 TDT；随后仍会叠加 `manual_shift_s`。

锚点选择规则：

- 锚点文件里的 `anchors.default` 是全局默认锚点。
- 锚点文件里的 `anchors.wells.<well-name>` 可以为单井覆盖 `well_top`、`horizon`、`event`、`twt_unit`。
- 直井锚点的解释层位 TWT 在井口 XY 处读取。
- 斜井 `deviated_with_tdt` 如果启用锚点粗标定，解释层位 TWT 在锚点 MD 对应的轨迹 XY 处读取。
- `event` 目前只写入 `anchor_report.csv` 作为审计信息，不参与层位采样或 TWT 修正。

锚点文件形如：

```yaml
anchors:
  default:
    well_top: <marker-name>
    horizon: <horizon-file>
    event: peak
    twt_unit: auto
  wells:
    <well-name>:
      well_top: <marker-name>
      horizon: <horizon-file>
```

手动偏移规则：

- `manual_shift.default_ms` 是全局默认值。
- `manual_shift.config_file` 指向单井手动偏移文件；文件里的 `manual_shift.wells_ms.<well-name>` 覆盖单井，优先级最高。

单井手动偏移文件形如：

```yaml
manual_shift:
  wells_ms:
    <well-name>: 0.0
```

默认配置下，只有 `vertical_anchor_from_tops` 启用锚点；`vertical_with_tdt` 和 `deviated_with_tdt` 的锚点粗标定关闭。所有路径的手动偏移默认都是 0。

### `reject`

`reject` 控制第四步自己的拒绝条件。曲线有效比例不在这里重复判定：第三步已经写出 `usable_p_sonic` 和 `usable_density`，第四步只消费这两个结果。

| 参数 | 含义 |
|------|------|
| `allow_near_outside` | 路由阶段是否允许 `survey_position == near_outside` 的井进入标定。默认 `false`，只接受工区内井。 |
| `min_tie_samples` | 时深表、曲线窗口或斜井裁剪后的最少样点数；低于该值则失败。 |
| `max_trajectory_outside_fraction` | 目标窗口内允许落到工区外的最大轨迹样点比例，仅对 `deviated_with_tdt` 生效。 |

斜井轨迹在目标窗口内采样时，部分 TWT 样点对应的轨迹 XY 可能落到工区之外。脚本的处理逻辑是：

- 出界比例 ≤ `max_trajectory_outside_fraction`：裁剪到最长连续工区内 TWT 段，在裁后的窗口内做标定。
- 出界比例 > `max_trajectory_outside_fraction`：整井失败，原因记 `trajectory_outside_fraction_exceeded`。
- 裁剪后样点数 < `min_tie_samples`：整井失败，原因记 `trajectory_inside_tie_samples_too_few`。

---

## 脚本在做什么

脚本分四步：**路由 → 准备 TDT 和曲线 → 取地震道 → 细标定**。

### 第一步：路由

join 前三步的井清单、曲线可用性和轨迹 QC 结果，生成 `well_tie_plan.csv`，为每口井匹配一条路径：

| 条件 | 路径 | 状态 |
|------|------|------|
| 直井，有时深，曲线可用 | `vertical_with_tdt` | 已实现 |
| 直井，无时深，有分层，曲线可用 | `vertical_anchor_from_tops` | 已实现 |
| 斜井，有时深，有轨迹，曲线可用 | `deviated_with_tdt` | 已实现 |
| 斜井，无时深，有轨迹+分层，曲线可用 | `deviated_anchor_from_tops` | 仅识别，未实现 |
| 不满足以上任一 | `rejected` | — |

`route_status` 有三种：`planned`（会执行）、`skipped_disabled`（路径未启用）、`rejected`（不满足条件）。

### 第二步：准备初始时深表、粗校正和测井窗口

三条路径都使用 MD 域时深表（横轴是测量深度 MD，纵轴是双程旅行时 TWT）：

- **有时深表的路径**（`vertical_with_tdt`、`deviated_with_tdt`）：读取 Petrel 时深表；如果启用了锚点粗校正或手动偏移，先整体平移 TWT；如果它只覆盖了目标窗的一部分，再用声波曲线从时深表端点向上或向下补齐。
- **锚点路径**（`vertical_anchor_from_tops`）：取一口井的某个分层 MD 和对应解释层位的 TWT 作为锚点，沿声波曲线向上向下积分，建出初始 TDT；如果配置了手动偏移，再整体平移这个初始 TDT。

如果时深表完全不覆盖目标窗口，直井会自动改走锚点路径；斜井目前直接失败。

### 第三步：取地震道

**直井**很简单：在井口 XY 处读一条地震道。

**斜井**不同——井眼不是垂直的，不同深度的轨迹点对应不同的地面 XY。如果还在井口读一条道，深部的标定就对不上。所以斜井需要沿轨迹逐点取道：

1. 把目标窗口的 TWT 轴上的每个样点，通过时深表转成 MD，再查轨迹得到该 MD 处的 XY 坐标。
2. 把每个 XY 吸附到最近的 inline/xline 地震道上。
3. 如果有少量样点落到了工区外面（≤ 5%），裁剪到最长连续工区内段。
4. 对其中用到的每条唯一地震道各读一次，再按 TWT 样点逐点取出对应时刻的振幅值，拼成一条"沿轨迹地震道"。

第一步版本用的是最近道采样，不做空间插值。每个样点落在哪条道上、是否在工区内，全部写进 `trace_sample_plan_<well>.csv`。

### 第四步：细标定

把准备好的 MD 域曲线、地震道和时深表交给 `wtie` 做优化：调整时深关系让合成记录与地震匹配得更好。输出优化后的 TDT、子波、合成记录和 QC 指标。

这里传给 `wtie` 的 `wellpath` 参数是 `None`，因为斜井的地震道已经在第三步沿轨迹拼好了，`wtie` 不需要再操心空间定位。

---

## 输出文件

所有文件在 `<output_root>/well_auto_tie_<timestamp>/` 下：

| 文件 | 内容 |
|------|------|
| `well_tie_plan.csv` | 一井一行的路由计划 |
| `well_tie_metrics.csv` | 已执行井的标定指标和输出路径 |
| `rejected_wells.csv` | 路由阶段被拒绝的井及原因 |
| `tie_window_report.csv` | 目标窗口、实际窗口、TDT 来源、粗校正量和裁剪原因 |
| `anchor_report.csv` | 使用锚点的井的锚点 MD/TWT、层位采样信息和锚点作用 |
| `wavelet_inventory.csv` | 成功井导出的候选子波清单，供第五步使用 |
| `wavelets/wavelet_201ms_<well>.csv` | 裁剪并能量归一化后的子波 |
| `time_depth/initial_tdt_<well>.csv` | 进入 `wtie` 前的初始 TDT，含 `source` 列 |
| `time_depth/optimized_tdt_<well>.csv` | `wtie` 优化后的 TDT |
| `petrel_checkshots/optimized_tdt_<well>.txt` | Petrel checkshots 格式的细标定后时深表，便于导入地质软件 |
| `filtered_las/filtered_logs_<well>.las` | 用本井 auto-tie 选中的日志滤波参数重建的 MD 域 LAS，供第五步和地质软件使用 |
| `synthetic_qc/tie_qc_<well>.csv` | 地震、反射系数、合成记录和残差 |
| `seismic_trace/seismic_trace_<well>.csv` | 实际用于标定的地震道（直井是井旁道，斜井是沿轨迹拼接道） |
| `trace_sample_plan/trace_sample_plan_<well>.csv` | 斜井样点级落道明细；直井通常没有 |
| `figures/<well>/*.png` | TDT 图、合成匹配图、子波图 |
| `run_summary.json` | 输入路径、路由统计、失败统计、逐井补充信息 |

`tie_window_report.csv` 会记录 `coarse_anchor_shift_ms`、`coarse_manual_shift_ms` 和 `coarse_total_shift_ms`。如果某口井没有启用锚点且手动偏移为 0，这三个值就是 0。

`time_depth/optimized_tdt_<well>.csv` 是工作流内部格式，保留正秒 `twt_s` 和正米 `md_m`；`petrel_checkshots/optimized_tdt_<well>.txt` 是地质软件导入格式，沿用 `export_vertical_tdt_to_petrel_checkshots()` 的口径导出。`filtered_las/filtered_logs_<well>.las` 只包含第五步需要的标准曲线：`DT_USM`（`us/m`）和 `RHO_GCC`（`g/cm3`）。

这里的“本井 auto-tie 选中的日志滤波参数”指 `wtie` 自动标定优化结束后返回的 `best_parameters` 中与日志滤波有关的三项：`logs_median_size`、`logs_median_threshold`、`logs_std`。脚本用它们在 MD 域重新滤波本井输入曲线，再把滤波后的 `Vp` 转回 `DT_USM`、`Rho` 写成 `RHO_GCC`。同一组 `best_parameters` 里的 `table_t_shift` 只用于时深表优化，不用于改写 `filtered_las` 曲线。

### `trace_sample_plan_<well>.csv`

斜井路径最重要的审计文件，每个 TWT 样点一行：

| 字段 | 含义 |
|------|------|
| `twt_s` / `md_m` | 样点的 TWT 和对应的 MD |
| `x_m` / `y_m` | 该 MD 在井轨迹上的 XY |
| `inline_float` / `xline_float` | XY 投影到工区的浮点线号 |
| `nearest_inline` / `nearest_xline` | 吸附到的最近道线号 |
| `inline_index` / `xline_index` | 最近道的数组索引 |
| `flat_idx` | 地震体内部 trace 编号 |
| `survey_position` | `inside` 或 `outside` |
| `used_for_tie` | 裁剪后是否实际进入标定 |

即使整井失败，这份文件通常也已经写出。对照它可以看出失败是轨迹出界、工区转换失败，还是最长 inside 窗口太短。

---

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| `tdt_no_target_window_overlap` | 时深表和目标窗口完全不重叠 | 检查 Petrel TDT 的时间范围、解释层位单位和目标窗配置 |
| `trajectory_outside_fraction_exceeded` | 斜井轨迹在目标窗内超过 5% 的样点出工区 | 看 `trace_sample_plan`，确认轨迹或工区几何是否有问题 |
| `trajectory_inside_tie_samples_too_few` | 裁剪后连续 inside 样点太少 | 放宽窗口 margin、检查轨迹，或暂时跳过这口井 |
| `TWT axis outside table range` | 地震采样轴超出了 TDT 范围 | 检查声波拓延是否有足够曲线覆盖 |
| `Seismic trace has zero standard deviation` | 读到的那段地震道完全没有振幅变化 | 检查地震体、窗口范围和道索引 |

---

## 如何阅读结果

### 第一步：看 `run_summary.json`

直接看顶层计数：

```
route_counts / route_status_counts / tie_status_counts
planned_run_count / successful_tie_count
```

`skipped_disabled` 多 → 配置没启用对应路径。`failed` 集中在斜井 → 优先看 `result_extras.<well>.trace_sampling` 和该井的 `trace_sample_plan`。

### 第二步：看 `well_tie_plan.csv`

确认每口井为什么走那条路径。常见标签：

- `route_disabled_deviated_with_tdt`：斜井有时深但路径被禁用
- `survey_position_outside`：井口不在工区内
- `unusable_p_sonic` / `unusable_density`：第三步判定曲线不可用
- `no_time_depth` / `no_well_trace` / `no_well_tops`：缺少路径需要的资产

### 第三步：看 `well_tie_metrics.csv`

- `initial_corr` / `optimized_corr`：优化前后的相关系数
- `optimized_nmae`：归一化误差，越低越好
- `best_table_shift_ms`：`wtie` 找到的整体时移量
- `tie_window_start_s` / `tie_window_end_s`：实际标定窗口
- `tdt_support_class`：TDT 来源（原始表、声波拓延、锚点积分）
- `petrel_checkshot_file`：细标定后时深表的 Petrel checkshots 导入文件
- `filtered_las_file`：第五步默认读取的滤波后 LAS 文件

相关系数提高不等于结果可靠——必须结合 TDT 图、合成匹配图和子波形态一起看。

### 第四步：看 `tie_window_report.csv`

回答"脚本到底拿哪段曲线和地震做的标定"。如果 `window_clip_reason` 里有 `trajectory_inside_crop`，说明斜井目标窗因为出界样点被裁剪过。

### 第五步：看图

每口成功井抽查这几张图：

- `figures/<well>/time_depth_table.png` — 初始 vs 优化 TDT 对比
- `figures/<well>/synthetic_match.png` — 合成记录与地震道匹配
- `figures/<well>/wavelet.png` — 提取的子波形态

关注：优化后的 TDT 有没有不合理的大幅扭曲？合成记录是不是只在局部强行对齐？子波有没有过窄、偏相或振铃异常？

---

## 留到第二轮

- `deviated_anchor_from_tops`：斜井无时深、有轨迹和分层的第四条路径。
- 斜井地震采样从最近道升级到双线性或多道加权。
- 斜井轨迹 inline/xline 随 TWT 变化的专用 QC 图。
- 密井网下多井落到同一 trace/time 样点时的冲突诊断。
