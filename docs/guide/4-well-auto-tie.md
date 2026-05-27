# 04 井震自动标定

本文讨论第四个规划脚本：`well_auto_tie.py`。

它接在 `log_preprocess.py` 后面，负责把井曲线、时深关系、地震道和子波提取流程组织起来，产出每口井的井震标定结果。虽然历史深度域脚本叫 `vertical_well_auto_tie_depth.py`，但时间域第一版建议不要继续叫 `vertical_well_auto_tie.py`，因为本脚本需要同时路由直井和斜井。

第一版按复杂度递增实现三条路径：

1. 有时深、直井。
2. 没有时深、有井分层、直井。
3. 有时深、有井轨迹、斜井。

没有时深、有井轨迹、有井分层的斜井路径先只识别并拒绝，不实现。

按当前已跑通的前三步产物估算，前三条可实现路径分别对应 `6 + 2 + 4 = 12` 口井。实现时不要三条路径并行铺开，建议先完成路径 1 的端到端闭环，再接路径 2 的锚点建表，最后处理路径 3 的斜井轨迹取道。

## 目标

`well_auto_tie.py` 回答五件事：

1. 每口井满足哪条井震标定路径，或为什么被拒绝。
2. 如何为该井构造初始时深关系。
3. 如何读取与井匹配的地震道：直井取井口处道，斜井沿井轨迹取道。
4. 如何调用 `wtie` auto-tie 流程微调时深关系并提取子波。
5. 如何输出可审计的标定指标、优化后时深表、合成记录和 QC 图。

第四步不再自己扫描原始 LAS 或猜测曲线是否可用。它读取前三步产出的 manifest 后做路由。

## 输入

- 第一阶段清单：`well_inventory.csv`。
- 第二阶段曲线筛选：`well_curve_screen.csv`、`las_curve_inventory.csv`。
- 第三阶段预处理状态：`well_preprocess_status.csv`。
- 第三阶段预处理 LAS：`preprocessed_las/*.las`。
- 时深表目录：`data/time_depth_table`。
- 井轨迹目录：`data/all_well_trace`。
- 井分层文件：`data/raw/well_tops`。
- 地震解释层位：例如 `data/interpre/H3-1`、`data/interpre/H7-1`。
- 地震体：`data/raw/obn-clipped-240-912-872-1544.zgy`。
- 子波提取网络：沿用深度域 auto-tie 的 `tutorial_model` 和 `tutorial_params`。
- 可选锚点配置：没有时深时，指定一组井分层和地震解释层位的对应关系。

建议配置片段：

```yaml
well_auto_tie:
  source_runs:
    mode: latest
    well_inventory_dir: null
    las_curve_screen_dir: null
    log_preprocess_dir: null

  inventory_file: null
  curve_screen_file: null
  preprocess_status_file: null
  preprocessed_las_dir: null

  time_depth_dir: time_depth_table
  well_trace_dir: all_well_trace
  well_tops_file: raw/well_tops
  interpretation:
    top_horizon: interpre/H3-1
    bottom_horizon: interpre/H7-1
  target_interval:
    top: H3-1
    bottom: H7-1
    margin_top_ms: 100.0
    margin_bottom_ms: 100.0
    twt_unit: auto
  seismic:
    file: raw/obn-clipped-240-912-872-1544.zgy
    type: zgy

  enabled_routes:
    - vertical_with_tdt
    - vertical_anchor_from_tops
    - deviated_with_tdt

  implementation_order:
    - vertical_with_tdt
    - vertical_anchor_from_tops
    - deviated_with_tdt

  coarse_anchor:
    enabled: true
    apply_to_routes:
      - vertical_anchor_from_tops
    config_file: experiments/well_auto_tie_anchors.yaml

  tutorial_model: tutorial/trained_net_state_dict.pt
  tutorial_params: tutorial/network_parameters.yaml
  target_crop_ms: 201.0

  search_space:
    logs_median_size_values: [51, 71, 91, 111]
    logs_median_threshold_bounds: [0.5, 3.0]
    logs_std_bounds: [20, 50]
    table_t_shift_bounds: [-0.030, 0.030]

  search_params:
    num_iters: 60
    similarity_std: 0.02

  wavelet_scaling:
    min_scale: 50000
    max_scale: 500000
    num_iters: 60

  reject:
    allow_near_outside: false
    min_valid_log_fraction: 0.7
    min_tie_samples: 64
```

`source_runs.mode: latest` 表示脚本自动从 `scripts/output` 下寻找最新的前置产物。若需要复现实验，可以显式填写 `inventory_file`、`curve_screen_file`、`preprocess_status_file` 和 `preprocessed_las_dir`。不要把 `YYYYMMDD_HHMMSS` 占位符长期写死在配置里。

`target_interval` 是实际井震标定窗口。脚本在井口 XY 处读取 `H3-1` 和 `H7-1` 解释层位的 TWT，再在时间域向上、向下各拓展默认 `100 ms`。路径一和路径二都会使用这个窗口裁剪井曲线、初始时深表和地震道。

`target_crop_ms` 是成功标定后导出候选子波时使用的中心裁剪长度，不再表示井震标定输入窗口。

`coarse_anchor` 是进入 `wtie` 细标定前的粗标定配置。它不是手填整体偏移，而是用人工指定的一组 `well_top -> horizon` 锚点自动建立初始时间基准：

- 对无时深直井路线，锚点用于给声波积分提供绝对 TWT 基准，第一版默认启用。
- 对有时深路线，锚点可以用于计算原始时深表的整体 TWT shift，但默认关闭；只有显式把 `vertical_with_tdt` 或 `deviated_with_tdt` 加入 `apply_to_routes` 时才执行。

当前工区第一版统一使用 `H3-1` 作为锚点。`H7-1` 作为解释层位资产保留，但不参与第一版锚定建表。

## 输出

默认输出目录建议为：

```text
scripts/output/well_auto_tie_<timestamp>/
```

核心文件：

- `well_tie_plan.csv`：一井一行的路由结果。
- `well_tie_metrics.csv`：一井一行的标定指标。
- `rejected_wells.csv`：被拒绝的井及原因。
- `wavelet_inventory.csv`：一条成功子波一行，供第五步作为候选子波清单。
- `tie_window_report.csv`：一井一行的目标窗口、实际窗口、TDT 支持类型和裁剪原因。
- `wavelets/wavelet_201ms_<well>.csv`：每口成功井的裁剪归一化子波。
- `time_depth/initial_tdt_<well>.csv`：窗口内初始时深表，包含 `source` 列。
- `time_depth/optimized_tdt_<well>.csv`：`wtie` 优化后的时深表。
- `synthetic_qc/tie_qc_<well>.csv`：地震、反射系数、合成记录和残差。
- `seismic_trace/seismic_trace_<well>.csv`：用于标定的井旁或轨迹地震道。
- `figures/<well>/*.png`：优化目标、时深关系、井震匹配、子波等 QC 图。
- `run_summary.json`：输入、配置、路由统计、失败统计。

`well_tie_plan.csv` 建议字段：

| 字段 | 含义 |
| --- | --- |
| `well_name` | 井名 |
| `route` | `vertical_with_tdt`、`vertical_anchor_from_tops`、`deviated_with_tdt`、`deviated_anchor_from_tops`、`rejected` |
| `route_status` | `planned`、`skipped_disabled`、`rejected` |
| `wellbore_class_initial` | 第一阶段基于井头底孔坐标的初分井型 |
| `wellbore_class_qc` | 第四步读取井轨迹后复核的井型；无轨迹时可为空 |
| `has_time_depth` | 第一阶段时深存在性 |
| `has_well_trace` | 第一阶段井轨迹存在性 |
| `has_well_tops` | 第一阶段井分层存在性 |
| `usable_p_sonic` | 第三阶段纵波时差可用性 |
| `usable_density` | 第三阶段密度可用性 |
| `input_las` | 预处理 LAS 路径 |
| `time_depth_file` | 时深表路径；没有则为空 |
| `well_trace_file` | 井轨迹路径；直井路径可为空 |
| `reasons` | 路由说明或拒绝原因 |

`well_tie_metrics.csv` 建议字段：

| 字段 | 含义 |
| --- | --- |
| `well_name` | 井名 |
| `route` | 实际执行路径 |
| `tie_status` | `success`、`failed` |
| `initial_corr` | 初始合成记录与地震相关系数 |
| `optimized_corr` | 优化后相关系数 |
| `optimized_nmae` | 优化后归一化 MAE |
| `best_table_shift_ms` | 最优整体时移 |
| `tie_window_start_s` | 实际标定窗口起点 |
| `tie_window_end_s` | 实际标定窗口终点 |
| `tdt_support_class` | 初始时深支持类型，如 `original_full_window`、`original_with_sonic_extension`、`anchor_integrated` |
| `original_tdt_window_fraction` | 原始时深表覆盖目标窗口的比例 |
| `wavelet_file` | 输出子波 |
| `optimized_tdt_file` | 优化后时深表 |
| `qc_figure_dir` | QC 图目录 |
| `reasons` | 警告或失败原因 |

`wavelet_inventory.csv` 建议字段：

| 字段 | 含义 |
| --- | --- |
| `source_well` | 子波来源井 |
| `route` | 来源井 auto-tie 路径 |
| `wavelet_file` | 子波 CSV 路径 |
| `dt_s` | 子波采样间隔 |
| `n_samples` | 子波采样点数 |
| `tie_corr` | 来源井第四步优化后相关系数 |
| `tie_nmae` | 来源井第四步优化后 NMAE |
| `usable_as_candidate` | 是否进入第五步候选池 |
| `reasons` | 跳过原因 |

## 路由规则

第四步先 join 三份 manifest：

- `well_inventory.csv`
- `well_curve_screen.csv`
- `well_preprocess_status.csv`

然后按下表路由：

| route | 条件 | 第一版动作 |
| --- | --- | --- |
| `vertical_with_tdt` | 工区内或允许边缘井；直井；有时深；`usable_p_sonic` 和 `usable_density` | 实现 |
| `vertical_anchor_from_tops` | 工区内或允许边缘井；直井；无时深；有井分层；`usable_p_sonic` 和 `usable_density`；有人工锚点配置 | 实现 |
| `deviated_with_tdt` | 工区内或允许边缘井；斜井；有时深；有井轨迹；`usable_p_sonic` 和 `usable_density` | 实现 |
| `deviated_anchor_from_tops` | 斜井；无时深；有井轨迹；有井分层；有曲线 | 暂时拒绝 |
| `rejected` | 其他情况 | 拒绝 |

说明：

- 这里的“有井轨迹”指 `data/all_well_trace` 中的井轨迹/井斜文件，不是 LAS 井径曲线 `caliper`。
- LAS 井径曲线可以作为井眼质量 QC 的辅助信息，但不是斜井路径的必要条件。
- 第一阶段的 `wellbore_class` 是初分。斜井路径执行前必须正式读取井轨迹并复核。
- `near_outside` 井默认拒绝，除非配置 `allow_near_outside: true`。

`well_tie_plan.csv` 应同时保留 `wellbore_class_initial` 和 `wellbore_class_qc`。路由可以先用初分井型生成候选 plan，但执行前必须用轨迹 QC 更新最终路径；如果复核结果和初分冲突，写入 `reasons`。

当前数据按前三步结果路由后，第一版优先实现的路径数量为：

| route | 数量 | 当前井 |
| --- | ---: | --- |
| `vertical_with_tdt` | 6 | `PH1`、`PH13`、`PH2`、`PH3`、`PH4`、`PH5` |
| `vertical_anchor_from_tops` | 2 | `B3`、`BG2` |
| `deviated_with_tdt` | 4 | `BA6S`、`PH6`、`PH7`、`PH8` |

这也是推荐实现顺序。`deviated_anchor_from_tops` 当前有较多候选井，但同时缺少时深表且需要斜井锚点积分，第一版只写入拒绝结果。

## 路径一：有时深、直井

`vertical_with_tdt` 是最接近现有深度域 `vertical_well_auto_tie_depth.py` 的路径，也是第四步第一版应最先落地的端到端闭环。

处理逻辑：

1. 读取预处理 LAS，取 `DT_USM` 和 `RHO_GCC`。
2. 将慢度 `DT_USM` 转成速度 `Vp`，和密度构造 `grid.LogSet`。
3. 读取该井已有时深表，构造 `grid.TimeDepthTable`。
4. 在井口 XY 处读取 `H3-1 -> H7-1` 目的层时间窗，并应用配置的上下拓展。
5. 若原始时深表完全不接触目的层窗口，则改走 `vertical_anchor_from_tops`，原因记录为 `tdt_no_target_window_overlap_reroute_anchor`。
6. 若原始时深表只覆盖部分窗口，则从原始 TDT 端点开始用 `DT_USM` 积分向上或向下拓延。
7. 按实际可用窗口裁剪井曲线、初始时深表和时间域地震道。
8. 用窗口内初始 table 调用 `autotie.tie_v1` 做微调。
9. 裁剪并能量归一化子波，输出优化后时深表和 QC。

这条路径不再像深度域脚本那样默认从 Vp 从零构造本地时深表，而是以已有时深表为初始约束。Vp 主要用于生成反射系数、合成记录，以及在原始时深表端点附近补齐目的层窗口。

可选锚点粗标定计算方式：

```text
anchor_md = well_top(H3-1).MD
anchor_twt = horizon(H3-1).TWT at well XY
tdt_twt_at_anchor = interp(original_tdt, anchor_md)
coarse_shift_s = anchor_twt - tdt_twt_at_anchor
shifted_tdt.twt = original_tdt.twt + coarse_shift_s
```

这样既保留原始时深表的形状，又用井分层和解释层位把整体时间基准拉到合理位置。`wtie` 后续只负责细标定。但这条路径默认不执行该步骤，避免在已有时深表质量未知前自动改动全部有时深井。

## 路径二：无时深、有井分层、直井

`vertical_anchor_from_tops` 用人工选定的层位锚点构造初始时深表。它是第二个实现路径：先只面向直井，先把锚点建表和 wtie 细标定跑通，再考虑更复杂的斜井无时深路径。

前置条件：

- 井分层文件中有该井的目标层位 MD。
- 地震解释层位能在井口 XY 处取到 TWT。
- 人工配置明确哪一个井分层对应哪一个地震强轴或解释层位。

当前工区路径 2 的锚点统一使用 `H3-1`。`H7-1` 可以继续作为解释层位资产保留，但第一版不参与锚定建表。

锚点配置示例：

```yaml
anchors:
  default:
    well_top: H3-1
    horizon: interpre/H3-1
    event: peak
    twt_unit: s
  wells:
    BG2:
      well_top: H3-1
      horizon: interpre/H3-1
      event: peak
```

处理逻辑：

1. 从井分层得到锚点 MD。
2. 从地震解释层位得到井口处对应锚点 TWT。
3. 从该锚点开始，沿井曲线向上、向下积分纵波慢度。
4. 按 `H3-1 -> H7-1` 目的层时间窗裁剪锚点积分得到的初始 table、井曲线和地震道。
5. 直井第一版近似 `MD ~= TVD`；如果后续发现直井也有明显偏斜，转入斜井路径。
6. 用窗口内初始 table 调用 `autotie.tie_v1` 微调。

积分关系按双程时间处理：

```text
dTWT_s = 2 * DT_USM * dZ_m * 1e-6
```

其中 `DT_USM` 是微秒每米，`dZ_m` 第一版对直井取 MD 增量。

这条路径只适用于经过复核仍可近似为直井的井。只要轨迹显示 MD 与 TVD 差异不可忽略，就不能使用这个积分近似，应转入斜井路径或拒绝。

第一版只允许一组锚点，避免多层位约束和声波积分之间的误差分配问题。后续如果确实需要顶底双锚点，再单独扩展建表逻辑。

注意：路径 1 和路径 2 可以使用同一份锚点配置，但默认只有路径 2 使用。路径 1 只有显式开启时，才用锚点修正已有时深表的整体时间偏移；路径 2 没有已有时深表，锚点本身就是声波积分的绝对时间基准。

## 路径三：有时深、有井轨迹、斜井

`deviated_with_tdt` 是第一版新增的关键路径，但实现顺序排在路径 1 和路径 2 之后。原因是它不再只取井口地震道，而要把已有时深表、井轨迹和地震取样串起来。

处理逻辑：

1. 读取预处理 LAS，构造 `grid.LogSet`。
2. 读取已有时深表，构造 `grid.TimeDepthTable`。
3. 读取井轨迹文件，得到 MD、X、Y、TVD/TVDSS 的关系。
4. 默认直接使用原始时深表；如果显式对 `deviated_with_tdt` 启用 `coarse_anchor`，才用锚点计算整体 TWT shift。斜井锚点的 horizon TWT 应在锚点 MD 对应的轨迹 XY 处读取，而不是默认井口 XY。
5. 将粗标定后的时深表和井轨迹对齐：用 TWT 反查 MD，再由 MD 插值得到对应 XY。
6. 沿时间采样轴，从地震体中按 `XY(TWT)` 抽取轨迹地震道。
7. 用轨迹地震道、井曲线和已有时深表调用 `autotie.tie_v1` 微调。
8. 输出轨迹地震道、优化后时深表、子波和 QC。

关键点：

- 斜井不能只取井口所在地震道。
- 只要涉及轨迹附近的真实物理距离、邻近道或插值，就必须使用 `open_survey()`、`line_to_coord()`、`coord_to_line()` 和真实 XY 计算，不能把 inline/xline 步长当米。
- 第一版可以先用最近道采样，但必须输出 `trace_sample_plan_<well>.csv`，记录每个 TWT 样点使用的 inline/xline 和 XY。若同一条轨迹跨越大量地震道，不能逐样点重复打开 ZGY；应先把轨迹映射到唯一道集合，批量读取后再按时间轴拼接。
- 最近道采样会引入阶梯状空间跳变。文档第一版接受这个近似，但 QC 图必须显示轨迹 inline/xline 随 TWT 的变化，便于判断是否需要升级到双线性或多道加权。

## 暂时拒绝路径

`deviated_anchor_from_tops` 暂时只识别，不实现。

原因是它同时缺少已有时深表，又需要沿斜井轨迹把 MD、TVD、XY、TWT 和层位锚点串起来。这里的误差来源比直井锚点路径多很多：

- 井轨迹采样和测井采样需要对齐。
- 井分层 MD 需要投影到轨迹 XY。
- 地震层位 TWT 需要在轨迹点而不是井口点读取。
- 纵波积分应沿 TVD 或实际路径讨论，不能简单套直井公式。

第一版把这类井写入 `rejected_wells.csv`，原因设为 `deviated_anchor_route_not_implemented`。

## 模块边界

第四步会推动 `cup` 新增井震标定相关能力。模块归属以 `斜井支持的 src/cup 重构规划` 为准：井轨迹、时深转换、空间样点和地震取样是跨步骤能力，不放进脚本，也不继续塞进 `cup.seismic.survey`。

### 建议新增

`cup.well.tie`

- `TieRoute`：路由枚举。
- `WellTiePlan`：单井路由和输入资产。
- `WellTieResult`：单井标定结果和指标。
- `build_tie_plan(...) -> list[WellTiePlan]`
- `run_vertical_with_tdt(plan, context) -> WellTieResult`
- `run_vertical_anchor_from_tops(plan, context) -> WellTieResult`
- `run_deviated_with_tdt(plan, context) -> WellTieResult`

`cup.well.td`

- `load_petrel_time_depth_table(path) -> grid.TimeDepthTable`
- `validate_time_depth_table(table, log_basis_md, trajectory=None)`
- `build_tdt_from_anchor(log, anchor_md, anchor_twt_s)`
- `merge_tdt_with_log_basis(table, log_basis_md)`

`cup.well.td` 负责时深表、MD/TWT、锚点建表和声波拓延。标准 LAS 到 `grid.LogSet` 的读取由 `cup.well.las.load_vp_rho_logset_from_standard_las()` 承担。`cup.well.tie` 只负责编排 route、调用 wtie Adapter 和整理 tie artifact，不直接承载时深/曲线转换细节。

`cup.well.trajectory`

- `WellTrajectory.from_petrel_trace(path)`
- `validate_trajectory_for_well(trajectory, log_basis_md)`
- `trajectory.to_wtie_wellpath()`

`cup.well.spatial_samples`

- `trajectory_position_at_md(trajectory, md)`
- `trajectory_position_at_twt(trajectory, table, twt)`

`cup.well.spatial_samples` 只负责把已经明确了 MD/TWT 的井样点落到 `x/y`、`inline/xline` 和 trace/sample 上；它不负责解析时深表，也不负责把慢度曲线转成速度。

`cup.seismic.trace_sampling`

- `import_time_trace_at_xy(x, y) -> grid.Seismic`
- `import_time_trace_along_xy(twt, x, y, method="nearest") -> grid.Seismic`

这些函数应放在专门的地震取样 Module 里，不要在脚本中散写 ZGY 索引和坐标转换。

`trace_sampling` 落地时还需要深化 `SurveyContext` 的 Interface：当前 `cup.seismic.survey` 公开协议主要是 `coord_to_line()`、`line_to_coord()` 和 `import_seismic_at_well()`，但斜井批量取道需要批量 `coord_to_index`、邻道/flat index 计划、sample window 解析和重复 trace 去重。不要在脚本里直接访问 ZGY/SEG-Y 私有细节，应由 `SurveyContext` 或专门 Adapter 暴露这些能力。

### 继续复用

- `cup.well.las`：读取第三步预处理 LAS。
- `cup.well.preprocess`：第三步已经决定曲线可用性，第四步不重复清洗。
- `cup.well.wavelet`：子波裁剪、采样间隔检查和 CSV 读取。
- `cup.petrel.load.import_well_heads_petrel()`、`import_well_tops_petrel()`、`import_interpretation_petrel()`：Petrel 文本读取。
- `cup.seismic.survey.open_survey()`：地震工区入口。

### API 验证

本仓库里的 `wtie` 已有 `grid.LogSet`、`grid.TimeDepthTable`、`grid.WellPath`、`InputSet` 和 `autotie.tie_v1`。当前源码中 `InputSet` 的字段顺序为 `logset_md, seismic, table, wellpath=None`，`tie_v1(inputs, wavelet_extractor, modeler, wavelet_scaling_params, ...)` 以 `InputSet` 作为第一个参数。但第四步不是照搬深度域脚本，而是把已有时深表、时间域地震道和斜井轨迹组合起来。因此落地代码前仍需要用一口直井和一口斜井做最小验证：

- `InputSet(logset_md, seismic, table, wellpath)` 在本项目数据中应使用 MD 域 table 还是 TVDSS table。
- 直井是否需要显式构造 `grid.WellPath(md, kb)`，而不是传 `None`。
- 已有时深表的 TWT 单位、起点和地震 TWT 轴是否一致。
- `wavelet_scaling` 配置键是否需要转成 `wavelet_min_scale`、`wavelet_max_scale`。
- `autotie.tie_v1` 输出中的 table、wavelet、synthetic 字段是否满足当前输出 schema。

## 脚本层负责

`well_auto_tie.py` 负责：

- 读取配置和前三步 manifest。
- 构建 `well_tie_plan.csv`。
- 按 route 调用对应 handler。
- 组织输出目录和文件命名。
- 逐井容错执行，失败井写入 `rejected_wells.csv` 或 `well_tie_metrics.csv`。
- 汇总 `run_summary.json`。

它不应该自己实现：

- LAS 解析。
- 时深表列名兼容。
- 井轨迹插值。
- 地震体坐标转换。
- 沿轨迹取道。
- `wtie` 输入对象构造细节。

## 已定策略

- 第四步使用一个脚本 `well_auto_tie.py`，内部按 route 分发，不拆成三个脚本。
- 第一版实现 `vertical_with_tdt`、`vertical_anchor_from_tops`、`deviated_with_tdt`，实现顺序固定为路径 1 -> 路径 2 -> 路径 3。
- `deviated_anchor_from_tops` 第一版拒绝。
- 斜井路径依赖井轨迹/井斜文件，不依赖 LAS 井径曲线。
- 有时深的路径以已有时深表为初始 table，再用 `wtie` 微调。
- 无时深直井路径需要人工锚点配置，当前锚点统一使用 `H3-1`，不能全自动猜强轴。
- 模块边界以 `deviated-well-src-cup-refactor.md` 为准：`cup.well.tie` 只放 auto-tie 编排和 wtie Adapter；时深关系、轨迹、空间样点、地震取样分别进入 `cup.well.td`、`cup.well.trajectory`、`cup.well.spatial_samples` 和 `cup.seismic.trace_sampling`；标准 LAS 到 `LogSet` 的读取进入 `cup.well.las`。

## 留到第二轮

- 沿斜井轨迹取道使用最近道、双线性还是多道加权。
- 斜井路径里 `wtie` 的 stretch/squeeze 搜索空间是否需要比直井更窄。
- 锚点层位是否允许多锚点，而不是单锚点向上向下积分。
- 没有时深的斜井路径如何处理。
- 井网很密时，多口井共用或竞争同一地震道的冲突如何进入 auto-tie 权重。
