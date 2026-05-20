# 斜井支持的 src/cup 重构规划

本文不是某一个脚本的设计文档，而是第六步 `lfm_precomputed.py` 和第十步 `well_constraints.py` 共享的重构规划。

前五步已经把斜井支持推到了工作流前台：第四步需要沿井轨迹取地震道，第六步低频模型需要决定斜井曲线在空间上的控制位置，第十步井约束需要把斜井阻抗约束落到多个地震道和多个时间样点上。单独在每个脚本里补特殊判断，会让斜井逻辑散落在脚本层，后续很难验证。

因此，斜井支持应先在 `src/cup` 内形成稳定 Module，再由脚本调用。

## 当前阻力

### `wtie.grid.WellPath` 不够表达斜井

`wtie.processing.grid.WellPath` 当前只表达：

- `md`
- `tvdss`
- `kb`

它能支持 `MD -> TVDSS` 或 `MD -> TWT` 的轴转换，但不能表达：

- 轨迹点的 `x/y`
- 轨迹点的 `inline/xline`
- 某个 TWT 样点对应哪一个地震道
- 一口斜井是否穿过多个地震道

所以 `grid.WellPath` 可以继续作为 wtie Adapter 使用，但不能作为项目内完整井轨迹模型。

### `cup.seismic.modeling.WellControl` 假设一口井一个平面点

当前 `WellControl` 的 Interface 是：

```text
well_name, property_log, inline, xline, horizon_values
```

`build_layer_constrained_model()` 在每个层段比例切片上，从每口井取一个值，然后用这口井固定的 `inline/xline` 做二维插值。

这对直井是自然的，但对斜井不成立。斜井在目标层内可能穿过多个 trace；同一口井的上部、下部甚至同一层段内的不同采样点，都可能对应不同的 `inline/xline`。

### `ginn.anchor.LogAIAnchorBundle` 是 trace 级约束

当前 GINN 的井约束数据最终落成：

```text
flat_indices
target_log_ai[n_anchor_trace, n_sample]
anchor_weight[n_anchor_trace, n_sample]
```

这个结构本身可以容纳斜井，因为它允许一条 trace 上只有部分时间样点有约束。但现在的生成逻辑更接近“井头最近道整条曲线约束”。斜井需要先生成 `trajectory point -> trace/time sample` 的稀疏点云，再聚合成现有 `LogAIAnchorBundle`。

## 重构目标

这次重构不按“先近似、后完整”的节奏推进。时间域脚本还没有形成历史包袱，深度域工作流后续大概率也不会继续作为主线，因此斜井应作为主路径一次性进入 `src/cup` 的数据模型。

目标是让脚本层只表达业务决策：

- 哪些井进入 LFM；
- 哪些井进入 well constraints；
- 斜井使用完整轨迹时，控制点如何抽稀、加权和冲突处理；
- 冲突点如何取舍。

脚本层不应该自己反复实现：

- 轨迹文件解析；
- `MD/TWT/TVDSS/XY/inline/xline` 互转；
- 斜井采样点落到地震道；
- 多井、多轨迹点冲突聚合；
- 直井与斜井的两套分支数据结构。

## 实际井轨迹文件格式

当前 `data/all_well_trace` 下共有 Petrel 导出的 `.txt` 文件。典型文件头如下：

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
| `Z` | Petrel 导出的绝对高程/深度坐标；从样例看通常等于 `KB - TVD` |
| `TVD` | 从 KB 起算的真垂深，向下为正，单位 m |
| `DX`, `DY` | 相对井口偏移，单位 m |
| `AZIM` | 方位角，单位度 |
| `INCL` | 井斜角，单位度 |
| `DLS` | 狗腿严重度 |

这意味着我们不需要从 `INCL/AZIM` 重新积分轨迹，第一优先级应使用文件中已经给出的 `X/Y/TVD/MD`。`INCL/AZIM/DLS` 只作为 QC 辅助字段。

需要特别处理的情况：

- 有些文件只有两行，例如直井从井口到井底；这依然是合法轨迹。
- 有些斜井文件采样很密，例如 0.5 m 或 0.125 m。
- `Z` 与 `TVD` 的关系应做一致性检查：通常 `Z ~= KB - TVD`。
- 文件夹真实名称是 `data/all_well_trace`，不是 `data/all_well_traces`。

### TVDSS 口径

这是斜井重构里最容易错一个 KB 的地方，必须作为 Module Interface 的一部分固定下来。

现有 `cup.petrel.load.import_checkshots_petrel(depth_domain="tvdss")` 会把 Petrel checkshots 的 `Z` 取绝对值后放进 `grid.TimeDepthTable(tvdss=...)`；`export_vertical_tdt_to_petrel_checkshots()` 又把 `tdt.tvdss` 写成负的 Petrel `Z`，并用 `MD = |Z| + KB` 导出。这说明当前代码里的 `TimeDepthTable.tvdss` 实际是“向下为正的 TVDSS/depth below MSL”口径，而不是严格数学符号的海拔坐标。

因此新 `WellTrajectory` 不应让脚本散写 `tvdss = tvd_kb - kb` 或 `tvdss = kb - z`。建议在 `cup.well.trajectory` 中集中实现：

```text
tvdss_m = tvd_kb_m - kb_m
```

并在读取 Petrel trace 时同时校验：

```text
z_m ~= kb_m - tvd_kb_m
```

如果未来决定改成带符号海拔口径，也必须只改 Adapter 和 `depth_time`，不要让不同脚本混用两种 TVDSS 语义。所有 `WellTimeDepth`、`WellTrajectory.to_wtie_wellpath()`、Petrel checkshots 导入导出都要共享同一个约定。

## 建议新增核心 Module

### `cup.well.trajectory`

这是斜井支持最关键的 Module。它的 Interface 应该比 `wtie.grid.WellPath` 更深，负责项目内完整井轨迹表达。

建议对象：

```text
WellTrajectory
```

建议字段：

| 字段 | 含义 |
| --- | --- |
| `well_name` | 井名 |
| `md_m` | 测深 MD，单位 m |
| `tvd_kb_m` | 从 KB 起算的 TVD，向下为正，单位 m |
| `tvdss_m` | 项目内部 TVDSS 口径，单位 m；必须由 Adapter 明确换算，不能在调用处临时猜 |
| `z_m` | 原始 Petrel `Z` 列，保留用于 QC |
| `x_m` | 轨迹点 X |
| `y_m` | 轨迹点 Y |
| `dx_m`, `dy_m` | 相对井口偏移 |
| `azim_deg`, `incl_deg`, `dls` | 轨迹 QC 字段 |
| `kb_m` | KB 高程 |
| `metadata` | 来源文件、单位、QC 信息 |

建议方法：

| 方法 | 功能 |
| --- | --- |
| `from_petrel_trace(path)` | 读取当前 Petrel well trace txt |
| `to_wtie_wellpath()` | 转成 `wtie.processing.grid.WellPath`，供 wtie 轴转换使用 |
| `with_inline_xline(survey)` | 用 `SurveyContext` 补充每个轨迹点的 `inline/xline` |
| `position_at_md(md)` | 按 MD 插值得到 `x/y/tvdss/inline/xline` |
| `position_at_tvdss(tvdss)` | 按 TVDSS 插值得到轨迹位置 |
| `position_at_twt(twt, time_depth_table)` | 先由 TWT 找 MD 或 TVDSS，再找空间位置 |
| `target_interval(top_twt, bottom_twt, table)` | 提取目标层时间窗内的轨迹片段 |
| `representative_position(policy)` | 返回井口、目标层中点、目标层加权中心等代表点 |

需要注意：`WellTrajectory` 是项目内主模型；`wtie.grid.WellPath` 只是一个 Adapter，不反过来主导项目设计。

### `cup.well.depth_time`

这个 Module 负责井曲线、时深表和轨迹之间的域转换。

建议对象或函数：

| 名称 | 功能 |
| --- | --- |
| `WellTimeDepth` | 包装 `grid.TimeDepthTable`，保留来源、domain、单位、QC |
| `read_time_depth_table(path)` | 读取已有时深表，并统一 TWT 单位和 TVDSS/MD 口径 |
| `validate_time_depth_table(table, log_basis_md, trajectory=None)` | 检查时深表与测井 MD、轨迹 TVDSS 的覆盖关系 |
| `convert_log_md_to_twt(log, table, trajectory, dt_s)` | 统一 MD 曲线到 TWT |
| `sample_log_on_twt(log_md, table, trajectory, twt_axis)` | 在指定 TWT 轴上采样曲线 |
| `sample_trajectory_on_twt(trajectory, table, twt_axis)` | 在 TWT 轴上采样轨迹点 |
| `convert_dt_usm_to_vp_log(dt_log)` | 在进入 wtie 或 AI 计算前，把标准慢度显式转换为 `Vp(m/s)` |

这里可以保持对象轻量，但要保证脚本不直接操作裸数组做长期传递。

### `cup.well.spatial_samples`

这个 Module 表达“井曲线采样点已经落到了地震空间中”。

建议对象：

```text
WellSpatialSampleSet
```

一行样点建议包含：

| 字段 | 含义 |
| --- | --- |
| `well_name` | 井名 |
| `md_m` | MD |
| `twt_s` | TWT |
| `tvdss_m` | TVDSS |
| `x_m`, `y_m` | 空间坐标 |
| `inline`, `xline` | 浮点线号 |
| `nearest_il_idx`, `nearest_xl_idx` | 最近地震道索引 |
| `flat_idx` | 展平 trace 索引 |
| `property_name` | `AI`、`Vp` 等 |
| `property_value` | 属性值 |
| `weight` | 样点置信权重 |
| `source` | `vertical_fixed_xy`、`deviated_trajectory`、`representative_xy` |

这个对象会同时服务：

- 第六步 LFM 的控制点生成；
- 第十步 well constraints 的锚点生成；
- 密集井网冲突诊断。

调用顺序需要明确：`WellTrajectory` 本身不依赖时深表；只有 `position_at_twt()` 或 `WellSpatialSampleSet.from_twt_axis()` 这类 TWT 采样方法需要 `TimeDepthTable`。因此第四步可以先读取轨迹并完成空间 QC，再用已有时深表或 auto-tie 输出 table 把 TWT 样点映射回 MD/XY，避免形成“轨迹依赖时深、时深又依赖轨迹”的隐性循环。

第一步的 `survey_position` 只是井口级早期标记。对于 `near_outside` 甚至少量 `outside` 井，轨迹目标段可能进入工区；最终是否可用于第四步、第六步或第十步，应以 `WellSpatialSampleSet` 的样点级覆盖统计为准，而不是只看井口 XY。

## 对 LFM 的改造

### 点云式 LFM 控制

时间域 LFM 不再把“井级固定 XY”作为核心假设。斜井 LFM 应把控制点从“井级”改成“样点级”。

需要给 `cup.seismic.modeling` 增加一个更通用的 Interface：

```text
LayerControlPoint
```

建议字段：

```text
well_name
property_name
inline
xline
sample_value
zone_name
u_in_zone
value
weight
```

`build_layer_constrained_model()` 的内部逻辑从：

```text
每个 slice -> 每口井插值一个属性值 -> kriging
```

逐步改成：

```text
每个 slice -> 收集落在该 slice 附近的控制点 -> 按权重/冲突策略聚合 -> kriging
```

为了不一次性打碎现有代码，可以先保留旧 `WellControl`，新增 Adapter：

| Adapter | 功能 |
| --- | --- |
| `well_controls_to_layer_points()` | 直井旧控制转成点云控制 |
| `spatial_samples_to_layer_points()` | 斜井样点转成层段比例控制 |

`spatial_samples_to_layer_points()` 不能只是字段改名。它需要把每个空间样点和 `TargetLayer` 关联起来：

1. 在样点的浮点 `inline/xline` 处读取所有层位解释值。
2. 判断样点 `twt_s` 落在哪个相邻层段 `[top_horizon, bottom_horizon]`。
3. 计算 `u_in_zone = (twt_s - top_twt) / (bottom_twt - top_twt)`。
4. 对超出目标层、层位缺失、层位反转或厚度过薄的样点写入 QC，不静默参与建模。
5. 结合 `TargetLayer.valid_control_mask`、轨迹样点覆盖率和密井冲突策略生成最终 `weight`。

这部分应成为 `cup.seismic.modeling` 或 `cup.well.spatial_samples` 的深 Interface，而不是写在 `lfm_precomputed.py` 里。否则第六步和第十步会各自实现一套层位映射逻辑，后续很难保证一致。

当点云路径稳定后，再考虑让 `WellControl` 退居兼容 Adapter。

既然本项目不急于快速跑出结果，`lfm_precomputed.py` 不应先落一个代表点近似版本。代表点策略可以保留为 QC 对照或降级 Adapter，但不作为默认路径。

## 对 well_constraints 的改造

well constraints 比 LFM 更应该直接使用点云。

建议流程：

```text
preprocessed LAS + optimized TDT + trajectory
  -> WellSpatialSampleSet
  -> AnchorPoint table
  -> LogAIAnchorBundle
```

`AnchorPoint` 建议字段：

| 字段 | 含义 |
| --- | --- |
| `well_name` | 井名 |
| `md_m` | MD |
| `twt_s` | TWT |
| `sample_idx` | 时间采样索引 |
| `x_m`, `y_m` | 空间坐标 |
| `inline`, `xline` | 浮点线号 |
| `flat_idx` | 约束落到的 trace |
| `ai` | 阻抗 |
| `log_ai` | `log(ai)` |
| `weight` | 约束权重 |
| `route` | 直井/斜井路径 |

聚合到 `LogAIAnchorBundle` 时，以 `(flat_idx, sample_idx)` 为基本单元。

冲突策略必须显式配置：

| 策略 | 含义 |
| --- | --- |
| `weighted_average` | 同一格点多个约束按权重平均 |
| `highest_confidence` | 保留权重最高者 |
| `drop_conflict` | 冲突点全部丢弃 |
| `fail_on_conflict` | 出现冲突直接报错 |

默认推荐 `weighted_average`，同时输出 `anchor_conflicts.csv`，不要静默覆盖。

## 对地震取样的改造

第四步和第十步都需要“沿轨迹落道”，所以这个能力不应放在脚本中。

建议在 `cup.seismic.trace_sampling` 中新增：

| 名称 | 功能 |
| --- | --- |
| `TraceSamplePoint` | 一个空间点对应的地震道采样计划 |
| `build_trace_sample_plan(x, y, survey)` | `x/y -> inline/xline -> nearest/bilinear trace` |
| `sample_volume_at_points(volume_or_survey, points, mode)` | 对一组点批量取地震样值或地震道 |
| `deduplicate_trace_reads(points)` | 斜井跨多道时合并重复 trace 读取 |

这里要继续遵守 AGENTS.md 的易错点：物理距离必须通过 `open_survey()`、`line_to_coord()` 和 `cup.seismic.spatial` 计算，不能把 `inline_step/xline_step` 当米制距离。

落地时还需要深化 `SurveyContext` 的 Interface。当前公开协议主要支持单点 `coord_to_line()`、`line_to_coord()` 和 `import_seismic_at_well()`；斜井批量取道需要批量坐标转索引、邻道/flat index 计划、sample window 解析、重复 trace 去重。这个复杂度应封装在 `cup.seismic.trace_sampling` 与 SEG-Y/ZGY Adapter 后面，而不是散落到 `well_auto_tie.py`、`lfm_precomputed.py` 或 `well_constraints.py`。

## 建议模块归并

为了避免文件膨胀，可以控制在这些 Module：

| Module | 负责内容 |
| --- | --- |
| `cup.well.trajectory` | 完整井轨迹、轨迹文件解析、轨迹到 wtie Adapter |
| `cup.well.depth_time` | 井曲线、时深表、慢度/速度和 MD/TWT/TVDSS 之间的域转换 |
| `cup.well.spatial_samples` | 把已经明确 MD/TWT 的井曲线样点落到 XY、inline/xline、trace/sample |
| `cup.seismic.modeling` | 层位约束建模；逐步从井级控制扩展到点级控制 |
| `cup.seismic.trace_sampling` | 批量地震道/地震样点采样计划 |

暂时不建议拆出很多小文件。比如 `time_depth`、`trajectory`、`wellpath_adapter` 如果太早拆开，Interface 会变浅，调用者反而需要知道更多实现细节。

## 一次性改造范围

“一次性”指数据模型和默认路径一次性按斜井点云设计，不先落代表点近似主路径。实际编码仍建议拆成可验证的 PR 切片，每个切片都保持主线设计不回退。

### 1. 建立斜井主模型

新增 `WellTrajectory`，能从 `data/all_well_trace` 读出：

- `md`
- `tvd_kb`
- `tvdss` 或可换算出的 TVDSS
- `x/y`
- `z`
- `dx/dy`
- `incl/azim/dls`
- `kb`

同时提供 `to_wtie_wellpath()`，让 wtie 相关转换继续可用。

验收标准：

- 直井也能表示为 `WellTrajectory`；
- `position_at_md()`、`position_at_twt()` 可测；
- 轨迹越界、非单调、缺坐标能给出清晰错误。

### 2. 统一井曲线到空间样点

新增 `WellSpatialSampleSet`，把 `LogSet + TimeDepthTable + WellTrajectory + SurveyContext` 统一成点云。

验收标准：

- 直井输出的所有样点落在同一个或近似同一个 trace；
- 斜井输出的样点可以跨 trace；
- 输出 `spatial_sample_qc.csv`，统计每口井跨过多少 trace、最大横向位移、越界样点比例。

### 3. 改造 LFM 为点级控制

新增 `LayerControlPoint`，让 `lfm_precomputed.py` 直接从 `WellSpatialSampleSet` 生成层段比例控制点。

验收标准：

- 直井旧 `WellControl` 可通过 Adapter 转为 `LayerControlPoint`；
- 斜井目标层内的多个空间控制点能参与 slice kriging；
- 输出 `lfm_layer_control_points.csv`，说明每个控制点的井名、TWT、XY、inline/xline、zone、u 和属性值；
- 密集井网冲突有显式权重或冲突报告。

### 4. 改造 well constraints 为点级锚点

`well_constraints.py` 使用 `WellSpatialSampleSet` 生成 `AnchorPoint`，再聚合为现有 `LogAIAnchorBundle`。

验收标准：

- 不需要改 GINN 训练主流程即可使用斜井约束；
- 输出 `anchor_points.csv`、`anchor_conflicts.csv`、`anchor_trace_summary.csv`；
- 同一 trace 不同时间窗的斜井约束不会被整条 trace 的单一权重覆盖。

建议实现切片：

| 切片 | 解锁能力 |
| --- | --- |
| A | `WellTrajectory` + Petrel trace 读取 + 轨迹 QC |
| B | `WellSpatialSampleSet` + survey 落道 + TWT/MD 空间采样 |
| C | `LayerControlPoint` + LFM 点级控制 |
| D | `AnchorPoint` + `LogAIAnchorBundle` 聚合 |

这些切片不是“先近似后完整”的业务阶段，而是同一设计下的工程验收边界。

## 旧函数处理建议

| 现有位置 | 建议 |
| --- | --- |
| `cup.petrel.load.load_vp_rho_logset_from_las` | 拆出 LAS 读取、单位转换、`LogSet` 构造；不要继续放在 Petrel I/O |
| `cup.seismic.lfm_time.LfmTimeWell.trajectory` | 改为接受项目 `WellTrajectory`，内部再转 wtie `WellPath` |
| `cup.seismic.modeling.WellControl` | 不再作为时间域主 Interface；保留为直井井级控制 Adapter |
| 脚本内 `coord_to_line` / 最近道逻辑 | 下沉到 `cup.seismic.trace_sampling` |
| `LogsetInput = Union[grid.LogSet, Dict[str, grid.Log]]` | 新时间域脚本不再使用；旧深度域脚本可保留 wrapper 到退出维护，避免为了新主线反向重写旧流程 |

## 决策点

这些决策会影响第六步和第十步文档：

1. `WellTrajectory` 是否必须包含 `x/y`。建议必须包含；只有 `MD/TVDSS` 的轨迹不算可用于斜井空间建模。
2. LFM 是否直接使用点云控制。建议直接使用，不先落代表点默认路径。
3. well constraints 是否直接使用点云锚点。建议直接使用。
4. `WellControl` 是否继续作为主 Interface。建议否；它可以作为直井兼容 Adapter。
5. 密集井冲突默认策略。建议默认 `weighted_average + report`，不要静默覆盖。

## 推荐结论

真正需要先改的不是 `lfm_precomputed.py` 或 `well_constraints.py` 的局部分支，而是 `src/cup` 里缺少一个“井轨迹空间样点”的深 Module。

推荐一次性路线：

1. 新增 `cup.well.trajectory.WellTrajectory`。
2. 新增 `cup.well.spatial_samples.WellSpatialSampleSet`。
3. 新增 `cup.seismic.modeling.LayerControlPoint`，让 LFM 直接消费点级控制。
4. 第十步 well constraints 使用同一套空间样点生成 trace/time 锚点。
5. `WellControl` 和代表点策略只作为 Adapter 或 QC 对照，不作为时间域主路径。

这样第六步和第十步共享同一个斜井事实来源：`WellTrajectory -> WellSpatialSampleSet`。复杂度集中在一个深 Module 后面，而不是分散在多个脚本里反复补丁。
