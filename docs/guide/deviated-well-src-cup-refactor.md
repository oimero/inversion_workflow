# 斜井支持的 src/cup 后续重构规划

本文只跟踪还没有落地的斜井主线能力。已经进入代码的内容不再作为规划项重复展开：`cup.well.trajectory.WellTrajectory`、`cup.well.spatial_samples.sample_trajectory_on_twt()`、`cup.seismic.trace_sampling` 的最近道采样计划、`SurveyContext.read_traces_at_indices()`，以及 `well_auto_tie.py` 的 `deviated_with_tdt` 路径。

后续主线仍然围绕同一个事实链条：

```text
WellTrajectory + TimeDepthTable + LogSet
  -> WellSpatialSampleSet
  -> trace/time sample points
  -> LFM 控制点或 GINN 井约束锚点
```

---

## 1. 路径四：`deviated_anchor_from_tops`

这是第四步 `well_auto_tie.py` 还没有落地的路线：**斜井、无 Petrel 时深表、有井轨迹、有井分层、有可用 `DT_USM/RHO_GCC` 曲线**。

它和已经实现的 `deviated_with_tdt` 最大区别是：没有现成的 `TWT -> MD` 表，所以必须先用井分层和解释层位建立一个初始 MD 域 TDT，然后才能复用已落地的斜井轨迹取道能力。

### 路由条件

`build_tie_plan()` 已能识别这条 route：

| 条件 | 要求 |
|------|------|
| 井型 | `wellbore_class_qc == deviated` |
| 曲线 | 第三步 `usable_p_sonic == true` 且 `usable_density == true` |
| 资产 | 无时深表；有井轨迹；有井分层 |
| 工区 | 井口在允许的 `survey_position` 内 |
| 配置 | route 被加入 `enabled_routes` 后才执行 |

当前执行层还没有 handler，因此即使 route 被识别，也不应在主配置中启用。

### 设计原则

- 锚点 TWT 必须在**锚点 MD 对应的轨迹 XY** 处读取，不能默认用井口 XY。
- 初始 TDT 仍应是 MD 域表：`grid.TimeDepthTable(twt=..., md=...)`。
- 轨迹只负责 `MD -> XY` 和后续 `TWT -> MD -> XY -> trace`；不要把轨迹里的 Petrel `Z` 当成 checkshot TDT。
- 第一版只支持单锚点，沿声波曲线向上、向下积分；多锚点误差分配留到后续。

### 建议处理流程

1. 读取预处理 LAS，构造 `grid.LogSet`。
2. 读取 Petrel 井轨迹，得到 `MD -> TVD_KB/TVDSS/XY`。
3. 从井分层表中找到配置的锚点层位 MD，例如 `H3-1`。
4. 用 `WellTrajectory.position_at_md(anchor_md)` 得到锚点处的 `x_m/y_m`。
5. 把锚点 XY 投影到工区 inline/xline，在解释层位上读取锚点 TWT。
6. 用 `DT_USM` 从锚点向上、向下积分，生成初始 MD 域 TDT。
7. 按 `H3-1 -> H7-1` 目标时间窗裁剪或拓展 TDT 和曲线。
8. 复用 `deviated_with_tdt` 的轨迹采样、最近道批读、出界比例检查和最长 inside 窗口裁剪。
9. 调用 `wtie.autotie.tie_v1`，输出同一套 wavelet、TDT、QC 和 metrics。

### 积分口径

直井锚点路径目前使用：

```text
dTWT_s = 2 * DT_USM * dMD_m * 1e-6
```

斜井不能直接套这个近似。建议第一版显式改成沿轨迹 TVD 增量积分：

```text
dTWT_s = 2 * DT_USM(MD) * dTVD_KB_m * 1e-6
```

这样初始 TDT 仍以 MD 为自变量，但时间增量使用轨迹给出的垂向深度变化。验收时必须把 `integration_axis = tvd_kb_from_trajectory` 写入 `tie_window_report.csv` 或 `run_summary.json`，避免后续误以为它是沿井眼 MD 积分。

如果轨迹 TVD 非单调、锚点 MD 超出轨迹、锚点层位在解释层位外、或生成的 TDT 非单调，应直接失败并写清原因。

### 需要新增或扩展的能力

| 位置 | 能力 |
|------|------|
| `cup.well.td` | 新增斜井锚点建表函数，例如 `build_deviated_tdt_from_anchor(logset_md, trajectory, anchor_md_m, anchor_twt_s)` |
| `scripts/well_auto_tie.py` | 新增 `_run_deviated_anchor_from_tops()` handler |
| `anchor_report.csv` | 增加锚点轨迹 XY、浮点 inline/xline、最近线号和层位采样状态 |
| `tie_window_report.csv` | 增加积分口径、锚点来源和轨迹裁剪原因 |

### 验收标准

- 一口候选斜井能从 `well_tie_plan.csv` 进入 `deviated_anchor_from_tops` 并完成 auto-tie。
- `anchor_report.csv` 证明锚点 TWT 取自锚点轨迹 XY，而不是井口 XY。
- `initial_tdt_<well>.csv` 为 MD 域，TWT 严格递增，`source` 能区分 anchor 和 sonic integration。
- `trace_sample_plan_<well>.csv` 与 `seismic_trace_<well>.csv` 的 TWT 轴一致。
- 失败井能区分 `anchor_md_outside_trajectory`、`anchor_horizon_outside_survey`、`deviated_anchor_tdt_not_monotonic`、`trajectory_outside_fraction_exceeded` 等原因。

---

## 2. LFM：点级控制而不是井级代表点

`lfm_precomputed.py` 后续不应再假设“一口井只有一个固定 inline/xline”。斜井在目标层内可能跨多个 trace，同一口井的不同 TWT 样点应作为不同控制点参与低频模型。

### 建议新增 `LayerControlPoint`

字段建议：

| 字段 | 含义 |
|------|------|
| `well_name` | 来源井 |
| `property_name` | 例如 `AI`、`Vp`、`Rho` |
| `twt_s` / `md_m` | 控制点所在时间和 MD |
| `x_m` / `y_m` | 控制点 XY |
| `inline_float` / `xline_float` | 浮点线号 |
| `flat_idx` | 最近 trace |
| `zone_name` | 所属层段 |
| `u_in_zone` | 在层段内的比例位置 |
| `value` | 控制值 |
| `weight` | 控制权重 |
| `source` | `vertical_trace`、`deviated_trajectory` 等 |

### 需要解决的问题

1. 在控制点所在 XY 处读取上下层位 TWT。
2. 判断样点属于哪个 `TargetZone`。
3. 计算 `u_in_zone = (twt_s - top_twt) / (bottom_twt - top_twt)`。
4. 对层位缺失、层位反转、厚度过薄、超出目标层的样点写 QC，不静默参与建模。
5. 密集井网中多个控制点落到同一 slice/trace 附近时，按显式策略聚合。

### 建议落地方式

保留现有 `cup.seismic.modeling.WellControl` 作为直井兼容 Adapter，但时间域主线新增点级路径：

```text
optimized TDT + preprocessed LAS + trajectory
  -> WellSpatialSampleSet
  -> LayerControlPoint
  -> slice-level interpolation
```

输出新增：

| 文件 | 内容 |
|------|------|
| `lfm_layer_control_points.csv` | 每个点的井名、TWT、MD、XY、inline/xline、zone、u、属性值、权重 |
| `lfm_control_qc.csv` | 每口井控制点数量、跨 trace 数、无效点比例 |
| `lfm_control_conflicts.csv` | 密集井网或同 trace/slice 冲突 |

验收标准：

- 直井旧控制能通过 Adapter 转成点级控制。
- 斜井目标层内多个空间控制点能参与 slice 插值。
- 不再把斜井压成目标层中点或井口代表点作为默认路径。

---

## 3. well constraints：点级锚点聚合

`well_constraints.py` 后续应直接从空间样点生成约束点，再聚合成现有 GINN 训练需要的 `LogAIAnchorBundle`。GINN 主训练流程可以继续消费：

```text
flat_indices
target_log_ai[n_anchor_trace, n_sample]
anchor_weight[n_anchor_trace, n_sample]
```

变化发生在 bundle 生成之前。

### 建议新增 `AnchorPoint`

字段建议：

| 字段 | 含义 |
|------|------|
| `well_name` | 来源井 |
| `md_m` / `twt_s` | 样点位置 |
| `sample_idx` | 时间采样索引 |
| `x_m` / `y_m` | 样点 XY |
| `inline_float` / `xline_float` | 浮点线号 |
| `flat_idx` | 约束落到的 trace |
| `ai` / `log_ai` | 阻抗及其对数 |
| `weight` | 约束权重 |
| `route` | 来源 auto-tie 路径 |
| `source` | `vertical_trace`、`deviated_trajectory` 等 |

### 聚合规则

聚合基本单元应为：

```text
(flat_idx, sample_idx)
```

同一个单元出现多个约束点时，必须显式配置冲突策略：

| 策略 | 含义 |
|------|------|
| `weighted_average` | 按权重平均，推荐默认 |
| `highest_confidence` | 保留权重最高者 |
| `drop_conflict` | 冲突单元全部丢弃 |
| `fail_on_conflict` | 有冲突就报错 |

默认建议 `weighted_average + report`，不要静默覆盖。

### 输出新增

| 文件 | 内容 |
|------|------|
| `anchor_points.csv` | 聚合前的点级井约束 |
| `anchor_conflicts.csv` | 同一 `(flat_idx, sample_idx)` 的冲突明细 |
| `anchor_trace_summary.csv` | 每条 trace 的锚点数量、时间覆盖、来源井列表 |

验收标准：

- 一口斜井跨多个 trace 时，约束分布到多个 `flat_idx`。
- 同一 trace 不同时间窗的约束不会被整条 trace 的单一权重覆盖。
- 不需要改 `ginn_train.py` 主训练循环即可使用新的 bundle。

---

## 4. 地震取样的后续升级

当前已落地的是最近道批读：样点落到最近 inline/xline，唯一 trace 去重读取，再拼成一条沿轨迹地震道。

后续可升级但不阻塞主线：

| 能力 | 用途 |
|------|------|
| 双线性采样 | 减少最近道阶梯跳变 |
| 多道加权 | 对斜井附近多个 trace 做稳定采样 |
| ZGY 批量块读取 | 减少多井、多轨迹点时的 IO |
| 轨迹 inline/xline QC 图 | 快速发现轨迹几何或工区转换异常 |

这些能力仍应放在 `cup.seismic.trace_sampling` 和 `SurveyContext` Adapter 后面，不应散写到脚本里。

---

## 推荐落地顺序

1. `deviated_anchor_from_tops`：补齐第四步最后一条 route。
2. 扩展空间样点到属性点：让 `WellSpatialSampleSet` 能携带 AI/Vp/Rho、权重和来源。
3. `LayerControlPoint`：让 `lfm_precomputed.py` 从井级控制转为点级控制。
4. `AnchorPoint`：让 `well_constraints.py` 从点级锚点聚合为 `LogAIAnchorBundle`。
5. 冲突诊断：统一输出密集井网下的 trace/time 冲突报告。

核心原则不变：斜井的空间事实只从 `WellTrajectory -> WellSpatialSampleSet` 出来；LFM 和 well constraints 不各自重写一套轨迹、TWT、inline/xline 转换逻辑。
