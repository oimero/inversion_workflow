# 斜井支持的后续基础设施规划

本文只记录时间域主线之外、仍未落地的斜井和井约束基础设施。已经进入主线的内容不再重复展开：

- `cup.well.trajectory.WellTrajectory`
- `cup.well.trajectory.sample_trajectory_on_twt()`
- `cup.seismic.trace_sampling`
- `well_auto_tie.py` 的 `deviated_with_tdt` 路径
- 第六步 `well_constraints.py` 的井约束主线规划
- 第七步 `lfm_precomputed.py` 的点级 LFM 主线规划
- 第八、九步 GINN 训练与反演主线规划

后续原则不变：斜井的空间事实只从同一条链路出来，LFM、well constraints 和训练脚本不各自重写一套 MD/TWT/XY/trace 映射。

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

1. 读取第四步同口径标准 LAS，构造 `grid.LogSet`。
2. 读取 Petrel 井轨迹，得到 `MD -> TVD_KB/TVDSS/XY`。
3. 从井分层表中找到配置的锚点层位 MD。
4. 用 `WellTrajectory.position_at_md(anchor_md)` 得到锚点处的 `x_m/y_m`。
5. 把锚点 XY 投影到工区 inline/xline，在解释层位上读取锚点 TWT。
6. 用声波曲线从锚点向上、向下积分，生成初始 MD 域 TDT。
7. 按目标时间窗裁剪或拓展 TDT 和曲线。
8. 复用 `deviated_with_tdt` 的轨迹采样、最近道批读、出界比例检查和最长 inside 窗口裁剪。
9. 调用 `wtie.autotie.tie_v1`，输出同一套 wavelet、TDT、QC 和 metrics。

### 积分口径

直井锚点路径目前使用：

```text
dTWT_s = 2 * DT_USM * dMD_m * 1e-6
```

斜井不应直接套这个近似。建议第一版显式改成沿轨迹 TVD 增量积分：

```text
dTWT_s = 2 * DT_USM(MD) * dTVD_KB_m * 1e-6
```

这样初始 TDT 仍以 MD 为自变量，但时间增量使用轨迹给出的垂向深度变化。验收时必须把 `integration_axis = tvd_kb_from_trajectory` 写入 `tie_window_report.csv` 或 `run_summary.json`。

如果轨迹 TVD 非单调、锚点 MD 超出轨迹、锚点层位在解释层位外、或生成的 TDT 非单调，应直接失败并写清原因。

V1 公式隐含假设 `DT_USM(MD)` 在每个轨迹 TVD 增量内变化较平缓。急弯井段可能累积近似误差，后续应补充按轨迹 TVD 增量的梯形或分段积分校正。

### 需要新增或扩展的能力

| 位置 | 能力 |
|------|------|
| `cup.well.td` | 新增斜井锚点建表函数，例如 `build_deviated_tdt_from_anchor(logset_md, trajectory, anchor_md_m, anchor_twt_s)` |
| `scripts/well_auto_tie.py` | 新增 `_run_deviated_anchor_from_tops()` handler |
| `anchor_report.csv` | 增加锚点轨迹 XY、浮点 inline/xline、最近线号和层位采样状态 |
| `tie_window_report.csv` | 增加积分口径、锚点来源和轨迹裁剪原因 |

---

## 2. Well Constraints：点级锚点聚合

第八步主线暂不要求井约束。后续如果要打开 `log_ai_anchor_file`，应从第六步空间样点生成点级约束，再聚合成现有 `src.ginn.anchor` 能消费的 bundle。

目标输出仍可以保持训练端现有契约：

```text
flat_indices
target_log_ai[n_anchor_trace, n_sample]
anchor_weight[n_anchor_trace, n_sample]
```

变化发生在 bundle 生成之前。

### 建议新增 `AnchorPoint`

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

---

## 3. 地震取样的后续升级

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
2. `AnchorPoint`：从点级样点聚合为 `LogAIAnchorBundle`。
3. 冲突诊断：统一输出密集井网下的 trace/time 冲突报告。
4. 地震取样升级：双线性或多道加权采样。
