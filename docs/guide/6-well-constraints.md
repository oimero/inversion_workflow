# 06 分频诊断与井约束

`well_constraints.py` 是时间域工作流的第六步。它负责把第四步、第五步之后可信的井曲线转换成一套统一的井约束数据，分别交给 GINN、低频模型和后续 enhance 使用。

---

## 快速开始

```bash
python scripts/well_constraints.py
python scripts/well_constraints.py --config experiments/common.yaml
python scripts/well_constraints.py --output-dir scripts/output/well_constraints_test
```

不带参数时，脚本自动发现最新的第四步和第五步产物，在 `scripts/output/well_constraints_<timestamp>/` 下写出结果。

---

## 运行前需要什么

| 来源 | 文件 | 用途 |
|------|------|------|
| 第四步 | `well_tie_metrics.csv` | 标定状态、路由、优化后时深表路径、滤波 LAS 路径 |
| 第四步 | `well_tie_plan.csv` | 井口坐标、井型初分 |
| 第四步 | 斜井样点计划文件 | 斜井沿优化后时深表生成的逐样点空间映射 |
| 第五步 | `batch_synthetic_metrics.csv` | 逐井批量合成相关系数和误差，用于筛选和定权重 |
| 地震数据 | 时间域地震体 | 提供时间轴和工区几何 |
| 解释层位 | 顶底解释层位文件 | 构建目标层范围和各层段划分 |

直井的空间位置来自井口坐标、优化后时深表和地震几何。斜井的空间位置必须来自第四步细标定后重新生成的样点计划文件——不能在本步骤退化成井口直井，也不能用另一套时深表或轨迹重新插值。

---

## 配置参考

```yaml
well_constraints:
  source_runs:
    mode: latest
    well_auto_tie_dir: null
    wavelet_generation_dir: null

  seismic:
    file: null
    type: segy

  target_interval:
    horizons:
      - <top-horizon-file>
      - <bottom-horizon-file>
    twt_unit: auto

  control_wells:
    min_batch_corr: 0.35
    max_batch_nmae: null
    include_wells: null
    exclude_wells: []

  frequency_split:
    mode: diagnose
    manual_cutoff_hz: null
    filter_order: 6
    candidate_cutoff_hz: [6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]
    selection_corr_tolerance: 0.02
    selection_nmae_tolerance: 0.03
    buffer_seconds: null
    buffer_mode: reflect
    qc_enabled: true
    qc_envelope_window_samples: 31

  anchor:
    include_deviated: false
    min_points_per_trace: 2

  conflicts:
    strategy: weighted_average

  weights:
    mode: corr
    corr_floor: 0.3
    corr_span: 0.4
    corr_min_weight: 0.6

  lfm_controls:
    n_slices: 20
    min_control_samples_per_well: 16

  motif:
    write_manifest: true
```

### `source_runs`

默认接上最新一次井震标定和全局子波结果。复现实验时，填写 `well_auto_tie_dir` 或 `wavelet_generation_dir` 固定输入。`mode` 目前只支持 `latest`。

### `control_wells`

不是所有标定成功的井都能成为控制井。每口井必须先过第五步的统一子波批量合成门槛：

| 参数 | 含义 |
|------|------|
| `min_batch_corr` | 批量合成相关系数低于此值的井被排除 |
| `max_batch_nmae` | 可选 NMAE 上限，为空时不按 NMAE 过滤 |
| `include_wells` | 白名单模式，只使用指定井 |
| `exclude_wells` | 人工排除可疑井 |

此外，井还必须同时具备第四步产出的滤波 LAS 和优化后时深表；斜井还需要对应的样点计划文件。

### `frequency_split`

决定井曲线中哪部分交给低频约束、哪部分交给 enhance 高频材料。支持两种模式：

- **`diagnose`**：对每个候选截止频率低通井上 log-AI，用第五步最终全局子波正演合成记录，再与第四步保存的井旁地震道计算 `corr`、`nmae` 和 `scale`。脚本先找多井波形拟合的近最佳平台，再在平台内选择较低 Hz，让低频分支保持保守。
- **`manual`**：跳过诊断，直接使用 `manual_cutoff_hz` 指定的截止频率。

无论哪种模式，最终选定的截止频率、滤波阶数、缓冲策略和诊断证据都会写入 `run_summary.json`。

分频结果不是为了让每口井各用各的截止频率。默认口径是全目标窗共享一个分频；分层 enhance 在此基础上统计每层的高频特征，但同一条训练链路里不应出现多套互相不兼容的频率拆分。

### `anchor`

控制 GINN 低频 anchor 的生成：

| 参数 | 含义 |
|------|------|
| `include_deviated` | 是否允许斜井进入低频 anchor。第一版默认关闭 |
| `min_points_per_trace` | 每条受控道上至少需要多少个有效样点 |

低频 anchor 的目标值来自分频后的低频 log-AI，而不是原始全频井曲线。这样它不会和第八步网络学习的高频残差互相争抢职责。

第一版默认只有直井进入低频 anchor 文件；斜井进入点级事实、高频监督和高频统计，但不进入低频 anchor。

### `conflicts`

密井网下，多口井可能落到同一个道位置和样点位置上。当前只支持 `weighted_average` 策略：按权重加权平均，冲突明细写入审计文件。密井网下不允许静默覆盖。

### `weights`

控制每个样点的训练权重。`mode: corr` 时，权重由第五步的批量合成相关系数映射而来：

- 相关系数低于 `corr_floor`，权重退到 `corr_min_weight`
- 相关系数在 `corr_floor` 到 `corr_floor + corr_span` 之间线性增长
- 相关系数达到上限后权重封顶为 1.0

`mode: uniform` 时，所有权重为 1.0。

### `lfm_controls`

| 参数 | 含义 |
|------|------|
| `n_slices` | 顺层切片数量。必须与第七步 LFM 配置中的同名参数一致 |
| `min_control_samples_per_well` | 单井最少有效样点数，低于此值的井被排除 |

第七步 LFM 会校验 `n_slices` 是否与本步骤一致，不一致则直接报错。

### `motif`

`write_manifest: true` 时，写出一个空的 motif 清单文件作为后续 synthetic generator 的接口占位。第一版不生成 motif 数据包。

---

## 脚本在做什么

脚本分六个阶段：**前置发现 → 点级事实生成 → 分频诊断与拆分 → 低频 anchor 输出 → 高频监督与统计输出 → LFM 控制点聚合**。

### 第一阶段：前置发现

1. 从配置或自动发现中定位第四步和第五步的产出目录。
2. 打开地震体，校验采样域为时间域且单位为秒。
3. 读取顶底解释层位，按平均时间从浅到深排序，相邻层位组成层段。

### 第二阶段：点级事实生成

对第四步标定成功的每口井，先过第五步的批量合成 QC 门槛，再过资产完整性检查。然后按井型分两条路径生成点级事实表：

**直井路径**

1. 从滤波 LAS 读取波阻抗曲线。
2. 用优化后时深表将深度域波阻抗转成时间域。
3. 在时间轴的每个样点处，判断是否落入目标层内；若是，记录该样点的时间、波阻抗、深度、层段和层内比例位置。
4. 空间坐标由井口 XY 和地震几何确定，直井的所有样点落在同一个道位置上。

**斜井路径**

1. 从滤波 LAS 读取波阻抗曲线。
2. 读取第四步的样点计划文件（细标定后的空间映射），只保留工区内样点。
3. 在计划文件给出的每个轨迹样点处，插值波阻抗值。
4. 判断每个样点是否落入目标层内，记录其空间坐标、时间、波阻抗、层段和层内比例位置。
5. 斜井的样点随轨迹分布在多个不同的空间位置上。

两路生成的点级事实表格式统一，基本单元是"某口井在某个时间样点上的空间和曲线事实"。规范坐标为浮点线号和正秒时间。整数道号、数组索引等只作为派生调试字段，不能当作跨工区稳定坐标。

### 第三阶段：分频诊断与拆分

如果配置为 `diagnose` 模式，对每个候选截止频率做一次完整的低通，然后用第五步最终全局子波做正演，和第四步保存的井旁地震道比较。诊断输出不是只看井曲线是否平滑，而是回答一个更直接的问题：这个 cutoff 下的低频 AI 正演出来，是否仍能解释真实地震波形。

多井聚合后，脚本先找 `median_corr` 最高的 cutoff，再按 `selection_corr_tolerance` 和 `selection_nmae_tolerance` 找近最佳平台；如果有平台，则选平台内较低的 Hz。时间域 Hz 越低，低通越强、低频分支越保守。

如果配置为 `manual` 模式，直接使用用户指定的截止频率。

确定截止频率后，对所有井的 log-AI 曲线做零相位低通滤波，拆成两部分：

- **低频部分**：滤波后的平滑趋势，交给 GINN 低频 anchor
- **高频部分**：原曲线减去低频部分后的残余，交给 enhance 统计和监督

每口井的分频结果会生成 QC 图：上方是完整的和低频的 log-AI 对比，下方是高频残余及其包络。

### 第四阶段：低频 anchor 输出

从分频后的低频 log-AI 中提取 anchor 样点，聚合成 GINN 训练端可直接读取的 anchor 数据包。anchor 样点只来自直井（除非显式打开斜井开关）。

如果多口井落到同一个道位置和样点位置上，按权重加权平均，冲突明细写入独立的冲突报告文件。每条受控道的井数、样点数、覆盖层段和权重统计写入道摘要文件。

### 第五阶段：高频监督与统计输出

从同一个点级事实表中，同时产出三套材料：

1. **逐点监督真值**：每口井每个样点上的高频残余值，写成 enhance 训练可读取的监督数据包。样点带有井名、空间位置、权重和层段归属标记。
2. **全窗和分层统计**：对全部井的高频残余做全局统计；再按每个层段分别统计。统计项包括振幅分布、事件密度、正负状态持续长度、转移矩阵、反射系数量级和频谱。
3. **收缩参数**：每层的经验统计按其可靠度向全窗统计收缩——可靠度高的层主要相信自己，可靠度低的层更多借用全窗先验。收缩因子一并写出。

可靠度由有效井数、有效样点数、事件数和空间覆盖共同决定。小样本层如果完全相信自身的经验统计，后续 enhance 合成器生成的样本会过拟合；向全窗收缩可以在信息不足时借用先验。

注意：这个阶段还没有低频模型（LFM 是第七步的产物），因此高频监督数据包中与 LFM 相关的字段是零值占位。下游 enhance 训练端如果需要底阻抗输入，应从 GINN 或 LFM 主文件中读取，而不是从这份高频监督包读。

### 第六阶段：LFM 控制点聚合

将点级事实表按单井、层段和切片聚合成代表控制点，写出 LFM 控制点文件。同一口井落入同一比例切片的多个样点只保留一个代表值。这样直井和斜井口径一致，也避免一小段斜井轨迹在同一张切片里被误当成多口独立井。

第七步 LFM 只读取这份控制点文件，不再自己读 LAS、时深表或井轨迹。

---

## 核心输出文件

所有文件在 `<output_root>/well_constraints_<timestamp>/` 下：

| 文件 | 内容 |
|------|------|
| `well_constraint_points.csv` | 点级事实表，每行一个时间样点的完整空间和曲线信息 |
| `well_high_supervision_qc.csv` | 逐井筛选结果、样点数和分频 QC 图路径 |
| `well_anchor_points.csv` | 聚合前用于低频 anchor 的点级约束 |
| `well_anchor_conflicts.csv` | 同一道位置 / 样点位置的井间冲突明细 |
| `well_anchor_trace_summary.csv` | 每条受控道的井数、样点数和权重统计 |
| `log_ai_anchor_time.npz` | GINN 训练可直接读取的低频井监督数据包 |
| `well_high_supervision_time.npz` | enhance 训练可直接读取的高频井监督数据包 |
| `well_high_stats_global.json` | 全目标窗高频残余统计 |
| `well_high_stats_by_layer.csv` | 每层的经验统计和可靠度 |
| `well_high_stats_shrinkage.json` | 每层收缩后的最终生成参数 |
| `well_high_motif_manifest.csv` | motif patch 清单（第一版为空占位） |
| `lfm_layer_control_points.csv` | 第七步 LFM 读取的控制点文件 |
| `lfm_control_qc.csv` | 逐井筛选结果和控制点数量 |
| `target_layer_qc/*` | 目标层 mask、层厚、层位有效性 QC |
| `frequency_split_diagnostics.csv` | 逐井逐候选 cutoff 的正演匹配指标 |
| `frequency_split_aggregate.csv` | 多井聚合后的候选 cutoff 指标 |
| `figures/frequency_split_cutoff_sweep.png` | 候选 cutoff 的全局 corr/nmae sweep 图 |
| `figures/frequency_split_wells/*.png` | 每口井的候选 cutoff corr/nmae sweep 图 |
| `frequency_split_qc/traces/*.csv` | 每口井分频前后的曲线数值 |
| `frequency_split_qc/figures/*.png` | 每口井的分频 QC 图 |
| `run_summary.json` | 输入路径、筛选统计、分频参数和所有输出路径 |

### `well_constraint_points.csv`

每行一个目标层内的井样点，以浮点线号和正秒时间为规范坐标。关键字段：

| 字段 | 含义 |
|------|------|
| 井名 / 路径 | 来源井和第四步标定路径 |
| 来源类型 | `vertical_trace`（直井井口）或 `deviated_trajectory`（斜井沿轨迹） |
| 时间 / 深度 | 样点在时间域和深度域的位置 |
| 平面坐标 | 样点的地面投影 XY |
| 浮点线号 | 投影到工区后的浮点线号 |
| 层段 / 层内比例 | 所属层段和层内比例位置（0 到 1） |
| 波阻抗 / 权重 | 样点的全频波阻抗值及训练权重 |
| 低频 / 高频 log-AI | 分频后的低频和高频成分 |
| 是否用于 anchor | 该样点是否进入 GINN 低频 anchor |

### `log_ai_anchor_time.npz`

第八步 GINN 读取的低频井监督数据包，包含：

| 键 | 含义 |
|----|------|
| 目标值数组 | 每个受控样点的低频波阻抗 |
| 掩码数组 | 哪些样点是有效控制点 |
| 权重数组 | 每个控制点的训练权重 |
| 井名 / 类型 / 坐标 | 每口控制井的标识和位置 |
| 元数据 | 分频参数、冲突策略和上游路径 |

### `well_high_supervision_time.npz`

enhance 训练读取的高频井监督数据包，包含每个受控样点上的全频 log-AI、低频成分、高频残余、掩码、权重和井名。样点保留层段归属标记，训练端可以按批抽取真实井位置，计算增强后的高频残余并与井上高频残余做对比损失。

### `well_high_stats_by_layer.csv`

每层一行，记录该层的统计特征和可靠度。关键列包括：有效井数、有效样点数、事件数、可靠度（决定收缩强度）、振幅分位数、事件密度、正负状态持续长度的分位数，以及转移矩阵。

### `well_high_stats_shrinkage.json`

每层一个条目，记录该层经验统计、全窗统计、收缩因子和收缩后的最终参数。最终参数就是 enhance 合成器要用到的生成参数。

---

## 如何阅读结果

### 第一步：看 `run_summary.json`

先看顶层计数：

- 入选井数、点级事实总数
- anchor 受控道数
- LFM 控制点数（聚合后）
- 分频诊断选出的截止频率

这些数字应该与第四步的标定成功井数、目标层覆盖范围在量级上匹配。如果入选井数远少于第四步的成功井数，优先看质量控制文件的拒绝原因分布。

### 第二步：看 `well_high_supervision_qc.csv`

确认每口井的入选状态和拒绝原因。常见拒绝原因：

- `batch_corr_below_threshold`：批量合成相关系数不够
- `missing_filtered_las_file`：缺少滤波 LAS
- `missing_optimized_tdt_file`：缺少优化后时深表
- `missing_optimized_trace_sample_plan`：斜井缺少样点计划文件
- `too_few_control_samples`：落入目标层的有效样点不足

对入选井，看样点数和唯一道数。直井的唯一道数应为 1；斜井应大于 1。如果斜井所有样点落在同一道上，要回头检查第四步的空间映射。

### 第三步：看 `well_anchor_conflicts.csv`

如果存在多口井落到同一个道位置和样点位置上的记录，说明存在密井冲突。看冲突文件中的值差距——同一位置上的波阻抗值差异越大，权重平均的效果就越值得怀疑。目前第一版用加权平均处理冲突，更精细的策略（如保留最高置信度、遇冲突直接失败）留到后续版本。

### 第四步：看 `well_high_stats_by_layer.csv`

关注每层的可靠度。可靠度低的层，在 enhance 合成器中生成的高频样本会更多依赖全窗先验——这不是错误，而是信息不足时的合理保守策略。但如果大多数层的可靠度都偏低，说明有效井数或样点覆盖不足以支撑分层统计，此时应回到第四步或第五步检查标定和批量合成的覆盖面。

### 第五步：看图

先看 `figures/frequency_split_cutoff_sweep.png`。这张图展示候选 cutoff 的多井 `median corr` 和 `median nmae`；虚线会标出波形相关性最佳 cutoff 和最终选中的 cutoff。如果两者不同，说明脚本在近最佳平台内选择了更保守的低 Hz。

再看 `figures/frequency_split_wells/` 中的逐井 sweep 图，确认最终选择不是由少数井单独拉动。

最后抽查几口井的分频 QC 图（`frequency_split_qc/figures/`）：

- 上方子图：完整的和低频的 log-AI。低频曲线应该平滑、跟随大趋势，而不是贴着原始曲线走。
- 下方子图：高频残余和包络。高频残余应在零值附近正负振荡，包络不应在某一段突然膨胀（说明那里的原始曲线有异常尖峰）。

同时抽查几口直井和斜井的点级事实——直井的空间坐标应恒定，斜井的空间坐标应随时时间变化，层内比例覆盖应尽量均匀。

---

### 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| `No selected well constraint points` | 所有井都被过滤或事实生成失败 | 检查质量控制文件的拒绝原因；多数情况是第四步成功井太少或第五步门槛过高 |
| 分频诊断在候选截止频率上全部不理想 | 候选范围不覆盖当前数据的合理分频点 | 扩大候选截止频率范围，或改用手动模式指定一个已知合适的值 |
| 某口斜井缺少样点计划文件 | 第四步未为该斜井写出细标定后的空间映射 | 回到第四步检查斜井路径是否执行成功 |
| 落入目标层的有效样点不足 | 时深表范围未覆盖目标层，或 LAS 曲线在目标层深度内没有值 | 检查时深表覆盖范围、目标层配置和 LAS 曲线深度范围 |
| LFM 控制点聚合后为空 | 所有入选井在切片聚合阶段被过滤 | 检查 `n_slices` 是否过大导致每张切片没有足够的控制点 |

---

## 留到第二轮

- 冲突策略从仅支持加权平均扩展到保留最高置信度、丢弃冲突点和遇冲突直接失败。
- motif patch 的提取、质量标签和数据包生成。
- 正演分频诊断进一步加入空间去偏聚合，避免密井平台主导 cutoff 选择。
- 斜井 anchor 支持（当前 `include_deviated` 默认为关）。
- 分频 QC 增加分层叠加图：在同一张图上按层段着色显示高频残余，方便对比不同层的分频效果。
