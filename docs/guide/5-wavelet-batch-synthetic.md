# 05 子波批量合成与全局选择

本文讨论第五个规划脚本：`wavelet_batch_synthetic.py`。

第四步 `well_auto_tie.py` 会为每口成功井产出一个微调时深表和一个自动提取子波。但实际反演通常只能使用一个全局子波。第五步的核心任务不是简单批量正演，而是把第四步产出的所有子波作为候选，在所有可评测井上互测，自动选出整体表现最稳的全局子波。

参考脚本：

- `scripts/wavelet_batch_synthetic_depth.py`
- `scripts/frequency_split_diagnosis_depth.py`

深度域的 `wavelet_batch_synthetic_depth.py` 是“给定一个子波，批量做井旁合成记录和 bulk shift”。本轮时间域第五步要再向前走一步：先做候选子波选择，再用选中的全局子波生成统一的批量合成结果。

## 目标

`wavelet_batch_synthetic.py` 回答五件事：

1. 第四步产出的哪些子波可以作为候选全局子波。
2. 每个候选子波在每口可评测井上的合成记录匹配效果如何。
3. 如何避免“来源井子波只在自己井上特别好”的过拟合。
4. 哪一个子波在全井集合上最稳，应该进入后续反演流程。
5. 使用该全局子波后，每口井的统一合成记录、指标和可选 bulk shift 结果是什么。

## 输入

- 第四阶段目录：`scripts/output/well_auto_tie_<timestamp>/`
- 第四阶段路由结果：`well_tie_plan.csv`
- 第四阶段标定指标：`well_tie_metrics.csv`
- 第四阶段候选子波清单：`wavelet_inventory.csv`
- 第四阶段优化时深表：`time_depth/optimized_tdt_<well>.csv`
- 第四阶段井旁或轨迹地震道：`seismic_trace/seismic_trace_<well>.csv`
- 第三阶段预处理 LAS：`preprocessed_las/*.las`
- 可选人工排除配置：排除某些井或某些候选子波

建议配置片段：

```yaml
wavelet_batch_synthetic:
  source_runs:
    mode: latest
    well_auto_tie_dir: null
    log_preprocess_dir: null

  source_auto_tie_dir: null
  preprocessed_las_dir: null

  candidate_filter:
    min_source_tie_corr: 0.35
    max_source_tie_nmae: null
    exclude_source_wells: []
    include_source_wells: null

  evaluation_wells:
    status: success
    exclude_wells: []
    include_wells: null
    require_optimized_tdt: true
    require_seismic_trace: true

  scoring:
    use_leave_source_out: true
    corr_weight: 1.0
    p10_corr_weight: 0.5
    nmae_weight: 0.5
    shift_penalty_weight: 0.2
    shift_penalty_scale_ms: 20.0
    min_eval_well_count: 3
    on_insufficient_eval_wells: select_best_source_tie
    min_leave_source_out_well_count: 2

  shift_scan:
    enabled: true
    min_shift_ms: -20.0
    max_shift_ms: 20.0
    step_ms: 2.0
    penalize_abs_shift_ms: true

  export:
    selected_wavelet_name: global_wavelet_201ms.csv
    write_unified_synthetics: true
    write_shifted_tdt: true
```

`source_runs.mode: latest` 表示默认自动发现最新的 `well_auto_tie` 和 `log_preprocess` 产物；复现实验时可以显式填写目录。

## 输出

默认输出目录建议为：

```text
scripts/output/wavelet_batch_synthetic_<timestamp>/
```

核心文件：

- `candidate_wavelets.csv`：进入评测的候选子波清单。
- `wavelet_candidate_metrics.csv`：候选子波 × 评测井的逐项指标。
- `wavelet_candidate_aggregate.csv`：每个候选子波的聚合指标。
- `selected_wavelet.csv`：最终选中的全局子波。
- `selected_wavelet_summary.json`：选择依据、配置、聚合指标和排除原因。
- `batch_synthetic_metrics.csv`：使用全局子波后的逐井合成记录指标。
- `synthetic_qc/*.csv`：每口井的地震、反射系数、合成记录和残差。
- `shift_scans/*.csv`：每口井的 bulk shift 扫描结果。
- `shifted_time_depth/*.csv`：可选，使用全局子波后得到的微调时深表。
- `figures/wavelet_candidate_matrix.png`：候选子波 × 井的相关系数矩阵。
- `figures/wavelet_candidate_aggregate.png`：候选子波聚合指标图。
- `figures/wells/*.png`：每口井使用全局子波后的 QC 图。
- `run_summary.json`：输入、配置、候选数量、评测井数量和推荐结果。

`wavelet_candidate_metrics.csv` 建议字段：

| 字段 | 含义 |
| --- | --- |
| `candidate_wavelet` | 候选子波名 |
| `source_well` | 子波来源井 |
| `eval_well` | 评测井 |
| `is_source_well` | 评测井是否就是来源井 |
| `route` | 评测井第四步路径 |
| `corr` | 最佳匹配相关系数 |
| `nmae` | 最佳匹配 NMAE |
| `best_shift_ms` | 最佳 bulk shift |
| `scale` | 最小二乘振幅缩放 |
| `n_eval_samples` | 参与评价的样点数 |
| `status` | `ok`、`failed`、`skipped` |
| `reasons` | 失败或跳过原因 |

`wavelet_candidate_aggregate.csv` 建议字段：

| 字段 | 含义 |
| --- | --- |
| `candidate_wavelet` | 候选子波名 |
| `source_well` | 子波来源井 |
| `n_eval_wells` | 成功评测井数 |
| `median_corr` | 全井中位相关系数 |
| `p10_corr` | 低分位相关系数，衡量短板 |
| `median_nmae` | 全井中位 NMAE |
| `median_abs_shift_ms` | 全井中位绝对时移 |
| `loo_median_corr` | 排除来源井后的中位相关系数 |
| `loo_p10_corr` | 排除来源井后的低分位相关系数 |
| `score` | 选择用综合分数 |
| `rank` | 排名 |
| `selected` | 是否被选中 |

## 处理逻辑

### 候选子波池

第五步不直接扫描 `wavelets/*.csv` 决定候选，而是读取第四步的 `wavelet_inventory.csv`。

进入候选池的条件：

- `usable_as_candidate == true`
- 子波文件存在并能读取。
- `dt_s` 与地震采样间隔一致。
- 子波长度、中心位置和能量归一化满足第四步约定。
- 来源井第四步指标不低于 `candidate_filter` 阈值。
- 未被人工排除。

这样可以避免失败井或质量很差的 auto-tie 子波进入全局评选。

### 评测井集合

评测井来自第四步成功标定的井，要求：

- `well_tie_metrics.tie_status == success`
- 有优化后时深表。
- 有第四步保存的井旁或轨迹地震道。
- 有第三步预处理 LAS，且 `usable_p_sonic` 和 `usable_density` 为 true。

直井和斜井都可以参与评测。区别在于地震道来源不同：直井使用井口道，斜井使用第四步已经保存的轨迹地震道。第五步不重新决定斜井怎么取道。

### 单井评测

对每个候选子波和每口评测井：

1. 读取该井预处理 LAS，构造 `Vp/Rho` 的 `grid.LogSet`。
2. 读取该井第四步优化后的 `TimeDepthTable`。
3. 将井曲线转换到 TWT，计算反射系数。
4. 读取第四步保存的地震道。
5. 用候选子波与反射系数卷积，生成合成记录。
6. 可选扫描一个较窄的 bulk shift，记录最佳相关系数、NMAE 和时移。

重要约束：

- 第五步不重新运行 `autotie.tie_v1`。
- 第五步不为每个候选子波重新优化时深表。
- bulk shift 只是残余对齐评测，范围应比第四步 auto-tie 更窄，并且选择全局子波时要惩罚过大的 shift。

这能让评测更像“同一套井约束下比较子波泛化能力”，而不是让每个候选子波重新过拟合。

### 全局子波选择

来源井上的分数天然偏高，所以默认用 leave-source-out 指标选子波：

- 如果候选子波来自 A1 井，计算聚合分数时优先看除 A1 以外的井。
- 如果排除来源井后的评测井数小于 `min_leave_source_out_well_count`，则退回全井聚合，但在 summary 中标注风险。
- 如果所有候选的成功评测井数都小于 `min_eval_well_count`，默认不声称完成全局优选，而是按 `on_insufficient_eval_wells` 执行降级策略。推荐降级策略是 `select_best_source_tie`：选择第四步来源井 auto-tie 指标最可靠的子波，并在 `selected_wavelet_summary.json` 中标记为 `selection_mode: insufficient_eval_fallback`。

推荐综合分数：

```text
score =
  corr_weight * loo_median_corr
  + p10_corr_weight * loo_p10_corr
  - nmae_weight * loo_median_nmae
  - shift_penalty_weight * min(median_abs_shift_ms / shift_penalty_scale_ms, 1.0)
```

这里 `p10_corr` 用来惩罚“中位数很好但少数井很差”的候选子波。`shift_penalty_scale_ms` 用来把毫秒级时移归一到与相关系数、NMAE 可比较的量纲，默认可取第五步残余 shift 扫描半宽。最终选择不只看最高中位相关系数，还要看低分位、NMAE、时移和成功覆盖井数。

如果两个候选分数接近，优先选择：

1. 评测井覆盖数更多的子波。
2. `p10_corr` 更高的子波。
3. `median_abs_shift_ms` 更小的子波。
4. 来源井第四步 auto-tie 指标更可靠的子波。

### 批量合成输出

选出全局子波后，再对所有评测井生成统一批量合成结果：

- `batch_synthetic_metrics.csv`
- `synthetic_qc/*.csv`
- `shift_scans/*.csv`
- 可选 `shifted_time_depth/*.csv`

这一部分才对应深度域 `wavelet_batch_synthetic_depth.py` 的角色。区别是时间域已有第四步的优化时深表，因此第五步默认只做窄范围 bulk shift，不再做大范围整体校正。

## 和第四步的关系

第四步产出的子波应放在：

```text
scripts/output/well_auto_tie_<timestamp>/wavelets/
```

但第五步不应该只靠这个文件夹扫文件，而应读取：

```text
scripts/output/well_auto_tie_<timestamp>/wavelet_inventory.csv
```

原因是文件夹只能告诉我们“有一个 CSV”，不能告诉我们这口井的 route、第四步指标、子波是否可用、是否应该进入候选池。

第四步同时产出的 `optimized_tdt` 和 `seismic_trace` 是第五步评测的固定基准。第五步的公平性来自：所有候选子波都在同一口井的同一条地震道、同一张优化时深表上比较。

## 模块边界

第五步优先复用第四步和深度域已有工具，少新增模块。模块归属以 `斜井支持的 src/cup 重构规划` 为准：`cup.well.tie` 只放 tie artifact、候选子波评测和 wtie Adapter；轨迹、空间样点和地震取样不放进这里。

### 建议新增到 `cup.well.tie`

- `load_tie_artifacts(auto_tie_dir) -> TieArtifactIndex`
- `load_candidate_wavelets(wavelet_inventory) -> list[WaveletCandidate]`
- `build_reflectivity_for_tie_eval(logset, table, dt_s)`
- `evaluate_wavelet_on_well(candidate, well_artifact, policy) -> WaveletWellMetric`
- `aggregate_wavelet_metrics(metrics, policy) -> DataFrame`
- `select_global_wavelet(aggregate, policy) -> WaveletSelection`

### 建议复用

- `cup.well.wavelet.load_wavelet_csv()`
- `cup.well.wavelet.infer_wavelet_dt()`
- `cup.well.wavelet.compute_wavelet_active_half_support_s()`
- 第四步 `cup.well.tie` 中的 tie artifact 索引与 wtie 输入构造能力。
- `cup.well.td` 中的 `TimeDepthTable` 读取和时深关系转换能力。
- `cup.well.spatial_samples` 中的斜井空间样点能力；第五步通常只读取第四步保存的地震道，不重新决定斜井取道。
- 深度域 `metrics_for_synthetic()`、`make_eval_mask()` 的思想，但应迁移到 `cup.well.tie` 或公共评测工具中，不要从脚本互相 import。

这里的模块分工与第四步保持一致：`cup.well.td` 负责 TDT、MD/TWT 和时深关系转换；`cup.well.las` 负责标准 LAS 中的 `DT_USM -> Vp` 读取；`spatial_samples` 只负责样点落到空间和 trace/sample。第五步若只消费第四步保存的 `seismic_trace`，通常不需要重新调用 `trace_sampling`。

## 脚本层负责

`wavelet_batch_synthetic.py` 负责：

- 解析配置和自动发现前置目录。
- 读取第四步 artifact index。
- 组织候选子波 × 评测井的批量评测。
- 写 CSV/JSON 报告和 QC 图。
- 复制或写出最终 `selected_wavelet.csv`。
- 用最终子波生成统一批量合成结果。

它不应该自己实现：

- LAS 解析和单位转换。
- 时深表解析。
- 斜井轨迹取道。
- 第四步 auto-tie 微调。

## 已定策略

- 第五步保留脚本名 `wavelet_batch_synthetic.py`，但职责扩展为“候选子波评测 + 全局子波选择 + 批量合成”。
- 第四步每口成功井的子波统一放在 `wavelets/`，并通过 `wavelet_inventory.csv` 暴露给第五步。
- 第五步不重新 auto-tie，只在固定优化时深表上评测候选子波。
- 默认使用 leave-source-out 聚合，降低来源井过拟合影响。
- 最终只输出一个全局子波给后续反演流程。

## 留到第二轮

- 是否允许按区块、层段或井型选择多个子波。
- 是否把候选子波先做相位/极性归一化后再评测。
- 是否用频谱相似度、主频、带宽作为选择惩罚项。
- 是否把密井网冲突作为权重，避免一簇密集井主导全局选择。
- 是否把第五步的统一 bulk shift 反写成后续低频模型和井约束使用的标准时深表。
