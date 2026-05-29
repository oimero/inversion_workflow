# 05 全局子波生成与批量合成

本文讨论第五个规划脚本：`global_wavelet_generation.py`。

第四步 `well_auto_tie.py` 会为每口成功井产出优化时深表、井旁或轨迹地震道，以及一条自动提取的子波。第四步输出的子波已经经过居中裁剪和 L2 能量归一化；第五步不把这些子波简单“再归一化后投票”，而是先校验它们满足第四步约定，再用它们张成一个低维子波形态空间，优化生成一条新的全局共识子波。

第五步仍会做批量合成和 QC。区别是：输出给后续反演流程的 `selected_wavelet.csv` 不一定来自某口井，而可能是 `optimized_consensus` 生成的新子波。

参考脚本：

- `scripts/wavelet_batch_synthetic_depth.py`
- `scripts/frequency_split_diagnosis_depth.py`
- `scripts/well_auto_tie.py`

深度域的 `wavelet_batch_synthetic_depth.py` 是“给定一个子波，批量做井旁合成记录和 bulk shift”。本轮时间域第五步要再向前走一步：用第四步的多井子波生成一个全局子波，再用这个全局子波生成统一的批量合成结果。

## 目标

`global_wavelet_generation.py` 回答六件事：

1. 第四步产出的哪些子波可以进入全局子波生成池。
2. 每条候选子波在每口可评测井上的合成记录匹配效果如何。
3. 如何避免“来源井子波只在自己井上特别好”的过拟合。
4. 如何用候选子波构造低维形态空间，并优化生成新的全局共识子波。
5. 生成的全局子波是否比任何原始候选子波更稳。
6. 使用全局子波后，每口井的统一合成记录、指标和可选 bulk shift 结果是什么。

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
global_wavelet_generation:
  source_runs:
    mode: latest
    well_auto_tie_dir: null
    log_preprocess_dir: null

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

  wavelet_qc:
    expected_l2_energy: 1.0
    l2_energy_tolerance: 1e-5
    max_center_abs_time_s: 1e-9
    allow_small_renormalization: true
    reject_zero_or_nan: true

  generation:
    mode: optimize_consensus          # select_existing / family_mean / optimize_consensus
    pca:
      n_components: 4
      coefficient_bounds: quantile
      coefficient_quantiles: [0.05, 0.95]
      include_mean_wavelet: true
    optimizer:
      strategy: random_then_nelder_mead
      random_trials: 512
      max_refine_iters: 120
      seed: 20260529
    objective:
      name: spatial_debiased_score
      corr_weight: 1.0
      p10_corr_weight: 0.5
      nmae_weight: 0.5
      shift_penalty_weight: 0.2
      shift_penalty_scale_ms: 20.0
      deviation_from_mean_weight: 0.15
      roughness_weight: 0.05
      bandwidth_drift_weight: 0.05
      max_allowed_side_lobe_ratio: null

  spatial_debias:
    enabled: true
    cluster_radius_m: 600.0
    aggregation: cluster_median_then_global_median
    leave_source_out_mode: source_cluster

  scoring:
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
scripts/output/global_wavelet_generation_<timestamp>/
```

核心文件：

- `candidate_wavelets.csv`：进入生成池的候选子波清单。
- `wavelet_qc.csv`：候选子波采样、中心、长度、L2 能量等 QC。
- `wavelet_candidate_metrics.csv`：原始候选子波 × 评测井的逐项指标。
- `wavelet_candidate_aggregate.csv`：原始候选子波的聚合指标。
- `wavelet_basis.csv`：均值子波和 PCA basis。
- `consensus_search_trials.csv`：共识子波优化搜索记录。
- `consensus_wavelet_metrics.csv`：生成子波 × 评测井的逐项指标。
- `selected_wavelet.csv`：最终输出的全局子波，可能是生成子波，也可能是降级选择的原始候选。
- `selected_wavelet_summary.json`：生成方式、优化参数、选择依据、配置、聚合指标和降级原因。
- `batch_synthetic_metrics.csv`：使用全局子波后的逐井合成记录指标。
- `synthetic_qc/*.csv`：每口井的地震、反射系数、合成记录和残差。
- `shift_scans/*.csv`：每口井的 bulk shift 扫描结果。
- `shifted_time_depth/*.csv`：可选，使用全局子波后得到的微调时深表。
- `figures/wavelet_basis.png`：均值子波、PCA basis 和解释方差。
- `figures/consensus_search.png`：优化搜索收敛和候选对比。
- `figures/wavelet_candidate_matrix.png`：候选子波 × 井的相关系数矩阵。
- `figures/wells/*.png`：每口井使用全局子波后的 QC 图。
- `run_summary.json`：输入、配置、候选数量、评测井数量和最终生成模式。

`wavelet_candidate_metrics.csv` 建议字段：

| 字段 | 含义 |
| --- | --- |
| `candidate_wavelet` | 候选子波名 |
| `source_well` | 子波来源井 |
| `eval_well` | 评测井 |
| `is_source_well` | 评测井是否就是来源井 |
| `spatial_cluster_id` | 评测井空间簇 |
| `route` | 评测井第四步路径 |
| `corr` | 最佳匹配相关系数 |
| `nmae` | 最佳匹配 NMAE |
| `best_shift_ms` | 最佳 bulk shift |
| `scale` | 最小二乘振幅缩放 |
| `n_eval_samples` | 参与评价的样点数 |
| `status` | `ok`、`failed`、`skipped` |
| `reasons` | 失败或跳过原因 |

`consensus_search_trials.csv` 建议字段：

| 字段 | 含义 |
| --- | --- |
| `trial_id` | 搜索编号 |
| `generation_mode` | `optimize_consensus` |
| `coef_0...coef_k` | PCA 系数 |
| `median_corr` | 全井中位相关系数 |
| `p10_corr` | 低分位相关系数 |
| `spatial_debiased_median_corr` | 空间去偏后的中位相关系数 |
| `spatial_debiased_p10_corr` | 空间去偏后的低分位相关系数 |
| `median_nmae` | 全井中位 NMAE |
| `median_abs_shift_ms` | 全井中位绝对时移 |
| `deviation_from_mean` | 与均值子波的距离 |
| `roughness` | 二阶差分粗糙度 |
| `score` | 优化目标 |
| `selected` | 是否最终选中 |

## 处理逻辑

### 候选子波池

第五步不直接扫描 `wavelets/*.csv` 决定候选，而是读取第四步的 `wavelet_inventory.csv`。

进入候选池的条件：

- `usable_as_candidate == true`
- 子波文件存在并能读取。
- `dt_s` 与地震采样间隔一致。
- 子波长度、中心位置和 L2 能量满足第四步约定。
- 来源井第四步指标不低于 `candidate_filter` 阈值。
- 未被人工排除。

第四步已经通过 `crop_wavelet_center_energy_normalize()` 将子波居中裁剪并做 L2 能量归一化。第五步读取后只做 QC：

- 若 L2 能量在容差内偏离 1，可在 `allow_small_renormalization: true` 时做一次数值重归一化，并在 `wavelet_qc.csv` 记录。
- 若中心不在 0 附近、采样间隔不一致、长度不一致或能量异常，默认拒绝该候选。
- 第五步不做自由相位校正，也不自动翻转极性；这些动作会改变第四步 auto-tie 的物理含义，必须作为显式配置另行设计。

### 评测井集合

评测井来自第四步成功标定的井，要求：

- `well_tie_metrics.tie_status == success`
- 有优化后时深表。
- 有第四步保存的井旁或轨迹地震道。
- 有第三步预处理 LAS，且 `usable_p_sonic` 和 `usable_density` 为 true。

直井和斜井都可以参与评测。区别在于地震道来源不同：直井使用井口道，斜井使用第四步已经保存的轨迹地震道。第五步不重新决定斜井怎么取道。

### 空间去偏

密井网中，一簇近井不应该拥有多倍投票权。第五步先给评测井建立空间簇，再用两级聚合评分。

第一版使用井口 XY 或第四步 tie window 的代表 XY。若斜井需要更严格，可后续切换为目标窗中点对应的轨迹 XY。

空间簇构建规则：

1. 计算评测井之间的平面距离。
2. 距离小于 `spatial_debias.cluster_radius_m` 的井连边。
3. 连通分量作为 `spatial_cluster_id`。
4. 输出 `evaluation_well_spatial_clusters.csv`，记录每口井的 XY、簇编号和簇大小。

候选或生成子波的聚合指标不直接对所有井取 median，而是：

```text
cluster_corr[c] = median(corr of wells in cluster c)
spatial_debiased_median_corr = median(cluster_corr)
spatial_debiased_p10_corr = p10(cluster_corr)
```

NMAE 和 bulk shift 同理先簇内聚合，再簇间聚合。这样 10 口近井只相当于一个空间簇的投票权。

### 单井评测

对每条原始候选子波、每条搜索生成子波和每口评测井：

1. 读取该井预处理 LAS，构造 `Vp/Rho` 的 `grid.LogSet`。
2. 读取该井第四步优化后的 `TimeDepthTable`。
3. 将井曲线转换到 TWT，计算反射系数。
4. 读取第四步保存的地震道。
5. 用子波与反射系数卷积，生成合成记录。
6. 可选扫描一个较窄的 bulk shift，记录最佳相关系数、NMAE 和时移。

重要约束：

- 第五步不重新运行 `autotie.tie_v1`。
- 第五步不为每个子波重新优化时深表。
- bulk shift 只是残余对齐评测，范围应比第四步 auto-tie 更窄，并且生成全局子波时要惩罚过大的 shift。

这能让评测更像“同一套井约束下比较子波泛化能力”，而不是让每条子波重新过拟合。

### 共识子波生成

默认生成模式为 `optimize_consensus`。它不是从候选里选一条，而是在候选子波张成的低维空间中搜索一条新的全局子波。

第一步：构造子波矩阵。

```text
W shape = [n_candidates, n_samples]
```

每行是一条已经通过 QC 的第四步子波。第四步子波应已 L2 归一化；第五步只做校验和容差内修正。

第二步：构造 PCA basis。

```text
mean_wavelet = mean(W)
pc_1 ... pc_k = PCA(W - mean_wavelet)
```

`n_components` 不宜太大，默认 4。候选井数量少时，实际维度取：

```text
k = min(generation.pca.n_components, n_candidates - 1)
```

第三步：用 PCA 系数表示全局子波。

```text
w(a) = mean_wavelet + a1 * pc_1 + a2 * pc_2 + ... + ak * pc_k
w_norm(a) = w(a) / ||w(a)||_2
```

第四步：限制搜索空间。

每条候选子波都可以投影到 PCA 空间，得到候选系数分布。默认用分位数限制搜索范围：

```text
a_j in [q05(candidate_coef_j), q95(candidate_coef_j)]
```

这样生成子波只能在已有可信子波的形态邻域内移动，不允许优化器发明远离候选集合的怪异波形。

第五步：优化目标。

推荐目标：

```text
score =
  corr_weight * spatial_debiased_median_corr
  + p10_corr_weight * spatial_debiased_p10_corr
  - nmae_weight * spatial_debiased_median_nmae
  - shift_penalty_weight * min(spatial_debiased_median_abs_shift_ms / shift_penalty_scale_ms, 1.0)
  - deviation_from_mean_weight * normalized_deviation_from_mean
  - roughness_weight * normalized_roughness
  - bandwidth_drift_weight * normalized_bandwidth_drift
```

这里：

- `spatial_debiased_median_corr` 控制整体稳定性。
- `spatial_debiased_p10_corr` 惩罚空间短板。
- `spatial_debiased_median_nmae` 防止只靠相关性忽略振幅误差。
- `median_abs_shift_ms` 防止子波靠残余时移补偿。
- `deviation_from_mean` 防止生成子波偏离候选子波族太远。
- `roughness` 防止振铃。
- `bandwidth_drift` 防止主频/带宽漂移到候选集合外。

第六步：搜索策略。

第一版建议用确定随机种子的两阶段优化：

1. `random_trials` 在系数范围内随机采样，评测每条生成子波。
2. 取前若干名作为初值，用 Nelder-Mead 或 Powell 做局部 refine。
3. 每次评测都写入 `consensus_search_trials.csv`。

由于维度很低，且评测井数量有限，这比在原始样点空间直接优化稳定得多。

### 降级策略

如果候选子波太少或优化不可信，脚本必须降级而不是硬生成：

- `n_candidates < 2`：不能 PCA，降级为 `select_existing`。
- `n_candidates == 2`：只允许一维 PCA，且加大 `deviation_from_mean_weight`。
- 所有候选评测井数不足：按 `on_insufficient_eval_wells` 降级。
- 优化出的最佳共识子波没有超过最佳原始候选的分数阈值：可选择原始候选，summary 写 `selection_mode: existing_candidate_wins`。
- 生成子波触发侧瓣、粗糙度或带宽硬限制：拒绝该 trial。

建议记录：

```text
selected_wavelet_summary.json.selection_mode
  = optimized_consensus
  | existing_candidate_wins
  | insufficient_eval_fallback
  | pca_unavailable_fallback
```

### 批量合成输出

生成全局子波后，再对所有评测井生成统一批量合成结果：

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

第四步同时产出的 `optimized_tdt` 和 `seismic_trace` 是第五步评测的固定基准。第五步的公平性来自：所有候选子波和生成子波都在同一口井的同一条地震道、同一张优化时深表上比较。

第四步子波的归一化事实：

- `scripts/well_auto_tie.py` 调用 `crop_wavelet_center_energy_normalize()` 写出 `wavelets/wavelet_201ms_<well>.csv`。
- 该函数会居中裁剪到目标长度，并将 L2 能量归一化到 1。
- 第五步需要校验这个约定，而不是把未归一化子波当作输入前提。

## 模块边界

第五步优先复用第四步和深度域已有工具，但不把所有新能力放进 `cup.well.tie`。模块归属以 `斜井支持的 src/cup 重构规划` 为准：`cup.well.tie` 只承接“第四步标定产物索引 + 固定时深表上的单井合成评测”；PCA 和共识优化进入独立模块；空间去偏拆成通用聚类/聚合工具加井震评测薄包装。轨迹、空间样点和地震取样不放进 `tie.py`。

### `cup.well.tie`

`tie.py` 现有职责是路由规划、搜索空间和合成记录指标。第五步可以扩展它，但不让它承担全局子波算法。

已新增或建议继续扩展：

- 已新增 `TieArtifactIndex`，从第四步输出目录读取 `well_tie_plan.csv`、`well_tie_metrics.csv`、`wavelet_inventory.csv`、`optimized_tdt` 和 `seismic_trace` 的路径索引。
- 已新增 `WaveletCandidate`、`TieEvaluationWell`、`WaveletWellMetric` 等 dataclass，避免脚本长期传裸 dict/DataFrame 行。
- 已新增 `load_tie_artifacts(auto_tie_dir) -> TieArtifactIndex`。
- 已新增 `load_candidate_wavelets(index, policy) -> list[WaveletCandidate]`。
- 已新增 `load_evaluation_wells(index, status="success") -> list[TieEvaluationWell]`。
- 已新增 `build_well_spatial_clusters(evaluation_wells, radius_m)` 作为井震评测薄包装，核心连通分量算法不放在 `tie.py`。
- 建议新增 `build_reflectivity_for_tie_eval(logset, table, dt_s)`。
- 建议新增 `evaluate_wavelet_on_well(wavelet, well_artifact, policy) -> WaveletWellMetric`。
- 扩展或复用现有 `scaled_synthetic_metrics()`；若需要 eval mask 和窄范围 bulk shift，优先把它做成这个评测接口背后的实现细节。
- 可新增 `aggregate_wavelet_metrics(metrics, policy) -> DataFrame`，但只做非空间去偏的基础聚合。

### `cup.well.wavelet_consensus`

已新建该模块，专门承载“从候选子波生成全局子波”的算法。它不读取 LAS、TDT、地震文件，也不关心井震路由，只消费已经对齐、同采样间隔、已通过 QC 的候选子波和评测器。

已新增对象和函数：

- `WaveletBasis`：保存均值子波、PCA basis、解释方差、候选系数范围。
- `ConsensusSearchPolicy`：保存 PCA 维度、系数边界、优化器参数和随机种子。
- `WaveletGenerationTrial`、`WaveletGenerationResult`：记录搜索过程和最终选择。
- `build_wavelet_pca_basis(candidates, policy) -> WaveletBasis`。
- `generate_consensus_wavelet(basis, coefficients) -> np.ndarray`。
- `optimize_consensus_wavelet(basis, evaluator, policy) -> WaveletGenerationResult`。

这样做的好处是：以后若要从 PCA 换成 family mean、robust mean、非负组合或贝叶斯优化，只改共识子波模块；`tie.py` 仍然只负责“给定一条子波，如何在某口井上公平评测”。

### `cup.utils.statistics`

空间去偏不单独新增 `cup.well.spatial_cluster`。核心算法是通用的半径连通分量和簇内/簇间聚合，已放进现有 `cup.utils.statistics`：

- `radius_connected_components(points_xy, radius)`：距离小于半径的点连边，输出连通分量编号。
- `aggregate_cluster_then_global(df, value_columns, cluster_column, group_columns, quantiles)`：先簇内 median，再跨簇 median/quantile。

井震特定的代表点选择留在 `cup.well.tie.build_well_spatial_clusters()` 或第五步脚本中。第一版代表点可以用井口 XY 或第四步 tie window 的代表 XY；若斜井后续需要更严格，再扩展为目标窗中点轨迹 XY。

### `cup.well.wavelet`

该模块已经有以下函数，第五步应直接复用，不要重复实现：

- 已有 `load_wavelet_csv()`。
- 已有 `infer_wavelet_dt()`。
- 已有 `compute_wavelet_active_half_support_s()`。
- 已有 `validate_wavelet_dt()`。
- 已有 `crop_wavelet_center_energy_normalize()`，第四步写出子波时已经调用。

已新增：

- `validate_wavelet_normalization()`：检查 L2 能量、中心、NaN/零能量，并返回结构化 QC 结果。
- `wavelet_l2_normalize()`：从 `crop_wavelet_center_energy_normalize()` 内部抽出 L2 归一化逻辑，供第五步容差内修正和共识子波生成复用。
- `wavelet_roughness()`：二阶差分粗糙度。
- `wavelet_spectrum_features()`：主频、带宽、谱重心等 QC/正则项。

### 建议复用

- `cup.well.td` 中的 `TimeDepthTable` 读取和时深关系转换能力。
- `cup.well.las` 中标准 LAS 到 `grid.LogSet` 的读取能力。
- `cup.well.trajectory` 中的斜井空间样点能力；第五步通常只读取第四步保存的地震道，不重新决定斜井取道。
- 深度域 `metrics_for_synthetic()`、`make_eval_mask()` 的经验可以参考，但不要从深度域脚本互相 import。当前 `cup.well.tie.scaled_synthetic_metrics()` 已经覆盖 correlation、NMAE 和振幅缩放，第五步优先复用或扩展它。

这里的模块分工与第四步保持一致：`cup.well.td` 负责 TDT、MD/TWT 和时深关系转换；`cup.well.las` 负责标准 LAS 中的 `DT_USM -> Vp` 读取；`cup.well.trajectory` 负责轨迹几何和轨迹在 TWT 轴上的空间样点；第五步若只消费第四步保存的 `seismic_trace`，通常不需要重新调用 `trace_sampling`。

## 脚本层负责

`global_wavelet_generation.py` 负责：

- 解析配置和自动发现前置目录。
- 读取第四步 artifact index。
- 组织候选子波、生成子波与评测井的批量评测。
- 调用 PCA basis 构建和共识子波优化。
- 写 CSV/JSON 报告和 QC 图。
- 写出最终 `selected_wavelet.csv`。
- 用最终子波生成统一批量合成结果。

它不应该自己实现：

- LAS 解析和单位转换。
- 时深表解析。
- 斜井轨迹取道。
- 第四步 auto-tie 微调。
- 神经网络训练或 dynamic gain 体建模。

## 已定策略

- 第五步脚本职责从“候选子波评测 + 全局子波选择 + 批量合成”升级为“候选子波评测 + 全局子波生成 + 批量合成”。
- 第四步每口成功井的子波统一放在 `wavelets/`，并通过 `wavelet_inventory.csv` 暴露给第五步。
- 第五步校验第四步子波的 L2 能量归一化，不默认重新改变其物理形态。
- 第五步不重新 auto-tie，只在固定优化时深表上评测候选子波和生成子波。
- 默认使用空间去偏聚合，降低密井网主导全局目标的风险。
- 默认使用 PCA 低维空间优化生成 `optimized_consensus` 子波；候选不足或优化不可信时降级为原始候选。
- 最终只输出一个全局子波给后续反演流程。

## 留到第二轮

- 是否允许按区块、层段或井型生成多个全局子波。
- 是否允许显式相位/极性搜索；默认不做。
- 是否在 PCA 前先做 wavelet family 聚类；第一版可以不做，先靠 PCA 系数范围和正则控制。
- 是否把第五步的统一 bulk shift 反写成后续低频模型和井约束使用的标准时深表。
- 是否让 dynamic gain 评估结果反向参与全局子波生成评分；第一版建议不要耦合，避免形成闭环。
