# 05 全局子波生成与批量合成

`wavelet_generation.py` 是时间域工作流的第五步。它读取第四步每口成功井产出的子波，在所有井上做交叉评测，再在候选子波张成的低维形态空间中优化出一条全局共识子波，最后用这条子波生成统一的批量合成记录。

---

## 快速开始

```bash
python scripts/wavelet_generation.py
python scripts/wavelet_generation.py --config experiments/common.yaml
python scripts/wavelet_generation.py --well <well-name>
python scripts/wavelet_generation.py --output-dir scripts/output/wavelet_generation_test
```

不带参数时，脚本自动发现最新的第四步产物，在 `scripts/output/wavelet_generation_<timestamp>/` 下写出结果。

用 `--well` 可以只在一口井上评测，方便调试——此时脚本跳过共识优化，直接选第四步来源井指标最好的候选子波。

---

## 运行前需要什么

第五步只依赖第四步产物：

| 来源 | 文件 | 用途 |
|------|------|------|
| 第四步 | `well_tie_plan.csv` | 每口井的路由、井口坐标 |
| 第四步 | `well_tie_metrics.csv` | 标定成功/失败、优化后相关系数和 NMAE、实际标定窗口 |
| 第四步 | `wavelet_inventory.csv` | 哪些子波可以作为候选、采样间隔、来源井指标 |
| 第四步 | `filtered_las/filtered_logs_<well>.las` | 读取第四步最优参数滤波后的 `DT_USM` 和 `RHO_GCC`，构造波阻抗和反射系数 |
| 第四步 | `time_depth/optimized_tdt_<well>.csv` | 优化后的时深表——第五步的固定评测基准 |
| 第四步 | `seismic_trace/seismic_trace_<well>.csv` | 井旁或沿轨迹地震道——第五步不重新取道 |

---

## 配置参考

```yaml
wavelet_generation:
  source_runs:
    mode: latest                  # 自动发现最新前置产物
    well_auto_tie_dir: null

  candidate_filter:
    min_source_tie_corr: 0.35     # 来源井 auto-tie 相关系数低于此值的子波不进入候选池
    max_source_tie_nmae: null     # 可选 NMAE 上限
    exclude_source_wells: []      # 人工排除某些井的子波
    include_source_wells: null    # 白名单模式

  evaluation_wells:
    status: success               # 只评测第四步标定成功的井
    exclude_wells: []
    include_wells: null

  wavelet_qc:
    expected_l2_energy: 1.0
    l2_energy_tolerance: 1e-5
    max_center_abs_time_s: 1e-9
    allow_small_renormalization: true

  generation:
    mode: optimize_consensus
    pca:
      n_components: 4
      coefficient_bounds: quantile
      coefficient_quantiles: [0.05, 0.95]
    optimizer:
      strategy: random_then_powell
      random_trials: 512
      max_refine_iters: 120
      seed: 20260529
    objective:
      corr_weight: 1.0
      p10_corr_weight: 0.5
      nmae_weight: 0.5
      deviation_from_mean_weight: 0.15
      roughness_weight: 0.05
      bandwidth_drift_weight: 0.05
      max_allowed_side_lobe_ratio: null

  spatial_debias:
    enabled: true
    cluster_radius_m: 600.0

  scoring:
    min_eval_well_count: 3
    on_insufficient_eval_wells: select_best_source_tie

  export:
    selected_wavelet_name: global_wavelet_201ms.csv
    write_unified_synthetics: true
```

### `candidate_filter`

第四步的 `wavelet_inventory.csv` 已经标记了每条子波是否 `usable_as_candidate`。第五步在这个基础上再做一层过滤：

- 来源井的 auto-tie 相关系数低于 `min_source_tie_corr` 的子波不进入候选池——来源井自己都标不好，子波不可信。
- 可以用 `exclude_source_wells` 临时剔除某口井的子波，比如怀疑它的时深表有问题。
- 设为白名单模式（填 `include_source_wells`）则只保留指定的几口井的子波。

### `wavelet_qc`

第四步输出的子波应该是居中、奇数长度、L2 能量为 1 的。第五步加载每条子波后做校验：

- 如果 L2 能量在容差内偏离 1.0，且 `allow_small_renormalization` 为 true，做一次数值重归一化并记录在 `wavelet_qc.csv`。
- 如果中心不在 0 附近、采样间隔不一致、长度不统一、或能量异常，直接拒绝该候选。
- 子波长度必须为奇数，确保中心样点明确落在零时刻。

第五步不做相位校正和极性翻转——这些会改变第四步 auto-tie 的物理含义。

### `generation.objective`

优化目标由六项组成：

| 项 | 作用 |
|------|------|
| `corr_weight` × 空间去偏中位相关系数 | 奖励整体匹配好的子波 |
| `p10_corr_weight` × 空间去偏 P10 相关系数 | 惩罚在少数井上特别差的子波 |
| `nmae_weight` × 空间去偏中位 NMAE | 防止只靠相关性忽略振幅误差 |
| `deviation_from_mean_weight` × 偏离均值距离 | 防止生成子波跑出候选子波族的形态范围 |
| `roughness_weight` × 粗糙度 | 防止振铃 |
| `bandwidth_drift_weight` × 带宽漂移 | 防止主频偏离候选子波族太远 |

权重都是正数，corr 项加分，NMAE/偏离/粗糙度/漂移项减分。

### `spatial_debias`

密井网中，一个平台上十几口井的同形态子波如果各自算一票，会主导全局目标。空间去偏的思路是：先把评测井按平面距离聚成空间簇（距离小于 `cluster_radius_m` 的井连通），评分时先算簇内中位数、再算簇间中位数。这样一簇近井只贡献一票。

如果不想用空间去偏，把 `enabled` 关掉即可——此时每口井等权投票。

### `scoring`

- 如果评测井总数不足 `min_eval_well_count`，脚本不尝试共识优化，降级为选择第四步来源井指标最好的候选子波。
- 共识优化产出的生成子波必须**严格优于**最佳候选子波的分数才会被选中；如果打平或更差，脚本选择最佳候选，并在 summary 中标记 `existing_candidate_wins`。

---

## 脚本在做什么

脚本分五个阶段：**加载 → 候选评测 → 共识生成 → 选择 → 批量合成**。

### 第一阶段：加载与 QC

1. 读取第四步的 `well_tie_plan.csv`、`well_tie_metrics.csv` 和 `wavelet_inventory.csv`，建立候选子波和评测井的索引。
2. 加载每条候选子波，校验中心位置、长度一致性、采样间隔和 L2 能量。通过校验的子波构成候选池。
3. 加载每口评测井的 LAS、优化 TDT 和地震道，预先计算好反射系数——这些在后续所有评测中保持不变，只算一次。

### 第二阶段：候选子波交叉评测

对候选池中的每条子波，在所有评测井上逐一做合成记录：

1. 用该井的优化 TDT 将 MD 域波阻抗转换到 TWT 域，计算反射系数。
2. 用候选子波与反射系数卷积，生成合成记录。
3. 与地震道比较，计算相关系数和 NMAE。

输出 `wavelet_candidate_metrics.csv`（候选 × 井的逐项指标）和 `wavelet_candidate_aggregate.csv`（空间去偏聚合后的综合指标）。

### 第三阶段：共识子波生成

默认不从候选里选一条，而是在候选子波张成的低维形态空间中搜索一条新的子波：

1. **构造 PCA 基底。** 把候选子波矩阵（每行一条子波）做 PCA，得到均值子波和若干主成分。实际维度不超过候选数减一，也不超过配置的 `n_components`。
2. **约束搜索范围。** 把每条候选子波投影到 PCA 空间，取系数分布的分位数作为搜索上下界。生成子波只能在已知可信子波的形态邻域内移动。
3. **两阶段搜索。** 先在系数范围内做大量随机采样（`random_trials` 次），每轮评测一条生成子波并记录分数；然后对分数最高的几个做局部细化，用无梯度优化器在系数空间内精调。
4. **每次评测都做完整交叉评测。** 每条生成子波和候选子波一样，在所有评测井上做合成记录、算空间去偏聚合指标、加上正则化项得到最终分数。

整个过程的所有 trial（随机 + 细化）都写入 `consensus_search_trials.csv`，可以追溯每一步的系数、各项指标和分数。

### 第四阶段：选择

比较最佳共识子波和最佳候选子波的分数：

- 共识子波严格优于候选 → 输出共识子波，`selection_mode = optimized_consensus`。
- 共识子波没有超过候选 → 输出最佳候选子波，`selection_mode = existing_candidate_wins`。
- 候选或评测井太少 → 降级为选第四步来源井指标最好的候选，`selection_mode = insufficient_eval_fallback`。

### 第五阶段：批量合成

用最终选定的全局子波，对所有评测井生成统一的合成记录和 QC 数据。

---

## 核心输出文件

所有文件在 `<output_root>/wavelet_generation_<timestamp>/` 下：

| 文件 | 内容 |
|------|------|
| `candidate_wavelets.csv` | 进入候选池的子波清单和来源井信息 |
| `wavelet_qc.csv` | 每条候选子波的 QC 结果：中心、长度、L2 能量、是否重归一化 |
| `evaluation_well_spatial_clusters.csv` | 每口评测井的空间簇编号和簇大小 |
| `wavelet_candidate_metrics.csv` | 候选子波 × 评测井的逐项指标（相关系数、NMAE、振幅缩放等） |
| `wavelet_candidate_aggregate.csv` | 候选子波的空间去偏聚合指标和综合分数 |
| `wavelet_basis.csv` | 均值子波和 PCA 主成分 |
| `consensus_search_trials.csv` | 共识搜索的每次 trial：系数、各项指标、分数、是否选中 |
| `consensus_wavelet_metrics.csv` | 共识子波 × 评测井的逐项指标 |
| `selected_wavelet.csv` | 最终输出的全局子波 |
| `selected_wavelet_summary.json` | 选择模式、分数对比、来源井、配置摘要 |
| `batch_synthetic_metrics.csv` | 使用全局子波后每口井的合成记录指标 |
| `synthetic_qc/*.csv` | 每口井的地震道、反射系数、合成记录和残差 |
| `figures/selected_wavelet.png` | 全局子波、最佳候选和共识子波的形态对比 |
| `run_summary.json` | 输入路径、候选数、评测井数、选择模式 |

### `wavelet_candidate_metrics.csv`

每条候选子波在每口评测井上各一行：

| 字段 | 含义 |
|------|------|
| `candidate_wavelet` | 候选子波名 |
| `source_well` | 子波来源井 |
| `eval_well` | 评测井 |
| `is_source_well` | 评测井是否就是来源井（自己评自己，通常偏高） |
| `spatial_cluster_id` | 评测井的空间簇编号 |
| `route` | 评测井的第四步标定路径 |
| `corr` | 零残余时移下的相关系数 |
| `nmae` | 零残余时移下的 NMAE |
| `scale` | 最小二乘振幅缩放因子 |
| `status` | `ok` 或 `failed` |
| `reasons` | 失败原因 |

### `consensus_search_trials.csv`

共识搜索的每次评测一行：

| 字段 | 含义 |
|------|------|
| `trial_id` | 搜索编号 |
| `coef_0 ... coef_k` | 该 trial 的 PCA 系数 |
| `spatial_debiased_median_corr` | 空间去偏中位相关系数 |
| `spatial_debiased_p10_corr` | 空间去偏 P10 相关系数 |
| `spatial_debiased_median_nmae` | 空间去偏中位 NMAE |
| `deviation_from_mean` | 与均值子波的形态偏离 |
| `roughness` | 二阶差分粗糙度 |
| `bandwidth_drift` | 主频相对候选子波均值的漂移 |
| `score` | 综合优化目标 |
| `selected` | 是否最终选中 |

---

## 如何阅读结果

### 第一步：看终端输出

```
=== Global Wavelet Generation ===
Output: scripts/output/wavelet_generation_<timestamp>
Selected: optimized_consensus (optimized_consensus), score=0.xxxx
```

`selection_mode` 告诉你最终输出了什么：`optimized_consensus`（成功生成新子波）、`existing_candidate_wins`（生成子波没有超过候选）、`insufficient_eval_fallback`（井太少，降级选择）。

### 第二步：看 `selected_wavelet_summary.json`

直接看 `selection_mode` 和分数对比：

```json
{
  "selection_mode": "optimized_consensus",
  "selected_score": 0.65,
  "best_candidate_score": 0.58,
  "candidate_count": 10,
  "evaluation_well_count": 12
}
```

如果 `selected_score` 只比 `best_candidate_score` 高一点点（< 0.02），说明共识子波的优势很微弱，建议检查对应 trial 的详细指标。

### 第三步：看 `wavelet_candidate_aggregate.csv`

按 `score` 降序排列，关注：

- 最高分的候选子波是哪口井的——它的 `spatial_debiased_median_corr` 和 `spatial_debiased_p10_corr` 是否明显优于其他候选。
- 某个候选的 `p10_corr` 特别低——说明它在少数井上表现很差，即使中位数不错也不该选。
- 空间去偏聚合和普通全井中位数的差异——如果某候选在去偏后分数大幅下降，说明它的高分主要靠一簇密井拉动。

### 第四步：看 `consensus_search_trials.csv`

在优化搜索记录中关注：

- 随机采样阶段分数最高的几个 trial 和细化后最终选中的 trial 之间的分数差——如果细化提升很小（< 0.01），说明随机采样已经找到了足够好的区域。
- 最终选中 trial 的各项指标是否都在候选子波族的合理范围内（`deviation_from_mean`、`roughness`、`bandwidth_drift` 不应显著高于候选的典型值）。

### 第五步：看图

`figures/selected_wavelet.png` 叠加了最终选中的全局子波、最佳候选子波和共识子波（如果是 consensus 模式）。三条曲线应该形态接近——如果共识子波有明显的旁瓣或相位偏移，检查对应的正则化项是否需要调整权重。

### 第六步：抽查合成记录

打开 `batch_synthetic_metrics.csv`，按 `corr` 排序，检查最低分的几口井对应的 `synthetic_qc/*.csv`——地震和合成记录是否在关键层位附近明显错位。如果少数井拖累了全局指标，考虑把它们加入 `evaluation_wells.exclude_wells` 后重跑。

---

### 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| `No candidate wavelets passed QC` | 所有子波都没通过校验 | 检查 `wavelet_qc.csv` 的具体失败原因：中心偏移、能量异常、长度不一致 |
| `No evaluation wells are available` | 第四步没有标定成功的井 | 检查第四步 `well_tie_metrics.csv` 的 `tie_status` |
| `No finite candidate wavelet metrics were produced` | 所有候选在所有井上的评测都失败了 | 检查子波采样间隔是否与地震道一致 |
| `wavelet dt does not match seismic trace dt` | 某条子波的采样间隔和地震道不一致 | 检查第四步子波是否是用正确的地震采样间隔导出的 |
| `insufficient_eval_fallback`（降级） | 评测井少于 `min_eval_well_count`，脚本降级为选最佳候选 | 检查第四步成功率；如果井确实少，这是预期行为 |

---

## 留到第二轮

- 是否允许按区块、层段或井型生成多个全局子波。
- 是否允许显式相位或极性搜索；默认不做。
- 是否在 PCA 前先做子波形态聚类；第一版先靠 PCA 系数范围和正则化控制。
- 是否增加只诊断、不反写的 residual shift scan。
- 是否让 dynamic gain 评估结果反向参与全局子波评分；第一版不要耦合，避免形成闭环。




