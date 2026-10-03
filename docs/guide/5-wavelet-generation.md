# 05 全局共识子波生成

`wavelet_generation.py` 是时间域工作流的第五步。它读取第四步每口成功井产出的子波，在所有井上做交叉评测，再在候选子波张成的低维形态空间中优化出一条全局共识子波，最后用这条子波生成统一的批量合成记录。

---

## 快速开始

```bash
python scripts/wavelet_generation.py
python scripts/wavelet_generation.py --config experiments/common/common.yaml
python scripts/wavelet_generation.py --well <well-name>
python scripts/wavelet_generation.py --debug
python scripts/wavelet_generation.py --output-dir scripts/output/<wavelet-generation-run>
```

不带参数时，脚本自动发现最新的第四步产物，在 `scripts/output/wavelet_generation_<timestamp>/` 下写出结果。

用 `--well` 可以把评测井限定为一口指定井，并将最小评测井数调整为 1；候选评测、共识搜索和最终选择仍按同一套流程执行。

---

## 运行前需要什么

| 来源 | 文件 | 用途 |
|------|------|------|
| 第四步 | `well_tie_plan.csv` | 每口井的路由、井口坐标 |
| 第四步 | `well_tie_metrics.csv` | 标定成功/失败、优化后相关系数和 NMAE、实际标定窗口 |
| 第四步 | `wavelet_inventory.csv` | 哪些子波可以作为候选、采样间隔、来源井指标 |
| 第四步 | `filtered_las/filtered_logs_<well>.las` | 读取第四步最优参数滤波后的 `DT_USM` 和 `RHO_GCC`，构造波阻抗和反射系数 |
| 第四步 | `time_depth/optimized_tdt_<well>.csv` | 优化后的时深表——第五步的固定评测基准 |
| 第四步 | `seismic_trace/seismic_trace_<well>.csv` | 井旁或沿轨迹地震道 |

---

## 配置参考

```yaml
spatial_debias:
  cluster_radius_m: 600.0

wavelet_generation:
  candidate_filter:
    min_source_tie_corr: 0.35     # 来源井 auto-tie 相关系数低于此值的子波不进入候选池
    max_source_tie_nmae: null     # 可选 NMAE 上限
    exclude_source_wells: []      # 人工排除某些井的子波
    include_source_wells: null    # 白名单模式

  evaluation_wells:
    exclude_wells: []
    include_wells: null

  wavelet_qc:
    expected_l2_energy: 1.0
    l2_energy_tolerance: 1e-5
    max_center_abs_time_s: 1e-9
    allow_small_renormalization: true

  generation:
    pca:
      n_components: 4
      coefficient_bounds: quantile
      coefficient_quantiles: [0.05, 0.95]
    optimizer:
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

  scoring:
    min_eval_well_count: 3
    on_insufficient_eval_wells: select_best_source_tie
```

### `source_runs`

默认自动接上最新一次井震标定结果。复现实验时可按需加入 `source_runs.well_auto_tie_dir` 固定输入。

### `candidate_filter`

第四步的 `wavelet_inventory.csv` 已经标记了每条子波是否 `usable_as_candidate`。第五步在这个基础上再做一层过滤：

- 来源井的 auto-tie 相关系数低于 `min_source_tie_corr` 的子波不进入候选池——来源井自己都标不好，子波不可信。
- 可以用 `exclude_source_wells` 临时剔除某口井的子波，比如怀疑它的时深表有问题。
- 设为白名单模式（填 `include_source_wells`）则只保留指定的几口井的子波。

### `wavelet_qc`

第五步会先确认候选子波的基本形态是否可比较：长度一致、中心明确、采样间隔一致、能量已经归一化。

- 如果能量只有很小的数值偏差，脚本可以重新归一化并记录在 `wavelet_qc.csv`。
- 如果中心不在 0 附近、采样间隔不一致、长度不统一、或能量异常，直接拒绝该候选。
- 子波长度必须为奇数，确保中心样点明确落在零时刻。

### `generation`

第五步固定执行共识优化：先学习候选子波的主要形态变化，再在该范围内寻找整体评分更好的子波。

#### `pca`

`generation.pca` 控制这个“形态范围”：

| 参数 | 含义 |
|------|------|
| `n_components` | 使用多少个主要形态变化方向。值越大，可搜索的子波形态越自由，也越容易追随局部噪声；候选井不多时不宜设太大。 |
| `coefficient_bounds` | 每个形态方向的搜索边界来源。`quantile` 用候选子波投影系数的分位数做边界，比直接用极值更稳。 |
| `coefficient_quantiles` | `quantile` 模式下的上下分位。`[0.05, 0.95]` 表示避开最极端的 5% 和 95% 以外形态。范围放宽会给优化更多自由，范围收窄会更保守。 |

#### `optimizer`

`generation.optimizer` 控制怎么在这个范围内搜索：

| 参数 | 含义 |
|------|------|
| `random_trials` | 随机尝试次数。次数越多越不容易漏掉好区域，但运行更慢。 |
| `max_refine_iters` | 局部细化最多迭代次数。值越大越愿意在当前高分区域继续打磨，但提升通常会递减。 |
| `seed` | 随机搜索种子。固定后，同一批输入会得到可复现的搜索过程。 |

实际调参时，通常先不要动 `n_components` 和分位范围；如果怀疑优化没有充分探索，优先增加 `random_trials`。如果共识子波形态变得不自然，优先收窄 `coefficient_quantiles` 或增加 `generation.objective` 里的形态惩罚。

#### `objective`

`generation.objective` 定义共识子波的评分方式。全局子波的评分同时考虑整体匹配、少数差井、振幅误差和子波形态是否合理：

| 项 | 作用 |
|------|------|
| `corr_weight` × 空间去偏中位相关系数 | 奖励整体匹配好的子波 |
| `p10_corr_weight` × 空间去偏 P10 相关系数 | 惩罚在少数井上特别差的子波 |
| `nmae_weight` × 空间去偏中位 NMAE | 防止只靠相关性忽略振幅误差 |
| `deviation_from_mean_weight` × 偏离均值距离 | 约束生成子波处于候选子波族的形态范围 |
| `roughness_weight` × 粗糙度 | 防止振铃 |
| `bandwidth_drift_weight` × 带宽漂移 | 防止主频偏离候选子波族太远 |

这些权重控制评分偏好：相关性负责奖励，误差和形态惩罚负责防止子波为了局部匹配而变得不可信。

### `spatial_debias`

密井网中，同一平台的近井数量较多时，逐井聚合会使该区域权重偏高。空间去偏先把近井归成空间簇，再按空间簇聚合评分，使每个空间簇以一个聚合结果参与全局评分。半径来自顶层 `spatial_debias.cluster_radius_m`。

如果不想用空间去偏，把 `enabled` 关掉即可——此时每口井等权投票。

### `scoring`

- 如果评测井总数不足 `min_eval_well_count`，脚本不尝试共识优化，降级为选择第四步来源井指标最好的候选子波。
- 共识优化产出的生成子波必须**严格优于**最佳候选子波的分数才会被选中；如果打平或更差，脚本选择最佳候选，并在 summary 中标记 `existing_candidate_wins`。

---

## 脚本在做什么

脚本按以下五个阶段处理时间域井震标定产物：

1. **加载与质量控制。** 读取第四步的路由计划、标定指标和子波清单，检查候选子波的中心、长度、采样间隔与 L2 能量，并为每口评测井预先计算波阻抗反射系数。
2. **候选子波交叉评测。** 将测深域波阻抗依据优化时深关系转换到双程旅行时域，与候选子波卷积生成合成记录，再计算相关系数和归一化平均绝对误差；按空间簇聚合逐井指标。
3. **共识子波搜索。** 对齐候选波形并建立 PCA 形态空间，在投影系数分位数范围内进行随机采样和局部细化；每个搜索点都重新执行全井正演、空间聚合和形态正则化评分。
4. **全局选择。** 比较最佳共识波形与最佳候选波形的综合分数；评测井数量低于配置下限时，选择第四步来源井指标最好的候选波形。
5. **批量合成。** 使用最终波形在全部评测井上生成合成记录、残差、批量指标和波形质量控制图，图中包含波阻抗、反射系数、合成地震、观测地震、残差和互相关结果。

---

## 核心输出文件

所有文件在 `<output_root>/wavelet_generation_<timestamp>/` 下：

### 主线决策输出

| 文件 | 什么时候看 | 内容 |
|------|------------|------|
| `selected_wavelet.csv` | 后续批量合成或反演流程读取时 | 最终输出的全局子波 |
| `selected_wavelet_summary.json` | 先看 | 选择模式、分数对比、来源井、配置摘要 |
| `wavelet_candidate_aggregate.csv` | 判断候选竞争关系时 | 候选子波的空间去偏聚合指标和综合分数 |
| `evaluation_well_spatial_clusters.csv` | 检查密井平台是否被去偏时 | 每口评测井的空间簇编号和簇大小 |
| `run_summary.json` | 复现实验时 | 输入路径、候选数、评测井数、选择模式和配置快照 |

### 批量合成 QC 输出

| 文件 | 什么时候看 | 内容 |
|------|------------|------|
| `batch_synthetic_metrics.csv` | 先按 `corr` 排序 | 使用全局子波后每口井的合成记录指标 |
| `figures/batch_synthetic_qc/*.png` | 排查单井波形时 | 六 panel 波形 QC 图（波阻抗、反射系数、合成 wiggle、观测 wiggle、残差 wiggle、互相关）；标题含 `corr`、`nmae`、`scale` 指标 |
| `synthetic_qc/*.csv` | 需要逐样点复核时 | 每口井的地震道、反射系数、合成记录和残差 |
| `figures/selected_wavelet.png` | 检查最终子波形态时 | 全局子波、最佳候选和共识子波的形态对比 |

### Debug 输出

这些文件默认不写。需要追溯候选过滤、PCA 或共识搜索过程时，使用 `--debug` 重跑。

| 文件 | 内容 |
|------|------|
| `candidate_wavelets.csv` | 进入候选池的子波清单和来源井信息 |
| `wavelet_qc.csv` | 每条候选子波的中心、长度、采样间隔和能量 QC |
| `wavelet_candidate_metrics.csv` | 候选子波 × 评测井的逐项指标 |
| `wavelet_basis.csv` | 均值子波和 PCA 主成分 |
| `consensus_search_trials.csv` | 共识搜索的每次 trial、系数、指标和分数 |
| `consensus_wavelet_metrics.csv` | 共识子波 × 评测井的逐项指标 |

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

直接看最终为什么选了这条子波：

```json
{
  "selection_mode": "optimized_consensus",
  "selected_score": 0.65,
  "best_candidate_score": 0.58,
  "candidate_count": 10,
  "evaluation_well_count": 12
}
```

如果 `selected_score` 只比 `best_candidate_score` 高一点点（< 0.02），说明共识子波的优势很微弱，建议把图和试验明细一起看，不要只凭分数接受。

### 第三步：看 `wavelet_candidate_aggregate.csv`

按 `score` 降序排列，关注：

- 最高分的候选子波是哪口井的——它的 `spatial_debiased_median_corr` 和 `spatial_debiased_p10_corr` 是否明显优于其他候选。
- 某个候选的 `p10_corr` 特别低——说明它在少数井上表现很差，即使中位数不错也不该选。
- 空间去偏聚合和普通全井中位数的差异——如果某候选在去偏后分数大幅下降，说明它的高分主要来自一个密集井簇。

### 第四步：必要时打开 debug 明细

如果对共识子波为什么赢有疑问，使用 `--debug` 重跑，再看 `consensus_search_trials.csv`：

- 随机采样阶段分数最高的几个试验和细化后最终选中的试验之间的分数差——如果细化提升很小（< 0.01），说明随机采样已经找到了足够好的区域。
- 最终选中试验的各项指标是否都在候选子波族的合理范围内（`deviation_from_mean`、`roughness`、`bandwidth_drift` 不应显著高于候选的典型值）。

### 第五步：看图

`figures/selected_wavelet.png` 叠加了最终选中的全局子波、最佳候选子波和共识子波（如果是 consensus 模式）。三条曲线应该形态接近——如果共识子波有明显的旁瓣或相位偏移，检查对应的正则化项是否需要调整权重。

### 第六步：抽查合成记录

打开 `batch_synthetic_metrics.csv`，按 `corr` 排序，检查最低分的几口井对应的 `figures/batch_synthetic_qc/*.png`，核对地震和合成记录在关键层位附近的对齐情况。如果少数井的指标明显偏低，可将其加入 `evaluation_wells.exclude_wells` 后重跑。

每井 QC 图第二个子图的红线是该井单独最小二乘缩放后的合成记录，标题里的 `scale` 就是这口井使用的缩放倍数。因此这张图适合看波形、相位和残差，不用于判断全局固定振幅尺度。

---

### 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| `No candidate wavelets passed QC` | 所有子波都没通过校验 | 使用 `--debug` 重跑，检查 `wavelet_qc.csv` 的具体失败原因：中心偏移、能量异常、长度不一致 |
| `No evaluation wells are available` | 第四步没有标定成功的井 | 检查第四步 `well_tie_metrics.csv` 的 `tie_status` |
| `No finite candidate wavelet metrics were produced` | 所有候选在所有井上的评测都失败了 | 检查子波采样间隔是否与地震道一致 |
| `wavelet dt does not match seismic trace dt` | 某条子波的采样间隔和地震道不一致 | 检查第四步子波是否是用正确的地震采样间隔导出的 |
| `insufficient_eval_fallback`（降级） | 评测井少于 `min_eval_well_count`，脚本降级为选最佳候选 | 检查第四步成功率；如果井确实少，这是预期行为 |
