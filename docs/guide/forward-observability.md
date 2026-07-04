# 正演可观测性分析

`forward_observability.py` 是工作流的研究旁路。它回答一个具体问题：**在当前全局子波、井震匹配和目标时窗下，某个频率的波阻抗扰动能否在地震上产生可辨识的响应。** 它的输出只用于可选的 `synthoseis-lite` probe benchmark 和 R0/R1 QC。

---

## 快速开始

```bash
python scripts/forward_observability.py
python scripts/forward_observability.py --config experiments/my_project.yaml
python scripts/forward_observability.py --output-dir /tmp/obs_test
python scripts/forward_observability.py --well W1  # 单井调试
```

不带参数运行时，脚本读取 `experiments/common/common.yaml`，自动发现最新合格的第三、四、五步产物，并在 `scripts/output/forward_observability_<timestamp>/` 下写出分析结果。

---

## 运行前需要什么

| 输入 | 来源 | 用途 |
|------|------|------|
| `selected_wavelet.csv` | 第五步 | 标称全局子波，所有人工扰动的基准 |
| `selected_wavelet_summary.json` | 第五步 | 子波来源校验（`source_auto_tie_dir` 必须匹配配置的第四步目录） |
| `wavelet_candidate_aggregate.csv` | 第五步 | 候选子波列表，与第四步子波清单联表加载 |
| `evaluation_well_spatial_clusters.csv` | 第五步 | 评测井的空间簇分配，用于跨簇证据聚合 |
| `batch_synthetic_metrics.csv` | 第五步 | 评测井的批量合成指标，用于筛选可分析井 |
| `well_tie_metrics.csv` | 第四步 | 井的标定状态、滤波 LAS 路径、优化 TDT 路径、地震道路径 |
| `well_tie_plan.csv` | 第四步 | 井的输入 LAS 路径（用于校验第四步与第三步 LAS 一致） |
| `wavelet_inventory.csv` | 第四步 | 候选子波的文件路径和可用性标记 |
| `well_preprocess_status.csv` | 第三步 | 预处理状态和预处理 LAS 路径 |
| 井分层文件 | Petrel | 构建目标窗口和相邻区域窗口 |

**来源校验规则：** 第五步的 `selected_wavelet_summary.json` 中记录的 `source_auto_tie_dir` 必须与配置的 `forward_observability.source_runs.well_auto_tie_dir` 指向同一目录——否则脚本以 `source_run_mismatch` 失败。第四步每口井的 `input_las` 也必须与显式配置的第三步预处理 LAS 路径一致。

---

## 配置参考

```yaml
# 以下所有字段都写在 experiments/common/common.yaml 的顶层或 forward_observability 段下。

# --- 必填 ---
forward_observability:
  frequency:
    max_hz: 80.0                                  # 必填

# --- 可选（source_runs 缺失时自动发现最新合格产物）---
forward_observability:
  source_runs:
    wavelet_generation_dir: scripts/output/wavelet_generation_<timestamp>
    well_auto_tie_dir: scripts/output/well_auto_tie_<timestamp>
    well_preprocess_dir: scripts/output/well_preprocess_<timestamp>

# --- 可选（有默认值）---
forward_observability:
  frequency:
    start_hz: 5.0                                 # 默认 5.0
    step_hz: 5.0                                  # 默认 5.0
  perturbation:
    epsilon_log_ai: 0.001                         # 默认 0.001
    tukey_alpha: 0.5                              # 默认 0.5
    phase_degrees: 10.0                           # 默认 10.0
    fractional_shift_samples: 0.5                 # 默认 0.5
    max_basis_condition_number: 1000000.0         # 默认 1e6
  thresholds:
    min_valid_samples: 50                         # 默认 50
    min_cycles: 2.0                               # 默认 2.0
    min_wells: 5                                  # 默认 5
    min_clusters: 3                               # 默认 3
    min_synthetic_rms: 1.0e-06                    # 默认 1e-6
    required_artificial_scenarios: 3              # 默认 3
    max_short_log_gap_s: 0.010                    # 默认 0.01

# 层位配置在 target_interval.horizons 下：
target_interval:
  horizons:
    - {name: top_a, well_top: Petrel Top A, file: interpre/top_a}
    - {name: middle_b, well_top: Petrel Marker B, file: interpre/middle_b}
    - {name: base_c, well_top: Petrel Base C, file: interpre/base_c}
```

### `source_runs`

三个目录均可留空——脚本默认从 `output_root` 下自动发现最新合格产物。显式填写时优先使用填写的路径，用于复现特定 run。每个目录要求包含特定文件——缺失任一文件直接报错。

### `target_interval.horizons`

从浅到深排列的层位列表，长度至少为 2。`name` 是输出使用的稳定内部层位 ID，`well_top` 是井分层 `Surface` 名，`file` 是解释层面文件；三项都必须显式配置。脚本用第一个和最后一个层位构建 `whole_target` 全目标窗口，用相邻层位对构建 `adjacent_zone` 相邻区域窗口。N 个层位产生 1 个全目标窗口和 N-1 个邻区窗口。内部层位 ID 大小写不敏感，但不允许重复。

### `frequency`

`max_hz` 必须显式配置，脚本内部还会用 0.45 倍奈奎斯特频率（`0.45 * 0.5 / dt`）做硬上限，取两者最小值。`start_hz` 和 `step_hz` 有默认值 5.0，可按需覆盖。

### `perturbation`

控制人工子波扰动的参数，用于评估算子对子波不确定性的敏感度。全部有默认值，可不配置。

#### `epsilon_log_ai`

波阻抗对数上的有限差分步长，默认 0.001。这个值在"太大会扭曲线性化近似"和"太小会让浮点噪声淹没梯度"之间折中。如果你的 AI 绝对值范围很窄（< 3 GPa·s/m），可以考虑略微增大。

#### `tukey_alpha`

时窗两端余弦锥度的比例，范围 [0, 1]，默认 0.5。控制 Tukey 窗的平坦段和衰减段的比例。alpha 越大衰减段越宽，可以抑制窗口边界的 Gibbs 效应，但也减少有效样本。

#### `phase_degrees` / `fractional_shift_samples`

人工扰动场景的幅度。脚本从标称子波出发，生成四个受控扰动：±10° 相位旋转（`artificial_phase`）和 ±0.5 样本分数时间偏移（`artificial_shift`）。这些扰动通过零填充的解析信号/傅里叶斜坡实现，天然避免了边界效应。

#### `max_basis_condition_number`

相位基矩阵的条件数上限，默认 1e6。当频率极低（窗口内不足一个周期）时，正弦和余弦列近乎线性相关，条件数会爆炸——此时脚本标记为 `ill_conditioned_phase_basis` 并跳过该场景。

### `thresholds`

#### `min_valid_samples`

窗口内最少有效样本数，默认 50。低于这个数直接标记为 `insufficient_valid_samples`。

#### `min_cycles`

每个频率在窗口内最少完整周期数，默认 2.0。实际计算：`required_samples = max(min_valid_samples, ceil(min_cycles / (freq * dt)))`。低频如果周期不够，在单口井内该频率的所有场景全标记为 `insufficient_cycles`。

#### `min_wells` / `min_clusters`

跨井证据聚合的门槛。在簇内做完中位数聚合后，跨簇取 P25 时至少有 `min_wells` 口井和 `min_clusters` 个空间簇才给出证据判定，否则标记为 `insufficient_evidence`。默认 5 口井、3 个簇——少于这个量级，证据不足以支撑频率级别的结论。

#### `required_artificial_scenarios`

每口井/窗口内必须成功的人工扰动场景数，默认 3、上限 4。如果某口井的某个窗口内有效人工场景不够（通常是低频时条件数过高导致失败），该组合不参与单井聚合。

#### `max_short_log_gap_s`

波阻抗对数曲线投影到双程旅行时轴后，允许线性插值填补的短间隙最大长度（秒），默认 0.010 s。超过此长度的间隙不填补，整个窗口被跳过。

---

## 脚本在做什么

脚本分五个阶段：**来源校验与子波加载 → 算子解析评估 → 频率网格上的逐井分析 → 场景/簇/证据三层聚合 → 实验区间推荐**。

### 第一阶段：来源校验与子波加载

校验三个来源运行目录的文件完整性，验证 `source_auto_tie_dir` 闭环。然后加载子波场景：

1. 从第五步的 `selected_wavelet.csv` 加载标称子波（`kind=nominal`）。
2. 将第五步的 `wavelet_candidate_aggregate.csv` 与第四步的 `wavelet_inventory.csv` 按规范化井名联表，只取 `usable_as_candidate=true` 的候选子波。联表失败或子波加载失败的井记入 `wavelet_scenario_qc.csv`。
3. 从标称子波出发，用 ±10° 相位旋转和 ±0.5 样本时间偏移生成四个受控人工扰动场景。

所有子波必须满足：L2 归一化（能量容差 1e-5）、奇数长度、中心在零时刻、时间轴与标称子波对齐。

### 第二阶段：算子解析评估

在显式频率网格上，计算每个子波场景的解析前向算子传递函数：

```
H(f) = W(f) * D(f)
W(f) = Σ wavelet[t] * exp(-i*2*pi*f*t)
D(f) = 0.5 * (1 - exp(-i*2*pi*f*dt))
```

其中 `D(f)` 是离散差分算子 `tanh((x[j]-x[j-1])/2)` 在频率域的精确传递函数，挂在下部样本上。`H(f)` 的归一化幅度被分类为 `core`（≥0.5）、`weak`（≥0.1）或 `unsupported`（<0.1）。所有场景的结果写入 `operator_transfer.csv`。

### 第三阶段：逐井逐频率分析

对第五步评测集中的每口井（`tie_status=success` 且 `preprocess_status=passed`）：

1. 通过优化 TDT 将滤波后的波阻抗对数（第四步）和预处理后的波阻抗对数（第三步）从测深投影到双程旅行时轴。短间隙（≤ `max_short_log_gap_s`）做线性插值填补。
2. 从有序层位构建全目标窗口和邻区窗口。
3. 对每个窗口，在连续的有效数据段上（`[window.start, window.end]` 内无长间隙），遍历频率网格和子波场景，执行 `analyze_frequency_scenario()`：

核心数值流程：

- **相位基构建：** 在 Tukey 加权下，对该频率的有限长正弦/余弦对做加权 Gram 矩阵特征分解，构造加权正交归一基，剔除近乎线性相关的基（条件数 > `max_basis_condition_number`）。
- **正演与失配基准：** 对滤波后的波阻抗对数做 Robinson 正演（`tanh` 反射率 + `numpy.convolve(mode="same")`，挂在下部样本），在 Tukey 加权下做观测值中心化/标准化后的正尺度零截距振幅拟合。输出 `corr`、`nmae` 和 `mismatch_rms`。
- **灵敏度计算：** 在相位基的两个方向上做 ±ε 对称有限差分正演，用加权 SVD 计算原始、固定尺度和尺度边缘化三种灵敏度。**保守灵敏度取尺度边缘化后的最小奇异值。**
- **可检测性：** `noise_equivalent_log_ai = mismatch_rms / sensitivity_scale_marginalized`，`detectability_ratio = preprocessed_log_ai_band_rms / noise_equivalent_log_ai`。

### 第四阶段：三层聚合

1. **场景聚合**（`well_frequency_aggregate.csv`）：每口井/窗口内跨场景取 P25 可检测比。要求：至少 1 个标称场景 + ≥ `ceil(0.5 * admitted_candidate_count)` 个候选场景（下限 3）+ ≥ `required_artificial_scenarios` 个人工场景。不够则标记 `insufficient_wavelet_scenarios`。
2. **簇聚合**（`cluster_frequency_aggregate.csv`）：每个空间簇内取中位数。
3. **证据聚合**（`frequency_evidence_bands.csv`）：跨簇取 P25。≥5 口井、≥3 个簇时给出证据状态：

| P25 可检测比 | 证据状态 |
|-------------|---------|
| ≥ 1.0 | `robust_detectable` |
| 中位数 ≥ 1.0 但 P25 < 1.0 | `conditional` |
| 中位数 < 1.0 | `not_detectable` |
| 井数或簇数不足 | `insufficient_evidence` |

### 第五阶段：实验区间推荐

将 `whole_target` 窗口的连续频率证据行合并为实验区间（`recommended_experiment_ranges.json`），按实验类别分类：

| 条件 | 类别 |
|------|------|
| 证据 `robust_detectable` 且保守算子支持 `core` | `must_recover` |
| 证据 `robust_detectable` 或 `conditional`，且保守算子支持 `core` 或 `weak` | `stress_test` |
| 其他（保守算子 `unsupported`） | `unsupported_or_unresolved` |

这些区间仅用于 `synthoseis-lite` 的设计。

---

## 核心输出文件

所有文件在 `<output_root>/forward_observability_<timestamp>/` 下：

### 1. `operator_transfer.csv` — 解析算子传递函数，每个场景/频率一行

| 字段 | 含义 |
|------|------|
| `wavelet_scenario` | 子波场景名 |
| `wavelet_scenario_kind` | `nominal` / `candidate` / `artificial_phase` / `artificial_shift` |
| `source_well` | 候选子波的来源井；标称和人工场景为空 |
| `frequency_hz` | 频率 |
| `wavelet_magnitude` | \|W(f)\| |
| `wavelet_phase_rad` | arg(W(f)) |
| `difference_magnitude` | \|D(f)\|，即 0.5 * \|1 - exp(-i*2*pi*f*dt)\| |
| `difference_phase_rad` | arg(D(f)) |
| `combined_magnitude_absolute` | \|W(f) * D(f)\| 绝对值 |
| `combined_magnitude_normalized` | 归一化到峰值的组合幅度 |
| `combined_phase_rad` | arg(W(f) * D(f)) |
| `operator_support_class` | `core` / `weak` / `unsupported`（基于归一化幅度） |
| `fft_convention` / `difference_convention` / `convolution_convention` | 算子约定文档字符串（值固定） |

### 2. `wavelet_scenario_qc.csv` — 候选子波联表和加载 QC

| 字段 | 含义 |
|------|------|
| `candidate_wavelet` | 候选子波名 |
| `source_well` | 来源井 |
| `wavelet_file` | 子波文件路径（加载成功时） |
| `status` | `ok` / `candidate_join_failed` / `invalid_wavelet` |
| `reasons` | 失败原因的详细描述 |

### 3. `well_status.csv` — 每口分析井的状态

| 字段 | 含义 |
|------|------|
| `well_name` | 井名 |
| `route` | 第四步标定路径 |
| `spatial_cluster_id` | 第五步空间簇 ID |
| `fifth_batch_corr` / `fifth_batch_nmae` / `fifth_batch_scale` | 第五步批量合成指标 |
| `status` | `ok` / `missing_input` / `source_run_mismatch` / `missing_horizon` / `outside_tdt_support` / `misordered_horizons` |
| `reasons` | 失败原因 |

### 4. `well_window_status.csv` — 每口井每个窗口的可用性

| 字段 | 含义 |
|------|------|
| `well_name` / `window_id` / `window_type` | 井和窗口标识 |
| `analysis_start_s` / `analysis_end_s` | 实际分析段起止时间 |
| `n_valid_samples` | 有效样本数 |
| `status` | `ok` / `outside_seismic_support` / `long_gap_inside_window` / `misordered_horizons` / `insufficient_valid_samples` |
| `reasons` | 失败原因 |

### 5. `well_frequency_sensitivity.csv` — 逐井逐窗口逐频率逐场景的详细指标

每行覆盖一个 `(well, window, frequency, wavelet_scenario)` 组合，几十个数值列——从 `sensitivity_raw` 到 `detectability_ratio`。OK 的行有完整指标；失败的行只有 `status` 和 `reasons`。每一行还通过联表附带了 `operator_magnitude_normalized` 和 `operator_support_class`。

### 6. `well_frequency_aggregate.csv` — 场景聚合后的单井证据

| 字段 | 含义 |
|------|------|
| `well_name` / `window_id` / `frequency_hz` | 标识 |
| `detectability_ratio` | 跨场景 P25 可检测比（NaN = 场景不足） |
| `scenario_status` | `ok` / `insufficient_wavelet_scenarios` |
| `valid_wavelet_scenario_count` | 有效场景总数 |
| `valid_candidate_wavelet_count` | 有效候选子波数 |
| `valid_artificial_perturbation_count` | 有效人工扰动数 |
| `admitted_candidate_count` / `required_candidate_wavelet_count` | 候选子波要求和实际数 |

### 7. `cluster_frequency_aggregate.csv` + `frequency_evidence_bands.csv` — 簇级和证据级聚合

`frequency_evidence_bands.csv` 的每行是一个 `(window_id, frequency_hz)` 的最终证据：

| 字段 | 含义 |
|------|------|
| `evidence_status` | `robust_detectable` / `conditional` / `not_detectable` / `insufficient_evidence` |
| `valid_well_count` / `valid_cluster_count` | 该频率下有效井数/簇数 |
| `cluster_median_detectability_ratio` / `cluster_p25_detectability_ratio` | 簇中位数和跨簇 P25 |
| `nominal_operator_support` / `conservative_operator_support` | 标称和跨场景 P25 算子支持 |
| `zone_warnings` | 全目标窗口行附带：哪些邻区的证据状态非 `robust_detectable` |

### 8. `recommended_experiment_ranges.json` — 实验区间

包含 `schema_version`、`semantics` 说明字符串和 `ranges` 列表。每个区间有 `experiment_class`（`must_recover` / `stress_test` / `unsupported_or_unresolved`）、起止频率、包含的频率列表和每个频率的证据状态与算子支持。

### 9. 图表

- `figures/operator_transfer.png` — 所有子波场景的归一化算子幅度 vs 频率，含 core/weak 阈值线。
- `figures/frequency_evidence.png` — 各窗口的簇 P25 可检测比 vs 频率，含 1.0 阈值线。
- `figures/wells/well_observability_<well>.png` — 每口可分析井的三面板 QC 图：第三步 vs 第四步波阻抗对数对比、观测地震道、单井场景 P25 可检测比 vs 频率。

### 10. `run_summary.json`

输入路径、层位列表、频率参数、扰动参数、阈值、子波场景清单、标称子波频谱特征（峰值频率、半幅频率）、井状态计数、按频率和状态分类的拒绝统计、警告和推荐实验区间。

---

## 如何阅读结果

### 第一步：看终端输出

```
Wells: 45 ok / 52 total
Wavelet scenarios: 21 (16 candidates)
Frequencies: 5-75 Hz
```

井数太少（< 10）说明第四/五步评测集太小，证据质量有限。候选子波太少（< 5）说明大部分井的子波在联表时被过滤或加载失败——查 `wavelet_scenario_qc.csv`。

### 第二步：看 `well_status.csv`

`missing_input` 的井通常是因为第四步标定失败或预处理未通过；`missing_horizon` 说明该井缺少某个配置层位的测深拾取值；`outside_tdt_support` 说明层位的测深超出了时深表覆盖范围。

### 第三步：看 `frequency_evidence_bands.csv`

筛选 `window_type == whole_target`，重点关注：

- `evidence_status == robust_detectable` 的频率——这些是 `must_recover` 候选。
- `evidence_status == not_detectable` 但 `conservative_operator_support != unsupported`——说明算子理论上能传递，但实测信噪比不够。检查 `zone_warnings` 是否有邻区拉了后腿。
- `valid_well_count < min_wells` 或 `valid_cluster_count < min_clusters`——证据不足，需要更多评测井或更大工区覆盖。

### 第四步：看 `recommended_experiment_ranges.json`

留意 `must_recover` 区间的宽度和频率范围。如果整个频带只有 `stress_test` 和 `unsupported_or_unresolved`，说明在当前数据和子波条件下，没有任何频率可以拍胸脯说"一定能恢复"——这是正常的，不代表工作流有问题。

### 第五步：抽查单井 QC 图

打开 `figures/wells/well_observability_<well>.png`：

- 左图（log AI）：第四步滤波后的波阻抗对数应该平滑地跟随第三步预处理波阻抗对数的趋势，不应出现大幅偏离。
- 中图（观测地震道）：确认目标窗口内有清晰可辨的地震反射。
- 右图（可检测比 vs 频率）：关注 P25 可检测比在哪些频率 > 1、哪些频率 < 1。如果整条线都远低于 1，这口井对任何频率都不会贡献正向证据。

### 第六步：看 `operator_transfer.png`

黑线是标称子波的算子传递幅度——它告诉你**子波本身**对不同频率的放大能力。如果 60 Hz 以上的黑线几乎贴着 0，那不管井数据多好，高频本身就传不过去——这是子波频宽决定的物理上限，不是数据质量问题。

---

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| `source_run_mismatch` | 第五步子波来源目录与配置的第四步目录不一致 | 检查 `selected_wavelet_summary.json` 中的 `source_auto_tie_dir` 是否指向正确的目录 |
| `candidate_join_failed` | 候选子波无法与第四步子波清单联表，或全部加载失败 | 检查第五步和第四步的井名是否一致、子波文件是否存在 |
| `ill_conditioned_phase_basis` | 频率太低导致窗口内正弦/余弦列近乎线性相关 | 提高 `start_hz` 或扩大目标窗口；低频这是正常现象 |
| `insufficient_valid_samples` / `insufficient_cycles` | 窗口太短或频率太低 | 放宽 `min_valid_samples` / `min_cycles`（但要理解这会让低频证据更不可靠） |
| `outside_seismic_support` | 该井的目标窗口完全在地震时间轴覆盖范围外 | 检查地震体时间范围和 TDT 覆盖 |
| `long_gap_inside_window` | 投影到 TWT 后的 log(AI) 曲线在窗口内有长间隙 | 检查 LAS 曲线在目标深度段的完整性，或放宽 `max_short_log_gap_s` |
| `configured_max_beyond_wavelet_support` | 配置的 `max_hz` 超过标称子波 1.5 倍右半幅频率 | `run_summary.json` 的 `warnings` 包含此信息；分析正常继续，但高频分析不可靠 |
| `No well/window observability records` | 所有井都被拒绝，没有产生任何分析结果 | 查看 `well_status.csv` 的拒绝原因分布 |

---

## 留到第二轮

- 三维体积级别的算子分析（当前只沿井轨迹逐道评估）。
- 逐层段而非仅窗口级别的可检测性。
- 显式的极性/相位搜索（当前用标称子波相位，靠人工扰动覆盖不确定性范围）。
- 与 `synthoseis-lite` 评估结果的双向反馈：实测恢复能力反向校准证据阈值。
- 该旁路当前只支持时间域工作流，待新增深度域 Jacobian/SVD 扩展。
