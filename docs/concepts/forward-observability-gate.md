# 前向可观测性闸门

## 文档地位

本文是[时间域反演重置](time-domain-inversion-reset.md)之后的当前研究入口。
它定义第五步之后第一个研究闸门的目的、输入、算法、输出和判定规则。

本闸门回答：

> 在当前全局子波、井震失配和目标时窗下，指定频率的阻抗扰动能否产生可辨识的地震响应？

它不回答神经网络能否恢复该频率，不选择永久性的 GINN cutoff，也不把诊断频率固化为
模型结构边界。通过本闸门后，建议频率范围只用于设计 `synthoseis-lite` 的恢复实验和
压力测试。

未来实现使用不带步骤编号的入口：

- CLI：`scripts/forward_observability.py`
- 核心模块：`src/cup/seismic/observability.py`

稳定生产链仍终止于第五步。本研究入口不是“第六步”。

## 1. 输入与来源约束

### 1.1 显式运行目录

运行必须显式配置：

- `wavelet_generation_dir`：第五步运行目录。
- `well_auto_tie_dir`：第四步运行目录。
- `well_preprocess_dir`：第三步运行目录。
- 有序井分层名称及井分层事实来源。名称必须对应井分层文件中的 `Surface` 值，
  不是 `interpre/...` 形式的地震解释层位资产路径。

最小配置形态为：

```yaml
forward_observability:
  source_runs:
    wavelet_generation_dir: <step-5-run>
    well_auto_tie_dir: <step-4-run>
    well_preprocess_dir: <step-3-run>
  horizons:
    ordered_names: [top_a, marker_b, base_c]
  frequency:
    start_hz: 5.0
    step_hz: 5.0
    max_hz: 80.0
```

`max_hz` 必须显式填写；示例中的 80 Hz 不是项目默认值。

禁止搜索 `latest`、按修改时间猜测运行目录、扫描子波文件夹或在输入缺失时回退到其他
运行。`selected_wavelet_summary.json.source_auto_tie_dir` 必须与显式指定的
`well_auto_tie_dir` 指向同一运行，否则整次运行失败。

路径解析遵循[核心 CSV 契约](csv-contracts.md)，坐标、时间和阻抗单位遵循
[数据与单位约定](data-and-coordinate-conventions.md)。

### 1.2 主基线与对照曲线

逐井使用以下事实：

- 第四步 `well_tie_metrics.csv.filtered_las_file`：主阻抗基线。
- 第四步 `well_tie_metrics.csv.optimized_tdt_file`：MD 到地震 TWT 的映射。
- 第四步 `well_tie_metrics.csv.seismic_trace_file`：实际井旁或轨迹地震道。
- 第三步 `well_preprocess_status.csv.preprocessed_las`：全频阻抗对照。

第四步 filtered LAS 直接投影到 optimized TDT，不再执行 Hampel、额外高斯平滑或
Butterworth 低通。第三步曲线必须投影到同一 TWT 轴、窗口和有效样点集合，用于量化
第四步条件化对可用阻抗幅度和非线性灵敏度的影响。

不得跨越长缺口插值。投影到 TWT 后允许线性填补不超过 `max_short_log_gap_s` 的内部短缺口，
默认 0.010 s；缺口两侧必须都有有限样点。分析仅在连续、有限且同时受 TDT、两套阻抗
曲线和地震支持的区间内进行。

### 1.3 子波场景

第五步 `selected_wavelet.csv` 是 nominal 子波。子波不确定性集合还包括：

1. 第五步准入的所有单井候选。
2. nominal 子波的 `-10 deg` 和 `+10 deg` 常相位旋转。
3. nominal 子波的 `-0.5 dt` 和 `+0.5 dt` 分数时移。

候选集合必须由第五步 `wavelet_candidate_aggregate.csv` 的
`source_well` 与第四步 `wavelet_inventory.csv.source_well` 做规范化井名后的 1:1 联接。
`candidate_wavelet` 只作为第五步场景标识，不用于匹配 `wavelet_file`。只有第五步实际
纳入汇总且第四步 `usable_as_candidate=True` 的记录有效。
联接缺失、重复或属性不一致时拒绝该候选并记录原因，不扫描目录补齐。

所有场景必须满足第五步的采样间隔、中心、奇数长度和 L2 归一化约束。人工扰动只改变
相位或亚采样时移，不增加独立振幅场景；逐井正约束最小二乘 scale 已承担整体振幅差异。
常相位旋转通过零填充后的 analytic signal 实现，分数时移通过零填充后的 Fourier phase
ramp 实现；两者都禁止 circular wrap-around，裁回原时间轴后重新 L2 归一化。时间轴、
`dt`、长度和 nominal 的零时刻样点位置保持不变，不对时移场景重新居中。

`+/-10 deg` 常相位与 `+/-0.5 dt` 时移在当前窄频带内部分相关。它们不是两个独立随机
维度，也不做笛卡尔积，只是两类可审计的小失配场景。候选子波集合负责提供形状和带宽
不确定性。

分析井集固定为第五步 `evaluation_well_spatial_clusters.csv` 中的评测井，并沿用其中的
`spatial_cluster_id`。这些井还必须在第四步 `well_tie_metrics.csv` 中为 `success`。
不得把第四步其他成功井自动加入本次分析。第五步 `batch_synthetic_metrics.csv` 仅用于
核对 nominal baseline 指标；闸门仍须在实际分析窗口和 mask 上重新计算 scale 与失配。

## 2. 通用窗口契约

配置接收任意长度、从浅到深排列的层位列表：

```yaml
forward_observability:
  horizons:
    ordered_names: [top_a, marker_b, base_c]
```

对于 `N >= 2` 个层位：

- 首层到末层生成一个 `whole_target` 窗口。
- 每对相邻层位生成 `N - 1` 个 `adjacent_zone` 窗口。
- 窗口标识由层位名称派生，不在代码中内置任何工区层位名。

井分层 MD 通过该井 optimized TDT 映射到 TWT。以下情况显式拒绝对应窗口：

- 任一边界层位缺失或非有限。
- 上下边界次序错误或窗口长度非正。
- 边界超出 optimized TDT 或地震支持范围。
- 有效连续区间不足以满足扫频的最小样点或周期要求。

全目标窗是全局判定的主窗口。相邻分层窗口只提供局部警告和诊断，不与全目标窗取硬交集，
也不能否决全目标窗结论。

## 3. 解析前向响应

令 `x = log(AI)`。相邻样点间的精确声阻抗反射系数为：

```text
r_i = (AI_(i+1) - AI_i) / (AI_(i+1) + AI_i)
    = tanh((x_(i+1) - x_i) / 2)
```

在小扰动和常值背景附近，离散线性化反射算子为 `0.5 D`。其频率响应与子波组合后为：

```text
H_0(f) = W(f) * 0.5 * (1 - exp(-i * 2*pi*f*dt))
```

该公式采用 NumPy FFT 的正变换约定
`X[k] = sum_n x[n] * exp(-i*2*pi*k*n/N)`，差分约定为
`D x[n] = x[n] - x[n-1]`。不得以连续近似 `W(f) * j*pi*f` 替代最终实现。

### 3.1 反射系数与卷积对齐

实现必须复用前五步现行 Robinson 正演约定：

- 输入 `log(AI)` 位于规则 TWT 样点 `t[0:N]`。
- `r[j] = tanh((x[j] - x[j-1]) / 2)`，`j=1,...,N-1`。
- 反射系数值挂在下部样点 `t[1:N]`，不改挂到理论中点 `t[j]-0.5dt`。
- 子波必须是奇数长度，零时刻位于中心样点。
- 使用 `numpy.convolve(wavelet, reflectivity, mode="same")`，输出沿用反射系数轴。
- observed、baseline synthetic 和所有扰动 synthetic 必须裁到同一
  `t[1:N]` 支持、使用同一 mask 和同一 Tukey 权重。

这一约定优先保证与前五步产物一致。若未来要研究半采样中点模型，必须作为不同 forward
model schema 明确比较，不能在本闸门实现中悄悄替换。

解析输出应分别记录：

- 子波 `W(f)` 的幅值和相位。
- 离散差分算子的幅值和相位。
- 联合算子 `H_0(f)` 的幅值和相位。
- FFT、差分、反射系数挂点、卷积 mode 和子波中心约定。

联合响应按每个子波场景自身的最大有限幅值归一化：

| 支持等级 | 判定 |
|----------|------|
| `core` | 归一化幅值 `>= 0.5` |
| `weak` | 归一化幅值 `>= 0.1` 且 `< 0.5` |
| `unsupported` | 归一化幅值 `< 0.1` |

这些等级只描述线性化前向算子的相对频率支持，不等同于经验可检测性。

## 4. 逐井扰动灵敏度

### 4.1 扫频、窗口与二维相位基

频率网格的默认起点和步长为 5 Hz，但 `configured_max_hz` 必须显式配置：

```text
f = 5 Hz, 10 Hz, 15 Hz, ...
f_max = min(0.45 * Nyquist, configured_max_hz)
```

`0.45 * Nyquist` 是抗 Nyquist 边界效应的硬上限，不是默认扫描终点。当前一次性工区
实例可从 `configured_max_hz=80` 开始审阅；实现不得把 80 Hz 写成项目默认值。
若配置上限超过 nominal 子波右侧半振幅频率的 1.5 倍，继续运行但在
`run_summary.json` 写入 `configured_max_beyond_wavelet_support` warning。

Tukey 权重固定使用 `alpha=0.5`。每个井窗、频率至少要求：

```text
min_required_samples = max(50, ceil(2 / (f * dt)))
```

即至少 50 个有效连续样点且至少容纳两个完整周期。不满足时分别记录
`insufficient_valid_samples` 或 `insufficient_cycles`，不能解释为该频率不可检测。
`run_summary.json` 必须按频率汇总被周期规则拒绝的井窗数量。

在相同连续 mask 上定义加权内积
`<u,v>_w = sum(w*u*v) / sum(w)`，其中 `w` 是 Tukey 权重。以未加窗的 sin/cos 列构成
`B`，令 `W=diag(w/sum(w))`，计算 Gram 矩阵 `Q=B^T W B`，再以
`B Q^(-1/2)` 得到单位加权 RMS、彼此正交的二维相位基。
`condition_number(Q) > 1e6` 或 Q 非正定时记录 `ill_conditioned_phase_basis`。

对主基线 `x_0 = log(AI_filtered)` 的两个正交基向量分别使用对称有限差分：

```text
g_k = (F(x_0 + epsilon*p_k) - F(x_0 - epsilon*p_k)) / (2*epsilon)
```

其中 `F` 使用第 3.1 节的精确 `tanh` 反射系数和卷积约定，`epsilon` 默认
`1e-3 log(AI)`。两个响应列构成 `G=[g_1,g_2]`。未知相位下的保守灵敏度来自该二维映射
的最小奇异值，不能用两个单独 RMS 的较小值代替。同一计算也在第三步全频曲线基线上
执行，用于报告条件化前后的非线性灵敏度比。

### 4.2 现实失配底

使用未加扰动的第四步主基线生成 baseline synthetic，并在同一井窗和 mask 上对实际地震
计算加权、带截距的 nuisance amplitude fit。先在 Tukey 权重下分别去均值，并将 observed
除以其加权标准差：

```text
d = (observed - weighted_mean(observed)) / weighted_std(observed)
s_0 = synthetic - weighted_mean(synthetic)
scale_0 = <d, s_0>_w / <s_0, s_0>_w
residual = d - scale_0 * s_0
```

`scale_0` 是消除整体振幅差异的回归斜率，不解释为物理增益。去均值等价于允许一个
nuisance intercept，不能因为短窗而改回无截距回归。有效窗长度由样点、周期和相位基
条件数共同控制，不另设任意的 100 样点门槛。

`weighted_std(observed)` 非正、`RMS_w(s_0) < 1e-6`、分母非正、scale 非有限或
`scale_0 <= 0` 时，分别标记 `invalid_observed_energy`、
`invalid_low_synthetic_energy` 或 `invalid_nonpositive_scale`。不得通过反极性获得
正相关。

`RMS_w(residual)` 是 standardized observed units 下的主要现实失配底。可另外在解析
算子的 weak/unsupported 范围内报告残差
谱能量，作为高频噪声诊断，但不得用它替换主要失配底或放宽判定。

### 4.3 同单位灵敏度与 scale 边缘化

有限差分响应必须转换到与失配底相同的 standardized observed units。对每个响应列：

```text
g_centered = g - weighted_mean(g)
g_fixed = scale_0 * g_centered
g_perp = g_centered - s_0 * <g_centered,s_0>_w / <s_0,s_0>_w
g_marginalized = scale_0 * g_perp
```

由两个 `g_fixed` 列和两个 `g_marginalized` 列分别构成响应矩阵。实际对
`W^(1/2) G` 求最小奇异值，在加权输出空间得到：

- `sensitivity_fixed_scale`：固定 baseline scale 时的保守灵敏度。
- `sensitivity_scale_marginalized`：允许整体 scale 重新拟合后的保守灵敏度。

主判定使用 `sensitivity_scale_marginalized`。前者只用于解释有多少响应会被整体振幅
自由度吸收。失配底与灵敏度若未进入同一振幅单位，结果必须视为实现错误。

### 4.4 窄带幅度与可检测性

在相同 Tukey 权重、频率、二维基和有效 mask 上，分别对第三步与第四步 `log(AI)` 做
加权最小二乘投影：

```text
a = (B^T W B)^(-1) B^T W x
x_band = B a
band_rms = RMS_w(x_band)
```

不能直接将未经 Gram 校正的 sin/cos 点积平方和解释为窄带幅度。每个频率至少报告：

```text
noise_equivalent_log_ai =
    mismatch_rms / sensitivity_scale_marginalized
detectability_ratio = preprocessed_log_ai_band_rms / noise_equivalent_log_ai
conditioning_amplitude_ratio =
    filtered_log_ai_band_rms / preprocessed_log_ai_band_rms
conditioning_sensitivity_ratio =
    filtered_baseline_sensitivity / preprocessed_baseline_sensitivity
```

灵敏度非正或非有限时不可计算 detectability ratio，必须保留带原因的无效记录。
第三步窄带幅度是井上可见阻抗变化的主估计，用于判断当前井曲线和 TDT 条件下是否存在
足够幅度的目标频率扰动；它可能包含测井噪声、环境影响、TDT 误差和插值高频，不自动
等同于地震可恢复的地质真值。第四步结果用于揭示经验滤波对结论的影响。

## 5. 聚合与证据分级

聚合顺序固定为：

1. 在同一井、窗口和频率内保守聚合子波场景；未知扰动相位已由二维最小奇异值处理。
2. 在同一空间簇内聚合有效井，避免密井平台重复加权。
3. 在空间簇之间计算中位数和 P25。

所有 P25 统一使用 lower empirical quantile，即 NumPy
`quantile(..., method="inverted_cdf")`，避免不同插值方法改变小样本结论。同一井的场景
P25 只有在以下条件同时满足时有效：

- nominal 场景有效。
- 有效候选数至少为 `max(3, ceil(0.5 * admitted_candidate_count))`，其中
  `admitted_candidate_count` 是第五步汇总表成功联接出的候选数。
- 四个人工扰动场景中至少三个有效。

否则记录 `insufficient_wavelet_scenarios`，不能把 nominal-only 结果当作子波不确定性下
的证据。逐表保留 `valid_wavelet_scenario_count`、`valid_candidate_wavelet_count` 和
`valid_artificial_perturbation_count`。

簇内使用有效井的中位数。全目标窗在每个频率至少需要 5 口有效井和 3 个有效空间簇，
否则状态为 `insufficient_evidence`。解析支持同时报告 nominal 等级与各有效子波场景
归一化联合响应 P25 对应的 conservative 等级。

证据状态为：

| 状态 | 判定 |
|------|------|
| `robust_detectable` | 跨簇 P25 detectability ratio `>= 1` |
| `conditional` | 跨簇中位数 `>= 1`，但 P25 `< 1` |
| `not_detectable` | 跨簇中位数 `< 1` |
| `insufficient_evidence` | 有效井或空间簇数量不足 |

相邻分层窗口使用相同统计方法，但证据不足只标记该分层结果，不阻断全目标窗输出。
解析支持等级与上述经验状态必须存为两个独立字段，不合并成单一 cutoff。

相邻频率点只有在状态相同且频率间隔等于配置步长时才能合并为连续证据区间。不得用插值
跨越 `insufficient_evidence`、无效点或状态边界。

## 6. `synthoseis-lite` 实验建议

本闸门输出建议实验区间，而不是生产目标频带：

| 经验证据 | conservative 解析支持 | 实验区间 |
|----------|-----------------------|----------|
| `robust_detectable` | `core` | `must_recover` |
| `robust_detectable` | `weak` | `stress_test`，附 operator warning |
| `conditional` | `core` 或 `weak` | `stress_test` |
| 任意经验证据 | `unsupported` | `unsupported_or_unresolved` |
| `not_detectable` 或 `insufficient_evidence` | 任意 | `unsupported_or_unresolved` |

每个建议区间附带分层窗口警告。例如，全目标窗可稳健检测但某一相邻层段证据不足时，
若 conservative 解析支持为 `core`，该区间仍可进入 `must_recover`，同时记录对应
zone warning。

后续合成基准必须独立验证逆问题的可恢复性。此处的频率范围不能直接成为网络输出 cutoff、
损失权重边界或多尺度架构分界。

## 7. 输出契约

每次运行建立独立输出目录。所有表必须包含来源运行标识、窗口定义、子波场景、有效样点数、
状态和拒绝原因；无效记录不得静默丢弃。

### `operator_transfer.csv`

逐子波场景、逐频率记录 `wavelet_magnitude/phase`、`difference_magnitude/phase`、
`combined_magnitude_absolute/normalized`、`combined_phase`、`operator_support_class`、
`fft_convention`、`difference_convention`、反射系数挂点、卷积 mode 和子波中心约定。

### `well_frequency_sensitivity.csv`

逐井、逐窗口、逐频率、逐子波场景记录：

- 井名、路由、空间簇、上下边界和窗口类型。
- 子波场景、频率、周期数、有效样点数、Tukey alpha 和 phase-basis condition number。
- baseline observed/synthetic RMS、scale、相关性、NMAE、总失配 RMS 和高频噪声诊断。
- raw、fixed-scale、scale-marginalized 灵敏度，第三步与第四步窄带幅度及
  conditioning ratios。
- noise-equivalent `log(AI)`、detectability ratio、解析支持等级和状态。
- 有效 nominal、候选、人工扰动场景数量及场景充分性状态。

### `well_frequency_aggregate.csv`

逐井、逐窗口和逐频率记录子波场景 lower empirical P25、三类有效场景数量以及
`insufficient_wavelet_scenarios` 状态。空间簇聚合只消费该表中场景充分的记录。

### `cluster_frequency_aggregate.csv`

逐空间簇、窗口和频率记录有效井数、簇内聚合 detectability ratio、解析支持和状态。

### `frequency_evidence_bands.csv`

记录全目标窗及相邻分层窗口的逐频率全局统计和连续区间，包括有效井数、有效簇数、
跨簇中位数、lower empirical P25、nominal/conservative 解析支持、场景数量 P25、
经验状态、scenario warning 及 zone warnings。

### `recommended_experiment_ranges.json`

记录 `must_recover`、`stress_test`、`unsupported_or_unresolved` 区间及其证据摘要，
并包含“仅用于 synthoseis-lite 实验设计，不是模型 cutoff”的语义声明。

### `run_summary.json`

至少记录：

- schema/version 与完整显式输入路径。
- 有序层位列表和实际生成的窗口。
- 频率网格、Tukey 参数、epsilon、最小周期数和证据阈值。
- nominal、候选和人工扰动子波场景。
- 接受/拒绝井窗数量及按原因、频率统计，特别列出 cycle requirement 拒绝数。
- nominal 子波右侧半振幅频率及 configured max 是否超出其 1.5 倍。
- 全目标窗证据摘要和建议实验区间。

### 图件

至少生成：

- 解析子波、差分算子和联合传递响应总览。
- 全目标窗的解析支持与经验证据带。
- 相邻分层窗口对比及警告。
- 逐井灵敏度、现实失配底和 conditioning bias QC。

## 8. 状态与失败策略

至少使用稳定、可机器读取的状态：

- `ok`
- `source_run_mismatch`
- `missing_input`
- `candidate_join_failed`
- `invalid_wavelet`
- `sampling_mismatch`
- `missing_horizon`
- `misordered_horizons`
- `outside_tdt_support`
- `outside_seismic_support`
- `insufficient_valid_samples`
- `insufficient_cycles`
- `long_gap_inside_window`
- `ill_conditioned_phase_basis`
- `invalid_observed_energy`
- `invalid_low_synthetic_energy`
- `invalid_nonpositive_scale`
- `invalid_sensitivity`
- `insufficient_wavelet_scenarios`
- `insufficient_evidence`

运行级来源矛盾、schema 不支持和 nominal 子波无效应使整次运行失败。单井、单窗口或候选场景
问题保留拒绝记录后继续，不得改用其他曲线、TDT、地震道、层位或 frequency target。

## 9. 实现约束与测试

跨函数和跨模块传递带采样轴的数据时，使用 `wtie.processing.grid.Log`、`Wavelet`、
`Seismic` 或项目内明确 dataclass；裸 `numpy.ndarray` 只用于局部数值计算。

实现至少覆盖以下测试：

1. 在小扰动及 `epsilon=0.1` 下验证精确 `tanh` 与线性化误差的大小和方向，能发现差分
   符号、epsilon 缩放或 Jacobian 实现错误。
2. 离散传递函数的幅值和相位均与有限差分正演响应一致，并覆盖下部样点挂点约定。
3. `scale <= 0` 被拒绝，不能依靠反极性得到有效结果。
4. 任意 `N >= 2` 的有序层位生成一个全窗和 `N - 1` 个相邻窗口。
5. 层位缺失、乱序及越界产生明确状态。
6. 候选子波只按规范化 `source_well` 联接第五步汇总与第四步 inventory，且不会把只在
   inventory 中存在的候选加入分析。
7. 空间簇去偏、P25 判定和 5 井/3 簇最小证据正确。
8. 连续区间提取不会跨越状态变化或证据缺口。
9. 第三步与第四步使用完全相同的 TWT 窗口、mask、Tukey 权重和 Gram 校正投影。
10. 来源运行不一致时整次运行失败，不发生自动回退。
11. 将 observed 同时乘任意正常数后，失配与灵敏度保持同单位，noise-equivalent
    `log(AI)` 和 detectability ratio 不发生非物理倍数漂移。
12. 只改变 baseline synthetic 整体幅度的扰动在 scale 边缘化后灵敏度接近零。
13. 非整数周期和 Tukey 窗下使用 Gram 归一化与响应矩阵最小奇异值，而非独立 sin/cos
    点积或较小 RMS。
14. 人工相位旋转和分数时移后，`dt`、时间轴、长度、零时刻定义和 L2 norm 保持契约；
    纯相位操作的幅度谱在数值容差内不变且无 circular wrap-around。
15. nominal 有效但候选或人工扰动场景不足时标记 `insufficient_wavelet_scenarios`。

截至 2026-06-15，本机曾有一次性运行目录 `wavelet_generation_20260613_015306` 及其来源
`well_auto_tie_20260613_013250`，当时包含 9 个汇总候选、11 口评测井和 7 个空间簇。
这些 gitignored 产物随时可能被清理，仅用于核对本文形成时的字段事实；文档、配置和实现
均不得依赖这些路径、数量或当前工区层位名称。

## 10. 下一研究闸门

本闸门的经验状态和解析支持只用于构造频率-振幅实验矩阵，不是反演模型的恢复上限。
下一阶段由
[Truth-First `synthoseis-lite` 基准](synthoseis-lite-benchmark.md)
生成已知阻抗真值、可控探针和二维地质场景，并在冻结报告卡上测量实际逆问题可恢复性。
