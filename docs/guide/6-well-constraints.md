# 06 三频带井约束

`well_constraints.py` 是时间域工作流的第六步。它从第三步的预处理 LAS 出发，对每口井做轻量清洗、转换到 TWT 域，然后把井上参考波阻抗拆成三个职责明确的频带：**LFM 低频背景**、**GINN 地震可恢复频带**、**Enhance 超分辨率残差**。同时输出 GINN 井 anchor、Enhance 监督包和 LFM 控制点，供后续步骤使用。

---

## 快速开始

```bash
python scripts/well_constraints.py
python scripts/well_constraints.py --config experiments/common.yaml
python scripts/well_constraints.py --output-dir scripts/output/well_constraints_test
```

不带参数时，脚本自动发现最新的第四、五步产物，在 `scripts/output/well_constraints_<timestamp>/` 下写出结果。

---

## 运行前需要什么

第六步的曲线真值来自第三步，时深关系来自第四步，子波和空间簇来自第五步：

| 来源 | 文件 | 用途 |
|------|------|------|
| 第三步 | `well_tie_plan.csv` → `input_las` | 预处理 LAS，含 `DT_USM` 和 `RHO_GCC`，作为参考曲线的唯一真值来源 |
| 第四步 | `well_tie_metrics.csv` | 标定状态、路径、井口 XY、时深表路径、地震道路径 |
| 第四步 | `seismic_trace/seismic_trace_<well>.csv` | 保存的井旁或沿轨迹地震道，用于 GINN cutoff 正演诊断 |
| 第四步 | `trace_sample_plan/optimized_trace_sample_plan_<well>.csv` | 斜井沿轨迹的空间采样映射 |
| 第五步 | `selected_wavelet.csv` | 共识子波，定义 LFM cutoff；也是 GINN cutoff 正演诊断的卷积算子 |
| 第五步 | `batch_synthetic_metrics.csv` | 含 `spatial_cluster_id`，用于 GINN cutoff 跨平台去偏 |
| 地震数据 | ZGY 或 SEG-Y 体、解释层位 | 读取井旁地震道，确定每个样点落在哪个目标层层段内 |

---

## 配置参考

```yaml
seismic:
  file: <seismic-volume-file>
  type: zgy
  zgy_inline_chunk_size: 16

well_constraints:
  target_interval:
    horizons:
      - <top-horizon>
      - <middle-horizon>          # 可选；提供一个或多个中间层位即可按层段输出统计
      - <bottom-horizon>
    twt_unit: auto

  control_wells:
    min_batch_corr: 0.35          # 第五步合成相关系数低于此值的井不参与
    max_batch_nmae: null          # 可选 NMAE 上限
    include_wells: null           # 白名单模式
    exclude_wells: []             # 人工剔除井

  reference_conditioning:
    max_short_gap_s: 0.010        # 允许线性填补的 DT/RHO 内部缺口最大时长
    hampel_window_samples: 7      # Hampel 去尖峰窗口（奇数，≥3）
    hampel_sigma: 4.0             # 尖峰判定 σ 倍数

  frequency_bands:
    lfm:
      cutoff_mode: wavelet_left_half_amplitude
      cutoff_scale: 1.0
      manual_cutoff_hz: null

    ginn:
      mode: diagnose
      manual_cutoff_hz: null
      candidate_cutoff_hz: null
      candidate_min_right_half_ratio: 0.4
      candidate_max_right_half_ratio: 1.3
      candidate_step_hz: 5.0
      selection_corr_tolerance: 0.02
      selection_nmae_tolerance: 0.03
      fail_on_candidate_boundary: true

    reference:
      cutoff_mode: ginn_octave
      ginn_multiplier: 2.0
      max_nyquist_fraction: 0.4
      manual_cutoff_hz: null

  anchor:
    include_deviated: false

  high_supervision:
    include_deviated: false

  weights:
    mode: corr
    corr_floor: 0.3
    corr_span: 0.4
    corr_min_weight: 0.6

  lfm_controls:
    min_control_samples_per_well: 16
```

### `source_runs`

默认自动接上最新的第四、五步结果。复现实验时可按需加入 `source_runs.well_auto_tie_dir` 和 `source_runs.wavelet_generation_dir` 固定路径。地震体来自顶层 `seismic`。

### `reference_conditioning`

控制从第三步 LAS 到参考 log-AI 的清洗过程：

| 参数 | 含义 |
|------|------|
| `max_short_gap_s` | DT 和 RHO 同时缺失的内部缺口，如果映射到 TWT 的时长 ≤ 此值，允许线性插值填补。插值样点仅用于滤波和重采样支撑，**不作为观测监督**。大于此值的缺口保持 NaN，两侧独立滤波。 |
| `hampel_window_samples` | Hampel 滤波器窗口宽度，必须是奇数。检测到孤立的尖峰时用局部中位数替换。 |
| `hampel_sigma` | 判定尖峰的 σ 倍数。值越小越激进（会替换更多样点）。被替换样点同样不作为观测监督。 |

默认 `max_short_gap_s: 0.010`（10 ms）。这个值来自当前 12 口井的缺口时长分布：99% 的内部缺口 ≤ 8 ms，第一个长缺口直接跳到 100 ms。

### `frequency_bands`

三条低通默认统一使用 6 阶零相位 Butterworth、完整连续段缓冲和 `reflect` 边界模式；频带 QC 始终生成。这些稳定实现细节不写入常用 YAML，但代码默认配置仍允许专项实验覆盖。

#### `lfm`

LFM 的低通截止频率。默认 `wavelet_left_half_amplitude`：取共识子波频谱主峰左侧归一化振幅 0.5（−6 dB）交点频率，直接作为零相位 Butterworth 的 cutoff。Butterworth 自身的 −3 dB 点和过渡带提供渐变，无需再人为下移。

`cutoff_scale` 只在需要显式实验时使用（如 `0.8` 将 cutoff 收窄到 80%）。本工区第一轮保持 `1.0`。

#### `ginn`

GINN 的高截止频率——表示"GINN 能从地震里稳定恢复到的最高频率"。默认 `diagnose` 模式做正演扫描：

1. 对每个候选 cutoff，取参考井 log-AI，低通后正演合成记录，与真实地震道比较相关性和误差。
2. 先在每个空间簇内取井指标中位数（防止 platform_001 的 34 口井垄断投票），再跨空间簇取中位数。
3. 找到全局最佳相关系数的近邻平台，在平台内选**最低的** cutoff——"再降低频率会明显损害拟合，再升高频率也不会显著改善"。
4. 如果选中值落在候选的最小或最大边界上，且 `fail_on_candidate_boundary: true`，脚本写出诊断后主动失败，要求扩展候选范围。

候选频率默认围绕共识子波右侧半峰值振幅（主频右翼 −6 dB 点）自动生成。手动覆盖时可填 `candidate_cutoff_hz`，或改 `candidate_min_right_half_ratio` / `candidate_max_right_half_ratio` 调整范围。

`selection_corr_tolerance` 和 `selection_nmae_tolerance` 定义"近最佳平台"的宽度——在此范围内的 cutoff 都算和最佳基本相当。

#### `reference`

Enhance 监督真值的上限截止频率。默认 `ginn_octave`：取 `2 × f_ginn` 和 `0.4 × fs`（Nyquist 的 40%）中的较小值。

三条低通统一使用同阶零相位 Butterworth，严格满足 `0 < f_lfm < f_ginn < f_reference < Nyquist`。

### `control_wells`

控制哪些井进入第六步。`min_batch_corr` 是第五步全局子波合成记录相关系数的最低门槛——低于此值的井整体跳过。可以额外用 `exclude_wells` 剔除某口井。

### `weights`

井样点权重来自第四步 auto-tie 相关系数，经线性映射得到。映射公式为：

```text
weight = corr_min_weight + (1 - corr_min_weight) × clamp((corr - corr_floor) / corr_span, 0, 1)
```

`corr_floor` 以下的井权重退化为 `corr_min_weight`；`corr_floor + corr_span` 以上的井接近满权重。设为 `mode: uniform` 则所有井等权。

### `lfm_controls`

`min_control_samples_per_well` 定义一口井至少要有多少观测样点才参与 LFM 插值。低于此值的井会被拒绝。

---

## 脚本在做什么

脚本分五个阶段：**条件化参考曲线 → 确定三个截止频率 → 三频带拆分 → 构建空间点表 → 聚合与导出**。

### 第一阶段：条件化参考曲线

对每口第四步标定成功、且第五步合成指标达标的井：

1. 从 `well_tie_plan.input_las` 读取第三步的标准 LAS，提取 `DT_USM` 和 `RHO_GCC`。保留原始 NaN——不做无条件线性贯通。
2. 用第四步输出的优化时深表，将 MD 域曲线映射到地震 TWT 采样轴上。
3. 识别 DT 和 RHO 同时无效的段：≤ 10 ms 的短缺口用端点值线性插值；> 10 ms 的长缺口保持 NaN。
4. 在 log-AI 域做 Hampel 去尖峰：对每段连续有效数据，用滑动窗口中位数检测孤立异常值，只替换尖峰不覆盖正常段。
5. 整条曲线重采样到地震 TWT 轴，保留三条 mask：`observed_mask`（真实观测）、`interpolation_mask`（短缺口插值）、`conditioned_mask`（Hampel 替换）。

观测样点数不足 `min_control_samples_per_well` 的井在此处被拒绝。

### 第二阶段：确定三个截止频率

**LFM cutoff**：计算共识子波频谱，找到主峰、左半峰值和右半峰值频率。LFM cutoff = 左半峰值 × `cutoff_scale`（默认 1.0）。本工区共识子波左半峰值约 10 Hz，所以第一轮 LFM cutoff ≈ 10 Hz。

**GINN cutoff**：在候选频率上做正演扫描。候选频率默认从 `右半峰值 × 0.4` 到 `右半峰值 × 1.3`，步长 5 Hz。对每个候选：

1. 取条件化后的参考 log-AI，用 Butterworth 低通到候选频率。
2. 转回 AI 计算反射系数，与共识子波卷积得到合成记录。
3. 与第四步保存的真实地震道比较相关系数和 NMAE。
4. 在每个空间簇内取中位数 → 再跨簇取中位数（两级聚合消除密井平台偏差）。
5. 在"近最佳平台"内选择最低频率。

**Reference cutoff**：取 `2 × f_ginn` 和 `0.4 × Nyquist` 的较小值。本工区 2 ms 采样下 Nyquist = 250 Hz，若 f_ginn ≈ 50 Hz，则 reference cutoff ≈ 100 Hz。

三条截止频率和实际 Butterworth 频率响应写入 `figures/frequency_band_response.png`。

### 第三阶段：三频带拆分

对每口井的条件化 log-AI，依次做三次 Butterworth 低通：

```text
reference_log_ai  = LP(conditioned_log_ai, f_reference)
ginn_target_log_ai = LP(conditioned_log_ai, f_ginn)
lfm_log_ai         = LP(conditioned_log_ai, f_lfm)
ginn_band_log_ai   = ginn_target_log_ai − lfm_log_ai
enhance_residual   = reference_log_ai − ginn_target_log_ai
```

长缺口段不参与滤波——每段连续有效数据独立低通，缺口处写 NaN。这保证了缺口两侧不存在滤波泄漏。

三条频带的线性组合严格等于 reference_log_ai，没有残差丢失。

### 第四阶段：构建空间点表

对每口井的 TWT 轴上每个落在目标层段内、且处于有效频带内的样点，写出完整的空间事实：

| 字段 | 含义 |
|------|------|
| `reference_ai` / `reference_log_ai` | 参考波阻抗 |
| `lfm_ai` / `lfm_log_ai` | LFM 频带 |
| `ginn_target_ai` / `ginn_target_log_ai` | GINN 目标（LFM + 地震可恢复频带） |
| `ginn_band_log_ai` | GINN 应恢复的纯中频部分 |
| `enhance_residual_log_ai` | Enhance 应学习的高频残差 |
| `observed_well_sample` | 是否为真实观测（非插值、非 Hampel 替换） |
| `short_gap_interpolated` | 是否为短缺口插值样点 |
| `hampel_conditioned` | 是否被 Hampel 替换 |
| `weight` | 样点权重 |
| `inline_float` / `xline_float` / `flat_idx` | 空间坐标 |

**直井**：井口 XY 映射到单个最近道，全井样点共享同一空间坐标。

**斜井**：读取 `optimized_trace_sample_plan.csv`，每个 TWT 样点都有自己的 XY 和最近道。当前默认斜井不参与 GINN anchor 和高频监督（由 `anchor.include_deviated` 和 `high_supervision.include_deviated` 控制），但直井路径没有此限制。

### 第五阶段：聚合与导出

1. **聚合**：将同一 flat_idx + seismic_sample_index 上的多点（不同井可能落到同一道同一样点）按权重加权平均，解决同道冲突。
2. **导出 GINN anchor**：`ginn_target_log_ai` 作为井 anchor 值。GINN 训练时以此为井监督目标——不再用 LFM 做 anchor，避免地震损失和井监督在同一频带内互相争夺。
3. **导出 Enhance 监督**：`enhance_residual_log_ai` 作为监督目标。只有 `observed_well_sample=true` 的样点才进入监督，插值和 Hampel 替换的样点不参与。
4. **导出 LFM 控制点**：`lfm_ai` 仅来自真实观测样点，供第七步 LFM 插值使用。
5. **写出诊断和 QC**：逐井、逐候选的正演指标；簇级和全局聚合；三频带数值 trace 和对比图。

---

## 核心输出文件

所有文件在 `<output_root>/well_constraints_<timestamp>/` 下：

### 主线产物

| 文件 | 语义 | 下游使用者 |
|------|------|-----------|
| `well_constraint_points.csv` | 逐样点三频带数值、mask、空间坐标和权重 | GINN QC、Enhance 准备 |
| `lfm_control_points.csv` | 仅 `observed_well_sample=true` 的 `lfm_ai` | 第七步 LFM 插值 |
| `log_ai_anchor_time.npz` | GINN 井 anchor，目标为 `ginn_target_log_ai` | GINN 训练 |
| `well_high_supervision_time.npz` | Enhance 监督包，目标为 `enhance_residual_log_ai`，schema `enhance_residual_supervision_v2` | Enhance 训练 |
| `well_high_stats_global.json` | Enhance 残差的全局统计 | Enhance 先验设置 |
| `well_high_stats_by_layer.csv` | 分目标层段的 Enhance 残差统计 | 分层增强策略 |
| `well_high_stats_shrinkage.json` | 分层收缩系数 | 层际融合权重 |

### QC 和审计

| 文件 | 内容 |
|------|------|
| `well_constraint_qc.csv` | 每口井的状态、控制样点数、mask 统计、拒绝原因 |
| `well_anchor_conflicts.csv` | 多井落到同一道样点时的 anchor 值冲突明细 |
| `well_high_supervision_conflicts.csv` | 高频监督的同道样点冲突明细 |

### 诊断产物

| 文件 | 内容 |
|------|------|
| `ginn_cutoff_diagnostics.csv` | 每口井在每个候选 cutoff 上的正演指标（corr、NMAE、scale） |
| `ginn_cutoff_cluster_aggregate.csv` | 每个 cutoff 在每个空间簇内的中位指标 |
| `ginn_cutoff_aggregate.csv` | 每个 cutoff 的跨簇聚合指标 |
| `figures/ginn_cutoff_sweep.png` | 全局 cutoff 扫描曲线（中位 corr/NMAE + P25–P75 区间） |
| `figures/ginn_cutoff_wells/*.png` | 逐井 cutoff 扫描图 |
| `figures/frequency_band_response.png` | 共识子波频谱、左右半峰值标记、三条 Butterworth 实际频率响应 |
| `frequency_band_qc/traces/*.csv` | 单井三频带完整 TWT 轴数值（含 NaN、mask） |
| `frequency_band_qc/figures/*.png` | 单井频带对比图（reference vs LFM vs GINN target vs enhance residual + envelope） |

---

## 如何阅读结果

### 第一步：看 `run_summary.json`

先看频率决策：

```json
{
  "frequency_bands": {
    "lfm_cutoff_hz": 10.2,
    "ginn_cutoff_hz": 45.0,
    "reference_cutoff_hz": 90.0,
    "wavelet_peak_hz": 29.7,
    "wavelet_left_half_amplitude_hz": 10.2,
    "wavelet_right_half_amplitude_hz": 54.5,
    "candidate_boundary_hit": false,
    "reason": "Selected the lowest GINN cutoff on the cluster-debiased near-best waveform-fit plateau."
  }
}
```

确认：
- `0 < lfm_cutoff_hz < ginn_cutoff_hz < reference_cutoff_hz`，顺序正确。
- `candidate_boundary_hit` 为 false——选中的 cutoff 不在候选边界。
- LFM cutoff ≈ 共识子波左半峰值（本工区约 10 Hz）。如果显著偏离，检查子波频谱是否异常。

再看计数：`selected_wells`、`point_count`、`anchor_trace_count`。如果选中的井数显著少于第四步 success 的井数，进入下一步排查。

### 第二步：看 `well_constraint_qc.csv`

按 `status` 筛选：
- `selected`：正常进入的井。
- `rejected`：被拒绝的井——看 `reasons` 列。常见原因：`too_few_control_samples`（目标窗内有效观测样点不够）、`batch_corr_below_threshold`（第五步合成指标太差）。
- `failed`：处理中出错——看 `reasons` 列的具体异常信息。

关注 `observed_samples` / `short_gap_interpolated_samples` / `hampel_conditioned_samples` 三列的相对比例。如果某井插值或 Hampel 替换比例异常高（> 20%），该井的 LAS 质量可能有问题。

### 第三步：看 `ginn_cutoff_sweep.png`

这是一张两张并排的曲线图：左图是相关系数随 cutoff 变化，右图是 NMAE 随 cutoff 变化。期望看到的模式：

- 相关系数在低频（~25 Hz）较低，然后快速上升到某个平台，之后基本持平或缓慢上升。
- NMAE 在低频较高，然后下降到平台。
- 选中频率（红色虚线）应在平台区的最左端——意味"更低的频率会让拟合变差，更高的频率不会显著改善"。
- 最佳相关系数对应的频率（黑色虚线）如果显著高于选中值，说明脚本正确地选了平台内更保守的那个。

如果选中频率在 25 Hz 以下，说明 GINN cutoff 可能过低——地震能恢复的很多频率被丢给了 Enhance。如果选中频率在 55 Hz 以上，检查对应的井是否只有少数几口、是否被单一平台主导。

### 第四步：看图

**`figures/frequency_band_response.png`**：确认三条 Butterworth 的实际 −3 dB 位置与你预期的 cutoff 一致。共识子波频谱的左右半峰值标记应和 LFM/GINN cutoff 的逻辑对齐。

**`figures/ginn_cutoff_wells/*.png`**：逐井扫描图可以暴露异常井——某口井在所有 cutoff 上的 corr 都远低于其他井，说明它的地震道或时深关系可能有问题。

**`frequency_band_qc/figures/*.png`**：抽查几口井的三频带 QC 图。上图是 reference log-AI 和 LFM log-AI 的叠加——LFM 应该是一条非常平滑的趋势线。下图是 enhance residual 和它的包络——长缺口处应为空白（NaN），不应有折线连接。

### 第五步：抽查 `well_constraint_points.csv`

对一口直井，验证：
- 同一口井的所有行 `inline_float` / `xline_float` 相同。
- `observed_well_sample=true` 的行，`short_gap_interpolated` 和 `hampel_conditioned` 都为 false。
- 三频带恒等式成立：`reference_log_ai ≈ lfm_log_ai + ginn_band_log_ai + enhance_residual_log_ai`（误差在浮点精度内）。

---

### 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| `No selected conditioned well logs` | 没有井通过条件化 | 检查第四步成功井数、第五步合成指标是否太低、`min_batch_corr` 是否太高 |
| `No finite aggregate forward-modeled metrics` | GINN cutoff 扫描中所有候选所有井都失败了 | 检查子波采样间隔和地震道是否一致、共识子波是否有效 |
| `Selected GINN cutoff lies on the candidate boundary` | 选中的 cutoff 落在候选扫描范围边界上 | 扩展 `candidate_min_right_half_ratio` / `candidate_max_right_half_ratio` 或手动设 `candidate_cutoff_hz` |
| `too_few_control_samples` | 某井目标窗内有效观测样点不足 | 检查第三步 LAS 在该井目标层段内是否有足够数据；长缺口占比是否过高 |
| `missing_preprocessed_las` | 找不到第三步的预处理 LAS | 确认第四步的 `well_tie_plan.csv` 中 `input_las` 路径正确 |
| `missing_spatial_cluster_id` | 某井在第五步的 `batch_synthetic_metrics.csv` 中没有空间簇 ID | 检查第五步是否开启了 `spatial_debias.enabled` |

---

## 留到第二轮

- 斜井的 GINN anchor 和高频监督（当前 `include_deviated: false`）。
- 按层段分别设定 GINN cutoff（当前全目标窗统一）。
- 参考 cutoff 从走测井频谱噪声拐点确定（当前用 `2 × f_ginn` 的简单规则）。
- LFM cutoff 的实验性缩比（`cutoff_scale < 1`）转为显式实验开关。
- 留井或留平台交叉验证：比较 `direct / 6 Hz / 8 Hz / 10 Hz` LFM cutoff 对最终反演结果的影响。
