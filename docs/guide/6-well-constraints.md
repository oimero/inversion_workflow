# 第六步：三频带井约束

入口：

```powershell
& "C:\Users\WangQinZhuo\miniconda3\envs\pinn_inversion\python.exe" scripts\well_constraints.py --config experiments\common.yaml
```

## 职责

第六步把井上参考 `log(AI)` 拆成三个职责明确的频带：

```text
reference_log_ai
  = lfm_log_ai
  + ginn_band_log_ai
  + enhance_residual_log_ai
```

定义为：

```text
reference_log_ai = LP(conditioned_log_ai, f_reference)
lfm_log_ai = LP(conditioned_log_ai, f_lfm)
ginn_target_log_ai = LP(conditioned_log_ai, f_ginn)
ginn_band_log_ai = ginn_target_log_ai - lfm_log_ai
enhance_residual_log_ai = reference_log_ai - ginn_target_log_ai
```

- LFM 消费 `lfm_ai`。
- GINN anchor 消费 `ginn_target_log_ai`。GINN 仍是物理约束反演，不是残差学习。
- Enhance 监督消费 `enhance_residual_log_ai`。
- 禁止用 `reference_log_ai - actual_ginn_prediction_log_ai` 作为 enhance 真值。

## 输入链

曲线真值来自第三步：

```text
well_tie_plan.csv.input_las
```

第四步只提供：

- `optimized_tdt_file`
- `seismic_trace_file`
- `optimized_trace_sample_plan_file`
- 井位、route 和时间关系

第五步提供：

- `selected_wavelet.csv`
- `batch_synthetic_metrics.csv`
- `spatial_cluster_id`

第四步 `filtered_las_file` 不参与第六步频带定义，避免 auto-tie 滤波参数反向定义 GINN 可恢复频带。

## 缺口和清洗

标准 LAS loader 保留原始 NaN。

- `<=10 ms` 的有界内部缺口允许线性插值，仅用于滤波、重采样和正演支撑。
- 插值样点的 `observed_well_sample=false`，不进入 LFM 控制、GINN anchor、enhance 监督、统计或冲突报告。
- `>10 ms` 的缺口保持无效，缺口两侧按连续段独立滤波。
- 轻量 Hampel 只替换孤立尖峰；被替换样点同样不作为观测监督。

默认配置：

```yaml
well_constraints:
  reference_conditioning:
    max_short_gap_s: 0.010
    hampel_window_samples: 7
    hampel_sigma: 4.0
```

## Cutoff

### LFM cutoff

默认取共识子波主峰左侧归一化振幅 `0.5` 交点：

```yaml
frequency_bands:
  lfm:
    cutoff_mode: wavelet_left_half_amplitude
    cutoff_scale: 1.0
```

这里的振幅 `0.5` 是 `-6 dB`；Butterworth cutoff 是 `-3 dB`。工作流把前者映射为名义 cutoff，并由 Butterworth 自身提供渐变过渡。`cutoff_scale < 1` 只作为显式实验。

### GINN cutoff

对每个候选 cutoff：

1. 将参考井 `log(AI)` 低通到候选频率。
2. 转回 AI 并计算反射系数。
3. 与第五步共识子波卷积。
4. 与第四步保存的真实井旁或轨迹地震比较 corr、NMAE 和 scale。
5. 先在空间簇内取井指标中位数，再跨空间簇取中位数。
6. 在近最佳平台内选择最低 cutoff。

默认候选围绕共识子波右侧振幅 `0.5` 交点自动生成。若最终值命中候选最低或最高边界，脚本写出诊断后失败，必须扩展候选范围。

### Reference cutoff

默认：

```text
f_reference = min(2 * f_ginn, 0.4 * fs)
```

三条低通统一使用同阶零相位 Butterworth，并满足：

```text
0 < f_lfm < f_ginn < f_reference < Nyquist
```

## 核心配置

```yaml
well_constraints:
  frequency_bands:
    filter_order: 6
    buffer_seconds: null
    buffer_mode: reflect
    qc_enabled: true
    qc_envelope_window_samples: 31

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
```

旧 `well_constraints.frequency_split` 不兼容，检测到后直接报迁移错误。

## 输出

正式产物：

| 文件 | 语义 |
|---|---|
| `well_constraint_points.csv` | 点级三频带事实、mask、空间坐标和权重 |
| `lfm_control_points.csv` | 仅真实观测样点的 `lfm_ai` |
| `log_ai_anchor_time.npz` | GINN 的 `ginn_target_log_ai` anchor |
| `well_high_supervision_time.npz` | Enhance 的 `enhance_residual_log_ai`，schema `enhance_residual_supervision_v2` |
| `well_high_stats_global.json` | Enhance residual 全局统计 |
| `well_high_stats_by_layer.csv` | 分层 Enhance residual 统计 |
| `well_high_stats_shrinkage.json` | 分层收缩统计 |

诊断产物：

| 文件 | 语义 |
|---|---|
| `ginn_cutoff_diagnostics.csv` | 逐井逐候选正演指标 |
| `ginn_cutoff_cluster_aggregate.csv` | 空间簇内聚合 |
| `ginn_cutoff_aggregate.csv` | 跨空间簇聚合 |
| `figures/ginn_cutoff_sweep.png` | 全局候选 sweep |
| `figures/ginn_cutoff_wells/*.png` | 逐井 sweep |
| `figures/frequency_band_response.png` | 共识子波频谱、左右半峰值和三条实际零相位 Butterworth 响应 |
| `frequency_band_qc/traces/*.csv` | 单井三频带数值 |
| `frequency_band_qc/figures/*.png` | 单井三频带 QC |

## 验收

1. `run_summary.json` 中三个 cutoff 顺序正确。
2. 选中的 GINN cutoff 不在候选边界。
3. 有效样点满足三频带重构恒等式。
4. `lfm_control_points.csv` 全部来自 `observed_well_sample=true`。
5. GINN anchor 元数据的 `anchor_target_band` 为 `lowpass_reference_to_ginn_cutoff`。
6. Enhance 包 schema 为 `enhance_residual_supervision_v2`。
7. 长缺口位置没有频带值，缺口两侧不存在滤波泄漏。
