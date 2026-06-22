# 模型消融与真实工区 R0/R1 · 冻结报告

本目录冻结 `synthoseis-lite` 模型消融阶段，以及真实工区 R0/R1 研究诊断阶段的可信报告。

当前 HEAD：

```text
1dd88ac969793a16cd94c0525e10e76675e82a18
```

## 当前结论

Synthetic gate 的主候选仍是 `trace1d_tcn_lateral_mixer_mismatch`：强 1D TCN 时间主干加浅横向 mixer，在多 seed 消融中整体优于纯 1D；`k5` 仍是下一轮候选，不替代当前 full-training `k3` 主候选。Post-hoc smoothing 不能替代 learned lateral mixer，tiny physics 暂未证明有效。

真实工区 R0/R1 已完成一版可复现诊断：使用 `real_field_lfm_v1`、`p99_abs_matched` seismic 输入变换和第五步子波 active half-support 的 valid-run erosion。修复 NaN 正演污染和图件纵轴后，R1 显示 zero-shot synthetic 与真实地震具备较强正演一致性；但这仍是 research output，不是生产反演结果，也不是穿井验证结论。

## 批次说明

| 批次 | 代码哈希 | 作用 | 当前状态 |
| --- | --- | --- | --- |
| `20260618_baseline_gate` | `bcaf573c6a0352e1426fafa5b81cdb8e20863d0e` | 冻结第一轮完整模型消融，确立 `trace1d_tcn_mismatch` 为强 1D 基线 | 被 20260619 lateral mixer 追加消融部分超越 |
| `20260619_lateral_mixer_gate` | `79e270e06eb797ab9e62227aa95217c2780543fb` | 冻结 tiny physics 与 lateral mixer 追加消融 | 支撑当前 synthetic 主候选 |
| `20260619_smoothing_width_gate` | `60aeb3a310cee090090041084344da72a218edd3` | 冻结 post-hoc smoothing、mixer width 和 k5 subset 多 seed 复核 | 当前 synthetic 推荐入口 |
| `20260622_real_field_gate` | `1dd88ac969793a16cd94c0525e10e76675e82a18` | 冻结新 LFM、R0 zero-shot 和 R1 forward diagnostic | 当前真实工区 R0/R1 推荐入口 |

## 真实工区冻结位置

```text
note/summary/final_audit/20260622_real_field_gate/
  README.md
  lfm/real_field_lfm_current_v4/
  r0/real_field_zero_shot_20260622_103213/
  r1/real_field_forward_diagnostic_20260622_103253/
  training_runs/ginn_v2_train_trace1d_tcn_mismatch_r0_s20260619/
```

关键指标来自：

```text
note/summary/final_audit/20260622_real_field_gate/r1/real_field_forward_diagnostic_20260622_103253/forward_diagnostic_metrics.csv
```

## 当前候选排序

| 排名 | 模型 | 状态 |
| ---: | --- | --- |
| 1 | `trace1d_tcn_lateral_mixer_mismatch` | Synthetic gate 综合主候选 |
| 2 | `trace1d_tcn_lateral_mixer_k5_subset` | 下一轮 full-training 候选 |
| 3 | `trace1d_tcn_mismatch` | 强 1D 对照基线 |
| 4 | `trace1d_tcn` | clean probe 参考 |
| - | `trace1d_tcn_mismatch + tiny physics` | coverage 实验，不作为提升候选 |

## 数据可靠性声明

`sample_kind` 过滤修复前的 mismatch-training base 指标全部废弃。20260618 baseline gate 只引用 `20260618_baseline_gate` 下的冻结副本；20260619 lateral mixer gate 只引用 `20260619_lateral_mixer_gate` 下的冻结副本；post-hoc smoothing 与 mixer width 复核只引用 `20260619_smoothing_width_gate` 下的冻结副本。

真实工区 R0/R1 只引用 `20260622_real_field_gate` 下的冻结副本。更早的 `real_field_*` 输出，尤其是旧 LFM、NaN 正演污染或图件纵轴修复前的结果，不再用于结论。

合成基准冻结位置：

```text
note/summary/source_data/synthoseis_lite_generate_20260617_202613
```
