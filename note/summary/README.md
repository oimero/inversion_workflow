# 模型消融与真实工区 R0/R1/R2 · 冻结报告

本目录冻结 `synthoseis-lite` 模型消融阶段，以及真实工区 R0/R1/R2 研究诊断阶段的可信报告。

当前 HEAD：

```text
0c216564d9a19e076665e133be791c40b7e31b4e
```

## 当前结论

Synthetic gate 的主候选仍是 `trace1d_tcn_lateral_mixer_mismatch`：强 1D TCN 时间主干加浅横向 mixer，在多 seed 消融中整体优于纯 1D；`k5` 仍是下一轮候选，不替代当前 full-training `k3` 主候选。

真实工区已完成六剖面 R0/R1/R2 诊断：zero-shot 正演一致性稳定，但井旁阻抗可信度尚未被钉牢。R2 证明全局常数 `log(AI)` bias 不能解决井旁 AI 偏差。2026-06-26 追加的旧 W0/W1 线性 sparse-well 支线已冻结为 rejected diagnostic：井侧线性改善存在，但全场应用主要表现为 LFM shrinkage 和 synthetic energy 塌缩。下一步不沿用 W0/W1 命名，改从 R2/R3 重新规划 real-delta adapter 验证。

## 批次说明

| 批次 | 代码哈希 | 作用 | 当前状态 |
| --- | --- | --- | --- |
| `20260618_baseline_gate` | `bcaf573c6a0352e1426fafa5b81cdb8e20863d0e` | 冻结第一轮完整模型消融，确立 `trace1d_tcn_mismatch` 为强 1D 基线 | 被 20260619 lateral mixer 追加消融部分超越 |
| `20260619_lateral_mixer_gate` | `79e270e06eb797ab9e62227aa95217c2780543fb` | 冻结 tiny physics 与 lateral mixer 追加消融 | 支撑当前 synthetic 主候选 |
| `20260619_smoothing_width_gate` | `60aeb3a310cee090090041084344da72a218edd3` | 冻结 post-hoc smoothing、mixer width 和 k5 subset 多 seed 复核 | 当前 synthetic 推荐入口 |
| `20260622_real_field_gate` | `1dd88ac969793a16cd94c0525e10e76675e82a18` | 冻结新 LFM、R0 zero-shot 和 R1 forward diagnostic | 当前真实工区 R0/R1 推荐入口 |
| `20260623_six_section_r2_gate` | `0c216564d9a19e076665e133be791c40b7e31b4e` | 冻结六剖面 R0/R1 和 R2 全局常数 bias 诊断 | 当前真实工区推荐入口 |
| `20260626_w0_w1_rejected_diagnostic` | `2e165832c2152701cd28c42038986f49572c03ed` | 冻结旧 W0/W1 线性 sparse-well 支线：W0 井侧正、W1 全场拒绝 | 废弃支线，仅作反例证据 |

## 真实工区冻结位置

```text
note/summary/final_audit/20260623_six_section_r2_gate/
  README.md
  configs/real_field_six_section_configs_20260623/
  r0/
  r1/
  r2/real_field_lowfreq_calibration_current/
```

旧 W0/W1 rejected diagnostic 位置：

```text
note/summary/final_audit/20260626_w0_w1_rejected_diagnostic/
```

关键指标来自：

```text
note/summary/final_audit/20260623_six_section_r2_gate/r2/real_field_lowfreq_calibration_current/calibration_bias_by_model.csv
note/summary/final_audit/20260623_six_section_r2_gate/r2/real_field_lowfreq_calibration_current/calibrated_forward_metrics.csv
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

真实工区 R0/R1/R2 当前只引用 `20260623_six_section_r2_gate` 下的冻结副本。2026-06-26 的旧 W0/W1 支线只引用 `20260626_w0_w1_rejected_diagnostic` 的结论页，不升级为主结论。更早的 `real_field_*` 输出，尤其是旧 LFM、NaN 正演污染、图件纵轴修复前或单剖面阶段的结果，不再作为当前主结论。

合成基准冻结位置：

```text
note/summary/source_data/synthoseis_lite_generate_20260617_202613
```
