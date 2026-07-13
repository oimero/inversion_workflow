# 模型消融与真实工区 R0/R1/R2 · 冻结报告

本目录冻结 `synthoseis-lite` 模型消融阶段，以及真实工区 R0/R1/R2 研究诊断阶段的可信报告。

## 工区标签

为避免把不同采样域的结果混在一起，本文使用匿名工区标签：

- `T1（时间域工区）`：synthoseis-lite、时间域 R0/R1/R2、W0/W1 和 PH5 诊断。
- `D1（深度域工区）`：四模型 GINN v2 消融和 LFM 反事实分解；深度采样轴、模型和结论均独立于 T1。

---

## T1 时间域

### 当前结论

- Synthetic 主候选是 full-training k3 lateral mixer，纯 1D TCN 保留为对照。
- 完整体 zero-shot 正演相关性约 `0.925`，相对 LFM-only 消除约 `84%` 的残差能量，但井旁 AI 改善不稳定。
- 0625 完整体反事实审计确认两个模型因果依赖 seismic：置零或 shuffle 后正演增益消失，极性反转后相关性约为 `-0.95`；prior-shortcut 假设被拒绝。
- PH5 from-scratch real-delta anchor 是 positive proof-of-signal，但仍有 delta 幅度偏低和井旁正演下降风险。

### 批次说明

| 批次 | 代码哈希 | 作用 | 当前状态 |
| --- | --- | --- | --- |
| `20260618_baseline_gate` | `bcaf573c6a0352e1426fafa5b81cdb8e20863d0e` | 冻结第一轮完整模型消融，确立 `trace1d_tcn_mismatch` 为强 1D 基线 | 被 20260619 lateral mixer 追加消融部分超越 |
| `20260619_lateral_mixer_gate` | `79e270e06eb797ab9e62227aa95217c2780543fb` | 冻结 tiny physics 与 lateral mixer 追加消融 | 支撑当前 synthetic 主候选 |
| `20260619_smoothing_width_gate` | `60aeb3a310cee090090041084344da72a218edd3` | 冻结 post-hoc smoothing、mixer width 和 k5 subset 多 seed 复核 | 当前 synthetic 推荐入口 |
| `20260622_real_field_gate` | `1dd88ac969793a16cd94c0525e10e76675e82a18` | 冻结新 LFM、R0 zero-shot 和 R1 forward diagnostic | 当前真实工区 R0/R1 推荐入口 |
| `20260623_six_section_r2_gate` | `0c216564d9a19e076665e133be791c40b7e31b4e` | 冻结六剖面 R0/R1 和 R2 全局常数 bias 诊断 | 当前真实工区推荐入口 |
| `20260625_volume_r0_r1_gate` | `471f6f9` / `210ad92` | 冻结完整体 R0 zero-shot 与 R1 forward diagnostic | 0626 W0/W1 的权威体模式对照 |
| `20260626_w0_w1_rejected_diagnostic` | `2e165832c2152701cd28c42038986f49572c03ed` | 冻结旧 W0/W1 线性 sparse-well 支线：W0 井侧正、W1 全场拒绝 | 废弃支线，仅作反例证据 |
| `20260627_r2_final_head_rejected_diagnostic` | `572a124a6f88f202c8654ea586db01f66e916f4d` | 冻结 final-head real-delta adapter：标量改善来自 delta collapse | 废弃支线，禁止进入 full-field application |
| `20260628_real_delta_anchor_ph5_positive_signal` | `c7d751afe8db3d408afc304eede663e308aba9ab` | 完整冻结 PH5 held-out real-delta anchor 输出、checkpoint 与图件 | Positive proof-of-signal；继续 GINN-v2 real-delta 研究 |
| `20260711_legacy_0625_counterfactual_audit` | `76726f6768d94864bf9a83c5be7cd64777fde863` + code snapshot | 冻结 0625 两模型完整体 seismic 反事实审计 | 确认 seismic 因果依赖，拒绝 prior-shortcut 假设 |

### 当前候选排序

| 排名 | 模型 | 状态 |
| ---: | --- | --- |
| 1 | `trace1d_tcn_lateral_mixer_mismatch` | Synthetic gate 综合主候选 |
| 2 | `trace_1d_dilated_tcn_mismatch` | 强 1D 对照基线 |

---

## D1 深度域

### 当前结论

- 四模型合成消融显示：nominal 与 mismatch training 的测试指标基本相同，k=3 横向 mixer 没有增益；mismatch 后追加 synthetic physics 反而使测试 RMSE 从约 `0.0706` 上升到 `0.0794`，相关系数从约 `0.8009` 降到 `0.7614`。因此当前 D1 R0 使用 `depth_tcn_mismatch_s20260712` 作为 trace 候选，physics 结果作为消融反例保留。
- Trend / proportional LFM 四格反事实分解表明，proportional kriging 的最终收益主要来自 LFM 直接背景替换：`Y_TT → Y_PT` 已带来平均井相关系数 `+0.1350`，而网络条件输入效应 `Y_PT → Y_PP` 只再带来 `+0.0085`。这说明当前 GINN v2 更接近"给定 LFM 的地震残差修正器"，而不是独立恢复绝对低频阻抗的模型。该结果仍受同井 LFM 约束影响，不能替代 blind-well 或 R1 正演验证。

### 批次说明

| 批次 | 代码哈希 | 作用 | 当前状态 |
| --- | --- | --- | --- |
| `20260713_D1_ginn_v2_lfm_audit` | `partial` | 冻结 D1 四模型 GINN v2 消融与 LFM 反事实分解 | 当前 D1 推荐入口 |

### 当前候选排序

| 排名 | 模型 | 状态 |
| ---: | --- | --- |
| 1 | `depth_tcn_mismatch_s20260712` | D1 R0 trace 候选 |

### 完整冻结目录

见 `note/summary/final_audit/20260713_D1_ginn_v2_lfm_audit/`。
