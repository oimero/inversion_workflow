# 模型消融实验 · 冻结报告

本目录冻结 `synthoseis-lite` 模型消融阶段的可信报告。当前最新代码提交：

```text
60aeb3a310cee090090041084344da72a218edd3
```

## 当前结论

`trace1d_tcn_lateral_mixer_mismatch` 是当前 synthetic gate 综合主候选。

它在强 1D TCN 时间编码基础上加入浅 `3x1` 横向 mixer。三 seed 下，相比纯 `trace1d_tcn_mismatch`，它稳定改善 base、mismatch、probe、probe+mismatch、几何横向梯度和 realization 拼接指标。代价是 0x false energy 小幅上升，后续真实工区和下一版 synthetic gate 需要继续观察。

20260619 smoothing/width 追加复核后，结论收敛为：

- post-hoc Gaussian smoothing 不是 learned lateral mixer 的等价替代；它能改善 probe NRMSE，但会损伤 base/mismatch 的全频阻抗 RMSE 和 geometry/stitching 指标。
- `k1` mixer 明显不如 `k3/k5`，说明收益不是单纯的时间编码后通道重映射。
- `k5` 在 subset 训练的小消融中表现稳定，值得下一轮 full-training / 多 seed 复核；但它目前仍是 subset-trained 证据，不能直接替代 full-training `k3` 三 seed 主候选。

`trace1d_tcn_mismatch + tiny physics` (`lambda_physics=0.001/0.01`) 基本中性：没有明显破坏，也没有可见收益，不能宣称为有效机制。

## Final Audit 结构

```text
note/summary/final_audit/
  README.md
  20260618_baseline_gate/
    README.md
    summaries/
    reports/
  20260619_lateral_mixer_gate/
    README.md
    summaries/
    reports/
    training_runs/
  20260619_smoothing_width_gate/
    README.md
    summaries/
    reports/
    training_runs/
```

## 批次说明

| 批次 | 代码哈希 | 作用 | 当前状态 |
| --- | --- | --- | --- |
| `20260618_baseline_gate` | `bcaf573c6a0352e1426fafa5b81cdb8e20863d0e` | 冻结第一轮完整模型消融，确立 `trace1d_tcn_mismatch` 为强 1D 基线 | 被 20260619 lateral mixer 追加消融部分超越 |
| `20260619_lateral_mixer_gate` | `79e270e06eb797ab9e62227aa95217c2780543fb` | 冻结 tiny physics 与 lateral mixer 追加消融 | 当前主结论 |
| `20260619_smoothing_width_gate` | `60aeb3a310cee090090041084344da72a218edd3` | 冻结 post-hoc smoothing、mixer width 和 k5 subset 多 seed 复核 | 当前推荐入口 |

## 当前候选排序

| 排名 | 模型 | 状态 |
| ---: | --- | --- |
| 1 | `trace1d_tcn_lateral_mixer_mismatch` | 综合主候选 |
| 2 | `trace1d_tcn_lateral_mixer_k5_subset` | 下一轮 full-training 候选，不替代当前主候选 |
| 3 | `trace1d_tcn_mismatch` | 强 1D 对照基线 |
| 4 | `trace1d_tcn` | clean probe 参考 |
| - | `trace1d_tcn_mismatch + tiny physics` | coverage 实验，不作为提升候选 |

## 模型说明

重点模型 README：

- `note/summary/models/trace1d_tcn_lateral_mixer_mismatch/README.md`
- `note/summary/models/trace1d_tcn_mismatch/README.md`
- `note/summary/models/trace1d_tcn_mismatch_tiny_physics/README.md`

旧模型 README 仍保留在 `note/summary/models/`，用于追溯 20260618 baseline gate。

## 数据可靠性声明

`sample_kind` 过滤修复前的 mismatch-training base 指标全部废弃。20260618 baseline gate 只引用 `20260618_baseline_gate` 下的冻结副本；20260619 lateral mixer gate 只引用 `20260619_lateral_mixer_gate` 下的冻结副本；post-hoc smoothing 与 mixer width 复核只引用 `20260619_smoothing_width_gate` 下的冻结副本。

合成基准冻结位置：

```text
note/summary/source_data/synthoseis_lite_generate_20260617_202613
```



