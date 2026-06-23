# 真实工区低频校准 Only

## 文档地位

本文定义 R0/R1 之后的 **R2-lowfreq-calibration-only** 研究阶段。首版 R2 的
“lowfreq calibration” 只指 `log(AI)` 的零频/常数偏置校正；它不拟合随时间、zone、剖面、井或空间变化的低频趋势。

R2 首版只回答一个窄问题：在不训练主模型、不改 wavelet/gain/phase/shift、不做 adapter 的
前提下，给 R0 的 `pred_log_ai` 加一个按模型全局拟合的常数低频偏置，是否能改善井旁低频
可信度，并且不破坏 R1 已观察到的正演一致性。

本文对应当前 R2 首版实现，但不注册为正式第六步，不宣称生产结果。当前入口为：

- CLI：`scripts/real_field_lowfreq_calibration.py`
- 核心逻辑：`src/ginn_v2/real_field_calibration.py`

稳定生产链仍终止于第五步。R2 输出仍是 research output，在完成后续 R3/R4 之前不得替代前五步
生产链。

## 1. 输入与阶段边界

R2 输入必须显式指定，不搜索 `latest`：

- 六条真实工区剖面的 R0 zero-shot 输出。
- 对应六条剖面的 R1 forward diagnostic 输出。
- 对应的 `real_field_lfm_v1` LFM 输入。

首版只处理两条 R0 主线：

| 角色 | 模型 |
| --- | --- |
| `no_lateral` | `trace1d_tcn_mismatch` |
| `lateral` | `trace1d_tcn_lateral_mixer_mismatch` |

不处理 k5、post-hoc smoothing、physics/tiny physics 或 adapter。若这些候选未来进入真实工区，
必须另建 R2 输入批次，不能混入首版主表。

R2 禁止：

- 训练或微调主模型。
- 训练 sparse-well adapter。
- 按井局部校正、按剖面校正或拟合横向 bias field。
- 追求进一步提高 waveform corr。
- 用 phase/shift/gain 扫描结果改写主预测。
- 把 R2 输出称为生产反演体。

## 2. 校正语义

R2 对每个 `model_role` 只拟合一个全局常数偏置：

```text
pred_log_ai_calibrated = pred_log_ai + bias_model_role
```

因为正演使用：

```text
r[j] = tanh((x[j] - x[j-1]) / 2)
```

所以给整条 `log(AI)` 加常数不会改变 `x[j] - x[j-1]`，理论上也不会改变反射系数和正演地震。
R2 首版因此是 **logAI intercept calibration**，不是 waveform calibration。

要求：

- `bias_model_role` 在 `log(AI)` 域估计，不在 AI 线性域操作。
- bias 来源为井旁 filtered LAS 与 R0 prediction 的 `log(AI)` 低频/均值残差，不用高频残差拟合 bias。
- 只使用 R1 纳入的直井；斜井不参与首版 R2。
- fit eligibility 至少要求 `wellbore_class=vertical`、`status=ok`、`well_ai_status=ok` 且有效样点数达到配置下限。
- filtered LAS waveform weak、`scale <= 0` 或 reference status 异常只作为 `reference_quality_flag` 记录；默认不自动排除 AI bias 拟合。
- 同一口井被多条剖面覆盖时，先按井聚合，再跨井等权聚合；同一井的多剖面结果合成一个 `well_bias`，不按剖面数量加权。
- lateral 与 no-lateral 分开拟合 bias、分开报告；不得把一个模型的 bias 套到另一个模型。

首版 bias 估计方法固定为：

```text
well_bias_i = median_t(filtered_las_log_ai(t) - pred_log_ai(t))
bias_model_role = median_i(well_bias_i)
```

若实现增加低通版本，只能作为同一报告中的诊断字段；主 bias 仍以上述井内 median residual 和跨井 median 为准。

R2 只改变 `log(AI)` 的 DC/intercept。它不改变 `pred_log_ai` 的时间差分、反射系数和中高频形态；若报告 `pred_delta_vs_lfm`，只能说明除零频/常数项外频谱结构不应改变。

首版允许同一批井用于拟合和报告，但必须明确写成 **calibration evidence**，不得称为
blind validation。

## 3. 输出契约

当前实现至少输出：

- `lowfreq_calibration_summary.json`
- `calibration_bias_by_model.csv`
- `calibration_bias_by_well.csv`
- `well_calibration_evidence.csv`
- `calibrated_forward_metrics.csv`
- `calibrated_well_ai_comparison.csv`
- 图件：校正前后井旁 AI、低频残差、正演 residual 对比。

所有输出必须记录：

- source R0/R1 目录。
- source LFM 目录。
- 参与井和剖面。
- 每个模型的 bias、有效样点数和拒绝原因。
- jackknife bias stability，包括 leave-one-well bias、bias std 和单井移除后的最大 bias shift。
- 代码版本或 git hash。

数组输出至少保留：

- `pred_log_ai_original`
- `pred_log_ai_calibrated`
- `bias_applied`
- `valid_mask_model`

校正不得覆盖 R0 原始预测目录。

## 4. 证据报告

R2 首版不设置硬通过线，只生成证据报告。报告必须同时比较校正前后。

井旁 AI 证据：

- lowfreq RMSE 是否改善。
- fullband RMSE 是否改善或至少不明显恶化。
- observable band RMSE/corr 是否不明显变差。
- highfreq/null-space 能量是否不增加。

正演一致性证据：

- 常数 `log(AI)` bias 不改变反射系数，校正后 synthetic 应与校正前 synthetic 数值一致。
- `calibrated_forward_metrics.csv` 主要是 invariance check，不是改善指标。
- 至少报告 `synthetic_max_abs_diff_before_after`、`residual_rms_before`、`residual_rms_after`、`residual_rms_delta` 和 `forward_invariance_status`。
- 若 waveform residual 明显变化，应优先判为实现错误、mask/NaN 边界问题或校正不再是纯常数，而不是 R2 成功。

模型对比证据：

- lateral 与 no-lateral 分开报告。
- 报告校正后两条模型的差异是否主要只剩低频 bias。
- 若 lowfreq RMSE 改善但 fullband 仍差，说明 R2 有用但仍缺中频/形态校正，应谨慎讨论 R3。
- 若 lowfreq RMSE 不改善或 jackknife bias 不稳定，应先回到 LFM/R1 井旁诊断。
- 若某些井改善、某些井恶化，不进入 global adapter，先做空间簇或地质分组分析。

## 5. 未来测试

未来实现必须覆盖：

- 输入路径显式，缺失则失败，不搜索 `latest`。
- 只消费 R1 中 `wellbore_class=vertical` 且 `status=ok` 的井旁记录。
- 同一井多剖面覆盖时先井内聚合，再跨井等权聚合。
- 主 bias 使用井内 median residual 和跨井 median，不能 pooled-sample mean。
- bias 估计使用 `log(AI)` 域，不在 AI 线性域操作。
- 校正只加常数；除 DC/intercept 外，不改变时间差分、反射系数或中高频形态。
- 校正前后 R1 forward operator 完全一致，且 forward invariance check 通过。
- `calibration_bias_by_well.csv` 和 jackknife bias stability 字段可追溯。
- `scale <= 0`、缺井、低样点、filtered LAS 弱参考等状态必须显式记录。
- R2 报告不得写成 blind validation 或 production candidate。
- `compileall` 通过；完整 pytest 仍由用户运行。

## 6. Assumptions

- R2 首版只基于当前六条剖面的直井证据。
- 当前 `filtered LAS` 是井旁校准参考，不是真值；报告必须保留这个限定。
- R2 使用全部参与井拟合和评价，因此结论只能是 calibration evidence。
- 若 R2 证据显示常数低频偏置不足，下一步不是扩大 R2 表达力，而是讨论 R3 adapter 或回到
  synthetic 消融。
