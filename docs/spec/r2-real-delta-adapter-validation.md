# R2 Real-Delta Adapter Validation

## 文档地位

R2 是 R1 之后的研究验证步骤：它冻结现有真实工区 LFM、R0 输入、R0 backbone 和合成
基准，只使用真实井上的 `delta` 标签重估 R0 的最终输出头，检验这种低容量监督能否迁移到
未参与拟合的空间簇。

```text
R1 real_field_forward_diagnostic
  -> R2 real-delta adapter validation
  -> R3 full-field adapter application（仅限 R2 通过的角色）
```

R2 不生成完整工区阻抗体，不重新训练 backbone，也不重新解释旧 W0/W1 的线性校正结果。
旧 W0/W1 已作为 rejected diagnostic 冻结；R2 不是
`a * R0 + b * LFM + c` 的延续。

规划入口和配置位置为：

```text
scripts/r2_real_delta_adapter.py
experiments/common/common.yaml::r2_real_delta_adapter
```

## 1. 研究问题与结论边界

R2 只回答：

```text
在冻结的生产 LFM 和 R0 上下文中，
用训练井的真实 well_delta 重估低容量 delta 输出头，
能否稳定提高未见空间簇的 delta corr，
同时不损害 full-AI 指标，并报告其与已有合成域能力的兼容性？
```

监督语义固定为：

```text
well_delta   = target_log_ai - lfm_log_ai
r0_delta     = r0_pred_log_ai - lfm_log_ai
pred_ai_r2   = lfm_log_ai + pred_delta_r2
```

这里的 held-out 只表示该井或空间簇的真实 `well_delta` 标签不参与 adapter 拟合。真实 LFM、
R0 输入和合成基准的统计构建已经使用过当前工区井信息，因此 R2 是
`conditional adapter-label transfer test`，不是端到端 blind-well validation。报告和 summary
不得省略这一限制。

R2 的正结果不代表完整工区校正已经成立。完整体、真实工区正演能量和井旁 halo 只能在 R3
生成候选体后重新运行 R1 才能判断。

## 2. 冻结输入与来源校验

### 2.1 显式来源

配置必须显式给出：

```yaml
r2_real_delta_adapter:
  forward_diagnostic_dir: scripts/output/<frozen-r1-run>
  zero_shot_dir: scripts/output/<frozen-r0-run>
  model_roles: [lateral, no_lateral]
  device: cuda
  lambda_candidates: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
  feature_reconstruction_tolerance_log_ai: 1.0e-5
  synthetic_scopes: [validation_base, validation_mismatch, test_base]
  thresholds:
    minimum_loco_clusters_for_decision: 5
    minimum_loco_cluster_improvement_fraction: 0.70
    minimum_median_delta_corr_gain: 0.0
    maximum_synthetic_error_relative_increase: 0.05
    maximum_synthetic_corr_decrease: 0.02
```

允许 CLI 覆盖单次运行来源和输出目录，但默认参数必须来自 `common.yaml`，使脚本可在
`scripts/` 下直接执行：

```text
python r2_real_delta_adapter.py
```

禁止搜索 `latest`、扫描目录猜测来源、缺失时切换到其他 checkpoint，或重新读取原始 ZGY
来替代冻结 R0 输入。R2 必须验证：

- R1 summary 记录的 R0 来源与显式 `zero_shot_dir` 是同一次运行。
- R1、R0 summary、每个模型 checkpoint、normalization、manifest、预测 NPZ 和合成基准文件
  的 SHA-256 与各自 provenance 一致。
- 两个角色的 R0 输入坐标轴、LFM、seismic 和有效掩码相同；角色 checkpoint 可以不同。
- R0 必须是 volume 模式，且角色只能是显式配置的 `lateral` 和 `no_lateral`。

任一校验失败均终止整次运行，不生成部分成功 summary。

### 2.2 R1 逐样点标签契约

R2 读取 R1 的 `well_ai_samples.csv`，只使用 `valid_for_fit == true` 的行。该表在实施 R2
前必须补入 `docs/concepts/csv-contracts.md`，至少冻结以下字段语义：

| 字段 | 语义 |
|------|------|
| `well_name` / `sample_index` / `twt_s` | 井名、井内样点序号和正秒 TWT |
| `inline` / `xline` / `x_m` / `y_m` | R1 用于三维体采样的轨迹坐标 |
| `spatial_cluster_id` / `spatial_cluster_size` | 600 m 空间簇标签和簇井数 |
| `target_log_ai` / `lfm_log_ai` | full-AI 模型网格井监督和冻结 LFM，单位均为 logAI |
| `model_role` / `r0_pred_log_ai` | R0 角色和 R1 同口径采样的预测 logAI |
| `r0_valid_mask` / `r0_blend_weight` | R0 有效性和 patch 拼接覆盖证据 |
| `well_twt_support_valid` | 井曲线在该 TWT 样点的支撑状态 |
| `valid_for_fit` / `valid_reason` | R2 是否可使用该行及拒绝原因 |
| `sampling_mode` / `sample_method` / `wellbore_class` | volume 采样、轨迹插值方法和井型 |

不得根据空值或字段名重新猜测 `valid_for_fit`。同一
`(model_role, well_name, sample_index)` 必须唯一；两个角色的井、样点、LFM 和井标签必须
逐行一致。

当前冻结数据有 11 口有效井、7 个空间簇，其中 PH6、PH7、PH8 为斜井。实现不得写死这些
名称或数量，但首版必须纳入所有有效直井和斜井，并按 `sample_method` 复用 R1 的轨迹坐标。

## 3. 模型边界与隐藏特征提取

### 3.1 唯一可训练参数

两个 R0 架构均冻结至倒数特征层，只允许重估原有最终 head：

- `no_lateral`：最后一个 `kernel_size=1` 的 `Conv1d`。
- `lateral`：`Trace1DTCNShallowLateralMixer.output` 的 `kernel_size=1` `Conv2d`。

当前两者都是 32 输入通道、1 输出通道，即 32 个权重和 1 个 bias。实现需要提供显式的
`forward_features` 与 `forward_head` 接口，并保持现有 checkpoint 的 state-dict key 不变。
所有 backbone 参数必须 `requires_grad=false`；R2 不新增 adapter 网络，也不允许解冻其他层。

对 `lateral` 角色，`forward_features` 必须返回 shallow lateral mixer 之后、1x1 `output`
head 之前的 feature map。不得只计算井所在 trace 的逐道 TCN feature，也不得用
`no_lateral` 的 feature 抽取路径代替 lateral mixer 输出。

head 输入是冻结模型已经由 seismic、LFM 和 valid mask 编码出的隐藏特征。因此 R2 学到的
仍是 feature-conditioned delta mapping，而不是对 R0/LFM 输出做全局重新加权。

### 3.2 查询式特征重建

R0 的 volume 为 `[inline, xline, twt]`，完整 32 通道隐藏特征体约 25 GB。R2 不保存或长期
构造该体，而采用查询式重建：

1. 从 R0 `predictions.npz` 读取冻结的 `seismic_input`、`lfm_input`、`valid_mask_model`、
   坐标轴和 `stitching_weight`。
2. 按 R0 manifest、normalization、patch spec、prediction index 和 stitch strategy 重放
   原 patch 推理，截取最终 head 之前的隐藏特征。
3. 由每个井样点的浮点 inline/xline/TWT 计算 R1 三维插值所需网格节点。
4. 只在这些节点累计重叠 patch 的隐藏特征和拼接权重，再按 R0 规则归一化。
5. 对拼接后的节点特征执行与 R1 相同的横向双线性加 TWT 线性插值，得到逐井样点特征。

最终 head、delta 反标准化、patch 拼接和上述插值都是仿射或线性运算，因此原 R0 head 作用于
重建特征后，应复现：

```text
r0_pred_log_ai - lfm_log_ai
```

每个有效样点的绝对误差必须不超过 `1e-5 logAI`。任何角色、井或样点超差都标记
`r0_feature_reconstruction_mismatch` 并拒绝整次运行，不能放宽容差继续拟合。

重建审计必须按角色、井、空间簇和全局分别输出：

```text
feature_reconstruction_max_abs_error
feature_reconstruction_p99_abs_error
feature_reconstruction_median_abs_error
feature_reconstruction_n_samples
```

R2 拟合和 held-out 指标只能使用通过 reconstruction check 的同一批样点。R1 CSV 中的
`r0_pred_log_ai` 只作为重建审计目标；正式 R0/R2 配对指标必须分别由原 head 和 R2 head
作用于同一批重建 feature 得到，不能混用 CSV R0 预测与 feature-based R2 预测。

特征缓存使用带 manifest 和 SHA-256 的 NPZ，加一张只保存索引、坐标、标签、角色和有效性的
CSV；不把 32 个无单位隐藏通道写入跨步骤 CSV。

## 4. 锚定 head 回归

### 4.1 目标与权重

拟合在 R0 checkpoint 的 normalized-delta 空间进行。`well_delta` 必须使用该角色 checkpoint
中原有的 delta mean/std 标准化，不得根据真实井重新估计 target normalization。

设训练集中有 `C` 个空间簇，簇 `c` 有 `W_c` 口训练井，井 `w` 有 `N_cw` 个有效样点，
样点权重固定为：

```text
q_cwn = 1 / (C * W_c * N_cw)
```

因此每个空间簇等权、簇内每口井等权、井内每个样点等权，全部权重和为 1。LOO 留出一口井
后，要在剩余训练井上重新计算该三级权重，不能沿用全数据权重。

### 4.2 训练折标准化与解析解

隐藏特征只使用当前训练折的上述权重计算均值和标准差。验证或测试特征不得参与标准化。
加权标准差为零或非有限的通道保持原 R0 head 权重，并从可拟合变量中移除；该处理及通道
列表必须进入 fold manifest。若所有通道均退化，则该 fold 状态为
`degenerate_training_features`。

将活跃特征标准化后，把原 R0 head 变换到同一坐标系，记参数为 `beta_0`。退化通道不进入
`beta`，但其原 R0 权重贡献始终包含在预测中。对每个候选 `lambda` 求唯一的锚定岭回归解：

```text
pred_i(beta) = z_i @ beta + sum_{j in D}(w0_j * h_ij)

beta(lambda) = argmin_beta [
    sum_i q_i * (pred_i(beta) - well_delta_normalized_i)^2
    + lambda * ||beta - beta_0||_2^2
]
```

这里 `z_i` 已增广常数 1，故 `beta` 包含活跃通道系数和 bias。

bias 包含在 `beta` 中并同样锚定。求解使用双精度线性代数；系统非有限或不可解时记录
`ridge_solve_failed`，不得切换 SGD、伪造 epsilon 或使用普通最小二乘兜底。

拟合参数必须转换回原始隐藏特征坐标后，才能保存为可替换 checkpoint 最终 Conv head 的
state dict。设活跃通道集合为 `A`，退化通道集合为 `D`，训练折标准化为：

```text
z_j = (h_j - mu_j) / sigma_j,  j in A
```

原 head 在 normalized-delta 空间的参数为 `w0_j, b0`。退化通道始终保留
`w_j = w0_j`，其原始特征贡献不并入活跃通道的标准化。原 head 的锚点转换为：

```text
beta0_j    = w0_j * sigma_j,                  j in A
beta0_bias = b0 + sum_{j in A}(w0_j * mu_j)
```

岭回归得到 `beta_j, beta_bias` 后，写回原特征坐标：

```text
w_j = beta_j / sigma_j,                       j in A
w_j = w0_j,                                   j in D
b   = beta_bias - sum_{j in A}(beta_j * mu_j / sigma_j)
```

转换后的 `w, b` 必须再次在训练与 held-out feature 上和标准化坐标预测逐样点比对。若原
checkpoint 的 delta 标准差为 `delta_std`，normalized-delta 容差固定换算为：

```text
feature_reconstruction_tolerance_normalized_delta =
    feature_reconstruction_tolerance_log_ai / delta_std
```

fold head 包保留 `mu/sigma/beta` 只作审计；运行时和 R3 不得依赖 fold-specific
feature-normalization wrapper。

训练目标只有加权 MSE。corr、梯度、频带指标和 synthetic replay 均不进入训练目标；这些
只用于 held-out 评价和保持性检查。

## 5. 嵌套拆分与 lambda 选择

### 5.1 主验证：外层 LOCO

设输入中通过 R1 有效性和 reconstruction check 的唯一空间簇数为 `K`。主验证对每个角色
独立执行 `K` 个 leave-one-cluster-out 外层 fold。当前冻结工区 `K=7`；实现不得写死该值。
held-out 簇内可以有一口或多口井，其全部标签必须同时隔离。每个外层 fold：

1. 只在外层训练簇上，对固定七档 lambda 运行 inner-LOCO。
2. 每个 inner fold 重算特征标准化和三级样点权重。
3. 对每个 lambda 先按井计算 held-out 指标，再在 held-out 簇内对井取中位数，最后对 inner
   簇等权汇总。
4. 合格 lambda 必须满足：inner median delta-corr 增益大于 0、inner median full-AI corr
   增益不小于 0、inner median full-AI RMSE 增量不大于 0，且所有 inner fold 有效。
5. 在合格候选中最大化 inner median delta-corr 增益；完全并列时选择更大的 lambda。
6. 用选定 lambda 在全部外层训练簇上重拟合一次，再且仅再评估 held-out 外层簇。

lambda 选择必须区分科学否定与不可判：

- 所有 inner fold 均有效，但没有候选同时满足 delta 改善和 full-AI 守门时，记录
  `no_eligible_lambda_due_to_guardrail`。若所有候选均无 delta 改善，归入
  `no_transfer_signal`；若存在 delta 改善但均被 full-AI 守门拒绝，归入
  `full_ai_guardrail_failed`。
- 任一 inner fold 因样点、特征、corr 或求解状态无效而无法完成候选比较时，记录
  `lambda_selection_invalid_inner_fold`，角色结论归入 `inconclusive_invalid_fold`。

两种情况都不得用 R0、最优失败候选或全数据 lambda 代替。任一外层 fold 无有效选定 head
时，该角色不能得到 R2 positive 判定，但 summary 必须保留上述科学失败或不可判的区别。

### 5.2 辅助验证：LOO

leave-one-well-out 使用同样的嵌套选择过程，但只用于：

- 定位异常井。
- 比较 LOO 与 LOCO 的乐观程度。
- 检查同一密井簇内近井标签是否造成表观迁移。
- 分别汇总 vertical 与 deviated 井表现。

LOO 不参与 R2 pass/fail，也不得与 LOCO fold 混合后计算一个总体比例。

### 5.3 all-well head

完成无偏的外层 LOCO 后，允许使用全部 `K` 个簇做固定 lambda 的完整 LOCO 比较，按相同规则
选择 all-well lambda，再用全部有效井拟合每个角色的 `all_wells_head.pt`。该 head 只为 R3
准备，不是 R2 验证证据。

如果全数据仍无合格 lambda，必须输出一个审计 head 包，其中保存原 R0 head，并明确记录：

```text
adapted = false
selection_status = no_eligible_lambda_due_to_guardrail | lambda_selection_invalid_inner_fold
eligible_for_r3 = false
validation_evidence = not_unbiased
```

不得随意选择一个失败 lambda 来满足产物存在性要求。

## 6. Held-out 指标与 synthetic preservation 诊断

### 6.1 井侧指标

每个 held-out 井至少报告 R0 与 R2 的：

- `delta_corr`、`delta_rmse`、`delta_bias` 和 `delta_rms`。
- `full_ai_corr`、`full_ai_rmse` 和 `full_ai_bias`。
- 上述 R2 减 R0 的配对增量。
- `r2_delta_rms / r0_delta_rms` 及相对真实 `well_delta_rms` 的幅度比。

corr 有效样点少于 8、目标或预测方差为零时状态必须显式失败。LOCO 汇总先在同一 held-out
簇内对井取中位数，再对 `K` 个簇等权；不能按样点数或井数直接池化。

head 只按全频拟合。沿用 R1 的频带定义计算 delta 和 full-AI 分频指标，用于判断改善来自
低频、可观测频带还是高频/零空间频带，但分频结果不参与 R2 pass/fail。
`r2_delta/r0_delta` 能量比也只报告；完整工区能量守门留给 R3 后的 R1。

### 6.2 合成能力保持

每个选定的 fold head 必须在对应角色的冻结 synthetic 数据上重新推理，并用原 R0 head 在
同一代码和同一 patch 集上形成配对基线。当前最小要求为：

```text
validation_base
validation_mismatch
test_base
```

synthetic preservation 必须直接安装第 4.2 节已经转换回原始隐藏特征坐标的 Conv head state
dict。不得在 synthetic feature 上重新估计 `mu/sigma`，也不得使用训练折标准化 wrapper；
该检查必须评价 R3 实际可加载的 head 行为。

对于当前报告中存在且状态为 `ok` 的 error 类指标，包括 RMSE、NRMSE、geometry RMSE 和
realization RMSE：

```text
r2_metric <= 1.05 * r0_metric
```

对于 corr 类指标：

```text
r2_corr >= r0_corr - 0.02
```

缺少必需 scope、无法形成配对基线或任一现有指标越界，都将
`synthetic_preservation_status` 标为 warning。它不否决真实井 LOCO 正结果，只描述 final
head 与合成域训练语义的兼容程度。当前冻结 synthetic 诊断没有 probe、paired increment 和
zero-x false-energy 的完整证据；summary 必须记录
`synthetic_probe_evidence_gap`，但 R2 不在本步骤补建这些产物，也不因历史缺口单独失败。

## 7. 正式判定与角色晋级

每个角色独立判定。令：

```text
K = 有效外层 LOCO 空间簇数
minimum_improved_clusters = ceil(
    minimum_loco_cluster_improvement_fraction * K
)
```

若 `K < minimum_loco_clusters_for_decision`，角色状态为
`inconclusive_insufficient_loco_clusters`，不得判为 positive 或 `no_transfer_signal`。

`r2_positive` 必须同时满足：

1. `K` 不小于 `minimum_loco_clusters_for_decision`，且全部 `K` 个外层 LOCO fold 有效。
2. delta corr 严格优于 R0 的 held-out 簇数不小于 `minimum_improved_clusters`。
3. 簇等权 median delta-corr 增益严格大于 `minimum_median_delta_corr_gain`。
4. 簇等权 median full-AI corr 增益不小于 0。
5. 簇等权 median full-AI RMSE 增量不大于 0。
6. all-well head 成功拟合。

当前冻结工区 `K=7`，默认改善比例为 `0.70`，因此第 2 条实例化为至少
`ceil(0.70 * 7) = 5` 个簇改善。`5/7` 只是当前运行的实例化结果，不是代码常量。

否则角色结论只能是以下之一：

```text
no_transfer_signal
full_ai_guardrail_failed
inconclusive_invalid_fold
inconclusive_insufficient_loco_clusters
```

任一角色通过即可进入 R3，且 R3 只能加载 `eligible_for_r3=true` 的 all-well head。如果两个
角色都通过，按以下顺序选择首个 R3 候选：

1. 更高的 LOCO cluster-equal median delta-corr 增益。
2. 完全并列时，更低的 median full-AI RMSE 增量。

失败角色的 all-well head 仍保留审计，但必须 `eligible_for_r3=false`；R3 加载器必须将其
视为硬错误。

## 8. 输出契约

默认输出目录：

```text
scripts/output/r2_real_delta_adapter_<timestamp>/
```

### `r2_fold_metrics.csv`

每个 `split_type / model_role / outer_fold / held_out_well` 一行，保存 R0、R2、配对增量、
样点数、井型、空间簇和 fold 状态。LOCO 与 LOO 必须由 `split_type` 明确区分。

### `r2_feature_reconstruction.csv`

按角色、井、空间簇和全局保存 reconstruction check 的样点数、max/P99/median absolute
error、容差和状态。该表中的通过样点集合是后续拟合和指标计算的唯一合法样点集合。

### `r2_cluster_metrics.csv`

每个 `split_type / model_role / outer_fold / held_out_spatial_cluster_id` 一行，保存 held-out
井数、簇内井中位 delta-corr 增益、full-AI corr 增益、full-AI RMSE 增量、synthetic
preservation 状态和 cluster status。正式判定只能消费该表，不能从井表临时采用另一套聚合。

### `r2_lambda_selection.csv`

每个 outer fold、inner fold 和 lambda 一行，保存训练/验证井簇、三级权重审计、全部 inner
指标、合格状态、拒绝原因和最终选择标志。all-well lambda 比较使用独立
`selection_scope=all_wells`。

### `r2_head_parameters.csv`

head 参数长表，至少包含角色、split、fold、lambda、参数名、R0 值、R2 值、差值、活跃通道
状态和相对参数漂移。隐藏特征通道只用稳定索引命名，不赋予地质语义。

### `r2_synthetic_preservation.csv`

每个 head、synthetic scope 和指标一行，保存 R0/R2 值、绝对及相对变化、阈值、状态和来源
报告哈希。

### `r2_band_metrics.csv`

逐 held-out 井、角色和 R1 频带保存 delta/full-AI 指标及 R2-R0 增量，仅作诊断。

### `r2_well_qc_index.csv` 与 `figures/wells/`

每口井、每个模型角色生成一张 R1 同款六联图，只使用 all-well head。第一子图严格只有
`target_log_ai` 和 `R2 all-well` 两条曲线；其余面板由 R2 all-well 的 reflectivity、synthetic、
R1 冻结的 observed seismic、residual 和 dynamic correlation 组成。不得叠加 R0、LFM 或
LOCO 曲线。标题固定包含 `all-well/trained-well view; not validation evidence`。

`r2_well_qc_index.csv` 逐图记录状态和路径。all-well head 无有效拟合时不允许用 R0 head
冒充，必须记录 `all_well_qc_unavailable`。LOCO 正式证据只存在于数值表。

### `r2_decision_table.csv`

每个模型角色一行，至少保存 `K`、有效 LOCO fold 数、改善簇数及比例、动态最小改善簇数、
三项 cluster-equal median 增量、synthetic/all-well 状态、decision、decision reason 和
`eligible_for_r3`。该表是人工审计入口，并与 summary JSON 的角色判定逐字段一致。

### `heads/`

保存每个有效外层 fold head、LOO 诊断 head 和每角色 `all_wells_head.pt`。head 包至少包含：

- schema、角色、来源 checkpoint 及 SHA-256。
- 原 head 和新 head state dict。
- lambda、训练井/簇、特征标准化和退化通道。
- split provenance、判定状态、`eligible_for_r3` 和
  `validation_evidence=not_unbiased`（all-well head；fold head 记录对应 held-out scope）。
- R2 summary SHA-256 可在 summary 写完后通过独立 manifest 反向关联，避免循环哈希。

### `r2_real_delta_adapter_summary.json`

schema 固定为 `r2_real_delta_adapter_summary_v2`；旧 filtered-label v1 summary/head 不兼容。
记录输入来源和哈希、冻结边界、样本及空间簇清单、模型接口、拆分规则、lambda 网格、逐角色
门槛结果、synthetic evidence gap、all-well head 路径和唯一的 `recommended_next_state`：

```text
r3_candidate_available
no_role_passed_r2
r2_inconclusive
```

除必需的逐井 QC 外，可选图件包括 LOCO delta/full-AI 配对增益、lambda 稳定性、head 参数
漂移、delta 能量比和 vertical/deviated 子组图。图件不得替代 CSV 判定。

## 9. 失败规则

以下情况不得降级继续：

- 输入列、来源文件、hash、schema、角色或坐标轴不一致。
- R0 head 无法精确重建 R1 样点预测。
- 少于 2 个训练空间簇、held-out 无有效样点或任一正式 LOCO fold 无效。
- `K` 小于 `minimum_loco_clusters_for_decision` 时允许完成诊断输出，但不得给出 positive 或
  `no_transfer_signal`，只能标记 `inconclusive_insufficient_loco_clusters`。
- 某角色 checkpoint 不是受支持的 1x1 最终 head 架构。
- 岭回归系统非有限、全部特征退化或没有合格 lambda。
- synthetic 配对 scope 缺失或门槛越界时记录 warning，不改变 real-LOCO 判定。
- 输出目录已存在。

脚本不提供角色替代、井标签补齐、自动改小模型、自动扩大 lambda 网格或切换优化器等兜底。

## 10. 实施与测试计划

实现时业务逻辑进入 `src/ginn_v2/`，脚本只负责配置、来源解析和输出编排。至少覆盖：

1. 原 head 经查询式隐藏特征重建后逐样点复现 R1 `r0_delta`。
2. 直井和斜井的浮点 inline/xline/TWT 插值与 R1 一致。
3. 簇等权、簇内井等权、井内样点等权之和为 1。
4. 外层 LOCO、inner-LOCO 和 LOO 不发生标签、标准化或 lambda 选择泄漏。
5. 人工线性特征能恢复已知 head；lambda 增大时解连续趋近原 R0 head。
6. 无迁移信号数据不会得到 `r2_positive`。
7. all-well head 不进入外层验证统计，失败角色始终不可被 R3 加载。
8. 相同输入、配置和设备重复运行得到相同拆分、lambda、head 和指标。
9. 缺列、全 NaN、单井、单簇、零方差特征、角色缺失、hash 不匹配、无合格 lambda 和
   synthetic preservation 越界给出固定 warning，且不会否决人工构造的 real-LOCO 正结果。

测试文件放在被忽略的 `tests/` 下，由用户本地运行。R2 在总览中作为 R1 后的研究验证入口，
不属于稳定生产链。
