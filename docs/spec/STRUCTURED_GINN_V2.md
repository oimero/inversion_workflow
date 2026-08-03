# Structured GINN V2 实施规格

## 1. 目标

Structured GINN V2 是一个由 seismic、LFM 和结构先验共同条件化的高分辨率生成器：

```text
seismic + full LFM + lateral context
→ observable-scale evidence
→ conditional semi-Markov generator
→ K 个横向连续的 high-resolution realizations
→ representative realization + ensemble summary
```

LFM 提供低频背景。网络恢复当前频带能够支持的阻抗增量、状态和有符号反射证据。
半马尔科夫先验补全亚调谐尺度的 duration、微边界和三参数 latent。

高分辨率 realization 表示观测与先验共同允许的一种地下模型，不表示地震逐层分辨了
全部微层。第一版回答：

> 在 LFM 提供低频背景的前提下，seismic 是否对状态、反射极性/幅度和阻抗剖面提供
> 可重复验证的增量信息，并能否据此生成连续、高分辨率且具有合理 coverage 的结构化
> AI ensemble？

第一版包含合成监督、横向 ensemble 和冻结 zero-shot。可微 forward loss、物理排序、
井监督和真实无标签 adaptation 在 zero-shot 结果之后规划。forward 只承担 producer
Oracle、目标物理闭环和诊断。

## 2. 已有证据与架构决定

目标和 head 只能由已发布证据或新的廉价目标审计支持。

### 2.1 Seismic 能恢复什么

冻结的 truth-segmentation、boundary-aware 实验表明，full 相对 no-seismic：

| 指标 | full | no-seismic |
| --- | ---: | ---: |
| projected log-AI RMSE | 0.03566 | 0.05404 |
| high-resolution log-AI RMSE | 0.04600 | 0.06449 |
| interface jump MAE | 0.14121 | 0.16227 |
| interface polarity accuracy | 0.64854 | 0.50261 |

within-parent seismic shuffle 会稳定破坏 AI 结果。随后 predicted-segmentation 实验中，
full 的 balanced state accuracy 为 0.5509，no-seismic 为 0.3437；projected log-AI RMSE
分别为 0.06981 和 0.13393。

因此第一版保留以下可验证目标：

- projected/model-grid log-AI increment；
- state emission；
- 有符号反射或 interface jump 的幅度与极性；
- decoded profile 和 high-resolution log-AI。

### 2.2 Seismic 不能证明什么

boundary observability audit 覆盖 16 个 parents 和 1157 个 truth boundaries：

- 调谐尺度中位数约 49.8 m；
- 最近边界间距中位数为 5 m；
- 没有一个边界同时属于 clean-forward-sensitive 和 isolated；
- raw boundary score 的 pairwise AUC 为 0.5535；
- boundary score 没有改善固定 segment-count 下的容差 F1。

三参数 Oracle 还表明，低 seismic mismatch 可以对应明显不同的 profile 和
`c0/c1/c2`。因此：

- exact micro-boundary、segment count 和单个三参数系数是 latent 审计坐标；
- micro-renewal 主要由 duration/transition prior 和 ensemble uncertainty 表达；
- boundary 指标不能单独证明 seismic 分辨了薄层。

### 2.3 当前 evidence target 的失败证据

2026-08-02 对 calibration 数据的无训练抽查发现：

```text
抽查 zone batches                         8
interface_activity 均值                  0.833
interface_activity > 0.5 的样点比例      97.1%
与 clean seismic envelope 的相关性       0.049
训练后 interface correlation             0.021
increment 相对 anchor-only RMSE 改善      2.17%
```

失败目标使用 `smooth(abs(diff(log_ai))) / trace_q95`。它把 segment 内连续 profile
梯度也计入活动，丢失反射极性和薄层相消/相长，并通过逐道归一化抹平绝对幅度。该目标
未通过标签健康度和观测对齐审计，不属于正式 evidence contract。

这项失败固定了一条实施原则：

> 全量训练用于估计模型能力，不用于发现标签是否退化。标签退化、物理错位和明显捷径
> 必须由训练前数据审计与短探针发现。

## 3. 固定数据与坐标合同

垂向数据使用 `SampleAxis`：

```text
depth: domain=depth, unit=m, depth_basis=tvdss
time:  domain=time,  unit=s, depth_basis=null
```

同一个 checkpoint 服务一种 domain；time/depth 使用同一 interface、各自权重和对应
forward adapter。固定约定为：

- `inline/xline` 负责几何寻址；
- 横向物理距离使用 `lateral_m` 或实际 XY；
- 工区 `xline_step=4` 只参与几何索引；
- 垂向 metrics 使用 sample、axis unit 或 tuning-scale fraction；
- canonical artifact 保存 clean seismic；
- dirty waveform 位于 reader 与模型之间并与 parent identity 原子绑定；
- schema、identity、domain、unit、axis、shape、dtype、mask 和几何严格校验。

canonical V2 包含 2400 个短 patch parents 和 47 个 section parents。
training/tuning/calibration 按 parent 原子划分为 1680/360/360，section parents 只参与
最终横向门禁。

## 4. 阶段 0：已完成的基础合同

阶段 0 已完成：

- canonical V2 单一 HDF5/index/manifest 发布；
- NumPy structured decoder、projection 和 forward Oracle；
- time/depth 同 interface smoke；
- SHA fingerprint 只作 provenance，不作 consumer 准入；
- `ConditionalGenerator.observe()` / `realize()` 深 interface；
- checkpoint、augmentation、section/volume inference 的基础实现。

阶段 1 的 L0–L2 observable target audit 已通过，并发布
`observable_target_contract.json`。正式 evidence 训练只接受该合同声明的三个目标与全局
尺度；checkpoint schema 为 V3。

## 5. 阶段 1：合成监督条件生成器

阶段 1 最多分三步。每一步都先使用廉价探针，再决定是否进入正式训练。

### 5.1 第一步：可观测目标审计与单道模型

#### 目标候选

首轮只审计三个具有现成证据的目标族：

1. `projected_log_ai_increment`
   - truth 为 model-grid log-AI 减 zone-linear LFM anchor；
   - old full/no-seismic 与 shuffle 已证明该目标存在 seismic 增量。
2. `signed_reflectivity`
   - truth 来自 published projected AI 进入 forward 前的有符号反射序列；
   - 保留全语料统一的物理幅度和极性；
   - time/depth adapter 必须复用各自正式 forward 的反射定义。
3. `state_emission`
   - truth 为 generator 的 low/background/high state；
   - 使用类别均衡监督；
   - 它是 HSMM emission，不等同于可独立观测的逐层分类结果。

`c0/c1/c2` 继续由完整 profile 和 decoder 监督。短 segment 或病态 basis 只监督 decoded
profile；单系数不作为目标选择门禁。

#### 训练前目标审计

目标构造由一个深模块承担：

```python
report = audit_observable_targets(corpus, output_dir, config=audit_budget)
```

调用者只提供 corpus、domain-aware target contract 和预算。模块内部完成抽样、forward
闭环、baseline、短探针、配对统计和图件。time/depth 是该 seam 的两个真实 adapter。

当前实现由 `audit_observable_targets()` 统一执行，并通过同一个 CLI 暴露：

```text
structured_ginn_v2.py audit-targets --corpus ... --output-dir ... [--smoke]
```

输出目录包含 `target_audit.json`、L0 对照图和日志。formal L0–L2 全部通过时才发布
`observable_target_contract.json`；smoke 会执行三个 level 的 interface，但不发布可供
正式训练使用的 target contract。科学门禁失败属于有效审计结果，报告照常完成。

每个候选必须发布：

- finite/support/axis/unit 检查；
- mean、std、分位数、符号比例、零值比例和动态范围；
- categorical target 的类别比例、entropy 和 constant-prior score；
- 按 geometry family、zone 和 parent 分组的同类统计；
- target 经正式 forward adapter 重建 clean seismic 的误差、相关性和 lag；
- full-LFM、zone-linear anchor 和 constant/marginal baseline；
- 至少 12 组 seismic、LFM、truth target 和 forward reconstruction 对照图。

目标使用 training split 的全局 robust scale 只做数值缩放。逐道、逐 zone 或逐 parent
归一化不得改变目标的物理幅度排序。

#### 训练预算闸门

预算按以下顺序执行，前一层失败时停止：

```text
L0  无训练 target audit       统计抽样，目标 < 2 分钟
L1  tiny overfit              4–8 parents，目标 < 5 分钟
L2  information probe        48 train + 24 tuning parents，目标 < 20 分钟
L3  formal training          只有 L0/L1/L2 全部通过后运行
```

L0 要求 target 具有非退化动态范围，物理闭环正确，且图件中 target 与 forward response
语义一致。L1 要求小网络能够明显拟合目标；失败优先解释为 loss、support 或实现错误。

L2 对每个 head 独立训练或评估：

```text
full
no-seismic
matched within-parent seismic shuffle
constant / anchor / full-LFM baseline
```

matched shuffle 在同一 parent、同一 zone 的邻道间循环选择 donor，只替换双方共同有效区，
recipient 的 support 保持不变；报告共同有效覆盖率、实际改动比例和波形差异 RMS。

每个正式 L2 探针至少执行与 L1 相同的更新步数；epoch 换算不足时采用 L1 步数下限，
同时服从 20 分钟总预算。

进入 L3 要求：

- full 相对 no-seismic 的 parent-paired 主指标改善 95% CI 方向正确；
- matched shuffle 使该改善显著下降；
- full 的收益不是由少数 parent 或单一 geometry family 提供；
- target baseline 与模型指标之间存在足够可解释的改进空间；
- 单 head 通过后才进入多任务模型，避免总 loss 掩盖失效 head。

未通过 L2 的 target 从正式 interface 中移除。它不会通过增加 epoch、增大网络或修改
loss 权重进入 L3。

#### 正式 evidence interface

通过 L2 后冻结：

```text
ObservableEvidence
├── background_lfm_linear
├── projected_log_ai_increment_mean / scale
├── signed_reflectivity_mean / scale
├── state_log_potential[..., 3]
├── local_tuning_scale
├── support
├── model/highres axes
└── physical lateral geometry
```

网络输入为固定语义通道：

```text
scaled seismic
scaled (full_lfm - background_lfm_linear)
observed validity
```

输入 scale 来自 training split 全局统计并写入 checkpoint。full/no-seismic 使用同一
初始化、split、训练顺序和预算。

当前 L3 实现采用单道编码，即一个 batch 可以包含多道，但每道 feature 不做横向混合。
网络分别输出两个连续 mean/scale head 和一个 log-normalized state potential head。训练
loss 的默认权重为 projected increment `1.0`、signed reflectivity `0.5`、state
`0.25`，两个 scale calibration 项合计使用 `0.1` 权重；日志逐项报告，不以总 loss
代替单 head 指标。

训练命令必须显式提供通过审计的 target contract：

```text
structured_ginn_v2.py train \
  --corpus ... \
  --target-contract .../observable_target_contract.json \
  --output-dir ... \
  --input-mode full|no_seismic [--smoke]
```

`--smoke` 固定使用 3 个 training parents、3 个 tuning parents 和 1 epoch。正式训练每个
epoch 发布 epoch checkpoint、`last.pt` 和当前 `best.pt`。统一 evaluator 同时报告
full、no-seismic、matched within-parent shuffle、full-LFM 和 zone-linear anchor。

当前 `ConditionalGenerator.observe()` 已发布 `ObservableEvidence`；`realize()` 在第二步
HSMM 合同落地前明确暂停，防止将 signed reflectivity 临时转换成未经审计的微边界概率。

正式单道门禁以少数主指标固定：

- projected log-AI increment RMSE；
- signed reflectivity MAE、correlation 和 polarity accuracy；
- balanced state accuracy / state proper score；
- decoded projected/high-resolution log-AI；
- full-no-seismic 与 matched-shuffle 的 parent 配对差值。

### 5.2 第二步：Conditional HSMM 与 ensemble

核心 interface 保持：

```python
evidence = generator.observe(observation_tile)
prediction = generator.realize(evidence, generation_policy)
```

HSMM 与 profile generator 只消费公开 `ObservableEvidence`：

- `state_log_potential` 条件化 state path；
- signed reflectivity 为 segment 端点与 profile amplitude 提供有符号证据；
- projected increment 约束 decoded model-grid AI；
- transition/duration prior 来自 training parents，duration 单位为 zone fraction。

第一版 micro-renewal 由 duration prior 与 state evidence 共同决定。独立 micro-boundary
head 只有在新的 target audit 证明可观测增量后才能进入 interface。相邻 segment 允许
state 相同，一次 posterior recursion 支持默认 `K=16` 的 backward sampling。

参数 head 使用 truth segments 和合法 boundary jitter 训练，并显式接收 state、duration、
extent 和公开 evidence。推理时在 sampled segments 上参数化。decoder 同时产生
high-resolution 和 projected AI；`c0/c1/c2` 是 prior-selected latent。

第二步先运行短 truth-substitution：

```text
predicted state evidence + predicted amplitude evidence
truth state evidence     + predicted amplitude evidence
predicted state evidence + truth amplitude evidence
truth state evidence     + truth amplitude evidence
prior-only
```

当前 state-duration 校准与 Oracle 通过统一 CLI 落地：

```text
structured_ginn_v2.py evaluate-hsmm \
  --corpus ... \
  --checkpoint .../generator.pt \
  --output-dir ... \
  --split tuning \
  --parents-per-family 4 [--smoke]
```

该命令从 training parents 的 canonical object catalog 标定 zone-fraction transition/duration
prior；formal 固定抽取 none、wedge、pinchout 各 4 个 parents，smoke 各取 1 个。网络证据
只前向一次。随后扫描 `state_evidence_weight = 0.5/1/2/4`、
`duration_temperature = 0.5/1/2` 和 `transition_temperature = 1/2/4` 的 36 个组合。
候选阶段使用每个 parent-zone 中具有完整上下文的中央 5 道，只运行 Viterbi；所选组合
再在全部道上运行完整 posterior recursion 与 marginals。报告同时标记所选值是否落在
搜索边界，供后续判断是否需要扩展范围。

固定基线为 `(1, 1, 1)`。候选必须同时满足：state balanced accuracy 不低于基线、MAP
segment-count bias 的绝对值不大于基线、truth-amplitude profile RMSE 不大于基线。合格
候选中，profile RMSE 位于最优值 `max(1%, 1e-5)` 容差内视为实际等价；再保留
segment-count bias 绝对值距最优值不超过 `0.25` 段的候选，最终选择 balanced accuracy
最高者。没有其他合格候选时保留基线。这一选择只校准公开证据与已标定 prior 的相对强度。

命令发布 `semi_markov_prior.json`、`hsmm_calibration.json` 和所选组合对应的
`hsmm_oracle.json`。Oracle 在 model grid 上复用同一个 exact HSMM，并报告：

- truth-state、predicted-state 和 prior-only 的 MAP state、state marginal、renewal marginal；
- same-state renewal、segment count 和 duration-fraction distribution；
- truth/predicted state 与 truth/predicted amplitude 的四格 substitution；
- 每个 MAP segment 内三参数 basis 拟合后的 projected increment。

renewal observation likelihood 固定为中性的 `0.5`。此时每条完整路径的 boundary/no-boundary
观测项相同，micro-renewal 只由 training prior 与 state evidence 条件化。signed
reflectivity 保留给后续 learned segment parameter head；它不会被临时转换成 micro-boundary
likelihood。Oracle 中的 segment-wise 三参数拟合是参数 head 之前的确定性上限诊断。

exact HSMM 当前同时发布全局 MAP、state marginal、renewal marginal 和 same-state renewal
marginal；一次 forward recursion 继续支持后续 K 次 backward sampling。`realize()` 在 learned
segment parameter head 和 high-resolution decoder 接入后发布完整 `StructuredPrediction`。

正式门禁为：

- full ensemble 相对 no-seismic/prior-only 改善 projected AI、state 和 profile；
- matched shuffle 破坏上述收益；
- bandlimited/projected truth 的 coverage 与 CRPS/energy score合理；
- segment count、duration 和 state occupancy 分布得到校准；
- truth-substitution 能定位剩余瓶颈；
- coefficient 与 exact micro-boundary 指标只作审计。

代表解必须是 K 个完整 realization 中的真实成员。逐道拼接和逐点 ensemble mean 不是
结构化代表解。

### 5.3 第三步：横向、dirty 与 section 门禁

单道 clean 门禁通过后接入 21 道、米制距离和显式 mask。横向收益通过 single-trace
与 neighbor-shuffle 证明。微结构连续性使用由 XY、zone identity 和 realization
identity 决定的 coordinate-stable correlated random fields。

真实观测统计 profile 在此步冻结，覆盖 phase、shift、gain、频带、振幅衰减和噪声等
nuisance。它只要求 synthetic observation 覆盖真实统计支持，不要求合成与真实波形
不可区分，也不改变地下 truth。

执行顺序仍采用 L0/L1/L2 小预算探针；clean 结论成立后才训练 dirty。最终一次运行
section benchmark，检查：

- full 相对 no-seismic、single-trace 和 neighbor-shuffle 的收益；
- clean/dirty 配对退化与不确定性变化；
- variogram、roughness、event continuity 和 birth/death；
- pinchout false bridging；
- direction reversal 与 tile/halo/stitching invariance；
- IID seed 与 combination holdout 结论一致性。

## 6. 阶段 2：真实工区冻结 zero-shot

先运行 6 口可信井剖面、3 口低质量井诊断剖面和至少 2 条 blind sections。full 与
no-seismic checkpoint 同时运行。inline/xline 分别生成 observable evidence，融合
calibrated evidence 后只生成一套微结构 ensemble。

section 保存 K 个完整成员。全体积保存：

- projected increment、signed reflectivity 和 state evidence；
- high-resolution ensemble mean/std；
- representative high-resolution log-AI 和 segment table；
- direction disagreement、seismic support 和 LFM sensitivity；
- support/stitching mask 与可局部重生成的 realization identities。

真实井门禁：

- full 相对 no-seismic 在井上对应频带的配对改善为正；
- full 相对 full-LFM 和 zone-linear anchor 有整体增量价值；
- high-resolution ensemble 对井曲线具有合理 coverage；
- 代表解在多数可信井上不相对 LFM 退化；
- 低质量井只作逐井诊断；
- blind sections 无 seam、方向条带或微结构随机跳变。

剖面门禁通过后才运行全体积。zero-shot 结果发布后再规划井监督、物理约束和真实
adaptation。

## 7. 验证与运行纪律

实现至少覆盖：

- target audit 的统计、图件、forward closure 和 family 分层；
- L0/L1/L2 失败能够阻止 formal training；
- full/no-seismic 使用相同随机合同；
- matched shuffle 只破坏 seismic-target 对应；
- target 全局 scale 不改变 parent 间幅度排序；
- time/depth target adapter 与正式 forward 一致；
- canonical V2 split 和 parent 原子隔离；
- small-sequence HSMM 与 brute-force posterior/sampling 对照；
- same-state renewal 与一次 recursion 的 K sampling；
- truth-substitution 对照；
- representative 是 ensemble 真实成员；
- spatial sampling 的 tile/order/batch determinism；
- inline/xline evidence 先融合、微结构只生成一次；
- section continuity、pinchout 和 false bridging；
- full/no-seismic 真实井配对报告。

正式训练命令必须在启动前读取通过的 target-audit artifact。训练按 batch 固定频率记录
每个 head 的独立 loss 和 baseline-relative metric；总 loss 只用于优化，不作为科学
验收指标。
