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

当前 V3 evidence checkpoint 保持可读取。接入 segment profile head 后 generator payload
升级为 V4；V4 只新增 profile head 配置、权重和 feature normalization，不复制 evidence
网络或另建诊断 checkpoint 体系。

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

`ConditionalGenerator.observe()` 发布 `ObservableEvidence`；`realize()` 消费冻结的
HSMM、profile head 和 coefficient variance 合同，生成完整结构化 ensemble。signed
reflectivity 不被临时转换成未经审计的微边界概率。

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

参数 head 显式接收 state、duration、extent 和公开 evidence。第一道廉价门禁只使用 truth
segments，并冻结 evidence 网络；它不让梯度穿过 segmentation，也不把 truth segment 信息
泄漏进 evidence head。只有这道门禁通过后才加入合法 boundary jitter，推理时则在 sampled
segments 上参数化。decoder 同时产生 high-resolution 和 projected AI；`c0/c1/c2` 是
prior-selected latent。

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
  --parents-per-family 4 \
  --prior-parents-per-family 32 [--smoke]
```

该命令从每类 32 个 training parents 的 trace-local model-grid truth path 标定
zone-fraction transition/duration prior。birth/death、pinchout 和 model-grid coarsening 删除
中间事件后形成的 same-state renewal 因而进入 transition prior；event catalog 不承担局部
trace adjacency 的统计。formal 固定抽取 none、wedge、pinchout 各 4 个 tuning parents，
smoke 各取 1 个。网络证据
只前向一次。随后扫描 `state_evidence_weight = 0.25/0.5/1/2/4`、
`duration_temperature = 0.25/0.5/1/2` 和
`transition_temperature = 0.5/1/2/4` 的 80 个组合。候选阶段使用每个
parent-zone 中具有完整上下文的中央 5 道，并对每个候选运行 exact posterior marginals；
所选组合再在全部道上运行完整 Oracle。报告同时标记所选值是否落在搜索边界。

固定基线为 `(1, 1, 1)`。MAP balanced accuracy、MAP segment-count bias 和
truth-amplitude profile RMSE 只作为防止退化的 guard；候选的主要选择分数由 state Brier、
renewal Brier 和 posterior expected segment-count bias 组成。这样 calibration 直接服务于
K-member sampling，不再用 MAP accuracy 代替 posterior calibration。

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

formal truth-substitution 已把瓶颈定位到 profile amplitude：所选 HSMM 的 MAP segment
count 与 truth 基本一致；predicted-state + truth-amplitude 明显优于使用 predicted-amplitude
的组合。因此 profile 路线优先于继续搜索微边界精度。

profile head 使用 LFM-relative high-resolution truth。每个 segment 按 decoder 的实际离散
basis 发布 rank 和 condition number，并把完整 profile 压缩成 Gram、cross 与平方均值三个
充分统计量：

- 所有 supervision-valid segment 进入 decoded profile likelihood；
- 仅 rank=3、condition 不大于 100 且未 clipping 的 segment 进入单系数 likelihood；
- rank=1/2 或病态 segment 不进入 `c0/c1/c2` correlation、NLL 和 coverage 门禁；
- evidence 网络冻结，head 只消费公开 increment、reflectivity、state occupancy、state、
  duration、extent 和预测 scale。

固定比较三种方法：

```text
segment 内 deterministic evidence fit
training truth 标定的 state-conditioned parameter prior
learned segment/profile head
```

正式命令为：

```text
structured_ginn_v2.py train-profiles \
  --corpus ... \
  --checkpoint .../generator.pt \
  --output-dir ... [--smoke]
```

命令每个 epoch 发布 `last.pt` 和当前 `best.pt`，结束后发布 V4 `generator.pt`、
`profile_prior.json` 和 `profile_evaluation.json`。smoke 只验证数据、训练、V3→V4 checkpoint
和推理 interface。正式科学门禁要求 learned profile RMSE 同时低于 deterministic fit 和
state-conditioned prior；只胜过其中一个不算通过。未通过时不把 learned head接入 ensemble，
而是根据最强 baseline 判断公开 evidence pooling 是否仍有增量空间。

当前 formal profile gate 使用 384 个 training parents、96 个 tuning parents 和 150509 个
training segments。learned profile RMSE 为 `0.05206`，优于 state-conditioned prior 的
`0.06555` 和 deterministic fit 的 `0.09943`；V4 head 因此保留。该结果是在 truth state
与 truth extent 条件下成立，不能替代完整 MAP reconstruction。

第一次完整 MAP reconstruction 通过同一个 CLI 实现：

```text
structured_ginn_v2.py evaluate-reconstruction \
  --corpus ... \
  --checkpoint .../generator.pt \
  --hsmm-contract-dir .../stage1_hsmm_calibration \
  --output-dir ... \
  --split calibration \
  --parents-per-family 4 [--smoke]
```

该命令固定执行：

```text
ObservableEvidence
→ selected HSMM MAP path
→ model-grid endpoint midpoint mapping
→ high-resolution SegmentExtent
→ V4 profile parameterization
→ LFM-anchored high-resolution decoder
→ parent 内全部 zone 合并
→ complete-support finite projection
```

model-grid 内部端点映射到相邻样点中心的中点；zone 首末端点扩展到完整
high-resolution zone support。projection 只评价 FIR 窗口完全落入已重建 zone union 的
model samples，避免用零填充或 truth halo制造边缘假象。

固定对照包括 deterministic evidence fit、state-conditioned prior、anchor-only、直接
bandlimited evidence，以及两个定位用 Oracle：同一 MAP extent 替换为 truth-majority state，
和同一 calibration parents 的 truth-segment profile control。正式门禁要求 learned MAP
同时改善 high-resolution 与 projected log-AI 的最强非 Oracle baseline；失败时由两个
Oracle 对照区分 state 错误、extent 分布偏移和 profile head 本身失效。

当前 12-parent calibration 结果中，truth-segment learned profile RMSE 为 `0.05257`，但
完整 MAP high-resolution RMSE 为 `0.09957`，弱于 deterministic MAP 的 `0.08357`；
truth-majority state 可将 learned MAP 改善到 `0.08948`。projected learned RMSE 为
`0.06948`，也弱于 deterministic MAP 的 `0.05673` 和 direct evidence 的 `0.05603`。
因此 teacher-forced profile 成绩不能直接进入 ensemble；瓶颈同时包含 state 错误以及
MAP extent 相对 truth extent 的分布偏移。

固定 MAP profile 探针通过统一 CLI 运行：

```text
structured_ginn_v2.py probe-map-profiles \
  --corpus ... \
  --checkpoint .../segment_profiles/generator.pt \
  --hsmm-contract-dir .../stage1_hsmm_calibration \
  --output-dir ... [--smoke]
```

探针冻结 evidence network 和所选 HSMM，先在 training parents 上生成固定 MAP extents，
再创建一个零残差 profile head。初始均值严格等于 deterministic evidence fit；优化目标
直接由完整 LFM-anchored high-resolution log-AI 与 complete-support finite projection
组成。truth-segment profile/可识别系数只提供弱的非负辅助损失，MAP segments 不执行
predicted-to-truth split/merge matching，也不接受逐段系数 NLL。每个 epoch 发布 checkpoint，
结束后在独立 calibration parents 上执行完整 MAP reconstruction。

formal 默认使用每类 64/16/4 个 training/tuning/calibration parents、中央 5 道和 2 epochs。
探针通过要求：learned high-resolution RMSE 低于 deterministic MAP；learned projected RMSE
不高于 direct bandlimited evidence；任一 geometry family 不能在两个分辨率上同时退化。
未通过时 K-member ensemble 使用 deterministic evidence fit，learned profile head 只作为
失败审计产物；通过后再单独校准 sampled coefficient variance。

coefficient variance 使用 calibration split 做闭式 post-hoc temperature calibration：

```text
structured_ginn_v2.py calibrate-profile-variance \
  --corpus ... \
  --checkpoint .../map_profile_probe/generator.pt \
  --output-dir ... [--smoke]
```

该命令冻结 evidence、HSMM 和 coefficient mean。正式预算为每类 32 个 calibration parents
及中央 5 道；只有 rank=3、condition 不大于 100 且未 clipping 的 truth segments 用于拟合
三个全局正 temperature。每个 temperature 是固定均值、对角 Gaussian coefficient NLL 的
闭式最优解，使对应 standardized coefficient residual 的 RMS 为 1。报告同时发布校准前后
50/80/95% coverage、Gaussian NLL 和按 state 分解；短段继续保留 prior-selected latent
语义，不用于单系数校准。

校准结果写入 V5 generator checkpoint，`parameterize_segments()` 返回乘过 temperature 的
`c0/c1/c2` scale。该 scale 只表示给定 segment state/extent 后的系数不确定性；segment
数量、state 和 boundary 的不确定性由 exact HSMM backward sampling 单独提供。校准过程
不改变均值，也不重新运行优化 epoch。geometry holdout 负责后续 ensemble coverage 的独立
最终门禁。

校准后的 V5 checkpoint 通过以下命令接入冻结 HSMM 并评价完整 ensemble：

```text
structured_ginn_v2.py evaluate-ensemble \
  --corpus ... \
  --checkpoint .../coefficient_variance/generator.pt \
  --hsmm-contract-dir .../hsmm_calibration \
  --output-dir ... \
  --split calibration \
  --parents-per-family 4 \
  --realization-count 16 \
  --figures-per-family 2 [--smoke]
```

命令只通过 `ConditionalGenerator.realize()` 运行生成链。每条有效 trace 执行一次 exact
posterior forward recursion，再执行 K 次 backward sampling；每个 sampled segment 从经过
temperature 校准的 `c0/c1/c2` 分布取样。decoder 生成 high-resolution AI，并使用完整支持
的有限 FIR 投影到 model grid。evaluator 先合并同一 parent 的全部 zone，再做一次有限支持
投影，避免在内部 zone 接缝丢失样点。state occupancy、renewal、segment count 和 duration
全部在 model grid 上与同尺度 truth 比较；raw high-resolution segment table 不作为
model-grid HSMM 的 count target。报告发布 high-resolution/projected ensemble mean RMSE、
代表解 RMSE、CRPS、50/80/95% coverage 及上述结构 posterior proper scores。

`--figures-per-family N` 使用同一评估数据流为每个 geometry family 的前 N 个 parent
发布横向连续性六联图：输入地震、high-resolution truth、固定 member、代表 member、
ensemble mean，以及叠加代表 segment 起点的 ensemble standard deviation。固定 member
和代表 member 是合法 realization；ensemble mean 只用于汇总，不作为结构化解释结果。

命令同时发布 V6 generator checkpoint。V6 在 V5 上只增加冻结的 semi-Markov prior 与
conditioning，推理时无需再次拼接实验目录。代表解按 projected AI 到 observable
bandlimited evidence 的距离，从 K 个完整成员中整体选取；并列时选择 conditional score
更高者。section/volume 的全局成员选择使用同一准则。

exact HSMM 同时发布全局 MAP、state marginal、renewal marginal 和 same-state renewal
marginal；K sampling 复用同一 forward table。`StructuredPrediction` 保存 realization
identities、可选的完整 K members、ensemble summary、真实成员代表解和 recursion 诊断。
最小 smoke 已验证 25 条有效 trace 对应 25 次 forward recursion 与 `25 × K` 次 backward
sampling；smoke 的 K=2 coverage 只验证接口，不作为科学结论。

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
