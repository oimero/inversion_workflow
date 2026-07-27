# Structured GINN V2 实施规格

## 1. 第一版目标

第一版只回答一个问题：

> 网络能否在完整 LFM 提供低频输入、zone-linear LFM anchor 提供 decoder 背景的前提下，通过半马尔科夫结构、界面证据和横向上下文，在结构化生成族内推断连续、高分辨率、可验证且可标记证据来源的 AI 解？

三参数对象表示为：

```text
profile = c0 + c1 * (2ξ - 1) + c2 * sin(πξ)
```

三参数是 generator、truth artifact 和 decoder 的精确坐标。冻结的 truth-boundary observability decision 已证明逐 segment 三参数在当前带限地震下不具有唯一可识别性；网络输出的三参数解释为由 seismic、LFM 和结构先验共同选择的 `prior-selected latent`。阶段 1 的主要交付是 state、boundary、interface jump、decoded profile 和 projected log-AI，单系数作为 generator/decoder 审计坐标完整保留。

zone 背景表示为：

```text
background = a + b * (2ζ - 1)
log_ai = background + profile
```

第一版的约束来自：

- 深度域；
- 已知 zone 边界；
- zone-linear LFM anchor 决定的 decoder 背景；
- 相对 zone-linear anchor 换基的三参数 profile；
- 固定 transition 和 duration prior 的 exact HSMM；
- 一维横向上下文和拓扑感知连续性；
- 高分辨率 decoder。

第一版使用：

- 单 zone 中心道预测；
- 21 道一维横向 patch；
- clean seismic 和 LFM 两个输入通道；
- 三状态 emission；
- boundary 与 signed interface jump 监督；
- segment decoded profile 监督；
- identifiable subset 的 state-conditioned `c0/c1/c2` 辅助监督；
- MAP、state/boundary posterior 和 jump/profile 方差。

第一版的结果是结构提案、界面变化、decoded profile、projected log-AI、prior-selected latent、横向连续性和证据来源诊断。真实工区只使用冻结模型推理。

波形 augmentation 属于模型训练侧。canonical dataset 保存 clean observation；训练时按显式随机 identity 在线生成 phase、shift、gain、static 和 noise 扰动。

第一版不加入可微 forward loss、物理排序或真实无标签微调。这里的约束是结构化 decoder、HSMM、已知 zone、zone-linear LFM anchor 和横向一致性；高分辨率先验输出不等同于地震直接分辨。深度域 differentiable forward seam 继续作为诊断基础设施保留，是否升级为训练约束只根据第一版结果重新规划。

## 2. 固定数据与坐标约定

time/depth workflow 使用相同的 in-memory record、writer、HDF5 layout 和 reader interface。

domain 差异只进入：

- sample domain；
- sample unit；
- sample axis；
- depth basis；
- forward context；
- domain-specific forward extras。

深度域约定：

```text
sample_domain = depth
sample_unit = m
depth_basis = tvdss
positive_direction = down
```

时间域约定：

```text
sample_domain = time
sample_unit = s
depth_basis = null
positive_direction = increasing_time
```

`inline` 和 `xline` 是几何身份。物理横向距离使用 `lateral_m`。工区 `xline_step=4` 来自 survey geometry，并随 parent identity 保存；它不是 writer 或模型中的常量，也不参与单道垂向 forward。

带采样轴的数据使用 `SampleAxis` 或 `wtie.processing.grid` 中的带轴对象。裸数组只存在于模块内部实现。

三种 mask 分离：

- `observed_valid`：模型网格 observation、LFM 和 forward 的合法支持；
- `truth_valid_highres`：高分辨率有限 truth；
- `segment_supervision_valid`：canonical artifact 中 segment 的端点、样点和参数记录结构合法。

`segment_supervision_valid` 不声明三个系数对观测可识别。阶段 1 在 LFM anchor/dataset seam 继续派生 profile supervision 和 parameter identifiability mask。

## 3. 阶段 0：canonical dataset 重构

状态：代码已落地，depth smoke 已通过。

## 4. 一次性迁移：20260725

## 5. 阶段 1：合成监督结构化模型

状态：阶段 0、完整迁移、sampled Oracle 和阶段 1 第一步均已完成。mean-pooling 前置审计以及 boundary-aware full、no-seismic、controls evaluator、truth-boundary observability Oracle 已冻结。当前实施项是第二步：单道结构提案与 exact HSMM。

前置审计使用 truth segment/state/duration 和 segment mean pooling。full model 相对 no-seismic 在 projected log-AI 上稳定改善，但参数恢复呈现明显分化：

```text
c0 R² = 0.831
c1 R² = 0.010
c2 R² = 0.040
```

这个结果证明 seismic 对 decoded profile 有贡献，也暴露了 mean pooling 与三参数 basis 不匹配：

- `2ξ-1` 在 segment 内均值为零，mean pooling 天然削弱 `c1` 证据；
- `sin(πξ)` 的均值与常值项混合，mean pooling 难以分离 `c0/c2`；
- truth state 已经提供了很强的 `c0` 条件信息。

boundary-aware 第一步随后完成。82 个 tuning parents 上，full model 相对 no-seismic 的 projected log-AI RMSE 从 0.05404 降至 0.03566，high-resolution log-AI RMSE 从 0.06449 降至 0.04600；within-parent-zone seismic shuffle 使两项指标稳定退化，证明模型使用了中心道 seismic 与目标之间的对应关系。

truth-boundary Oracle 固定 truth segmentation、LFM anchor、zone 外 truth 和深度域 forward context，只优化合法 segment 的三参数。truth-init 最大 seismic normalized RMSE 为 0.000039；随机初值可将该误差压至中位 0.01422，但 profile RMSE 中位仍为 0.08196。每个样本的连续参数数中位为 118.5，有效 seismic 样点数中位为 68.5。冻结结论为：

- state、boundary、interface jump、decoded profile 和 projected log-AI 是后续主要输出与门禁；
- `c0/c1/c2` 保持为 decoder 所需的 `prior-selected latent` 和审计坐标；
- 单系数恢复不构成阶段推进条件，也不解释为 seismic 唯一分辨；
- 第一步完整证据位于 `note/summary/final_audit/20260726_structured_ginn_v2_boundary_observability/`。

阶段 1 后续只分三步。

### 第一步：可观测性对齐的单道 profile head

状态：完成并冻结。

使用宽度 1 的 `StructuredPatch`、truth segments 和合法 boundary jitter，建立 boundary-aware、basis-aware 的 segment parameterization。保留冻结的 mean-pooling full/no-seismic 结果作为旧实现对照，不把旧 checkpoint 直接视为新 head 的验收结果。

segment evidence 至少包含：

- segment interior mean feature；
- top/bottom 的 inside/outside feature；
- segment center feature；
- 按 `1`、`2ξ-1` 和 `sin(πξ)` 加权的 feature moments；
- state embedding、duration fraction 和 extent。

top/bottom signed log-AI jump truth 与 polarity 由 structured truth 派生，只作为辅助监督和评价目标，绝不作为 parameter head 的输入。

parameter head 继续输出 decoder 所需的 `c0/c1/c2` 分布，但主要监督和报告对象调整为：

- decoded high-resolution profile；
- projected 5 m log-AI；
- top/bottom interface jump；
- interface polarity；
- identifiable subset 的单系数 likelihood。

state 与 `c0` 采用层级关系：state 描述离散地质类别和序列拓扑，`c0` 描述给定 state 后的连续幅度。禁止并行生成互不约束的 state 与 `c0` 结论。

第一步固定比较：

- frozen mean-pooling full model；
- boundary-aware full model；
- 配对的 boundary-aware no-seismic；
- state-duration analytic baseline。

第一步验收结果：

- boundary-aware full model 相对 frozen mean-pooling full model 的 projected log-AI RMSE 平均降低 0.00904，95% CI 为 [0.00813, 0.00998]；
- full model 相对 no-seismic 的 projected log-AI、high-resolution log-AI 和 interface jump 改善在全部 tuning parents 上为正；
- seismic shuffle 后 profile、projected AI、interface jump 和三个单系数指标均退化，其中 profile/AI 退化最稳定；
- `c0` 是最容易学习的系数，`c1/c2` 只有弱的 seismic-sensitive signal；truth-boundary Oracle 进一步证明三参数整体不具有唯一可识别性；
- decoder、projection、depth forward 和 truth-init 数值闭环通过。

第一步使用以下 truth-boundary Oracle 可观测性诊断：

```text
truth boundaries + truth LFM
→ 多初值优化 c0/c1/c2
→ differentiable depth forward
→ 参数、profile、projected AI 和多解离散度
```

该诊断只判断在结构完全正确时地震是否能区分三参数，不进入训练 loss，不产生候选排序，也不用于真实工区推理。冻结 decision manifest 的结论为 `profile_primary_coefficients_audit_only`：阶段 1 后续以 state、boundary、interface jump、decoded profile 和 projected log-AI 为主输出；完整三参数 truth、辅助监督、posterior 和 decoder 继续保留，但不承担唯一参数恢复的科学声明。

### 第二步：单道结构提案与 exact HSMM

预计新增生产代码 1000–1600 行。

在第一步冻结的 profile head interface 上增加：

- 三状态 emission head；
- boundary 与 interface polarity/jump evidence head；
- exact HSMM MAP；
- forward-backward marginals；
- zone-fraction duration prior；
- predicted MAP segmentation 的端到端验证。

单道闭环固定为：

```text
single-trace StructuredPatch
→ DirectionalEvidence
→ exact HSMM MAP + marginals
→ unique predicted segments
→ boundary-aware parameterize_segments()
→ LFM-anchored decoder
→ CenterTracePosterior
```

parameter head 仍只使用 truth segments 和合法 boundary jitter 训练。predicted segments 只进入端到端验证，不实现 predicted/truth split/merge matching，也不让梯度穿过 MAP segmentation。

第一版只输出唯一 MAP segments 和 posterior marginals，不实现 top-K HSMM、beam search、物理排序或局部边界优化。只有当 MAP 失败主要来自明确的多峰结构 posterior 时，才在阶段 1 完成后重新讨论候选集合。

第二步验收要求：

- 小序列 MAP 与 brute-force 枚举完全一致；
- forward-backward marginals 归一且与 brute-force 对照一致；
- zone-fraction duration prior 在不同采样间隔下保持物理一致；
- predicted segments 能生成唯一、合法的 segment/profile 表；
- boundary F1、segment IoU、duration error、segment count bias 和 projected AI 没有结构性异常；
- full model 相对 no-seismic 出现稳定的结构或 profile 增益。

### 第三步：横向能力和 sim-to-real 防线

预计新增生产代码 1200–2000 行。

将同一 interface 扩展到 21 道米制 patch，完成：

- masked lateral mixer；
- event identity 与 topology mask；
- topology-aware lateral consistency；
- observation augmentation profile；
- clean/dirty 固定配对；
- matched center、neighbor 和 parent shuffle；
- posterior、variance 和 evidence calibration；
- full-LFM、anchor、no-seismic、single-trace 和 lateral baseline；
- geometry holdout 最终门禁；
- 供阶段 2 使用的冻结 full/no-seismic checkpoints。

闭环固定为：

```text
21-trace clean/dirty StructuredPatch
→ lateral DirectionalEvidence
→ exact HSMM
→ boundary-aware segment profile
→ calibration
→ geometry holdout
→ frozen real-field checkpoints
```

第三步验收使用 5.6 节完整阶段 1 门禁：AI/profile 增量价值、seismic contribution、lateral contribution 和 clean/dirty 对称门禁分别通过；不增加 pinchout false-bridging；state/boundary posterior、profile variance 和 `seismic_support_score` 完成冻结校准。

augmentation、calibration 和 geometry holdout 属于同一个最终实施切片，不再拆出额外步骤。

### 5.1 Zone-linear LFM anchor 与离散换基

zone 顶底来自已知层位。对每个 zone，在 `observed_valid` 支持内将完整 LFM 解析投影为：

```text
background_lfm_linear = a_lfm + b_lfm * (2ζ - 1)
```

`a_lfm/b_lfm` 是确定量，网络不预测 zone `a/b`。完整 LFM 是模型输入和敏感性参考；decoder baseline 是 `background_lfm_linear`，两者不能混称。

正式分别报告：

- `prediction - background_lfm_linear`：结构化网络增量；
- `full_lfm - background_lfm_linear`：zone 线性投影舍弃的 LFM 分量；
- prediction 与完整井曲线的最终指标。

artifact 中的 generator truth 继续保留 raw、projected、effective 三套参数。模型监督使用相对 zone-linear LFM anchor 换基后的 effective 三参数。

连续形式只用于解释。定义：

```text
Δa = a_truth - a_lfm
Δb = b_truth - b_lfm
```

对 zone 内中点为 `ζ_mid`、厚度比例为 `f` 的常规 segment：

```text
c0_lfm = c0_effective + Δa + Δb * (2ζ_mid - 1)
c1_lfm = c1_effective + Δb * f
c2_lfm = c2_effective
```

正式 implementation 使用 decoder 的实际离散 sample、mask 和 basis 求 coefficient adjustment：

```text
Δbackground[j] = Δc0 + Δc1 * (2ξ[j] - 1)
```

离散合同固定为：

- `xi` 的分母使用 `max(segment thickness, highres sample interval)`；
- 非末段使用 `[top, bottom)`；
- 末段使用 `[top, bottom]`；
- `c2` 保持不变；
- rank-2 segment 求唯一 `Δc0/Δc1`；
- singleton 使用 `Δc1=0`、`Δc0=Δbackground[sample]`；
- 换基结果经过与原 decoder 相同的 clipping；
- 每个 segment 发布离散换基 residual，超过 parity tolerance 直接失败。

换基逻辑集中在一个 interface：

```text
anchor_to_lfm(StructuredSample) -> LfmAnchoredStructuredSample
```

reader、dataset、模型和 decoder 不各自实现换基公式。`LfmAnchoredStructuredSample` 保存：

- 完整 LFM；
- `background_lfm_linear`；
- 换基后的 effective 参数；
- 离散换基 residual；
- `profile_supervision_valid`；
- `parameter_supervision_valid`；
- `parameter_identifiability_rank`；
- `parameter_basis_condition`；
- 原 artifact 参数身份。

后三类参数可识别性字段由 anchor/dataset seam 根据 segment 的实际离散 basis 派生，不写回 canonical HDF5，也不触发 artifact 重迁移。对 segment 中每个有效高分辨率样点构造：

```text
B[j] = [1, 2ξ[j] - 1, sin(πξ[j])]
```

首版规则固定为：

- canonical segment 合法、至少含一个有限 truth sample 且 decoder clipping 可复现时，`profile_supervision_valid=true`；
- `rank(B)=3`、`condition(B) <= 100` 且 segment 内没有 clipping 时，`parameter_supervision_valid=true`；
- rank 不足或严重病态时，只监督 decoded profile，不监督单个 `c0/c1/c2`；
- singleton、rank-2 和 clipping segment 仍参加适用的 profile/decoder parity，但不因 generator 保存了唯一参数而被视为三参数可识别；
- 参数相关性、parameter NLL 和区间 coverage 只统计 `parameter_supervision_valid` 子集。

第一版不实现 SVD 可识别线性组合的独立 likelihood。若后续发现 rank 不足样本占比足以影响 profile 学习，再单独设计该合同。

NumPy 与 Torch implementation 必须在普通、singleton、首段、末段、pinchout 和 clipping 情形下证明：

```text
generator background + original effective profile
==
background_lfm_linear + rebased effective profile
```

### 5.2 数据划分

`development_pool` 按 parent 原子划分。划分使用固定 seed，按 section、duration mode 和 geometry family 分层，并发布不可变 split manifest：

- 70% training；
- 15% tuning validation；
- 15% calibration。

职责固定为：

- tuning validation：选择模型、loss 权重和训练轮次；
- calibration：拟合 posterior temperature、parameter variance scale 和 evidence classification threshold；
- geometry holdout：只运行一次阶段 1 最终门禁。

同一 parent 的 lateral traces、clean/dirty 配对和所有 zone 必须进入同一 split。禁止逐道随机划分。dirty random identities 在 calibration 前冻结。

`geometry_holdout` 不参与：

- 模型训练；
- 超参数选择；
- augmentation 上限选择；
- posterior temperature 或参数区间校准；
- evidence threshold 冻结。

training、tuning validation、calibration 和 geometry holdout 的 parent 集合必须两两不相交。

### 5.3 模型 interface

模型读取一维横向 patch：

```text
StructuredPatch
├── seismic              # [21, vertical]
├── lfm                  # [21, vertical]
├── observed_valid
├── zone top / bottom
├── sample axes
├── relative_lateral_m   # 以中心道为 0
└── lateral_valid
```

标准横向采样为 25 m。patch 使用 21 道，覆盖中心道两侧各约 250 m。缺失邻道和 survey 边缘只通过 `lateral_valid` 表示，不复制边缘道。

模型的第一层 interface 只编码中心道证据：

```text
encode_patch(StructuredPatch) -> DirectionalEvidence

DirectionalEvidence
├── emission_log_potential
├── boundary_log_potential
├── interface_polarity_log_potential
├── interface_jump_mean / log_scale
└── center_feature_sequence
```

`interface_jump` 的正方向固定为深度增加方向上的：

```text
q[j] = log_ai[j+] - log_ai[j-]
```

它只在界面两侧 truth 均合法、endpoint identity 明确时监督。polarity 分为 `increase/decrease/no_resolved_jump`；`no_resolved_jump` 是观测标签，不是额外 HSMM state。interface jump 是局部观测证据，state 是 segment 级序列变量，两者分别保留。

参数化是独立的第二层 interface：

```text
parameterize_segments(
    center_feature_sequence,
    externally_supplied_segments
) -> SegmentParameterDistributions
```

`externally_supplied_segments` 必须包含 state、duration fraction 和 segment extent。parameterization module 在内部构造 boundary-aware、basis-aware segment evidence，调用者不需要了解 pooling 实现。它至少使用：

- state embedding；
- duration fraction；
- segment extent；
- interior mean、center feature；
- top/bottom inside/outside feature；
- 按 `1`、`2ξ-1` 和 `sin(πξ)` 加权的 feature moments。

禁止只用无位置的 segment mean 作为 profile head 的唯一地震摘要。basis-aware moments 使用 segment 的实际离散 basis、endpoint convention 和 valid mask；top/bottom context 不越过 zone 或 observation support。

完整单方向推理组合两个 interface：

```text
DirectionalEvidence
→ exact HSMM MAP + marginals
→ unique segments
→ parameterize_segments()
→ LFM-anchored decoder
→ CenterTracePosterior
```

implementation 包含：

- 共享的逐道垂向编码器；
- 使用相对米制距离和 lateral mask 的横向 mixer；
- high-resolution 三状态 emission、boundary、interface polarity/jump head；
- exact HSMM MAP 和 posterior marginals；
- boundary-aware segment 三参数均值与受限方差 head；
- LFM-anchored Torch decoder。

parameter head 训练只使用 truth segments 和合法 boundary jitter。对 `parameter_supervision_valid` segment 使用三参数 likelihood；对只有 `profile_supervision_valid` 的 segment，只通过 LFM-anchored decoder 监督离散 profile。predicted MAP 或 fused segments 只用于端到端验证和推理。第一版不实现 predicted/truth segment matching，不对 MAP segmentation 反向传播。

parameter head 在推理时仍为每个合法 segment 输出完整三参数分布，但 rank 不足或病态 segment 的单系数结果不声明为可识别参数，也不进入单系数门禁。

producer state 保持原语义：相对 generator 背景的 `low_impedance/background/high_impedance`。换基不重标 state。parameter head 使用 state conditioning，并报告：

- `state_parameter_inconsistency_rate`；
- state-conditioned parameter NLL；
- profile mean-sign conflict；
- state support violation。

初步抽样中，换基后的非 background profile mean-sign conflict 约为 0.085%，因此首版不引入 state relabeling。

transition 和 duration prior 只从 training truth 标定，训练期间固定。duration prior 的统计单位是：

```text
segment thickness / zone thickness
```

即 zone fraction。HSMM 根据当前 zone、sample axis 和 endpoint convention 将其离散为合法 duration bins。`duration_samples` 只用于当前 artifact 的监督和 QC，不作为跨工区先验单位。

首版没有 merge、unknown 或“低证据时删除薄层”的特殊状态。地震证据不足主要表现为：

- state/boundary posterior 展宽；
- interface jump evidence 变弱；
- decoded profile 幅度减弱；
- profile/参数方差增大；
- matched-seismic intervention sensitivity 降低。

在一个 parent 内，`(zone_id, object_id)` 是跨 lateral 的生成事件 identity。`object_id` 在 lateral loop 前分配，不是逐道重新编号；`lateral_index` 只标识该事件在某道上的实例。

dataset 在施加横向 consistency 前必须验证：

- event identity 跨道稳定；
- object 顺序不反转；
- 同一 event 的 producer state 一致；
- zero-sample 或 supervision-invalid segment 只表示 birth、death 或 pinchout；
- topology transition 区域不施加横向连续性损失。

### 5.4 训练课程

训练课程与阶段 1 的三个实施步骤完全一致：

- 第一步只使用单道 truth+jitter segments，完成 boundary-aware profile、interface jump/polarity 和参数可观测性决策；
- 第二步加入 state/boundary sequence supervision、exact HSMM 和 predicted-segment 端到端验证；
- 第三步加入 lateral patch、topology-aware consistency、在线 waveform augmentation，并冻结 clean/dirty/calibration/geometry 合同。

损失只作用于合法 mask，至少包含：

- 三状态 emission 与 semi-Markov sequence likelihood；
- boundary supervision；
- interface jump likelihood 与 polarity supervision；
- duration prior；
- rebased `c0/c1/c2` 参数 likelihood，仅作用于 `parameter_supervision_valid`；
- decoded high-resolution log-AI/profile loss，作用于 `profile_supervision_valid`；
- projected 5 m log-AI；
- topology-aware lateral consistency。

横向一致性只约束具有相同 event identity 且 supervision-valid 的对象、状态和边界。pinchout、对象产生/消失和显式 topology transition 不施加跨事件平滑。

阶段 1 训练不使用 forward seismic loss。第一步按需触发的 truth-boundary Oracle 只是一项可观测性诊断，不改变训练合同。

### 5.5 真实统计约束的观测扰动

canonical HDF5 始终保存 clean seismic。训练前冻结一份 observation augmentation profile，统计来源为：

- 真实深度域 seismic 的振幅、噪声颜色和邻道相干范围；
- 6 口可信井震标定的 phase/shift 不确定性；
- 冻结 wavelet。

profile 不读取真实地下标签，不拟合真实 AI，也不生成第二套 materialized seismic views。

在线 augmentation 位于 reader 与模型之间：

```text
LfmAnchoredStructuredSample
→ SeismicAugmentationPipeline(profile, random identity)
→ augmented seismic + unchanged LFM/truth
→ StructuredPatch
```

支持：

- bounded phase rotation；
- bounded wavelet shift；
- depth-domain static；
- global/tracewise positive gain；
- 平滑的深度—横向振幅衰减；
- white、colored 和 coherent noise；
- 弱反射压低场景。

每个 dirty sample 与同一 clean truth 配对。扰动超出冻结统计边界、产生非有限值或破坏 valid support 时直接拒绝。

证据干预分为四种不同合同：

- `no-seismic control`：独立训练的 baseline；
- `matched center-seismic shuffle`：从相同 zone geometry 和匹配振幅、频谱统计的其他 parent 替换中心 seismic；
- `neighbor shuffle`：中心 seismic 不变，只破坏邻道横向对应；
- `parent shuffle`：替换整个 patch，用于整体伪相关对照。

第一版不把全零 seismic 当正式 evidence 指标。只有未来训练合同显式包含 seismic dropout，才允许使用 zero-input sensitivity。

evidence classification threshold 只在 dirty calibration set 上冻结。它根据正确 seismic 相对 matched intervention 是否改善结构、参数和 decoded profile 恢复来标定，不按单一输出变化幅度直接分档。阶段 1 发布连续 `seismic_support_score` 及 `seismic_supported/mixed_support/non_seismic_supported` 阈值；`non_seismic_supported` 不区分 LFM 与 HSMM prior 的贡献。

### 5.6 阶段 1 评价与门禁

固定比较：

- full-LFM-only；
- zone-linear anchor-only；
- no-seismic；
- single-trace；
- lateral model；
- matched center-seismic shuffle；
- parent-shuffle；
- neighbor-shuffle。

`full-LFM-only` 直接使用完整 LFM；`zone-linear anchor-only` 使用 `background_lfm_linear`；`no-seismic` 是保留完整 LFM 输入、但不接收 seismic 的独立训练 control。三者不能合并为同一个 baseline。

至少报告：

- state accuracy、segment IoU 和 duration error；
- boundary polarity accuracy、signed interface jump error 和 jump interval coverage；
- 全部 `profile_supervision_valid` segment 的 decoded profile error；
- decoded high-resolution 与 projected 5 m log-AI error；
- identifiable subset 的 `c0/c1/c2` MAE、R²、相关性、符号一致率、NLL 和区间 coverage；
- clean/dirty 配对性能差；
- segment count bias；
- false-positive/false-negative boundary rate；
- high-confidence wrong segment rate；
- `non_seismic_supported` segment rate；
- pinchout false-bridging；
- 横向粗糙度相对 truth 的偏差；
- state/boundary posterior calibration；
- profile、jump 和 identifiable-parameter interval calibration。

所有主模型与 baseline 的比较使用 parent 配对 bootstrap，并报告置信区间。

四个主门禁指标固定为：

- projected 5 m log-AI RMSE；
- boundary F1；
- segment IoU；
- decoded segment profile RMSE。

interface jump/polarity 是观测对齐诊断；单系数指标只用于判断 decoder 坐标的可识别性。二者均完整报告，但不替代上述四个主门禁。

baseline 按输出能力分工，不要求每个 baseline 在不具备的输出上参与比较：

**AI 增量价值**

- 主模型相对 full-LFM-only 的 projected 5 m log-AI RMSE 配对改善置信区间为正；
- 主模型相对 zone-linear anchor-only 的 projected 5 m log-AI RMSE 配对改善置信区间为正。

**地震贡献**

- 主模型相对 no-seismic control 在四个主门禁指标上的预先声明聚合结论为正；
- matched center-seismic shuffle 后，主模型相对 no-seismic 的结构、interface jump 或 profile 收益下降；
- parent-shuffle 后，地震贡献下降。

**横向贡献**

- lateral model 相对 single-trace 在 boundary F1、segment IoU 和横向粗糙度上的聚合结论为正；
- neighbor-shuffle 后横向收益下降；
- lateral model 不增加 pinchout false-bridging。

其余指标用于诊断和解释，不要求每一项都达到统计显著。聚合规则、指标方向和 bootstrap 单位必须在 geometry holdout 解封前写入 evaluation manifest。

进入阶段 2 需要同时满足：

- AI 增量价值、地震贡献和横向贡献三组门禁分别通过；
- dirty/clean segment count bias 的置信区间覆盖 0；
- dirty 输入不会系统性增加 false-positive 或 false-negative boundary；
- `non_seismic_supported` segment rate 不超过 calibration set 冻结上限；
- dirty 输入不会增加 high-confidence wrong segment；
- 证据变弱时 posterior 展宽，jump/profile 方差增大；
- 第一步的参数可观测性 decision manifest 已冻结，阶段门禁将完整 `c0/c1/c2` 解释为 `prior-selected latent` 和审计坐标；
- dirty holdout 上的 state/boundary posterior、jump/profile 区间和适用的参数区间完成校准；
- geometry holdout 不出现与 development validation 相反的结论。

如果模型在弱反射 dirty holdout 上通过系统性合并或过度切分 segment 获得更低误差，阶段 1 失败，不进入真实工区。

## 6. 阶段 2：真实工区冻结模型推理

预计生产代码量：900–1400 行。

阶段 2 只判断冻结的合成模型是否在真实工区保留可迁移信号。checkpoint、先验、normalization、augmentation profile、patch geometry 和证据分类阈值均在真实推理前冻结。

### 6.1 输入与展开顺序

输入使用：

- 真实深度域 seismic；
- TVDSS sample axis；
- 已解释 zone 顶底层位；
- 比例切片克里金 LFM，作为主输入；
- 趋势 LFM，作为敏感性输入；
- 冻结的 full model checkpoint；
- 与 full model 配对的冻结 no-seismic control checkpoint。

两个 checkpoint 使用相同的 model family、非 seismic 模块容量、parent split、初始化 seed、训练预算和 calibration 流程。no-seismic control 从训练开始就通过显式 modality-availability mask 跳过 seismic encoder 输入，只保留 LFM 和同一结构化 head；不能只在 full model 推理时临时把 seismic 清零来替代。

先运行固定剖面：

- 6 口可信井剖面；
- 3 口低质量井震标定诊断剖面；
- 至少 2 条不穿井的 blind section。

固定剖面必须同时运行 full model 和 no-seismic control。只有剖面门禁通过后，才使用完全相同的 full model checkpoint、标准化、LFM variant 和推理参数运行 `601 × 801 × 551` 全体积。首版不要求 no-seismic control 运行全体积。

剖面门禁失败时不启动全体积。

### 6.2 双方向横向推理

synthetic training parent 是一维 lateral section，不把它解释成二维训练样本。同一个一维横向模型在真实体上分别沿 inline 和 xline 方向运行。

双方向采用两遍推理。

第一遍只生成方向证据：

```text
inline patch → encode_patch() → inline DirectionalEvidence
xline patch  → encode_patch() → xline DirectionalEvidence

等权融合 calibrated emission/boundary potentials
→ 一次 fused exact HSMM
→ 唯一 fused state sequence 和 fused segments
```

第二遍只在 fused segments 上预测 profile：

```text
parameterize_segments(inline features, fused segments)
parameterize_segments(xline features, fused segments)
→ directional profile/parameter mixtures
```

两个方向各自的 MAP segments 不进入参数融合，也不按 segment index 对齐。

融合规则固定为：

- calibrated emission log potentials 等权平均；
- calibrated boundary log potentials 等权平均；
- 融合后每道只运行一次 exact HSMM；
- 两个方向使用同一个 state-conditioned parameter head；
- 参数 mean 使用等权 mixture mean；
- total variance 分解为 within-direction 与 between-direction；
- survey 边缘只有一个合法方向时使用该方向；
- 两个方向都无合法上下文时该位置无 prediction support。

方向参数 mixture 为：

```text
directional_mixture_mean
    = (μ_inline + μ_xline) / 2

within_direction_variance
    = (σ²_inline + σ²_xline) / 2

between_direction_variance
    = (μ_inline - μ_xline)² / 4

directional_total_variance
    = within_direction_variance + between_direction_variance
```

该 variance 是等权两分量 mixture 的 total variance，表示单方向不确定性与方向间分歧。inline/xline 来自同一 seismic 体和同一个模型，不能把它解释成独立证据融合后的 Bayesian posterior variance。

`inline/xline` 只用于几何寻址。相对距离使用实际米制坐标；`xline_step=4` 不进入横向卷积、attention 或 HSMM。

阶段 2 不对 state、segment boundary、参数或 AI 做额外横向平滑。

### 6.3 真实输出合同

每个真实位置输出：

- MAP state sequence；
- segment table；
- state/boundary posterior；
- interface polarity 与 signed interface jump；
- decoded segment profile 及其区间；
- zone-linear LFM anchor-relative `c0/c1/c2`，解释范围服从阶段 1 冻结的参数可观测性 decision manifest；
- `directional_mixture_mean`；
- `directional_total_variance`；
- `within_direction_variance`；
- `between_direction_variance`；
- decoded high-resolution log-AI；
- projected 5 m log-AI；
- `background_lfm_linear`；
- `full_lfm - background_lfm_linear`；
- inline/xline direction disagreement；
- `seismic_support_score`；
- `lfm_variant_sensitivity_score`；
- matched-seismic intervention 的原始诊断量；
- stitching weight、direction support 和 valid mask。

`seismic_support_score` 根据 full model 相对 matched center-seismic intervention 的结构、interface jump 和 profile 稳定性，在 synthetic dirty calibration set 上标定。`lfm_variant_sensitivity_score` 独立描述比例切片克里金 LFM 与趋势 LFM 两次推理之间的变化；两个分数不能合并为单一“先验支配度”。

使用 synthetic dirty holdout 冻结的阈值，将每个 segment 保守标记为：

```text
seismic_supported
mixed_support
non_seismic_supported
```

`non_seismic_supported` 只表示当前干预没有证明 seismic 是主要支持来源。它可能由完整 LFM、HSMM prior、模型饱和或干预强度不足造成，不能解释为 `prior_dominated`。只有未来增加并校准显式 prior intervention 后，才允许发布 `prior_dominated` 标签。

高分辨率结果始终可以输出，但 `mixed_support` 和 `non_seismic_supported` segment 不解释为地震直接分辨的薄层。

比例切片克里金 LFM 的结果是主交付。趋势 LFM 使用同一 checkpoint 完整重跑，用于标记在两套输入下均稳定存在的 `LFM-robust` segment。两次推理各自使用对应完整 LFM 的 zone-linear anchor；LFM sensitivity 不通过修改参数或后处理主结果实现。

### 6.4 阶段 2 评价与门禁

固定检查：

- 真实输入是否落在训练和 augmentation 的统计支持范围；
- 所有有效位置是否具有 prediction support；
- 是否存在 patch seam 或方向性条带；
- segment 密度、duration、jump/profile 及适用的参数分布是否落在 dirty synthetic 校准范围；
- inline/xline direction disagreement 是否异常；
- directional variance 分解是否有限且闭合；
- `non_seismic_supported` 是否主要集中在弱反射或 OOD 区；
- 两套 LFM 下的 segment 稳定性。

井评价分为完整 AI 和网络增量：

- 6 口可信井参与聚合；
- 固定剖面上的可信井同时评价 full model 与配对 no-seismic control；
- 完整 AI 与井曲线的相关性和误差照常报告；
- 完整 LFM 与 `background_lfm_linear` 分别作为背景参考报告；
- 主要门禁比较 `prediction - background_lfm_linear` 与井上对应频带的 `logAI - background_lfm_linear`；
- full model 相对 no-seismic control 的配对增量改善必须为正，用于隔离真实 seismic 的贡献；
- 可信井的中位表现必须优于 zone-linear anchor-only；
- 多数可信井不能相对 zone-linear anchor-only 退化；
- 3 口低质量标定井只做逐井诊断，不进入聚合门禁。

full model 相对 zone-linear anchor-only 的改善回答网络整体是否增加有效信息；full model 相对 no-seismic control 的改善回答增加的信息是否来自 seismic。两项门禁必须分别报告。真实井没有可靠 segment truth 时，以 projected log-AI RMSE 和井上对应频带增量误差作为主门禁；boundary、segment 和参数指标只在具有相应可信解释的井段报告。

剖面门禁通过还要求：

- blind section 没有明显拼接缝、方向性条带或 jump/profile 爆炸；
- 高分辨率 segment 不因真实波形变弱而系统性粗化；
- direction disagreement、LFM sensitivity 和 `non_seismic_supported` 比例均可定位并输出；
- 主结果与敏感性结果使用同一冻结模型和推理合同。

门禁失败时，根据 OOD、方向分歧、LFM 敏感性、`non_seismic_supported` 比例、full/no-seismic 差异和井上增量指标定位失败来源。本阶段不通过真实标签调参，不进行无标签联合微调。

## 7. 实现验证合同

未来 implementation 至少覆盖：

### 7.1 LFM anchor 与 decoder

- 普通多采样点 segment 的离散换基 parity；
- singleton canonical parameterization；
- zone 首段与末段 endpoint；
- pinchout 与 zero-sample segment；
- clipping 前后 parity；
- NumPy/Torch 换基和 decoder parity；
- 每个 segment 的换基 residual gate；
- rank-1、rank-2、rank-3 和病态 basis 的 identifiability 分类；
- `parameter_supervision_valid` 只在 rank-3、condition threshold 内且无 clipping 时成立；
- 参数监督失效时 profile supervision 仍然保留；
- `full_lfm - background_lfm_linear` 与网络增量字段不混写。

### 7.2 HSMM 与参数 head

- signed interface jump 的方向、endpoint 和 valid-mask 约定；
- boundary polarity 与 jump magnitude 的 supervision mask 分离；
- top/bottom inside/outside feature 不越过 zone 或 observation support；
- basis-aware moments 对 `c1` 符号翻转和 `c0/c2` 分离具有响应；
- boundary-aware head 与 frozen mean-pooling baseline 使用相同 parent split 配对比较；
- 小序列 brute-force MAP 对照；
- forward-backward marginal 对照及归一化；
- zone-fraction duration 在不同采样间隔下产生一致的物理先验；
- truth+jitter 参数训练不需要 predicted/truth segment matching；
- state-conditioned parameter NLL 和单系数指标只统计 identifiable subset；
- rank 不足 segment 只产生 decoded profile loss；
- predicted segmentation 只进入端到端验证。

### 7.3 横向与双方向推理

- cross-lateral event identity 和顺序验证；
- birth、death、pinchout 和 topology mask；
- neighbor shuffle 只破坏邻道对应；
- inline/xline segmentation 不同时，两遍推理仍产生唯一 fused parameter table；
- parameter head 在 externally supplied fused extents 上构造相同的 boundary-aware、basis-aware evidence；
- directional mixture variance 分解恒等式；
- `xline_step=4` 不改变米制 patch 几何。

### 7.4 Split、dirty gate 与 evidence

- training、tuning validation、calibration 和 geometry holdout parent 集合两两不相交；
- dirty identities 在 calibration 前冻结；
- matched center-seismic shuffle 保持指定的 zone、振幅和频谱统计；
- parent shuffle、center shuffle 和 neighbor shuffle 分别只破坏目标信息；
- zero seismic 不进入正式 evidence classification；
- clean/dirty segment count bias 双向检查；
- false-positive/false-negative boundary rate；
- `non_seismic_supported` 和 high-confidence wrong segment gate；
- full/no-seismic checkpoint 的配对训练合同；
- 阶段 1 baseline 按 AI、地震和横向指标分工；
- 阶段 2 固定剖面同时生成 full/no-seismic 井上配对报告；
- 没有 prior intervention 时不发布 `prior_dominated`。

## 8. 当前停止条件

当前工作顺序固定为：

```text
阶段 1 前置 teacher-forcing audit 已冻结
→ 第一步：boundary-aware profile head + 参数可观测性 decision 已冻结
→ 第二步：单道 evidence + exact HSMM MAP/marginals
→ 第三步：lateral patch + augmentation + calibration + geometry holdout
→ 冻结 full/no-seismic 真实剖面推理
→ 剖面门禁
→ 冻结全体积推理
```

阶段 1 门禁未通过时不进入真实工区。真实剖面门禁未通过时不运行全体积。

后续是否加入可微 forward、物理排序、二维 synthetic training、井监督 adaptation 或真实无标签 adaptation，只根据阶段 1 和阶段 2 的结果重新规划，不预先进入第一版 implementation。
