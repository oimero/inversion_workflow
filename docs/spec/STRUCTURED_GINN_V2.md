# Structured GINN V2 最终实施规格

## 1. 项目目标

Structured GINN V2 是一个：

> 地震带限证据指导下、具有横向连续性的高分辨率地质生成器。

完整计算链为：

```text
seismic + full LFM + physical lateral geometry
→ BandlimitedEvidence
→ producer-calibrated structured prior
→ section-level ordered EventFields
→ K 个合法 high-resolution realizations
→ representative member + ensemble summary + evidence diagnostics
```

三类信息具有不同职责：

- LFM 提供 zone 内低频背景；
- 地震提供当前频带与调谐尺度能够支持的阻抗增量、反射极性和状态占据证据；
- 由真实井标定的 producer prior 补全亚调谐事件、持续时间、profile 和横向演化。

“高分辨率”表示生成器能够输出高分辨率、满足结构合同的条件 realization。它不表示地震
逐层分辨了全部微层。弱地震证据下，多个高分辨率 realization 可以具有近似相同的带限
响应；这种非唯一性必须由 ensemble 保留，而不是由一个过度确定的预测隐藏。

第一版的科学问题是：

> 在 LFM 负责低频背景的前提下，地震能否为井标定结构先验提供稳定的带限条件，使生成器
> 输出横向连续的高分辨率地质 realization，并在合成基准与真实井上证明地震带来了增量
> 信息？

首版范围包括合成监督、冻结 zero-shot 真实剖面和后续二维体扩展。可微正演 loss、物理
排序、井监督和真实无标签适配属于 zero-shot 结果之后的独立研究决策。

## 2. 已知证据与固定科学边界

### 2.1 既有实验已经证明的内容

冻结实验支持以下结论：

- full model 相对 no-seismic 能改善 projected log-AI、state 和反射极性；
- matched/within-parent seismic shuffle 会破坏上述改善，地震没有被 LFM 完全取代；
- exact micro-boundary、精确 segment count 和单个 `c0/c1/c2` 不是可靠的地震直接监督量；
- projected AI、tuning-scale evidence 和 decoded profile 是更稳定的训练与评价对象；
- profile coefficient 只在离散 basis 可识别时适合单独评价，短段应评价 decoded profile；
- 逐道生成候选再做横向匹配，只能减少局部跳变，不能建立稳定的跨道事件身份；
- ensemble mean 的平滑来自平均，不证明任一合法 member 具有横向连续性。

详细数值和图件保存在：

- `note/summary/final_audit/20260803_structured_ginn_v2_observable_hsmm/`；
- `note/summary/final_audit/20260804_structured_ginn_v2_profiles_ensemble/`；
- `note/summary/final_audit/20260806_structured_ginn_v2_event_track_generation/`。

### 2.2 地震峰数不是地质事件密度

同样厚度的 zone 中，合成地震可能显示十余个波峰波谷，真实地震只显示四五个；这首先
说明真实观测压低或混叠了更多反射，并不能直接推出真实地下事件更少。

地质密度正式使用：

- event count per zone；
- duration as zone fraction；
- event survival/birth/death over physical lateral distance。

波峰数只属于观测域诊断。生成器不能把“可见峰更少”直接翻译成“高分辨率事件更少”。
在弱证据下，事件密度应回归同一真实井标定的 producer prior，同时扩大条件不确定性。

第一版不设置独立的 `density_multiplier`，也不建立第二套拍脑袋的真实工区微结构先验。
若 zero-shot 井结果证明 producer prior 的事件密度系统失配，再以井证据重新标定 producer
prior，并将其作为新的版本化科学合同。

### 2.3 投影塌缩不等于同状态续生

当前 producer 的高分辨率状态转移矩阵对角线为零。原始 event sequence 因此没有可供训练
的独立同状态续生过程。

对 24 个 canonical parents 的直接抽查得到：高分辨率相邻同状态 event edge 为 0，而
model grid 上“dominant state 相同、dominant object 不同”的 edge 为 2750。这个数量差异
直接证明两种语义不能混用。

有限支撑投影到 model grid 后，薄的中间状态可能不再成为任何 model sample 的 dominant
state。例如：

```text
high resolution: state 0 / object 19
                 state 1 / object 20  ← 薄层
                 state 0 / object 21

model grid:      dominant state 0 / object 19
                 dominant state 0 / object 21
```

这是 `projection_collapse`：中间事件仍存在于高分辨率 truth，只是在 5 m 或对应时间采样
网格上被混合。它不是同状态续生，也不是应当拒绝的坏样本。此类样本正是本项目希望表达的
亚调谐非唯一性。

固定合同为：

- EventField truth 只由 high-resolution `(zone_id, object_id)` 构造；
- model-grid state 使用三状态 fraction/occupancy，不使用 argmax state 反推事件；
- model-grid object change 只作 projection-collapse 诊断，不作 renewal label；
- 第一版 prior 遵守 producer 的异状态转移合同；
- topology 使非相邻全局事件在局部接触时，保留其全局事件身份并单独标记；
- 真实同状态续生只有在 producer 和高分辨率语料都发布显式监督后才成为研究扩展。

当前 canonical corpus 不因投影塌缩重新生成，也不拒绝这些 parent。

### 2.4 真实观测覆盖

sim-to-real 在本项目中指真实观测扰动覆盖，不指把合成地震改造成与真实地震无法区分。
处理范围是：

- wavelet 频带、相位和小幅 shift；
- vertical static；
- 正增益与平滑振幅衰减；
- white、colored 和 coherent noise；
- 弱反射压低及可见峰数量下降。

这些扰动保持同一 synthetic truth。它们用于覆盖 `p(x | z)` 的合理变化，不修改地质标签，
也不宣称解决合成地质先验与真实地下之间的全部差异。

## 3. 单一公开生成接口

### 3.1 深模块边界

包根只暴露：

```python
ConditionalGenerator
ObservationTile
GenerationPolicy
BandlimitedEvidence
StructuredEnsemble
train_generator(...)
evaluate_generator(...)
infer_section(...)
```

核心接口为：

```python
ensemble = generator.generate(observation_tile, generation_policy)
```

调用者不运行 HSMM、不匹配逐道 segment、不平滑 coefficient、不拼接成员，也不负责选择
代表解。带限证据、结构先验、EventField 生成、栅格化和 ensemble 选择全部由
`ConditionalGenerator` 内部拥有。

### 3.2 ObservationTile

```text
ObservationTile
├── model_axis / highres_axis
├── seismic[lateral, model_sample]
├── full_lfm[lateral, model_sample]
├── observed_valid
├── zone_top / zone_bottom
├── lateral_m
├── optional x_m / y_m
└── identity
```

输入必须携带 sample domain、unit 和 depth basis。横向网络只使用物理米制距离；
`inline/xline` 只承担寻址，工区 `xline_step=4` 不表示四倍或单位横向距离。

### 3.3 BandlimitedEvidence

公开证据只包含地震频带可以合理监督的量：

```text
BandlimitedEvidence
├── background_lfm_linear
├── projected_log_ai_increment_mean / scale
├── signed_reflectivity_mean / scale
├── state_fraction[..., 3]
├── local_tuning_scale
├── model/highres support
├── model/highres axes
└── physical lateral geometry
```

`state_fraction` 表示一个 model-grid 支持窗口内三种高分辨率状态的占据比例。它不是硬状态，
也不是事件边界。`signed_reflectivity` 使用与 projection/tuning operator 一致的带限反射目标，
不把每个微边界都标成独立可见界面。

证据网络的 hidden feature 不属于生成器的隐式捷径。结构生成器只消费已发布的
`BandlimitedEvidence`，从而可以独立审计“网络看见了什么”和“先验补了什么”。

### 3.4 ProducerPrior

`ProducerPrior` 是 Synthoseis-lite producer 与 GINN 生成器共享的单一版本化合同：

```text
ProducerPrior
├── zone-conditioned initial state
├── zero-diagonal state transition
├── duration distribution in zone fraction
├── event/profile distribution
├── identifiable coefficient statistics
├── lateral correlation and survival statistics
├── topology family contract
└── calibration provenance
```

它由真实井标定结果、合成配置和 producer 科学合同共同发布。GINN consumer 不重新拟合一套
含义不同的先验，也不从 model-grid argmax state 估计转移。

producer 和 generator 使用同一个 prior identity。fingerprint 只记录 provenance 和稳定随机
identity；consumer 不重算上游文件 SHA，也不因已保存 fingerprint 不一致拒绝输入。

### 3.5 EventField

高分辨率结构使用有序的 `EventField`：

```text
EventField
├── zone_id
├── event_id
├── state_id
├── presence over physical locations
├── duration_fraction over physical locations
├── c0/c1/c2 or decoded-profile latent
├── identifiability mask
└── topology mask
```

一维剖面上的 EventField 是 EventTrack；二维体上的 EventField 是 EventSurface。两者共享：

- 事件按垂向顺序具有全局身份；
- 每个位置的有效 duration 非负且总和为一；
- top/bottom 由 duration 累积得到，边界不会交叉；
- birth、death 和 pinchout 通过 presence/topology 表达；
- profile 和 duration 在物理坐标上连续，但不跨 topology transition 强制平滑；
- 每个 realization 先生成完整 EventField，再栅格化为各位置的 segment。

横向连续性因此是 latent representation 的性质，而不是逐道候选生成后的修补结果。

### 3.6 StructuredEnsemble

```text
StructuredEnsemble
├── BandlimitedEvidence
├── K 个完整 EventField realizations
├── K 个 high-resolution / model-grid log-AI
├── state occupancy / event presence summaries
├── high-resolution / projected mean、std、coverage
├── representative realization
└── evidence、prior、topology 与 support diagnostics
```

代表解必须是 K 个完整 realization 中的真实成员。它按完整 section 的冻结条件分数选择，
不逐道挑选，也不把逐点 ensemble mean 当作合法地质 realization。

## 4. 数据、坐标与当前基础

### 4.1 Canonical corpus V2

当前 depth/TVDSS corpus 可继续作为正式开发语料：

- 2400 个 25 道 short-patch parents；
- 训练、调参与 calibration 分别为 1680、360、360；
- none、wedge、pinchout 在每个 split 内均衡；
- 47 个 full-section parents，仅用于最终横向门禁；
- 其中 24 个 IID sections，23 个 combination holdout sections；
- 每个 parent 内 `(zone_id, object_id)` 是稳定的 high-resolution event identity；
- artifact 只有一套 canonical HDF5、索引和 manifest；
- clean seismic 持久化，dirty observation 在线生成。

短 patch 中的 25 道是一个联合训练样本。21 道是模型的完整上下文窗口；两侧道用于显式
mask 与边缘行为。full sections 不参加参数训练、模型选择或不确定性 calibration。

样本量只按独立 parent 报告，不把同一 parent 派生的 25 道或 dirty views 计成新的独立样本。
阶段 1 在固定模型与训练预算下发布 420、840、1680 training parents 的学习曲线。只有验证
误差仍随 parent 数稳定下降，且失败不能由模型容量或标签退化解释时，才扩充 synthetic
corpus；47 个 full sections 不转入训练集。

现有语料的 47/48 section 配额缺口是已发布 warning，不影响开发。新增语料只由明确的科学
缺口触发，不为追求整数配额重新运行数小时 benchmark。

### 4.2 投影语义补充

训练 adapter 必须从 high-resolution truth 确定性生成：

```text
state_fraction_model
boundary_mass_model             # diagnostic only
dominant_object_id_model        # diagnostic only
projection_collapse_count
projection_collapse_fraction
hidden_transition_count
```

这些字段优先由当前 HDF5 和 object tables 在线派生，使现有 GB 级结果继续可用。未来 producer
版本可以把相同字段写入 canonical HDF5，但不能增加第二套 truth 旁路。

### 4.3 时深对称

所有垂向语义来自 `SampleAxis`：

```text
depth: domain=depth, unit=m, depth_basis=tvdss
time:  domain=time,  unit=s, depth_basis=null
```

统一规则为：

- metrics 使用 sample 数、axis unit 或 tuning-scale fraction；
- augmentation 使用 `vertical_static_samples`；
- tuning scale 在 time 中为 `1 / (2 f_dom)`，在 depth 中为 `Vp / (4 f_dom)`；
- depth 的 `Vp` 是显式域适配输入；
- time/depth 共享 writer、reader、target、loss、generation 和 diagnostic interface；
- 正式 benchmark 当前运行 depth/TVDSS，time fixture 负责接口对称性门禁。

## 5. 阶段 0：补齐 producer prior 与 projection 合同

### 5.1 实施内容

仓库清理已经完成，`src/ginn_v2` 当前保持 10 个深模块。阶段 0 的剩余工作集中在科学合同：

1. 在 `cup.synthetic` 发布 `ProducerPrior` writer/reader；
2. 允许 transition 矩阵合法地包含零对角线；
3. 从 high-resolution object catalog 构造 EventField truth；
4. 增加 state fraction、boundary mass 和 projection-collapse 派生器；
5. 将 producer decoder、projection 和 Oracle 保持在 `cup.synthetic`；
6. 使 GINN Torch decoder 与同一 producer contract 做 parity；
7. 将当前 evidence seam 收敛为公开的 `BandlimitedEvidence` 合同；
8. 为相同 truth 构造 clean、普通 dirty 和 peak-poor dirty 配对 fixture。

当前 corpus 已包含完成这些工作的必要 high-resolution identity 和 profile 字段，因此阶段 0
不要求重跑 Synthoseis-lite。只有正式 producer schema 被改到无法由现有 artifact 无歧义派生时，
才发布新 corpus 版本。

### 5.2 门禁

- EventField truth 栅格化后严格重建 high-resolution log-AI；
- finite-support projection 与 canonical model-grid truth 一致；
- 全量 parent 中 high-resolution event identity、顺序、state 和 topology 通过审计；
- true high-resolution same-state adjacency 与 projection-collapse 分开计数；
- model-grid argmax state 不进入 EventField truth；
- `ProducerPrior` 与 producer calibration/config 的数值合同一致；
- clean/dirty/peak-poor 三者共享同一 truth identity；
- time/depth fixture 使用同一 adapter 和 public types；
- SHA fingerprint mismatch 不构成 consumer 拒绝条件。

阶段 0 通过后冻结 prior、target 与 split manifest；后续阶段不得在训练结果不理想时静默改
标签定义。

## 6. 阶段 1：学习 BandlimitedEvidence

### 6.1 标签进入训练前的证据门禁

任何新监督 head 必须先通过固定 target preflight：

1. 从 high-resolution truth 和冻结 operator 确定性产生；
2. 轴、support、数值范围和 mask 闭合；
3. 标签分布不退化为全零、全一或全 zone 高活跃；
4. 简单常数、LFM-only 和局部滤波 baseline 被记录；
5. 固定 mini-corpus 可以过拟合；
6. matched seismic shuffle 能破坏预期可观测信息；
7. clean/dirty 配对不会改变 truth target。

preflight 是 `train_generator()` 的固定前置步骤，不建立另一套长期诊断旁路。失败时在训练前
停止，而不是训练数小时后再解释 head 为什么没有信息。

### 6.2 模型

证据网络读取完整 short patch，并为所有有效道输出带限证据。模型由：

- 逐道垂向 encoder；
- 使用 `lateral_m`、相对距离和显式 mask 的 lateral mixer；
- projected increment、signed reflectivity、state fraction 和 scale heads；
- support-aware uncertainty calibration。

固定训练三个同预算模型：

- full：seismic + full LFM；
- no-seismic：full LFM；
- single-trace：中心道 seismic + full LFM。

训练目标只作用于带限量：

- projected log-AI increment Gaussian NLL/RMSE；
- signed reflectivity NLL/相关性/极性一致率；
- state fraction proper score；
- scale calibration 与 coverage。

exact micro-boundary、event count 和单个 profile coefficient 作为后续结构审计，不作为本阶段
的主要证据 head。

### 6.3 观测扰动课程

训练同时保留 clean 和在线 dirty 配对。真实观测 profile 从真实地震、冻结子波和可信井震
标定中提取振幅、频谱、相位、shift 与噪声支持范围。

必须包含一组 peak-poor 配对：逐步压低弱反射、收窄有效频带并增加相干噪声，使可见峰数
下降，但 high-resolution truth 和 ProducerPrior identity 保持不变。该配对专门检查模型是否
错误地把观测复杂度当成地质事件密度。

### 6.4 门禁

- full 相对 no-seismic 在 projected increment 和 signed reflectivity 上具有正的 parent-paired
  改善；
- full 相对 full-LFM 与 zone-linear anchor 具有 AI 增量价值；
- full 的 state-fraction proper score 优于 no-seismic；
- matched center-seismic shuffle 和 parent shuffle 破坏地震收益；
- neighbor shuffle 破坏 lateral model 相对 single-trace 的收益；
- dirty 与 peak-poor 输入增加合理 uncertainty，不制造系统性虚假反射；
- evidence 输出在 projection-collapse 与普通样本上分别报告；
- calibration split 冻结 scale/temperature，full sections 不参与调参。

本阶段只证明“地震提供了什么带限信息”，不生成高分辨率事件。

## 7. 阶段 2：section-level EventTrack 条件生成器

### 7.1 生成语义

EventTrack decoder 消费 `BandlimitedEvidence` 和冻结 `ProducerPrior`，一次生成一个 zone 的
完整横向事件系统：

```text
ProducerPrior renewal/duration process
+ bounded bandlimited evidence potentials
→ ordered event identities and states
→ lateral presence / birth / death / pinchout
→ duration-fraction fields
→ profile fields
→ cumulative non-crossing boundaries
→ high-resolution section
```

事件数量由 zone-fraction duration/renewal process 自然产生。模型不设置“一个波峰对应几个
事件”的 head，也不使用可见峰数作为 stop condition。地震 evidence 只能通过冻结范围内的
条件 potential 调整 prior，弱 evidence 下回归 ProducerPrior。

event state 遵守 producer transition。第一版不采样独立同状态续生。projection collapse 通过
隐藏的异状态中间事件表达，而不是生成两个相邻同状态段。

duration 和 profile 是 event-level 横向场。其随机性以 event identity 与物理坐标为单位；
每道独立采样再匹配不属于合法实现。

### 7.2 训练

EventTrack truth 直接来自 high-resolution object catalog，训练使用 teacher forcing 和逐步减少
teacher forcing 的课程：

- ordered event/state likelihood；
- lateral presence、birth/death 与 topology；
- duration simplex 和 thickness field；
- state/duration-conditioned profile distribution；
- identifiable coefficient likelihood；
- 所有 event 的 decoded-profile loss；
- rasterized high-resolution reconstruction；
- projected/tuning-scale consistency with BandlimitedEvidence。

短段或病态 basis 不进入单系数门禁。它们通过 decoded profile 与 ensemble coverage 监督。
梯度不需要穿过离散代表解选择。

### 7.3 横向连续性合同

横向连续性至少满足：

- 同一 event 在邻道保持 event identity；
- thickness 和 profile 随米制距离连续变化；
- event 可以在允许的 topology 位置 birth、death 或 pinchout；
- state、duration 和 profile 不跨 topology mask 被强制平滑；
- direction reversal 不改变统计结果；
- section 宽度变化不改变局部生成语义。

评价以 EventTrack 为主，不以像素梯度代替。固定指标包括：

- event matched/survival fraction；
- thickness log-ratio 与沿轨迹 variation；
- profile mean/RMS jump；
- birth/death localization；
- pinchout false bridging；
- high-resolution 与 projected reconstruction；
- projection-collapse recovery by track identity。

固定图件展示 seismic、truth、predicted EventTracks、high-resolution AI、projected AI 和
topology mask。横向门禁必须同时通过数值与图件审查。

### 7.4 deterministic 门禁

- truth EventTracks 经同一 rasterizer 完整闭环；
- 固定 mini-corpus 可以过拟合 event identity、duration 和 profile；
- deterministic/MAP 结果优于 prior-only EventTracks；
- full 相对 no-seismic 改善 projected AI 与可识别 event/profile 指标；
- lateral model 相对 single-trace 改善 event survival 与 thickness/profile continuity；
- wedge/pinchout 不增加 false bridging；
- peak-poor 输入不造成 event count 系统性坍缩；
- weak evidence 区域的结果回归 prior，而不是输出重复的确定性微段。

deterministic 门禁通过后，才实现 K-member sampling。

## 8. 阶段 3：联合 ensemble 与 full-section 合成门禁

### 8.1 Ensemble sampling

默认 `K=16`。一个 member 的采样单位是完整 section EventTracks：

```text
sample ordered events once per zone
→ sample each event's lateral presence and duration field
→ sample each event's lateral profile field
→ condition on BandlimitedEvidence
→ rasterize all traces together
→ decode one complete section realization
```

随机 identity 由 run、parent/section、zone、event、member 和物理坐标共同决定。tile 大小、
执行顺序和 GPU batch 不改变结果。每个 member 都必须满足 EventField、axis、support、duration
和 topology 合同。

representative 以完整 section 的冻结 joint conditional score 从 K 个 member 中选择。ensemble
mean 只作为统计摘要。

### 8.2 不确定性与证据归因

calibration split 冻结：

- evidence scale；
- event count/duration dispersion；
- profile variance；
- high-resolution/projected coverage；
- full 与 no-seismic 的 paired sensitivity threshold。

输出使用两个独立诊断量：

- `seismic_conditioning_sensitivity`：full 与 matched/no-seismic 条件结果的差异；
- `sub_tuning_prior_uncertainty`：在带限证据近似不变时，高分辨率 members 的分歧。

它们不被压缩成“地震支持/先验支配”的过强单标签。亚调谐事件可以输出，但必须伴随其
ensemble 不确定性和带限不可区分性。

### 8.3 Full-section 门禁

calibration 完成后只运行一次当前 47-parent full-section benchmark。检查：

- K members 和 representative 都具有连续 event identity；
- ensemble mean 的平滑不掩盖 member 跳变；
- event count、duration、state occupancy 和 profile distribution coverage；
- high-resolution/projected CRPS 与 coverage；
- event survival、thickness/profile variation 和 topology；
- projection-collapse 子集的 hidden-event coverage；
- clean、dirty、peak-poor 的配对稳定性；
- peak-poor 条件下可见峰数下降，而 posterior event density 不系统性欠分；
- matched seismic shuffle 破坏地震收益，neighbor shuffle 破坏横向收益；
- IID 与 combination holdout 结论一致；
- tile/halo/stitching 与 random identity invariance。

每个 geometry family 固定输出 truth、representative、两个固定 members、ensemble summary、
EventTracks 和 topology 图件。合法 member 的视觉连续性是正式验收项。

## 9. 阶段 4：真实工区冻结 zero-shot 剖面

阶段 4 固定运行：

- 6 口可信井剖面；
- 3 口低质量井震标定诊断剖面；
- 至少 2 条 blind sections；
- full 与 no-seismic 两个冻结 checkpoint；
- 比例切片克里金主 LFM 与趋势 LFM 敏感性对照。

每条剖面输出：

- BandlimitedEvidence 与统计支持检查；
- K 个完整 EventTrack realizations；
- representative high-resolution AI 与 event table；
- ensemble high-resolution/projected mean、std 和 coverage；
- event presence、duration 和 profile summary；
- seismic conditioning sensitivity；
- sub-tuning prior uncertainty；
- LFM variant sensitivity；
- support、topology 和 stitching mask。

可信井门禁为：

- full 相对 no-seismic 在井上对应频带的配对改善为正；
- full 相对 full-LFM 和 zone-linear anchor 有整体增量价值；
- high-resolution ensemble 对井曲线具有合理 coverage；
- representative 在多数可信井上不相对 LFM 退化；
- 井上 event density 与 ProducerPrior 的失配被单独报告；
- blind sections 不出现 event identity 跳变、随机块状纹理或 seam。

3 口低质量井只作逐井诊断。若 full 不优于 no-seismic，结论是地震条件没有成功迁移；若
带限指标通过而高分辨率井 coverage 失败，结论是 producer prior 或 profile distribution 需要
重新标定。两类失败不能互相替代解释。

## 10. 阶段 5：二维 EventSurface 与全体积

真实 1D 剖面通过后，使用同一 EventField 语义扩展二维空间：

```text
inline/xline observations
→ one fused BandlimitedEvidence field
→ one ordered EventSurface generator
→ one set of 2D event identities
→ volume realizations
```

inline 和 xline 可以分别编码带限证据，但在微结构生成前融合。禁止分别生成两套 EventTracks
再按 segment 对齐或平均。

二维实施前，Synthoseis-lite 增加小型 2D EventSurface fixture 和 holdout，用于验证 surface
ordering、birth/death、pinchout、tile/halo 和方向不变性。它复用同一 ProducerPrior，不建立
新的地质语义。

全体积采用可重生成的流式 ensemble：第一遍累计 summary 与全局代表分数，第二遍按选定
member identity 重生成代表解。chunk 顺序不改变结果。

全体积只在以下条件满足后发布：

- 2D synthetic EventSurface 门禁通过；
- 真实固定剖面门禁通过；
- inline/xline evidence disagreement 在 calibration 支持范围内；
- 体内没有方向性条带、tile seam、event-surface 断裂或 false bridging；
- 局部重生成与全体积结果身份一致。

## 11. 风险与停止条件

### 11.1 P0 风险

**观测复杂度泄漏为事件密度。** 模型可能学到“峰少即层少”。peak-poor paired gate 若失败，
停止 EventTrack 训练并修正 evidence/augmentation，不通过调高结构 loss 掩盖。

**横向连续性只存在于 ensemble mean。** 任一 member 仍逐道跳变时，停止 full-section 发布并
修正 EventField decoder，不增加事后平滑或候选匹配。

**ProducerPrior 与真实井失配。** real well high-resolution coverage 系统失败时，重新审计井
标定、duration 和 profile prior；不把失败归咎于地震噪声后继续全体积。

**LFM 取代地震。** full 不优于 no-seismic 时，zero-shot 地震增量门禁失败。

### 11.2 P1 风险

- none/wedge/pinchout 不能覆盖断层、盐边界等真实 topology；
- 25 道短 patch 对长距离事件生存的监督有限；
- K=16 可能低估多峰 posterior；
- 一维 EventTrack 成功不能自动证明二维 EventSurface 成功；
- coefficient 看似准确可能来自 producer 偏好，而非地震可辨识性；
-真实输入可能落在 observation profile 的统计支持之外。

这些风险必须在对应 gate 中形成可见报告。报告缺失本身视为门禁未完成。

## 12. 运行与发布纪律

- schema、status、ID、domain、unit、axis、shape、dtype、mask 和 geometry 在长任务前严格
  preflight；
- 标签退化、truth closure 和 mini-corpus overfit 在正式训练前完成；
- 每个 epoch 原子发布 `last` 与 `best` checkpoint；
- 日志按固定 batch/parent 频率报告进度和剩余规模；
- 单个随机 realization 或诊断图失败写入 warning，并发布已完成结果；
- producer Oracle 的数值边界告警写入 report，不使数小时生成在末尾丢失；
- fingerprint 记录 provenance，不作为 consumer equality gate；
- full-section、真实剖面和全体积各有独立输出目录与 manifest；
- 所有科学阈值在 calibration split 冻结，geometry holdout 和真实井不得用于静默调参。

## 13. 当前状态与下一步

当前已经完成：

- canonical corpus V2：2400 个 short patches 和 47 个 full sections；
- producer Oracle、decoder/projection parity 与 time/depth smoke；
- SHA consumer gate 清理；
- `src/ginn_v2` 历史实验链清理，包收敛为 10 个深模块；
- `ObservationTile`、证据 seam、EventTrack 基础类型和 `ConditionalGenerator` 外壳；
- 全量 parent event identity/topology 审计；
- 既有带限 evidence、profile、ensemble 和横向失败路线的冻结审计。

当前尚未完成的是新的 section-level EventField generator。实施顺序固定为：

```text
ProducerPrior + projection-collapse contract
→ BandlimitedEvidence 正式训练与 controls
→ deterministic EventTrack generator
→ K-member section ensemble
→ real zero-shot sections
→ 2D EventSurface / full volume
```

下一次代码实施从阶段 0 的 `ProducerPrior` 与 projection target 开始。当前 corpus 继续使用，
无需因 model-grid 同状态现象重跑合成 benchmark。
