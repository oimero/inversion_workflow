# Structured GINN V2 确定性实施规格

## 1. 项目目标

Structured GINN V2 的目标是：

> 地震带限证据指导下、具有横向连续性的高分辨率地质生成器。

第一版采用确定性条件生成：

```text
seismic + full LFM
→ ObservableEvidence
→ evidence-guided renewal + deterministic exact HSMM MAP
→ macro state bodies + same-state sub-resolution renewals
→ canonical deterministic segment profile
→ high-resolution log-AI
```

LFM 负责 zone 内低频背景；地震网络恢复当前频带和调谐尺度能够约束的增量、状态与
有符号反射证据。结构解码先用反射证据定位宏观地质体变化，再用显式亚调谐先验
填充高分辨率细节。状态持续与 segment renewal 是两个独立变量：renewal 不强制状态变化。
最终交付是一个完整、合法、可复现的结构化预测。

高分辨率表示生成器能够依据观测与井标定先验构造细网格地质模型。它不表示地震逐层
分辨了所有微层。每个预测同时发布证据强度、方向分歧和先验敏感性，区分地震支持的
带限结构与先验补全的亚调谐细节。

当前 checkpoint 是逐道模型，同一剖面上各道共享 LFM、层位、HSMM prior 和确定性
解码合同。先在真实固定剖面上验证 zero-shot 可行性，再根据真实失败形式决定是否
训练一维横向模型。

## 2. 已有证据与固定决定

### 2.1 地震具有可验证的增量信息

冻结的 truth-segmentation、boundary-aware 和 predicted-segmentation 实验表明，full model
相对 no-seismic 能改善 projected log-AI、high-resolution log-AI、state 和反射极性。
within-parent seismic shuffle 会稳定破坏这些改善。

可进入正式 interface 的监督目标为：

- projected/model-grid log-AI increment；
- state emission 或 state log potential；
- 有符号 reflectivity/interface jump；
- decoded segment profile 和 high-resolution log-AI。

目标必须先通过无训练数据审计、tiny overfit 和 full/no-seismic/shuffle 信息探针。全量训练
用于估计能力，不承担发现退化标签的职责。

### 2.2 微边界主要由结构先验决定

boundary observability audit 表明，合成语料中的边界间距远小于调谐尺度，raw boundary
score 接近随机排序。exact micro-boundary、精确 segment count 和单个 `c0/c1/c2` 不能作为
地震直接分辨能力的主门禁。

正式评价分为两层：

- 带限层：projected AI、state occupancy、signed reflectivity 和 tuning-scale structure；
- 高分辨率层：合法性、井标定先验一致性、profile 重建和真实井 coverage。

高分辨率结果可以优于 LFM，但不能仅凭合成微边界指标宣称真实地震分辨了薄层。

### 2.3 横向随机候选路线的结论

完整 section 实验显示，逐道随机路径和随机 presence 会把连续事件切成局部轨道：

- 真值相邻道事件数跳变为 `0.0054`，预测为 `0.6135`；
- 真值轨道邻道存活率为 `0.9994`，预测为 `0.7871`；
- 代表解全局事件轨道数平均偏多约 `41`；
- evidence projected RMSE 为 `0.0656`，代表解退化到 `0.1032`。

ensemble mean 的平滑来自成员平均，不代表任一合法成员具有连续性。第一版因此使用唯一的
确定性 MAP path；不确定性通过解析 marginals、方向分歧和输入干预表达。

### 2.4 相邻同状态 segment

producer 的基础状态转移矩阵表达“发生状态变化时转向哪一类”。segment renewal
单独表达 profile 参数、厚度或地层事件开始新周期。一次 renewal 可以保持原状态，
因此高阻、背景或低阻地质体可以跨越多个亚调谐 segment 持续存在。

same-state renewal probability 是独立、版本化的亚调谐生成先验。V3 producer 的
直接对象转移不包含同状态；横向 birth/death 后实际同状态接触约占 `0.49%`。
这一统计只负责 provenance，不代替真实工区的细分密度先验。

相邻同状态 segment 参数化后，若合并成一个 profile 的误差低于公开 evidence scale
规定的阈值，则发布为一个 canonical segment。只有参数差异能够形成可审计的 profile
变化时，same-state seam 才保留在 segment table 中。

### 2.5 真实剖面暴露的解码缺口

首轮真实剖面表明 band-limited increment evidence 具有可读性，但结构解码存在先验驱动
周期条带。直接原因是 renewal evidence 使用中性常数，duration prior 频繁切段，
同时零对角 transition 强制低/高阻段回到背景态。真实结果中约 `80%` 的段厚不超过
`10 m`，而 state evidence 中位最大概率仅为 `0.482`。

该结果将 band-limited evidence 标记为有效阶段产物，将当前 high-resolution deterministic
realization 标记为未通过真实剖面门禁。结构解码需要先完成 evidence-guided renewal、
same-state renewal 和带限投影一致性，再进入真实井解释。

### 2.6 不确定性范围

第一版发布以下确定性诊断：

- exact HSMM state/renewal marginals 与 entropy；
- inline/xline directional disagreement；
- full 与 no-seismic 的预测差；
- 主 LFM 与敏感性 LFM 的预测差；
- 真实输入相对 synthetic/augmentation support 的 OOD 指标。

这些诊断不生成多个高分辨率候选，也不参与最终结构解码。

## 3. 固定数据、坐标与先验合同

### 3.1 时深对称

垂向数据携带 `SampleAxis`：

```text
depth: domain=depth, unit=m, depth_basis=tvdss
time:  domain=time,  unit=s, depth_basis=null
```

同一套 interface 服务 time/depth，两种 domain 使用各自权重、wavelet 和 forward adapter。
metrics 使用 sample、axis unit 或 tuning-scale fraction，不在共享模块中写死 `5 m` 语义。

### 3.2 横向几何

- `inline/xline` 负责几何寻址；
- 横向距离使用 `lateral_m` 或实际 XY 坐标；
- 工区 `xline_step=4` 只参与索引映射；
- patch 边缘使用显式 lateral mask；
- direction reversal 只改变数组顺序，不改变物理结果。

### 3.3 LFM anchor

每道、每个已知 zone 将 full LFM 离散投影为 zone-linear anchor：

```text
background_lfm_linear = a_lfm + b_lfm * (2ζ - 1)
```

full LFM 是网络输入和敏感性参考。decoder 的低频背景固定为
`background_lfm_linear`，网络预测其上的结构化增量。

`lateral_valid` 表示 zone-linear anchor 的离散计算支持：zone 顶底有效，并且当前
`SampleAxis` 上至少有两个 `observed_valid` 样点落在 zone 内。只有一个模型样点的薄端或
pinchout 道标记为 unsupported，不使用单点常数拟合或 zone 外插值。

正式报告分别给出：

- `prediction - background_lfm_linear`；
- `full_lfm - background_lfm_linear`；
- prediction、full LFM 和真实井曲线的最终比较。

### 3.4 SemiMarkovPrior

`SemiMarkovPrior` 是独立、版本化的科学合同：

- initial probability 和 conditional state-change transition 从 training parents 的 high-resolution truth 标定；
- renewal 的 state persistence mass 与 conditional state-change transition 分开发布；
- duration 单位为 segment thickness / zone thickness；
- 运行时依据当前 SampleAxis 离散为合法 duration bins；
- density 由 duration prior 控制，不焊接在网络停止概率中；
- renewal likelihood 由 signed reflectivity 的均值/尺度构造，弱证据不为微边界提供高置信度；
- 主 prior 在 tuning/calibration 后冻结；
- 真实工区的 dense-prior sensitivity 作为独立结果发布。

更密的高分辨率细分通过版本化 duration prior 表达。主结果与 density sensitivity 使用同一
checkpoint 和同一地震证据，不能逐道调整 prior 来追随地震波峰数量。

### 3.5 SHA 与错误合同

schema、status、identity、domain、unit、axis、shape、dtype、mask 和几何语义严格校验。
fingerprint 只承担 provenance 和稳定随机 identity，不作为 consumer 拒绝输入的 equality gate。

## 4. 单一公开模块

包根只暴露一个深模块：

```python
prediction = generator.predict(observation_tile)
evidence = generator.observe(observation_tile)  # 诊断与对照
```

`ConditionalGenerator.predict()` 内部拥有：

```text
input normalization
→ ObservableEvidence
→ reflectivity-to-renewal calibration
→ exact forward-backward + Viterbi MAP
→ segment extents
→ deterministic profile means
→ equivalent same-state canonicalization
→ high-resolution decoder
→ finite-support projection
```

公开输入：

```text
ObservationTile
├── model_axis / highres_axis
├── seismic / full_lfm / observed_valid
├── depth-only vp_model_mps
├── known zone top / bottom
├── lateral_m
├── optional x_m / y_m
└── identity
```

公开输出：

```text
StructuredPrediction
├── ObservableEvidence
├── MAP state path / segment table
├── state and renewal marginals
├── LFM-relative c0/c1/c2 means
├── decoded high-resolution log-AI
├── projected model-grid log-AI
├── evidence attribution diagnostics
└── support / stitching masks
```

调用者不运行 HSMM、不拼接 segment、不平滑 coefficient，也不选择候选。内部数值模块可以
分别测试，但训练、评估和真实推理必须穿过同一个公开 interface。

## 5. 阶段 0：确定性核心收敛（已完成）

### 5.1 已落地内容

- `ConditionalGenerator.observe()` 与 `ConditionalGenerator.predict()` 构成唯一结构解码入口；
- exact semi-Markov forward-backward、marginals 和 Viterbi 保留，MAP 是生产输出；
- HSMM prior calibration 读取 canonical high-resolution segment table，以 producer object
  为 duration 统计单位，并将非对角接触归一化为 conditional state-change transition；
- same-state renewal mass 由独立、版本化的 decode policy 提供，不从 producer 中稀少的
  偶发同状态接触反推；
- segment profile head 输出确定性 mean，系数 variance 作为诊断校准量；
- HSMM calibration 将 prior、选定 conditioning、decode policy 和 profile head 一并发布为
  V8 deterministic generator checkpoint；
- 随机候选、ensemble summary、空间随机耦合和成员选择代码已从生产链移除；
- section/volume inference 先融合 evidence，再执行一次结构解码；
- 结构化 artifact 只保存一个确定性预测、marginals、segment table 和 support；
- `cup.synthetic` 继续负责 NumPy decoder、projection 和 forward seam；
- `scripts/audit_sha_contract.py` 通过，fingerprint equality 不参与 consumer admission。

阶段 0 的实现只收敛运行时主链；已有 corpus、checkpoint 和历史实验目录作为审计输入保留，
阶段 1 再按当前 deterministic contract 逐项确认兼容性。

### 5.2 阶段门禁

- 同一输入和 checkpoint 重复运行得到逐位一致的 segment table；
- time/depth fixture 通过同一个 `predict()` interface；
- Viterbi MAP 与小序列 brute-force 最优路径一致；
- posterior marginals 归一且有限；
- deterministic prediction 经 decoder/projector 数值闭合；
- 生产 CLI 和包根只要求 observation、checkpoint、prior 和 domain contract。

## 6. 阶段 1：单道确定性结构反演

### 6.1 ObservableEvidence

正式 evidence 为：

```text
ObservableEvidence
├── background_lfm_linear
├── projected_log_ai_increment_mean / scale
├── signed_reflectivity_mean / scale
├── state_log_potential[..., 3]
├── local_tuning_scale
├── model/highres axes
└── support
```

输入通道使用 training split 的全局 robust scale：

```text
scaled seismic
scaled (full_lfm - background_lfm_linear)
observed validity
```

当前已冻结 full model。no-seismic control 和 matched within-parent seismic shuffle 用于后续
地震归因审计，不是首轮真实剖面生成的运行依赖。每个 head 独立报告指标，
总 loss 只用于优化。

### 6.2 Deterministic HSMM MAP

每道、每个 zone 执行：

```text
state log potential + signed-reflectivity renewal evidence
+ state-persistence/change transition + frozen duration prior
→ exact forward-backward
→ state/renewal marginals
→ deterministic Viterbi MAP
```

renewal probability 使用 `abs(reflectivity_mean) / reflectivity_scale` 的版本化单调变换。
阈值、温度、上下限和 same-state renewal mass 是显式 decode policy，随每次预测产物发布。

Viterbi 的并列规则固定并写入合同。MAP 必须覆盖整个 zone，segment duration 合法，
相邻 segment 可以具有同一状态。证据权重、prior temperature 和 decode policy 只在
tuning/calibration 上选择一次，真实剖面不逐道调参。

HSMM 首轮使用冻结 evidence，不让梯度穿过 MAP。参数 head 先用 truth segments 与合法
boundary jitter 训练；随后仅用 MAP segments 做端到端评价。

### 6.3 Segment profile

parameter head 接收：

- state embedding；
- duration fraction 和 segment extent；
- segment 内公开 evidence pooling；
- LFM anchor 与局部 tuning scale。

输出为 LFM-relative `c0/c1/c2` mean。普通 segment 同时监督 decoded profile 和可识别系数；
rank 不足或病态 segment 只监督 decoded profile。最终 decoder 在 high-resolution axis 上生成
log-AI。相邻同状态 profile 经过 evidence-scale canonicalization，将观测上不可区分的 seam 合并。
带限主结果为 `background_lfm_linear + projected_log_ai_increment_mean`；high-resolution
realization 的 finite-support projection 是一致性诊断，完整垂向预测组装后统一执行。

### 6.4 阶段门禁

主门禁只使用少数指标：

- projected log-AI RMSE；
- high-resolution log-AI RMSE；
- balanced state accuracy / state proper score；
- tuning-scale boundary displacement；
- segment-count bias 与 duration distribution；
- identifiable subset profile NLL/MAE。

科学对照固定为：

- zone-linear anchor；
- full LFM；
- no-seismic structured model；
- matched seismic shuffle；
- prior-only HSMM。

进入真实固定剖面要求：

- full model 的 evidence、HSMM、profile 和 projection 在合成 calibration/section 上数值闭合；
- MAP reconstruction 不显著破坏带限 evidence；
- renewal 相对普通位置在反射 evidence 峰值上显著富集；
- prior-only 路径与 full evidence 路径在宏观 state body 上存在明确差异；
- same-state policy 不产生周期性背景插入，canonicalization 不改变带限结果；
- 冻结 checkpoint 可在同一公开 interface 上处理深度域真实 `ObservationTile`；
- exact one-sample micro-boundary 不作为推进条件。

## 7. 阶段 2：真实工区冻结剖面可行性

### 7.1 目标与输入

该阶段先回答：

> 冻结的单道条件生成器在真实深度域剖面上，能否生成数值合法、视觉可读、
> 与 LFM 和层位合同一致的高分辨率结构化 log-AI？

首轮使用真实深度域 seismic、TVDSS `SampleAxis`、已解释 zone 顶底层位、
比例切片克里金 full LFM 和冻结 full deterministic generator checkpoint。趋势 LFM
作为第二次敏感性输入。深度域 AI–Vp relation 只用于 tuning-scale 与 forward 诊断。

real-field adapter 只负责把 seismic、LFM、层位、XY 和纵向轴组装为
`ObservationTile`。结构解码继续调用 `ConditionalGenerator.predict()`。

### 7.2 展开顺序

先运行少量固定剖面：

- 2 条穿可信井剖面；
- 1 条穿低质量井震标定剖面；
- 1 条不穿井 blind section。

每个已知 zone 独立组装 tile 并解码。`lateral_m` 由实际 XY 路径累计，
`xline_step=4` 只用于从线号解析 SEG-Y 几何。模型不额外平滑预测。

### 7.3 输出与诚实边界

每个 zone 保存：

- 原始 seismic、full LFM 与 zone-linear anchor；
- projected increment evidence、signed reflectivity 和 state potential；
- deterministic state path 与 segment table；
- band-limited evidence log-AI、high-resolution log-AI 与完整垂向 projected consistency；
- state/renewal marginals 与 entropy；
- support mask、输入尺度统计与完整剖面图件。

真实剖面首先判断带限 evidence 是否可读，再判断 high-resolution decoder 是否通过。
带限结果通过不自动使高分辨率结果通过。

### 7.4 剖面门禁

报告和人工图件审查至少检查：

- 真实 seismic 和 LFM 有效区的 robust scale 是否落在 target contract 支持范围；
- 线性 anchor、evidence、projected/high-resolution prediction 是否有限且轴对齐；
- segment density、duration 和 state occupancy 是否明显偏离合成 calibration；
- 是否存在逐道随机跳段、层位附近伪边界、周期性条带或支持缝隙；
- 同状态续生是否形成可读地质体，而非数值近似的碎段表；
- projected high-resolution 在有效 FIR 支持内是否与 band-limited evidence 一致；
- 主 LFM 与趋势 LFM 下的带限结构是否稳定；
- 穿井剖面上 prediction、full LFM、zone-linear anchor 与井 log-AI 的分频带对比。

前 4 条剖面中出现非有限输出、大范围 unsupported、明显周期条带或密度崩塌时，
停止全剖面/全体积展开，先定位输入支持、sim-to-real 或 prior 问题。

real-field adapter 和 `generate-real-section` 已完成 478 道 interface smoke。带限 evidence
通过可读性审查；零对角 transition + neutral renewal 生成的 high-resolution realization
未通过周期条带门禁。修订 decode policy 必须在同一剖面与合成 section 上对照验证。

## 8. 阶段 3：横向能力与全体积

### 8.1 真实观测统计 profile

sim-to-real 在本项目中表示真实观测扰动覆盖。统计来源包括真实地震、可信井震标定和冻结
wavelet，覆盖：

- phase、small shift 和 vertical static；
- global/tracewise positive gain；
- 平滑深度—横向振幅衰减；
- white、colored 和 coherent noise；
- 弱反射压低与可见峰减少。

augmentation 保持同一 synthetic truth，并在 calibration 前冻结 profile 和 dirty identity。
clean/dirty 成对评价，检查真实输入是否落入训练支持范围。

### 8.2 横向能力决策

阶段 2 的真实结果决定实现路径：

- 若单道结果已经满足业务剖面连续性，先补齐 no-seismic/干预归因和真实观测覆盖；
- 若出现明显逐道跳段，训练 21 道、约 `±250 m` 的一维 lateral evidence model，
  使用米制距离和显式 mask；
- 若主要失败是 seismic/LFM OOD，先做 clean/dirty 配对训练；
- 若主要失败是 segment 密度，优先校准版本化 duration prior，不改动 evidence 网络。

需要 lateral model 时，中心道 evidence 一次产生，pinchout/topology mask 不施加
强制平滑。真实三维推理先融合 inline/xline evidence，然后只执行一次确定性
结构解码。横向模型只有在连续性改善且不增加 pinchout false bridging 时才取代
single-trace checkpoint。

### 8.3 全体积门禁

全体积前补齐 no-seismic checkpoint 与 matched seismic intervention。可信井上要求 full
相对 no-seismic 的对应频带配对改善为正，并同时报告 full LFM 和 zone-linear
anchor。低质量井只进入逐井诊断。

固定剖面、LFM sensitivity、真实观测支持和井上归因门禁通过后，使用完全
相同的 checkpoint、prior、标准化、LFM 和推理参数运行全体积。

## 9. 全体积输出

全体积保存：

- projected evidence mean/scale；
- state potential/marginal 与 renewal marginal；
- deterministic high-resolution log-AI；
- projected model-grid log-AI；
- segment table；
- direction disagreement；
- full-no-seismic contribution；
- LFM sensitivity；
- support/stitching/OOD masks。

chunk、GPU batch 和执行顺序不改变结果。体推理通过重叠 halo 和中心裁剪保证 stitching
一致性，不需要保存候选成员或进行第二遍代表成员选择。

## 10. 停止条件与后续研究

以下失败直接停止向真实全体积推进：

- full model 没有稳定优于 no-seismic；
- deterministic MAP 明显破坏带限 evidence；
- lateral model 只降低 roughness，却增加 pinchout false bridging；
- 真实剖面落在 augmentation/OOD 支持之外；
- 可信井的 full-no-seismic 配对改善方向错误。

第一版完成后，再根据 zero-shot 结果决定井监督、可微 forward 约束、真实 adaptation 或二维
横向网络。它们不属于当前实现链的依赖。

## 11. 最小验证清单

- target audit、tiny overfit 和 full/no-seismic/shuffle probe；
- time/depth 同 interface smoke；
- high-resolution truth 到 decoder/projector 的 NumPy/Torch parity；
- discrete LFM rebasing parity；
- 小序列 exact HSMM marginals/Viterbi 与 brute force 对照；
- renewal/state-change 语义分离，exact HSMM 支持对角 transition；
- reflectivity SNR 到 renewal probability 的变换单调、有界且时深无关；
- 相邻同状态 profile 的 canonical merge 只移除 evidence-scale 下不可区分的 seam；
- duration zone fraction 在不同采样间隔下语义一致；
- 短 segment identifiability mask；
- prediction 重复运行逐位一致；
- lateral mask、direction reversal 和 stitching invariance；
- inline/xline evidence 先融合、结构只解码一次；
- section continuity、pinchout 和 false bridging 图件；
- 真实剖面的 full/no-seismic/LFM 配对报告；
- forward diagnostic 只产生诊断，不改变预测。
