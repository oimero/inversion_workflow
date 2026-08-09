# Structured GINN V2 确定性实施规格

## 1. 项目目标

Structured GINN V2 的目标是：

> 地震带限证据指导下、具有横向连续性的高分辨率地质生成器。

第一版采用确定性条件生成：

```text
seismic + full LFM + 1D lateral context
→ ObservableEvidence
→ deterministic exact HSMM MAP
→ deterministic segment profile
→ high-resolution log-AI
```

LFM 负责 zone 内低频背景；地震网络只恢复当前频带和调谐尺度能够约束的增量、状态与
有符号反射证据；半马尔科夫先验负责亚调谐 segment 的状态转移、持续时间和微边界。
最终交付是一个完整、合法、可复现的结构化预测。

高分辨率表示生成器能够依据观测与井标定先验构造细网格地质模型。它不表示地震逐层
分辨了所有微层。每个预测同时发布证据强度、方向分歧和先验敏感性，区分地震支持的
带限结构与先验补全的亚调谐细节。

第一版的最高空间复杂度是一维横向模型。真实三维体由同一个模型沿 inline 和 xline
分别提取证据，融合证据后执行一次确定性结构解码。

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

producer 的基础状态转移矩阵不包含对角转移；横向 birth/death 或 pinchout 移除中间对象后，
两个同状态 producer objects 仍可能在一条道上接触。确定性路线将连续同状态样点规范为一个
最大 state run：接触处的 object seam 不构成 HSMM renewal。

transition 和 duration prior 从这些 high-resolution 最大 state runs 标定，主对角线固定为零；
model-grid 审计也只在状态实际变化处建立 renewal。这样可以避免在弱证据真实数据上产生
数值近似、地震上不可区分的连续同状态段。

### 2.5 不确定性范围

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

- transition 和 initial probability 从 training parents 的 high-resolution truth 标定；
- duration 单位为 segment thickness / zone thickness；
- 运行时依据当前 SampleAxis 离散为合法 duration bins；
- density 由 duration prior 控制，不焊接在网络停止概率中；
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
→ exact forward-backward + Viterbi MAP
→ segment extents
→ deterministic profile means
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
- HSMM prior calibration 读取 canonical high-resolution segment table，将接触的同状态对象
  规范为最大 state run，再以 zone fraction 统计 duration；
- segment profile head 输出确定性 mean，系数 variance 作为诊断校准量；
- HSMM calibration 将 prior、选定 conditioning 和 profile head 一并发布为 V7
  deterministic generator checkpoint；
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

固定训练 full model 和 no-seismic control；matched within-parent seismic shuffle 作为推理
干预。每个 head 独立报告指标，总 loss 只用于优化。

### 6.2 Deterministic HSMM MAP

每道、每个 zone 执行：

```text
state log potential + frozen transition/duration prior
→ exact forward-backward
→ state/renewal marginals
→ deterministic Viterbi MAP
```

Viterbi 的并列规则固定并写入合同。MAP 必须覆盖整个 zone，segment duration 合法，相邻状态
符合 transition prior。证据权重和 prior temperature 只在 tuning/calibration 上选择一次。

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
log-AI，再以正式 finite-support projection 返回 model grid。

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

进入阶段 2 要求：

- full 相对 no-seismic 的 projected/high-resolution AI 配对改善置信区间为正；
- matched shuffle 后地震收益下降；
- MAP reconstruction 不显著破坏带限 evidence；
- segment count 和 duration 不因 dirty 输入产生系统偏移；
- exact one-sample micro-boundary 不作为推进条件。

## 7. 阶段 2：一维横向连续性

### 7.1 横向模型

模型读取 21 道、约 `±250 m` 的一维横向 patch，只预测中心道 evidence。逐道垂向 encoder
后接使用 `lateral_m` 和显式 mask 的 lateral mixer。

训练短 patch 保留 25 道，使内侧 5 道具有完整上下文；边缘道只验证 mask 行为。横向 loss
作用在 synthetic truth 中可比较的对象上：

- tuning-scale evidence；
- state potential/occupancy；
- projected log-AI；
- 可对应 segment 的 duration/profile。

pinchout、birth/death 和 topology transition 通过 producer event identity mask 排除强制平滑。

### 7.2 Section 推理

完整 section 以滑动 patch 运行，每个中心道只产生一套 evidence。全部 evidence 组装完成后，
每道执行一次相同的 deterministic HSMM 和 profile decoder。

横向连续性来自：

1. 邻道上下文改变中心道 evidence；
2. 横向一致性监督约束 evidence 与 profile；
3. 所有道共享同一冻结 prior 和确定性并列规则。

segment alignment 只用于评价，不进入推理，也不修改最终预测。

### 7.3 三维体的双方向融合

同一个一维模型沿 inline 和 xline 分别运行：

```text
inline ObservableEvidence
+ xline ObservableEvidence
→ calibrated evidence fusion
→ one deterministic HSMM/profile decode
```

连续 mean 采用校准精度加权；state log potentials 在统一 temperature 后融合；scale 同时包含
方向内尺度和方向间分歧。survey 边缘只有一个合法方向时使用该方向。

微结构在 evidence 融合后只生成一次。首版三维推理不训练 inline-xline 二维输入网络。

### 7.4 Section 门禁

calibration 完成后，固定 section parents 只运行一次最终门禁。报告必须包含每类 geometry 的
truth、evidence、prediction、increment 和 residual 图件；人工图件审查是正式验收的一部分。

数值检查包括：

- projected/high-resolution lateral roughness 与 variogram；
- 相邻道 segment-count jump；
- tuning-scale boundary displacement consistency；
- event survival、birth/death 和 pinchout false bridging；
- direction reversal invariance；
- patch/halo/stitching invariance；
- neighbor shuffle 后横向收益下降；
- 输出相对带限 evidence 的退化量。

进入真实工区要求：

- lateral model 相对 single-trace 改善横向连续性；
- pinchout false bridging 没有增加；
- prediction 保留阶段 1 的地震增量价值；
- none、wedge、pinchout 的固定图件不存在明显条带、跳段和 stitching seam。

## 8. 阶段 3：真实观测覆盖与冻结 zero-shot

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

### 8.2 固定剖面

使用冻结的 full 与 no-seismic checkpoint，先运行：

- 6 口可信井剖面；
- 3 口低质量井震标定诊断剖面；
- 至少 2 条不穿井 blind section。

比例切片克里金 LFM 是主输入，趋势 LFM 是敏感性输入。每条剖面保存：

- deterministic state path 和 segment table；
- high-resolution 与 projected log-AI；
- state/renewal entropy；
- inline/xline direction disagreement；
- full-no-seismic seismic contribution；
- LFM variant sensitivity；
- support/stitching/OOD masks。

### 8.3 真实井门禁

- full 相对 no-seismic 在井上对应频带的配对改善为正；
- full 相对 full LFM 和 zone-linear anchor 具有整体增量价值；
- deterministic prediction 在多数可信井上不相对主 LFM 退化；
- 高分辨率井对比按 tuning scale 与更细诊断尺度分别报告；
- 低质量井只进入逐井诊断；
- blind section 不出现 seam、方向条带或 segment 随机跳变。

剖面门禁通过后，使用完全相同的 checkpoint、prior、标准化、LFM 和融合参数运行全体积。

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
- high-resolution 最大 state run 的 transition diagonal 为零；
- duration zone fraction 在不同采样间隔下语义一致；
- 短 segment identifiability mask；
- prediction 重复运行逐位一致；
- lateral mask、direction reversal 和 stitching invariance；
- inline/xline evidence 先融合、结构只解码一次；
- section continuity、pinchout 和 false bridging 图件；
- 真实剖面的 full/no-seismic/LFM 配对报告；
- forward diagnostic 只产生诊断，不改变预测。
