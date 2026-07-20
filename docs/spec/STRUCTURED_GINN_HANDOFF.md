# Structured GINN 纵向原型 Handoff

## 1. 状态与定位

本文是 Structured GINN 纵向原型的活动实施规格。目标是把 Synthoseis-lite
中的三状态对象、持续期和三参数剖面从数据生成机制变成 GINN 显式推断的低维
隐变量模型。

本路线是下一阶段研究主线。[Enhance v2](ENHANCE_V2_HANDOFF.md) 保留为冻结的
替代路线：它在固定中频反演之后补充井统计条件微纹理；Structured GINN 则直接
推断对象级后验。两者首轮不串联。

Synthoseis-lite 的父实现、双索引、视图和域 Adapter 继续遵循
[ADR 001](../adr/001-synthoseis-shared-pipeline.md)。本文只定义新增的结构化监督
合同和推断路径，不复制已有 benchmark 编排合同。

## 2. 科研问题与解释边界

当前连续 GINN 解决的是：

```text
seismic + LFM
    -> model-grid canonical increment
    -> model-grid log AI
```

Structured GINN 要验证的是：

```text
seismic + LFM
    -> object segmentation posterior
    -> per-zone and per-object parameter posterior
    -> deterministic object decoder
    -> high-resolution log AI
    -> projection and seismic forward
```

高分辨率网格是结果的表达分辨率，不是观测的信息分辨率。网络的自由输出限于
对象分段和少量参数；状态网格、边界网格、反射系数、高分辨率阻抗和合成地震均
由同一组对象变量确定性展开。

真实地震对目标高频尺度通常是非唯一的。未来真实工区输出只能解释为地震、LFM
和标定地质先验共同约束的对象后验，不能解释为唯一恢复的地下真实薄层。正演
闭合不是单独的成功标准。

## 3. 当前仓库事实

- Synthoseis-lite 已生成 high-resolution log AI、RGT、state、object、对象内部
  坐标、zone、boundary、object catalog 和逐横向位置的最终 `c0/c1/c2`。
- GINN v2 的活动 reader 只读取同网格 seismic、canonical background、canonical
  increment 和 `valid_mask`。
- 活动模型输入为 seismic、LFM 和 `valid_mask` 三通道，输出为一个连续 increment
  通道；合成监督为逐点 MSE。
- physics block 使用 waveform loss 和 increment L2。地震弱敏感或零空间内的
  高频分量不会被 waveform loss 可靠约束，同时会受到 increment L2 惩罚。
- `valid_mask` 当前同时承担数据支持、网络输入和监督范围。结构化路径必须拆开
  这些语义，且不把 mask 当作地质观测通道。
- 当前 truth generator 是包含随机采样、参数投影、边界条件、裁剪和 QC 的 NumPy
  生成过程，不是可以直接用于训练的共享可微 decoder。
- 对象剖面的完整公式还包含逐 realization、逐 zone 的背景 `a/b`；当前 benchmark
  没有直接发布这两个变量。

旧连续 GINN v2 的 reader、模型、checkpoint 和部署合同保持可运行。Structured
GINN 使用独立子包、配置和产物 schema，不在原合同上增加分支语义。

## 4. 首版范围

首版包含：

- 时间域和深度域共享的 latent contract、object decoder、HSMM 和 projection seam；
- 更新 Synthoseis-lite producer，直接发布完整结构化 latent；
- Oracle contract round-trip；
- 深度域、单道、单个可配置 zone 的合成监督原型；
- Direct、Calibration-prior、LFM-only Structured-HSMM、Structured-HMM 和
  Structured-HSMM 比较；
- state/boundary 边缘概率、MAP 分段、参数分布和 posterior realization；
- 合成 holdout、厚度分箱、posterior calibration 和 seismic sensitivity 报告。

首版不包含：

- 真实工区推理或无标签 physics 适配；
- 真实井监督、留井评估或 posterior adaptation；
- 外部真实 LFM 对 high-resolution posterior 的低频替换；
- 跨 zone 联合推断；
- 横向对象拓扑或二维 semi-Markov 模型；
- mixture posterior、diffusion 或 MCMC；
- amplitude、noise 和 registration views 的主消融；
- 逐点独立分类作为正式科学消融；
- 旧 benchmark 的 `a/b` 反推或迁移工具。

## 5. 代码边界

目标结构：

```text
src/ginn_v2/structured/
├── contracts.py       # experiment、checkpoint、prediction 和 latent 合同
├── data.py            # domain-neutral latent reader、split、normalization、batch
├── decoder.py         # object decoder 的 Torch 接口
├── hsmm.py            # forward-backward、Viterbi、backward sampling
├── model.py           # 两通道 encoder、emission 和参数 posterior
├── losses.py          # segment、parameter、reconstruction、projection、forward
├── training.py        # synthetic-only 阶段和固定验证
├── prediction.py      # MAP、marginal 和 posterior realization
└── reporting.py       # latent、闭合、calibration 和 sensitivity 报告
```

共享科学实现属于 `cup.synthetic`：

- domain-neutral structured latent schema；
- 确定性 NumPy object decoder；
- 与 NumPy decoder 等价的 Torch backend；
- 已有 projection 和 time/depth forward operator。

边界规则：

- `cup.synthetic` 不依赖 `ginn_v2`；
- `ginn_v2.structured` 可以依赖公开的 synthetic、projection 和 physics 接口；
- 新代码不复制时间域、深度域 decoder 或 HSMM；
- 域 Adapter 只提供轴名、单位、采样间隔和域专属 forward 参数；
- 旧连续 `ginn_v2.models`、`ginn_v2.data` 和 checkpoint schema 不改语义；
- 首版使用独立入口与 experiment schema，不允许旧 checkpoint 被结构化 reader 接受。

## 6. Structured latent supervision 合同

新增 schema：

```text
structured_latent_supervision_v1
```

它由 Synthoseis-lite producer 与 benchmark 同时发布。核心 benchmark 保持父实现拥有
权和 view 索引语义；latent 只属于 base parent，seismic view 复用相同 latent。

### 6.1 轴合同

每个 parent 必须发布：

```text
observed_axis
latent_axis
observed_sample_interval
latent_sample_interval
latent_oversampling_factor
sample_domain            time | depth
sample_unit              s | m
depth_basis              null | tvdss
lateral_m
```

要求：

- observed 和 latent 轴规则、严格递增；
- oversampling factor 是正整数并由 artifact 声明，不在 reader 中硬编码；
- latent 轴与 projection contract 严格嵌套；
- 时间域和深度域使用相同公共字段，域专属别名只存在于 Adapter；
- xline/inline 标签不作为物理横向间距，横向算子只使用 `lateral_m`。

### 6.2 网格 latent

每个 base parent 至少发布：

```text
log_ai_highres          float32 [lateral, latent]
rgt_highres             float32 [lateral, latent]
state_id_highres        int8    [lateral, latent]
object_id_highres       int32   [lateral, latent]
object_xi_highres       float32 [lateral, latent]
zone_id_highres         int16   [lateral, latent]
boundary_mask_highres   bool    [lateral, latent]

observed_valid_mask     bool    [lateral, observed]
latent_valid_mask       bool    [lateral, latent]
object_supervision_mask bool    [lateral, latent]
```

mask 语义固定：

- `observed_valid_mask`：观测网格上的 projection/forward/loss 支持；
- `latent_valid_mask`：目标 zone 内具有合法 state/object 标签的高分辨率位置；
- `object_supervision_mask`：允许计算 state、boundary、duration 和参数监督的位置；
- 三种 mask 均不进入 encoder 输入；
- padding 只由 batch mask 表达，不与地质 zone mask 合并。

### 6.3 对象与背景表

对象表按 realization、zone、object 和 lateral position 建立显式主键，发布：

```text
state / state_id
object_top_coordinate / object_bottom_coordinate
duration_fraction / duration_samples / thickness
c0_raw / c1_raw / c2_raw
c0_profile_projected / c1_profile_projected / c2_profile_projected
c0_effective / c1_effective / c2_effective
profile_projection_scale
c0_conditioning_adjustment
segment_supervision_valid
```

三参数的阶段语义固定为：

1. `raw` 是随机场采样完成、尚未施加对象剖面约束的生成器变量，也是首版网络的
   监督目标；
2. `profile_projected` 是向标定中心收缩、满足 profile metric 约束后的变量；
3. `effective` 是进一步按背景、状态和绝对 AI bounds 调整 `c0` 后，实际进入解析式
   的变量。

producer 必须直接发布三组值。`profile_projection_scale` 和
`c0_conditioning_adjustment` 只用于审计，consumer 不得靠它们重建缺失阶段。共享 decoder
只接受 raw 参数进入完整约束路径；若内部函数直接展开 effective 参数，该函数必须以
不同类型或不同入口命名，不能再次执行 projection 或 conditioning。

一个 object 定义为一个最大常值 state segment。相邻 object 不允许具有相同 state，
transition matrix 的对角元素必须严格为零。因此 state sequence、boundary、object count
和 duration 存在唯一一致的相互转换。

`object_supervision_mask` 是逐点合同。对象表中的 `segment_supervision_valid` 由 producer
唯一决定：当且仅当该 object 的全部 latent samples 都位于
`object_supervision_mask` 内时为真。duration 和三参数的 segment-level loss 只消费该
布尔值，consumer 不得自定覆盖率阈值。

新增 zone background 表：

```text
realization_id
zone_id
background_a
background_b
```

`background_a/background_b` 必须在生成时直接记录。不允许 consumer 从 high-resolution
AI 或随机种子反推。

### 6.4 身份与可消费性

latent manifest 记录：

- parent benchmark identity；
- impedance calibration identity；
- transition/duration model identity；
- object profile contract；
- projection contract；
- forward contract；
- random stream contract；
- producer schema 和 science revision。

生成器 duration 的科学坐标保持为 zone fraction。对象表同时记录该 fraction、对应
物理厚度和当前 latent grid 上的整数 `duration_samples`。HSMM 对每条 trace 使用
`duration_samples` 枚举合法路径，但其 duration potential 由候选样本数、latent interval
和该 trace 的 zone extent 映射回 zone fraction 后计算，并在该离散支持上重新归一化。
duration identity 因而同时绑定连续 fraction calibration、sample domain、latent interval、
active zone 和离散化规则。时间域与深度域共享算法和接口，但分别发布自己的离散化
identity，不能直接复用另一域的样本数分布。

只有结构、轴、有限性、主键或 fingerprint 不可解释时标记不可消费。对象数量、状态
比例、参数分布或科学指标异常只产生 warning。

更新 producer 后重新生成结构化训练 benchmark。旧 benchmark 继续供连续 GINN 使用，
不提供兼容或迁移路径。

## 7. 共享 object decoder

decoder 输入是确定性的完整对象描述：

```text
latent_axis
zone top/bottom
zone background a/b
ordered segments
segment state
segment top/bottom or duration
segment raw c0/c1/c2
calibrated profile and AI bounds
```

对 zone 内坐标 `ζ` 和对象内坐标 `ξ`，输出固定为：

```text
background = a + b * (2ζ - 1)
profile    = c0 + c1 * (2ξ - 1) + c2 * sin(pi * ξ)
log_ai     = background + profile
```

decoder 必须复用 producer 的对象端点、最后对象闭区间、profile projection、c0
conditioning 和 AI bounds 语义，并返回 raw、profile-projected 和 effective 三阶段参数。
解析式只使用 effective 参数。NumPy 和 Torch backend 对相同输入逐点等价。

Oracle round-trip 顺序固定：

```text
published latent
  -> decoded high-resolution log AI
  -> existing finite-support projection
  -> model-grid log AI
  -> existing time/depth seismic forward
```

必须分别复算并比较 high-resolution truth、model-grid truth 和 model-consistent seismic。
Oracle 不训练模型，也不使用网络输出。任何一级不能在声明容差内闭合，结构化训练都
不得启动。

Oracle 的名称固定为 `Oracle contract round-trip`。它只证明 latent artifact 足以重放
生成过程以及 decoder、projection、forward 合同闭合，不代表 seismic 可达到的理论
上限。

## 8. 数据与 split

首轮 active config 选择一个占位符 zone。选择原则为物理厚度较大、有效 parent 较多、
对象持续期覆盖较丰富；公共文档和配置模板不写真实 zone 名。

每个训练 item 是一个 parent 中的一条 lateral trace 和一个完整 zone：

- 输入在 observed grid，包含 forward operator 要求的 zone 上下文；
- latent 输出只覆盖该 zone；
- 保留真实物理采样间隔，不把不同厚度 zone 拉伸到统一归一化长度；
- batch 使用右侧 padding，并分别返回 observed、latent 和 segment mask；
- parent split 先于 lateral trace 展开，同一 parent 的任何 trace 不跨 split；
- sampler 先均衡 parent，再选 lateral trace，不能让宽剖面获得更高权重。

split 继续使用固定 parent identity、seed 和 geometry holdout 语义。Direct、
Calibration-prior、LFM-only、Structured-HMM 和 Structured-HSMM 共用同一 split。

normalization 只从 train parent 的 clean base 计算：

- seismic：有效点 centered mean/RMS；
- LFM：有效点冻结 mean/std；
- latent 参数：使用 calibration 中相应 state/zone 的中心和尺度；
- validation/test、seismic perturbation 和所有比较复用同一 normalization identity。

## 9. 模型与推断

### 9.1 Encoder 输入

encoder 输入固定为：

```text
channel 0: normalized clean base seismic
channel 1: normalized LFM
```

invalid 和 padding 位置标准化后填零。深度、RGT、zone、horizon、valid mask 和井距离
都不作为 encoder channel。坐标和 zone 信息只用于裁剪、prior、decoder 和 loss。

### 9.2 Encoder 与 emission

首版使用单道 1D dilated TCN：

- `in_channels=2`；
- model-grid 编码；
- 按 artifact 的 oversampling factor 投影到 latent grid；
- 输出每个 latent 位置的三状态 emission logits；
- 不输出独立 boundary 或 duration head。

### 9.3 HSMM

transition matrix、初始状态概率和 state-specific duration distribution 固定读取 impedance
calibration。首版不学习或修正这些先验。

公共 HSMM 实现提供：

- log-space exact forward-backward 和 `logZ`；
- state marginal；
- boundary marginal；
- segment posterior statistics；
- Viterbi MAP segmentation；
- exact backward sampling；
- variable-length 和 padding mask 支持。

状态、边界、对象数量和 duration 全部由同一个 segmental posterior 推导。

主模型使用 calibration 的非参数或参数化 duration-fraction distribution。用于 duration
消融的 Structured-HMM 仍走相同 segmental inference、transition、参数 head 和 decoder，
但把 duration potential 换成在相同合法支持上归一化的 memoryless geometric law；其
期望长度与 calibration 的 state-specific 期望匹配。两组之间只改变 duration law。

### 9.4 参数 posterior

每个 zone 输出 `a/b` 的 independent bounded transformed-normal；每个 segment 输出 raw
`c0/c1/c2` 的同类分布：

```text
background: pre_mean_a, pre_log_std_a, pre_mean_b, pre_log_std_b
object:     pre_mean_c0, pre_log_std_c0, pre_mean_c1, pre_log_std_c1,
            pre_mean_c2, pre_log_std_c2
```

网络参数化无界标准化空间中的独立 Normal，再以 sigmoid 映射到 calibration 声明的
硬 lower/upper bounds。NLL 在有界物理参数空间计算，必须包含 inverse transform 和
log-Jacobian；禁止用 sample 后 clamp 代替分布变换。checkpoint 保存 pre-transform
location/scale 的定义以及 bounds identity，prediction 同时发布有界 raw posterior 参数和
经 decoder 诱导得到的 effective posterior samples。边界点按合同 epsilon 进入可逆变换，
该 epsilon 属于 schema identity。

参数 head 使用 segment 内 latent feature pooling、state 和归一化 segment extent；不在
每个高分辨率网格点自由预测三参数。训练监督使用 producer 发布的 raw 参数，decoder
唯一负责 profile projection、c0 conditioning 和 AI bounds。effective posterior 是 raw
posterior 经非线性 decoder 诱导的分布，不假定仍为 Gaussian。

训练分两阶段：warm-up 使用真实 segmentation teacher forcing；随后按显式 schedule
混合真实分段与边界扰动分段。扰动只移动内部边界 `±1/±2` latent samples，保持对象数、
状态顺序、最短持续期和 zone 端点合法，因此参数监督仍有唯一对象对应。schedule、扰动
概率和随机身份写入 experiment/checkpoint。首版不在训练中对预测 segment 做启发式真值
匹配。验证和预测只使用 MAP 或采样 segmentation，不使用真实边界。必须分别报告 true、
jittered 和 predicted segmentation 条件下的参数指标，以暴露 segmentation-to-parameter
断层。

### 9.5 Posterior 物化

每个样本固定输出：

- state marginal；
- boundary marginal；
- MAP segment table；
- zone background posterior；
- MAP segment parameter posterior；
- 32 个 posterior samples。

posterior sampling 使用版本化随机身份，由 experiment seed、parent、trace、zone 和
sample index 决定。每个 sample 物化 segment table、high-resolution log AI、reflectivity、
projected log AI 和 synthetic seismic。每个 sampled segmentation 都重新执行 segment
feature pooling 和参数 posterior；禁止复用 MAP segments 的参数分布。

## 10. 训练合同

训练只使用 clean base synthetic parent。首版包含四个独立监督量：

1. true segmentation 的 segmental HSMM NLL；
2. true zone `a/b` 的 bounded transformed-normal NLL；
3. true segment raw `c0/c1/c2` 的 bounded transformed-normal NLL；
4. teacher-forced 参数均值经过 decoder 后的 high-resolution、projection 和 forward
   reconstruction loss。

各项先按自己的自然计数归一化：latent sample、zone parameter、object parameter、有效
AI sample 和有效 waveform sample。所有权重显式写入配置、manifest、history 和
checkpoint；不依据 batch 内对象数隐式改变权重。

increment L2 不进入结构化训练。高频不可见部分通过固定 HSMM 和参数 posterior 约束，
不通过把连续 increment 压向零来约束。

checkpoint selection 使用固定的合成 latent validation score。主 score 只由归一化后的
segmentation proper score 和 raw object-parameter NLL 显式加权组成；`a/b` 不进入主
selection score。报告必须分别发布 `segmentation_score`、`object_parameter_score` 和
`background_score`，不能只发布加权总分。high-resolution、projection、forward 和
sensitivity 指标只报告，不参与首轮选模。三个结构化训练组使用相同 selection 定义和
validation parent；Direct 保持其连续输出可定义的冻结选模合同，并与结构化组共用最终
holdout。

## 11. Oracle 门禁与固定实验矩阵

### Oracle contract round-trip

它是训练前门禁，不属于模型实验。使用真实 segmentation、`a/b` 和 raw `c0/c1/c2`
完成完整生成约束与 round-trip，验证 artifact 和共享 decoder 的合同闭合。

### Direct

复用当前连续 model-grid canonical increment 架构，但使用相同新 benchmark、clean base、
parent split、normalization 和训练预算。它不输出对象 posterior；只比较 model-grid AI、
seismic closure 和上采样后 high-resolution AI 的有限基线。

### Calibration-prior

不训练 encoder，也不读取 seismic 或 LFM 特征。segmentation、duration、state 和参数
直接来自固定 calibration prior，只使用 zone extent、轴和 forward 所需的结构上下文。
输出同样的 MAP、marginal 和 32 个 realization，是测量未条件化地质先验的基准。

### LFM-only Structured-HSMM

从头训练与主模型相同的 encoder、HSMM、参数 head、预算和 selection 合同。LFM 通道
保持不变，seismic 通道恒定置零。它隔离 LFM 通过学习模型能够提供的信息，不能用主
模型在评估时临时置零 seismic 来替代。

### Seismic+LFM Structured-HMM

使用两通道 encoder 和与主模型相同的 segmental inference、transition、参数 head 与
decoder，只把 calibrated duration potential 换成第 9.3 节的 memoryless geometric law。
它是 calibrated duration 价值的单因素消融。

### Seismic+LFM Structured-HSMM

使用相同 encoder 和参数 posterior，emission 进入固定 calibration HSMM。该组是首轮
候选主模型。

逐点 Structured-independent 只保留为实现调试基线。它改变了对象数量、segment pooling
和参数任务规模，不能用来解释纯粹的 HSMM 或 duration 价值。

实验首先运行一个训练 seed。人工机制审查通过后，固定 benchmark、split、normalization、
训练预算、validation 和报告合同，再补到三个训练 seed。新增 seed 不改变 generator seed。

## 12. 报告与科学门禁

### 12.1 合成 holdout

按 parent、scenario、state、对象厚度和相对调谐尺度报告：

- state accuracy、macro F1 和 posterior NLL；
- object count error；
- boundary precision、recall、F1 和最近边界距离；
- duration NLL、MAE 和 credible-interval coverage；
- `a/b` 与 raw `c0/c1/c2` 的 NLL、MAE、RMSE 和 coverage；
- effective `c0/c1/c2` 的 MAE、RMSE 和 sample-based coverage；
- high-resolution AI MAE、RMSE、correlation 和频谱；
- projected AI 指标；
- model-consistent seismic closure；
- posterior predictive coverage；
- MAP、posterior mean 和单 realization 的差异。

### 12.2 Seismic sensitivity

在完全相同的 parent、LFM 和 posterior random identity 上评估：

- seismic 置零；
- 在相同 split 和 zone 内进行 parent 间 seismic 打乱；
- 对局部 observed window 做 parent 间替换。

报告 state/boundary posterior 的 JS divergence、MAP segment 变化、参数 posterior 位移、
latent accuracy 下降和 forward closure 变化。

判读原则：

- clean seismic 必须相对独立训练的 LFM-only 模型提供可报告的 latent 改善，才可称为
  seismic-conditioned object inference；
- 主模型 seismic 置零后的 posterior 变化只作为敏感性诊断；它不能替代 LFM-only
  训练基线；
- 平静波形区域允许 posterior 变宽，不能把稳定 prior realization 描述为地震恢复；
- 只改善正演闭合而不改变 segmentation/duration posterior，表示 seismic 主要约束连续
  参数，而没有识别薄层组合。

### 12.3 与基线比较

固定报告：

```text
LFM-only Structured-HSMM - Calibration-prior
Seismic+LFM Structured-HSMM - LFM-only Structured-HSMM
Seismic+LFM Structured-HSMM - Seismic+LFM Structured-HMM
Seismic+LFM Structured-HSMM - Direct
Oracle contract round-trip - Seismic+LFM Structured-HSMM
```

这些差值依次回答 LFM 条件信息、seismic 在 LFM 之上的新增信息、calibrated duration
价值、整套结构化路线相对连续路线的效果，以及 latent inference gap。最后两项不是
单因素消融：Oracle gap 不是 seismic 可达到的理论空间，Structured-HSMM 与 Direct 的
差异同时包含表示、损失、decoder 和先验变化。

### 12.4 失败分级

以下情况硬失败：

- schema、science contract 或 fingerprint 不匹配；
- 数据集、主键、shape、轴、单位或域不可解释；
- 必需数据非有限；
- Oracle、NumPy/Torch 或 projection/forward closure 超出声明数值容差；
- HSMM 没有合法路径或数学计算产生 NaN/Inf；
- checkpoint 与 benchmark、split 或 normalization identity 不一致。

以下情况只告警、记录并继续：

- latent 恢复差；
- boundary、duration、参数或 posterior calibration 指标差；
- 某些 scenario、state 或厚度区间覆盖不足；
- seismic sensitivity 弱；
- Calibration-prior、LFM-only 与主 HSMM 差异小；
- seed 间差异大；
- validation、forward 或真实诊断结果不理想。

科学指标不决定 artifact 的结构可消费性。是否从单 seed 进入三 seed、是否从合成进入
真实工区，均由人工根据完整报告裁决。

## 13. 合成门禁后的真实阶段

本文不要求实现真实推理，只冻结进入下一阶段时必须回答的问题：

- 如何用 external LFM 替换生成器 `a/b` 低频背景，同时保持 high-resolution、projection
  和 deployment closure；
- 真实 seismic 小扰动是否导致合理的 posterior 响应；
- 趋势 LFM 与克里金 LFM 是否只改变长波背景，还是支配对象结构；
- 可信井的留井 posterior predictive 是否优于 Calibration-prior 和 LFM-only；
- 低质量标定井是否仅保留非聚合诊断角色；
- 参数是否贴住 synthetic support 边界，真实数据是否明显分布外；
- 多 seed 的主要边界是否稳定，不确定性是否随地震信息强弱变化。

真实阶段开始前必须再次使用 `grill-me` 冻结 LFM closure、井角色、适配目标和对外表述。

## 14. 实施顺序

### 阶段 0：producer 与 latent contract

- 定义 `structured_latent_supervision_v1`；
- 统一时间/深度公共字段和 Adapter；
- producer 直接写出 zone `a/b`、三阶段 `c0/c1/c2`、segment supervision 标记和三类
  mask；
- 固结 maximal-state-segment、零对角 transition 和 duration-fraction 离散化合同；
- 更新 manifest identity 和 reader；
- 重新生成结构化训练 benchmark。

### 阶段 1：共享 decoder 与 Oracle

- 从 truth generator 提取确定性对象展开逻辑；
- 实现 NumPy 与 Torch backend；
- 实现 time/depth projection 与 forward round-trip；
- Oracle 全部通过后才开放训练入口。

### 阶段 2：数据与 HSMM

- 建立 parent-owned split 和 trace/zone dataset；
- 实现 train-only normalization；
- 实现 exact HSMM、HMM-like duration 消融、Viterbi 和 backward sampling；
- 用小序列穷举验证所有概率量。

### 阶段 3：最小模型

- 实现两通道 TCN 和 latent upsampling；
- 实现 emission、zone background 和有界 segment parameter posterior；
- 实现 warm-up teacher forcing、边界扰动 schedule、loss、checkpoint 和固定验证；
- 用 development-limited benchmark 完成 smoke run。

### 阶段 4：单 seed 比较

- 运行 Direct、Calibration-prior、LFM-only HSMM、Structured-HMM 和
  Structured-HSMM；
- 生成 holdout、厚度分箱、posterior calibration 和 seismic sensitivity 报告；
- 记录人工机制审查结论。

### 阶段 5：三 seed 或停止

- 机制审查通过则补到三个 seed并生成统一报告；
- 未通过则保留所有可用 artifact，记录失败属于表示、decoder、HSMM、参数 inference、
  seismic sensitivity 或 synthetic prior 的哪一层；
- 不因科学结果差删除或拒绝已有合法 artifact。

## 15. 测试计划

测试继续放在被 `.gitignore` 忽略的 `tests/`，由用户运行。

必须覆盖：

- time/depth latent reader 公共字段与域 Adapter；
- observed、latent、object-supervision mask 严格分离；
- producer 直接发布 `a/b`，consumer 不反推；
- producer 直接发布 raw、profile-projected 和 effective 三参数，decoder 不重复变换；
- maximal constant-state segment 与 object 一一对应，transition 对角严格为零；
- duration fraction 到 time/depth latent sample count 的离散化、归一化和 identity；
- `segment_supervision_valid` 只在整个 segment 均被监督时成立；
- object endpoints、最后对象闭区间和 ξ/ζ 定义；
- NumPy/Torch decoder parity；
- Oracle high-resolution AI、projection 和 forward round-trip；
- 小序列穷举对照 HSMM `logZ`、marginal 和 Viterbi；
- backward sampling 频率与精确 posterior 一致；
- variable-length zone、padding 和 batch mask；
- bounded transformed-normal 的 shape、有限性、硬支持、inverse transform、Jacobian NLL
  和固定 seed 复现；
- encoder tensor 严格只有 seismic/LFM 两通道；
- parent split 无 lateral trace 泄漏；
- normalization 只使用 train clean base；
- Direct、LFM-only、Structured-HMM 和 Structured-HSMM 使用相同 split/parent；
- LFM-only 从头训练且 seismic tensor 恒定置零；
- true segmentation、`±1/±2` 边界扰动和 predicted segmentation 下的参数稳定性；
- 每个 sampled segmentation 独立执行 pooling 和参数 posterior；
- 32 个 posterior samples 的身份和重跑一致；
- zero、shuffle、local replacement 不改变 LFM 和随机身份；
- 科学指标差只产生 warning；
- schema、轴、closure 和 fingerprint 错误明确失败；
- 正式结构化入口不接受旧连续 checkpoint。

## 16. 产物合同

建议版本：

```text
structured_ginn_experiment_v1
structured_ginn_checkpoint_v1
structured_ginn_prediction_v1
structured_ginn_report_v1
```

checkpoint 至少保存：

- architecture 和 posterior parameterization；
- transformed-normal 的 pre-transform 定义、bounds identity 和 inverse epsilon；
- latent、calibration、projection 和 forward identities；
- benchmark、parent split 和 normalization identities；
- active zone 占位符配置；
- loss weights、训练 seed 和 posterior sampling contract；
- best epoch、selection metric、history 和实际 parent/sample 计数。

prediction 至少发布：

- MAP segment table；
- state/boundary marginals；
- background/object posterior tables；
- raw、profile-projected 和 effective 参数表；
- posterior sample index；
- high-resolution AI、reflectivity、projected AI 和 seismic；
- observed/latent axes 和三类 mask；
- 每个输出的 parent、trace、zone 和 checkpoint identity。

## 17. Suggested skills

- `improve-codebase-architecture`：在修改 producer 前确定共享 decoder、latent schema 和
  域 Adapter 的最小边界，避免把 monolithic truth generator 整体搬入训练代码。
- `diagnose`：为 Oracle round-trip、NumPy/Torch parity 和 time/depth parity 建立最小
  可重复反馈环，再处理任何数值差异。
- `grill-me`：合成门禁通过后，用于冻结 external LFM closure、真实井角色、posterior
  adaptation 和真实输出的解释边界。

## 18. 固定假设

- 新结构化 benchmark 由更新后的 producer 重新生成，不迁移旧 artifact；
- 首个 active zone 使用配置占位符，不在公共文档记录真实名称；
- encoder 只有 seismic 和 LFM 两个输入通道；
- v1 使用固定 calibration transition/duration 和 independent bounded
  transformed-normal 参数 posterior；
- posterior realization 数量固定为 32；
- 首轮只做深度域单 zone 实验，但共享基础必须同时支持时间域；
- 首轮止于合成 holdout，不实现真实推理；
- Enhance v2 保留为冻结替代路线，不删除、不实施、不串联。
