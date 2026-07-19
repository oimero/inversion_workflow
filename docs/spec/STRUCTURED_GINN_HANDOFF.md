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
- Oracle decoder round-trip；
- 深度域、单道、单个可配置 zone 的合成监督原型；
- Prior-only、Direct、Structured-independent 和 Structured-HSMM 比较；
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
duration / thickness_fraction
c0 / c1 / c2
profile_projection_scale
c0_conditioning_adjustment
```

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
segment c0/c1/c2
calibrated profile and AI bounds
```

对 zone 内坐标 `ζ` 和对象内坐标 `ξ`，输出固定为：

```text
background = a + b * (2ζ - 1)
profile    = c0 + c1 * (2ξ - 1) + c2 * sin(pi * ξ)
log_ai     = background + profile
```

decoder 必须复用 producer 的对象端点、最后对象闭区间、profile projection、c0
conditioning 和 AI bounds 语义。NumPy 和 Torch backend 对相同输入逐点等价。

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

split 继续使用固定 parent identity、seed 和 geometry holdout 语义。Direct、Prior-only、
Independent 和 HSMM 共用同一 split。

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

### 9.4 参数 posterior

每个 zone 输出 `a/b` 的对角高斯；每个 segment 输出 `c0/c1/c2` 的对角高斯：

```text
background: mean_a, log_std_a, mean_b, log_std_b
object:     mean_c0, log_std_c0, mean_c1, log_std_c1,
            mean_c2, log_std_c2
```

分布在 calibration 标准化空间中参数化，输出和样本必须落在相应已标定支持范围内。
参数 head 使用 segment 内 latent feature pooling、state 和归一化 segment extent；不在
每个高分辨率网格点自由预测三参数。

训练时参数 head 使用真实 segmentation teacher forcing。验证和预测使用 MAP 或采样
segmentation，不使用真实边界。

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
projected log AI 和 synthetic seismic。

## 10. 训练合同

训练只使用 clean base synthetic parent。首版包含四个独立监督量：

1. true segmentation 的 segmental HSMM NLL；
2. true zone `a/b` 的 Gaussian NLL；
3. true segment `c0/c1/c2` 的 Gaussian NLL；
4. teacher-forced 参数均值经过 decoder 后的 high-resolution、projection 和 forward
   reconstruction loss。

各项先按自己的自然计数归一化：latent sample、zone parameter、object parameter、有效
AI sample 和有效 waveform sample。所有权重显式写入配置、manifest、history 和
checkpoint；不依据 batch 内对象数隐式改变权重。

increment L2 不进入结构化训练。高频不可见部分通过固定 HSMM 和参数 posterior 约束，
不通过把连续 increment 压向零来约束。

checkpoint selection 使用固定的合成 latent validation score。score 由归一化后的
segment NLL、background NLL 和 object-parameter NLL 显式加权组成；high-resolution、
projection、forward 和 sensitivity 指标只报告，不参与首轮选模。所有组使用相同
selection 定义和 validation parent。

## 11. 固定实验矩阵

### Oracle decoder

不训练。使用真实 segmentation、`a/b` 和 `c0/c1/c2` 完成 round-trip，验证 artifact
和共享 decoder 的理论上限。

### Direct

复用当前连续 model-grid canonical increment 架构，但使用相同新 benchmark、clean base、
parent split、normalization 和训练预算。它不输出对象 posterior；只比较 model-grid AI、
seismic closure 和上采样后 high-resolution AI 的有限基线。

### Prior-only

不读取 seismic。segmentation、duration、state 和参数直接来自固定 calibration prior，
LFM 仅作为背景条件。输出同样的 MAP、marginal 和 32 个 realization，是判断 seismic
新增信息量的正式基线。

### Structured-independent

使用相同 encoder 和参数 posterior，但三状态在 latent grid 上逐点独立分类。MAP 状态
经过 run-length 编码形成 segment；不使用 transition 或 duration prior。

### Structured-HSMM

使用相同 encoder 和参数 posterior，emission 进入固定 calibration HSMM。该组是首轮
候选主模型。

实验首先运行一个训练 seed。人工机制审查通过后，固定 benchmark、split、normalization、
训练预算、validation 和报告合同，再补到三个训练 seed。新增 seed 不改变 generator seed。

## 12. 报告与科学门禁

### 12.1 合成 holdout

按 parent、scenario、state、对象厚度和相对调谐尺度报告：

- state accuracy、macro F1 和 posterior NLL；
- object count error；
- boundary precision、recall、F1 和最近边界距离；
- duration NLL、MAE 和 credible-interval coverage；
- `a/b/c0/c1/c2` NLL、MAE、RMSE 和 coverage；
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

- clean seismic 必须相对 Prior-only 提供可报告的 latent 改善，才可称为 seismic-conditioned
  object inference；
- seismic 置零后 posterior 应向 Prior-only 靠近；
-平静波形区域允许 posterior 变宽，不能把稳定 prior realization 描述为地震恢复；
-只改善正演闭合而不改变 segmentation/duration posterior，表示 seismic 主要约束连续
  参数，而没有识别薄层组合。

### 12.3 与基线比较

固定报告：

```text
Structured-HSMM - Prior-only
Structured-HSMM - Structured-independent
Structured-HSMM - Direct
Oracle - Structured-HSMM
```

这些差值分别回答 seismic 新增信息、半马尔科夫约束价值、结构化表示价值和 latent
inference 剩余空间。

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
- Prior-only 与 HSMM 差异小；
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
- 可信井的留井 posterior predictive 是否优于 Prior-only；
- 低质量标定井是否仅保留非聚合诊断角色；
- 参数是否贴住 synthetic support 边界，真实数据是否明显分布外；
- 多 seed 的主要边界是否稳定，不确定性是否随地震信息强弱变化。

真实阶段开始前必须再次使用 `grill-me` 冻结 LFM closure、井角色、适配目标和对外表述。

## 14. 实施顺序

### 阶段 0：producer 与 latent contract

- 定义 `structured_latent_supervision_v1`；
- 统一时间/深度公共字段和 Adapter；
- producer 直接写出 zone `a/b` 和三类 mask；
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
- 实现 exact HSMM、Viterbi 和 backward sampling；
- 用小序列穷举验证所有概率量。

### 阶段 3：最小模型

- 实现两通道 TCN 和 latent upsampling；
- 实现 emission、zone background 和 segment parameter posterior；
- 实现 teacher forcing、loss、checkpoint 和固定验证；
- 用 development-limited benchmark 完成 smoke run。

### 阶段 4：单 seed 比较

- 运行 Direct、Prior-only、Independent 和 HSMM；
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
- object endpoints、最后对象闭区间和 ξ/ζ 定义；
- NumPy/Torch decoder parity；
- Oracle high-resolution AI、projection 和 forward round-trip；
- 小序列穷举对照 HSMM `logZ`、marginal 和 Viterbi；
- backward sampling 频率与精确 posterior 一致；
- variable-length zone、padding 和 batch mask；
- Gaussian posterior shape、有限性、范围和固定 seed 复现；
- encoder tensor 严格只有 seismic/LFM 两通道；
- parent split 无 lateral trace 泄漏；
- normalization 只使用 train clean base；
- Direct、Independent 和 HSMM 使用相同 split/parent；
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
- latent、calibration、projection 和 forward identities；
- benchmark、parent split 和 normalization identities；
- active zone 占位符配置；
- loss weights、训练 seed 和 posterior sampling contract；
- best epoch、selection metric、history 和实际 parent/sample 计数。

prediction 至少发布：

- MAP segment table；
- state/boundary marginals；
- background/object posterior tables；
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
- v1 使用固定 calibration transition/duration 和对角 Gaussian 参数 posterior；
- posterior realization 数量固定为 32；
- 首轮只做深度域单 zone 实验，但共享基础必须同时支持时间域；
- 首轮止于合成 holdout，不实现真实推理；
- Enhance v2 保留为冻结替代路线，不删除、不实施、不串联。
