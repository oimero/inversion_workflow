# 物理约束与残差学习路线大纲

## 1. 研究目标

本路线将波阻抗反演拆成两个尺度明确的任务：

```text
真实地震 + 低频模型
    -> GINN V2：物理约束的主体尺度反演
    -> Enhance V2：井统计约束的高频残差学习
    -> 高分辨率 log-AI
```

GINN V2 恢复地震能够稳定约束的主体结构。Enhance V2 在主体结构上补充井中存在、但地震弱可观测的高频地质纹理。

论文的核心表达为：

> 先以物理正演约束主体尺度，再以残差学习补充高频信息。

## 2. 尺度分解

使用一个米制参数定义 GINN V2 与 Enhance V2 的职责分界：

```text
body_smoothing_fwhm_m = 15.0

body     = gaussian_smooth(full, fwhm_m=body_smoothing_fwhm_m)
residual = full - body
```

`body_smoothing_fwhm_m` 表示主体尺度高斯平滑核的半高全宽：

- GINN V2 输出 `body`，固定平滑层定义主体尺度的最细表达范围，并作为输出尺度保险丝；
- Enhance V2 输出 `residual`，补充井中存在的高频地质纹理。

第一版工作值为 `15 m`。最终数值由现有井尺度分解结果、正演保持程度和主体结构图件共同确定。该尺度以米制物理厚度表达，再由域 adapter 依据采样轴和速度/TDT 离散化。

FWHM 表示连续高斯核的物理宽度，而不是三个采样点的滑动平均。`15 m` 在 `5 m` 网格上对应约 `1.27` 个采样点的高斯标准差：它强烈抑制约 `15 m` 波长的振荡，同时仍会部分衰减约 `30 m` 波长的结构。因此该参数承担明确的主体/残差尺度分工，而不单独承担训练正则化职责。

第七步 LFM 的滤波尺度独立记录为 `lfm_cutoff_wavelength_m`。当前深度域工作值为 `400 m`，其职责是提供低频初始模型，不参与主体/残差分界。

## 3. Wtie 标定资产

Wtie 独立完成：

- 时深关系与井震位置标定；
- 冻结子波；
- 测井预处理与标定质量报告；
- 可信井及其有效区间的确定。

Wtie 的测井滤波参数服务于井震标定。GINN V2 的 `body_smoothing_fwhm_m` 服务于反演尺度控制，两者分别配置。

Wtie 向后续模块提供冻结的子波、对齐测井、采样轴和有效支持，不参与 GINN V2 或 Enhance V2 的参数优化。

## 4. 第六步与第七步适配

### 4.1 第六步：统一井控事实

第六步负责将时间标定或深度平移后的井曲线转换为统一井控事实。它保留两层互补数据：

```text
model-axis control
├── target SampleAxis
├── filtered log-AI
├── inline/xline/XY
└── valid mask

native well control
├── aligned preprocessed full log-AI
├── native vertical coordinates
├── valid mask
└── alignment provenance
```

`model-axis control` 延续当前第六步合同，服务于第七步 LFM。`native well control` 从时间域对齐后的 preprocessed LAS 或深度域 `shifted_preprocessed_las` 读取，服务于 GINN V2 和 Enhance V2。

第六步同时携带上游井震标定的相关性、误差、状态和有效支持。井数据能否进入 GINN 井监督、Enhance 残差库或仅作诊断，由各实验配置显式选择。

第六步发布井事实和 provenance。主体目标与残差由下游按当前 `body_smoothing_fwhm_m` 计算：

```text
well_body_target
    = gaussian_smooth(native_full_log_ai, body_smoothing_fwhm_m)

well_high_frequency_residual
    = native_full_log_ai - well_body_target
```

### 4.2 第七步：低频模型

第七步继续发布两种基线方法：

- `trend`：每井拟合 zone 线性趋势参数，在 XY 米制空间分别克里金参数并重建低频体；
- `proportional_kriging`：在相邻层位间建立比例切片，每个切片独立做 XY 克里金，再沿垂向插值重建。

第一版角色固定为：

```text
proportional_slice_baseline -> GINN V2 主 LFM
trend_baseline              -> LFM 敏感性对照
```

LFM artifact 向 GINN V2 提供 `log_ai`、SampleAxis、有效掩码、variant 身份和实际低通尺度。当前深度域 `lfm_cutoff_wavelength_m=400 m`，与 `body_smoothing_fwhm_m=15 m` 分别配置。

GINN 的 LFM 锚定在第七步 variant 的实际低通尺度上比较：

```text
lowpass(predicted_body, lfm_cutoff_wavelength_m)
    <-> selected LFM variant
```

Enhance V2 以 GINN V2 的 body AI 和第六步井残差为输入，不直接消费第七步 LFM。

## 5. GINN V2：主体尺度物理反演

### 5.1 输出合同

GINN V2 输出确定性的 body-scale log-AI：

```text
u = network(seismic, LFM, lateral context)
x_body = gaussian_smooth(u, fwhm_m=body_smoothing_fwhm_m)
```

物理正演、井监督和最终交付都使用 `x_body`。固定且可微的主体尺度平滑层位于网络输出与正演算子之间，使更细尺度振荡无法用于降低地震拟合损失。

### 5.2 自监督预训练

自监督阶段使用真实地震体，不要求井标签：

```text
横向地震 patch
    -> 将完整中心道替换为固定缺失值，并提供显式 missing-trace mask
    -> 保留邻道地震、中心道 LFM 和横向米制几何
    -> GINN V2 只输出中心道 x_body
    -> 冻结子波正演中心道
    -> 在中心道的上游有效样点重建原始地震
```

主要损失：

```text
L_self
    = masked seismic reconstruction
    + Step-7-scale LFM anchor
```

第一版只使用完整中心道掩码。训练、验证和 checkpoint 选择共享同一套冻结中心道 identity。输入归一化只使用可见邻道，中心道原始地震不参与任何输入特征或归一化统计。

重建 support 直接使用上游发布的中心道观测有效掩码。输入准备、正演和损失之间不派生 halo、erode、taper 或垂向窗口；中心道全部有效样点采用一致权重。这样每个被监督样点都具有相同的输入可见性和损失语义。

第一版使用 inline/xline 二维剖面共享网络权重。横向位置由实际米制坐标表达，xline 步长只用于几何寻址。

### 5.3 半监督微调

半监督阶段继续使用真实地震物理闭环，并在可信井位置加入主体尺度监督：

```text
L_finetune
    = full seismic physics loss
    + well body-scale loss
    + Step-7-scale LFM anchor
```

井监督目标为：

```text
gaussian_smooth(aligned full well log-AI, body_smoothing_fwhm_m)
```

低频锚定在所选第七步 variant 的实际低通尺度上比较预测与 LFM，控制背景漂移。高频井残差不进入 GINN V2 的监督目标。

### 5.4 早停与 checkpoint 选择

早停是 GINN V2 抑制正演多解性过拟合的主要训练机制。固定主体平滑层提供表达尺度保险，早停负责在网络开始利用剩余自由度追逐正演模型误差之前结束训练。

训练每个 epoch 保存可独立推理的 checkpoint，并分别记录：

- 留出空间块上的波形误差和相关性；
- 可信井上的 body-scale log-AI 误差；
- 预测 body 的短波能量和垂向粗糙度；
- 相对所选 LFM 的低频漂移。

正式 checkpoint 使用“最早可接受”规则选择：

1. 留出波形拟合达到预先冻结的可接受区间或进入平台；
2. 可信井 body-scale 指标处于已达到的最佳容差范围内；
3. 短波能量、粗糙度和低频漂移保持在冻结上限内。

自监督预训练以冻结 validation center traces 的整道重建平台触发早停。半监督微调同时使用留出地震和可信井 body-scale 指标。波形相关性是闭环充分性指标，不单独决定最佳 checkpoint；同等满足条件时选择 epoch 更早的模型。

### 5.5 GINN V2 交付

- body-scale high-resolution log-AI；
- model-grid log-AI；
- 正演地震及其与真实地震的闭环指标；
- 有效支持与井锚支持；
- inline/xline 推理结果及方向分歧。

## 6. Enhance V2：高频残差学习

### 6.1 残差定义

可信井上的监督残差为：

```text
r_well
    = full well log-AI
    - gaussian_smooth(full well log-AI, body_smoothing_fwhm_m)
```

这些残差构成高频地质纹理样本库。样本保留物理厚度、振幅、所在层段和有效支持。

### 6.2 训练样本

训练阶段沿用残差增强思路：

```text
base body AI + sampled well residual patch -> enhanced target AI
```

井残差贴片按物理尺度抽取，并通过平移、幅度变化和有限尺度变形扩充。随机贴片只负责构造训练配对，网络在完整剖面或体数据上学习连续 residual field。

### 6.3 网络任务

```text
r_pred = residual_network(body AI, structural context)
x_full_pred = body AI + r_pred
```

第一版以 GINN V2 body AI 为主输入。地震、正演地震或波形差异可作为后续可选条件通道，但不承担逐贴片择优功能。

网络直接预测 `delta_log_ai`。同一个卷积网络作用于完整剖面或体数据，使增强结果继承 base AI 的横向结构，并形成连续的超分辨率效果。

### 6.4 Enhance V2 交付

- predicted high-frequency residual；
- enhanced high-resolution log-AI；
- body 与 residual 的尺度分账；
- 增强前后正演地震对比；
- 井位残差幅度和纹理对比；
- 横向连续性图件。

## 7. 时深统一管线

GINN V2 和 Enhance V2 均采用一套时深统一实现。训练循环、网络、损失、井监督、checkpoint 结构、指标和推理 interface 在两个域中保持一致，域差异集中在内部 adapter。

共同采样轴合同至少包含：

```text
SampleAxis
├── sample_domain        # time | depth
├── sample_unit          # s | m
├── coordinates
├── sample_interval
├── positive_direction
└── depth_basis          # depth 时固定为 tvdss
```

两个正式 adapter 为：

```text
TimeDomainAdapter
├── TWT/s 采样轴
├── 定常时间褶积正演
├── Vp 或 TDT 驱动的主体平滑宽度换算
└── 时间域井曲线与层位准备

DepthDomainAdapter
├── TVDSS/m 采样轴
├── Vp 相关的非定常深度正演
├── 直接离散米制主体平滑宽度
└── 深度域井曲线与层位准备
```

两个域都使用以秒为横轴的冻结时间子波。深度 adapter 依据 TVDSS 轴和 Vp 在内部构造非定常正演算子。

`body_smoothing_fwhm_m` 是跨域共享的科学参数：

```text
body_smoothing_fwhm_m -> 深度轴上的局部平滑宽度
body_smoothing_fwhm_m + Vp/TDT -> 时间轴上的局部 TWT 平滑宽度
```

时间域缺少体速度或 TDT 时，配置可以显式使用秒制尺度；相应 checkpoint 和 artifact 记录该尺度语义。正式结果的 checkpoint 始终记录训练域、采样单位、尺度单位和 forward adapter 身份。

统一训练链为：

```text
domain adapter.prepare(raw inputs)
    -> common observation batch
    -> shared neural network
    -> vertical scale adapter.smooth_body()
    -> shared body/residual composition
    -> forward adapter.forward()
    -> shared loss and reporting
```

数据模块以带轴对象承载单道数据，以显式 `SampleAxis` 承载 batch 和体数据。横向距离使用实际米制坐标；inline/xline 线号及 xline 步长只承担几何寻址。

当前深度/TVDSS 工作流作为正式生产 adapter。时间/TWT adapter 使用独立小 fixture 覆盖数据准备、尺度换算、训练 batch、正演、损失、推理和 artifact 写读的相同 interface。

## 8. 模块 seam

三个模块通过少量稳定 interface 连接：

```text
TieAssets = Wtie.calibrate(well, seismic)

BodyResult = BodyInverter.predict(
    seismic,
    lfm,
    wavelet,
    geometry,
)

EnhancedResult = ResidualEnhancer.enhance(
    body_result,
    geometry,
)
```

完整中心道掩码、尺度离散化、正演调用、井残差抽样和剖面拼接分别隐藏在模块实现中。调用方只处理带采样轴、单位、支持和几何语义的结果对象。

## 9. 实施顺序

1. 建立共同采样轴、common batch、time/depth forward adapter 和 vertical-scale adapter。
2. 扩展第六步 native well control，并锁定第七步主 LFM 与敏感性 LFM 的 artifact 合同。
3. 固定 `body_smoothing_fwhm_m` 和可微高斯算子，建立 GINN V2 的 body-scale 正演闭环。
4. 冻结波形充分性、井尺度误差、短波能量、粗糙度和低频漂移的 checkpoint 选择合同。
5. 使用真实地震剖面完成掩码物理自监督预训练，并按最早可接受规则早停。
6. 加入可信井 body-scale 监督，完成半监督微调与确定性真实工区推理。
7. 从对齐井建立高频残差样本库，复刻并简化残差增强训练链。
8. 训练 Enhance V2，输出完整剖面的连续高频 residual field。
9. 联合报告 body、residual、最终 log-AI 和正演闭环结果。

## 10. 第一版判断标准

### GINN V2

- 真实地震闭环相关性和误差达到当前物理反演基线；
- 输出中无明显小于 `body_smoothing_fwhm_m` 的锯齿；
- 可信井上的 body-scale AI 优于 LFM；
- 掩码预训练后横向连续性优于直接半监督训练。
- 最终交付来自满足波形、井尺度、粗糙度和低频漂移合同的最早 checkpoint。

### Enhance V2

- 增强结果具有明显而连续的井统计高频纹理；
- residual 振幅和主导尺度落在可信井残差范围内；
- 横向图件表现为对 body AI 的连续超分辨增强；
- 增强后的正演闭环保持在可接受范围内。

### 联合结果

- body 与 residual 的贡献能够分别展示；
- 井位、剖面和体数据使用同一套尺度与推理合同；
- time/depth fixture 通过相同的模型、损失、正演和 artifact interface；
- 域相关代码集中在 time/depth adapter，训练和推理模块不包含分散的域判断；
- 最终结果同时具备地震约束的主体结构和井先验提供的高频纹理。
