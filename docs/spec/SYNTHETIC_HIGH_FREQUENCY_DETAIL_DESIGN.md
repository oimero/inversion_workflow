# Synthetic 高频细节能力设计

> 状态：阶段 1 已完成；阶段 2 被父对象尺度分解支撑问题阻塞
> 范围：Synthoseis-lite 时间域与深度域 field-conditioned 数据的高频细节校准、生成、正演与产物契约
> 本文区分已确定的工程契约和待原型验证的建模候选。阻塞项通过原型验收前，不进入完整生产实现。

## 1. 目标

在 `src/cup/synthetic` 的现有父对象模型内增加一层受井数据约束的高频 detail-event 模型，使同一个地质 realization 同时产出相互配对的低尺度基准、高频增量和完整真值，并分别计算基准真值与完整真值的正演响应。

首版保留当前背景趋势、三态划分、Semi-Markov 父对象序列和三参数对象剖面。高频能力只解释三参数父对象剖面没有拟合的对象内部细节，不重新定义父对象，也不改变现有训练主目标的含义。

核心产物定义为：

```text
base_log_ai   = 当前背景趋势 + 三参数父对象剖面
detail_log_ai = 父对象内部的高频 detail-event 增量
full_log_ai   = base_log_ai + detail_log_ai
```

时间域与深度域共享校准模型结构、生成语义和产物契约。两域的数值 calibration、物理尺度和 realization topology 分别从各自项目输入独立产生，不共享统计数值。

## 2. 非目标

首版不包含：

- 重新分解完整井曲线或重新定义现有三态父对象；
- 对已经被当前三态划分识别为短父对象的高频结构重新归类；
- detail event 沿横向 birth/death、分支、合并或跨父对象延伸；
- 二维 Markov random field、GAN、扩散模型或其他生成式神经网络；
- 将真实井残差片段直接移植、拼接到合成剖面；
- 缺少工区证据时使用内置 generic prior；
- 读取、推断或自动升级旧 calibration、benchmark 或 reader schema；
- 修改 `src/enhance` 或复用其中的逐道 Markov packet 作为生产实现。

`src/enhance` 仅作为历史实验参考。其全局随机数、逐道独立拓扑、减均值闭合和固定事件数语义不属于本设计。

## 3. 术语与数据分层

### 3.1 父对象与 detail event

父对象沿用当前模型的 zone、state、object ID、局部坐标和三参数剖面：

```text
base(x, ξ) = background(x, ξ)
           + c0(x)
           + c1(x) * (2ξ - 1)
           + c2(x) * sin(πξ)
```

其中 `ξ ∈ [0, 1]` 是父对象内部坐标。detail event 完全定义在父对象局部坐标内，并继承父对象的 zone、state、几何事件和横向范围。

一个激活的父对象在允许激活的横向支撑内共享同一套 event 顺序。各 event 的持续长度和振幅可以随横向连续变化，但首版不允许 event 在该支撑内产生、消失、交叉或改变顺序。

这里的 positive、negative 和 silent segment 是特定尺度算子下的 detail event，不宣称对应真实岩性层、真实层厚或地质界面。因此 topology 产物使用 `detail_latent_event_*` 命名，不使用 `microbed_*`。若未来要获得地质微层语义，必须另行设计滤波前 change-point 或 latent bed inference。

### 3.2 base、detail 与 full

- `base_log_ai` 是现有生成器在加入高频能力前产生的 log-AI 语义；
- `detail_log_ai` 是经高频残差算子、边缘处理、幅度恢复和边界调节后的最终增量；
- `full_log_ai` 是进入完整真值正演和训练主目标的 log-AI；
- 三者使用相同的轴、shape、valid mask 和 log-AI 单位；
- 无 detail 的有效位置必须显式写为 0，不使用 NaN 表示未激活；无效区域仍遵循现有 valid mask 契约。

现有 `truth_log_ai_highres` 和 `model_target_log_ai` 在新 schema 中明确表示 full，不改变训练主字段语义。

### 3.3 物理距离与线号

inline/xline 是工区线号，不是数组下标，也不是米制距离。任意线号起点和步长都只参与工区网格定位，不得进入横向相关尺度公式。

所有 detail-event 持续长度和振幅的横向随机场必须使用 section 的 `lateral_m`。`lateral_m` 必须由 `SurveyLineGeometry` 转换出的真实 XY 路径累计距离构建。禁止使用 xline 差、inline 差、线号步长、道序号或数组下标近似物理距离。生产代码不得硬编码任何具体工区的 inline/xline 起点、终点或步长。

## 4. 校准输入与细节提取

### 4.1 输入来源

高频校准从现有对象剖面样本中的 `fit_residual` 开始：

```text
fit_residual = observed_object_residual - fitted_three_parameter_profile
```

该残差可能同时包含薄层、中频欠拟合、测井噪声和处理伪影，因此不能直接作为地质微层真值。正式校准必须先执行域相关的物理尺度分解。

首版的能力边界必须写入 calibration summary：高频统计只描述当前父对象内部的 `fit_residual`，不代表井曲线全部高频能量。

### 4.2 分辨率算子与频谱纯度

首版候选定义低尺度平滑算子 `L` 和高频残差算子 `H`：

```text
H(r) = r - L(r)
```

`L` 是平滑算子，不假设满足 `L² = L`，`H = I - L` 也不称为严格投影。首版不通过反复应用 `H` 将 `L(detail)` 逼近零，以免把能量推向 Nyquist、改变事件宽度或增强振铃。

`L` 必须是零相位、保持轴长度、边界行为明确的局部平滑算子。算子的具体族、参数推导方式、边界模式和实现版本必须通过原型选型后冻结到 calibration 契约中。

**时间域。** 使用冻结时间子波的主周期推导秒制平滑尺度。算子在规则 TWT truth grid 上工作，配置记录主周期提取方法、周期到窗口尺度的比例以及最终秒制和样点制参数。

**深度域。** 使用冻结时间子波与冻结 reference velocity，将主周期转换为随深度变化的局部米制分辨率。reference velocity 在 calibration 阶段由冻结 AI–Vp 关系和明确的 reference log-AI 来源一次性产生。base、detail 和 full 必须使用完全相同的冻结 `L`，不得分别从各自 AI/Vp 重新构造算子。配置记录局部窗口上下限、reference velocity 来源、哈希及算子离散规则。

时间域与深度域不得共享一个无单位的窗口样点数。改变 truth 采样间隔后，物理尺度保持不变，离散窗口随之重新计算。

### 4.3 Detail-event 提取

对 `H(fit_residual)` 使用三态滞回分段：

- 绝对值超过进入阈值时进入正或负 active state；
- 回落到退出阈值以内时返回 silent state；
- 进入阈值必须大于退出阈值；
- 阈值使用稳健噪声尺度的倍数表达；
- 阈值处的等号归属、首尾状态和非有限值行为必须显式定义并测试。

低于最小物理持续长度的 active 或 silent segment 必须按冻结规则合并到相邻 segment。合并依据必须确定且与遍历方向无关；相邻候选同等合理时使用文档化的稳定 tie-break，不得随机决定。

最小 event 持续长度使用局部物理分辨率的比例表示：时间域单位为秒，深度域单位为米。校准产物同时记录物理持续长度和 truth-grid 样点数。该长度描述滤波后事件支撑，不解释为地质层厚。

首版采用 post-`H` 事件口径：校准对象和最终生成目标都是 `H(fit_residual)` 所在的信号空间。每个校准 event 除状态和持续长度外，还保存归一化波形模板、polarity、峰值、RMS、面积和父对象边缘位置。生成器直接合成 post-`H` 波形，不对合成结果再次应用 `H`。

归一化模板使用 event 局部坐标保存，并记录原始物理支撑。模板的对齐点、重采样方法、长度归一化和模板池抽样权重由阶段 1 原型冻结。若模板方法不能在留出对象上复现持续长度、波形与 PSD 联合分布，则阶段 2 不得开始。

### 4.4 统计量

每个可用的 `zone × parent state` 至少校准：

- detail 激活概率；
- 单位父对象物理长度的 transition/event density；
- 正、负和 silent event 的物理持续长度分布；
- 正负振幅分布、detail RMS 和绝对峰值分布；
- 归一化 event shape、面积及 shape 与持续长度的联合分布；
- 父对象首尾的状态、振幅和 taper 支撑统计；
- 井数、空间 cluster 数、父对象数、active 对象数和有效 event 数；
- 三态 event 转移计数与转移概率；
- 各统计量的 evidence level 与实际来源组成。

事件数不得作为不考虑父对象长度的主要统计量。所有幅度统计使用尺度分解后的 log-AI detail，不使用线性 AI 差值。

### 4.5 部分池化与失败规则

`zone` 和 `parent state` 是平行证据来源，不构成固定先后层级。组合统计采用由证据量冻结决定的部分池化：

```text
θ(z,s) = w(z,s)θ(z,s) + w(z)θ(z) + w(s)θ(s) + w(global)θ(global)
```

权重规则按参数类型冻结，并记录每个来源的样本数和最终权重。持续长度、振幅、转移概率和激活率可以使用不同的 evidence 门限，但不得由实现者临时选择偏向 zone 或 state。

必须区分“没有足够观察”和“观察充分且 active 数为零”。前者进入部分池化；后者是该组合的有效零激活证据，不得回退到全工区后重新制造 detail。

全工区仍没有达到冻结最低证据门限时，校准失败。禁止生成通用 detail-event 先验或从 `src/enhance` 读取默认幅度、事件数和持续长度。

### 4.6 门限冻结

以下参数必须由正式配置显式给出并进入 calibration fingerprint：

- 主周期提取方法和分辨率比例；
- 滞回进入、退出阈值的稳健尺度倍数；
- 最小 event 持续长度的局部分辨率比例；
- 激活及各级统计的最低 evidence；
- 平滑算子的实现和频谱纯度门限；
- 父对象边缘 taper 规则；
- detail 低频泄漏、base shift 和代数闭合容差；
- 对象级最小允许振幅缩放比例；
- detail 横向厚度与振幅的相关长度场景值。

诊断工具可以根据井数据输出建议值和敏感性图，但正式校准只读取已冻结配置，不得根据当前批次自动改写门限。

## 5. 二维 detail-event 生成

### 5.1 激活与拓扑

每个父对象根据其最终 evidence model 抽样是否激活 detail。激活后根据父对象 reference trace 的物理长度执行条件 renewal process：

1. 抽样初始 event state；
2. 根据冻结的三态转移矩阵抽样下一状态；
3. 根据 `state × zone × parent state` 条件分布抽样 event 持续长度；
4. 填充至父对象 reference 长度；
5. 最后一个 event 按冻结规则截断；截断后低于最小持续长度时整套 topology 重采样。

滞回提取得到的是最大连续 segment，因此转移矩阵对角线固定为 0；相邻同状态 event 不允许存在。silent 是否位于两个 active event 之间由校准转移概率决定，不使用手写交替规则。

reference trace 使用确定性规则选择：在父对象有效道中，取厚度最接近有效道厚度中位数的道；如有并列，取最小 lateral index。reference trace ID、厚度和选择差值写入对象目录。该规则避免以 wedge 最大厚度端生成系统性偏多的 event。

同一父对象的所有允许激活道共享该 topology。状态顺序固定，event 边界由各道的正持续长度累计得到，因此边界必须有序且不交叉。

父对象过薄、无法容纳满足最小持续长度要求的 topology 时不激活 detail，并记录 `insufficient_parent_support`。这属于由几何决定的显式非激活结果，不从其他统计回填 event。

对 wedge/pinchout，先以父对象横向最小厚度确定允许激活支撑。detail amplitude 在进入不可容纳区之前按冻结 taper 平滑降为 0；不可容纳区的 event ID 为无激活值。最小持续长度只在允许激活支撑内强制执行。该策略允许 detail 支撑随父对象共同终止，但不改变支撑内部的共享 topology。

### 5.2 横向连续性

event duration composition 和振幅 magnitude 分别使用 irregular AR(1) 场扰动。reference trace 上先定义：

```text
p_k = duration_k0 / sum_j(duration_j0)
u_k(x) = log(p_k) + σ_duration * f_k(x)
π_k(x) = exp(u_k(x)) / sum_j(exp(u_j(x)))
duration_k(x) = parent_duration(x) * π_k(x)
amp_k(x) = sign_k * exp(log_magnitude_k0 + σ_amp * g_k(x))
```

该 logistic-normal composition 保证 `duration_k(x) > 0` 且所有 event 严格填满该道父对象。它建模的是 `duration composition | parent duration`，不是对无条件物理持续长度做事后归一化。

`f_k` 和 `g_k` 以 `lateral_m` 为坐标。有效相关长度不得小于冻结的最小采样支撑，并记录 requested、effective 和 empirical correlation length。

detail 的横向相关长度是独立场景参数，不复用宏对象的 `correlation_length_fraction`。场景目录必须能区分宏对象相关尺度和 detail 相关尺度。

生成统计的目标是 `duration composition | parent duration`。原型阶段必须比较条件 renewal process 与 logistic-normal 横向变换后的 event density、持续长度和 composition 联合分布。

### 5.3 raw detail 构造

raw detail 在父对象内部按 latent event topology、post-`H` 波形模板和振幅栅格化。三态顺序只在 topology 创建时抽样一次，各横向道不得重新抽样。符号由 event state 固定，横向振幅场只改变 magnitude，不得跨零改变状态。

raw detail 必须按以下固定顺序处理：

1. 根据校准模板直接合成 post-`H` detail；
2. 父对象边缘 taper；
3. 恢复到抽样得到的目标 RMS 或幅度尺度；
4. 检查频谱纯度和低尺度泄漏；
5. 执行硬物理 AI 边界处理；
6. 重新检查频谱纯度与最小振幅比例。

生成阶段不得再次应用 `H`。taper 和幅度/边界调节仍可能改变事件形态及低频能量，因此原型必须量化处理前后的持续长度、振幅、PSD 和振铃，处理后的复检不得省略。

### 5.4 随机流隔离

detail 必须使用新的命名随机流，至少区分：

```text
detail_activation
detail_topology
detail_duration
detail_amplitude
detail_lateral_duration
detail_lateral_amplitude
```

禁止从全局随机状态取样，也禁止插入 detail 抽样后改变现有宏对象随机流的调用顺序。硬契约为：在 calibration、scenario、realization ID、global seed 和宏对象配置相同的情况下，`detail_enabled=false` 与 `detail_enabled=true` 得到的 base 必须 bitwise identical。

## 6. 闭合与 AI 边界

### 6.1 代数闭合

最终数组必须满足：

```text
full_log_ai = base_log_ai + detail_log_ai
```

代数误差定义为：

```text
algebraic_closure_error = max(abs(full - (base + detail)))
```

容差按 dtype 的数值精度冻结。写入前和 reader 读取后都必须验证。不得先单独裁剪 full 再保留原 detail。

### 6.2 尺度增量与频谱纯度 QC

现有三参数 base 不要求是 `L` 的不动点，因此不得要求 `L(full) ≈ base`。在 base、detail 和 full 共用同一个冻结线性 `L` 时，以下恒等式可用于复算：

```text
L(full_log_ai) - L(base_log_ai) = L(detail_log_ai)
```

普通平滑算子不满足 `L²=L`，因此 `detail = (I-L)r` 不意味着 `L(detail)=0`。`L(detail)` 是允许非零的频谱纯度 QC，不通过迭代高通逼近 0。至少记录：

```text
detail_lowpass_leakage = RMS(L(detail)) / RMS(detail)
base_shift_error = RMS(L(full) - L(base)) / RMS(L(base))
```

这些指标使用原型确定的非零容差。RMS 的 mask、分母 epsilon、无 detail 对象的定义和聚合层级必须冻结。无 detail 对象不参与比值分布，不得用 0/0 伪造通过结果。

`base_shift_error` 和 `detail_lowpass_leakage` 共同描述增量污染；二者都不被表述为严格正交投影闭合。深度域 `L` 是局部非平稳算子，首版不定义基于单一全局 cutoff 的 `detail_lowband_energy_fraction`。若后续引入局部时频指标，必须另行冻结窗口、overlap、taper、局部 cutoff 和边界规则。

### 6.3 AI 软包络与硬边界

现有 zone p01/p99 是井数据统计包络，不是硬物理边界。超过 p01/p99 只记录 exceedance fraction、tail severity 和状态条件分布偏差，不强制缩放。

硬边界必须来自项目配置中具有明确物理含义的绝对 AI 有效范围，并与 p01/p99 分开命名。只有 full 触及硬边界时才允许对 detail 使用统一缩放系数：

```text
detail_adjusted = scale_object * detail
```

缩放粒度由原型比较“全父对象统一缩放”和“保持横向连续的局部 envelope”后冻结；无论选择哪种，都必须保留 event 符号、零点和横向连续性。禁止逐样点裁剪 detail 或 full。

若可行缩放低于配置冻结的最小比例，或者缩放后频谱纯度、硬 AI 边界仍不通过，则拒绝整个 realization。拒绝原因必须区分低频泄漏、最小缩放和硬边界不可行。原型必须报告 detail amplitude 与 base 到软包络/硬边界距离的相关性，防止边界机制系统性吃掉高频。

## 7. 正演与 QC 分层

### 7.1 降采样与正演

base 和 full 必须分别从各自的 high-resolution truth 出发，经过相同域的抗混叠降采样和正演流程：

```text
base_highres → antialias/downsample → base_model → forward → seismic_base
full_highres → antialias/downsample → full_model → forward → seismic_full
```

时间域保存：

```text
seismic_detail = seismic_full - seismic_base
```

`seismic_detail` 是正演响应差，不等同于对 `detail_log_ai` 独立正演，因为反射率到 log-AI 的关系不是线性关系。深度域在 paired-forward 语义冻结前使用下面两个显式诊断名。

时间域使用冻结时间域正演。深度域阶段 1 同时计算两类 paired-forward 诊断：

```text
seismic_detail_self_consistent
    base/full 分别通过同一冻结 AI–Vp 关系派生各自 Vp

seismic_detail_fixed_velocity
    base/full 共用 calibration 中冻结的 reference Vp
```

前者包含阻抗 detail 引起的反射率与运动学共同变化，后者隔离共同速度下的反射率可观测性。原型分别报告两者的事件位置、RMS、相关性和差值能量，再决定阶段 4 benchmark 的主生产字段；在此之前不得把二者笼统命名为唯一的 `seismic_detail`。

full 继续作为 `model_target_log_ai`、nominal seismic 和 observed mismatch 的来源。base 正演用于成对实验与可观测性诊断，不替换现有主观测字段。

### 7.2 Macro QC

Macro QC 只作用于 base，并沿用当前：

- 三参数 profile metric；
- 父对象 state 语义；
- c0 conditioning；
- reversal fraction；
- 父对象几何与厚度约束。

不得用 full 的 detail-event 局部极值评价三参数 profile 是否违规。

### 7.3 Detail QC

Detail QC 至少包含：

- 激活率、事件密度和 event 持续长度分布；
- 正负振幅、RMS、绝对峰值；
- `detail_lowpass_leakage` 和 `base_shift_error`；
- 对象级振幅缩放比例；
- 边缘 taper 支撑；
- latent detail-event 边界有序、非交叉和共享 topology；
- requested、effective、empirical 横向相关长度。

### 7.4 Full QC

Full QC 至少包含：

- AI 软统计包络与硬物理边界；
- 代数闭合；
- high-resolution 到 model-grid 的抗混叠结果；
- base/full 各自正演闭合；
- 时间域 `seismic_detail`、深度域两类 paired-forward difference 可复算；
- 正演有效 mask 和数值稳定性。

## 8. 配置、schema 与产物契约

### 8.1 Schema 升级

实施时必须同时升级 impedance calibration 和 benchmark schema。旧 calibration 和 benchmark 不能进入新生成器，reader 不得根据字段存在性推断版本或自动补默认值。

错误信息必须包含实际 schema、期望 schema 和重新运行 `calibrate`、`generate` 的入口。

### 8.2 Calibration 产物

新 calibration 至少记录：

- detail operator 的域、单位、实现版本和冻结参数；
- evidence model、部分池化来源、权重和样本计数；
- 激活、持续长度、事件密度、转移概率和振幅统计；
- 滞回与最小持续长度规则；
- 频谱纯度、缩放及失败门限；
- 冻结子波与深度域 AI–Vp 输入的路径和哈希；
- detail 配置快照及其 fingerprint。

对象 profile sample 表继续保存原始 `fit_residual`，并新增尺度分解后 detail、滞回状态和最终 segment ID，便于复核事件提取过程。

### 8.3 Benchmark 数组

高分辨率层至少新增：

```text
base_log_ai_highres
detail_log_ai_highres
detail_latent_event_id_highres
detail_latent_event_boundary_mask_highres
detail_active_mask_highres
```

`detail_latent_event_*` 只描述生成器 topology，不声称边界等于最终 detail 波瓣、零交叉或地质界面。`detail_active_mask_highres` 由最终边界调节后的 `detail_log_ai_highres` 按冻结显著性规则计算。若需要评价最终波形事件，必须从最终 detail 重新派生独立标签，不能复用 latent ID。

现有 `truth_log_ai_highres` 表示 `full_log_ai_highres`。

模型网格至少新增：

```text
base_log_ai_model
detail_log_ai_model
seismic_base_model_consistent
seismic_detail_model_consistent
```

`seismic_detail_model_consistent` 是时间域冻结字段。深度域在阶段 1 使用两个诊断字段；阶段 4 根据原型结论确定主字段名称与语义，不提前复用时间域名称掩盖速度口径。

现有 `model_target_log_ai` 和 `seismic_model_consistent` 分别表示 full 模型网格和 full 一致地震。

`detail_log_ai_model` 必须由 `full_model - base_model` 定义并验证，不得假设独立降采样 detail 与该差值在所有边界处理下天然相等。

对象级目录至少新增 detail evidence、激活状态、topology 标识、目标/实际 event 数、振幅缩放比例和频谱纯度指标。realization 级 QC 保存聚合统计和拒绝原因。

### 8.4 Metadata 与 reader

所有新增 HDF5 dataset 必须声明：

- `sample_domain`；
- unit；
- axis path 和 axis order；
- base/detail/full role；
- operator/calibration fingerprint；
- dtype 和 shape 契约。

时间域与深度域 reader 必须显式读取并验证新增字段。manifest、sample index、scenario catalog 和 benchmark fingerprint 必须包含 detail schema、detail 场景参数及直接上游 calibration 契约。

## 9. 严格失败规则

以下情况必须报错或拒绝 realization，不得静默降级：

- 缺少冻结子波、深度域 AI–Vp 关系或其哈希不一致；
- 全工区 detail evidence 不足；
- 分辨率算子单位、采样域或轴不匹配；
- 滞回阈值、最小持续长度或频谱纯度门限缺失；
- 父对象允许激活支撑无法容纳合法 detail-event topology；
- detail-event 边界交叉或横向 topology 不一致；
- 低频泄漏或 base shift 超过门限；
- 硬 AI 边界缩放低于最小比例；
- base/detail/full 代数闭合失败；
- base/full 正演或地震差值复算失败；
- 读取旧 schema、错域、错单位或缺少新增 metadata。

## 10. 实施阶段

### 阶段 1：诊断与参数冻结

- 实现只读诊断，展示各域 `fit_residual` 的物理尺度、尺度分解结果、滞回分段和门限敏感性；
- 验证 post-`H` event 模板能否在不二次应用 `H` 的条件下复现留出对象的持续长度、振幅、波形和 PSD；
- 验证 conditional renewal 与 logistic-normal composition 能否严格填满父对象并复现条件持续长度统计；
- 对比深度域 self-consistent velocity 与 fixed reference velocity 两类 paired-forward 诊断；
- 验证 latent event 标签、最终 active mask 和最终波形事件之间的边界差异；
- 输出建议门限，不修改正式配置；
- 用户根据诊断结果冻结正式参数。

阶段 1 的只读诊断由 `scripts/synthoseis_detail_diagnostic.py` 和 `cup.synthetic.detail_diagnostics` 实现，正式结果见 [阶段 1 结果](SYNTHETIC_HIGH_FREQUENCY_DETAIL_STAGE1_RESULTS.md)。post-H 模板、duration composition、paired-forward 和标签语义已经形成结论；同时发现父对象普遍远短于局部分辨率窗口。连续 well-zone 尺度分解原型通过前不得进入阶段 2–4。

### 阶段 2：共享校准核心

- 增加域相关分辨率算子接口；
- 增加滞回分段、segment 合并、统计和 evidence-weighted 部分池化；
- 升级时间域和深度域 calibration schema 与 QC 产物。

### 阶段 3：共享二维 detail-event 生成

- 增加父对象内共享 topology、条件 renewal process 和 pinchout 支撑策略；
- 增加独立 detail 横向相关场；
- 增加 post-`H` 波形生成、边缘处理、频谱纯度 QC 和硬边界调节；
- 将 Macro、Detail、Full QC 分离。

### 阶段 4：双支路正演与 benchmark schema

- 增加 base/detail/full 高分辨率和模型网格字段；
- 时间域与深度域分别完成 base/full 正演；
- 升级 HDF5、manifest、index 和 reader；
- 保持现有主字段明确指向 full。

### 阶段 5：训练消费与消融

- 在新 benchmark 契约稳定后，再设计 full AI、条件 full AI 和 residual 三类训练任务；
- 训练端不得在缺少显式 role metadata 时猜测输入字段语义；
- 该阶段不属于本设计文档的首轮实现验收。

## 11. 测试规范

测试由实现方编写，用户使用项目指定 Python 环境运行。至少覆盖：

1. 人工父对象上的滞回进入/退出、首尾状态、确定性 tie-break、最小持续长度合并和正负振幅统计。
2. 时间域平稳算子与深度域局部变尺度算子的单位、边界、shape、采样率不变性和固定输入确定性。
3. 固定随机种子下 detail 激活、topology、横向场和部分池化结果完全可复现；启用/禁用 detail 时 base bitwise identical。
4. base/detail/full 在 high-resolution 和 model grid 上满足冻结代数容差。
5. `detail_lowpass_leakage` 和 `base_shift_error` 的解析样例、线性恒等式、mask、非零容差和零 detail 行为。
6. 原型冻结的硬边界调节策略保持 event 符号、零点、横向连续性和代数闭合，且代码路径不存在逐样点裁剪。
7. detail-event 边界有序、非交叉、允许激活支撑内共享 topology；wedge/pinchout 过薄区按冻结 taper 终止且拒绝率有限。
8. detail 横向相关长度独立于宏对象相关长度，requested/effective/empirical 指标可复算。
9. base/full 分别完成抗混叠降采样和正演闭合，`seismic_detail = seismic_full - seismic_base` 可复算。
10. 至少一个 inline 或 xline 使用非单位步长、非零起点的 section fixture 中，相关场只使用米制 `lateral_m`；改变线号编码但保持 XY 路径不变时，生成的横向相关场不变。
11. 时间域与深度域 reader 正确读取新增数组和 metadata。
12. 旧 schema、错域、错单位、错轴、错 fingerprint 和缺失冻结参数明确失败。
13. 全工区实测 detail evidence 不足时校准失败，不产生 generic prior。
14. 同一 realization 的 base/full split、scenario role 和 parent realization ID 保持一致，避免成对数据跨训练/测试集合泄漏。
15. 生成 event density、持续长度、振幅、RMS 和频谱纯度与井校准分布匹配，并包含留出井或留出父对象验证。
16. 时间域使用 `PSD(full)-PSD(base)`，深度域使用与冻结局部算子一致的局部谱诊断；增量不得在低频或 Nyquist 附近异常堆积。
17. 深度域分别记录 self-consistent velocity 与 fixed reference velocity 下的 `RMS(seismic_full-seismic_base)/RMS(seismic_full)` 分布，区分反射率可观测性和运动学贡献。
18. 按 geometry family、硬边界处理和频谱纯度拒绝原因统计接受率，任何正式场景必须达到冻结的最低接受率。

## 12. 验收标准

- 时间域和深度域共享模型结构与 topology 语义，数值 calibration、物理尺度和 realization 分别生成；
- 每个成功 realization 都提供可验证的 base/detail/full 数组和 base/full 正演结果；
- 代数闭合、频谱纯度、AI 软硬边界和三层 QC 均有冻结指标与明确失败行为；
- 现有训练主字段明确表示 full，reader 不猜测字段角色；
- detail 横向连续性完全基于真实米制路径，生产逻辑不包含具体工区线号步长；
- calibration 和 benchmark schema 同步升级，旧产物必须重建；
- 无实测 detail 证据时流程失败，不生成通用兜底数据；
- 所有门限由正式配置冻结，诊断结果不会自动改写生产参数。

## 13. 主要风险与控制

| 风险 | 控制 |
|------|------|
| `fit_residual` 混入噪声和中频欠拟合 | 物理尺度分解、滞回分段、最小持续长度门限和诊断冻结 |
| 当前三态划分已吸收部分薄层 | 在 calibration summary 中明确能力边界，不宣称恢复全部井上高频 |
| 滤波波瓣被误认为地质层 | 使用 detail-event 语义和字段名，不输出地质微层标签 |
| taper 或振幅恢复重新引入低频 | 固定处理顺序并复检频谱纯度，不迭代高通逼近零 |
| full 越界后裁剪破坏闭合 | 区分软统计包络和硬物理边界，禁止逐点裁剪 |
| detail 与宏对象横向尺度耦合 | detail 使用独立场景参数和独立随机流 |
| 线号步长被误当物理距离 | 只接受由 XY 路径构建的 `lateral_m`，使用非单位步长测试 |
| 深度域固定米窗忽略速度变化 | 使用冻结子波和 AI–Vp 关系构造局部米制尺度 |
| 新旧数据字段语义混用 | schema 强制升级、role metadata 和 reader 严格校验 |
| 成对样本在 split 中泄漏 | 所有 base/full/detail 共享 parent realization ID 并整体划分 |
