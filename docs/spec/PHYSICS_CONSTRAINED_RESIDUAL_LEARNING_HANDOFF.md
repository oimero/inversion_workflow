# 物理约束与残差学习实施 HANDOFF

## 1. 用途

本文将 `PHYSICS_CONSTRAINED_RESIDUAL_LEARNING.md` 转换成可连续实施的阶段。阶段只在以下位置结束：

- 需要人工查看井、剖面或体数据图件并作科学判断；
- 需要用户运行长时间训练或全体积推理命令。

每个阶段由执行代理完成代码、短 smoke、错误修复和受限训练循环。长时间命令可以由执行代理持续监控；用户只在阶段产物通过自动门禁后检查最终图件，或在受限循环耗尽后决定是否扩大搜索范围。对应实现尚未存在时，HANDOFF 只规定命令职责；实际脚本和参数必须在 smoke 通过后给出。

## 2. 固定科学合同

- 当前生产域为深度/TVDSS；时间/TWT 使用相同 interface 的小 fixture 验证。
- `body_smoothing_fwhm_m=15 m` 定义 GINN V2 body 与 Enhance V2 residual 的尺度分工。
- `lfm_cutoff_wavelength_m=150 m` 是第七步 LFM 尺度，与主体平滑分别配置。
- 第六步发布 model-axis control 和 native well control。
- 第七步以比例切片克里金为主 LFM，以趋势模型为敏感性对照。
- 所有成功井控用于第七步 LFM；神经网络井监督和残差库分别使用训练前冻结的显式井名单，且只监督原始观测有效样点。
- 冻结子波保持标定相位和极性，并做单位能量归一化。自监督和半监督共用振幅不变的波形形状 loss；原始振幅 MSE、解析增益和 raw-amplitude residual 只作诊断。
- 自监督只掩盖完整中心道。中心道不参与输入归一化；损失侧可以在上游有效样点上归一化目标道，不派生 halo、erode、taper 或垂向窗口。
- GINN V2 输出确定性 body-scale log-AI；Enhance V2 输出确定性高频 residual field。
- GINN 每个 epoch 保存 checkpoint，使用最早可接受规则选择模型。
- xline 步长只用于几何寻址；横向运算使用实际米制坐标。

## 3. 阶段 0：基础合同与输入资产

### 执行代理连续完成

- 建立共同 SampleAxis、common batch、time/depth forward adapter 和 vertical-scale adapter。
- 实现 `body_smoothing_fwhm_m` 的可微高斯算子及 NumPy/Torch 对照。
- 扩展第六步 artifact，使同一口井同时发布 model-axis control 和 native well control。
- 固定第七步主 LFM、敏感性 LFM、实际低通尺度和有效支持的读取合同。
- 建立最小 GINN body 输出到冻结正演的闭环，但不开始正式训练。
- 执行代理运行深度 smoke 和时间 fixture，修复所有可复现错误。

正式命令链固定为：

```text
scripts/real_field_well_controls.py
    -> scripts/real_field_lfm.py
    -> 后续 GINN 训练入口
```

第六步和第七步各自发布正式 artifact。阶段 0 的检查直接读取这些 artifact，不设置独立的生产脚本或中间 artifact。

### 交给用户的人工门禁

用户查看第六步和第七步正式产物已有的检查图，确认：

- native full log-AI 与 model-axis filtered log-AI 对齐正确；
- 比例切片 LFM 和趋势 LFM 的层位、支持与低频幅度合理；
- `15 m` body/residual 分解仍符合已有井尺度实验的视觉判断；
- 进入 GINN 井监督、Enhance 残差库和仅诊断三种角色的显式井名单。

用户确认后，阶段 0 完成。该阶段没有长训练命令。

## 4. 阶段 1：GINN V2 主体反演与受限训练循环

### 4.1 训练模块 interface

执行代理只调用一个正式训练入口。模块内部完成：

```text
固定 baseline 与 split
    -> 完整中心道掩码 batch
    -> 中心道可见的地震物理 batch
    -> 可信井 body-scale batch
    -> 每个 epoch 的统一评估
    -> 最早可接受 checkpoint
    -> 固定井剖面和 blind section 图件
```

同一个网络处理两种中心道可见性，并显式接收 missing-trace mask：

- 掩码 batch：完整中心道不可见，只学习邻道、LFM 与地震形状之间的映射；
- 混合半监督 batch：中心道可见，同时保留掩码 batch，加入可信井绝对 log-AI 和第七步尺度 LFM anchor；
- 生产推理：中心道可见，与混合半监督 batch 的输入语义一致。

训练固定为一次共享预训练和一次最多三个微调 epoch：

```text
pretrain epoch 1       masked pretraining，发布共同 checkpoint
finetune epoch 1–3     masked + visible-center + trusted-well 三路 batch 轮换
```

每个 epoch 都发布可独立恢复的 checkpoint、统一指标和固定 validation 图件。训练进程不得只把最佳权重保存在内存。

### 4.2 执行代理连续完成

- 实现横向 patch reader、完整中心道缺失输入、中心道可见输入和显式 missing-trace mask。
- 网络只预测中心道 body log-AI；归一化冻结子波只正演该中心道。
- 实现中心道支持内的振幅不变波形形状 loss，以及不反向传播的非负解析增益诊断。
- 从第六步 native well control 的原始观测有效样点生成 `15 m` body-scale 井监督目标。
- 冻结井角色名单：`2-ANP-2A-RJS`、`L1-NW1`、`L5-NW5`、`L9-NW4A`、`NW11` 五口井进入 body loss；`L3-NW2A`、`L6-NW3A`、`NW7`、`NW8` 只作诊断；全部成功井继续用于第七步 LFM。`NW8` 的目的层井震相关性低，不以目的层上方密度常值补齐后的相关性作为监督依据。
- 输入保留完整时窗作为上下文；地震闭环、LFM anchor、井 body loss、checkpoint 指标和图件统一使用第七步 LFM 发布的 `base_of_salt` 至 `base_of_itp` 目的层 mask。mask 不做 halo、侵蚀或过渡带。
- masked pretraining 只运行一次并发布共同 checkpoint；所有半监督候选从该 checkpoint 分支，预训练学习率为 `1e-3`，微调学习率为 `2e-4`。
- masked/visible batch 使用 seismic shape 与 LFM anchor；trusted-well batch 使用按井 body 标准差归一化的 well-body loss、物理坐标导数项与 LFM anchor，并可在配置启用时以低权重拟合该井低相关的井旁地震波形。
- LFM anchor 直接复用所选第七步 variant 记录的 Butterworth 阶数、截止频率和边界处理，不用同名 Gaussian 尺度替代。
- 冻结空间训练/验证块和 validation center-trace identities。五口可信井目的层有效样点全部进入 well loss，不在井内保留留出区间；泛化检查由固定剖面和远井区域完成。
- 实现一个命令内的共享预训练、分支微调、逐 epoch 评估、固定井剖面和 blind section 推理。
- 使用极小数据完成 masked、visible-center、trusted-well 三种 batch 的前向/反向和端到端 smoke。

### 4.3 冻结 baseline 与验收指标

正式训练前只计算一次 `LFM-only` baseline，随后所有微调 epoch 复用同一份 baseline、split 和 validation identity。每个 checkpoint 必须报告：

- masked validation center traces 的相关性、归一化 shape loss 和逐道改善率；
- visible-center validation traces 的相关性和归一化 shape loss；
- 可信井全部有效样点上的 body log-AI RMSE、绝对 bias 和逐井改善；
- `150 m` 低通后的 LFM drift；
- 小于 `body_smoothing_fwhm_m` 的短波能量和相对可信井 body target 的垂向粗糙度；
- 非负解析增益与 raw-amplitude residual，仅作诊断；
- 非有限值、support 连续性和 inline/xline 方向分歧。

预训练门禁（相对 LFM-only baseline）：

1. masked 相关性中位改善至少 `0.01`，归一化 shape loss 不高于 baseline 的 `99%`；未达到时停止，不进入半监督。

半监督验收门禁：

1. masked validation 相关性相对共同 masked-pretrain 的中位下降不超过 `0.01`；
2. pooled body RMSE 优于共同 masked-pretrain checkpoint，相对比值不超过 `1.0`。

以下指标照常计算、随每个 checkpoint 发布，但不参与验收：visible-center 相关性、归一化 shape loss、低频漂移、短波能量、垂向粗糙度、解析增益、raw-amplitude residual、support 连续性与方向分歧。原始振幅 MSE 不参与验收；自监督 epoch 不单独要求绝对 body 对比度。

### 4.4 逐 epoch 停止与 checkpoint 选择

执行代理在每个 epoch 结束后立即评估并保存 checkpoint。单次微调跑满配置的微调 epoch 数，不提前停止：

- 每个 epoch 统一评估、门禁判定、checkpoint 与五口井 waveform QC 图件；
- 所有通过 masked 相关性保持门禁的 epoch 为候选，没有任何 epoch 通过时按诊断发布 `completed_not_accepted`；
- 候选 epoch 中固定选择按归一化 well RMSE 排序最早的 checkpoint，不以最高地震相关性覆盖较早模型；
- 出现非有限值、support 错位、中心道泄漏或 checkpoint 无法恢复时，按实现错误处理，修复并重新 smoke。

### 4.5 单次微调

masked pretraining 只运行一次。半监督微调从共同 checkpoint 开始，按配置的 batch 比例 `masked : visible : trusted-well = 1 : 1 : 2` 单次运行配置的微调 epoch 数，不自动调整 loss 权重，不修改网络架构、井名单、split、子波、`15 m` body 尺度、第七步 LFM 滤波合同、增益模型或训练数据。

微调结束后发布 `selected_checkpoint.json`、固定井剖面和 blind sections，并冻结 GINN V2 body checkpoint；没有任何 epoch 通过验收门禁时发布 `completed_not_accepted` 和最佳诊断 checkpoint，不抛出训练异常。

### 4.6 阶段结束

自动门禁通过后，用户只需查看最终 review package：

- 可信井附近的 body-scale AI 相对 LFM 是否具有合理增量；
- 井监督区与非井区是否保持一致预测语义；
- 剖面是否无明显锯齿、方向条带或逐道跳变；
- 最早可接受 checkpoint 是否优于更晚的高相关 checkpoint。

真实工区全体积推理留到阶段 3。

## 5. 阶段 2：Enhance V2 残差学习

本阶段由执行代理连续完成代码、smoke、正式训练、一次受限调整和最终图件。残差
atlas 在训练开始前生成，但本轮不作为暂停点；用户第二天直接检查完整 review package。

### 5.1 冻结输入

当前正式实现使用以下冻结产物：

```text
Step 6 filtered well controls
    scripts/output/real_field_well_controls_20260812_203252

Step 7 primary LFM
    scripts/output/real_field_lfm_20260812_205054
    variant = proportional_slice_baseline

GINN V2 body model
    experiments/ginn_v2/results/run_20260812_stage1_final/selected_checkpoint.pt

GINN V2 config
    experiments/ginn_v2/ginn_v2.yaml

depth forward inputs
    scripts/output/depth_forward_model_inputs_20260719_172553/forward_model_inputs.json
```

残差库固定使用五口 GINN body 监督井：

```text
2-ANP-2A-RJS
L1-NW1
L5-NW5
L9-NW4A
NW11
```

第六步的 native 曲线已经是 Wtie 标定级 filtered log-AI。残差库以该 native 曲线为唯一
full log-AI 来源；主体输入由上述冻结 GINN checkpoint 生成。

### 5.2 残差与组合合同

定义物理尺度算子：

```text
S15(x) = physical Gaussian smoothing with FWHM = body_smoothing_fwhm_m
H15(x) = x - S15(x)
```

井残差库保存纯纹理：

```text
well_residual = H15(filtered_full_well_log_ai)
```

```text
training_target    = frozen_ginn_body + sampled_well_residual
predicted_residual = H15(raw_network_output)
enhanced_log_ai    = frozen_ginn_body + predicted_residual
```

训练直接复刻已经获得良好观感的“向 base AI 贴入井残差”实验。`H15` 输出投影保证 Enhance
只表达约定尺度的纹理。井上评估分别报告：

```text
GINN body                vs S15(filtered full well)
predicted residual       vs H15(filtered full well)
enhanced full curve      vs filtered full well
```

### 5.3 新生产模块与唯一入口

Enhance V2 的源码包为 `src/enhance`。生产链使用当前第六步井控、冻结 GINN body 和物理
尺度残差库。

模块对外只保留两个主要 interface：

```python
run = train_enhancer(config)
result = enhance_sections(checkpoint, frozen_body_sections)
```

内部负责：

- 从第六步 native filtered well control 构建米制残差库；
- 加载冻结 GINN checkpoint 并生成训练、验证和 review body section；
- 构造配对残差样本；
- 训练、逐 epoch 评估和 checkpoint 保存；
- 确定性 section 推理和 review package。

正式入口和配置固定为：

```text
scripts/enhance_v2.py
experiments/enhance_v2/enhance_v2.yaml
experiments/enhance_v2/train_network.ps1
```

PowerShell 脚本接收一个结果标识，将产物写入：

```text
experiments/enhance_v2/results/<run_id>
```

脚本是本阶段唯一夜间入口，内部依次执行：

```text
prepare residual library
    -> smoke
    -> first formal training
    -> evaluate all epochs
    -> optional single adjustment
    -> final review package
```

代码 Agent 完成实现和静态检查后，必须在同一任务中立即执行：

```powershell
.\experiments\enhance_v2\train_network.ps1 20260813_nightly
```

不得以“代码已完成、等待用户运行”为结束条件。只有上述命令发布最终状态，或发生无法由
实现修复解决的外部输入缺失时，本阶段任务才结束。

### 5.4 残差库与配对训练样本

每口井在 native 物理轴上计算 `S15/H15`，随后按目标 `SampleAxis` 重采样残差 patch。
残差库至少保存：井名、native 轴、目的层支持、残差值、物理跨度、RMS、绝对振幅分位数
和来源层段。全部五口井都参与残差学习。

首轮数据预算写入 YAML，默认值为：

```text
training_pairs = 2048
fixed_monitor_pairs = 256
batch_size = 16
formal_epochs = 4
maximum_adjustment_runs = 1
adjustment_epochs = 2
```

训练与 monitor 的 base center identities 分开冻结。这里的 monitor 只负责比较各 epoch 和零
residual baseline，不建立新的数据生产旁路。

训练样本以冻结 GINN body 为 base。第一版延续旧 Enhance 的确定性超分辨思路。每个固定
配对样本按以下顺序构造：

```text
frozen GINN body
    + sampled well residual
    -> enhanced target log-AI
    -> frozen time/depth forward difference

delta seismic
    = forward(enhanced target log-AI)
    - forward(frozen GINN body)

injection gain
    = non-negative analytic gain between
      forward(frozen GINN body) and observed real seismic

paired training seismic
    = observed real seismic + injection gain * delta seismic

paired training seismic + frozen GINN body
    -> residual network
    -> sampled well residual
```

该配对方式保留真实地震的噪声、频谱和处理特征，只用冻结正演注入已知 residual 对波形造成
的增量。网络由 paired training seismic 获得与 residual 对应的条件，不能只根据 body 学习
一个与输入无关的随机 donor。地震输入使用与 GINN 相同的逐道波形形状标准化，不用原始
绝对振幅决定 residual 尺度；injection gain 只把正演差分映射到该道观测振幅尺度，在配对
生成时解析计算并冻结，不参与网络预测或反向传播。残差尺度由 residual 标签与井 residual
anchor 约束。训练样本中
固定 `20%` 为零注入，即 `delta seismic=0`、目标 residual 为零，避免网络在所有位置强制
生成纹理。井残差 patch：

本实施固定启用主规格为 Enhance 预留的 seismic 条件通道；GINN body 仍是主体结构输入，
seismic 负责区分不同 residual 注入后形成的训练目标。

- 按物理跨度截取和重采样，不使用固定样点数表达尺度；
- donor 随机，正式训练前冻结每个 base patch 的 donor identity、平移、振幅和尺度参数；
- 允许有限的平移、正比例振幅变化和物理跨度变化；所有范围写入 YAML；
- 相邻 review traces 使用共享 donor 和随横向位置平滑变化的平移、振幅参数；禁止逐道独立
  抽取 donor；
- patch 超出目的层或有效支持时只截断到合法范围，不在目的层外生成 residual。

目的层支持直接采用第七步发布的 mask；残差 loss 和组合结果均不派生 halo、侵蚀、taper 或
过渡带。

网络第一版使用共享的一维垂向卷积模型预测中心道 residual。真实推理输入相同标准化合同下
的真实地震，同一个模型直接作用于完整 section；横向连续性继承连续的真实地震、GINN body
和共享卷积映射。训练和推理都以一维垂向网络为单位。

训练输入至少包含：

```text
normalized seismic waveform
normalized frozen GINN body
physical vertical derivative of frozen GINN body
target-zone/support mask
```

输出只包含一个 `delta_log_ai` 通道，并在组合前经过 `H15`。五口井的直接 residual anchor
使用井位真实地震、冻结 GINN body 和 `H15(filtered full well)` 构成监督样本。正演结果同时
进入增强前后 review package。

网络、损失、训练循环和 checkpoint 在时间域与深度域共用。域差异只由 `SampleAxis`、物理
尺度换算和 forward adapter 处理；当前正式训练使用 TVDSS/depth，smoke 另运行一个 TWT/time
小 fixture。xline 步长 `4` 只用于定位 trace，所有横向距离读取米制坐标。

### 5.5 残差库图件与 smoke

执行代理先生成：

```text
residual_library.npz
residual_library_summary.json
review_package/residual_atlas/<well>/residual_atlas.png
```

每口井的 atlas 使用直接可读的三轨图：filtered full、`15 m` body、high-frequency
residual；另选少量目的层窗口展示 full/body 叠合和 residual，不生成复杂频谱审计图。

随后运行自动 smoke：

1. 五口井均能构建非空残差；
2. 生成至少 32 个固定配对样本；
3. 完成一个 batch 的 forward、backward 和优化；
4. 保存并重新加载 checkpoint，数值输出一致；
5. 深度域与时间 fixture 分别完成一组 paired-sample forward；
6. 对 inline/xline 各一个短 section 完成确定性推理；
7. residual 在目的层外严格为零，所有有效输出有限；
8. 相同 checkpoint、输入和 chunk 顺序变化产生相同结果。

smoke 中的接口、坐标、非有限值、checkpoint 恢复和 section stitching 错误由执行代理直接
修复并重跑。smoke 通过后立即启动正式训练，atlas 与训练结果一起交给用户查看。

### 5.6 正式训练

正式训练最多四个 epoch。每个 epoch 必须：

- 保存独立 checkpoint；
- 记录训练 loss、固定配对 monitor 的 residual RMSE 和 residual RMS ratio；
- 记录五口井的高频 residual RMSE、相关性和振幅比；
- 记录增强前后完整井曲线 RMSE；
- 记录增强前后井位正演相关性，仅作诊断；
- 输出同一组固定 section 图件。

每个 epoch 结束后先写 `last.pt` 和独立的 `epoch_XXX.pt`，再运行评估。评估或绘图中断时，
正式入口从最近一个完整 checkpoint 恢复，不重做已经完成的 epoch。

零 residual 是正式 baseline。一个 epoch 成为候选只需满足：

1. 输出与 checkpoint 恢复均有限；
2. 固定配对 monitor 的 residual RMSE 优于零 residual baseline；
3. 预测 residual RMS / target residual RMS 位于 YAML 配置的宽容区间，首轮默认
   `[0.35, 1.75]`；
4. 五口井 pooled 高频 residual RMSE 优于 GINN body-only，且至少四口井改善。

候选中选择最早 epoch。增强后的正演相关性、完整曲线 RMSE、横向粗糙度和低频泄漏均发布
为诊断，不以单一数值阻塞夜间训练；最终由用户根据图件决定是否交付。

### 5.7 夜间受限循环

执行代理先完整跑完首轮四个 epoch，再根据结果最多追加一轮训练。不得做多参数搜索：

- 若固定配对 monitor 和井位输出都明显坍缩到零 residual，只将 residual RMS underfit
  权重乘 `2`，从首轮最佳 checkpoint 续训最多两个 epoch；
- 若固定配对 monitor 已学习 residual，但井位高频指标没有改善，只将直接井 residual
  anchor 权重乘 `2`，从首轮最佳 checkpoint 续训最多两个 epoch；
- 若 residual 振幅过强或更晚 epoch 出现过密纹理，不重训，直接选择较早 checkpoint；
- 若出现横向贴片缝、逐道随机跳变、support 错位或 chunk 顺序改变结果，按实现错误修复，
  重跑 smoke 后只重跑受影响的正式轮次。

一次受限调整后仍没有候选，正常发布 `completed_not_accepted` 和最接近的诊断 checkpoint，
不继续自动调权重、网络深度、井名单、`15 m` 尺度或残差库。

执行代理监控正式命令直至最终状态文件出现。训练日志每个固定 batch 和每个 epoch 输出
进度；长时间无新日志时先检查进程与最近 checkpoint，再判断是否为实现错误。不得因为门禁
未通过而把正常完成的训练当作异常退出。

### 5.8 夜间最终产物

无论结果是否 accepted，正式命令都必须退出并发布：

```text
enhance_status.json
selected_checkpoint.json（存在候选时）
best_diagnostic_checkpoint.pt（无候选时）
last.pt
metrics.csv
resolved_config.yaml
residual_library.npz
review_package/
```

review package 至少包含：

- 五口井 residual atlas；
- 五口井 filtered full / GINN body / enhanced 曲线叠合图；
- 五口井 target/predicted residual 对比；
- 固定 inline 和 xline section 的 GINN body、predicted residual、enhanced log-AI；
- 增强前后正演差异图；
- 同一 section 的横向放大图，便于检查重复纹理、逐道跳变和贴片缝。

用户起床后只需检查：

- residual 是否像对 body 的连续超分辨增强，而不是噪声或周期纹理；
- 横向是否存在重复贴片边界和逐道跳变；
- 振幅是否与五口井 residual atlas 同量级；
- 增强后正演结果是否保持 GINN 已有的主要波瓣关系。

用户确认 Enhance checkpoint 后，阶段 2 完成。

## 6. 阶段 3：联合真实工区交付

### 执行代理连续完成

- 冻结 GINN V2 与 Enhance V2 checkpoint、归一化参数和两个尺度合同。
- 实现 inline/xline 一致的确定性分块推理和无缝拼接。
- 先用固定小区域完成端到端 smoke，并验证 chunk 顺序不改变输出。
- 准备唯一的正式全体积推理命令。

### 长时间命令分割点

用户运行全体积推理命令。命令发布：

- body-scale log-AI；
- predicted high-frequency residual；
- enhanced high-resolution log-AI；
- GINN 正演地震和闭环指标；
- 支持、方向分歧和 provenance；
- 固定剖面及井位对比图。

全体积产物和最终图件通过人工检查后，第一版工作流完成。

## 7. 每次交接的最小信息

每个阶段结束时只需记录：

- 本阶段实际修改的文件；
- smoke 结果；
- 需要用户运行的一条长命令；
- 长命令预计时间与主要输出；
- 人工需要查看的图件；
- 已冻结的 artifact、checkpoint 和配置路径；
- 下一阶段唯一入口。

详细失败实验与历史选择继续由 Git 和 `note/summary/final_audit` 保存，不扩展当前生产 interface。
