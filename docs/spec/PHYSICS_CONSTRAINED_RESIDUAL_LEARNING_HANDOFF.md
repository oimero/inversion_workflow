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

训练固定为一次共享预训练和最多两个微调 epoch：

```text
pretrain epoch 1       masked pretraining，发布共同 checkpoint
finetune epoch 1–2     mixed masked + visible-center + trusted-well training
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
- masked/visible batch 使用 seismic shape 与 LFM anchor；trusted-well batch 使用按井 body 标准差归一化的 well-body loss 与 LFM anchor，不同时拟合该井的低相关地震波形。
- LFM anchor 直接复用所选第七步 variant 记录的 Butterworth 阶数、截止频率和边界处理，不用同名 Gaussian 尺度替代。
- 冻结空间训练/验证块、validation center-trace identities，以及不进入 well loss 的井内连续验证区间。
- 实现一个命令内的共享预训练、分支微调、逐 epoch 评估、固定井剖面和 blind section 推理。
- 使用极小数据完成 masked、visible-center、trusted-well 三种 batch 的前向/反向和端到端 smoke。

### 4.3 冻结 baseline 与验收指标

正式训练前只计算一次 `LFM-only` baseline，随后所有 trial 复用同一份 baseline、split 和 validation identity。每个 checkpoint 必须报告：

- masked validation center traces 的相关性、归一化 shape loss 和逐道改善率；
- visible-center validation traces 的相关性和归一化 shape loss；
- 可信井留出区间的 body log-AI RMSE、绝对 bias 和逐井改善；
- `150 m` 低通后的 LFM drift；
- 小于 `body_smoothing_fwhm_m` 的短波能量和相对可信井 body target 的垂向粗糙度；
- 非负解析增益与 raw-amplitude residual，仅作诊断；
- 非有限值、support 连续性和 inline/xline 方向分歧。

一个 checkpoint 只有同时满足以下门禁才可接受：

1. 共同 masked-pretrain checkpoint 相对 LFM-only 的 masked 相关性中位改善至少 `0.01`，归一化 shape loss 不高于 baseline 的 `99%`；未达到时停止，不进入半监督；
2. 半监督 checkpoint 的 masked/visible 相关性相对共同 pretrain 的中位下降均不超过 `0.01`，归一化 shape loss均不超过共同 pretrain 的 `1.05` 倍；
3. 半监督 pooled body RMSE 优于共同 masked-pretrain checkpoint，至少三口可信井改善；相对 LFM-only 的 pooled RMSE 不超过 `1.5` 倍。相对 LFM 的改善继续报告，不作为硬性要求；
4. `150 m` 低通后的 LFM drift RMSE 不高于 `0.03 log-AI`；
5. 小于 `15 m` 的短波能量占比不高于 `1%`，井位垂向粗糙度不高于对应 `15 m` body target 的 `1.25` 倍；
6. 输出、梯度和指标全部有限，validation support 无缺口；inline/xline 分歧作为诊断，固定剖面无明显方向条带、逐道跳变或锯齿。

原始振幅 MSE、解析增益和 raw-amplitude residual 不参与验收。自监督 epoch 不单独要求绝对 body 对比度；绝对尺度只在混合半监督 epoch 验收。

### 4.4 逐 epoch 停止与 checkpoint 选择

执行代理在每个 epoch 结束后立即评估：

- 当前 checkpoint 首次通过全部门禁时，停止训练并选择它；
- 波形相关性继续上升，但 well RMSE、LFM drift 或粗糙度相对上一 epoch 恶化超过 `5%` 时，停止当前 trial 并保留上一 checkpoint；
- 所有未通过门禁的主指标相对上一 epoch 改善均小于 `1%` 时，判定平台并停止当前 trial；
- 出现非有限值、support 错位、中心道泄漏或 checkpoint 无法恢复时，按实现错误处理，修复并重新 smoke，不计入调参 trial。

若多个 checkpoint 通过，固定选择 epoch 最早者，不以最高地震相关性覆盖较早模型。

### 4.5 受限 trial loop

masked pretraining 只运行一次。每个半监督 trial 从同一共同 checkpoint 开始，最多运行两个 epoch。若没有 checkpoint 通过，根据首个失败门禁只改变一个量：

| 首个失败门禁 | 下一 trial 唯一允许动作 |
|---|---|
| masked shape | 微调学习率减半 |
| visible-center shape | 将 seismic-shape loss 权重乘 `2` |
| trusted-well body | 将 well body loss 权重乘 `2` |
| LFM drift | 将 LFM anchor 权重乘 `2` |
| roughness/短波能量 | 学习率减半，并优先检查更早 checkpoint |
| 方向分歧、方向条带或 support | 作为实现错误修复，不允许用 loss 权重掩盖 |

每个调整必须记录父 trial、唯一变化、目标门禁和结果；目标门禁没有改善时立即撤销。自动循环最多运行三个正式 trial。执行代理不得在循环中修改网络架构、井名单、split、子波、`15 m` body 尺度、第七步 LFM 滤波合同、增益模型或训练数据。

三个 trial 内通过时，发布 `selected_checkpoint.json`、trial 对比表、固定井剖面和 blind sections，并冻结 GINN V2 body checkpoint。三个 trial 均未通过时停止循环。

### 4.6 阶段结束

自动门禁通过后，用户只需查看最终 review package：

- 可信井附近的 body-scale AI 相对 LFM 是否具有合理增量；
- 井监督区与非井区是否保持一致预测语义；
- 剖面是否无明显锯齿、方向条带或逐道跳变；
- 最早可接受 checkpoint 是否优于更晚的高相关 checkpoint。

真实工区全体积推理留到阶段 3。

## 5. 阶段 2：Enhance V2 残差学习

### 执行代理连续完成

- 从可信井构建 `full - body` 高频残差库和尺度、振幅、层段 metadata。
- 生成残差贴片 atlas，供用户在正式训练前快速检查。
- 复刻并简化完整剖面残差学习；训练输出连续 residual field，而不是逐位置检索结果。
- 实现 body、residual、enhanced log-AI 和增强前后正演对比图。
- 执行代理完成小残差库和短剖面 smoke。

### 人工检查与长时间命令分割点

用户先确认残差 atlas 的振幅和纹理具有地质意义。确认后，执行代理给出唯一的 Enhance V2 正式训练命令，用户运行该命令。

### 交给用户的最终门禁

用户检查：

- residual 是否表现为对 body 的连续超分辨增强；
- 横向上是否存在贴片边界、重复纹理或随机跳变；
- 增强振幅是否落在可信井残差范围内；
- 增强后正演闭环是否仍在可接受范围。

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
