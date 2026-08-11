# 物理约束与残差学习实施 HANDOFF

## 1. 用途

本文将 `PHYSICS_CONSTRAINED_RESIDUAL_LEARNING.md` 转换成可连续实施的阶段。阶段只在以下位置结束：

- 需要人工查看井、剖面或体数据图件并作科学判断；
- 需要用户运行长时间训练或全体积推理命令。

每个阶段由 Codex 完成代码、短 smoke、错误修复和命令准备。用户只承担长时间运行和无法由数值指标代替的图件判断。对应实现尚未存在时，HANDOFF 只规定命令职责；实际脚本和参数必须在 smoke 通过后给出。

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

### Codex 连续完成

- 建立共同 SampleAxis、common batch、time/depth forward adapter 和 vertical-scale adapter。
- 实现 `body_smoothing_fwhm_m` 的可微高斯算子及 NumPy/Torch 对照。
- 扩展第六步 artifact，使同一口井同时发布 model-axis control 和 native well control。
- 固定第七步主 LFM、敏感性 LFM、实际低通尺度和有效支持的读取合同。
- 建立最小 GINN body 输出到冻结正演的闭环，但不开始正式训练。
- Codex 运行深度 smoke 和时间 fixture，修复所有可复现错误。

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

## 4. 阶段 1：GINN V2 自监督预训练

### Codex 连续完成

- 实现横向 patch reader、完整中心道缺失输入和显式 missing-trace mask。
- 网络只预测中心道 body log-AI；冻结正演只重建该中心道。
- 实现归一化子波、中心道支持内的振幅不变波形形状 loss，以及不反向传播的非负解析增益诊断；不建立可训练的逐样点或空间增益场。
- 可见输入道分别由自身支持归一化；中心道目标只在损失侧归一化。自监督 checkpoint 的 body 对比度尺度明确保持未校准，只作为阶段 2 初始化。
- 实现固定 training/validation center-trace identities。
- 每个 epoch 保存 checkpoint，并记录归一化波形形状闭环、观测振幅诊断、短波能量、粗糙度和 LFM 漂移。
- Codex 使用极小数据完成前向、反向、checkpoint 恢复和推理 smoke。

### 长时间命令分割点

smoke 通过后，Codex 给出唯一的正式自监督训练命令。用户运行该命令，训练输出每个 epoch 的 checkpoint、指标和固定 validation 图件。

### 交给用户的人工门禁

Codex 汇总各 epoch 后，用户重点查看：

- 波形形状拟合达到可接受水平后，body 是否开始出现额外短波振荡；
- 固定 validation center traces 是否存在明显横向断裂；
- raw-amplitude residual 是否只表现为诊断 gap，而没有驱动 body 对比度随局部增益失真；
- 最早可接受 checkpoint 的图件是否优于更晚 checkpoint。

阶段 1 不对绝对 body 对比度作定量验收。

用户确认一个预训练 checkpoint 后，阶段 1 完成。

## 5. 阶段 2：GINN V2 半监督与剖面门禁

### Codex 连续完成

- 从第六步 native well control 的原始观测有效样点生成 `15 m` body-scale 井监督目标。
- 冻结井角色名单：全部成功井继续用于第七步 LFM，可信井进入 body loss，低质量井只作诊断。
- 在阶段 1 checkpoint 上加入可信井绝对 log-AI 监督和第七步尺度 LFM anchor；地震项继续使用阶段 1 的振幅不变波形形状 loss。
- 实现半监督 checkpoint 选择、固定井剖面和 blind section 推理。
- Codex 完成单井、单 batch、单剖面的训练与推理 smoke。

### 长时间命令分割点

smoke 通过后，Codex 给出唯一的正式半监督微调命令。用户运行后，Codex分析全部 epoch，并用候选 checkpoint 生成固定真实剖面。

### 交给用户的人工门禁

用户确认：

- 可信井附近的 body-scale AI 相对 LFM 有合理增量；
- 可信井提供的绝对对比度没有被 raw seismic amplitude mismatch 拉偏；
- 井监督区与非井区没有突然变化的预测语义；
- 剖面无明显锯齿、方向条带或逐道跳变；
- 选择的 checkpoint 是满足闭环和井约束的较早模型。

阶段 2 结束时冻结 GINN V2 body checkpoint。真实工区全体积推理留到阶段 4。

## 6. 阶段 3：Enhance V2 残差学习

### Codex 连续完成

- 从可信井构建 `full - body` 高频残差库和尺度、振幅、层段 metadata。
- 生成残差贴片 atlas，供用户在正式训练前快速检查。
- 复刻并简化完整剖面残差学习；训练输出连续 residual field，而不是逐位置检索结果。
- 实现 body、residual、enhanced log-AI 和增强前后正演对比图。
- Codex 完成小残差库和短剖面 smoke。

### 人工检查与长时间命令分割点

用户先确认残差 atlas 的振幅和纹理具有地质意义。确认后，Codex 给出唯一的 Enhance V2 正式训练命令，用户运行该命令。

### 交给用户的最终门禁

用户检查：

- residual 是否表现为对 body 的连续超分辨增强；
- 横向上是否存在贴片边界、重复纹理或随机跳变；
- 增强振幅是否落在可信井残差范围内；
- 增强后正演闭环是否仍在可接受范围。

用户确认 Enhance checkpoint 后，阶段 3 完成。

## 7. 阶段 4：联合真实工区交付

### Codex 连续完成

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

## 8. 每次交接的最小信息

每个阶段结束时只需记录：

- 本阶段实际修改的文件；
- smoke 结果；
- 需要用户运行的一条长命令；
- 长命令预计时间与主要输出；
- 人工需要查看的图件；
- 已冻结的 artifact、checkpoint 和配置路径；
- 下一阶段唯一入口。

详细失败实验与历史选择继续由 Git 和 `note/summary/final_audit` 保存，不扩展当前生产 interface。
