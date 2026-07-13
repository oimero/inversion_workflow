# GINN v2 Canonical Increment 与 Synthoseis-lite 微纹理实施 HANDOFF

## 1. 文档目的

本文是两份设计规格之间的实施交接单：

- [GINN v2 Canonical Increment 语义重构规格](GINN_V2_CANONICAL_INCREMENT_SEMANTICS.md)；
- [Synthoseis-lite 微纹理生成方法论规格](SYNTHOSEIS_MICROTEXTURE_GENERATION_DESIGN.md)。

它记录已经锁定的公共语义、实现顺序、每个阶段的完成条件和当前状态，供后续开启 goal 模式后逐项推进。本文本身只描述实施边界和验收合同；本轮没有修改生产代码、配置、测试或历史产物。

## 2. 当前结论

### 2.1 唯一的阻抗分解语义

在最终规则采样轴上，对物理单位的 `log(AI)` 定义：

```text
canonical_background_log_ai = P(target_log_ai)
target_increment_log_ai     = target_log_ai - canonical_background_log_ai
predicted_log_ai            = external_lfm_log_ai + predicted_increment_log_ai
```

`P` 是固定的零相位 Butterworth 低通算子。它是低通算子而不是数学投影，因此 `P(P(m)) == P(m)` 不是验收条件；增量也不是“地震唯一恢复的全部高频”。

公共字段采用以下名称：

| 角色 | 字段 |
| --- | --- |
| 合成完整真值 | `model_target_log_ai` |
| 规范背景 | `canonical_background_log_ai` |
| 合成监督目标 | `target_increment_log_ai` |
| 外部或合成输入 LFM | `input_lfm_log_ai` |
| 网络输出 | `predicted_increment_log_ai` |
| 最终重组结果 | `predicted_log_ai` |

地质对象扰动、微纹理、观测与正演之间的波形误差、阻抗预测误差分别保留各自对象名称；它们不共享一个无修饰的 `residual` 或 `delta` 语义。

### 2.2 共享数值合同

| 采样域 | 设计截止 | 单程设计阶数 | 前后向等效阶数 | reflect buffer |
| --- | ---: | ---: | ---: | ---: |
| 时间域 | 15 Hz | 6 | 12 | 0.4 s |
| 深度域 | 最小波长 400 m | 6 | 12 | 400 m |

实现使用 SciPy Butterworth SOS 与 `sosfiltfilt`。采样轴和滤波输入使用 float64；每个连续有限段独立处理，NaN/invalid 间隙不跨越。标签在完整道上生成，训练 patch 只做直接切片。

深度域的纵向长度、薄层厚度和纹理相关长度使用米；横向坐标继续使用实际 inline/xline 坐标，xline 步长 4 不改变纵向单位。

### 2.3 微纹理首版模式

首版固定三组独立 benchmark，三者共享一个宏观父样本：

| 实验 | 宏观 profile | 微纹理 | 用途 |
| --- | --- | --- | --- |
| A | `three_parameter` | `none` | 宏观基线与 false-texture 对照 |
| B | `three_parameter` | `thin_bed_cluster` | 参数化交替薄层先验 |
| C | `three_parameter` | `canonical_well_texture` | 训练井 canonical increment 去宏观纹理先验 |

HSMM 状态序列、对象厚度、几何事件、横向坐标、LFM 退化和 seismic mismatch 使用 common-random 规则。任一模式失败，整个 paired group 拒绝；只有三种模式都成功时才进入 A/B/C benchmark。

微纹理只在对象内部的高分辨率真值轴生成，随后与宏观 profile 合成，统一经过既有抗混叠和降采样。最终 `target_increment_log_ai` 始终从完整 composite 重新计算；微纹理组件不成为第二套监督目标。

## 3. 实施边界

### 3.1 本轮已确定

- canonical increment 是唯一网络输出语义；
- canonical operator、真实井处理顺序和外部 LFM 元数据合同已在主规格中锁定；
- Synthoseis-lite 继续使用 `synthoseis_lite_v4` 作为微纹理扩展后的 benchmark 版本；
- A/B/C 分开训练，另建共享跨模式测试集用于交叉评估；
- 微纹理模式不是网络输入通道，也不改变 GINN v2 的三通道输入；
- physics 只承接已完成的监督祖先；标准部署 checkpoint 仍由 dense synthetic supervised 指标选择；
- R0/R1 同时报告 increment、canonical closure、deployment closure、LFM-only 与最终 AI 指标；
- 20260706 full-correction benchmark、旧 checkpoint 和历史报告作为冻结基线记录。

### 3.2 本轮尚未实施

- 共享 canonical operator 的生产代码；
- v4 benchmark writer/reader 与完整道标签写入；
- 微纹理 bank、三种 emitter 和 paired-group 生成器；
- GINN v2 v2/v5 schema、increment batch/loss/checkpoint；
- 真实井 canonical increment reader；
- canonical/deployment closure、R0/R1 新字段和反事实报告；
- 旧生产实现的清理提交。

这些事项是后续 goal 的工作内容，不在本 HANDOFF 中假定已经完成。

## 4. 推荐实施顺序与完成条件

每一步都应保持主分支可测试；下一步只能消费上一步已冻结的字段和数值合同。

### 阶段 1：共享 canonical 基础算子

- [ ] 增加领域无关的 canonical lowpass/decomposition 模块；
- [ ] 实现时间域和深度域合同解析；
- [ ] 支持连续有限段、NaN 间隙、invalid mask 和短段明确失败；
- [ ] 按权威 `sample_interval` 检查 float64 规则轴；
- [ ] 输出 `canonical_background_log_ai` 与 `target_increment_log_ai`。

完成条件：时间域、深度域、NaN 间隙、短段和 cutoff 响应 fixture 全部通过；`target == background + increment` 在有限点成立。

### 阶段 2：Synthoseis-lite v4 生产端

- [ ] 在完整合成道上写入规范背景和规范增量；
- [ ] 写入多个 `input_lfm_log_ai` variant，并保证同一父 realization 共享 target increment；
- [ ] 写入 canonical contract 和 LFM producer metadata；
- [ ] 为微纹理组件和对象目录预留 v4 字段；
- [ ] 保持历史 benchmark 目录不变。

完成条件：v4 fixture 可独立复算 composite、canonical background、target increment 和 LFM variant；patch 标签等于完整道标签的直接切片。

### 阶段 3：微纹理 bank 与领域无关 emitter

- [ ] 固定 `MicrotextureEmission` 接口、物理单位、seed 和 metadata；
- [ ] 在最终规则时间轴/TVDSS 轴上建立训练井 canonical texture bank；
- [ ] 检查训练井、留出井和 bank coverage 隔离；
- [ ] 实现 A 的 `none`、B 的 `thin_bed_cluster`、C 的 `canonical_well_texture`；
- [ ] 对对象边界、厚度、振幅、端点跳变、clipping 和 reversal 做 QC。

完成条件：固定 seed 完全确定；对象外扰动为零；所有厚度和长度以秒/米表达；A/B/C 的 emitter 输出均可独立复算。

### 阶段 4：paired A/B/C benchmark

- [ ] 一次生成 `macro_parent`，再实例化三个 mode；
- [ ] 实现组级全成功/全拒绝和原因记录；
- [ ] 保持三组 benchmark 的 parent 集合、split、LFM、mismatch 和预算一致；
- [ ] 生成 `shared_none_test`、`shared_thin_bed_cluster_test`、`shared_canonical_well_texture_test`。

完成条件：三组 benchmark 的 `macro_parent_id` 集合一致；任一 mode 失败不会留下不完整 paired group；跨模式测试矩阵可运行。

### 阶段 5：最小 GINN v2 synthetic supervised 垂直切片

- [ ] v4 reader 只暴露 canonical 字段；
- [ ] 先选择最简单 trace 架构打通 `v4 -> batch -> increment MSE`；
- [ ] 输出 `predicted_increment_log_ai`，并生成 `predicted_log_ai`；
- [ ] 写入新 experiment/checkpoint/prediction schema；
- [ ] 验证零初始化输出增量为零且最终结果等于输入 LFM。

完成条件：一个最小 epoch smoke 可以从 v4 fixture 完成监督训练、恢复 checkpoint 和增量预测；旧 schema 输入明确失败。

### 阶段 6：四类架构与 synthetic closure

- [ ] 将同一 canonical 接口扩展到四种纯架构；
- [ ] 实现 canonical closure 与 deployment closure；
- [ ] 分别报告 increment fidelity、canonical closure AI 和 deployment closure AI；
- [ ] 接入低频 QC 与 physics 前后增量漂移诊断。

完成条件：四种架构输出形状、零初始化和有限梯度一致；两类 closure 可以由保存数组复算。

### 阶段 7：synthetic physics 与部署资格

- [ ] physics 使用 canonical background 或明确的 deployment LFM 组合；
- [ ] `increment_l2_weight` 作为唯一增量正则字段；
- [ ] physics 阶段必须从已完成的 synthetic/real-well supervised checkpoint 初始化；
- [ ] waveform-best physics checkpoint 仅作为诊断；
- [ ] real-well-plus-physics 默认保留 experimental 标记。

完成条件：监督后 physics 前向、反向和有限梯度通过；非法 physics-first、zero-initialized physics 和不具备部署资格的 checkpoint 明确失败。

### 阶段 8：真实 LFM 与真实井监督

- [ ] 真实井按 MD -> 轨迹/时深映射 -> 最终规则轴重采样 -> canonical P -> well increment 顺序处理；
- [ ] 真实 LFM 只校验 producer metadata，并透传最终体 complement-response QC；
- [ ] 井上监督目标命名为 `well_target_increment_log_ai`；
- [ ] 井间 physics 漂移、LFM-only 指标和最终 AI 指标分开报告。

完成条件：时间域和深度域 golden fixture 在最终规则轴上复算一致；井监督与 synthetic increment 语义一致。

### 阶段 9：R0、R1 与反事实报告

- [ ] R0 生成 `predicted_increment_log_ai` 和 `predicted_log_ai`；
- [ ] valid 点全部得到有限输出，invalid 点保持 NaN；
- [ ] 支持 inline 步长 1、xline 步长 4、多剖面和完整体；
- [ ] 输出 support count、最大上下文有效率和低支持区域；
- [ ] 运行 paired LFM 反事实并区分增量条件效应和 LFM 直接替换效应；
- [ ] R1 使用明确的 deployment closure 字段。

完成条件：R0/R1 只接受新 checkpoint/LFM schema；覆盖、坐标和 closure 数组可复算。

### 阶段 10：A/B/C GINN 消融与默认入口切换

- [ ] A/B/C 使用相同架构、split、seed、normalization 和训练预算分别训练；
- [ ] 交叉测试 false texture、薄层漏检、井纹理覆盖和跨模式迁移；
- [ ] 对照 full-correction 基线，单独报告合同正确性和模型效果；
- [ ] 只有完整回归通过后才切换默认入口；
- [ ] 按清单退役被替代的旧生产代码，历史产物由冻结目录和 Git 历史保证可复现。

完成条件：第 10.1–10.5 验收矩阵和完整 smoke 通过；默认入口与文档入口使用 canonical increment 语义。

## 5. 关键验收矩阵

### 数值与语义

- [ ] `target_log_ai == canonical_background_log_ai + target_increment_log_ai`（有限点）；
- [ ] NaN/invalid 间隙两侧独立；
- [ ] 采样轴和 cutoff 合同在时间域、深度域一致；
- [ ] patch 不重新计算低通或增量；
- [ ] `predicted_log_ai == input/external LFM + predicted_increment_log_ai`；
- [ ] canonical increment 的频带统计使用保守低频响应指标，不解释为正交能量占比。

### 微纹理

- [ ] A/B/C 共享 `macro_parent_id` 和宏观随机流；
- [ ] `none` 发射器严格为零；
- [ ] 薄层数量、厚度、对比度和交替方向符合物理单位配置；
- [ ] 井纹理 bank 只来自训练井，状态/zone/长度 coverage 可查询；
- [ ] 微纹理只在对象边界内生成，失败原因显式记录；
- [ ] composite、组件、降采样和 forward visibility 可复算。

### GINN v2 与部署

- [ ] 四类架构均使用三通道输入和单一 increment 输出；
- [ ] 零初始化严格输出 LFM；
- [ ] synthetic/real-well supervised 的梯度有限；
- [ ] supervised 后 physics 合法，physics-first 非法；
- [ ] physics checkpoint 的部署资格按 dense synthetic supervised 指标控制；
- [ ] R0 有效点全覆盖且无有效点 NaN；
- [ ] R1、LFM-only、canonical closure 和 deployment closure 指标分开。

### 迁移与清理

- [ ] v4 benchmark、experiment v2、checkpoint v5、prediction v3 和 R0/R1 新 schema 相互匹配；
- [ ] 旧 schema 只触发明确失败，不进入新 reader/loader；
- [ ] 旧 full-correction 产物只作为冻结基线；
- [ ] 正式生产路径中的旧字段、alias、fallback 和双语义分派按主规格清理；
- [ ] 清理提交附带删除清单和回归 smoke 结果。

## 6. 后续 goal 的首个任务

建议下一个 goal 从阶段 1 开始，只实现共享 canonical 基础算子和对应 fixture。不要同时接入微纹理、physics、真实井或 R0。阶段 1 通过后，再按本 HANDOFF 的阶段 2 到阶段 10 推进，每完成一项就在对应复选框勾选并记录测试命令、fixture 和产物路径。

## 7. 相关文档

- [Canonical Increment 语义规格](GINN_V2_CANONICAL_INCREMENT_SEMANTICS.md)
- [Synthoseis-lite 微纹理生成规格](SYNTHOSEIS_MICROTEXTURE_GENERATION_DESIGN.md)
- [GINN v2 可组合训练设计](GINN_V2_COMPOSABLE_TRAINING_DESIGN.md)
