# GINN v2 Canonical Increment 与 Synthoseis-lite 微纹理实施 HANDOFF

## 1. 文档目的

本文是 canonical increment、Synthoseis-lite 阶段 2.5 和微纹理设计之间的实施交接单：

- [GINN v2 Canonical Increment 语义重构规格](GINN_V2_CANONICAL_INCREMENT_SEMANTICS.md)；
- [Synthoseis-lite 微纹理生成方法论规格](SYNTHOSEIS_MICROTEXTURE_GENERATION_DESIGN.md)。
- [Synthoseis-lite 阶段 2.5 边界重构规格](SYNTHOSEIS_LITE_STAGE_2_5_REFACTOR_AND_REVIEW_FIXES.md)。

它记录已经锁定的公共语义、实现顺序、每个阶段的完成条件和当前状态，供后续开启 goal 模式后逐项推进。阶段 1/2 的代码已落地，阶段 2.5 负责完成公共合同、probe 关闭和包边界收口；阶段 3 先稳定现有 v4 主链，微纹理及其 A/B/C benchmark 延后到主链验收之后；历史产物目录保持不变。

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
| C | `three_parameter` | `canonical_well_patch` | 训练井 canonical increment 去宏观纹理先验 |

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

- 微纹理 bank、三种 emitter 和 paired-group 生成器；
- GINN v2 v2/v5 schema、increment batch/loss/checkpoint；
- 真实井 canonical increment reader；
- canonical/deployment closure、R0/R1 新字段和反事实报告；
- 旧生产实现的清理提交。

这些事项是后续 goal 的工作内容，不在本 HANDOFF 中假定已经完成。阶段 1 和阶段 2 已落地在 `cup.impedance`、Synthoseis-lite v4 writer/reader、LFM variant writer 和生成 manifest 中。微纹理规格中的阶段 3/4 不作为阶段 2.5 的直接后继；先完成 v4 主链和 GINN v2 核心链路，再完成延期的 physics 阶段，随后才进入本 HANDOFF 的阶段 10–12。

### 3.3 阶段 2.5 对称性收口与单一 mask（本轮）

本轮把两个域的消费合同统一到 observed high-resolution forward：

- 配置必须声明 `seismic_input.policy=observed_highres_forward`；时间域还必须将
  `forward_qc.highres_forward.enabled/required` 同时设为 `true`；
- manifest 写入 `seismic_input_contract`，明确 model axis、`seismic_observed` 网络
  输入和仅供 physics/closure 的 `seismic_model_consistent`；
- time 将高分辨率 truth forward/downsample 结果写入
  `seismic/seismic_observed`；depth 沿用 AI–Vp、深度正演和抗混叠结果写入同名字段；
- 两域 `sample_index.csv` 都显式记录
  `seismic_input_dataset`、`seismic_model_consistent_dataset` 和
  `valid_mask_dataset`；变体还记录 family、operator source 和参数 JSON；
- 两个 reader 通过同一最小 sample protocol 暴露 `sample_axis`、canonical 字段、
  `seismic_input`、`seismic_model_consistent` 和单一 `valid_mask`。缺少新
  `mask_contract` 或仍包含旧 mask index 字段的 v4 直接失败；不提供字段回退；
- LFM component 顺序、variant catalog、随机流目的和 QC metadata 共用同一编排层，
  但时间域的秒制算子与深度域的米制/AI–Vp 算子保持各自实现。
- 时间域和深度域均按 `local_missing_control_bias -> over_smoothing` 执行 LFM
  退化；两者的组合顺序由本地回归测试锁定。
- 深度域受控 LFM 退化改为按 `valid_mask` 做幅度缩放和空间平滑，无效区继续写 NaN；
- 深度 forward 增加 `seismic_forward.backend=auto|numpy|torch_cuda` 与 float64 合同，
  auto 在 CUDA 可用时复用 Torch depth backend；manifest 记录 requested/resolved backend；
- 高分辨率 forward support 只作为生成 QC，reader、训练协议和 loss 只消费 `valid_mask`；
  physics halo/central-crop 仍留在后续 GINN 训练阶段。

阶段 2.5 本身不改微纹理、physics、R0/R1，也不迁移旧 v4 产物；当前阶段 8 收口只涉及
GINN v2 synthetic-supervised 读取、评估和 parser 边界，见 3.4。

阶段调整：原计划中的“阶段 3：微纹理 bank 与 emitter”和“阶段 4：paired A/B/C benchmark”分别延期为本 HANDOFF 的阶段 10 和阶段 11；A/B/C GINN 消融为阶段 12。physics 训练、真实 physics 适配、部署资格和 R1 正演闭环统一延期为阶段 9，作为进入微纹理前的最后一个大阶段。当前正在运行的完整 v4 工区生成属于阶段 3 的主链稳定化门禁；在 v4、GINN v2 核心链路、R0 以及阶段 9 的 physics 门禁完成前，不引入微纹理变量，以免把生成合同、物理适配和模型消融问题混在一起。

### 3.4 阶段 8 检阅收口（当前增量）

本轮只收紧当前可部署的 synthetic-supervised 主链，不提前实现阶段 9：

- 新配置解析器和实验 runner 对 `physics` loss block 直接失败，并明确提示阶段 9；
  手工构造的配置也在 runner 入口再次拒绝，避免绕过 parser；
- `PatchDataset` 和 canonical reader 直接要求物化的
  `canonical_background_log_ai`，缺字段即失败，不从 `target - increment` 静默重建；
- 真实井 QC 图使用已生成的 `well_target_increment_log_ai`，不再用 `well_log_ai - lfm_log_ai`
  代替监督增量；
- `lfm_ideal` 只作为评估基线按需从预测产物或 benchmark 加载，不进入训练 batch、网络输入或
  checkpoint 训练状态；
- 阶段 9 的真实工区变换 provenance、physics 数据路径和 R0/R1 历史别名仍按下方
  allowlist 保留，当前不把它们伪装成已经完成的功能。

这些收口不改变 `ginn_v2_prediction_v3` 的公共预测语义；没有 `lfm_ideal` 时，评估报告仍输出
空的基线表和 `n_ok=0` 摘要，表示该诊断未计算，而不是把它混入训练数据。

本轮验证：45 个 canonical/v4/GINN ignored tests 通过，`compileall -q src/cup src/ginn_v2
scripts/ginn_v2.py` 通过；使用既有 6-patch prediction smoke 重新执行 report，产出
`scripts/output/ginn_v2_report_20260714_184124_507557`，`lfm_ideal` 仍能从 benchmark
按需评估。

## 4. 推荐实施顺序与完成条件

每一步都应保持主分支可测试；下一步只能消费上一步已冻结的字段和数值合同。

### 阶段 1：共享 canonical 基础算子

- [x] 增加领域无关的 canonical lowpass/decomposition 模块；
- [x] 实现时间域和深度域合同解析；
- [x] 支持连续有限段、NaN 间隙、invalid mask 和短段明确失败；
- [x] 按权威 `sample_interval` 检查 float64 规则轴；
- [x] 输出 `canonical_background_log_ai` 与 `target_increment_log_ai`。

完成条件：时间域、深度域、NaN 间隙、短段和 cutoff 响应 fixture 全部通过；`target == background + increment` 在有限点成立。状态：代码已落地，并通过阶段 2.5 本地合同验收。

### 阶段 2：Synthoseis-lite v4 生产端

- [x] 在完整合成道上写入规范背景和规范增量；
- [x] 写入多个 `input_lfm_log_ai` variant，并保证同一父 realization 共享 target increment；
- [x] 写入 canonical contract 和 LFM producer metadata；
- [x] 为微纹理组件和对象目录预留 v4 字段；
- [x] 保持历史 benchmark 目录不变。

完成条件：v4 fixture 可独立复算 composite、canonical background、target increment 和 LFM variant；patch 标签等于完整道标签的直接切片。状态：writer/reader 已切换新合同并通过本地 fixture，完整工区需要按阶段 2.5 合同重新生成。

### 阶段 2.5：边界重构与检阅建议修复

- [x] `cup.canonical_increment` 收口到 `cup.impedance`，增加 contract version 与双轴容差；
- [x] LFM producer contract 和 increment/LFM compatibility validator 落地；
- [x] v4 删除 probe 生成、writer、reader 和正式 baseline report 分支，并对旧 probe row 明确失败；
- [x] synthetic 按 `core/time/depth/readers/reporting` 实际依赖拆分；
- [x] 深度生成记录抽到 `depth/model.py`，保留其余数值 seam 的渐进拆分边界；
- [x] 删除提前落地的 microtexture emitter；
- [x] canonical contract 对象入口、LFM profile/QC 状态和实现版本命名完成稳定化；
- [x] 更新本地 ignored fixture，跑完 canonical、writer/reader、LFM 合同和 probe 失败测试；
- [x] 两域 observed input、model-consistent 参照、`valid_mask` 和 mismatch metadata
  使用统一索引字段；
- [x] 两域统一到 `mask_contract=single_valid_mask_v1` 与单一 `valid_mask`；
- [x] 深度域增加 auto CUDA/显式 NumPy CPU 的 float64 forward backend，并移除重复
  model-grid closure forward；
- [x] 使用当前可用深度 source 完成 writer → manifest → reader → baseline evaluator 小烟测；
- [x] 用新合同重新生成需要消费的 v4 benchmark，不覆盖冻结目录。

完成条件：新包 import/compile smoke 通过，LFM profile 和 QC 状态语义通过，旧实现版本命名不再出现在正式 reader/config/CLI 路径，probe 正式路径仍明确失败（保留失败测试与 observability 旁路），reader 拒绝缺少新合同的旧 v4。状态：单一 mask 合同、深度 GPU backend、本地 fixture 和完整工区重生成均已完成；严格无告警 baseline 仍由阶段 3 最后一项决定。

说明：历史冻结 report-card 和 `cup.seismic.observability` 仍作为参考资料保留；当前
`scripts/ginn_v2.py summarize`、v4 writer/reader、baseline evaluator 和 GINN v2 report
只消费 canonical increment/closure 字段，不生成或读取 probe coverage gate。

### 阶段 3：Synthoseis-lite v4 主链稳定化与基线冻结

- [x] 完成当前 v4 深度域全量工区生成，确认 preflight、writer、manifest、reader 和 baseline evaluator 能消费同一新目录；
- [x] 对生成结果执行 sample index、canonical decomposition、observed/model-consistent seismic、mask、LFM variant 和 split 的完整检查；
- [x] 保持 `microtexture_mode=none` 的现有主链，不在本阶段加入纹理 emitter 或 paired benchmark；
- [x] 记录场景接受率、拒绝原因和 `development_limited` 等状态，不把 attempt-level 拒绝误报为生成成功；
- [ ] 形成一份可冻结的 v4 baseline manifest 和评估报告，不覆盖 20260706 等历史目录。

时间域真实 source 可用前只做 fixture reader/baseline 验证，不声称完成时间域工区 smoke。
深度域完整工区已在当前实现上用新输出目录重新生成；严格无告警 baseline 仍由最后一项单独决定。

完整工区新运行 `experiments/synthoseis_lite/results/20260714/generate_field_conditioned`
已完成：manifest 为 `synthoseis_lite_v4`、`mask_contract=single_valid_mask_v1`，
`auto -> torch_cuda`/float64；sample index 共 6,942 行（534 个 base、6,408 个
seismic variant），全部为 `status=ok`，reader 和 baseline evaluator 均已读取通过。
生成状态为 `completed_with_warnings`：840 个 planned attempts 中 534 个接受，3 个
wedge 场景低于 0.5 acceptance threshold（`lx0.3/a0.25`、`lx0.3/a0.5`、
`lx1/a0.5` 的 left-to-right），因 enforcement=warn 保留了已接受样本。该目录可作
主链验收产物，但在把它标为严格全场 baseline 前，应先决定是否接受这 3 个场景的警告。

完成条件：新输出目录可被 v4 reader 和 baseline evaluator 独立读取；有限点的 canonical 重组成立；observed input、model-consistent 参照、mask、LFM variant 和坐标轴合同完整；旧 probe 和旧 v4 schema 明确失败；生成 acceptance/report 已记录。当前 3 个 wedge 场景低于阈值的告警仍未被标记为严格全场 baseline。

### 阶段 4：最小 GINN v2 synthetic supervised 垂直切片

- [x] v4 reader 只向训练 batch 暴露 canonical 字段；
- [x] 选择 `trace_conv1d` 打通 `v4 -> batch -> increment MSE`；
- [x] 输出 `predicted_increment_log_ai`，并生成 `predicted_log_ai`；
- [x] 写入 `ginn_v2_experiment_v2`、`ginn_v2_checkpoint_v5` 和 `ginn_v2_prediction_v3`；
- [x] 验证零初始化输出增量为零且最终结果等于输入 LFM。

完成条件已满足：一个最小 epoch smoke 从 20260714 v4 benchmark 完成监督训练、checkpoint 恢复、canonical increment prediction 和 report；旧 experiment schema 明确失败。

阶段 4 垂直切片证据：

```text
配置快照：experiments/ginn_v2/results/canonical_increment_contract_smoke_final_20260714/train.yaml
架构：trace_conv1d(hidden_channels=8, depth=2)
训练产物：experiments/ginn_v2/results/canonical_increment_contract_smoke_final_20260714
预测产物：scripts/output/ginn_v2_canonical_increment_contract_predict_final_20260714
报告产物：scripts/output/ginn_v2_canonical_increment_contract_report_final_20260714
训练：1 epoch / 2 steps，CUDA，synthetic_increment.mse=0.0090645645
checkpoint：ginn_v2_checkpoint_v5，成功恢复 Trace1DNet
prediction.npz：包含 predicted_increment_log_ai、predicted_log_ai、
  target_increment_log_ai、input_lfm_log_ai、valid_mask；
  max|predicted_log_ai-(input_lfm_log_ai+predicted_increment_log_ai)| = 0
零初始化：trace_conv1d 输出最大绝对值 = 0
合同复验：checkpoint 写入完整 increment_contract、training_sources、stage_lineage；
  prediction_summary 按实际 benchmark 记录 increment_contract、sample_axis_contract 和
  benchmark contract fingerprint；benchmark override 先做合同兼容性检查
smoke 标记：run_mode=smoke，validation_semantics=duplicated_training_patch，
  development_limited=true，deployment_eligible=false；该 checkpoint 只作为合同烟测产物，
  R0 入口会拒绝使用它
回归测试：43 passed；compileall src/cup src/ginn_v2 scripts/ginn_v2.py 通过
旧配置：ginn_v2_experiment_v1 明确拒绝，期望 ginn_v2_experiment_v2
```

本切片只覆盖 synthetic supervised 和最简单 trace 架构；physics、其部署资格和其余架构仍按后续阶段实施。
当前证据是合同烟测而非正式模型部署；正式训练需使用独立 validation parent，并由后续阶段重新评估部署资格。
活动配置随后切换为 `run_mode=standard` 的全量 `trace_conv1d` 训练，烟测证据以结果目录中的配置快照为准。

### 阶段 5：四类架构与 synthetic closure

- [x] 将同一 canonical 接口扩展到四种纯架构；`trace_conv1d`、`trace_dilated_tcn`、
  `trace_lateral_mixer` 和 `patch_conv2d` 均通过同一三通道/单增量输出接口；
- [x] 实现 canonical closure 与 deployment closure 的数组诊断；
- [x] 分别报告 increment fidelity、canonical closure AI 和 deployment closure AI；
- [x] 接入低频 QC；报告使用 canonical operator，局部段不足时写
  `not_computed`，不把 QC 当作训练或部署 gate；本阶段不启用 physics loss 或 physics 参数更新。

完成条件：四种架构输出形状、零初始化和有限梯度一致；两类 closure 可以由保存数组复算。
已验证：四架构 CPU forward/backward fixture；现有完整 checkpoint 的 6-patch
closure smoke 输出 `scripts/output/ginn_v2_stage5_closure_smoke_20260714_final2`。

### 阶段 6：真实 LFM 与真实井监督

- [x] 真实井按 MD -> 轨迹/时深映射 -> 最终规则轴重采样 -> canonical P -> well increment 顺序处理；
- [x] 真实 LFM 只校验 producer metadata，并透传最终体 complement-response QC；
- [x] 井上监督目标命名为 `well_target_increment_log_ai`；
- [x] 真实井监督支持四类架构，并保留空间簇均衡与 held-out 井切分。

完成条件：时间域和深度域 golden fixture 在最终规则轴上复算一致；井监督与 synthetic increment 语义一致。
已验证：深度域 200 点规则轴 golden fixture 生成并读取 `well_target_increment_log_ai`；
真实工区 source 目录当前不在工作区，因此尚未声称真实井训练 smoke。

### 阶段 7：R0 与增量报告

- [x] R0 生成 `predicted_increment_log_ai` 和 `predicted_log_ai`；
- [x] valid 点全部得到有限输出，invalid 点保持 NaN；
- [x] 支持 inline 步长 1、xline 步长 4、多剖面和完整体；
- [x] 输出 support count、最大上下文有效率和低支持区域；
- [x] 提供 paired LFM 反事实指标函数，区分增量条件效应和 LFM 直接替换效应。

完成条件：R0 只接受新 checkpoint/LFM schema；覆盖、坐标、增量和最终结果数组可复算。已验证：
32×128 深度/xline-step-4 fixture 输出覆盖率 1.0 且新字段有限。R1 正演闭环留待阶段 9。

### 阶段 8：核心主链对照、默认入口切换与旧代码清理

- [x] 对照冻结的 full-correction 基线，单独报告 canonical contract 正确性和模型效果；
- [x] 确认 v4 → GINN v2 → R0 主链在新 schema 下可重复运行；
- [x] 只有完整回归通过后才切换默认入口；活动训练入口固定为 `ginn_v2_experiment_v2`，
  活动摘要读取 manifest 与显式 closure 字段；
- [x] 按清单退役被替代的旧生产代码，历史产物由冻结目录和 Git 历史保证可复现。
  `real_delta.py` 已收口为 `real_well_supervised.py`，真实井字段、采样 QC 和评估指标
  使用 canonical increment 命名；旧 probe report-card 和 evaluation helper 已从正式路径删除。
  `LEGACY_KEYS` 仅保留为新 parser 的明确失败入口，R0 中供阶段 9 R1 使用的旧别名仍列为
  延期 allowlist，不作为新训练或新预测字段。
- [x] 当前主链不接受阶段 9 之前的 physics block；canonical background 缺失、真实井 QC
  增量来源和 `lfm_ideal` 训练耦合问题已收紧。

完成条件：主链合同、默认入口、R0 覆盖和旧代码清理验收通过。已验证：45 个 canonical/
v4/GINN 测试通过、compileall 通过、canonical summarize smoke 产出
`canonical_increment_ready`。阶段 9 仍负责 R1 物理闭环和 R0 延期别名的最终清理；physics
本身保持为下一阶段的独立门禁。

### 阶段 9（延期至微纹理前）：synthetic/real physics、R1 与部署资格

- [ ] physics 使用 canonical background 或明确的 deployment LFM 组合；
- [ ] `increment_l2_weight` 作为唯一增量正则字段；
- [ ] physics 阶段必须从已完成的 synthetic/real-well supervised checkpoint 初始化；
- [ ] waveform-best physics checkpoint 仅作为诊断；
- [ ] real-well-plus-physics 默认保留 experimental 标记；
- [ ] 真实工区 physics 记录前后 increment 漂移、LFM-only 指标、井间 QC 和 selection metric；
- [ ] R1 使用明确的 canonical/deployment closure 字段。

完成条件：synthetic/real physics 前向、反向和有限梯度通过；非法 physics-first、zero-initialized physics 和不具备部署资格的 checkpoint 明确失败；R1 闭环可复算。只有本阶段完成后才进入微纹理。

### 阶段 10（延期）：微纹理 bank 与领域无关 emitter

- [ ] 固定 `MicrotextureEmission` 接口、物理单位、seed 和 metadata；
- [ ] 在最终规则时间轴/TVDSS 轴上建立训练井 canonical texture bank；
- [ ] 检查训练井、留出井和 bank coverage 隔离；
- [ ] 实现 A 的 `none`、B 的 `thin_bed_cluster`、C 的 `canonical_well_patch`；
- [ ] 对对象边界、厚度、振幅、端点跳变、clipping 和 reversal 做 QC。

完成条件：固定 seed 完全确定；对象外扰动为零；所有厚度和长度以秒/米表达；A/B/C 的 emitter 输出均可独立复算。只有阶段 9 physics 门禁通过后才开启本阶段。

### 阶段 11（延期）：paired A/B/C benchmark

- [ ] 一次生成 `macro_parent`，再实例化三个 mode；
- [ ] 实现组级全成功/全拒绝和原因记录；
- [ ] 保持三组 benchmark 的 parent 集合、split、LFM、mismatch 和预算一致；
- [ ] 生成 `shared_none_test`、`shared_thin_bed_cluster_test`、`shared_canonical_well_patch_test`。

完成条件：三组 benchmark 的 `macro_parent_id` 集合一致；任一 mode 失败不会留下不完整 paired group；跨模式测试矩阵可运行。

### 阶段 12（延期）：A/B/C GINN 消融与后续入口评估

- [ ] A/B/C 使用相同架构、split、seed、normalization 和训练预算分别训练；
- [ ] 交叉测试 false texture、薄层漏检、井纹理覆盖和跨模式迁移；
- [ ] 只在合同、主链回归和 paired benchmark 全部通过后，评估是否切换微纹理默认入口；
- [ ] 微纹理实验与无纹理 v4 baseline 分开报告，不覆盖既有主链结果。

完成条件：微纹理 A/B/C 的数据生成、训练和评估均可独立复现；效果结论作为消融报告，不反向修改 canonical increment 语义。

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

阶段 3 的主链证据和阶段 4 的最小监督切片已经完成；20260714 生成目录中的 3 个 wedge
acceptance warning 仍作为生成质量记录，不阻塞当前 canonical 主链。阶段 5–8 已完成
四架构 closure、真实井 canonical increment 接线、R0 覆盖/反事实报告以及生产路径清理。
下一步只进入阶段 9：集中接入 physics 与 R1，并在该阶段完成 R0 延期别名的最终清理；
阶段 9 通过后才开启阶段 10–12 的微纹理与 paired A/B/C。每完成一项就在对应复选框勾选
并记录测试命令、fixture 和产物路径。

## 7. 阶段 1/2 验证证据

阶段 1/2 的历史验证（迁移前路径）已运行：

```text
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_canonical_increment.py tests/test_synthoseis_v4_canonical.py
```

结果：`11 passed`。该结果不替代阶段 2.5 迁移后的新路径验证。

覆盖内容：

- 时间域和深度域完整道分解恒等式；
- NaN 间隙独立滤波、短段显式失败和规则采样轴校验；
- v4 writer 写入 `canonical_background_log_ai`、`target_increment_log_ai`；
- 时间域和深度域 writer 均写入 canonical 字段；
- canonical 与 controlled 两个 LFM variant；

- LFM producer 的 canonical background 和 variant 复算；
- v4 reader 暴露 canonical 字段；
- v4 facade 拒绝冻结 v3 manifest；
- 固定 seed 的微纹理 `none` emitter 接口预留。

完整工区 v4 生成尚未作为长时间 smoke 运行；运行前应使用新的 v4 配置和新的输出目录，不覆盖 20260706 冻结目录。

补充验证：

- `python -m compileall -q src/cup` 通过；
- 当前深度 v4 组合配置可由 `load_composed_config` 与 `parse_depth_config` 解析；
- 读取 `experiments/synthoseis_lite/results/20260706/generate_field_conditioned` 时明确拒绝冻结的 v3 benchmark。

阶段 2.5 实施后的本地验证命令和结果：

```text
$env:PYTHONPATH = "src"
python -m compileall -q src/cup src/ginn_v2
python -m pytest -q -p no:cacheprovider tests/test_canonical_increment.py tests/test_synthoseis_v4_canonical.py tests/test_synthoseis_stage_2_5.py
```

结果：稳定化和对称性测试 `29 passed`。`compileall` 与 `cup.impedance`、synthetic
time/depth reader import smoke 同样通过。测试文件继续被 `.gitignore` 忽略；本节只记录
本机结果，不声称测试已提交。

稳定化烟测记录：

- 时间域：当前工作区没有可直接复用的真实时间域 source，本阶段只做 fixture reader/baseline 验证；
- 深度域：先由 `calibrate` 生成
  `scripts/output/synthoseis_lite_calibrate_20260713/impedance_calibration.json`，
  再以 `field_conditioned`、`--debug-attempt-limit 1` 和
  `--geometry-family none` 完成 writer → manifest → reader smoke，输出目录为
  `scripts/output/synthoseis_lite_symmetry_depth_smoke_20260713_fixed`；4 个场景生成成功，
  2 个对象参数组合在 preflight 被拒绝，整体状态为 `development_limited`；
- 深度域 baseline evaluator 使用 `--max-samples 2` 成功读取同一 smoke benchmark，
  输出目录为 `scripts/output/synthoseis_lite_evaluate_depth_smoke_20260713_fixed_rerun3`；
- 旧 v3 与阶段 2 之前生成的 v4 目录不覆盖、不迁移。

## 8. 本轮单一 mask 与深度 GPU 验证证据

本轮实现后的本地验证：

```text
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_canonical_increment.py \
  tests/test_synthoseis_v4_canonical.py \
  tests/test_synthoseis_stage_2_5.py
```

结果：`33 passed`；`compileall -q src/cup src/ginn_v2` 通过。新增回归覆盖深度静态
mismatch 的 halo 源支持与 GINN patch 的单一 mask/有限零填充。

深度域实际 smoke 输出：

```text
scripts/output/synthoseis_lite_single_valid_mask_depth_smoke_gpu2_20260714
```

该目录使用 `field_conditioned --debug-attempt-limit 1 --geometry-family none`，
manifest 记录 `requested_backend=auto`、`resolved_backend=torch_cuda`、`dtype=float64`；
HDF5 realization 只有 `masks/valid_mask`，reader 能加载 ROI 内有限的 observed 和
model-consistent seismic，baseline evaluator 也成功消费同一目录。完整 v4 工区仍需
在新输出目录重新生成，历史目录不覆盖。

显式 CPU 覆盖也已完成同样的 writer → manifest → reader smoke：

```text
scripts/output/synthoseis_lite_single_valid_mask_depth_smoke_cpu3_20260714
```

manifest 记录 `requested_backend=numpy`、`resolved_backend=numpy`、`dtype=float64`，
4 个场景生成成功，2 个对象参数组合在 preflight 被拒绝，整体状态为
`development_limited`。该 smoke 同时验证了带 halo 的深度静态 mismatch 先使用完整
源支持平移、再与公共 ROI 相交；所有变体最终共享同一 `valid_mask`。baseline
evaluator 也已用 `--max-samples 2` 读取该 CPU smoke 目录。

## 9. 相关文档

- [Canonical Increment 语义规格](GINN_V2_CANONICAL_INCREMENT_SEMANTICS.md)
- [Synthoseis-lite 微纹理生成规格](SYNTHOSEIS_MICROTEXTURE_GENERATION_DESIGN.md)
- [GINN v2 可组合训练设计](GINN_V2_COMPOSABLE_TRAINING_DESIGN.md)
