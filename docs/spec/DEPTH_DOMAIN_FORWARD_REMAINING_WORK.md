# 深度域正演重构剩余工作清单

> 盘点日期：2026-07-04  
> 盘点基线：[深度域正演能力重构设计](DEPTH_DOMAIN_FORWARD_REFACTOR.md)  
> 结论：统一物理内核、深度域 Synthoseis、真实井控制与 LFM 主体已经落地；尚未闭合的是深度域 GINN v2 物理损失、R0 轴契约、R1 正演闭环、重复正演实现清理和端到端验收。

本文只记录当前代码中仍需完成的工作。测试仅检查了源码与现有测试用例是否存在，未在本次盘点中执行。

## 1. 已完成且不列入剩余工作的范围

- `src/cup/physics/` 已提供 NumPy/PyTorch 的时间域和深度域正演、深度算子分块计算、AI–Vp 关系及网格适配器。
- 深度域 Synthoseis 已使用 `forward_depth`，并保存 N 点深度域地震、速度和 TVDSS 轴。
- 真实井控制与 LFM 已使用统一的 `WellControlSet`、统一构建器、显式 variant 和通用采样轴；深度域要求 TVDSS。
- `SurveyLineGeometry` 使用显式线号轴；现有测试覆盖 inline 步长 1、xline 步长 4 的几何换算、深度 Synthoseis、井控与 LFM 局部路径。
- 深度域 Jacobian/SVD 可观测性扩展仍是非阻塞旁路，不属于本轮验收条件。
- `forward_model_inputs_sha256` 和逐文件 SHA-256 是历史契约。当前实现采用契约指纹和直接上游引用，剩余改造不得重新引入旧字段。

## 2. P0：闭合深度域生产主链

### 2.1 GINN v2 深度域物理损失

现状证据：

- `src/ginn_v2/training.py` 明确拒绝深度 benchmark 与非零物理损失的组合。
- 同一文件仍调用本地 `_torch_forward_log_ai`，输出 N-1 点，并把观测地震和掩码裁成 `[..., 1:]`。
- 模型 manifest 的物理算子标识仍是时间域旧实现，patch 字段仍使用 `twt_*`；manifest 未把采样域、采样单位和深度基准发布为模型语义。

剩余工作：

- [ ] 删除深度域物理损失的显式禁用分支。
- [ ] 训练路径按 benchmark 的采样域分派到 `cup.physics.torch_backend.forward_time` 或 `forward_depth`。
- [ ] 深度分支显式提供 TVDSS 轴、秒制子波时间轴，以及由冻结 AI–Vp 关系得到的正速度；不得从样点数或字段名猜测。
- [ ] 时间域和深度域物理损失均使用 N 点合成地震和 N 点有效掩码，不再裁掉首样点。
- [ ] 从模型 manifest 到 checkpoint 发布并校验采样域、单位、深度基准、正演输入上游契约和正演算子标识。
- [ ] R0 加载模型时校验模型域与所选 LFM/地震域完全一致，拒绝把时间域模型用于深度域输入。
- [ ] 增加深度物理损失训练 smoke test、N 点对齐测试、梯度测试和错域模型拒绝测试。

验收条件：深度 benchmark 可以启用非零物理损失完成一个最小训练步；生产训练正演只经过 `cup.physics`；模型产物足以独立判断其采样域和正演来源。

### 2.2 R0 真正使用通用采样轴

现状证据：

- `src/ginn_v2/real_field.py` 的 `RealFieldSection`、`RealFieldVolume`、patch 索引和预测产物仍以 `twt_s`/`twt_*` 表达采样轴，即使输入 manifest 声明为深度域。
- `scripts/real_field_zero_shot.py` 对任意域都计算 `dt_s`、Hz 频带、秒制边缘侵蚀，并在摘要中写 `n_twt`、`twt_start_s` 和 `twt_stop_s`。
- 深度轴当前会被当作秒轴参与频谱和边缘宽度计算；这不是命名问题，会产生错误数值。
- R0 只核对 LFM 与顶层地震的域，尚未核对模型训练域。
- R0 当前强制解析时间域第五步的 `selected_wavelet.csv`；深度链的冻结子波和 AI–Vp 关系实际来自岩石物理分析的 `forward_model_inputs.json`。

剩余工作：

- [ ] 用显式的通用采样轴值对象替代 R0 内部的 `twt_s` 轴；轴必须携带 domain、unit 和 depth basis。
- [ ] 预测 NPZ、patch index、模型摘要和总摘要按域发布 TWT 或 TVDSS，不在深度产物中写秒制轴字段。
- [ ] patch 尺寸与步长使用样点数语义；物理宽度配置按域分别使用秒或米，错域字段立即报错。
- [ ] 深度域禁用 Hz 频谱 QC，或另行设计明确的米制空间频率 QC；不得把米制采样间隔传给 `rfftfreq` 后标成 Hz。
- [ ] 体导出使用通用 samples 轴并保留 depth/TVDSS 元数据。
- [ ] 深度 R0 的来源契约引用 `forward_model_inputs_v2` 及其直接上游，不把时间域 `wavelet_generation_dir` 伪装成深度正演来源。
- [ ] 补充深度 section、深度 volume、错域配置、时间模型配深度 LFM、深度轴摘要和体导出的测试。

验收条件：深度 R0 的内存对象、NPZ、CSV、JSON 和体导出中均没有把 TVDSS 当作 TWT/秒使用；时间域原路径继续保持时间语义。

### 2.3 R1 TVDSS 正演闭环

现状证据：

- `scripts/real_field_forward_diagnostic.py` 只导入并调用 `forward_time`。
- 井旁 QC 通过 TDT 把 LAS 投影到 TWT，采样函数、绘图和摘要都只接受 TWT。
- 主合成、候选子波扫描、频带 QC 和契约指纹都固定为时间域正演。
- 当前扫描只有子波相位/时间方向偏移，没有与其独立的米制深度静差扫描。

剩余工作：

- [ ] R1 从 R0 契约读取 domain、sample unit、depth basis 和通用采样轴，并严格校验所有模型共享同一轴。
- [ ] 深度分支读取冻结子波和 AI–Vp 关系，由预测 logAI 生成正速度并调用 `forward_depth`。
- [ ] 深度分支直接在 TVDSS 上读取 shifted LAS/井控，不经过 TDT 或 MD→TWT 投影。
- [ ] 合成、观测、预测 logAI 和 TVDSS 均保持 N 点逐点一致；反射率和 forward-valid mask 保持 N-1 点。
- [ ] 子波秒制相位/时间扰动与米制深度静差使用独立配置、独立扫描维和独立摘要字段。
- [ ] 深度域不运行以等间隔 TWT 为前提的 Hz 频带诊断；保留适用于 TVDSS 的波形闭环指标。
- [ ] 发布新的 R1 契约版本，记录采样域、轴、正演输入上游契约、扰动参数、严格轴检查和闭环指标；旧 R1 摘要必须明确失败。
- [ ] 增加深度 section/volume R1、轴错配、非法速度、错单位、独立扰动和旧 schema 拒绝测试。

验收条件：给定深度 R0 产物，R1 可以在 TVDSS 原生轴上完成正演闭环，且所有合成地震只经过 `cup.physics.forward_depth`。

## 3. P1：统一实现与契约收口

### 3.1 删除重复正演实现

现状证据：

- `src/ginn_v2/training.py` 仍定义 `_torch_forward_log_ai`。
- `src/cup/seismic/observability.py` 仍定义 `acoustic_reflectivity_from_log_ai` 和 `forward_log_ai`。
- `src/cup/synthetic/canonical.py`、`generation.py`、`forward.py`、`probes.py` 仍直接消费旧 `forward_log_ai`，沿用 N-1 点时间域语义。

剩余工作：

- [ ] 将 GINN v2、时间域 Synthoseis 和可观测性旁路迁到 `cup.physics` 的公共函数。
- [ ] 调用方显式传递秒制子波时间轴，并按 N 点合成地震契约调整掩码和对齐。
- [ ] 删除 `_torch_forward_log_ai`、`forward_log_ai` 和重复反射率公式；不保留兼容包装。
- [ ] 搜索门禁保证新生产代码中不再出现重复 Robinson 卷积或重复反射率实现。

验收条件：新通用工作流的正演和反射率实现来源只有 `src/cup/physics/`；旁路模块可以保留自身诊断实现，但不能保留第二套生产正演内核。

### 3.2 深度 real-delta 的正演 QC

现状证据：

- real-delta 的控制点和标签路径已有深度域测试。
- `src/ginn_v2/real_delta.py` 的井旁正演图仍固定调用 `forward_time`，并使用 `twt`、TDT 窗口和时间域体采样。

剩余工作：

- [ ] real-delta 正演 QC 按井控/LFM 域分派；深度域使用 TVDSS、冻结 AI–Vp 关系和 `forward_depth`。
- [ ] 深度域 QC 窗口、图轴、指标和产物字段使用米制语义。
- [ ] 增加深度 real-delta 正演图与错域来源测试。

验收条件：深度 real-delta 不再进入任何 TWT 专用正演或采样路径。

### 3.3 旧 schema 与域错配的统一失败行为

剩余工作：

- [ ] 数据集、模型、R0 和 R1 分别增加旧 schema、缺失域、错单位、错 depth basis 和上游契约不一致的失败测试。
- [ ] 错误信息包含实际版本、期望版本和对应重建入口。
- [ ] 不通过字段存在性推断旧产物的域，不为旧产物注入默认值。

验收条件：任何缺少明确采样域和轴契约的旧产物都无法进入深度生产主链。

## 4. P1：几何和端到端验收

现有 xline=4 测试主要覆盖几何换算和局部模块，尚未覆盖完成后的整条深度生产主链。

- [ ] 增加 inline 步长 1、xline 步长 4 的深度 volume 集成 fixture，贯穿体读取、井控、LFM、R0、R1 和体导出。
- [ ] fixture 至少使用三个 xline，断言实际线号轴为等差 4，而数组下标仍为连续 0、1、2。
- [ ] 在井位置、section 抽取、volume patch、R1 观测采样和导出回读处分别断言线号与 XY 坐标。
- [ ] 用户使用项目指定 Python 环境执行阶段 4/5 的现有测试和新增测试，并记录真实工区 smoke run 结果。

验收条件：端到端结果不存在 `line_number - first_line` 下标假设，xline=4 时不会取到错误道或写到错误位置。

## 5. P2：规格与教程同步

现状证据：

- 基线规格开头仍写“阶段 4–5 待实施”，实施顺序却写“阶段 4 已实施、R0 已切 v2”。
- 基线规格引用的 `SHA256_CONTRACT_SLIMMING.md` 和 `UNIFIED_REAL_FIELD_LFM_V2.md` 在当前 `docs/spec/` 中不存在。
- 当前代码已经使用 Synthoseis v3、LFM v3 和契约指纹，而基线规格若干段落仍描述 v1/v2 与旧 SHA 字段。
- R0/R1 教程仍只描述 TWT、秒制边缘和时间域正演。

剩余工作：

- [ ] 在 P0/P1 完成后更新基线规格的阶段状态、schema 名称和验收结论。
- [ ] 恢复缺失的现行规格，或把失效链接改到实际存在的规范；只保留一个现行契约来源。
- [ ] 更新深度域工作流、R0 和 R1 教程，使其只描述届时实际存在的配置、轴、产物和失败规则。
- [ ] 文档明确区分秒制时间子波、TWT 地震轴和米制 TVDSS 地震轴。

## 6. 建议实施顺序

1. 先把模型 manifest/checkpoint 的采样域契约补齐，并完成深度物理损失。
2. 再把 R0 改为通用采样轴，消除深度轴上的秒制计算。
3. 在稳定的 R0 产物契约上实现 R1 TVDSS 闭环。
4. 迁移剩余时间域调用并删除重复正演实现。
5. 补齐 xline=4 端到端测试，由用户执行本地测试和真实工区 smoke run。
6. 最后同步基线规格和教程，关闭阶段 4–5。
