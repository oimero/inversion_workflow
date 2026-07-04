# 深度域正演重构剩余工作清单

> 盘点日期：2026-07-04
> 基线：[深度域正演能力重构设计](DEPTH_DOMAIN_FORWARD_REFACTOR.md)
> 结论：深度域 GINN v2 物理损失、R0 通用采样轴、R1 TVDSS 正演闭环和规格/教程同步已闭合。尚未完成的是重复正演实现清理、real-delta 深度正演 QC、旧 schema 错配测试和 xline=4 端到端验收。

本文只记录当前代码中仍需完成的工作。已完成的 P0/P2 项从本文移除，实现细节见 git 历史。

## 1. P1：统一实现与契约收口

### 1.1 删除重复正演实现

现状：

- `src/cup/seismic/observability.py` 仍定义 `acoustic_reflectivity_from_log_ai` 和 `forward_log_ai`。
- `src/cup/synthetic/canonical.py`、`generation.py`、`forward.py`、`probes.py` 仍直接消费旧 `forward_log_ai`，沿用 N-1 点时间域语义。

剩余工作：

- [ ] 将时间域 Synthoseis 和可观测性旁路迁到 `cup.physics` 的公共函数。
- [ ] 调用方显式传递秒制子波时间轴，并按 N 点合成地震契约调整掩码和对齐。
- [ ] 删除 `forward_log_ai` 和重复反射率公式；不保留兼容包装。
- [ ] 搜索门禁保证新生产代码中不再出现重复 Robinson 卷积或重复反射率实现。

验收条件：新通用工作流的正演和反射率实现来源只有 `src/cup/physics/`；旁路模块可以保留自身诊断实现，但不能保留第二套生产正演内核。

### 1.2 深度 real-delta 的正演 QC

现状：

- `src/ginn_v2/real_delta.py` 的井旁正演图仍固定调用 `forward_time`，并使用 `twt`、TDT 窗口和时间域体采样。

剩余工作：

- [ ] real-delta 正演 QC 按井控/LFM 域分派；深度域使用 TVDSS、冻结 AI–Vp 关系和 `forward_depth`。
- [ ] 深度域 QC 窗口、图轴、指标和产物字段使用米制语义。
- [ ] 增加深度 real-delta 正演图与错域来源测试。

验收条件：深度 real-delta 不再进入任何 TWT 专用正演或采样路径。

### 1.3 旧 schema 与域错配的统一失败行为

剩余工作：

- [ ] 数据集、模型、R0 和 R1 分别增加旧 schema、缺失域、错单位、错 depth basis 和上游契约不一致的失败测试。
- [ ] 错误信息包含实际版本、期望版本和对应重建入口。
- [ ] 不通过字段存在性推断旧产物的域，不为旧产物注入默认值。

验收条件：任何缺少明确采样域和轴契约的旧产物都无法进入深度生产主链。

## 2. P1：几何和端到端验收

xline=4 测试目前覆盖几何换算和局部模块，尚未覆盖整条深度生产主链。

- [ ] 增加 inline 步长 1、xline 步长 4 的深度 volume 集成 fixture，贯穿体读取、井控、LFM、R0、R1 和体导出。
- [ ] fixture 至少使用三个 xline，断言实际线号轴为等差 4，而数组下标仍为连续 0、1、2。
- [ ] 在井位置、section 抽取、volume patch、R1 观测采样和导出回读处分别断言线号与 XY 坐标。
- [ ] 用户使用项目指定 Python 环境执行现有测试和新增测试，并记录真实工区 smoke run 结果。

验收条件：端到端结果不存在 `line_number - first_line` 下标假设，xline=4 时不会取到错误道或写到错误位置。

## 3. 建议实施顺序

1. 迁移剩余时间域调用并删除重复正演实现。
2. 补齐 real-delta 深度正演 QC。
3. 补齐旧 schema 错配失败测试。
4. xline=4 端到端测试，由用户执行本地测试和真实工区 smoke run。
