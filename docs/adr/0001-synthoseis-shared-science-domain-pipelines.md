# ADR-0001：Synthoseis-lite 共享科学数值与分域工程管线

- 状态：已接受
- 日期：2026-07-17

## 背景

Synthoseis-lite 同时支持时间域和深度域。两个域共享井约束地下真值、规则轴投影、
canonical decomposition、受控低频模型退化、地震变体、随机流和公共数值 QC；输入坐标、
正演物理、上游来源、运行方式和 Artifact 表达存在真实差异。

完整统一工程编排会扩大 Interface，引入大量可选字段、hook 和领域条件。这样的 Module
缺少 Depth，也会降低领域行为的 Locality。分别维护工程管线不会损害科学一致性，前提是
Pipeline 不实现域无关科学算法。

## 决策

Synthoseis-lite 采用以下架构：

1. 域无关科学与数值 Module 保持单一 Implementation，包括 calibration record、地下真值、
   projection、support、canonical decomposition、LFM degradation、seismic variants、
   随机流和公共数值 QC。
2. 时间正演和深度正演构成真实 Seam，分别由 `TimeForwardAdapter` 和
   `DepthForwardAdapter` 满足公共 Forward Interface。
3. 时间域和深度域工程 Pipeline 分别管理配置、上游输入、attempt 生命周期、日志、进度、
   manifest、sample index、图件以及领域 Artifact 字段。
4. Writer 和 Reader 可以共享合同与执行工具，但不以单一域中立 Implementation 为目标。
   稳定消费 Interface 为 `SynthoseisBenchmark` 和样本 Protocol。
5. Pipeline 只决定何时调用、调用多少次、输入来源和结果记录；改变相同无量纲输入科学
   数值结果的 Implementation 必须位于共享 Module。
6. science v2 的随机命名合同保持冻结。任何移除 Artifact schema 随机命名字段的调整都
   必须进入新的 science revision。

## 架构门禁

- 科学 Module 不导入 HDF5 Writer、Reader、reporting、GINN 或分域 Pipeline。
- Pipeline 不维护私有 FIR、projection、LFM、noise、gain 或随机流 Implementation。
- 两个域对相同无量纲输入的公共数值结果通过跨域测试。
- Forward Adapter 之外不实现领域正演。
- Writer 只物化已构建产品；Reader 严格验证合同，不修补或猜测旧字段。

## 后果

- 科学变更集中在共享 Module，获得跨域 Leverage 和维护 Locality。
- 时间域与深度域工程工作流可以按各自数据来源和物理需求独立演化。
- 部分生命周期、Writer 和 Reader 代码会保持平行；代码行数重复本身不构成抽取理由。
- 新增域无关科学能力时必须同时满足时间域和深度域合同测试。

## 重审条件

仅在以下情况重审本决策：

- 两个 Pipeline 出现相同的域无关科学或数值 Implementation；
- 新增第三个物理域，使现有两个 Adapter 的 Interface 无法保持稳定；
- 出现第二个 Benchmark 外的完整正演观测调用方，需要独立的观测产品 Module；
- Writer 或 Reader 的合同漂移反复产生同类缺陷，证明更深的公共 Module 能提供实际
  Leverage。
