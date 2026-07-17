# ADR 001：Synthoseis-lite 共享 Pipeline 与域 Adapter

## 状态

已采用。

## 背景

时间域和深度域需要共享同一套父实现、地震视图、索引、接受率和 reader 语义。把这些逻辑分别实现会使一个视图或一个合同修订在两个入口产生不同的采样分布和产物含义。

## 决策

- 域无关编排进入共享 `SyntheticBenchmarkPipeline`；
- 时间域和深度域通过 `TimeSyntheticDomainAdapter` 与 `DepthSyntheticDomainAdapter` 接入同一 Seam；
- `SeismicViewPipeline` 使用原子算子目录和有序视图列表，base 独立于 view；
- benchmark 使用 v5 双索引：父实现索引与 seismic view 索引；
- 每个父实现只物化一份 canonical background，训练输入直接引用该背景；
- GINN v2 由实验套件拥有 split，按父实现先抽样再抽样视图，训练和验证权重显式记录。

## 结果

共享 Pipeline 的修改同时覆盖时间域和深度域；两个 Adapter 只承载采样轴、单位、域专属上游适配和正演执行。视图数量不会隐式改变父实现概率，索引和 checkpoint 可以独立复算其身份与权重。

v5 是不兼容版本。旧 schema、旧平铺索引、旧低频退化字段和旧训练字段不进入当前 reader 或 checkpoint provenance。

## 复查条件

若新增域必须修改共享父实现循环、视图循环、双索引或接受率聚合，而不是只增加 Adapter 实现，说明 Seam 仍未集中足够的域无关逻辑；若某个算子没有独立的合同 fingerprint 和随机身份，也不得进入活动配置。
