# 反演工作流

当前稳定时间域工作流截止第五步。旧后半程实现已经从活动树移除，
后续反演从研究闸门重新开始。

开始后续工作前，请先阅读[时间域反演重置](concepts/time-domain-inversion-reset.md)。
该文档是重构原则、研究顺序和清理边界的唯一权威入口。

[前向可观测性闸门](concepts/forward-observability-gate.md)已经落地，对应无编号脚本
`scripts/forward_observability.py`。

当前活动设计入口是
[Truth-First `synthoseis-lite` 基准](concepts/synthoseis-lite-benchmark.md)，用于建立
已知真值的二维合成评测闸门。
