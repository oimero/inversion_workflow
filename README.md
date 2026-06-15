## 自动井震标定与时间域反演工作流

当前稳定生产链截止第五步：井资产盘点、曲线筛选、测井预处理、井震自动标定和全局子波生成。
第五步之后的旧实现已从活动树移除，后续反演先经过可观测性分析、truth-first 合成基准和模型消融。

重构原则、清理记录和研究顺序以
[时间域反演重置](docs/concepts/time-domain-inversion-reset.md)为唯一权威入口。

当前已经落地的下一项研究工具是
[前向可观测性闸门](docs/concepts/forward-observability-gate.md)，运行入口为
`scripts/forward_observability.py`；它不属于编号生产步骤。

当前活动设计入口是
[Truth-First `synthoseis-lite` 基准](docs/concepts/synthoseis-lite-benchmark.md)。
它将建立已知真值的二维合成评测闸门，暂不定义神经网络架构。
