"""cup.seismic: 地震体 Adapter、工区几何与地震处理工具。

子模块
------
- survey: SEG-Y/ZGY 地震体 Adapter、采样轴与井旁道提取。
- geometry: 规则 inline/xline 工区几何、XY 变换、采样窗口与 XY 距离计算。
- horizon: 单层位解释面构建、异常点剔除与双线性采样。
- target_zone: 多层位目标层段构建、厚度修复、QC 掩码与三维采样。
- modeling: 采样域无关的层位约束低频模型构建核心。
- lfm_time: 时间域低频模型完整流程（曲线滤波、时深转换、建模）。
- lfm_depth: TVDSS 深度域低频模型完整流程。
- facies_control_depth: 深度域岩相控制点交互式 QC 与空间混入。
- process: 旧层位插值入口（已废弃，请使用 horizon 模块）。
- viz: 可视化占位模块。
"""
