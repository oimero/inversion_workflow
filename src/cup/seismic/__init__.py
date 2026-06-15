"""cup.seismic: 地震体 Adapter、工区几何与地震处理工具。

子模块
------
- survey: SEG-Y/ZGY 地震体 Adapter、采样轴与井旁道提取。
- geometry: 规则 inline/xline 工区几何、XY 变换、采样窗口与 XY 距离计算。
- horizon: 单层位解释面构建、异常点剔除与双线性采样。
- wavelet: 地震子波加载、生成、归一化与频谱属性。
- wavelet_consensus: 对齐候选子波的 PCA 共识搜索。
- trace_sampling: 直井与斜井的轨迹落道计划和地震道组装。
- viz: 前五步井震与子波 QC 绘图。
- observability: 第五步后的前向可观测性研究闸门数值核心。
"""
