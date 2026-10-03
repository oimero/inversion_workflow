# cup 模块职责

模块按输入、输出及计算含义组织。物理算子、井资产、地震工区和工作流产物各自维护对应的单位与数据口径。

| 模块 | 职责 | 主要入口 |
|---|---|---|
| `cup.physics` | 声学正演、AI–Vp 关系、多井物性关系拟合及正演后端执行 | `numpy_backend`、`torch_backend`、`relations`、`rock_physics`、`execution` |
| `cup.well` | 井资产、测井曲线、轨迹、时深转换和井震标定 | `inventory`、`petrel`、`las`、`trajectory`、`tie`、`controls` |
| `cup.seismic` | 工区几何、地震读取和采样、层位、子波及体导出 | `survey`、`petrel`、`geometry`、`trace_sampling`、`wavelet`、`forward_inputs`、`volume_export` |
| `cup.synthetic` | 合成基准的目标定义、固定阻抗分解、采样与生成 | `core.canonical`、`core.pipeline`、`time`、`depth` |
| `cup.lfm` | 第七步真实工区低频模型构建、变体与产物读取 | `builders`、`pipeline`、`artifacts` |
| `cup.config` | 工区配置、来源运行目录选择与工作流产物规则 | `workflow`、`sources`、`artifacts` |
| `cup.utils` | 通用转换、布尔区间、基础统计、文本、路径、序列化和日志 | `coerce`、`masks`、`statistics`、`text`、`io`、`logging` |
