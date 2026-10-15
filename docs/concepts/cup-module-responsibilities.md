# cup 模块职责

模块按输入、输出及计算含义组织。物理算子、井资产、地震工区和工作流产物各自维护对应的单位与数据口径。

| 模块 | 职责 | 主要入口 |
|---|---|---|
| `cup.physics` | 声学正演、AI–Vp 关系、多井物性关系拟合和正演有效区间 | `numpy_backend`、`torch_backend`、`relations`、`rock_physics`、`execution`、`support` |
| `cup.well` | 井资产、测井曲线、轨迹、时深转换和井震处理 | `inventory`、`petrel`、`las`、`trajectory`、`gaps`、`auto_tie`、`wavelet_generation`、`tie`、`evaluation_support`、`controls` |
| `cup.seismic` | 工区几何、地震读取和采样、层位、子波及体导出 | `survey`、`petrel`、`geometry`、`trace_sampling`、`wavelet`、`forward_inputs`、`volume_export` |
| `cup.synthetic` | 合成基准的目标定义、固定阻抗分解、采样与生成 | `core.canonical`、`core.pipeline`、`time`、`depth` |
| `cup.lfm` | 第七步真实工区低频模型构建、变体与产物读取 | `builders`、`pipeline`、`artifacts` |
| `cup.config` | 工区配置、来源运行目录选择与工作流产物规则 | `workflow`、`sources`、`artifacts` |
| `cup.utils` | 通用转换、布尔区间、基础统计、文本、路径、序列化和日志 | `coerce`、`masks`、`statistics`、`text`、`io`、`logging` |

## `cup.well` 内部边界

`cup.well` 按井资产、工作流步骤和跨步骤能力分层。

| 模块 | 所属范围 | 职责 |
|---|---|---|
| `inventory`、`petrel`、`las`、`curves`、`preprocess`、`trajectory` | 井资产基础 | 读取井资产和曲线，完成预处理、轨迹及 MD、TWT、TVDSS 之间的采样转换。 |
| `auto_tie` | 第四步 | 组织井震标定路线、输入和结果。 |
| `wavelet_generation` | 第五步 | 根据标定产物评价候选子波并生成全局子波。 |
| `tie` | 第四步与第五步共用 | 提供公共标定数据结构、连续窗口、地震重采样和匹配指标。 |
| `gaps` | 第四步至后续井控步骤共用 | 统一处理短缺口和有效性掩码。 |
| `controls` | 第六步 | 组装井控、生成监督曲线并执行井控质检。 |
| `evaluation_support` | 第六步与第八步共用 | 定义井曲线和波形的固定评价区间。 |
