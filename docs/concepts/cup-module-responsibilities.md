# cup 模块职责

模块按输入、输出及计算含义组织。物理算子、井资产、地震工区和工作流产物各自维护对应的单位与数据口径。

| 模块 | 职责 | 主要入口 |
|---|---|---|
| `cup.physics` | 声学正演、AI–Vp 关系、多井物性关系拟合及正演后端执行 | `numpy_backend`、`torch_backend`、`relations`、`rock_physics`、`execution` |
| `cup.well` | 井资产、测井曲线、轨迹、时深转换、井震标定和井震评价 | `inventory`、`petrel`、`las`、`trajectory`、`auto_tie`、`wavelet_generation`、`tie`、`evaluation_support`、`controls` |
| `cup.seismic` | 工区几何、地震读取和采样、层位、子波及体导出 | `survey`、`petrel`、`geometry`、`trace_sampling`、`wavelet`、`forward_inputs`、`volume_export` |
| `cup.synthetic` | 合成基准的目标定义、固定阻抗分解、采样与生成 | `core.canonical`、`core.pipeline`、`time`、`depth` |
| `cup.lfm` | 第七步真实工区低频模型构建、变体与产物读取 | `builders`、`pipeline`、`artifacts` |
| `cup.config` | 工区配置、来源运行目录选择与工作流产物规则 | `workflow`、`sources`、`artifacts` |
| `cup.utils` | 通用转换、布尔区间、基础统计、文本、路径、序列化和日志 | `coerce`、`masks`、`statistics`、`text`、`io`、`logging` |

## `cup.well` 内部边界

`cup.well` 按井资产、工作流步骤和跨步骤基础能力分层。第四步和第五步各自拥有步骤专用模块；两步之间只通过 `tie` 中定义的公共数据契约和数值操作衔接。

| 模块 | 所属范围 | 职责 |
|---|---|---|
| `inventory`、`petrel`、`las`、`curves`、`preprocess`、`trajectory` | 井资产基础 | 读取井头、层位、轨迹和 LAS，筛选曲线，完成井曲线预处理及 MD、TWT、TVDSS 之间的采样转换。 |
| `auto_tie` | 第四步 | 选择标定路线，生成标定计划和搜索空间，准备时间域或深度域井震输入，执行逐井自动标定，并导出标定曲线、时深关系和标定结果。Wtie 预训练资产的解析也属于这一步的标定资源入口。 |
| `wavelet_generation` | 第五步 | 读取候选子波与标定产物，定义候选和评测井数据对象，统一准备地震、反射系数与阻抗，组织井的空间簇，在时间网格或固定原生深度网格上评价子波。共识形态搜索使用 `cup.seismic.wavelet_consensus`。 |
| `tie` | 第四步与第五步共用 | 定义标定计划和标定结果等公共数据契约；统一按井名组织标定记录，处理短缺口与连续有效段，完成地震重采样和合成记录匹配指标计算。 |
| `controls` | 第六步 | 组装井控数据集，按井用途筛选监督井和低频背景井，执行井控质检，并序列化后续步骤使用的井控产物。 |
| `evaluation_support` | 第六步与第八步共用 | 根据目标层段、观测地震和井曲线支持范围生成并读取固定的井级评价区间，使井控质检和反演评价使用同一评价支持。 |
