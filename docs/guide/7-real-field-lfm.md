# 07 真实工区低频模型

`real_field_lfm.py` 是工作流的第七步。它读取第六步井控数据、当前地震体和目标层位，在指定输出网格上构建低频模型，并按配置写出基线模型、修饰结果和变体比较。

完成后可将选定变体作为[第八步主体反演](8-ginn-v2-body-inversion.md)的初始模型，并同时提供第六步井控。

---

## 快速开始

```powershell
python scripts/real_field_lfm.py
python scripts/real_field_lfm.py --config experiments/<project>.yaml
python scripts/real_field_lfm.py --output-dir <output_dir>
```

运行前确保第六步已完成。脚本会自动发现最近一次包含所需文件且采样域匹配的第六步运行，或通过配置中的 `well_control_run_dir` 显式指定。已有输出目录会被拒绝。

---

## 运行前需要什么

| 来源 | 文件 | 用途 |
|------|------|------|
| 第六步 | `run_summary.json` | 井控数据格式、采样域和来源信息 |
| 第六步 | `well_control_manifest.csv` | 成功井清单和逐井 NPZ 路径 |
| 第六步 | `wells/<well_name>.npz` | 每井规范化 log(AI) 和逐样点位置 |
| 数据目录 | 时间域地震体 | 目标时间采样轴、工区几何和线网 |
| 数据目录 | 层位文件 | 解释层位，定义目标区间的几何边界 |
| 可选 | framework body CSV | 仅使用 framework modifier 时需要 |

---

## 配置参考

```yaml
workflow_config: experiments/<workflow_config>.yaml

real_field_lfm:
  source_runs:
    well_control_run_dir:                        # 留空自动发现最新第六步

  output_geometry:
    mode: volume                                 # 全工区规则体

  baselines:
    trend_main:
      method: trend
      filter:
        enabled: true
        cutoff_hz: <cutoff_hz>
        order: 6
        buffer_mode: reflect
        buffer_axis_units: <buffer_seconds>
      fit:
        min_valid_samples_per_well: 32
        huber_f_scale_log_ai: 0.05
      spatial: {variogram: spherical, exact: true, nugget: 0.0}

    proportional_slice_main:
      method: proportional_kriging
      filter:
        enabled: true
        cutoff_hz: <cutoff_hz>
        order: 6
        buffer_mode: reflect
        buffer_axis_units: <buffer_seconds>
      n_slices: 32
      spatial: {variogram: spherical, exact: true, nugget: 0.0}

  modifiers: {}

  variants:
    - variant_id: trend_baseline
      baseline_id: trend_main
      modifier_ids: []
    - variant_id: proportional_slice_baseline
      baseline_id: proportional_slice_main
      modifier_ids: []

  comparisons: []
```

### 顶层配置

本指南按时间域工区说明第七步。组合后的工作流配置中，`seismic.domain` 应为 `time`，并提供时间域地震文件。顶层配置的 `target_interval.horizons` 必须声明至少两个层位的名称和文件路径。首个层位和末尾层位定义趋势窗的顶和底；中间层位参与目标区间的几何定义和连续性检查，不增加趋势参数。路径相对于 `data_root`，层位值进入模型前统一到时间采样轴。

配置中的数据路径和层位名称应使用当前工区的实际值，例如：

```yaml
data_root: <data_root>
target_interval:
  horizons:
    - {name: <top_horizon_name>, file: <top_horizon_file>}
    - {name: <bottom_horizon_name>, file: <bottom_horizon_file>}
```

### `source_runs`

指向第六步产出的井控集运行目录。留空时自动发现最近一次 `real_field_well_controls_*` 运行。加载时检查数据模式、采样域和单位、采样轴、坐标及数组形状，并确认井控记录的目标地震体与当前配置一致。

### `output_geometry`

控制低频模型输出的空间范围。三种模式互斥，每种有各自的参数要求。所有第六步成功井都参与建模——`output_geometry` 只控制最终写出体的网格范围，不减少使用的控制井。

#### `volume` — 全工区体

输出轴与源地震严格一致。不需要任何额外参数。

```yaml
output_geometry:
  mode: volume
```

> 输出轴与源地震严格一致，只有 volume 模式能导出 SEG-Y/ZGY。

#### `window` — 矩形子体

从全工区中裁出一个矩形窗口。六个参数缺一不可，端点必须精确落在当前地震采样轴上；inline 和 xline 端点必须落在真实线网上。

```yaml
output_geometry:
  mode: window
  inline_min: <inline_min_on_axis>
  inline_max: <inline_max_on_axis>
  xline_min: <xline_min_on_axis>
  xline_max: <xline_max_on_axis>
  sample_min: <sample_min_on_axis>
  sample_max: <sample_max_on_axis>
```

| 参数 | 含义 |
|------|------|
| `inline_min` / `inline_max` | inline 起止线号（含） |
| `xline_min` / `xline_max` | xline 起止线号（含） |
| `sample_min` / `sample_max` | 时间采样轴起止（含） |

下标轴只接受真实线号。线号步长不为单位步长时，相邻有效线号仍然不能当成数组下标。

#### `section` — 二维剖面

由折线路径定义，沿路径按 XY 米制距离均匀采样 `n_traces` 个道位置。输出线号可以是浮点值，不要求重新吸附到地震线网。

```yaml
output_geometry:
  mode: section
  points:
    - {inline: <inline_start>, xline: <xline_start>}
    - {inline: <inline_end>, xline: <xline_end>}
  n_traces: <n_traces>
  sample_min: <sample_min_on_axis>
  sample_max: <sample_max_on_axis>
```

| 参数 | 含义 |
|------|------|
| `points` | 至少两个 `{inline, xline}` 端点，定义折线路径 |
| `n_traces` | 沿路径均匀采样的道数，至少为 2 |
| `sample_min` / `sample_max` | 时间采样轴起止（含） |

`n_traces` 至少为 2。在 XY 米制空间均匀采样后，路径位置再转换为浮点 inline/xline；它只决定剖面采样密度。

### `baselines`

两种基线模型方法各解决不同的偏差-方差权衡。

#### trend

该方法用一条相对层位坐标上的直线表示每口井的纵向阻抗趋势。

1. 对每口有效控制井，在首层位到底层位之间计算归一化层位坐标，并用 Huber 回归拟合截距和斜率。
2. 在真实 XY 米制坐标上分别插值每口井的截距和斜率，得到两个横向参数场。
3. 将两个参数场代回层位坐标公式，重建输出网格中的对数阻抗。

控制语义：

| 拟合有效井数 | 行为 |
|---:|---|
| 0 | 变体失败 |
| 1 | 生成 `single_control_constant` 场，方差为零 |
| ≥2 且参数值退化 | 变体失败 |
| ≥2 且正常 | XY ordinary kriging |

#### proportional_kriging

该方法直接估计多个相对层位位置的横向场，再沿层位坐标重建纵向变化。

1. 对每对相邻层位定义的层段，按等距相对层位位置在每口井上采样滤波后的对数阻抗。
2. 每个切片独立进行基于真实 XY 坐标的普通克里金插值。
3. 沿相对层位坐标在相邻切片之间线性插值，重建层段内所有样点。
4. 按层位顺序拼接所有层段。

切片控制不足时的补齐行为：

| 有限控制数 | 行为 | mode |
|---:|------|------|
| 0 | 同层段存在其他有效切片时，由最近上下切片线性补齐 | `neighbor_slice_fill` |
| 1 | 全平面使用该控制值 | `single_control_constant` |
| ≥2 且非退化 | XY ordinary kriging | `kriging` |
| ≥2 且退化 | 失败 | 无产物 |

补齐不跨越层段边界。整个层段没有任何有效切片时变体失败。

#### 共享低通

两种基线模型都先对每口井的输入阻抗曲线进行低通处理。滤波在每个连续有限段内独立执行，空值间隙两侧的有限段不会互相影响。

时间域用 `cutoff_hz`（cycles/second）。`filter.enabled: false` 时所有其他 filter 参数必须整体删除。

每个连续有效段都需满足滤波器的最小长度要求：最小样点数为滤波器二阶节数的六倍再加三。阶数为 6 时至少需要 21 个样点；过短的有限段会导致变体失败。

#### 共享 XY ordinary kriging

两种基线模型都在真实 XY 米制坐标上进行普通克里金插值：

- 控制点和输出网格都使用真实 XY 米制坐标。
- 变差函数的范围取最近邻距离中位数和名义道间距的较大值，基台取控制值方差。
- 至少两个控制值存在但全部近似相同时失败；基台非正时失败。
- 不设置隐式各向异性、搜索半径或最大控制点数。
- 变差函数类型、块金、精确插值设置以及实际范围和基台写入方法质控表。

### 与合成基准的低频分解

真实工区低频模型由井控数据约束，并通过井间空间插值扩展到目标网格；配置可以同时请求多个基线、修饰器和变体。合成基准的低频分解使用完整合成模型上下文上的固定低通契约，先完成连续有限段滤波，再施加公开目标掩码；它不使用井控拟合、XY 克里金、变体图或框架修饰器。

### `modifiers`

当前只有 `framework` 一种修饰器。它根据配置的平面范围、层段位置和阻抗倍率，在基线模型上施加明确的空间修改。

#### 逐 class 配置

```yaml
modifiers:
  <modifier_id>:
    method: framework
    bodies_file: <framework_bodies_file>
    classes:
      <framework_class>:
        top_horizon: <top_horizon_name>
        bottom_horizon: <bottom_horizon_name>
        linear_ai_multiplier: 1.06
        edge_taper_m: 100.0
        top_taper_fraction: 0.1
        bottom_taper_fraction: 0.1
```

- `top_horizon` / `bottom_horizon` 必须是目标区间中声明的相邻层位。
- `linear_ai_multiplier` 是 AI 倍率，必须为正且不等于 1。例如 1.06 表示该 class 覆盖区域的 AI 比背景高 6%。
- `edge_taper_m` 是 polygon 边缘向内的羽化距离（米）。
- `top_taper_fraction` / `bottom_taper_fraction` 是纵向 raised-cosine taper 的占比，必须落在 (0, 0.5)。

#### Body CSV 格式

`framework_bodies.csv` 固定字段：

```text
body_id,framework_class,u_top,u_bottom,vertex_order,inline,xline
```

- 同一 `body_id` 的多个行定义一个 polygon，`framework_class`、`u_top`、`u_bottom` 必须一致。
- `u_top` / `u_bottom` 定义 body 在母层段内的相对纵向窗（0~1），需满足 `0 ≤ u_top < u_bottom ≤ 1`。
- `vertex_order` 从 0 开始连续，定义 polygon 顶点顺序。
- polygon 至少三个互异顶点，不能自交或退化。
- 所有顶点必须落在显式 survey 线网上。
- 同一平面位置允许多个纵向窗不同的 body。


### `variants`

显式列表，不会自动生成基线模型与修饰器的笛卡尔积：

```yaml
variants:
  - variant_id: trend_baseline
    baseline_id: trend_main
    modifier_ids: []
  - variant_id: trend_with_<modifier_id>
    baseline_id: trend_main
    modifier_ids: [<modifier_id>]
```

`variant_id` 必须是描述性的、可用作目录名的标识符。`M0`、`M1` 等编号式命名会被拒绝。修饰器按 `modifier_ids` 顺序依次应用。

### `comparisons`

显式 pair 列表，只比较配置中声明的变体对：

```yaml
comparisons:
  - comparison_id: trend_vs_slice
    left_variant_id: trend_baseline
    right_variant_id: proportional_slice_baseline
```

每一对要求同网格同 mask。输出线性 AI 差值、logAI 差值、百分比差、井旁指标和剖面图。comparison 不输出 winner/best 字段，也不自动把左侧解释为基准。

---

## 脚本在做什么

脚本依次完成数据加载、网格确定、基线模型构建、空间修改和结果发布。计算先写入临时目录，所有变体和比较均完成后才形成正式运行目录；任一阶段失败时本次运行不发布为成功结果。

### 第一阶段：加载

1. 读取第六步井控数据，检查时间域、单位、采样轴和目标地震体。
2. 解析基线、修饰器、变体和比较关系，检查名称唯一性及引用关系。

### 第二阶段：构建上下文

1. 加载目标层位并统一时间单位，建立层位面、采样轴和层间时间间隔约束。
2. 按输出模式确定网格：全工区模式使用源地震的完整轴，矩形窗口模式使用指定的轴范围，二维剖面模式沿折线路径生成等距道位置。

### 第三阶段：构建基线模型

按变体关系去重后构建所需的每个基线模型：

1. **提取井内低频信息。** 在每口井的连续有效段内独立执行低通滤波，保留曲线缺口。
2. **趋势基线。** 在每口井上拟合首末层位之间的纵向线性趋势，用普通克里金分别插值截距和斜率，再把参数场还原到输出网格。
3. **比例切片基线。** 在相邻层位之间建立多个相对位置切片，逐片用普通克里金估计横向场，再沿相对层位坐标插值重建曲线。没有直接井控的切片由邻近有效切片补齐；整个层段没有有效切片时该基线失败。

### 第四阶段：应用修饰器

按每个变体声明的顺序应用修饰器，每次修饰包含三步：

1. **计算空间概率。** 在平面多边形内部按距边界的距离形成渐变权重，在层段内的相对时间窗中按上下边界的余弦渐变形成纵向权重。两者相乘得到样点概率。
2. **合并同类区域。** 同一类别包含多个空间区域时，逐样点取概率最大值，形成该类别的概率场。
3. **叠加阻抗增量。** 将类别概率与阻抗倍率的自然对数相乘，叠加到父模型；不同类别的增量相加。输出网格和有效掩码保持不变。

每个类别的贡献为：

```
修饰后的对数声阻抗 = 父模型对数声阻抗 + 类别概率 × ln(阻抗倍率)
```

倍率必须为正且不能等于 1。概率为 1 的区域获得完整对数增量，渐变区域按概率缩放，概率为 0 的区域不改变基线。

### 第五阶段：原子发布

1. 为每个变体写出主模型、方法字段、修饰字段、质控表和图件。
2. 为每个比较关系写出总体差异、井旁差异和对比图。
3. 写出变体清单和运行摘要。
4. 全工区模式额外导出线性 AI 的 SEG-Y 或 ZGY 体。
5. 全部成功后将临时目录改名为正式输出目录。

---

## 核心输出文件

```text
real_field_lfm_<run_timestamp>/
├── lfm_run_summary.json
├── variant_manifest.csv
├── comparisons/
│   └── <comparison_id>/
│       ├── metrics.csv
│       ├── well_metrics.csv
│       └── figures/
│           └── overview.png
└── variants/
    └── <variant_id>/
        ├── <variant_id>_linear_ai.<segy_or_zgy>  # 仅 volume 模式
        ├── lfm.npz
        ├── method_fields.npz
        ├── modifier_fields.npz        # 仅含 modifier 时
        ├── variant_summary.json
        └── qc/
            ├── *.csv
            └── figures/
```

### `lfm_run_summary.json`

数据模式固定为 `real_field_lfm_run_v3`。记录业务配置、井控集来源、输出几何、请求的变体和比较关系，以及产物路径。

### `variant_manifest.csv`

一行一个 variant，关键列：

| 列 | 含义 |
|------|------|
| `variant_id` | 描述性标识符 |
| `baseline_id` / `baseline_method` | 来源 baseline |
| `modifier_chain` | 分号分隔的 modifier ID 列表 |
| `lfm_path` | 主 NPZ 路径 |
| `method_fields_path` | 方法 sidecar 路径 |
| `contract_fingerprint_sha256` | 当前变体的发布标识 |

### `variants/<variant_id>/lfm.npz`

主低频模型体，跨方法同构。只包含：

| 键 | dtype | 含义 |
|------|------|------|
| `log_ai` | float32 | ln(AI)，mask 内有限、mask 外 NaN |
| `valid_mask_model` | bool | 权威掩码 |
| `ilines` / `xlines` / `samples` | float64 | 输出轴 |
| `metadata_json` | 标量字符串 | 完整 variant metadata |

a/b、kriging variance、framework probability 等方法专属字段只在 sidecar NPZ 中。

### `variants/<variant_id>/variant_summary.json`

完整 metadata：变体身份、基线模型/修饰器链、业务配置、直接上游契约、产物路径、体统计量（valid 样点数、波阻抗对数范围）和当前变体唯一契约指纹。

### QC 表格

| 文件 | 内容 |
|------|------|
| `trend_well_fit.csv` | 每井 a/b 拟合参数、残差 RMS、有效样点数 |
| `trend_parameter_model.csv` | 每个参数的 kriging mode、sill、range |
| `proportional_slice_qc.csv` | 每个切片的原始 mode、最终 mode、控制井、上下来源切片 |
| `framework_body_qc.csv` | 每个 body 的顶底、面积、有效 trace 数、概率统计 |
| `framework_class_qc.csv` | 每类的 multiplier、概率统计、修改 sample 数 |
| `well_framework_effect_qc.csv` | 每井在 modifier 作用下的 logAI 偏移量 |

### 图表

| 文件 | 内容 |
|------|------|
| `lfm_representative_section.png` | 代表性剖面的 log(AI) 和 linear AI，标注层位 |
| `framework_map_and_sections.png` | 每个 framework class 的 polygon map 和概率剖面 |
| `comparisons/<id>/figures/overview.png` | 七面板：两个 variant 的 logAI/AI、差值、百分比差 |

---

## 如何阅读结果

### 第一步：看终端输出

```
=== Unified Real-field LFM v3 ===
Output: scripts/output/real_field_lfm_<run_timestamp>
Variants: <variant_count>
Status: ok
```

确认变体数与配置一致、status 为 `ok`。

### 第二步：看 `lfm_run_summary.json`

确认井控来源、输出网格、基线方法、变体和比较清单与预期一致，再根据记录的产物路径查看具体结果。

### 第三步：看 variant QC 图

打开 `variants/<variant_id>/qc/figures/lfm_representative_section.png`：

- 左图的波阻抗对数应该在 8~10 左右，层位之间整体趋势合理。
- 右图的线性 AI 确认数值在地质合理范围内。
- 两种基线的剖面差异反映其纵向参数化和切片插值结果的差异。

### 第四步：看逐井拟合 QC

**Trend：** 打开 `trend_well_fit.csv`。关注每口井的 a 和 b 是否与邻井一致。如果某口井的 b 与周边井符号相反，说明该井的趋势斜率异常——可能是井曲线质量问题或该井穿过了特殊地质体。

**Proportional kriging：** 打开 `proportional_slice_qc.csv`。按 `original_mode` 分组：

- `kriging` 是最理想的情况——该切片有足够多井控制直接插值。
- `neighbor_slice_fill` 说明该切片没有直接控制值，使用相邻有效切片补齐。
- `single_control_constant` 表示只有一口井控制该切片，横向变化为零。

### 第五步：看框架修饰效果

如果使用了 framework 修饰器，打开 `well_framework_effect_qc.csv`：

- `delta_log_ai_mean` 的正负和大小反映了修饰器在各井位置的平均影响。
- `*_probability_mean/max` 列显示每口井落入修饰器概率场的程度。概率接近零的井完全不受修饰器影响——这是预期行为；如果本来想让某口井受修饰器影响却概率为零，检查 body polygon 是否覆盖了该井位置。

### 第六步：看 comparison

打开 `comparisons/<id>/metrics.csv`：

- `mean_delta_log_ai` 和 `mean_percent_difference` 给出两个变体的全局差异量级。
- 打开 `figures/overview.png` 的右三面板（差值图），观察差异的空间分布。差异集中在某些特定层段是合理的；如果整个体都有系统性偏差，说明两个基线模型或修饰器的差异超出了随机波动。

---

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| WellControlSet schema 不是 v7 | 第六步未完成或产物不符合当前井控契约 | 重建第六步 |
| WellControlSet 几何/采样轴不一致 | 第六步使用了不同的地震几何或采样轴 | 用当前地震重建第六步 |
| 配置段缺少必要字段 | baseline/modifier/variant 配置不完整 | 对照配置参考补全 |
| variant ID 使用 `M0`/`M1` | 禁止编号式命名 | 使用描述性 ID |
| baseline ID 或 modifier ID 重复 | ID 必须全局唯一 | 重命名冲突的 ID |
| filter cutoff 无效 | `cutoff_hz` 不是正值或不低于时间采样轴的 Nyquist 频率 | 调整截止频率，使其低于当前时间采样轴的 Nyquist 频率 |
| 有限井段短于滤波最小长度 | 某口井的连续有效段过短 | 检查上游曲线缺口，或降低滤波阶数 |
| 两口以上控制值退化 | 所有井的参数几乎相同，kriging sill 非正 | 检查井数据是否存在系统性偏差 |
| 整层段无有效切片 | proportional_kriging 在某层段没有任何井控制 | 检查层位解释范围是否覆盖井位 |
| framework polygon 自交或顶点不在 survey 线上 | body CSV 顶点坐标有误 | 在解释软件中修正 polygon |
| body 在离散网格上完全不可见 | polygon 太小或位于输出几何之外 | 检查 body 位置或调整输出几何范围 |
| 仅部分 variant 成功 | 原子发布规则：一次 run 全部成功才算成功 | 检查失败 variant 的具体错误 |
