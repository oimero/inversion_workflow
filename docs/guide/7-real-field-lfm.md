# 07 构建真实工区低频模型

`real_field_lfm.py` 是工作流的第七步。它按配置中显式声明的基线模型、修饰器和变体图，一次性构建所有请求的低频模型变体。

---

## 快速开始

```powershell
python scripts/real_field_lfm.py
python scripts/real_field_lfm.py --config experiments/my_project.yaml
python scripts/real_field_lfm.py --output-dir scripts/output/lfm_test
```

运行前确保第六步已完成且产物可消费。脚本会自动发现最新的第六步运行，或通过配置中的 `well_control_run_dir` 显式指定。已有输出目录会被拒绝。

---

## 运行前需要什么

| 来源 | 文件 | 用途 |
|------|------|------|
| 第六步 | `run_summary.json` | WellControlSet schema/domain 校验和直接契约身份 |
| 第六步 | `well_control_manifest.csv` | 成功井清单和逐井 NPZ 路径 |
| 第六步 | `wells/<well_name>.npz` | 每井 canonical log(AI) 和逐样点位置 |
| 数据目录 | 地震体 | 目标 SampleAxis、survey geometry 和 inline/xline 线网 |
| 数据目录 | 层位文件 | Petrel 导出的解释层位，定义 TargetZone 的几何边界 |
| 可选 | framework body CSV | 仅使用 framework modifier 时需要 |

---

## 配置参考

```yaml
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
        cutoff_hz: 15.0
        order: 6
        buffer_mode: reflect
        buffer_axis_units: 0.4
      fit:
        min_valid_samples_per_well: 32
        huber_f_scale_log_ai: 0.05
      spatial: {variogram: spherical, exact: true, nugget: 0.0}

    proportional_slice_main:
      method: proportional_kriging
      filter:
        enabled: true
        cutoff_hz: 15.0
        order: 6
        buffer_mode: reflect
        buffer_axis_units: 0.4
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

顶层配置的 `target_interval.horizons` 必须声明至少两个层位的名称和文件路径。首个层位和末尾层位定义趋势窗的顶和底；中间的层位只参与 TargetZone 几何定义和连续性校验，不增加趋势参数。层位文件为 Petrel 导出格式，通过 `import_interpretation_petrel` 读取，内部做绝对值处理，路径相对于 `data_root`。

### `source_runs`

指向第六步产出的井控集运行目录。留空时自动发现最新的 `real_field_well_controls_*`。第七步加载井控集时校验数据模式、domain/unit、采样轴、坐标和 shape 等显式语义，并把第六步发布的契约指纹记为直接输入。消费者不重算地震体或逐井 NPZ 的 SHA；若地震体等业务输入发生变化，必须重建第六步，形成新的不可变运行。

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

从全工区中裁出一个矩形窗口。六个参数缺一不可，端点必须精确落在 survey 线网上。

```yaml
output_geometry:
  mode: window
  inline_min: 1000
  inline_max: 1200
  xline_min: 3000
  xline_max: 3400
  sample_min: 1.0
  sample_max: 2.5
```

| 参数 | 含义 |
|------|------|
| `inline_min` / `inline_max` | inline 起止线号（含） |
| `xline_min` / `xline_max` | xline 起止线号（含） |
| `sample_min` / `sample_max` | 深度或时间采样轴起止（含） |

下标轴只接受真实线号。例如 xline 步长为 4 时 `5003`、`5007`、`5011` 是不同有效线号，不能当成数组下标。

#### `section` — 二维剖面

由折线路径定义，沿路径按 XY 米制距离均匀采样 `n_traces` 个道位置。

```yaml
output_geometry:
  mode: section
  points:
    - {inline: 1100.0, xline: 3000.0}
    - {inline: 1100.0, xline: 3200.0}
  n_traces: 101
  sample_min: 1.0
  sample_max: 2.5
```

| 参数 | 含义 |
|------|------|
| `points` | 至少两个 `{inline, xline}` 端点，定义折线路径 |
| `n_traces` | 沿路径均匀采样的道数，至少为 2 |
| `sample_min` / `sample_max` | 深度或时间采样轴起止（含） |

`n_traces` 不需要和 survey 线网对齐——函数在 XY 米制空间均匀采样后反算浮点 inline/xline。写成不同数值只是改变采样密度，都不会报错。

### `baselines`

两种基线模型方法各解决不同的偏差-方差权衡。

#### trend

高偏差、低方差。逻辑是"每口井的背景趋势可以用两个数描述"：

1. 对每口有效控制井，在首层位到底层位的目标窗内计算归一化相对位置 `u = (sample - top) / (bottom - top)`，然后拟合 `logAI = a + b·(2u - 1)`。使用 Huber 回归，`f_scale` 由 `huber_f_scale_log_ai` 控制。
2. 在真实 XY 米制坐标上，分别对 `a` 和 `b` 做 ordinary kriging，得到 `a_field` 和 `b_field`。
3. 逐道用 `logAI = a_field + b_field·(2u - 1)` 重建体。

控制语义：

| 有效井数 | 行为 |
|---:|---|
| 0 | variant 失败 |
| 1 | 生成 `single_control_constant` 场，方差为零 |
| ≥2 且参数值退化 | variant 失败 |
| ≥2 且正常 | XY ordinary kriging |

#### proportional_kriging

降低纵向结构偏差，但更依赖井分布和变差函数：

1. 对每对相邻层位定义的层段，按 `n_slices` 个等距比例地层切片（`u = k/(n_slices-1)`）在每口井上采样滤波后的波阻抗对数。
2. 每个切片独立做 XY ordinary kriging，得到该切片的横向场。
3. 逐道在切片之间线性插值，重建层段内所有样点的值。
4. 跨层段拼接成完整体。

completion 行为（切片控制不足时）：

| 有限控制数 | 行为 | mode |
|---:|------|------|
| 0 | 同层段存在其他有效切片时，由最近上下切片线性补齐 | `neighbor_slice_fill` |
| 1 | 全平面使用该控制值 | `single_control_constant` |
| ≥2 且非退化 | XY ordinary kriging | `kriging` |
| ≥2 且退化 | 失败 | 无产物 |

补齐不跨越层段边界。整个层段没有任何有效切片时变体失败。

#### 共享低通

两种基线模型各自配置 filter 参数，但底层调用同一个 `apply_lfm_lowpass` 实现。滤波永远在每口井上逐连续有限段独立执行——空值间隙两侧的有限段不会互相污染。

时间域用 `cutoff_hz`（cycles/second），深度域用 `cutoff_wavelength_m`（metres/cycle）。错域 cutoff 参数直接失败。`filter.enabled: false` 时所有其他 filter 参数必须整体删除。

每个连续有限段至少需要 `3·(2·order + 1)` 个样点才能执行指定阶数的 Butterworth 滤波。过短的有限段会导致变体失败。

#### 共享 XY ordinary kriging

两种基线模型的空间插值调用同一个 `ordinary_krige_xy` 原语：

- 控制点和输出网格都使用真实 XY 米制坐标。
- 变差函数固定使用 `range = max(median(nearest_neighbor_distance), nominal_bin_spacing)` 和 `sill = control_values_variance`。
- 至少两个控制值存在但全部近似相同时失败；sill 非正时失败。
- 不设置隐式 anisotropy、搜索半径或最大控制点数。
- variogram、nugget、exact 和实际 range/sill 全部写入 QC。

### `modifiers`

当前只有 `framework` 一种修饰器。framework 是显式注入地质场景假设的修饰器。框架正确时可能改善初始模型，框架错误时会主动注入系统误差。

#### 逐 class 配置

```yaml
modifiers:
  reef_scenario:
    method: framework
    bodies_file: experiments/real_field_lfm/framework_bodies.csv
    classes:
      reef:
        top_horizon: horizon_a
        bottom_horizon: horizon_b
        linear_ai_multiplier: 1.06
        edge_taper_m: 100.0
        top_taper_fraction: 0.1
        bottom_taper_fraction: 0.1
```

- `top_horizon` / `bottom_horizon` 必须是 TargetZone 中声明的相邻层位。
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

显式列表，不会自动生成基线模型 × 修饰器的笛卡尔积：

```yaml
variants:
  - variant_id: trend_baseline
    baseline_id: trend_main
    modifier_ids: []
  - variant_id: trend_with_reef
    baseline_id: trend_main
    modifier_ids: [reef_scenario]
```

`variant_id` 必须是描述性的、可用作目录名的标识符。`M0`、`M1` 等编号式命名会被拒绝。修饰器按 `modifier_ids` 顺序依次应用。

### `comparisons`

显式 pair 列表，只比较配置中声明的变体对：

```yaml
comparisons:
  - comparison_id: trend_vs_slice
    left_variant_id: trend_baseline
    right_variant_id: slice_baseline
```

每一对要求同网格同 mask。输出线性 AI 差值、logAI 差值、百分比差、井旁指标和剖面图。comparison 不输出 winner/best 字段，也不自动把左侧解释为基准。

---

## 脚本在做什么

脚本分五个阶段：**加载 → 构建上下文 → 构建基线模型 → 应用修饰器 → 原子发布**。所有阶段在临时目录中执行，任一失败整次运行失败，临时目录保留诊断信息。

### 第一阶段：加载

1. 加载第六步井控集，校验数据模式、契约身份和采样轴与当前地震体一致。
2. 解析配置，校验所有基线模型、修饰器、变体和 comparison 的 ID 全局唯一且引用有效。

### 第二阶段：构建上下文

1. 加载并归一化目标层位，构建 TargetZone（层位面 + 采样轴 + 最小厚度约束）。
2. 按输出模式确定网格：`volume` 用全 survey 轴，`window` 用指定的 inline/xline/sample 子范围，`section` 由折线路径插值生成道网格。

### 第三阶段：构建基线模型

按变体图中引用的基线模型去重后依次构建。两种方法共享低通滤波和 XY kriging 原语。

**Trend：** 高偏差、低方差。对每口控制井在目标窗内拟合一条 logAI = a + b·(2u-1) 的直线，然后在 XY 平面上分别对 a 和 b 做 ordinary kriging，逐道按参数场重建体。单井控制时直接生成常数场。

**Proportional kriging：** 降低纵向结构偏差。把目标区间按层段切分为等距比例切片，每口井在每个切片位置采样滤波后的波阻抗对数，逐切片独立做 XY kriging，再在切片间线性插值重建体。控制不足的切片由相邻切片补齐，整层段无控制时失败。

### 第四阶段：应用修饰器

对每个变体声明的修饰器列表按序应用。当前只有 framework 修饰器，它的逻辑是在基线模型上叠加地质场景假设——比如"这里有一块礁体，AI 比背景高 6%"。

修饰过程分三步：

**第一步：计算每个 body 的三维概率场。**

一个 body 由一个 polygon（平面范围）加一个纵向窗（层段内的相对深度区间）定义。对体中的每个样点：

- 横向：判断该道是否在 polygon 内。polygon 边缘有羽化过渡带（`edge_taper_m`），从边界向内概率从 0 平滑上升到 1，避免多边形边缘出现硬边界。
- 纵向：判断该样点是否落在 body 的纵向窗内。窗的顶和底各有一段 raised-cosine 渐变带（各占纵向窗的 `top_taper_fraction` / `bottom_taper_fraction`），从窗边界向内概率从 0 平滑上升到 1。窗的中部概率为 1。

三维概率 = 横向概率 × 纵向概率。polygon 外或纵向窗外概率为零，完全落在内部且不在任何渐变带的样点概率为 1。

**第二步：同 class 内取最大值。**

如果同一个 class 下有多个 body，每个样点取所有 body 概率的最大值。这样可以在不合并 polygon 的前提下处理复杂形状——多个小 polygon 拼出一个不规则地质体。不同 class 之间完全独立，后续线性叠加。

**第三步：叠加到基线模型。**

在波阻抗对数域上，每个 class 独立贡献一项：

```
logAI_modified = logAI_parent + P_class × ln(multiplier)
```

`multiplier` 是 AI 倍率。例如 `multiplier=1.06` 表示该 class 覆盖区域 AI 比背景高 6%，贡献量 `ln(1.06) ≈ 0.058`。`P_class` 接近 1 的区域获得完整倍率效果，渐变带内按概率比例减弱，完全在外的区域不受影响。

多个 class 各自独立计算、直接累加。修饰后的体保持与基线相同的网格和有效掩码。

### 第五阶段：原子发布

1. 逐变体写出主 NPZ、方法 sidecar、修饰器 sidecar 和变体摘要。
2. 逐 comparison 写出差异统计表和七面板对比图。
3. 写入 `variant_manifest.csv` 和 `lfm_run_summary.json`。
4. volume 模式额外导出线性 AI 的 SEG-Y 或 ZGY 体。
5. 全部成功后原子 rename 临时目录为最终目录。

---

## 核心输出文件

```text
real_field_lfm_<timestamp>/
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
        ├── lfm.npz
        ├── method_fields.npz
        ├── modifier_fields.npz        # 仅含 modifier 时
        ├── variant_summary.json
        └── qc/
            ├── <variant>_linear_ai.zgy  # 仅 volume 模式
            ├── *.csv
            └── figures/
```

### `lfm_run_summary.json`

数据模式固定为 `real_field_lfm_run_v3`。记录业务配置、井控集直接上游契约、输出几何、请求的变体/comparison ID 列表和产物路径，并发布一个运行级契约指纹。

### `variant_manifest.csv`

一行一个 variant，关键列：

| 列 | 含义 |
|------|------|
| `variant_id` | 描述性标识符 |
| `baseline_id` / `baseline_method` | 来源 baseline |
| `modifier_chain` | 分号分隔的 modifier ID 列表 |
| `lfm_path` | 主 NPZ 路径 |
| `method_fields_path` | 方法 sidecar 路径 |
| `contract_fingerprint_sha256` | 当前 variant 唯一契约指纹 |

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
Output: scripts/output/real_field_lfm_<timestamp>
Variants: 2
Status: ok
```

确认变体数与配置一致、status 为 `ok`。

### 第二步：看 `lfm_run_summary.json`

关注契约指纹和输入契约：前者是本次不可变运行的唯一身份，后者只列直接上游。配置、直接上游或主产物变化都会形成新指纹。

### 第三步：看 variant QC 图

打开 `variants/<variant_id>/qc/figures/lfm_representative_section.png`：

- 左图的波阻抗对数应该在 8~10 左右，层位之间整体趋势合理。
- 右图的线性 AI 确认数值在地质合理范围内。
- 如果 trend 和 proportional_kriging 的剖面看起来很相似，说明井控充分、纵向趋势简单。如果差异明显，说明 proportional_kriging 捕捉到了 trend 的线性假设无法描述的结构。

### 第四步：看逐井拟合 QC

**Trend：** 打开 `trend_well_fit.csv`。关注每口井的 a 和 b 是否与邻井一致。如果某口井的 b 与周边井符号相反，说明该井的趋势斜率异常——可能是井曲线质量问题或该井穿过了特殊地质体。

**Proportional kriging：** 打开 `proportional_slice_qc.csv`。按 `original_mode` 分组：

- `kriging` 是最理想的情况——该切片有足够多井控制直接插值。
- `neighbor_slice_fill` 说明该切片控制不足，由相邻切片补齐。如果大量切片都是补齐的，考虑增加 `n_slices` 或在配置中为特定层段提供更多控制。
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
| WellControlSet schema 不是 v2 | 第六步未完成或用旧版脚本生成 | 重建第六步 |
| WellControlSet 几何/采样轴不一致 | 第六步使用了不同的地震契约 | 用当前地震重建第六步；不要靠文件哈希替代轴语义校验 |
| 配置段缺少必要字段 | baseline/modifier/variant 配置不完整 | 对照配置参考补全 |
| variant ID 使用 `M0`/`M1` | 禁止编号式命名 | 使用描述性 ID |
| baseline ID 或 modifier ID 重复 | ID 必须全局唯一 | 重命名冲突的 ID |
| filter cutoff 错域 | 时间域用了 `cutoff_wavelength_m` 或反过来 | 时间域用 `cutoff_hz`，深度域用 `cutoff_wavelength_m` |
| 有限井段短于滤波最小长度 | 某口井的连续有效段过短 | 放宽有效样点要求或减小 filter order |
| 两口以上控制值退化 | 所有井的参数几乎相同，kriging sill 非正 | 检查井数据是否存在系统性偏差 |
| 整层段无有效切片 | proportional_kriging 在某层段没有任何井控制 | 检查层位解释范围是否覆盖井位 |
| framework polygon 自交或顶点不在 survey 线上 | body CSV 顶点坐标有误 | 在解释软件中修正 polygon |
| body 在离散网格上完全不可见 | polygon 太小或位于输出几何之外 | 检查 body 位置或调整输出几何范围 |
| 仅部分 variant 成功 | 原子发布规则：一次 run 全部成功才算成功 | 检查失败 variant 的具体错误 |

---

## 留到第二轮

- 是否支持更多基线模型方法（如外部导入的低频模型体直接作为变体）。
- 是否支持更多修饰器类型（如断层扰动、流体替代场景）。
- 是否支持按井型或区块分组建模（同一运行内的不同 well group）。
- 是否在 comparison 中增加沿井轨迹的逐样点对比图。
- 是否支持为合成训练自动生成低频模型变体的组合策略。
- trend 基线模型是否支持多层位分段拟合（当前只用首末层位）。
