# 07 实际工区 LFM 变体构建

`real_field_lfm.py` 是工作流的第七步。它从第六步冻结的 WellControlSet 出发，按配置中显式声明的 baseline、modifier 和 variant 图，一次性构建所有请求的低频模型变体。

这是统一 LFM v2 架构的核心：**同一套 builder 在时间域和深度域之间共享，framework 是可选 modifier 而非第三种 baseline。** 时间域与深度域的差异只体现在 cutoff 单位、采样轴单位和 source adapter——builder 和 modifier 内部逻辑完全相同。

---

## 快速开始

```powershell
python scripts/real_field_lfm.py
python scripts/real_field_lfm.py --config experiments/my_project.yaml
python scripts/real_field_lfm.py --output-dir scripts/output/lfm_test
```

运行前确保第六步已完成且产物可消费。脚本会自动发现最新的第六步 run，或通过配置中的 `well_control_run_dir` 显式指定。已有输出目录会被拒绝。

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

### 层位要求

`target_interval.horizons` 必须声明至少两个层位的名称和文件路径。首个层位和末尾层位定义趋势窗的顶和底；中间的层位只参与 TargetZone 几何定义和连续性校验，不增加趋势参数。

层位文件为 Petrel 导出格式，通过 `import_interpretation_petrel` 读取，内部做绝对值处理。路径相对于 `data_root`。

### 第六步契约身份

Step 7 加载 WellControlSet 时校验 schema、domain/unit、采样轴、坐标和 shape 等显式语义，并把第六步发布的 `contract_fingerprint_sha256` 记为直接输入。消费者不重算地震体或逐井 NPZ 的 SHA；若地震体等业务输入发生变化，必须重建 Step 6，形成新的不可变 run。

---

## 配置参考

Step 7 的配置是整份规范中最丰富的一段。它分为五个子段：`source_runs`、`output_geometry`、`baselines`、`modifiers`、`variants`、`comparisons`。

```yaml
real_field_lfm:
  source_runs:
    well_control_run_dir: scripts/output/real_field_well_controls_<timestamp>

  output_geometry:
    mode: window
    inline_min: 1600
    inline_max: 1800
    xline_min: 5003
    xline_max: 6203
    sample_min: 1.0
    sample_max: 2.5

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

    slice_main:
      method: proportional_kriging
      filter:
        enabled: true
        cutoff_hz: 15.0
        order: 6
        buffer_mode: reflect
        buffer_axis_units: 0.4
      n_slices: 32
      spatial: {variogram: spherical, exact: true, nugget: 0.0}

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

  variants:
    - variant_id: trend_baseline
      baseline_id: trend_main
      modifier_ids: []
    - variant_id: trend_with_reef
      baseline_id: trend_main
      modifier_ids: [reef_scenario]
    - variant_id: slice_baseline
      baseline_id: slice_main
      modifier_ids: []

  comparisons:
    - comparison_id: trend_vs_slice
      left_variant_id: trend_baseline
      right_variant_id: slice_baseline
    - comparison_id: framework_effect
      left_variant_id: trend_baseline
      right_variant_id: trend_with_reef
```

### `output_geometry`

三种模式：

| mode | 含义 | 允许导出体 |
|------|------|-----------|
| `volume` | 全工区规则体，轴与源地震严格一致 | 是 |
| `window` | 矩形子体，需显式指定 inline/xline/sample 起止 | 否 |
| `section` | 二维剖面，由折线路径 + 采样点数定义 | 否 |

`window` 和 `section` 只裁输出，不减少 baseline 可使用的控制井集合——所有第六步成功井都参与建模，裁的是最终写出的体范围。

`section` 需配置 `points`（至少两个 `{inline, xline}` 端点）和 `n_traces`（至少 2）。剖面端点间的折线段数自动检测，不要求每条线段长度相等。

inline/xline 始终是真实线号。例如 xline 步长为 4 时，`5003`、`5007`、`5011` 是不同有效线号，不能把它们当成数组下标或步长 1。

### `baselines`

两种 baseline 方法各解决不同的偏差-方差权衡。

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

1. 对每对相邻层位定义的层段，按 `n_slices` 个等距比例地层切片（`u = k/(n_slices-1)`）在每口井上采样滤波后的 log(AI)。
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

补齐不跨越层段边界。整个层段没有任何有效切片时 variant 失败。

#### 共享低通

两种 baseline 各自配置 filter 参数，但底层调用同一个 `apply_lfm_lowpass` 实现。滤波永远在每口井上逐连续有限段独立执行——NaN gap 两侧的有限段不会互相污染。

时间域用 `cutoff_hz`（cycles/second），深度域用 `cutoff_wavelength_m`（metres/cycle）。错域 cutoff 参数直接失败。`filter.enabled: false` 时所有其他 filter 参数必须整体删除。

每个连续有限段至少需要 `3·(2·order + 1)` 个样点才能执行指定阶数的 Butterworth 滤波。过短的有限段会导致 variant 失败。

#### 共享 XY ordinary kriging

两种 baseline 的空间插值调用同一个 `ordinary_krige_xy` 原语：

- 控制点和输出网格都使用真实 XY 米制坐标。
- 变差函数固定使用 `range = max(median(nearest_neighbor_distance), nominal_bin_spacing)` 和 `sill = control_values_variance`。
- 至少两个控制值存在但全部近似相同时失败；sill 非正时失败。
- 不设置隐式 anisotropy、搜索半径或最大控制点数。
- variogram、nugget、exact 和实际 range/sill 全部写入 QC。

### `modifiers`

当前只有 `framework` 一种 modifier。framework 是显式注入地质场景假设的修饰器。框架正确时可能改善初始模型，框架错误时会主动注入系统误差。

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

#### 修饰公式

对每个样点，先独立计算每个 body 的三维概率 `P_body = P_map(x,y) × P_vertical(v_body)`，再对同一 class 内的所有 body 取最大值 `P_class = max(P_body)`。最后在 log(AI) 域叠加：

```text
logAI_modified = logAI_parent + Σ(P_class × ln(multiplier_class))
```

先形成三维 body 再取最大值——不先合并二维 polygon。不同 class 的 body 独立计算、线性叠加。mask 外保持 NaN。

### `variants`

显式列表，不会自动生成 baseline × modifier 的笛卡尔积：

```yaml
variants:
  - variant_id: trend_baseline
    baseline_id: trend_main
    modifier_ids: []
  - variant_id: trend_with_reef
    baseline_id: trend_main
    modifier_ids: [reef_scenario]
```

`variant_id` 必须是描述性的、可用作目录名的标识符。`M0`、`M1` 等编号式命名会被拒绝。modifier 按 `modifier_ids` 顺序依次应用。

### `comparisons`

显式 pair 列表，只比较配置中声明的 variant 对：

```yaml
comparisons:
  - comparison_id: trend_vs_slice
    left_variant_id: trend_baseline
    right_variant_id: slice_baseline
```

每一对要求同网格同 mask。输出线性 AI 差值、logAI 差值、百分比差、井旁指标和剖面图。comparison 不输出 winner/best 字段，也不自动把左侧解释为基准。

---

## 脚本在做什么

脚本分五个阶段：**加载与校验 → 构建 context → 构建 baseline → 应用 modifier → 原子发布**。所有阶段在临时目录中执行，全部成功后才发布到最终目录。

### 第一阶段：加载与校验

1. 加载第六步 WellControlSet，校验 `run_summary.json` schema、直接契约身份，以及 manifest/逐井 NPZ 的结构、轴、dtype、shape 和 mask 语义。
2. 校验 WellControlSet 与当前地震的 domain、SampleAxis 和 survey geometry 一致；不重算地震文件哈希。
3. 校验 WellControlSet 的 SampleAxis 与当前地震体的 SampleAxis 逐点一致。
4. 解析 `real_field_lfm` 配置段，校验各 ID 全局唯一、引用存在、无自动组合。

### 第二阶段：构建 LfmContext

1. 加载并归一化所有 target horizon。
2. 构建 TargetZone（层位面 + 采样轴 + 最小厚度约束）。
3. 按 `output_geometry.mode` 解析输出网格：
   - `volume`：全 survey 轴，XY 网格由 survey geometry 提供。
   - `window`：子轴必须落在显式 survey 线网上，端点必须精确匹配已有坐标。
   - `section`：由折线端点 + 道数插值生成 inline/xline 轴，XY 由 survey geometry 换算。

### 第三阶段：构建 baseline

按 variant 图中引用的 baseline ID 去重后依次构建：

1. **读取 baseline 配置**，校验 method、filter、fit（仅 trend）、n_slices（仅 proportional）和 spatial 字段。
2. **对每口控制井应用低通滤波**，输出仍是同 basis、同单位的 `grid.Log`。滤波参数来自 baseline 自己的 filter 配置。
3. **构建 baseline 体：**
   - trend：每井拟合 a/b → XY kriging 插值参数场 → 逐道重建 → 用 target mask 写值。
   - proportional_kriging：逐层段切片采样 → 逐切片 XY kriging → neighbor_slice_fill 补齐空切片 → 逐道线性插值 → 跨层段拼接。
4. 生成 method sidecar（`a_field`、`b_field`、`kriging_variance`、`distance_to_control_m`、`slice_u` 等）和 QC 表格。

### 第四阶段：应用 modifier

对每个 variant 声明的 modifier 列表，按顺序应用：

1. **framework modifier：** 加载 body CSV，校验 polygon 拓扑和 class 参数 → 逐 class 计算三维概率 → 在 parent 的 logAI 上累加 `P_class × ln(multiplier)` → 生成 class probability sidecar 和 body/class QC。
2. 修饰后的体保持 parent 的 `valid_mask_model`、axes 和 shape 不变。
3. 额外生成 `well_framework_effect_qc.csv`：逐井统计 modifier 在各井位置上的 logAI 偏移量和 class 概率。

### 第五阶段：原子发布

1. **逐 variant 写入：**
   - `lfm.npz`：只包含 `log_ai`（float32）、`valid_mask_model`（bool）、`ilines`/`xlines`/`samples`（float64）、`metadata_json`。
   - `method_fields.npz`：a/b field、kriging variance 等。
   - `modifier_fields.npz`：class probability、class map QC 等（仅含 modifier 时）。
   - `variant_summary.json`：完整 metadata、统计量、产物路径和当前 variant 唯一契约指纹。
2. **逐 comparison 写入：** `metrics.csv`（全局差异统计）、`well_metrics.csv`（每井差异）、`figures/overview.png`（七面板对比图）。
3. **volume 模式导出体：** 对每个 variant 输出线性 AI 的 SEG-Y 或 ZGY，mask 外保留 NaN。
4. **写入 `variant_manifest.csv` 和 `lfm_run_summary.json`，然后原子 rename 临时目录为最终目录。**

任一 variant 或 comparison 失败，整次 run 失败——临时目录被重命名为 `_failed_<uuid>` 保留诊断信息，不发布可被 R0 自动发现的 manifest。

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

Schema 固定为 `real_field_lfm_run_v3`。记录业务配置、WellControlSet 直接上游契约、输出几何、请求的 variant/comparison ID 列表和产物路径，并发布一个 run 级契约指纹。

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

主 LFM 体，跨方法同构。只包含：

| 键 | dtype | 含义 |
|------|------|------|
| `log_ai` | float32 | ln(AI)，mask 内有限、mask 外 NaN |
| `valid_mask_model` | bool | 权威掩码 |
| `ilines` / `xlines` / `samples` | float64 | 输出轴 |
| `metadata_json` | 标量字符串 | 完整 variant metadata |

a/b、kriging variance、framework probability 等方法专属字段只在 sidecar NPZ 中。

### `variants/<variant_id>/variant_summary.json`

完整 metadata：variant 身份、baseline/modifier 链、业务配置、直接上游契约、产物路径、体统计量（valid 样点数、logAI 范围）和当前 variant 唯一契约指纹。

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
Variants: 3
Status: ok
```

确认 variant 数与配置一致、status 为 `ok`。

### 第二步：看 `lfm_run_summary.json`

关注 `contract_fingerprint_sha256` 和 `input_contracts`：前者是本次不可变 run 的唯一身份，后者只列直接上游。配置、直接上游或主产物变化都会形成新指纹。

### 第三步：看 variant QC 图

打开 `variants/<variant_id>/qc/figures/lfm_representative_section.png`：

- 左图的 log(AI) 应该在 8~10 左右，层位之间整体趋势合理。
- 右图的线性 AI 确认数值在地质合理范围内。
- 如果 trend 和 proportional_kriging 的剖面看起来很相似，说明井控充分、纵向趋势简单。如果差异明显，说明 proportional_kriging 捕捉到了 trend 的线性假设无法描述的结构。

### 第四步：看逐井拟合 QC

**Trend：** 打开 `trend_well_fit.csv`。关注每口井的 a 和 b 是否与邻井一致。如果某口井的 b 与周边井符号相反，说明该井的趋势斜率异常——可能是井曲线质量问题或该井穿过了特殊地质体。

**Proportional kriging：** 打开 `proportional_slice_qc.csv`。按 `original_mode` 分组：

- `kriging` 是最理想的情况——该切片有足够多井控制直接插值。
- `neighbor_slice_fill` 说明该切片控制不足，由相邻切片补齐。如果大量切片都是补齐的，考虑增加 `n_slices` 或在配置中为特定层段提供更多控制。
- `single_control_constant` 表示只有一口井控制该切片，横向变化为零。

### 第五步：看 framework effect

如果使用了 framework modifier，打开 `well_framework_effect_qc.csv`：

- `delta_log_ai_mean` 的正负和大小反映了 modifier 在各井位置的平均影响。
- `*_probability_mean/max` 列显示每口井落入 modifier 概率场的程度。概率接近零的井完全不受 modifier 影响——这是预期行为；如果本来想让某口井受 modifier 影响却概率为零，检查 body polygon 是否覆盖了该井位置。

### 第六步：看 comparison

打开 `comparisons/<id>/metrics.csv`：

- `mean_delta_log_ai` 和 `mean_percent_difference` 给出两个 variant 的全局差异量级。
- 打开 `figures/overview.png` 的右三面板（差值图），观察差异的空间分布。差异集中在某些特定层段是合理的；如果整个体都有系统性偏差，说明两个 baseline 或 modifier 的差异超出了随机波动。

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

- 是否支持更多 baseline 方法（如外部导入的 LFM 体直接作为 variant）。
- 是否支持更多 modifier 类型（如断层扰动、流体替代场景）。
- 是否支持按井型或区块分组建模（同一 run 内的不同 well group）。
- 是否在 comparison 中增加沿井轨迹的逐样点对比图。
- 是否支持为合成训练自动生成 LFM variant 的组合策略。
- trend baseline 是否支持多层位分段拟合（当前只用首末层位）。
