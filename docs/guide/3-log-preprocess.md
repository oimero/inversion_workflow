# 03 测井预处理

`log_preprocess.py` 接在曲线筛选之后，对选出的曲线做规范化、清洗和复核，产出可直接交给井震标定的预处理 LAS。

读完这篇你会知道：数据怎么流转、每一步在做什么校验、出问题时怎么从报告里定位。

---

## 快速开始

```bash
python scripts/log_preprocess.py
python scripts/log_preprocess.py --config experiments/my_project.yaml
python scripts/log_preprocess.py --output-dir /tmp/preprocess_test
```

不带参数时，脚本自动从输出目录发现最新的曲线筛选产物，在 `<output_root>/log_preprocess_<timestamp>/` 下写出结果。

---

## 脚本在做什么

预处理分两趟完成。

### 第一趟：逐井逐曲线规范化

对每口通过第二步筛选的井，从原始 LAS 中加载第二步识别出的全部曲线（不只是 primary，还包括同类 secondary），依次做：

1. **缺失哨兵替换** — 把 `-999.0`、`-999.25`、`-9999.0`、`-99999.0`、LAS header 声明的 NULL 值等已知占位符统一转为 NaN。
2. **缩写规范化**（可关闭）— 把原始 mnemonic 映射到标准名，比如原始 `DT`、`DTC`、`AC` 都映射为 `DT_USM`。
3. **单位规范化**（可关闭）— 声波统一为 `us/m` 慢度，密度统一为 `g/cm3`。同时做两层单位 QC：
   - **硬失败**：声波慢度单位中位数 > 1000、声波速度单位中位数 < 1000、密度 g/cm3 中位数 > 100、密度 kg/m3 中位数 < 10 — 直接判该曲线失效。
   - **软提示**：us/ft 中位数像 us/m、us/m 中位数像 us/ft、密度 g/cm3 中位数偏高 — 只记入 QC 报告。
4. **连续常值段替换** — 把长度 ≥ 阈值的严格连续相同值段置为 NaN。井径默认跳过（防止误伤）。
5. **收集全局阈值样本** — 把每条通过单位硬校验的 step2-primary 曲线的清洗后数据按标准曲线名汇集。

### 第二阶段：全局阈值 + 逐井复核

1. **计算全局分位数** — 按标准曲线名（`DT_USM`、`RHO_GCC` 等）分别统计 q01/q99。样本量不足时跳过并标记。
2. **阈值优先级** — 对每条曲线，按「该井该曲线的手动配置 → 该曲线的全局手动配置 → 自动分位数」顺序解析最终使用的上下限。
3. **极值替换** — 超出上下限的有限值置为 NaN。
4. **可用性判定** — 检查最终有效点是否满足最低数量和相对初始有效点的最低比例。
5. **Primary 接管** — 如果某 category 的 primary 曲线不可用，按顺序尝试同类 secondary。接管只在同一 category 内发生。

---

## 配置参考

```yaml
log_preprocess:
  # 来源 — null 表示自动发现最新第二步产物
  screen_file: null
  input_las_dir: null
  curve_inventory_file: null
  classification_dir: null

  # 输出子目录
  output_las_dir: preprocessed_las

  # 必须同时具备的类别（缺任一个则整井 failed）
  required_categories: [p_sonic, density]

  # 需要处理的类别集合
  selected_categories:
    - caliper
    - gamma_ray
    - s_sonic
    - p_sonic
    - density
    - resistivity
    - spontaneous_potential
    - porosity
    - permeability
    - water_saturation

  mnemonic_standardization:
    enabled: true

  unit_standardization:
    enabled: true
    unit_mismatch_qc: true

  constant_runs:
    enabled: true
    min_run_length: 8
    min_run_length_by_category:
      p_sonic: 16
      s_sonic: 16
      density: 16
      gamma_ray: 16
      resistivity: 16
      spontaneous_potential: 16
      porosity: 16
      permeability: 16
      water_saturation: 16
    replacement: null
    exclude_categories: [caliper]

  outliers:
    enabled: true
    strategy: global_quantile_with_override
    lower_quantile: 0.01
    upper_quantile: 0.99
    replacement: null
    range_override_file: experiments/log_preprocess_ranges.yaml
    min_samples_for_auto_threshold: 1000

  usable_thresholds:
    min_valid_samples: 100
    min_valid_fraction_of_initial: 0.70

  export:
    null_value: -999.25
    write_fmt: "%.6f"
```

### `required_categories`

井必须同时拥有这些类别（经过预处理并有可用曲线）才能 `passed`。默认 `[p_sonic, density]`。如果某口井缺少任一 required，状态降为 `failed`。

### `selected_categories`

脚本只处理这些类别。不在列表中的即使第二步识别了也不加载。类别名是工作流语义类别（`p_sonic`、`density`），不是 LAS 原始 mnemonic。

### `mnemonic_standardization.enabled`

`true`（默认）时，导出的预处理 LAS 使用标准 mnemonic（`DT_USM`、`RHO_GCC` 等）。`false` 时保留原始 mnemonic。无论哪种模式，`mnemonic_mapping.csv` 始终记录原始名到最终名的映射。

### `unit_standardization.enabled`

`true`（默认）时执行单位转换和硬先验检查。`false` 时跳过，但缺失哨兵替换和后续清洗仍按正常流程进行。

### `constant_runs.min_run_length` 与 `min_run_length_by_category`

连续相同值段长度 ≥ 阈值时替换为 null。`min_run_length` 是兜底默认值，`min_run_length_by_category` 中的配置优先。不同类别应使用不同阈值：声波和密度默认 16 点，GR 默认 16 点，而 `min_run_length: 8` 只对未在 by_category 中列出的类别生效。

### `constant_runs.exclude_categories`

这些类别跳过连续常值段替换。默认只排除 `caliper`，因为井径曲线长时间保持同一值是正常现象。排除的类别仍然记录在报告中（`action: skip_caliper`），只是不做替换。

### `outliers.range_override_file`

手动指定阈值上下限的 YAML 文件。格式：

```yaml
global:
  DT_USM:
    min: 120.0
    max: 520.0
  RHO_GCC:
    min: 1.6
    max: 3.0

well_curve:
  A1:
    DT_USM:
      min: 130.0
      max: 480.0
```

优先级：`well_curve.<well>.<standard>` → `global.<standard>` → 自动分位数。每一级都可以只填 `min` 或 `max`，未填的边回退到下一级。

### `usable_thresholds`

判定一条曲线是否可用的双条件：最终有效点数 ≥ `min_valid_samples`，且最终有效点数相对初始有效点数的比例 ≥ `min_valid_fraction_of_initial`。

### `export`

`null_value` 是导出 LAS 时写入的缺失值（`-999.25`），与数据清洗无关（清洗阶段全部用 NaN 标记缺失）。

---

## 单位处理规则

### 声波类（`p_sonic` / `s_sonic`）

| 输入单位 | 处理 | 硬失败条件 |
|----------|------|-----------|
| `us/ft`、`μs/ft` 等 | × 3.28084 → `us/m` | 中位数 > 1000 |
| `us/m`、`μs/m` 等 | 保留 | 中位数 > 1000 |
| `m/s` | 1e6 / value → `us/m` | 中位数 < 1000 |
| 其他 | 不转换，标记硬失败 | — |

非正数的物理量为 NaN（慢度、速度不可能 ≤ 0）。`us/ft` 中位数在 40-160 范围内时追加 QC 提示（看起来更像 `us/m`），`us/m` 中位数在 40-160 时追加反向提示（看起来更像 `us/ft`）。这些都是软提示，不阻止处理。

### 密度类（`density`）

| 输入单位 | 处理 | 硬失败条件 |
|----------|------|-----------|
| `g/cm3`、`g/cc` 等 | 保留 | 中位数 > 100 |
| `kg/m3`、`kg/m^3` | ÷ 1000 → `g/cm3` | 中位数 < 10 |
| 其他 | 不转换，标记硬失败 | — |

`g/cm3` 中位数 > 10 时追加 QC 提示（疑似实际为 kg/m3 但数值异常）。

### 其他类别

不做单位转换，原始数值保留，原始单位作为标准单位原样带出。

### 后续需要速度怎么办

预处理 LAS 里的声波始终是慢度 `DT_USM`（`us/m`）。进入井震标定或构造 `LogSet` 时需要显式转换：

```text
Vp (m/s) = 1e6 / DT_USM (us/m)
```

不要把慢度曲线直接命名为 `Vp`。

---

## 输出文件

所有文件在 `<output_root>/log_preprocess_<timestamp>/` 下：

### `preprocessed_las/*.las`

通过预处理的井的标准 LAS。曲线使用标准 mnemonic，单位已统一，缺失值填 `-999.25`。只有 `preprocess_status == passed` 的井才会导出。

### `well_preprocess_status.csv` — 每井一行

| 字段 | 含义 |
|------|------|
| `well_name` | 井名 |
| `preprocess_status` | `passed` / `failed` |
| `usable_p_sonic` | 纵波时差是否最终可用 |
| `usable_density` | 密度是否最终可用 |
| `usable_caliper` | 井径是否最终可用 |
| `final_p_sonic` | 最终使用的标准纵波曲线名 |
| `final_density` | 最终使用的标准密度曲线名 |
| `final_caliper` | 最终使用的标准井径曲线名 |
| `preprocessed_las` | 导出 LAS 路径 |
| `reasons` | 失败原因 |

后续步骤从这里判断每口井的可用性，不再回查第二步。

### `preprocess_summary.csv` — 每井每条曲线一行

最全的明细。包含原始名、标准名、单位转换动作、硬失败原因、QC 标记、每步替换点数、最终有效点数和占比、使用的阈值上下限及来源、最终是否被选为该类别的可用曲线。

### `mnemonic_mapping.csv`

原始 mnemonic → 标准 mnemonic 的映射，附带类别、是否第二步 primary、原始 LAS 路径。

### `unit_conversion_report.csv`

每条曲线的单位转换详情：转换动作、转换前后统计量（中位数、P01、P99）。

### `unit_mismatch_qc.csv`

触发单位错配软提示的曲线清单，含 QC 标记和当时的中位数/P01/P99。

### `primary_reselection_report.csv`

primary 接管记录。每行记录：哪个 category 的哪条 primary 失效、由哪条 secondary 接管、失效原因。

### `constant_run_report.csv`

每个被替换的连续常值段一行：起止 MD、段长、常值、操作（`set_null` 或 `skip_caliper`）。

### `outlier_report.csv`

每条曲线的极值替换明细：上下限、阈值来源、替换点数。

### `range_thresholds.csv`

每个标准曲线名一行：最终使用的全局上下限、来源（`global_quantile` / `manual_global` / `skipped_insufficient_samples`）、参与统计的样本数。

### `skipped_wells.csv` / `skipped_curves.csv`

无法处理的井和曲线，附原因标签。

### `run_summary.json`

输入路径、配置摘要、各项计数。

---

## 如何阅读结果

### 第一步：看终端输出

```
Log preprocess summary: 38 step2-passed wells, 35 passed, 3 failed, 35 LAS exported.
```

如果 `failed` 井数多，打开 `well_preprocess_status.csv` 看 `reasons` 列。

### 第二步：定位 individual curve failures

`skipped_curves.csv` 按原因标签分类——`no_finite_samples`、`unsupported_sonic_unit`、`density_g_cm3_unit_impossible_median_gt_100` 等。数量集中的原因说明系统性数据问题。

### 第三步：检查 primary reselection

`primary_reselection_report.csv` 非空时，说明有井的 primary 曲线在预处理中失效并被 secondary 接替。关注接管后是否影响全局阈值统计（primary 失效意味着该曲线的值未进入阈值计算）。

### 第四步：检查单位 QC

`unit_mismatch_qc.csv` 列出所有疑似单位与数值不匹配的曲线。是软提示——脚本不自动纠正，但第二轮迭代前应该人工确认这些曲线的单位字段。

### 第五步：抽查极端阈值

`range_thresholds.csv` 中 `source == skipped_insufficient_samples` 的条目——有效样本太少，没有自动阈值。这些标准曲线名下所有井的极值处理都不会生效，后续可能需要手动指定。

### 第六步：抽查一口井的完整轨迹

打开 `preprocess_summary.csv` 筛选某口井，检查每条曲线的 `conversion_action` → `constant_replaced_points` → `outlier_replaced_points` → `final_valid_fraction` 链条是否符合预期。

---

## 留到第二轮

- 单位错配软提示是否自动纠正还是只报告。
- 全局阈值是否按层段、井型或工区分区细化。
- 是否为极值处理生成直方图 QC 图。
- 连续常值段阈值是否需要继续细化。
