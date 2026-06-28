# 03 测井预处理

`well_preprocess.py` 接在曲线筛选之后，对选出的曲线做规范化、清洗和复核，产出可直接交给井震标定的预处理 LAS。

---

## 快速开始

```bash
python scripts/well_preprocess.py
python scripts/well_preprocess.py --config experiments/my_project.yaml
python scripts/well_preprocess.py --output-dir /tmp/preprocess_test
```

不带参数时，脚本自动从输出目录发现最新的曲线筛选产物，在 `<output_root>/well_preprocess_<timestamp>/` 下写出结果。

---

## 运行前需要什么

| 输入 | 用途 |
|------|------|
| `well_screen.csv` | 确定第二步 passed 井、原始 LAS 路径和 primary 曲线 |
| `las_curve_inventory.csv` | 读取每条曲线的类别、单位、primary 标记 |
| `curve_classification/*.json` | 复原逐井分类细节，支持 primary 接管 |
| `selected_las/*.las` | 读取第二步导出的瘦身 LAS 并做清洗 |
| 可选阈值 override YAML | 人工指定曲线值域上下限 |

---

## 配置参考

```yaml
well_curves:
  required_categories: [p_sonic, density]
  selected_categories: [...]

well_preprocess:
  md_resampling:
    step_m: 0.1
    max_interpolation_gap_m: 0.5

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
    exclude_categories: [caliper]

  outliers:
    enabled: true
    lower_quantile: 0.01
    upper_quantile: 0.99
    range_override_file: experiments/common/well_preprocess_ranges.yaml
    min_samples_for_auto_threshold: 1000

  usable_thresholds:
    min_valid_samples: 100
    min_valid_fraction_of_initial: 0.70
```

### `source_runs`

默认自动接上最新一次曲线筛选结果。复现实验时可按需加入 `source_runs.well_screen_dir` 固定整套输入。

### `md_resampling`

第三步输出的标准 LAS 必须使用规则 MD 网格。原始 LAS 可以是不规则采样，但必须在这里显式规则化一次，后续 Step 4/5 不再各自猜测或重复重采样。

规则化之前，原生 LAS 曲线由 `IrregularMdCurveSet` 显式承载；只有生成规则 MD 轴后才构造 `grid.Log`。不得把 irregular MD 伪装成要求等采样的 `grid.Log`。

- `step_m`：输出 MD 采样间隔，单位米。当前工区按原 LAS 名义 `STEP` 固定为 `0.1 m`。
- `max_interpolation_gap_m`：允许插值的相邻有限源样点最大距离。超过该距离的缺口在规则网格上保持 NaN；当前配置为 `0.5 m`。

规则网格以原始首个 MD 样点为起点，只覆盖原始 MD 支撑范围。脚本要求原始 MD 有限且严格递增，不排序、不去重，也不跨长缺口填值。

### `required_categories`

决定一口井预处理后是否还能进入井震标定。默认要求纵波声波和密度都可用；只要其中一类在清洗后不可用，这口井就会被挡在第三步之外。

### `selected_categories`

脚本只处理这些类别。不在列表中的即使第二步识别了也不加载。类别名是工作流语义类别（`p_sonic`、`density`），不是 LAS 原始 mnemonic。

### `constant_runs`

#### `min_run_length`

用于识别”长时间完全不变”的可疑曲线段。这类段落常见于仪器、导出或填充值问题，会被置为空值。不同曲线对常值段的容忍度不同，可通过 `min_run_length_by_category` 让声波、密度、GR 等类别单独设阈值；没有单独配置的类别才使用 `min_run_length`。

#### `exclude_categories`

这些类别跳过连续常值段替换。默认只排除 `caliper`，因为井径曲线长时间保持同一值是正常现象。排除的类别仍然记录在报告中（`action: skip_caliper`），只是不做替换。

### `outliers`

#### `range_override_file`

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
  <well-name>:
    DT_USM:
      min: 130.0
      max: 480.0
```

优先级：`well_curve.<well>.<standard>` → `global.<standard>` → 自动分位数。每一级都可以只填 `min` 或 `max`，未填的边回退到下一级。

### `usable_thresholds`

用于避免“清洗后只剩一点点数据”的曲线继续被当成可靠曲线。脚本同时检查最终有效点数量和保留下来的有效比例，两项都过关才认为曲线可用。

---

## 脚本在做什么

预处理分两趟完成。

### 第一趟：逐井逐曲线规范化

对每口通过第二步筛选的井，从原始 LAS 中加载第二步识别出的全部曲线（不只是 primary，还包括同类 secondary），依次做：

1. **缺失值识别** — 把 LAS 中常见的缺失占位符统一转为空值。
2. **缩写规范化**（可关闭）— 把原始 mnemonic 映射到标准名，比如原始 `DT`、`DTC`、`AC` 都映射为 `DT_USM`。
3. **单位规范化**（可关闭）— 声波统一为 `us/m` 慢度，密度统一为 `g/cm3`。数值明显不符合单位常识的曲线会被判为不可用；疑似单位写错但还能处理的曲线会进入 QC 报告。
4. **连续常值段替换** — 把长时间完全不变的可疑段落置为空值。井径默认跳过，避免误伤真实井径响应。
5. **收集全局阈值样本** — 把每条通过单位硬校验的 step2-primary 曲线的清洗后数据按标准曲线名汇集。

### 第二阶段：全局阈值 + 逐井复核

1. **计算全局分位数** — 按标准曲线名（`DT_USM`、`RHO_GCC` 等）分别统计 q01/q99。样本量不足时跳过并标记。
2. **阈值优先级** — 优先使用单井手动阈值，其次使用全局手动阈值，最后才使用自动分位数。
3. **极值替换** — 超出上下限的有限值置为 NaN。
4. **可用性判定** — 检查最终有效点是否满足最低数量和相对初始有效点的最低比例。
5. **Primary 接管** — 如果某 category 的 primary 曲线不可用，按顺序尝试同类 secondary。接管只在同一 category 内发生。
6. **MD 规则化** — 对最终入选曲线统一重采样到配置的规则 MD 网格，只在不超过 `max_interpolation_gap_m` 的有限样点之间插值。
7. **派生 AI** — 在规则化后的 `DT_USM` 和 `RHO_GCC` 上重新计算 AI，然后导出标准 LAS。

---

## 单位处理规则

### 声波类（`p_sonic` / `s_sonic`）

| 输入单位 | 处理 | 硬失败条件 |
|----------|------|-----------|
| `us/ft`、`μs/ft` 等 | × 3.28084 → `us/m` | 中位数 > 1000 |
| `us/m`、`μs/m` 等 | 保留 | 中位数 > 1000 |
| `m/s` | 1e6 / value → `us/m` | 中位数 < 1000 |
| 其他 | 不转换，标记硬失败 | — |

非正数的慢度或速度会被视为空值。脚本还会检查“单位写法”和“数值范围”是否相互匹配：例如单位写成 `us/ft` 但数值更像 `us/m`，或反过来。这类情况只进入 QC 报告，不会自动改单位，也不会阻止处理。

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

### 波阻抗派生

通过预处理的井固定派生：

```text
AI (m/s*g/cm3) = (1e6 / DT_USM) * RHO_GCC
```

只有 `DT_USM` 和 `RHO_GCC` 同时有限且为正的样点才计算 AI；其他样点保持缺失并在 LAS 中写为 `NULL`。

---

## 核心输出文件

所有文件在 `<output_root>/well_preprocess_<timestamp>/` 下：

### `preprocessed_las/*.las`

通过预处理的井的标准 LAS。MD 已按 `md_resampling.step_m` 规则采样，LAS 头中的 `STRT/STOP/STEP` 与实际输出轴一致。曲线使用标准 mnemonic，单位已统一，固定包含 `DT_USM`、`RHO_GCC` 和规则化后重新派生的 `AI`；其他入选辅助曲线照常保留。长缺口保持缺失并写为 `-999.25`。只有 `preprocess_status == passed` 的井才会导出。

### `md_resampling_report.csv`

每井每条导出曲线一行，记录原始/输出 MD 范围与样点数、原始步长 min/median/max、输出步长、插值样点数、保留空值数和未跨越的长缺口数量。

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
| `md_original_regular` | 原始 MD 是否已经规则采样 |
| `md_output_step_m` | 输出规则 MD 步长 |
| `md_output_regular` | 输出 MD 规则性校验结果 |
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

### 第二步：定位单条曲线为什么失效

`skipped_curves.csv` 会列出每条失效曲线的原因。如果某一类原因大量出现，通常不是单井问题，而是数据命名、单位或导出规则存在系统性偏差。

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
