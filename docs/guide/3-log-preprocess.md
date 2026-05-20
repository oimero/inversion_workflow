# 03 测井预处理

本文讨论第三个规划脚本：`log_preprocess.py`。

它接在 `las_curve_screen.py` 后面，处理第二步导出的有用曲线。当前阶段做三件事：

1. 缩写规范化。
2. 单位规范化。
3. 异常值处理：连续截断值/常值段、极值。

暂时不做标准化，也不做扩径段处理。

参考 notebook：

- `notebooks/well_replace_constant_value@20260403.ipynb`

## 目标

`log_preprocess.py` 的目标不是把曲线“修漂亮”，而是生成一套可追溯的预处理版本：

- 原始 mnemonic 被映射到了哪个标准 mnemonic。
- 原始单位被转换到了哪个标准单位。
- 哪些值被识别为截断值、常值段或极值。
- 每口井、每条曲线替换了多少点。
- 是否发现“单位字段和实际数值分布明显不匹配”的可疑曲线。

所有操作都必须输出报告，方便第二轮迭代时回看。

## 输入

- 第二阶段筛选结果：`well_curve_screen.csv`。
- 第二阶段导出的瘦身 LAS：`selected_las/*.las`。
- 第二阶段曲线清单：`las_curve_inventory.csv`。
- 可选人工范围配置：例如 `experiments/log_preprocess_ranges.yaml`。
- 可选缩写映射配置：例如 `experiments/log_mnemonic_map.yaml`。

建议配置片段：

```yaml
log_preprocess:
  screen_file: scripts/output/las_curve_screen_YYYYMMDD_HHMMSS/well_curve_screen.csv
  input_las_dir: scripts/output/las_curve_screen_YYYYMMDD_HHMMSS/selected_las
  curve_inventory_file: scripts/output/las_curve_screen_YYYYMMDD_HHMMSS/las_curve_inventory.csv
  output_las_dir: preprocessed_las

  mnemonic_standardization:
    enabled: true

  unit_standardization:
    enabled: true
    p_sonic_target: slowness_us_per_m
    density_target: g_cm3
    unit_mismatch_qc: true

  constant_runs:
    enabled: true
    min_run_length: 8
    replacement: null
    exclude_categories: [caliper]

  outliers:
    enabled: true
    strategy: global_quantile_with_override
    lower_quantile: 0.01
    upper_quantile: 0.99
    replacement: null
    range_override_file: experiments/log_preprocess_ranges.yaml
```

这里的 `replacement: null` 表示替换为缺失值。导出 LAS 时再统一写成 `-999.25`。

## 输出

默认输出目录建议为：

```text
scripts/output/log_preprocess_<timestamp>/
```

核心文件：

- `preprocessed_las/*.las`：预处理后的 LAS。
- `preprocess_summary.csv`：一井一曲线一行的汇总。
- `well_preprocess_status.csv`：一井一行的最终可用性清单。
- `mnemonic_mapping.csv`：原始 mnemonic 到标准 mnemonic 的映射。
- `unit_conversion_report.csv`：单位转换报告。
- `unit_mismatch_qc.csv`：单位字段和数值分布疑似不匹配的曲线。
- `primary_reselection_report.csv`：primary 曲线在预处理后失效时的同类曲线接管记录。
- `constant_run_report.csv`：连续常值段报告。
- `outlier_report.csv`：极值处理报告。
- `range_thresholds.csv`：每类曲线最终使用的全局上下限。
- `skipped_wells.csv`：无法读取或无法导出的井。
- `skipped_curves.csv`：无法处理的曲线。
- `run_summary.json`：输入、配置和统计摘要。

`well_preprocess_status.csv` 建议字段：

| 字段 | 含义 |
| --- | --- |
| `well_name` | 井名 |
| `preprocess_status` | `passed`、`failed`、`partial` |
| `usable_p_sonic` | 纵波时差是否在预处理后可用 |
| `usable_density` | 密度是否在预处理后可用 |
| `usable_caliper` | 井径是否在预处理后可用 |
| `final_p_sonic` | 最终使用的标准纵波曲线名 |
| `final_density` | 最终使用的标准密度曲线名 |
| `final_caliper` | 最终使用的标准井径曲线名；没有则为空 |
| `preprocessed_las` | 输出 LAS 路径 |
| `reasons` | 失败或警告原因 |

## 缩写规范化

第二步负责“识别和选择曲线”，第三步负责“形成标准输出名”。

建议保留两套信息：

- `original_mnemonic`：原始 LAS 里的曲线名。
- `standard_mnemonic`：工作流内部使用的标准曲线名。

第一版标准名可以先保持少量：

| category | standard_mnemonic | 说明 |
| --- | --- | --- |
| `p_sonic` | `DT_USM` | 纵波时差，统一为 `us/m` |
| `s_sonic` | `DTS_USM` | 横波时差，统一为 `us/m` |
| `density` | `RHO_GCC` | 密度，统一为 `g/cm3` |
| `gamma_ray` | `GR` | 自然伽马 |
| `caliper` | `CALI` | 井径 |
| `resistivity` | `RT` | 电阻率 |
| `sp` | `SP` | 自然电位 |
| `porosity` | `POR` | 孔隙度 |
| `permeability` | `PERM` | 渗透率 |
| `water_saturation` | `SW` | 含水饱和度 |

导出的预处理 LAS 可以使用标准 mnemonic，但 `mnemonic_mapping.csv` 必须能追溯到原始曲线。

## 单位规范化

声波类建议只保留慢度，不再同时保留速度和慢度两种口径。

理由：

- 当前工区绝大多数原始测井是声波时差曲线。
- 统一到慢度后，极值阈值可以按全局 `DT_USM`、`DTS_USM` 分布统计。
- 后续如果需要速度，可以在井震标定或物性资产阶段由慢度确定性转换得到。

代价是：下游凡是进入 `wtie` 或其他外部库的地方，通常仍需要 `Vp` 速度 `m/s`。因此第三步的标准 LAS 保留 `DT_USM`，但第四步构造 `grid.LogSet` 时必须显式转换出 `Vp`，不能把慢度曲线直接命名为 `Vp`。

建议规则：

| 输入物理量 | 输入单位 | 输出 |
| --- | --- | --- |
| 声波时差 | `us/ft`、`μs/ft`、`µs/ft` | 乘 `3.280839895` 得到 `us/m` |
| 声波时差 | `us/m`、`μs/m`、`µs/m` | 保留数值，单位规范为 `us/m` |
| 声波速度 | `m/s` | `1e6 / velocity_mps` 得到 `us/m` |
| 密度 | `kg/m3` | 除以 `1000` 得到 `g/cm3` |
| 密度 | `g/cm3`、`g/cc`、`g/cm^3` | 保留数值，单位规范为 `g/cm3` |

### 单位错配 QC

单位字段不一定可信，所以第三步应顺便做数值分布 QC。

第一版可以用简单统计规则标注疑似问题，不自动修：

- 标准化前后记录均值、中位数、P1、P99。
- 如果曲线标注为 `us/m`，但中位数更像 `us/ft`，写入 `unit_mismatch_qc.csv`。
- 如果曲线标注为 `m/s`，但中位数落在典型慢度范围，也写入 QC。
- 如果密度单位标注为 `g/cm3`，但中位数在 `1600-3000`，疑似实际是 `kg/m3`。

这些规则只做提示，阈值应放进配置，不要硬编码进函数。岩性差异会让“典型范围”失效，例如高密度矿物或特殊岩性会触发误报。是否自动纠正，留到第二轮迭代。

## Primary 曲线复核

第二步选出的 primary 曲线只是基于 header、mnemonic、单位字段和人工 override 的选择。第三步完成单位规范化和异常值处理后，还需要复核 primary 是否仍然可用。

建议第一版使用简单规则：

- 如果 primary 曲线无法完成单位规范化，标记为失效。
- 如果 primary 曲线在常值段和极值处理后有效点比例过低，标记为失效。
- 如果同一类别存在 secondary 候选曲线，并且 secondary 能通过预处理，则允许自动接管。
- 如果没有可接管曲线，则整井或该关键类别失败。

接管只在同一类别内发生，例如 `DT` 失效后可以尝试 `DTC`，但不能用速度或密度曲线跨类别补位。所有接管行为写入 `primary_reselection_report.csv`，并同步更新 `preprocess_summary.csv` 中的最终曲线名。

## 连续截断值/常值段

当前 notebook 的逻辑是：

- 将 LAS 读成每井 `dict[str, grid.Log]`。
- 对每条曲线做严格相等判断。
- 连续相同值长度 `>= min_run_length` 时，整段替换为 `anomaly_value`。
- `-999.0`、`-999.25`、`-99999`、`NaN` 不参与连续段判断。

第一版保留这个思路，但按你的判断调整为：

- 除 `caliper` 井径以外，所有类别默认启用连续常值段替换。
- 替换目标为 null，导出 LAS 时写成 `-999.25`。
- 报告里记录常值段的起止 MD、原始值和长度。

井径排除的原因是：井径曲线本来就可能长时间保持同一数值，直接替换会误伤。

井径虽然排除连续常值段替换，但仍然要参与单位规范化、基础有限值检查和可用性统计。这里的 `usable_caliper` 指 LAS 井径曲线是否可用，只服务井眼质量和可选扩径段 QC；斜井路径需要的是 `data/all_well_trace` 里的井轨迹/井斜数据。

连续段报告字段建议：

| 字段 | 含义 |
| --- | --- |
| `well_name` | 井名 |
| `original_mnemonic` | 原始曲线名 |
| `standard_mnemonic` | 标准曲线名 |
| `category` | 曲线类别 |
| `start_md` | 起始 MD |
| `end_md` | 结束 MD |
| `run_length` | 连续点数 |
| `constant_value` | 被替换的原始常值 |
| `action` | `set_null`、`skip_caliper` |

## 极值处理

第一版采用全局自动阈值：

- 对所有通过第二步筛选并完成单位规范化的井做全局统计。
- 按标准曲线名统计阈值，例如 `DT_USM`、`DTS_USM`、`RHO_GCC`。
- 默认使用全局 `q01/q99`。
- 如果人工配置给了某个标准曲线或某井某曲线的上下限，人工配置优先。
- 超出上下限的点置 null。

示例人工范围配置：

```yaml
global:
  DT_USM:
    min: 120.0
    max: 520.0
  DTS_USM:
    min: 200.0
    max: 900.0
  RHO_GCC:
    min: 1.6
    max: 3.0

well_curve:
  A1:
    DT_USM:
      min: 130.0
      max: 480.0
```

如果某一类有效点太少，不硬算分位数：

- 有效点数少于阈值时，只使用人工范围。
- 没有人工范围时只报告，不处理。
- 每个阈值记录来源：`manual`、`global_quantile`、`skipped_insufficient_samples`。

## 对 `process.py` 的看法

`process.py` 现在混乱的根源是对象层级没分清。

`wtie.processing.grid.LogSet` 要求至少包含 `Vp` 和 `Rho`，本质是“标准物性曲线集合”。但第三个脚本处理的是第二步导出的任意曲线集合，里面可能有 `GR`、`CALI`、`SP`、`PERM`，也可能暂时没有标准命名的 `Vp/Rho`。所以这个阶段不应该强行用 `LogSet`。

建议引入项目内对象：

```text
WellCurveSet
```

它表示“同一口井、同一采样轴上的任意 LAS 曲线集合”，不要求必须有 Vp/Rho。

如果以后某个步骤确认已经有标准 `Vp/Rho`，再从 `WellCurveSet` 显式构造 `grid.LogSet`。

## 模块边界

为了避免 `cup.well` 被拆得太碎，第三步先合并为两个主要模块：一个放井曲线集合对象，一个放预处理策略和算法。等 `preprocess.py` 变得太大时，再拆出 `standardize.py`、`cleaning.py` 或 `thresholds.py`。

### 建议新增

`cup.well.curves`

- `WellCurveSet`：任意曲线集合，不要求 Vp/Rho。
- `validate_shared_basis(curves)`：检查曲线是否共享 MD 轴。
- `select_curves_by_category(curve_set, categories)`：按第二步分类选择曲线。

这些对象可以和第二步的 `CurveInfo`、`CurveSelection` 放在同一个 `curves.py` 里，避免 `curve_set.py` 只有少量薄包装。

`cup.well.preprocess`

- `MnemonicStandardizationPolicy`
- `UnitStandardizationPolicy`
- `ConstantRunPolicy`
- `OutlierPolicy`
- `PreprocessPolicy`
- `CleaningReport`
- `standardize_mnemonic(curve_info, policy)`
- `standardize_log_unit(log, category, target_unit)`
- `detect_unit_mismatch(log, category, unit_info)`
- `replace_constant_runs(log, policy) -> tuple[grid.Log, CleaningReport]`
- `remove_outliers(log, policy) -> tuple[grid.Log, CleaningReport]`
- `compute_global_quantile_thresholds(curve_sets, curve_inventory, q_low, q_high)`
- `merge_threshold_overrides(auto_thresholds, manual_thresholds)`
- `reselect_primary_curve(curve_set, category, failed_primary, policy)`
- `preprocess_curve_set(curve_set, policies) -> PreprocessedCurveSet`

### 建议移动或拆分

`src/cup/well/process.py` 不建议继续变成大杂烩，但也不必为了第一版强行拆出多个很薄的文件。迁移策略可以是：

- 先新增 `cup.well.preprocess`，承接缩写规范化、单位规范化、单位错配 QC、异常值、常值段、极值阈值。
- `clip_logsets_by_well_tops()` 可以暂时留在 `process.py`，等井分层相关函数超过一两个后再拆 `cup.well.tops`。
- 任意曲线集合对象先放 `cup.well.curves`，与第二步的 `CurveInfo`、`CurveSelection` 同处一处。

旧的 `replace_constant_value_intervals_in_log_dicts()` 可以在第一轮重构中保留 wrapper，但新脚本应调用更清楚的新函数。`LogsetInput = Union[grid.LogSet, Dict[str, grid.Log]]` 这种签名建议逐步淘汰，因为它把“标准物性集合”和“任意曲线集合”混在一起了。

明确迁移路径：

1. 第二、三步内部统一使用 `WellCurveSet` 表达任意 LAS 曲线集合。
2. 只有当第四步确认 `DT_USM` 和 `RHO_GCC` 都可用，并把 `DT_USM` 转成 `Vp(m/s)` 后，才构造 `grid.LogSet({"Vp": ..., "Rho": ...})`。
3. 旧函数若仍接受 `LogsetInput`，只作为兼容 wrapper，内部立即转换到新对象或明确拒绝不合适的输入。

### 脚本层负责

`log_preprocess.py` 负责：

- 读取第二步 manifest。
- 组织输入输出路径。
- 调用规范化、清洗和阈值模块。
- 写 CSV/JSON 报告。
- 导出预处理后 LAS。

它不应该自己实现连续段扫描、分位数阈值计算、单位转换、primary 接管或单位错配判断。

## 前三步后的建议文件布局

第一轮重构先保持文件数量克制。`cup` 的边界可以暂定为：

### `src.cup.petrel`

`cup.petrel` 只负责 Petrel 交换格式，不再承载 LAS 曲线业务逻辑。

| 文件 | 职责 |
| --- | --- |
| `petrel/load.py` | 读取 Petrel 文本：井头、井分层、解释层位、checkshots。 |
| `petrel/export.py` | 导出 Petrel 文本：例如 checkshots、解释或后续 Petrel 专用表。 |

需要迁出的内容：

- 地震体读取应归到 `cup.seismic`。
- LAS header 扫描、LAS 曲线提取、LAS 导出应归到 `cup.well.las`。
- 单位转换、异常值处理和曲线选择不属于 Petrel。

### `src.cup.well`

`cup.well` 负责井资产、井曲线和井相关处理。

| 文件 | 职责 |
| --- | --- |
| `well/assets.py` | 井头对象、井资产清单、工区内外状态、近邻井对 QC。 |
| `well/curves.py` | 曲线类别 schema、mnemonic 候选、曲线分类结果、primary 选择、`WellCurveSet`。 |
| `well/las.py` | LAS header 扫描、曲线提取、瘦身 LAS 和预处理 LAS 导出。 |
| `well/preprocess.py` | 缩写规范化、单位规范化、单位错配 QC、常值段、极值、primary 接管；第一版也可以暂存少量井分层裁剪函数。 |
| `well/tops.py` | 可选拆分点；等井分层相关函数变多后再创建。 |
| `well/trajectory.py` | 井轨迹读取和直井/斜井复核；第四步前再细化。 |
| `well/wavelet.py` | 保留现状，小波生成、读取和采样间隔校验。 |
| `well/viz.py` | 井曲线、直方图和 QC 图。 |

如果后续 `well/preprocess.py` 或 `well/curves.py` 变得过大，再拆出 `standardize.py`、`cleaning.py`、`thresholds.py` 或 `curve_screen.py`。当前阶段先不拆。

## 已定策略

- 声波统一成慢度 `us/m`。
- 极值超限点置 null。
- 自动阈值按全局分位数统计。
- 除井径以外的所有类别都启用连续常值段替换。
- 单位规范化、缩写规范化和异常值处理放在同一个预处理脚本里。
- 第四步使用 `well_preprocess_status.csv` 判断 `usable_p_sonic`、`usable_density` 和可选的 `usable_caliper`，不直接使用第二步的原始曲线存在性。斜井路径是否可走由第一步的 `has_well_trace` 和第四步轨迹 QC 决定。

## 留到第二轮

- 单位错配 QC 是否自动纠正，还是只报告。
- 全局阈值是否需要按层段、井型或工区分区细化。
- 是否为极值处理生成直方图 QC 图。
- 连续常值段是否需要按类别设置不同 `min_run_length`。
