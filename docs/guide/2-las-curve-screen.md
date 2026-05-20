# 02 LAS 曲线筛选与导出

本文讨论第二个规划脚本：`las_curve_screen.py`。

它接在 `well_inventory.py` 后面，只处理已经通过第一步空间筛选的井：在工区内或工区边缘，并且同时有井头数据和 LAS 文件。

参考 notebook：

- `notebooks/well_select_and_io@20260404.ipynb`
- `notebooks/well_import_and_export@20260404.ipynb`

这两个 notebook 的核心流程是：读取 LAS 文件头，用 LLM 对曲线归类；再按归类结果提取有用曲线，导出新的 LAS。单位和缩写的最终规范化放到第三步预处理里完成。

## 目标

`las_curve_screen.py` 回答四件事：

1. 每口候选井有哪些曲线、单位、描述和 LAS 采样轴信息。
2. 哪些曲线对应井径、GR、横波时差、纵波时差、密度、电阻率、SP、孔隙度、渗透率、含水饱和度等类别。
3. 哪些井同时包含“纵波声波/时差”和“密度”，可以进入后续合成记录和井震标定。
4. 哪些井正式包含 LAS 井径曲线，可作为井眼质量 QC 的候选。
5. 将筛选出的有用曲线导出为瘦身 LAS，并保留完整的分类、选择和跳过原因。

这个脚本可以调用 LLM，但 LLM 不应是唯一判断来源。推荐优先级：

```text
人工 override > 已缓存 LLM 分类 > 本地 mnemonic 规则 > 本次 LLM 分类
```

如果没有配置 LLM，本脚本也应该能用本地 mnemonic 规则跑出一个可审计的初版结果。

## 输入

- 第一阶段清单：`well_inventory.csv`。
- 原始 LAS 目录：`data/all_well_las`。
- 曲线分类 schema：建议项目内维护一个 YAML，而不是继续依赖临时 `import_template.csv`。
- 可选人工 override：例如 `experiments/curve_alias_overrides.yaml`。
- 可选 LLM 配置：模型名、API key 路径、是否启用、失败重试次数。

建议配置片段：

```yaml
las_curve_screen:
  inventory_file: scripts/output/well_inventory_YYYYMMDD_HHMMSS/well_inventory.csv
  las_dir: all_well_las
  include_survey_positions: [inside, near_outside]
  required_categories: [p_sonic, density]
  selected_categories:
    - caliper
    - gamma_ray
    - s_sonic
    - p_sonic
    - density
    - resistivity
    - sp
    - porosity
    - permeability
    - water_saturation
  curve_schema_file: experiments/curve_schema.yaml
  curve_override_file: experiments/curve_alias_overrides.yaml
  llm:
    enabled: false
    cache_dir: scripts/output/las_curve_screen_cache
    max_retry: 1
  export:
    selected_las_dir: selected_las
    null_value: -999.25
    write_fmt: "%.6f"
```

`write_fmt` 第一版可以用统一默认值，保证实现简单和可重复。后续如果 QC 发现文件过大或不同曲线需要不同精度，再扩展为按类别配置，例如密度、孔隙度、电阻率分别指定格式。

## 输出

默认输出目录建议为：

```text
scripts/output/las_curve_screen_<timestamp>/
```

核心文件：

- `las_curve_inventory.csv`：每条 LAS 曲线一行，记录 mnemonic、unit、descr、category、分类来源。
- `well_curve_screen.csv`：每口井一行，记录是否通过筛选、各类主曲线、失败原因。
- `selected_las/*.las`：只导出 `screen_status == passed` 的井，即同时具备 `p_sonic` 和 `density` 的井。
- `curve_classification/*.json`：逐井分类缓存，便于断点续跑和人工复查。
- `skipped_wells.csv`：未通过筛选的井及原因。
- `skipped_curves.csv`：分类命中但提取失败、歧义未解或被 override 禁用的曲线。
- `run_summary.json`：输入路径、配置、统计摘要和失败原因汇总。

`well_curve_screen.csv` 建议字段：

| 字段 | 含义 |
| --- | --- |
| `well_name` | 井名 |
| `las_file` | 原始 LAS 路径 |
| `screen_status` | `passed`、`failed`、`partial` |
| `has_p_sonic` | 是否有纵波声波/时差 |
| `has_density` | 是否有密度 |
| `has_caliper` | 是否有正式归类的井径曲线 |
| `primary_p_sonic` | 选定的纵波曲线 mnemonic |
| `primary_density` | 选定的密度曲线 mnemonic |
| `primary_caliper` | 选定的井径曲线 mnemonic；没有则为空 |
| `selected_curve_count` | 成功提取并导出的曲线数 |
| `exported_las` | 瘦身 LAS 路径 |
| `reasons` | 失败或警告原因 |

`las_curve_inventory.csv` 建议字段：

| 字段 | 含义 |
| --- | --- |
| `well_name` | 井名 |
| `mnemonic` | LAS 曲线名 |
| `unit` | 原始单位 |
| `description` | LAS 描述 |
| `category` | 标准类别 |
| `is_primary` | 是否为该类别主曲线 |
| `classification_source` | `override`、`llm_cache`、`mnemonic_rule`、`llm`、`unclassified` |
| `confidence` | 可选置信度；规则分类可给 1.0 |
| `notes` | 歧义或人工备注 |

## 分类类别

建议使用英文稳定 key，中文只作为展示名：

| key | 中文 | 示例 mnemonic |
| --- | --- | --- |
| `caliper` | 井径/井眼质量 | `CAL`、`CALI`、`BS` |
| `gamma_ray` | GR/泥质 | `GR`、`GR1`、`VSH` |
| `p_sonic` | 纵波声波/时差 | `DT`、`AC`、`DTC`、`DTCO`、`VP` |
| `s_sonic` | 横波声波/时差 | `DTS`、`DTSM`、`DTSH` |
| `density` | 密度 | `DEN`、`RHOB`、`RHOZ`、`RHO` |
| `resistivity` | 电阻率 | `RT`、`LLD`、`LLS`、`MSFL`、`ILD`、`AT90` |
| `sp` | 自然电位 | `SP` |
| `porosity` | 孔隙度 | `POR`、`PHIE`、`PHIT` |
| `permeability` | 渗透率 | `PERM` |
| `water_saturation` | 含水饱和度 | `SW`、`SWE`、`SWT` |

当前 `src/cup/well/mnemonics.py` 已有部分候选集，但缺少电阻率、SP，并且常量以下划线开头。第二个脚本落地前，应该把它整理成公开的类别配置。

## 处理逻辑

### 候选井

只读取第一步清单中满足以下条件的井：

- `has_well_head == true`
- `has_las == true`
- `survey_position in [inside, near_outside]`
- `inventory_status == usable_for_las_screen`

这样第二个脚本不再重新决定空间范围，也不扫描工区外井。

### LAS header 扫描

优先只读取 LAS header 和 `~Curve` 段，拿到：

- 井名、STRT/STOP/STEP/NULL。
- 每条曲线的 mnemonic、unit、descr。
- 曲线数量、采样点数、采样轴是否单调。

不建议像 notebook 里那样把整个 LAS 数据区都喂给 LLM。对 LLM 来说，曲线表和必要的 well/header 信息已经足够；数据区又大又容易泄露无关内容。

### 曲线分类

分类合并顺序：

1. 人工 override：最高优先级，用于修正 LLM 或规则误判。
2. LLM cache：如果已有逐井 JSON，则直接复用。
3. 本地 mnemonic 规则：对常见标准名直接分类。
4. 本次 LLM：只处理仍未分类或存在歧义的井/曲线。

LLM 输出必须校验：

- 只能使用 schema 中定义的类别。
- 返回的 mnemonic 必须真实存在于该 LAS。
- 一个 mnemonic 可以出现在多个类别时，需要进入 `ambiguous` 或要求 override。
- 每个类别如果有多个候选，需要按选择策略确定 primary。

### 人工 override

人工 override 是第二步最重要的人工干预入口。它不是“手工改结果文件”，而是一份可追溯配置，用来覆盖规则或 LLM 的判断。

落地顺序可以分两层：第一版至少实现全局类别优先级和单井单类别 primary；单井禁用、单井强制归类可以先按 schema 预留，等真实井暴露出需要时再实现。

建议至少支持四类 override：

| override | 作用 |
| --- | --- |
| 全局类别优先级 | 例如 `p_sonic: [DT, DTC, DTCO, AC, VP]` |
| 单井单类别 primary | 例如 A1 井的 `p_sonic` 强制使用 `DTC` |
| 单井单曲线禁用 | 例如某井的 `DEN_BAD` 不参与密度候选 |
| 单井单曲线强制归类 | 例如某井的 `RHOZ2` 强制归为 `density` |

这样可以处理“有一口井想用这个缩写，另一口井想用另一个缩写”的情况。全局优先级只提供默认选择，单井 override 永远优先。

示例：

```yaml
global_priority:
  p_sonic: [DT, DTC, DTCO, AC, VP]
  density: [DEN, RHOB, RHOZ, RHO]

wells:
  A1:
    primary:
      p_sonic: DTC
    disabled_curves: [DT_BAD]
  B2:
    force_category:
      RHOZ2: density
    primary:
      density: RHOZ2
```

### 主曲线选择

后续合成记录最低要求是：

- `p_sonic`
- `density`

如果一类里有多个候选，建议先按配置中的优先级选择 primary；例如：

```text
p_sonic: DT > DTC > DTCO > AC > VP
density: DEN > RHOB > RHOZ > RHO
```

没有 `p_sonic` 或没有 `density` 的井，`screen_status = failed`，不导入后续井震标定流程。

`screen_status = partial` 只用于报告：表示井里有一些有用曲线，但不满足最低要求。partial 井不写入 `selected_las`，避免第三步误处理注定不能进入合成记录的井。

### 单位与缩写

第二个脚本只记录原始 mnemonic、原始 unit 和曲线类别，不做最终单位规范化和标准缩写改名。

原因是单位规范化需要和数值分布 QC 结合起来看。例如同一条声波曲线既可能是慢度，也可能是速度，单位字段还可能写错。这个判断更适合放到第三步 `log_preprocess.py`，和异常值处理、单位错配 QC 一起完成。

第二个脚本也不计算 AI。它只负责筛选和导出输入曲线；AI 生成可以放到标准井资产准备或井震标定阶段。

### 井径状态

第二步完成曲线分类后，`has_caliper` 是 LAS 井径曲线的正式存在性标记：

- 如果井径曲线被规则、LLM cache、LLM 或人工 override 归为 `caliper`，则 `has_caliper = true`。
- 如果只有疑似井径名但分类歧义未解，则 `has_caliper = false`，原因写入 `reasons` 或 `skipped_curves.csv`。
- 如果有多条井径曲线，按 override 和全局优先级选 `primary_caliper`。

这里的 `caliper` 指 LAS 井径曲线，不等同于 `data/all_well_trace` 里的井轨迹/井斜文件。后续斜井路径应以第一步的 `has_well_trace` 和第四步的轨迹 QC 为准；`has_caliper` 只服务井眼质量和可选扩径段 QC。

### LAS 导出

导出瘦身 LAS 时建议：

- 保留原始 mnemonic，避免过早改名导致人工追踪困难。
- 在 `well_curve_screen.csv` 中记录每个类别的 primary mnemonic。
- 导出所有选中类别的曲线，而不是只导出 p_sonic/density。
- 同时写 `selected_curve_count` 和 `skipped_curves.csv`，避免“导出了文件但关键曲线失败”的假阳性。

是否另导出一份标准命名 LAS，可以留到后续资产准备脚本讨论。这里先保持职责简单。

## 模块边界

第二个脚本应该推动 `cup` 做一次井曲线相关重构。为了避免文件过碎，先按“对象/IO/流程”三类合并，等模块真正变大后再拆。

### 建议新增

`cup.well.curves`

- `CurveCategory`：稳定类别 key、展示名、候选 mnemonic、单位类型。
- `CurveInfo`：单条 LAS 曲线的 mnemonic、unit、descr、index。
- `CurveClassification`：曲线到类别的归类结果和来源。
- `CurveSelection`：每口井最终选中的曲线、primary 曲线、失败原因。
- `classify_curves_by_rules(curves, schema)`
- `merge_curve_classifications(rule_result, llm_result, overrides)`
- `select_primary_curves(classification, policy)`
- `screen_well_curves(...) -> CurveSelection`

`cup.well.las`

- `scan_las_header(path) -> LasHeader`
- `scan_las_curves(path) -> list[CurveInfo]`
- `extract_las_curve(las_file, mnemonic) -> grid.Log`
- `extract_selected_curves(path, selection) -> dict[str, grid.Log]`
- `export_curve_sets_to_las(curve_sets, output_dir, ...)`

`cup.well.las` 只做 LAS 读写和轻量结构扫描，不做曲线分类、单位转换或异常值处理。

### 建议调整

`src/cup/well/mnemonics.py`

- 去掉只给内部用的下划线命名，改成公开的 `CURVE_CATEGORY_MNEMONICS`。
- 补齐 `resistivity`、`sp`。
- 把“曲线类别”和“单位归一规则”分开，避免 mnemonics 模块承担业务逻辑。

`src/cup/petrel/load.py`

- `extract_any_log_from_las()` 更像通用 LAS 功能，不属于 Petrel 文本读取。建议移动到 `cup.well.las.extract_las_curve()`，原位置只保留兼容 wrapper 或直接在本轮重构中删除旧入口。
- `load_vp_rho_logset_from_las()` 也不适合继续放在 Petrel 模块，但不能只是机械移动。它现在混合了曲线选择、单位转换、插值和 `grid.LogSet` 构造。重构时应拆成：LAS 提取放 `cup.well.las`，单位转换放第三步 `cup.well.preprocess`，确认 `Vp/Rho` 后再显式构造 `grid.LogSet`。

`src/cup/petrel/export.py`

- `export_logsets_to_las()` 也是通用 LAS 导出能力，不属于 Petrel。建议移动到 `cup.well.las.export_curve_sets_to_las()`，Petrel 模块只保留 Petrel 专有文本导出。

### 脚本层保留

LLM 请求、API key、模型名、重试和缓存属于工作流脚本层，不建议放进 `cup` 核心模块。核心模块只接收“分类结果”，不依赖外部服务。

## 决策点

这些是第二个脚本真正需要拍板的点：

- LLM 是否首轮启用：建议默认 `false`，先跑规则分类和人工 override。
- 瘦身 LAS 是否保留原始 mnemonic：建议保留。
- 是否导出未通过 p_sonic/density 的部分井：不进入 `selected_las`，只在 `well_curve_screen.csv` 和 `skipped_wells.csv` 中完整记录。
- 多条同类曲线如何选 primary：建议配置化优先级，先不要让 LLM 决定 primary。
- 是否在本脚本计算 AI：建议不计算。
