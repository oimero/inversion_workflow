# 02 LAS 曲线筛选与导出

`well_screen.py` 是工作流的第二步。它读取第一步的井资产清单，按配置的位置范围筛选候选井，用本地曲线名规则分类并选择各类别的代表曲线，最后为通过筛选的井导出精简 LAS。

---

## 快速开始

```bash
python scripts/well_screen.py
python scripts/well_screen.py --config experiments/<project>.yaml
python scripts/well_screen.py --output-dir <OUTPUT_DIR>
```

不带参数运行时，脚本读取 `experiments/common/common.yaml`，自动发现配置中 `output_root` 下最新的井资产盘点结果，并在 `output_root/well_screen_<timestamp>/` 下写出结果；未配置 `output_root` 时使用 `scripts/output`。复现实验时可以在 `source_runs.well_inventory_dir` 中指定固定的上游目录。

当前版本只使用可复现的本地曲线名规则和人工 override。

---

## 运行前需要什么

| 输入 | 用途 |
|------|------|
| `well_inventory.csv` | 确定候选井、LAS 是否存在、井口是否在工区内 |
| 原始 LAS 目录 | 读取曲线头、分类 mnemonic，并导出瘦身 LAS |
| 可选曲线规则 YAML | 覆盖或扩展内置 mnemonic 分类规则 |
| 可选人工 override YAML | 指定 primary、禁用坏曲线、强制类别 |

只有 `screen_status == passed` 的井会导出 `selected_las/*.las` 并进入第三步。

---

## 配置参考

```yaml
data_root: <DATA_ROOT>
output_root: <OUTPUT_ROOT>

assets:
  las_dir: <LAS_DIRECTORY>

well_curves:
  required_categories: [p_sonic, density]
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

well_screen:
  candidate_filter:
    include_survey_positions: [inside, near_outside]

  classification:
    curve_schema_file: null                  # null = 使用内置分类规则
    curve_override_file: "<curve-override-file>"
```

### `source_runs`

默认自动接上最新一次井资产盘点结果，因此常用配置不显示 `source_runs`。复现实验时可按需加入 `source_runs.well_inventory_dir` 固定输入。

### `well_curves`

#### `required_categories`

决定一口井能不能进入后续井震流程。当前要求同时找到纵波声波和密度；缺少任意一个，就不会导出给第三步使用的瘦身 LAS。

#### `selected_categories`

决定第二步重点关心哪些曲线类别。脚本会为列表中的每个类别尽量选出一条代表曲线；不在列表中的曲线即使能识别，也只留在分类明细里，不进入导出的瘦身 LAS。

### `classification`

#### `curve_schema_file`

自定义分类规则的 YAML 文件。不填则使用内置的曲线名称分类规则。格式：

```yaml
categories:
  p_sonic:
    display: "纵波声波/时差"
    mnemonics: [DT, DTC, DTCO, AC, VP]
  my_custom_category:
    mnemonics: [ABC, XYZ]
```

#### `curve_override_file`

人工干预入口，用来处理规则无法可靠判断的井或曲线。它的优先级高于内置规则，适合固定某口井的 primary、跳过坏曲线，或把项目里特殊命名的曲线强制归类。

```yaml
global_priority:                  # 覆盖全局 primary 选择优先级
  p_sonic: [DT, DTC, AC, VP]
  density: [RHOB, DEN, RHOZ]

global_force_category:            # 工区级特殊 mnemonic → 标准类别
  <custom-sonic-curve>: p_sonic
  <custom-density-curve>: density

wells:
  <well-name>:                    # 单井配置（井名大小写不敏感）
    primary:                      # 单井单类别指定 primary
      p_sonic: DTC
    disabled_curves: [DT_BAD]    # 跳过该井的某些曲线
    force_category:               # 强制将某曲线归入某类别
      <curve-name>: density
```

- `global_priority`：调整每类曲线的优先顺序，例如优先选原始测井曲线，少选派生产品。
- `global_force_category`：为当前项目统一命名的特殊曲线名补充分类，不改变默认分类规则。
- `primary`：指定某口井某一类必须使用哪条曲线。
- `disabled_curves`：跳过明确知道有问题的曲线。
- `force_category`：把特殊命名但含义明确的曲线强制归入指定类别。

分类结果中的 `classification_source` 字段会标记 `override`，`notes` 会记录具体原因。

---

## 脚本在做什么

1. **筛选候选井。** 从资产清单中选择同时具备井头和测井文件，且工区位置与资产状态满足要求的井。
2. **识别曲线类别。** 读取测井文件头，依据曲线名称和可选人工规则，将曲线归入声波、密度、井径等类别。
3. **选择代表曲线。** 同一类别有多条候选时，依次使用单井指定、全局优先顺序、内置优先顺序和原文件顺序作出选择。类别不明确或被人工禁用的曲线不参与选择。
4. **判断井是否通过。** 检查所选曲线是否覆盖全部必需类别，记录通过、部分满足或失败的状态及原因。
5. **导出精简测井文件。** 对通过的井保留各类别的代表曲线，并检查必需曲线是否确实写入导出文件。

---

## 曲线分类原理

### 本地曲线名规则

内置规则维护了一份“常见曲线名 → 语义类别”的字典。每条曲线会先做轻量规范化，例如统一大小写、去掉空格、裁剪 LASIO 自动添加的重复曲线后缀，然后再与规则表匹配：

- 匹配到**唯一**类别 → 直接归类，`classification_source = mnemonic_rule`
- 匹配到**多个**类别 → 标记为 `ambiguous`，`confidence = 0`，不进入 primary 选择。触发条件是同一个规范化曲线名同时出现在多个类别规则中；内置规则会避免这种情况
- 没匹配到任何类别 → 标记为 `unclassified`

### LASIO 后缀处理

lasio 读取 LAS 时，同名曲线可能被自动添加 `:1`、`:2` 等后缀。

脚本区分两种曲线名概念：

| 概念 | 处理方式 | 示例 |
|------|------|------|
| **精确名** | 保留 LAS 中的实际后缀 | `CALIBRATEDSONICLOG:1` → `CALIBRATEDSONICLOG:1` |
| **规范化名** | 去除自动添加的后缀 | `CALIBRATEDSONICLOG:1` → `CALIBRATEDSONICLOG` |

分类匹配用的是规范化名（`CALIBRATEDSONICLOG:1` 和 `:2` 都归入 `p_sonic`）。Primary 选择优先用精确名匹配，规范化名作为兜底——这样你可以 override primary 到具体的 `CALIBRATEDSONICLOG:1` 而不会误选 `:2`。

### Primary 选择

每个类别可能命中多条曲线。第二步会按以下顺序选出一条代表曲线：

1. 单井 `primary` override（精确匹配）→ 精确命中
2. 单井 `primary` override（规范化匹配）→ 多候选时取 index 最小者
3. 全局配置或内置规则的优先级顺序 → 第一个匹配者
4. 兜底：取 index 最小的候选曲线

选出的 primary 曲线名是**精确名**（LAS 里实际出现的曲线名），可直接用于后续提取。

---

## 核心输出文件

脚本在 `<output_root>/well_screen_<timestamp>/` 下生成：

### 1. `las_curve_inventory.csv` — 每条 LAS 曲线一行

| 字段 | 含义 |
|------|------|
| `well_name` | 井名 |
| `mnemonic` | 原始 LAS 曲线名（精确名） |
| `unit` | 原始单位 |
| `description` | LAS 描述文本 |
| `category` | 标准类别 key，或 `unclassified`/`ambiguous`/`disabled` |
| `is_primary` | 是否为该类别的 primary |
| `classification_source` | `override`、`mnemonic_rule`、`unclassified` |
| `confidence` | 规则分类为 1.0，ambiguous 为 0.0 |
| `notes` | 歧义说明或 override 原因 |

### 2. `well_screen.csv` — 每口井一行

| 字段 | 含义 |
|------|------|
| `well_name` | 井名 |
| `las_file` | 原始 LAS 路径（repo-relative） |
| `screen_status` | `passed`、`partial`、`failed` |
| `has_p_sonic` | 是否有 p_sonic 的 primary |
| `has_density` | 是否有 density 的 primary |
| `has_caliper` | 是否有 caliper 的 primary |
| `primary_p_sonic` | 选定的 p_sonic 精确 mnemonic |
| `primary_density` | 选定的 density 精确 mnemonic |
| `primary_caliper` | 选定的 caliper 精确 mnemonic |
| `selected_curve_count` | 成功选出 primary 的曲线数 |
| `exported_las` | 导出 LAS 路径（repo-relative）；failed/partial 时为空 |
| `reasons` | 分号分隔的失败或警告原因 |

### 3. `selected_las/*.las` — 瘦身 LAS

只导出 `screen_status == passed` 的井，并在导出后再次检查必需的 primary 曲线（如 DT、DEN）是否写入 LAS；缺失时该井标记为 `failed`，不产出 LAS。

导出内容：LAS 索引道 + 所有选出的 primary 曲线。保留原始曲线名（不做标准命名），NULL 值统一为 `-999.25`。

### 4. `curve_classification/*.json` — 逐井分类详情

每口候选井一份 JSON，包含 header 摘要、分类结果、primary 选择、reasons。用于断点复查和人工复核。

### 5. `skipped_wells.csv`、`skipped_curves.csv`、`run_summary.json`

| 文件 | 内容 |
|------|------|
| `skipped_wells.csv` | 未通过筛选的井及原因 |
| `skipped_curves.csv` | 分类为 ambiguous/disabled、或导出时缺失的曲线 |
| `run_summary.json` | 输入路径、候选井数、各状态计数、LAS 导出数、分类来源分布 |

---

## 如何阅读结果

### 第一步：看终端输出

```
LAS curve screen summary: <CANDIDATE_COUNT> candidates, <PASSED_COUNT> passed, <PARTIAL_COUNT> partial, <FAILED_COUNT> failed, <EXPORTED_COUNT> LAS exported.
```

四数之和应等于 candidates。如果 `partial` 比例很高（>30%），说明工区内许多井缺 density 或 p_sonic——需要检查是 LAS 数据本身缺失，还是曲线名规则没覆盖。

### 第二步：看 `well_screen.csv`

按 `screen_status` 分组查看：

- `passed` — 具备 p_sonic + density，已导出瘦身 LAS，进入第三步
- `partial` — 有一些有用曲线但不满足 required，不导出 LAS，不进入第三步
- `failed` — 完全缺少关键曲线，或导出后的必需曲线契约校验失败

LAS 文件缺失、解析异常或导出异常属于运行错误，脚本会直接抛出，不会降级成某口井的 `failed` 结果。

关注 `reasons` 列：`missing_p_sonic`、`missing_density`、`export_missing_required_p_sonic` 等标签能快速定位失败原因。

`has_caliper` 在第三步用于决定是否跳过井径曲线的连续常值段替换。

### 第三步：看 `las_curve_inventory.csv`

查询具体某口井的曲线分类详情。关注 `category == ambiguous` 的行——这些曲线命中多个类别，需要人工指定或补充分类规则。

### 第四步：抽查 `curve_classification/*.json`

对任何 `failed` 或 `partial` 的井，打开对应 JSON 查看完整的曲线头和分类判断，确认是数据本身缺失还是规则需要调整。
