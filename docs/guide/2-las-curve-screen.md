# 02 LAS 曲线筛选与导出

`las_curve_screen.py` 是工作流的第二步。它读取第一步的 `well_inventory.csv`，对工区内且有 LAS 的每口井扫描曲线头，用本地 mnemonic 规则把每条曲线归类，选出每类的 primary 曲线，最后把通过筛选的井导出瘦身 LAS。

---

## 快速开始

```bash
python scripts/las_curve_screen.py
python scripts/las_curve_screen.py --config experiments/my_project.yaml
python scripts/las_curve_screen.py --output-dir /tmp/screen_test
```

不带参数运行时，脚本读取 `experiments/common.yaml`，自动发现最新的 `well_inventory_*/well_inventory.csv`，在 `scripts/output/las_curve_screen_<timestamp>/` 下写出结果。

当前版本 LLM 分类默认**不启用**。仅凭本地 mnemonic 规则即可产出可审计的初版结果。如果配置里打开了 LLM，脚本会直接抛出 `NotImplementedError`——LLM 路径预留给后续迭代。

---

## 配置参考

```yaml
las_curve_screen:
  inventory_file: null                       # null = 自动发现最新 step1 输出
  las_dir: all_well_las                      # 原始 LAS 目录，相对于 data_root
  include_survey_positions: [inside, near_outside]
  required_categories: [p_sonic, density]    # 必须同时具备才能 passed
  selected_categories:                       # 需要从 LAS 中提取的类别
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
  curve_schema_file: null                    # null = 使用内置 CURVE_CATEGORY_MNEMONICS
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

### `inventory_file`

`null` 时脚本自动扫描 `<output_root>/well_inventory_*/well_inventory.csv`，取最新一份。如果要复现固定实验，显式填写第一步产出的路径。

### `required_categories`

必须同时具备的类别。当前默认 `[p_sonic, density]`——缺任一个则 `screen_status = failed` 或 `partial`（有其他曲线时为 partial，完全没有则 failed）。

### `selected_categories`

脚本会从 LAS 中提取这些类别的曲线，并各选一条 primary。不在这个列表里的曲线即使能被识别也不会被选中导出，但仍会出现在 `las_curve_inventory.csv` 的分类记录中。

这里的类别名是工作流内部的语义类别，不是 LAS 原始 mnemonic。比如 `spontaneous_potential` 是自然电位类别，匹配的常见 LAS mnemonic 是 `SP`。

### `curve_schema_file`

自定义分类规则的 YAML 文件。不填则使用 `cup.well.mnemonics.CURVE_CATEGORY_MNEMONICS` 内置规则。格式：

```yaml
categories:
  p_sonic:
    display: "纵波声波/时差"
    mnemonics: [DT, DTC, DTCO, AC, VP]
  my_custom_category:
    mnemonics: [ABC, XYZ]
```

### `curve_override_file`

人工干预配置，优先级高于规则和 LLM。支持四类操作：

```yaml
global_priority:                  # 覆盖全局 primary 选择优先级
  p_sonic: [DT, DTC, AC, VP]
  density: [RHOB, DEN, RHOZ]

wells:
  A1:                             # 单井配置（井名大小写不敏感）
    primary:                      # 单井单类别指定 primary
      p_sonic: DTC
    disabled_curves: [DT_BAD]    # 跳过该井的某些曲线
    force_category:               # 强制将某曲线归入某类别
      RHOZ2: density
```

- `global_priority`：定义每类中选 primary 的优先顺序。内置优先级（`CURVE_CATEGORY_PRIORITY`）仅含可信原始测井曲线，不含派生产品。
- `primary`：直接指定某口井某类用哪条曲线。它的值使用**精确 mnemonic**（含 LASIO 重复曲线后缀如 `:1`），不会做规范化裁剪。
- `disabled_curves`：跳过某些曲线，不参与分类和选择。支持含 LASIO 重复曲线后缀的精确名和基础名。填 `DT:1` 只禁用 `DT:1`；填 `DT` 会禁用 `DT`、`DT:1`、`DT:2` 等所有同名变体。下划线不是这里说的后缀，`DT_BAD` 是另一条独立 mnemonic，不会被 `DT` 禁用。
- `force_category`：覆盖规则判断，将一条曲线强制归入指定类别。曲线名的精确名/基础名规则与 `disabled_curves` 一致。

所有 override 均可追溯：分类结果中的 `classification_source` 字段会标记 `override`，`notes` 会记录具体原因。

---

## 曲线分类原理

### 本地 mnemonic 规则

内置规则在 `cup.well.mnemonics.CURVE_CATEGORY_MNEMONICS` 中维护。每条曲线按 mnemonic 规范化后（大写、去前后空格、裁剪 LASIO 的 `:1`/`:2` 这类重复曲线后缀）与规则表匹配：

- 匹配到**唯一**类别 → 直接归类，`classification_source = mnemonic_rule`
- 匹配到**多个**类别 → 标记为 `ambiguous`，`confidence = 0`，不进入 primary 选择。触发条件是同一个规范化 mnemonic 同时出现在多个类别规则中；内置规则会避免这种情况，通常只会在自定义 `curve_schema_file` 把同一个简称放进多个类别时出现，例如把 `GR` 同时放进 `gamma_ray` 和 `resistivity`
- 没匹配到任何类别 → 标记为 `unclassified`

### LASIO 后缀处理

lasio 读取 LAS 时，同名曲线可能被自动添加 `:1`、`:2` 等后缀。这里的“后缀”只指 mnemonic 尾部的冒号加数字，正则形式是 `:\d+$`；它不是下划线，也不是 `GR_NORM`、`DT_BAD` 这类名字的一部分。脚本区分两种 mnemonic 概念：

| 概念 | 函数 | 示例 |
|------|------|------|
| **精确名** | `exact_mnemonic()` | `CALIBRATEDSONICLOG:1` → `CALIBRATEDSONICLOG:1`（保留后缀） |
| **规范化名** | `normalize_mnemonic()` | `CALIBRATEDSONICLOG:1` → `CALIBRATEDSONICLOG`（裁剪后缀） |

分类匹配用的是规范化名（`CALIBRATEDSONICLOG:1` 和 `:2` 都归入 `p_sonic`）。Primary 选择优先用精确名匹配，规范化名作为兜底——这样你可以 override primary 到具体的 `CALIBRATEDSONICLOG:1` 而不会误选 `:2`。

### Primary 选择

对每个 `selected_categories` 中的类别，按以下顺序选出 primary：

1. 单井 `primary` override（精确匹配）→ 精确命中
2. 单井 `primary` override（规范化匹配）→ 多候选时取 index 最小者
3. `global_priority` 或 `CURVE_CATEGORY_PRIORITY` 的优先级顺序 → 第一个匹配者
4. 兜底：取 index 最小的候选曲线

选出的 primary mnemonic 是**精确名**（LAS 里实际出现的 mnemonic），可直接用于后续提取。

---

## 输出文件

脚本在 `<output_root>/las_curve_screen_<timestamp>/` 下生成：

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

### 2. `well_curve_screen.csv` — 每口井一行

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

只导出 `screen_status == passed` 的井，且必须经过 `_exported_contains_required` 校验——若导出后发现 required primary（如 DT、DEN）并未写入 LAS，该井会被降级为 `failed`，不产出 LAS。

导出内容：LAS 索引道 + 所有选出的 primary 曲线。保留原始 mnemonic（不做标准命名），NULL 值统一为 `-999.25`。

### 4. `curve_classification/*.json` — 逐井分类详情

每口候选井一份 JSON，包含 header 摘要、分类结果、primary 选择、reasons。用于断点复查和人工审计。

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
LAS curve screen summary: 61 candidates, 38 passed, 22 partial, 1 failed, 38 LAS exported.
```

四数之和应等于 candidates。如果 `partial` 比例很高（>30%），说明工区内许多井缺 density 或 p_sonic——需要检查是 LAS 数据本身缺失，还是 mnemonic 规则没覆盖。

### 第二步：看 `well_curve_screen.csv`

按 `screen_status` 分组查看：
- `passed` — 具备 p_sonic + density，已导出瘦身 LAS，进入第三步
- `partial` — 有一些有用曲线但不满足 required，不导出 LAS，不进入第三步
- `failed` — 完全缺少关键曲线，或 LAS 读取/导出失败

关注 `reasons` 列：`missing_p_sonic`、`missing_density`、`export_missing_required_p_sonic` 等标签能快速定位失败原因。

`has_caliper` 在第三步用于决定是否跳过井径曲线的连续常值段替换。

### 第三步：看 `las_curve_inventory.csv`

查询具体某口井的曲线分类详情。关注 `category == ambiguous` 的行——这些曲线命中多个类别，需要人工指定或补充分类规则。

### 第四步：抽查 `curve_classification/*.json`

对任何 `failed` 或 `partial` 的井，打开对应 JSON 查看完整的曲线头和分类判断，确认是数据本身缺失还是规则需要调整。

---

## 注意事项

- **LLM 分类第一版不启用。** 如果 `llm.enabled: true`，脚本会直接抛出 `NotImplementedError`。LLM 路径的接口已预留（classifications 可合并 LLM 结果），但请求逻辑和缓存未实现。
