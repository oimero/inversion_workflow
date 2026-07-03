# 06 真实井控制准备

`real_field_well_controls.py` 是工作流的第六步。它只做一件事：**把不同上游标定产物的波阻抗对数统一转换成相同的井控事实，冻结为与任何低频模型无关的标准井控集。**

这一步是整个统一低频模型架构的根基。第七步的所有基线模型和修饰器只消费同一种井控集接口，永远不需要知道井控来自时间域标定还是深度域合成旁路。

---

## 快速开始

```powershell
python scripts/real_field_well_controls.py
python scripts/real_field_well_controls.py --config experiments/my_project.yaml
python scripts/real_field_well_controls.py --output-dir scripts/output/well_controls_test
```

不带 `--output-dir` 时，脚本在输出目录下自动创建 `real_field_well_controls_<timestamp>/`。已有目录会被拒绝，避免覆盖历史运行。

---

## 运行前需要什么

第六步依赖上游标定产物和井资产信息。它需要多少东西取决于你用的是时间域还是深度域。

| 来源 | 文件 | 用途 |
|------|------|------|
| 第四步（时间域）或第五步（深度域） | `run_summary.json` | schema/domain 校验和直接上游契约身份 |
| 第四步（时间域）或第五步（深度域） | 指标 CSV | 成功井清单和产物路径 |
| 上游 LAS | 每井 filtered LAS 或 shifted filtered LAS | `AI [m/s*g/cm3]` 曲线 |
| 上游转换表 | 每井优化 TDT | MD→TWT 映射（仅时间域） |
| 上游轨迹计划 | 每井 trace sample plan | 斜井逐样点 inline/xline/XY（仅时间域斜井） |
| 第一步 | `well_inventory.csv` | 井口坐标、KB 高程、井型 |
| 项目资产 | 井轨迹文件 | MD→TVDSS 映射（仅深度域斜井） |
| 数据目录 | 地震体 | 目标 SampleAxis 和 survey geometry |

### Source adapter 概念

配置必须显式声明一种来源运行类型，它同时决定了脚本从上游读取哪些字段、如何做域转换、以及如何校验数据模式：

| `source_run_type` | 目标域 | 上游 |
|---|---|---|
| `well_auto_tie` | time + s | 第四步 `well_tie_metrics.csv`，每井 filtered LAS + 优化 TDT |
| `wavelet_batch_synthetic_depth` | depth + tvdss + m | 第五步（深度域） `wavelet_batch_metrics.csv`，每井 shifted filtered LAS |

脚本不会根据目录名、CSV 某个字段或 LAS 文件名猜测适配器。适配器与上游运行摘要的数据模式或采样域不一致时直接失败。

### 时间域特有要求

- 只接受第四步 `tie_status=success` 的井。
- 每井必须有 filtered LAS（含 AI 曲线，单位 `m/s*g/cm3`）和优化 TDT 表。
- 斜井还必须有优化轨迹采样计划文件，提供每个双程旅行时采样点的线号、道号和平面坐标，以及测线位置标记。

### 深度域特有要求

- 只接受第五步（深度域）`status=ok` 的井。
- 每井必须有深度平移后的过滤 LAS 路径。
- shifted filtered LAS 的纵轴仍按 MD 解释——它只是被平移到更好的深度位置，坐标体系不变。
- 斜井需要项目井轨迹文件（Petrel 导出），用于 MD→TVDSS 转换。
- 直井走测深减补心海拔公式，不需要轨迹文件。

---

## 配置参考

```yaml
workflow_config: experiments/common/common.yaml

real_field_well_controls:
  source_run_type: well_auto_tie
  source_run_dir: scripts/output/well_auto_tie_<timestamp>
  well_inventory_file: scripts/output/well_inventory_<timestamp>/well_inventory.csv
  well_trace_dir: all_well_trace
```

### `source_run_type`

必填。只能是 `well_auto_tie`（时间域）或 `wavelet_batch_synthetic_depth`（深度域）。不能为空、不能缩写、不能自动推断。

### `source_run_dir`

指向第四步（时间域）或第五步（深度域）的运行目录。留空时脚本做模式感知自动发现：按来源运行类型前缀在输出目录下找到最新的、数据模式和采样域匹配的成功运行。显式填路径则固定使用该目录，适合复现。

### `well_inventory_file`

指向第一步产出的 `well_inventory.csv`。脚本从中读取每口井的井型（直井或斜井）、井口坐标、线号和道号，以及补心海拔。

### `well_trace_dir`

深度域斜井需要的井轨迹目录。时间域直井不需要，但配置段必须存在（可为空路径）。

---

## 脚本在做什么

脚本的核心逻辑只有一层：**逐井做域转换，把上游 LAS 的 MD 域波阻抗对数映射到目标地震的采样轴上，同时为每个有效样点附上空间坐标。**

### 第一步：校验来源

1. 读取上游 `run_summary.json`，校验 schema 版本、domain、depth basis 和 status。
2. 读取指标 CSV（时间域 `well_tie_metrics.csv`，深度域 `wavelet_batch_metrics.csv`），只保留状态成功的井。
3. 读取 `well_inventory.csv`，与指标 CSV 做井名规范化匹配。上游成功但不在 inventory 中的井被拒绝。

### 第二步：域转换（时间域）

对每口时间域井：

1. **读取 filtered LAS。** 校验 AI 曲线存在且单位为 `m/s*g/cm3`，AI 值全为正。对原值取自然对数得到 MD 域的波阻抗对数。
2. **用优化 TDT 做 MD→TWT 投影。** 从 TDT 表中读取测深列和双程旅行时列，插值得到每个目标 TWT 采样点对应的 MD。插值不做外推——目标 TWT 落在 TDT 覆盖范围外即标记无效。
3. **在 MD 域逐连续有限段插值。** 这是关键细节：LAS 的有限段之间可能有空值间隙。脚本不会跨缺口插值——每个连续有限段独立处理，缺口处严格输出空值。这保证了"井控事实不补洞"的契约。
4. **处理空间位置。** 直井把井资产盘点中的固定线号、道号和 XY 广播到全部样点。斜井从轨迹采样计划读取每个 TWT 的逐样点线号、道号和 XY，只取测线位置在工区内的行，工区外的行置空值后逐有限段独立插值——避免把工区内外之间的间隙错误连接。

### 第三步：域转换（深度域）

对每口深度域井：

1. **读取 shifted filtered LAS。** 同上校验 AI 单位和正值。
2. **做 MD→TVDSS 转换。** 直井用"亚海真垂深等于测深减补心海拔"。斜井从项目轨迹文件加载亚海真垂深列和测深列，插值得到 shifted LAS MD 对应的 TVDSS，再用 TVDSS 反查到目标采样轴的 MD。
3. **空间定位。** 直井把井资产盘点位置广播到所有样点。斜井从轨迹中插值出每个样点的 XY，再通过测网几何换算为线号和道号。如果某个样点的 XY 换算后与测网线网不一致，该样点标记无效。
4. **逐有限段插值。** 同样遵守不跨缺口原则。

### 第四步：构建 WellControl 并校验

每个成功井生成一个 `WellControl` 内存对象：

| 字段 | 含义 |
|------|------|
| `log_ai` | 对齐到目标 SampleAxis 的 ln(AI)，`grid.Log` 对象 |
| `inline_by_sample` / `xline_by_sample` | 每个样点的地震线号 |
| `x_m_by_sample` / `y_m_by_sample` | 每个样点的真实米制 XY |
| `valid_mask` | 布尔数组，标记 log_ai 和四个位置数组同时有限的样点 |
| `sampling_mode` | 时间直井 `vertical_inventory_position`，时间斜井 `optimized_trace_plan`，深度直井 `vertical_md_minus_kb`，深度斜井 `trajectory_tvdss` |
| `provenance` | 来源 LAS 与转换表路径；其发布 run 身份由 summary 的 `input_contracts` 表达 |

构建时还会做几何一致性校验：对每个有效样点，用测网几何把真实 XY 反算线号，与记录的线号/道号逐点对照。不一致的井被拒绝。

### 第五步：写入磁盘

1. **逐井 NPZ。** `wells/<normalized_well_name>.npz`，固定字段为 `samples`、`log_ai`、`inline`、`xline`、`x_m`、`y_m`、`valid_mask`、`metadata_json`。无效样点的波阻抗对数和位置为空值。禁止使用 pickle 序列化或对象数组。
2. **Manifest。** `well_control_manifest.csv`，每口候选井一行，记录状态、来源路径、井型、采样模式、有效样点数和 NPZ 路径等。失败的井也保留行，但 `well_npz_path` 为空。
3. **运行摘要。** `run_summary.json`，数据模式固定为 `real_field_well_controls_v3`，记录来源适配器、采样轴、直接上游契约、成功/失败井统计、产物路径和唯一契约指纹。

这些属于 variant-specific label preparation（Step 7 之后的下游），与井控事实无关。

---

## 核心输出文件

```text
real_field_well_controls_<timestamp>/
├── run_summary.json
├── well_control_manifest.csv
└── wells/
    ├── <well_a>.npz
    └── <well_b>.npz
```

### `well_control_manifest.csv`

每口候选井一行，关键列：

| 列 | 含义 |
|------|------|
| `well_name` | 规范化井名 |
| `status` | `ok` 或 `failed` |
| `reason` | 失败原因（成功时为空） |
| `source_run_type` | `well_auto_tie` 或 `wavelet_batch_synthetic_depth` |
| `wellbore_class` | `vertical` 或 `deviated` |
| `sampling_mode` | 具体采样方式 |
| `n_samples` / `n_valid_samples` | 总样点数 / 有效样点数 |
| `well_npz_path` | NPZ 路径（失败时为空）；消费者不再重算逐井文件哈希 |

`run_summary.json` 使用 `real_field_well_controls_v3`，通过 `input_contracts` 记录直接上游，并只发布一个 `contract_fingerprint_sha256`。

### `wells/<well_name>.npz`

每井固定包含八个数组：

| 键 | dtype | 形状 | 含义 |
|------|------|------|------|
| `samples` | float64 | [N] | SampleAxis 采样值 |
| `log_ai` | float32 | [N] | ln(AI)，无效处为 NaN |
| `inline` | float64 | [N] | 逐样点 inline 线号 |
| `xline` | float64 | [N] | 逐样点 xline 线号 |
| `x_m` | float64 | [N] | 逐样点 X 米制坐标 |
| `y_m` | float64 | [N] | 逐样点 Y 米制坐标 |
| `valid_mask` | bool | [N] | 有效掩码 |
| `metadata_json` | 标量字符串 | — | 井名、schema、provenance |

### `run_summary.json`

记录业务配置、来源适配器、采样轴描述、成功/失败计数、产物路径、直接上游契约和唯一契约指纹。

---

## 如何阅读结果

### 第一步：看终端输出

```
=== Real-field Well Controls ===
Output: scripts/output/real_field_well_controls_<timestamp>
Successful wells: 12
```

成功井数至少为 1 即表示第六步完成。具体哪些基线模型能建模是第七步的事。

### 第二步：看 `well_control_manifest.csv`

按 `status` 列分组：

- **ok 井：** 关注有效样点数。有效样点数远少于总样点数说明该井的 LAS 或 TDT/轨迹覆盖范围与目标采样轴重叠有限。这在目标窗口边缘是正常的；如果一口井的有效覆盖率异常低，检查上游 LAS 的深度范围或 TDT 表的时间范围。
- **failed 井：** 看原因列。常见原因见下一节。

### 第三步：抽查一口井的 NPZ

如果你需要确认某口井的域转换是否正确，可以直接加载它的 NPZ：

- 检查有效掩码对应的波阻抗对数值范围是否合理（波阻抗对数值通常在 8~10 左右，对应线性 AI 约 3000~22000 m/s*g/cm3）。
- 检查直井的线号和道号是否为常数，斜井是否随样点变化。

---

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| schema_version 不匹配 | 上游 run summary 不是 v2 schema | 用当前版脚本重建上游 run |
| source adapter/domain 不一致 | `source_run_type` 与上游 summary 的 domain 不匹配 | 时间域用 `well_auto_tie`，深度域用 `wavelet_batch_synthetic_depth` |
| AI 单位不是 `m/s*g/cm3` | LAS 中 AI 曲线单位错误或缺失 | 检查上游 LAS 导出配置 |
| AI 包含非正值 | LAS 中有零或负的 AI 值 | 检查上游测井曲线质量 |
| TDT 或 trajectory 缺失 | 时间域井缺优化 TDT，或深度域斜井缺项目轨迹 | 补充对应文件 |
| inventory 行缺失 | 上游成功的井在 well_inventory.csv 中找不到 | 重新运行第一步或检查井名是否变化 |
| XY 与线号不一致 | 井的 physical XY 与 survey geometry 反算的线号不匹配 | 检查 inventory 中井口坐标或斜井轨迹是否正确 |
| 有效样点为零 | 井的 LAS 覆盖范围与目标 SampleAxis 完全不重叠 | 检查目标窗口是否设得合理 |

---

## 留到第二轮

- 是否支持时间域和深度域之外的第三种 source adapter（如直接从 Petrel 导出的 TWT 域 LAS）。
- 是否在 manifest 中增加逐井目标窗口覆盖率的百分比指标。
- 是否对斜井轨迹做更精细的 QC（如轨迹点间距检查、狗腿度告警）。
- 是否在 Step 6 中直接输出每井的 TDT/TVDSS 转换质量图。
