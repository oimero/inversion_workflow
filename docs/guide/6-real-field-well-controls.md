# 06 真实工区井控数据集

`real_field_well_controls.py` 是工作流的第六步。它只做一件事：**把不同上游标定产物的波阻抗对数统一转换成相同的井控事实。**

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

必填。只能是 `well_auto_tie` 或 `wavelet_batch_synthetic_depth`。不能为空、不能缩写、不能自动推断。脚本不会根据目录名、CSV 字段或 LAS 文件名猜测适配器，与上游数据模式或采样域不一致时直接失败。

| 值 | 目标域 | 上游 |
|---|---|---|
| `well_auto_tie` | time + s | 第四步 `well_tie_metrics.csv`，每井 filtered LAS + 优化 TDT |
| `wavelet_batch_synthetic_depth` | depth + tvdss + m | 第五步（深度域） `wavelet_batch_metrics.csv`，每井 shifted filtered LAS |

时间域输入要求：

- 只接受第四步 `tie_status=success` 的井。
- 每井必须有 filtered LAS（含 AI 曲线，单位 `m/s*g/cm3`）和优化 TDT 表。
- 斜井还需要优化轨迹采样计划文件。

深度域输入要求：

- 只接受第五步（深度域）`status=ok` 的井。
- 每井必须有深度平移后的过滤 LAS 路径。其纵轴仍按 MD 解释——只是被平移到了更好的深度位置，坐标体系不变。
- 斜井需要项目井轨迹文件（Petrel 导出），用于 MD→TVDSS 转换；直井走 `md - kb` 公式，不需要轨迹文件。

### `source_run_dir`

指向第四步（时间域）或第五步（深度域）的运行目录。留空时脚本做模式感知自动发现：按来源运行类型前缀在输出目录下找到最新的、数据模式和采样域匹配的成功运行。显式填路径则固定使用该目录，适合复现。

### `well_inventory_file`

指向第一步产出的 `well_inventory.csv`。脚本从中读取每口井的井型（直井或斜井）、井口坐标、线号和道号，以及补心海拔。

### `well_trace_dir`

深度域斜井需要的井轨迹目录。时间域直井不需要，但配置段必须存在（可为空路径）。

---

## 脚本在做什么

脚本的核心任务很简单：**把上游 LAS 里的波阻抗对数，从测深域投影到目标地震的采样轴上，并为每个有效样点附上空间坐标。** 分三个阶段：适配 → 域转换 → 写入。

### 第一阶段：适配

1. 读取上游 `run_summary.json`，确认 schema、domain、status 匹配当前配置的 `source_run_type`。
2. 读取指标 CSV，只保留状态成功的井。
3. 读取 `well_inventory.csv`，与上游成功井做井名匹配。上游成功但 inventory 中不存在的井被拒绝。

### 第二阶段：域转换

这是脚本的主体。时间域和深度域走不同的转换路径，但目标一致——把 MD 域的 ln(AI) 映射到地震采样轴上，同时不跨越 LAS 数据中的空值缺口。

**时间域：**

对每口井，先通过优化 TDT 表把每个 TWT 采样点映射到 MD 轴上的一个位置，再从 LAS 的 MD 轴上读出该位置的 ln(AI) 值。空间位置方面，直井直接把井口的固定线号道号广播到所有样点；斜井从第四步产出的轨迹采样计划读取逐样点的线号、道号和 XY。

**深度域：**

更简单——没有 TDT 投影这个环节。直井直接用"测深减补心海拔"把 LAS 的 MD 轴转成 TVDSS 轴，在 TVDSS 上对齐到地震深度采样轴。斜井通过项目轨迹文件做 MD→TVDSS 映射，再从 TVDSS 反查到采样轴上。空间位置同样由井型决定：直井广播固定井位，斜井从轨迹插值 XY 再换算线号道号。

两条路径共享一个关键约束：**LAS 中连续有限段之间的空值间隙不会被填补**。每个连续数据段独立插值，缺口处严格保留空值。这保证了井控事实不引入人为插值。

### 第三阶段：校验与写入

转换完成后，脚本对每口成功井做几何一致性校验——用测网几何把真实 XY 反算线号，与记录的线号道号逐点对照，不一致的井被拒绝。

然后写入三类产物：

1. **逐井 NPZ。** `wells/<well>.npz`，固定包含采样轴、ln(AI)、线号、道号、XY、有效掩码和元数据 JSON。无效样点对应数组值为 NaN。
2. **Manifest CSV。** 每口候选井一行，记录状态、井型、采样模式、有效样点数和 NPZ 路径。失败的井也保留行，但 NPZ 路径为空。
3. **运行摘要 JSON。** 记录来源适配器、采样轴、上游契约指纹、井数统计和产物路径。

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
