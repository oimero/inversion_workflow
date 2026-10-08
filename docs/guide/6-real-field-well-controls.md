# 06 真实工区井控数据集

`real_field_well_controls.py` 是工作流的第六步。本文按时间域工区说明：脚本直接沿用第四步井震标定为每口井选出的滤波参数和曲线，将已标定的滤波曲线对齐到地震时间轴，同时保留原生测井样点及其对应的双程旅行时坐标。

---

## 快速开始

```powershell
python scripts/real_field_well_controls.py
python scripts/real_field_well_controls.py --config experiments/<project>.yaml
python scripts/real_field_well_controls.py --output-dir <output_dir>
```

不带 `--output-dir` 时，脚本在配置的输出根目录下自动创建 `real_field_well_controls_<run_timestamp>/`。已有目录会被拒绝。

---

## 运行前需要什么

| 来源 | 文件 | 用途 |
|------|------|------|
| 来源运行 | `run_summary.json` | schema/domain 校验和直接上游契约身份 |
| 时间域第四步 | `well_tie_metrics.csv` | 成功井清单、每井已标定滤波后的 LAS 和优化 TDT 路径 |
| 时间域第四步 | 每井已标定滤波后的 LAS | 原生 `AI [m/s*g/cm3]` 曲线 |
| 时间域第四步 | 每井优化 TDT | MD→TWT 映射 |
| 时间域第四步 | 每井 trace sample plan | 斜井逐样点 inline/xline/XY（仅斜井） |
| 第一步 | `well_inventory.csv` | 井口坐标、KB 高程、井型 |
| 数据目录 | 时间域地震体 | 目标采样轴和工区几何 |

---

## 配置参考

```yaml
workflow_config: experiments/<workflow_config>.yaml

real_field_well_controls:
  source_run_type: well_auto_tie
  source_run_dir: scripts/output/<source_run_dir>
  well_inventory_file: scripts/output/<well_inventory_run_dir>/well_inventory.csv
```

### `source_run_type`

时间域填写 `well_auto_tie`。该值必须明确填写，不能缩写或自动推断。

| 值 | 目标域 | 上游 |
|---|---|---|
| `well_auto_tie` | time + s | 第四步成功井、滤波后的 LAS 和优化 TDT |

输入要求：

- 只接受第四步 `tie_status=success` 的井。
- 每井必须有滤波后的 LAS（含 AI 曲线，单位 `m/s*g/cm3`）和对应的域转换信息。
- 斜井需要第四步生成的优化轨迹采样计划。

### `source_run_dir`

指向当前 `source_run_type` 对应的上游运行目录。留空时脚本按来源前缀在输出根目录下查找最近一次包含所需文件且通过域和状态检查的运行。显式填路径则固定使用该目录。

### `well_inventory_file`

指向第一步产出的 `well_inventory.csv`。脚本从中读取每口井的井型（直井或斜井）、井口坐标、线号和道号，以及补心海拔。

### 缺口

缺口处理由上游滤波后的 LAS 负责。第六步只在各个有限段内投影到目标采样轴，不跨缺口插值，也不在模型轴上重新填补。观测支撑掩码表示投影后的上游有效支撑，缺口样点在模型轴阻抗字段和掩码中保持无效。

---

## 脚本在做什么

1. **确定来源与井清单。** 检查第四步运行的数据格式、时间域和成功状态，按规范化井名与井资产盘点结果匹配。缺少盘点记录的井进入失败记录。
2. **转换测井坐标。** 读取每井已标定滤波后的声阻抗并取自然对数，使用优化时深关系把原生测深样点映射为双程旅行时，保留原生曲线与有效性信息。
3. **对齐到地震时间轴。** 在原生曲线的连续有效段内插值到地震时间样点。超出覆盖范围和曲线缺口的位置保持无效。
4. **确定逐样点位置。** 直井使用固定井口坐标；斜井将优化轨迹采样计划中的位置插值到地震时间轴，仅使用计划中位于工区内的连续有效段。
5. **建立有效性掩码。** 分别记录原生曲线和时间轴曲线的有效性。时间轴上的观测支撑来自上游有效曲线段，最终有效样点还要求阻抗值和空间位置均为有限值。
6. **核对几何并写出。** 由米制坐标反算线号，逐点检查其与已记录线号的一致性，写出逐井数据、井控清单和运行摘要。

---

## 核心输出文件

```text
real_field_well_controls_<run_timestamp>/
├── run_summary.json
├── well_control_manifest.csv
├── qc/
│   ├── manifest.json
│   └── evaluation_support.json
├── wells/
│   ├── <well_name_a>.npz
│   └── <well_name_b>.npz
```

### `well_control_manifest.csv`

每口候选井一行，关键列：

| 列 | 含义 |
|------|------|
| `well_name` | 规范化井名 |
| `status` | `ok` 或 `failed` |
| `reason` | 失败原因（成功时为空） |
| `source_run_type` | `well_auto_tie` |
| `wellbore_class` | `vertical` 或 `deviated` |
| `sampling_mode` | 具体采样方式 |
| `n_samples` / `n_valid_samples` | 总样点数 / 有效样点数 |
| `n_observed_samples` / `n_interpolated_samples` | 模型轴上由输入曲线投影得到的有效样点数 / 有效但未标记为观测的样点数（当前实现为 0） |
| `n_native_samples` / `n_valid_native_samples` | 滤波曲线原生采样总样点数 / 有效样点数 |
| `well_npz_path` | NPZ 路径（失败时为空） |

`run_summary.json` 使用 `real_field_well_controls_v7`，记录原生已标定滤波曲线来源和缺口处理方式，并保存第六步固定评价支撑、直接上游与产物路径。

`qc/evaluation_support.json` 固定每口井用于后续井标签和波形质检的共同有效区间；它由目标层位、已标定滤波曲线、地震样点和空间位置共同确定。

第七步和[第八步物理约束神经网络反演](8-ginn.md)读取这一版本的逐井文件。生成井控后，将下游配置中的井控目录指向本次输出，并基于这份井控生成相应的低频模型。

### `wells/<well_name>.npz`

每井固定包含模型轴、原生井轴和元数据字段：

| 键 | dtype | 形状 | 含义 |
|------|------|------|------|
| `samples` | float64 | [N] | 地震时间采样值，单位为秒 |
| `model_grid_filtered_log_ai` | float32 | [N] | 目标采样轴上的滤波 LAS ln(AI)，无效处为 NaN |
| `inline` | float64 | [N] | 逐样点 inline 线号 |
| `xline` | float64 | [N] | 逐样点 xline 线号 |
| `x_m` | float64 | [N] | 逐样点 X 米制坐标 |
| `y_m` | float64 | [N] | 逐样点 Y 米制坐标 |
| `valid_mask` | bool | [N] | 有效掩码 |
| `observed_valid_mask` | bool | [N] | 未经缺口内插的观测支撑 |
| `native_coordinates` | float64 | [M] | 对齐后的原生 TWT 坐标 |
| `native_filtered_log_ai` | float32 | [M] | 滤波 LAS 原生采样的 ln(AI)，无效处为 NaN |
| `native_valid_mask` | bool | [M] | 滤波 LAS 原生采样曲线有效掩码 |
| `metadata_json` | 标量字符串 | — | 井名、schema、provenance |

### `run_summary.json`

记录业务配置、来源类型、采样轴描述、成功/失败计数和产物路径。

---

## 如何阅读结果

### 第一步：看终端输出

```
=== Real-field Well Controls ===
Output: scripts/output/real_field_well_controls_<run_timestamp>
Successful wells: <successful_well_count>
```

成功井数打印后，第六步完成。

### 第二步：看 `well_control_manifest.csv`

按 `status` 列分组：

- **ok 井：** 关注有效样点数。有效样点数远少于总样点数说明该井的 LAS 或 TDT/轨迹覆盖范围与目标采样轴重叠有限。这在目标窗口边缘是正常的；如果一口井的有效覆盖率异常低，检查上游 LAS 的深度范围或 TDT 表的时间范围。
- **failed 井：** 看原因列。常见原因见下一节。

### 第三步：抽查一口井的 NPZ

如果你需要确认某口井的域转换是否正确，可以直接加载它的 NPZ：

- 检查有效掩码对应的波阻抗对数值范围是否合理（波阻抗对数值通常在 8~10 左右，对应线性 AI 约 3000~22000 m/s*g/cm3）。
- 检查直井的线号和道号是否为常数，斜井是否随样点变化。

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| schema_version 不匹配 | 上游运行摘要的格式与当前读取接口不匹配 | 用当前版脚本重新生成对应步骤的产物 |
| source adapter/domain 不一致 | `source_run_type` 与上游 summary 的 domain 不匹配 | 使用时间域第四步 `well_auto_tie` |
| AI 单位不是 `m/s*g/cm3` | LAS 中 AI 曲线单位错误或缺失 | 检查上游 LAS 导出配置 |
| AI 包含非正值 | LAS 中有零或负的 AI 值 | 检查上游测井曲线质量 |
| TDT 缺失 | 井缺优化 TDT 表 | 重新运行第四步确保该井标定成功 |
| inventory 行缺失 | 上游成功的井在 well_inventory.csv 中找不到 | 重新运行第一步或检查井名是否变化 |
| XY 与线号不一致 | 井的物理 XY 与工区几何反算的线号不匹配 | 检查 inventory 中井口坐标或斜井轨迹是否正确 |
| 有效样点为零 | 井的 LAS 覆盖范围与目标 SampleAxis 完全不重叠 | 检查目标窗口是否设得合理 |
