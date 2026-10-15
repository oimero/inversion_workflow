# 06 真实工区井控数据集

`real_field_well_controls.py` 是工作流的第六步。本文按时间域说明：脚本读取第四步的井震标定产物，将可信井的已标定滤波曲线对齐到地震时间轴，同时读取背景井的第三步曲线，为第七步提供完整的低频建模控制集。可信井还可以在同一原生井轴上生成主体平滑曲线，供第八步作为井监督输入；滤波曲线始终单独保留，继续供第七步使用。

时间域第四步需要保存完整源时深表，供直井对齐目的层外的可用曲线。若来源运行缺少该产物，先重新运行第四步生成完整源时深关系，再生成井控数据集。

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
| 来源运行 | `run_summary.json` | 来源运行、采样域和产物路径 |
| 第四步 | `run_summary.json`、井清单和每井产物 | 可信井的标定结果、滤波曲线、源/优化时深表和轨迹采样计划 |
| 第五步 | `selected_wavelet.csv` | 时间域井控质检使用的共识子波 |
| 第三步 | `preprocessed_las/*.las` | 背景井的原生曲线；这些井不进入第四、第五步 |
| 第一步 | `well_inventory.csv` | 井口坐标、KB 高程、井型 |
| 数据目录 | 时间域地震体 | 目标时间采样轴和工区几何 |

---

## 配置参考

```yaml
workflow_config: experiments/<workflow_config>.yaml

well_usage:
  trusted_well_names: null
  background_only_well_names: []

real_field_well_controls:
  supervision_curve: filtered  # filtered 或 body
  body_smoothing:
    fwhm_axis_units: <body-smoothing-fwhm-axis-units>
  source_runs:
    well_auto_tie_dir: <well-auto-tie-run-dir>   # 留空自动发现
    well_preprocess_dir: <well-preprocess-run-dir> # 留空时从第四步摘要解析
    well_inventory_dir: <well-inventory-run-dir>  # 留空时从第四步摘要解析
  qc:
    wavelet_generation_run_dir: <wavelet-generation-run-dir> # 留空自动发现
    dynamic_correlation_window_s: <correlation-window-s>
```

`supervision_curve` 决定第八步读取可信井的曲线，默认使用 `filtered`，设为 `body` 时使用主体平滑曲线。主体曲线在原生采样轴的连续有效段内对对数波阻抗做高斯平滑，再投影到地震模型轴；时间域的 `fwhm_axis_units` 以秒填写，质检使用同一条主体曲线。

### `source_runs`

三个来源目录应属于同一轮工作流，并与当前地震域、采样轴和工区几何一致。留空时脚本自动发现运行；填写目录后固定使用该次产物。

| 配置项 | 来源 | 用途 |
|---|---|---|
| `well_auto_tie_dir` | 第四步 | 可信井的标定结果、滤波曲线和域转换信息 |
| `well_preprocess_dir` | 第三步 | 背景井的预处理曲线 |
| `well_inventory_dir` | 第一步 | 井名、井型、井口位置和补心高程 |

可信井只接受第四步标定成功的井；背景井不需要第四、第五步产物，但必须在第三步中有可用曲线。

### `qc`

`qc` 集中保存第六步质检所需的配置。时间域的动态相关窗口使用秒填写；第五步运行目录留空时自动发现与当前域匹配的运行。

### 井用途

第六步从顶层 `well_usage` 读取角色。可信井进入井控清单并可被第八步用作监督；背景井参与第七步低频模型，但不进入第八步井监督。名单外井不进入本次井控构建。主体曲线只对可信井生成，选择 `body` 不改变第七步使用的滤波曲线。

### 缺口

第六步沿用第四步运行摘要中的 `max_short_log_gap_s`。可信井读取第四步已修复的滤波曲线；背景井读取第三步曲线后按同一阈值修复内部短缺口。缺口时长按井的时深关系换算为双程旅行时；长缺口、端点缺口和无法建立时长的缺口保持无效。

- 短缺口修复值参与滤波、重采样、正演和井监督。
- 原生井轴和模型轴分别保存有效掩码与原始观测掩码；长缺口分段处理，主体平滑也不跨段。
- 井曲线评价使用目的层内最长共同有效区间；波形评价排除长缺口边缘的不完整响应，并记录实际评价边界和样点数。

---

## 脚本在做什么

1. **确定来源与井清单。** 检查第四步、第三步和第一步产物，按井角色选择可信井或背景井，并按规范化井名匹配来源。
2. **读取标定曲线。** 可信井读取第四步的滤波曲线、完整源时深表和最佳整体时移；背景井读取第三步曲线。直井保留源映射支持内的可用长曲线。
3. **建立主体监督曲线。** 对可信井的连续有效段做配置尺度的高斯平滑，再把结果投影到地震模型轴；背景井保留滤波曲线。
4. **对齐到地震时间轴。** 在每个连续有效段内插值到地震时间样点。超出覆盖范围和曲线缺口的位置保持无效。
5. **确定逐样点位置。** 直井使用固定井口坐标；斜井将优化轨迹采样计划中的位置插值到地震时间轴，仅使用计划中位于工区内的连续有效段。
6. **建立有效性掩码。** 分别记录原生曲线和模型轴曲线的有效性。缺口不跨段插值，最终有效样点还要求阻抗值和空间位置均为有限值。
7. **核对几何并写出。** 由米制坐标反算线号，逐点检查与已记录线号的一致性，写出逐井数据、井控清单和运行摘要。

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
| `status` | `ok`、`failed` 或 `excluded` |
| `reason` | 失败原因（成功时为空） |
| `sample_domain` | `time` |
| `well_role` | `trusted`、`background_only` 或 `excluded` |
| `wellbore_class` | `vertical` 或 `deviated` |
| `sampling_mode` | 具体采样方式 |
| `n_samples` / `n_valid_samples` | 总样点数 / 有效样点数 |
| `n_observed_samples` / `n_interpolated_samples` | 模型轴上原始观测有效样点数 / 有效但未标记为原始观测的修复或投影样点数 |
| `n_native_samples` / `n_valid_native_samples` | 滤波曲线原生采样总样点数 / 有效样点数 |
| `well_npz_path` | NPZ 路径（失败时为空） |

`run_summary.json` 记录输入来源、井角色、监督曲线及其平滑尺度、短缺口阈值和评价范围。

`qc/evaluation_support.json` 固定每口井的井曲线评价区间，由目的层、滤波曲线、地震样点和空间位置共同确定。实际波形评价范围记录在 `qc/metrics.csv` 中。

第七步和[第八步物理约束神经网络反演](8-ginn.md)读取这一版本的逐井文件。第七步固定读取滤波曲线；第八步按 `supervision_curve` 读取可信井的滤波曲线或主体平滑曲线。生成井控后，将下游配置中的井控目录指向本次输出，并基于这份井控生成相应的低频模型。

### `wells/<well_name>.npz`

每井固定包含模型轴、原生井轴和元数据字段：

| 键 | dtype | 形状 | 含义 |
|------|------|------|------|
| `samples` | float64 | [N] | 地震时间采样值，单位为秒 |
| `model_grid_filtered_log_ai` | float32 | [N] | 目标采样轴上的滤波 LAS ln(AI)，无效处为 NaN；第七步使用此曲线 |
| `model_grid_body_log_ai` | float32 | [N] 或缺省 | 目标采样轴上的主体平滑 ln(AI)，仅在生成主体曲线时写出；第八步选择 `body` 时使用 |
| `inline` | float64 | [N] | 逐样点 inline 线号 |
| `xline` | float64 | [N] | 逐样点 xline 线号 |
| `x_m` | float64 | [N] | 逐样点 X 米制坐标 |
| `y_m` | float64 | [N] | 逐样点 Y 米制坐标 |
| `valid_mask` | bool | [N] | 有效掩码 |
| `observed_valid_mask` | bool | [N] | 未经缺口内插的观测支撑 |
| `native_coordinates` | float64 | [M] | 对齐后的原生 TWT 坐标 |
| `native_filtered_log_ai` | float32 | [M] | 滤波 LAS 原生采样的 ln(AI)，无效处为 NaN |
| `native_body_log_ai` | float32 | [M] 或缺省 | 原生井轴上的主体平滑 ln(AI)，仅可信井生成；无效处为 NaN |
| `native_valid_mask` | bool | [M] | 滤波 LAS 原生采样曲线有效掩码 |
| `native_observed_mask` | bool | [M] | 滤波 LAS 原生采样的原始观测掩码；短缺口修复样点为 false |
| `metadata_json` | 标量字符串 | — | 井名与曲线来源等元数据 |

---

## 如何阅读结果

### 第一步：看终端输出

```
=== Real-field Well Controls ===
Output: scripts/output/real_field_well_controls_<run_timestamp>
Successful wells: <successful_well_count>
```

### 第二步：看 `well_control_manifest.csv`

按 `status` 列分组：

- **ok 井：** 关注有效样点数。有效样点数远少于总样点数说明该井的 LAS 或 TDT/轨迹覆盖范围与目标采样轴重叠有限。这在目标窗口边缘是正常的；如果一口井的有效覆盖率异常低，检查上游 LAS 的深度范围或 TDT 表的时间范围。
- **failed 井：** 看原因列。常见原因见下一节。
- **excluded 井：** 这是井角色配置明确排除的井，不作为失败处理，也不会写入逐井井控文件。

### 第三步：抽查一口井的 NPZ

如果你需要确认某口井的域转换是否正确，可以直接加载它的 NPZ：

- 检查有效掩码对应的波阻抗对数值范围是否合理（波阻抗对数值通常在 8~10 左右，对应线性 AI 约 3000~22000 m/s*g/cm3）。
- 检查直井的线号和道号是否为常数，斜井是否随样点变化。

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| schema_version 不匹配 | 上游运行摘要的格式与当前读取接口不匹配 | 用当前版脚本重新生成对应步骤的产物 |
| 来源域不一致 | 第四步、第三步或当前地震的 domain 不匹配 | 用同一域配置重新生成上游产物 |
| AI 单位不是 `m/s*g/cm3` | LAS 中 AI 曲线单位错误或缺失 | 检查上游 LAS 导出配置 |
| AI 包含非正值 | LAS 中有零或负的 AI 值 | 检查上游测井曲线质量 |
| TDT 缺失 | 井缺优化 TDT 表 | 重新运行第四步确保该井标定成功 |
| inventory 行缺失 | 上游成功的井在 well_inventory.csv 中找不到 | 重新运行第一步或检查井名是否变化 |
| XY 与线号不一致 | 井的物理 XY 与工区几何反算的线号不匹配 | 检查 inventory 中井口坐标或斜井轨迹是否正确 |
| 有效样点为零 | 井的 LAS 覆盖范围与目标采样轴完全不重叠 | 检查目标窗口是否设得合理 |
