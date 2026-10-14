# 06 真实工区井控数据集

`real_field_well_controls.py` 是工作流的第六步。本文按时间域说明：脚本读取第四步的井震标定产物，将可信井的已标定滤波曲线对齐到地震时间轴，同时读取背景井的第三步曲线，为第七步提供完整的低频建模控制集。

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
| 第四步 | `run_summary.json`、井清单和每井产物 | 可信井的标定结果、滤波曲线和优化 TDT |
| 第五步 | `selected_wavelet.csv` | 时间域井控质检使用的共识子波 |
| 第三步 | `preprocessed_las/*.las` | 背景井的原生曲线；这些井不进入第四、第五步 |
| 第四步 | 每井优化 TDT | MD→TWT 映射 |
| 第四步 | 每井 trace sample plan | 斜井逐样点 inline/xline/XY（仅斜井） |
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
  source_runs:
    well_auto_tie_dir: <well-auto-tie-run-dir>   # 留空自动发现
    well_preprocess_dir: <well-preprocess-run-dir> # 留空时从第四步摘要解析
    well_inventory_dir: <well-inventory-run-dir>  # 留空时从第四步摘要解析
  qc:
    wavelet_generation_run_dir: <wavelet-generation-run-dir> # 留空自动发现
    body_smoothing_fwhm_s: <smoothing-fwhm-s>
    dynamic_correlation_window_s: <correlation-window-s>
```

### `source_runs`

三个来源目录应属于同一轮工作流，并与当前地震域、采样轴和工区几何一致。留空时脚本按输出根目录自动发现可用运行；填写目录后固定使用该次产物。

| 配置项 | 来源 | 用途 |
|---|---|---|
| `well_auto_tie_dir` | 第四步 | 可信井的标定结果、滤波曲线和域转换信息 |
| `well_preprocess_dir` | 第三步 | 背景井的预处理曲线 |
| `well_inventory_dir` | 第一步 | 井名、井型、井口位置和补心高程 |

可信井只接受第四步标定成功的井。第五步使用这些井的候选子波生成共识结果；背景井不需要第四步或第五步产物，但必须在第三步中有可用曲线。

### `qc`

`qc` 集中保存第六步质检所需的配置。时间域使用秒制平滑尺度和动态相关窗口；`wavelet_generation_run_dir` 留空时自动发现与当前域匹配的第五步运行。

### 井用途

第六步从顶层 `well_usage` 读取角色。可信井进入井控清单并可被第八步用作监督；背景井也进入井控清单并参与第七步低频模型，但不进入第八步井监督。名单外井不进入本次井控构建。

### 缺口

缺口处理由上游滤波后的 LAS 负责。第六步只在各个有限段内投影到目标采样轴，不跨缺口插值，也不在模型轴上重新填补。井控曲线和井监督掩码保留缺口；短缺口如需参与卷积或波形质检，只在该次计算的中间数组中处理，不回写到持久化曲线或观测支撑。观测支撑掩码表示投影后的上游有效支撑，缺口样点在模型轴阻抗字段和掩码中保持无效。

---

## 脚本在做什么

1. **确定来源与井清单。** 检查第四步、第三步和第一步产物，按井角色选择可信井或背景井，并按规范化井名匹配来源。
2. **读取标定曲线。** 可信井读取第四步标定后的滤波曲线和优化 TDT；背景井读取第三步曲线。
3. **对齐到地震时间轴。** 在每个连续有效段内插值到地震时间样点。超出覆盖范围和曲线缺口的位置保持无效。
4. **确定逐样点位置。** 直井使用固定井口坐标；斜井将优化轨迹采样计划中的位置插值到地震时间轴，仅使用计划中位于工区内的连续有效段。
5. **建立有效性掩码。** 分别记录原生曲线和模型轴曲线的有效性。缺口不跨段插值，最终有效样点还要求阻抗值和空间位置均为有限值。
6. **核对几何并写出。** 由米制坐标反算线号，逐点检查其与已记录线号的一致性，写出逐井数据、角色、井控清单和运行摘要。

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
| `sample_domain` | `time` |
| `well_role` | `trusted` 或 `background_only` |
| `wellbore_class` | `vertical` 或 `deviated` |
| `sampling_mode` | 具体采样方式 |
| `n_samples` / `n_valid_samples` | 总样点数 / 有效样点数 |
| `n_observed_samples` / `n_interpolated_samples` | 模型轴上由输入曲线投影得到的有效样点数 / 有效但未标记为观测的样点数（当前实现为 0） |
| `n_native_samples` / `n_valid_native_samples` | 滤波曲线原生采样总样点数 / 有效样点数 |
| `well_npz_path` | NPZ 路径（失败时为空） |

`run_summary.json` 记录采样域、井角色、原生曲线来源和缺口处理方式，并保存第六步固定评价支撑、直接上游与产物路径。

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
| 来源域不一致 | 第四步、第三步或当前地震的 domain 不匹配 | 用同一域配置重新生成上游产物 |
| AI 单位不是 `m/s*g/cm3` | LAS 中 AI 曲线单位错误或缺失 | 检查上游 LAS 导出配置 |
| AI 包含非正值 | LAS 中有零或负的 AI 值 | 检查上游测井曲线质量 |
| TDT 缺失 | 井缺优化 TDT 表 | 重新运行第四步确保该井标定成功 |
| inventory 行缺失 | 上游成功的井在 well_inventory.csv 中找不到 | 重新运行第一步或检查井名是否变化 |
| XY 与线号不一致 | 井的物理 XY 与工区几何反算的线号不匹配 | 检查 inventory 中井口坐标或斜井轨迹是否正确 |
| 有效样点为零 | 井的 LAS 覆盖范围与目标 SampleAxis 完全不重叠 | 检查目标窗口是否设得合理 |
