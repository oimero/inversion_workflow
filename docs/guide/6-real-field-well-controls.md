# 06 真实工区井控数据集

`real_field_well_controls.py` 是工作流的第六步。它把预处理后的完整井曲线统一到地震采样轴，同时保留原生采样井曲线，并为每口成功井生成正演质控图。

> 深度域工区使用第五步（`wavelet_batch_synthetic_depth`）作为上游，配置方式见文末。

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
| 第四步 | `run_summary.json` | schema/domain 校验和直接上游契约身份 |
| 第四步 | `well_tie_metrics.csv` | 成功井清单和产物路径 |
| 第三步与第四步 | 每井预处理 LAS | 原生 `AI [m/s*g/cm3]` 曲线 |
| 第四步 | 每井优化 TDT | MD→TWT 映射 |
| 第四步 | 每井 trace sample plan | 斜井逐样点 inline/xline/XY（仅斜井） |
| 第一步 | `well_inventory.csv` | 井口坐标、KB 高程、井型 |
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
  max_internal_gap_axis_units: 0.008

real_field_well_controls_qc:
  forward_model_inputs_run_dir:
  body_smoothing_fwhm_m: 15.0
  dynamic_correlation_window_m: 75.0
  event_threshold_fraction: 0.10
  max_event_windows_per_well: 4
```

### `source_run_type`

必填。时间域工区填 `well_auto_tie`。深度域工区填 `wavelet_batch_synthetic_depth`（详见文末）。不能为空、不能缩写、不能自动推断。

| 值 | 目标域 | 上游 |
|---|---|---|
| `well_auto_tie` | time + s | 第四步成功井、预处理 LAS 和优化 TDT |
| `wavelet_batch_synthetic_depth` | depth + tvdss + m | 深度域第五步成功井和平移后的预处理 LAS |

输入要求：

- 只接受第四步 `tie_status=success` 的井。
- 每井必须有预处理 LAS（含 AI 曲线，单位 `m/s*g/cm3`）和对应的域转换信息。
- 斜井还需要优化轨迹采样计划文件。

### `source_run_dir`

指向第四步的运行目录。留空时脚本按 `source_run_type` 前缀在输出目录下自动发现最新的成功运行。显式填路径则固定使用该目录，适合复现。

### `well_inventory_file`

指向第一步产出的 `well_inventory.csv`。脚本从中读取每口井的井型（直井或斜井）、井口坐标、线号和道号，以及补心海拔。

### `well_trace_dir`

深度域斜井使用该目录中的轨迹文件；直井不读取轨迹文件。

### 内部缺口

`max_internal_gap_axis_units` 的单位由采样轴决定：时间域为秒，深度域为米。脚本只线性内插不超过该宽度的内部缺口；曲线两端和更长缺口保持为空。当前深度域主配置使用 10 m。

---

## 脚本在做什么

脚本发布同一口井的两层事实：模型轴上的完整井控服务于低频模型和后续训练，原生采样上的完整井曲线服务于主体监督和高频残差。处理顺序为适配、域转换、写入和正演质控。

### 第一阶段：适配

1. 读取上游 `run_summary.json`，确认 schema、domain、status 匹配当前配置的 `source_run_type`。
2. 读取 `well_tie_metrics.csv`，只保留 `tie_status=success` 的井。
3. 读取 `well_inventory.csv`，与上游成功井做井名匹配。上游成功但 inventory 中不存在的井被拒绝。

### 第二阶段：域转换

对每口井，通过优化 TDT 表把每个 TWT 采样点映射到 MD 轴上的一个位置，再从 LAS 的 MD 轴上读出该位置的 ln(AI) 值。空间位置方面，直井直接把井口的固定线号道号广播到所有样点；斜井从第四步产出的轨迹采样计划读取逐样点的线号、道号和 XY。

模型轴记录两个掩码：`observed_valid_mask` 表示直接由原生有限段投影得到的样点，`valid_mask` 表示可供低频模型使用的样点，其中可以包含配置允许的短内部缺口内插。两端缺口和长缺口在两个掩码中都保持无效。

### 第三阶段：校验与写入

转换完成后，脚本对每口成功井做几何一致性校验——用测网几何把真实 XY 反算线号，与记录的线号道号逐点对照，不一致的井被拒绝。

然后写入三类产物：

1. **逐井 NPZ。** `wells/<well>.npz` 同时包含模型轴井控和原生完整井控。两层各自携带采样坐标、波阻抗对数和有效掩码；模型轴层同时携带线号、道号和米制坐标。
2. **Manifest CSV。** 每口候选井一行，分别记录两层数据的总样点数、有效样点数和 NPZ 路径。失败的井也保留行，但 NPZ 路径为空。
3. **运行摘要 JSON。** 记录来源适配器、采样轴、上游契约指纹、井数统计和产物路径。

### 第四阶段：每井正演质控

深度域运行读取第五步发布的冻结子波和 AI–Vp 关系。每口成功井生成三张目的层图件：

1. 完整井曲线的六联正演质控图；
2. 15 m 主体曲线的六联正演质控图；
3. 若干真实地震波瓣窗口中的完整曲线、主体曲线及两套合成波形对比图。

完整曲线与主体曲线使用同一个由完整曲线估计的振幅系数，因此第三张图保留两套合成波形之间的真实振幅差异。井曲线覆盖不完整时，图件显示目的层内最长的共同有效区间。

---

## 核心输出文件

```text
real_field_well_controls_<timestamp>/
├── run_summary.json
├── well_control_manifest.csv
├── wells/
│   ├── <well_a>.npz
│   └── <well_b>.npz
└── qc/
    ├── manifest.json
    ├── metrics.csv
    └── figures/<well_name>/
        ├── full_waveform_qc.png
        ├── body_waveform_qc.png
        └── event_waveform_comparison.png
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
| `n_observed_samples` / `n_interpolated_samples` | 原始观测样点数 / 短缺口内插样点数 |
| `n_native_samples` / `n_valid_native_samples` | 原生完整井曲线总样点数 / 有效样点数 |
| `well_npz_path` | NPZ 路径（失败时为空）；消费者不再重算逐井文件哈希 |

`run_summary.json` 使用 `real_field_well_controls_v5`，通过 `input_contracts` 记录直接上游，并发布一个生产者契约指纹。

### `wells/<well_name>.npz`

每井固定包含模型轴、原生井轴和元数据字段：

| 键 | dtype | 形状 | 含义 |
|------|------|------|------|
| `samples` | float64 | [N] | SampleAxis 采样值 |
| `log_ai` | float32 | [N] | ln(AI)，无效处为 NaN |
| `inline` | float64 | [N] | 逐样点 inline 线号 |
| `xline` | float64 | [N] | 逐样点 xline 线号 |
| `x_m` | float64 | [N] | 逐样点 X 米制坐标 |
| `y_m` | float64 | [N] | 逐样点 Y 米制坐标 |
| `valid_mask` | bool | [N] | 有效掩码 |
| `observed_valid_mask` | bool | [N] | 未经缺口内插的观测支撑 |
| `native_coordinates` | float64 | [M] | 对齐后的原生 TWT 或 TVDSS 坐标 |
| `native_full_log_ai` | float32 | [M] | 原生采样的完整 ln(AI)，无效处为 NaN |
| `native_valid_mask` | bool | [M] | 原生完整井曲线有效掩码 |
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
QC figures: scripts/output/real_field_well_controls_<timestamp>/qc/figures
```

成功井数和图件目录均打印后，第六步完成。

### 第二步：看 `well_control_manifest.csv`

按 `status` 列分组：

- **ok 井：** 关注有效样点数。有效样点数远少于总样点数说明该井的 LAS 或 TDT/轨迹覆盖范围与目标采样轴重叠有限。这在目标窗口边缘是正常的；如果一口井的有效覆盖率异常低，检查上游 LAS 的深度范围或 TDT 表的时间范围。
- **failed 井：** 看原因列。常见原因见下一节。

### 第三步：抽查一口井的 NPZ

如果你需要确认某口井的域转换是否正确，可以直接加载它的 NPZ：

- 检查有效掩码对应的波阻抗对数值范围是否合理（波阻抗对数值通常在 8~10 左右，对应线性 AI 约 3000~22000 m/s*g/cm3）。
- 检查直井的线号和道号是否为常数，斜井是否随样点变化。

### 第四步：检查三张图件

- `full_waveform_qc.png` 检查完整井曲线正演与真实地震的相位、振幅和局部相关性。
- `body_waveform_qc.png` 检查 15 m 主体尺度是否保留主要地震响应。
- `event_waveform_comparison.png` 直接观察几个地震波瓣内 full 与 body 合成响应的差异。

---

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| schema_version 不匹配 | 上游 run summary 不是 v2 schema | 用当前版脚本重建上游 run |
| source adapter/domain 不一致 | `source_run_type` 与上游 summary 的 domain 不匹配 | 时间域用 `well_auto_tie` |
| AI 单位不是 `m/s*g/cm3` | LAS 中 AI 曲线单位错误或缺失 | 检查上游 LAS 导出配置 |
| AI 包含非正值 | LAS 中有零或负的 AI 值 | 检查上游测井曲线质量 |
| TDT 缺失 | 井缺优化 TDT 表 | 重新运行第四步确保该井标定成功 |
| inventory 行缺失 | 上游成功的井在 well_inventory.csv 中找不到 | 重新运行第一步或检查井名是否变化 |
| XY 与线号不一致 | 井的 physical XY 与 survey geometry 反算的线号不匹配 | 检查 inventory 中井口坐标或斜井轨迹是否正确 |
| 有效样点为零 | 井的 LAS 覆盖范围与目标 SampleAxis 完全不重叠 | 检查目标窗口是否设得合理 |

失败井保留在 manifest 中，并记录明确原因。
