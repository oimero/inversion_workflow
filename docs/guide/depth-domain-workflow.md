# 深度域工作流

深度域工作流复用井资产盘点、测井曲线筛选、测井预处理、岩石物理分析、井控集和低频模型构建。第四步提取单井固定子波，第五步使用该子波进行全井合成与深度平移。

---

## 总览

时间域和深度域分别执行第四、第五步：

- 时间域第4步做全井自动标定，输出每井候选子波；
- 深度域第4步只跑一口指定井，输出一条固定子波；
- 时间域第5步做全井交叉评测和共识子波优化；
- 深度域第5步用固定子波做全井时移扫描和深度平移 LAS 导出。

---

## 第 4 步：固定子波提取

第四步针对一口指定井提取固定时间子波，供后续批量合成使用。深度换算采用直井近似：钻井测量深度减去补心高程，得到相对海平面的垂向深度。

### 快速开始

```powershell
python scripts/vertical_well_auto_tie_depth.py --config "<config-yaml>" --well "<source-well>"
python scripts/vertical_well_auto_tie_depth.py --config "<config-yaml>" --well "<source-well>" --output-dir "<tie-output-dir>"
```

### 配置参考

配置位于 `vertical_well_auto_tie_depth` 段。尖括号内容替换为本地路径或井名；以下数值展示搜索范围的配置形式，实际范围需结合测井采样间隔和振幅尺度设置。

```yaml
vertical_well_auto_tie_depth:
  well_name: "<source-well>"        # 子波来源井
  source_runs:
    well_preprocess_dir: "<preprocess-run-dir>"  # 留空时自动发现第三步产物
  las_vp_unit: us/m                 # DT 单位
  las_rho_unit: g/cm3               # 密度单位
  target_crop_ms: 201.0             # 最终子波目标长度 (ms)

  tutorial_model: "<pretrained-model-file>"       # 相对数据根目录的模型路径
  tutorial_params: "<network-parameters-file>"    # 相对数据根目录的参数路径

  search_space:                     # autotie 搜索空间定义
    logs_median_size_values: [3, 5, 7, 9, 11, 15, 21]
    logs_median_threshold_bounds: [0.05, 0.25]
    logs_std_bounds: [0.05, 0.35]
    table_t_shift_bounds: [-0.04, 0.04]

  search_params:                    # autotie 搜索参数
    num_iters: 80
    similarity_std: 0.1

  wavelet_scaling:                  # 子波振幅缩放
    min_scale: 0.5
    max_scale: 2.0
    num_iters: 20
```

测井预处理运行目录相对仓库根目录解析。来源目录留空时，脚本在配置的输出根目录内查找第三步产物。预训练模型和网络参数文件相对数据根目录解析。

此外还需顶层 `seismic` 段声明深度域：

```yaml
seismic:
  domain: depth
  depth_basis: tvdss
  file: "<seismic-file>"
  type: zgy
```

地震配置需要使用深度域和海平面以下垂深口径。文件类型按实际数据选择；读取 SEG-Y 时还需在同一配置段填写道头位置和线号步长。

### 脚本在做什么

1. **准备井曲线与井旁地震。** 读取预处理后的纵波速度和密度曲线，对其中的缺失值做线性插值，并根据井口坐标提取深度域地震道。

2. **建立局部时深关系。** 按直井近似把测井深度换算到海平面以下垂深，裁取井曲线与地震道的共同深度窗。沿深度积分纵波慢度，得到双程旅行时，并以共同窗口顶部作为相对时间零点。

3. **转换到规则时间轴。** 使用局部时深关系，把深度域地震插值到预训练模型要求的时间采样间隔。测井曲线也按同一时深关系转换到时间域，用于计算反射系数。

4. **搜索标定参数。** 联合搜索测井去尖峰、平滑和整体时间平移参数。每组参数对应一条估计子波及卷积合成记录，通过与地震道的相似度选择结果，再估计子波的振幅尺度。

5. **裁剪、归一化与评价。** 以零时刻为中心裁取目标长度的奇数样点子波，使振幅平方和为一。用裁剪子波重新生成合成记录，拟合展示用振幅尺度，计算相关系数和归一化绝对误差。输出时深关系、收敛过程、波形对比和子波频谱图。

### 核心输出文件

| 文件 | 内容 |
|------|------|
| `wavelet_201ms_<source-well>.csv` | 最终子波（columns: `time_s`, `amplitude`），奇数长度，能量归一化 |
| `run_summary_<source-well>.json` | autotie 参数、重叠窗口范围、裁剪信息、合成指标 |
| `wavelet_raw/auto_well_tie_wavelet_raw_<source-well>.csv` | 裁剪前的 raw wavelet |
| `depth_match/local_tdt_md_<source-well>.csv` | 局部时深表（columns: `tvdss_m`, `twt_s`, `md_m`, `vp_mps`） |
| `depth_match/seismic_twt_from_depth_<source-well>.csv` | 从深度域转到 TWT 域的地震道 |
| `synthetic_qc/auto_well_tie_synthetic_qc_*.csv` | raw 和 cropped 合成记录 QC |
| `figures/qc_*.png` | 5 张 QC 图 |

---

## 第 5 步：批量合成与深度平移

第五步使用第四步的固定子波与测井滤波参数，对各井计算合成记录，确定整体时间平移量，并导出深度平移后的全曲线与滤波曲线两套测井文件。

### 快速开始

```powershell
python scripts/wavelet_batch_synthetic_depth.py --config "<config-yaml>"
python scripts/wavelet_batch_synthetic_depth.py --config "<config-yaml>" --well "<well-name>"
python scripts/wavelet_batch_synthetic_depth.py --config "<config-yaml>" --output-dir "<batch-output-dir>"
```

用 `--well` 可以只跑一口井调试。

### 配置参考

所有配置在 `wavelet_batch_synthetic_depth` 段下：

```yaml
wavelet_batch_synthetic_depth:
  source_runs:
    well_preprocess_dir: "<preprocess-run-dir>"
    vertical_well_auto_tie_depth_dir: "<tie-run-dir>"
  las_vp_unit: us/m
  las_rho_unit: g/cm3

  source_well_name: "<source-well>"        # 与第四步保持一致
  skip_shift_scan_well_names:             # 按配置保持零时移的井
    - "<zero-shift-well-a>"
    - "<zero-shift-well-b>"

  shift_min_ms: -20.0                    # 时移扫描下限
  shift_max_ms: 20.0                     # 时移扫描上限
```

同样需要顶层 `seismic.domain: depth` + `seismic.depth_basis: tvdss`。

默认自动接上最新一次测井预处理结果和最新一次包含指定来源井产物的深度域第4步结果。复现实验时可按需填写 `source_runs` 下对应的运行目录固定输入。

脚本从发现的深度域第4步目录读取子波和 `run_summary_<source-well>.json`，并使用该次标定选出的测井中值滤波窗口、去尖峰阈值和高斯平滑标准差。摘要缺少任一参数时脚本直接报错。

### 脚本在做什么

1. **构造每口井的时间域输入。** 对速度和密度的缺失值做线性插值，按直井近似裁取共同深度窗，并由速度积分建立局部时深关系。使用第四步选出的去尖峰和平滑参数处理测井，计算时间域反射系数，把井旁地震转换到同一规则时间轴，并对地震振幅做标准化。

2. **确定整体时间平移量。** 按配置的上下限扫描时间平移，步长等于子波的时间采样间隔。将平移后的反射系数与固定子波卷积，拟合振幅尺度并计算波形相关性，选择相关性最高的候选。配置为零时移的井在零点完成合成与评价，采用零平移量。

3. **将时间平移换算为深度平移。** 使用每口井的局部时深关系，把原时间及平移后的时间分别映射回深度，两者之差形成随深度变化的平移曲线。平移曲线覆盖范围外使用端点值延伸，并记录受影响样点数量。

4. **导出两套深度平移曲线。** 全曲线版重新读取第三步的原始预处理文件，按连续有效段进行深度搬移，保留原有缺口，并由声波时差和密度重新计算声阻抗。滤波版保存合成计算所用的声波时差、密度和声阻抗，按同一平移曲线导出。第六步井控集读取滤波版，合成基准分别使用两套曲线。

5. **汇总合成结果。** 输出每井波形对比、扫描曲线和深度平移统计。汇总图使用各井实际采用的平移量，另报告来源井的批量平移与第四步标定平移之间的差值。

零时移名单需使用本次测井目录中的井名，来源井保留正常扫描；重复井名或未知井名会导致配置失败。所有成功导出的井都进入下游井控处理。

### 核心输出文件

| 文件 | 内容 |
|------|------|
| `wavelet_batch_metrics.csv` | 全井汇总：时移策略、扫描最佳时移、实际采用时移、corr、NMAE、深度平移统计、产物路径 |
| `run_summary.json` | 输入路径、LAS 契约说明、成功/失败计数 |
| `shifted_preprocessed_las/<well>.las` | 深度平移后的 Step 3 全曲线 LAS |
| `shifted_filtered_las/<well>.las` | 深度平移后的 filtered DT_USM/RHO_GCC/AI |
| `shift_scans/shift_scan_<well>.csv` | 执行扫描井的时移扫描明细（shift_s, corr, nmae, scale） |
| `depth_shift_curves/depth_shift_curve_<well>.csv` | 每井深度平移曲线（twt_s, tvdss_m, depth_shift_m） |
| `synthetic_qc/synthetic_qc_<well>.csv` | 每井实际采用时移下的合成记录 QC |
| `figures/qc_01_batch_metric_summary.png` | 批量指标汇总（corr, NMAE, applied shift） |
| `figures/qc_02_batch_depth_shift_summary.png` | 深度平移汇总（median + P10-P90） |
| `figures/qc_<well>_synthetic_vs_seismic.png` | 每井 R1 风格六 panel 波形 QC 图 |
| `figures/qc_<well>_shift_scan.png` | 执行扫描井的时移扫描 corr 曲线 |

---

## 旁路：深度域正演输入冻结

这一步将岩石物理分析的声阻抗—纵波速度关系与第四步的固定子波组装为统一正演输入，供井控质检、深度域合成基准和主体反演读取。

岩石物理分析和子波提取各自独立重跑，本旁路只在子波或关系发生变化时才需要重跑，避免更新子波时必须重跑整个岩石物理拟合。

### 快速开始

```powershell
python scripts/depth_forward_model_inputs.py
python scripts/depth_forward_model_inputs.py --config experiments/common/common.yaml
python scripts/depth_forward_model_inputs.py --config "<config-yaml>" --output-dir "<forward-input-output-dir>"
```

不带参数时，脚本自动发现最新的岩石物理分析和深度域第 4 步产物。

### 配置参考

所有配置在 `depth_forward_model_inputs` 段下：

```yaml
depth_forward_model_inputs:
  source_runs:
    rock_physics_analysis_dir: "<rock-physics-run-dir>"
    vertical_well_auto_tie_depth_dir: "<tie-run-dir>"
  source_well_name: "<source-well>"       # 与第四步保持一致
```

同时需要顶层 `seismic.domain: depth` + `seismic.depth_basis: tvdss`。

### 脚本在做什么

1. 读取成功的岩石物理拟合与单井标定结果，核对深度口径和子波来源井。
2. 读取子波的时间与振幅，检查规则采样、奇数样点数和零时刻居中，通过卷积计算检查正演输入条件。
3. 读取声阻抗与纵波速度的线性关系，检查公式、系数、物理单位和参与拟合的井清单。
4. 汇总子波路径、采样参数、关系系数及来源记录，写出统一正演输入文件，供各下游步骤共用。

### 核心输出文件

| 文件 | 内容 |
|------|------|
| `forward_model_inputs.json` | 冻结的正演输入：子波路径与参数、AI–Vp 关系路径与系数、来源运行契约指纹 |
| `run_summary.json` | 来源运行发现方式、`source_well_name`、契约指纹 |

---

## 第 6 步：深度域井控集与正演质检

第六步读取批量深度平移后的滤波测井，将井曲线与位置对齐到地震深度轴，同时保留滤波曲线的原生采样版本。入口与主教程相同：

```powershell
python scripts/real_field_well_controls.py --config "<config-yaml>" --output-dir "<well-control-output-dir>"
```

### 配置参考

以下配置与前面的工区配置放在同一个文件中。来源运行目录和井资产清单相对仓库根目录解析，轨迹目录相对数据根目录解析。

```yaml
assets:
  well_trace_dir: "<well-trajectory-dir>"

real_field_well_controls:
  source_run_type: wavelet_batch_synthetic_depth
  source_run_dir: "<batch-run-dir>"
  well_inventory_file: "<well-inventory-csv>"

real_field_well_controls_qc:
  forward_model_inputs_run_dir: "<forward-input-run-dir>"
  body_smoothing_fwhm_m: 25.0
  dynamic_correlation_window_m: 75.0
  event_threshold_fraction: 0.10
  max_event_windows_per_well: 4
```

源目录留空时，脚本按批量深度平移的运行前缀查找产物。正演输入目录留空时，脚本查找与本次深度口径一致的正演输入。质检配置中的四个数值参数均需填写。

### 脚本在做什么

1. **读取成功井的滤波曲线。** 根据第五步汇总表选择成功井，读取平移后的滤波声阻抗，并保留原生测井轴上的曲线与有效性信息。
2. **转换井深与位置。** 直井采用补心高程换算垂深并使用井口位置；斜井沿轨迹将测深转换为海平面以下垂深和平面位置。
3. **对齐到地震深度轴。** 在连续有效曲线段内插值到地震的规则深度样点，保存每个样点的井位置和有效性，写出逐井数据与汇总清单。
4. **比较两种阻抗的正演结果。** 在目标层段内，分别对滤波后的完整阻抗曲线和经高斯平滑的主体曲线做深度正演。两套合成地震共用由完整曲线拟合得到的振幅增益，计算整体与局部波形相似度。
5. **比较地震事件窗口。** 从真实地震中选择振幅满足阈值的事件，截取局部窗口，展示两种阻抗、两者之差及对应合成地震，输出逐井图片和指标。

### 质检输出

| 文件 | 内容 |
|---|---|
| `qc/figures/<well>/full_waveform_qc.png` | 滤波后完整阻抗曲线的正演对比 |
| `qc/figures/<well>/body_waveform_qc.png` | 平滑主体曲线的正演对比 |
| `qc/figures/<well>/event_waveform_comparison.png` | 地震事件窗口中的阻抗与波形对比 |
| `qc/metrics.csv` | 每井相关系数、共用增益和事件窗口数量 |
| `qc/manifest.json` | 质检参数、来源和图件路径 |

井控清单和逐井数组的通用结构见[第六步主教程](6-real-field-well-controls.md)。深度域中，`samples` 表示地震的海平面以下垂深样点，`native_coordinates` 表示原生测井样点转换后的海平面以下垂深，两者单位均为米。这里的完整曲线与主体曲线都以第五步的滤波测井为输入。

## 第 7 步：深度域低频模型

第七步使用深度域井控集和解释层位，按实际平面坐标构建低频模型。基线方法、修饰器和变体组织见[第七步主教程](7-real-field-lfm.md)；本节说明深度轴和滤波单位。

```powershell
python scripts/real_field_lfm.py --config "<config-yaml>" --output-dir "<lfm-output-dir>"
```

### 配置参考

```yaml
real_field_lfm:
  source_runs:
    well_control_run_dir: "<well-control-run-dir>"
  output_geometry:
    mode: volume
  baselines:
    depth_trend:
      method: trend
      filter:
        enabled: true
        cutoff_wavelength_m: "<cutoff-wavelength-m>"
        order: 6
        buffer_mode: reflect
        buffer_axis_units: "<depth-buffer-m>"
      fit:
        min_valid_samples_per_well: 32
        huber_f_scale_log_ai: 0.05
      spatial:
        variogram: spherical
        exact: true
        nugget: 0.0
  modifiers: {}
  variants:
    - variant_id: depth_trend_baseline
      baseline_id: depth_trend
      modifier_ids: []
  comparisons: []
```

截止波长和缓冲长度均按米填写，占位符需要替换为正数。井控集、地震和解释层位需要采用相同的海平面以下垂深口径，目标层位文件来自顶层工区配置。

### 脚本在做什么

1. **建立共同网格。** 读取深度域井控、地震采样轴和目标层位，确定输出体、窗口或剖面的平面与深度范围。
2. **提取井内低频信息。** 把截止波长换算为每米的空间频率，对每口井的连续有效阻抗段做双向低通，边界缓冲长度按深度采样间隔换算为样点数。
3. **建立基线。** 趋势方法按层段相对深度拟合每井趋势参数，再沿平面插值；比例切片方法在层段相对深度上建立切片，逐片插值并还原到地震深度轴。
4. **形成变体与输出。** 按配置叠加框架修饰和比较项，保存低频模型、有效性信息、拟合质检与图件。模型的深度轴以米表达，平面插值距离也以米表达。

---

## 第 8 步：深度域主体反演

第八步读取第六步深度域井控、第七步低频模型与统一正演输入，依次进行地震自监督预训练和可信井约束微调。训练阶段、主体分解、模型选择与产物说明见[第八步主教程](8-ginn-v2-body-inversion.md)。

```powershell
python scripts/body_train.py --config "<body-config-yaml>" --output-dir "<training-output-dir>"
```

训练入口可用 `--lfm-run-dir`、`--variant-id`、`--well-control-run-dir` 和 `--forward-model-inputs-run-dir` 覆盖对应配置；分阶段微调还需要 `--pretrain-checkpoint`。

在第八步配置中，将上游输入指向本次深度域成果：

```yaml
ginn_v2_body_inversion:
  inputs:
    lfm_run_dir: "<lfm-run-dir>"
    variant_id: "<lfm-variant-id>"
    well_control_run_dir: "<well-control-run-dir>"
    forward_model_inputs_run_dir: "<forward-input-run-dir>"
  training:
    trusted_well_names:
      - "<trusted-well-a>"
      - "<trusted-well-b>"
    body_smoothing_fwhm_m: <body-smoothing-fwhm-m>
    waveform_qc_dynamic_window_m: <waveform-qc-dynamic-window-m>
    patch_radius: 8
    orientations: [inline, xline]
    validation_gap_m: <validation-gap-m>
    loss_weights:
      seismic_shape: 1.0
      trusted_well_body: 1.0
      trusted_well_derivative: 0.5
      lfm_anchor: 1.0
    selection_weights:
      well_rmse: 1.0
      amplitude_mapping: 0.0
    warnings:
      pretrain_masked_corr_improvement: 0.01
      pretrain_masked_shape_ratio: 0.99
      masked_corr_drop_tolerance: 0.01
      well_pooled_rmse_ratio_max: 1.0
      seismic_body_amplitude_spearman_max: 1.0
```

训练配置与这些输入放在同一个配置文件中，公共工区配置使用深度域与海平面以下垂深口径。统一正演输入沿用本篇旁路生成的子波与声阻抗—纵波速度关系，速度可由该关系和初始阻抗计算。

| 配置 | 深度域语义 |
|------|------------|
| `body_smoothing_fwhm_m` | 主体高斯平滑半高全宽，沿 TVDSS 深度轴计算，单位为米且必须为正 |
| `waveform_qc_dynamic_window_m` | 局部波形统计窗口，沿 TVDSS 深度轴计算，单位为米且必须为正 |
| `forward_model_inputs_run_dir` | 本篇旁路生成的正演输入目录，包含固定子波和 AI–Vp 关系 |
| `validation_gap_m` | 平面空间验证块与训练区之间的米制间隔 |

主体平滑直接使用地震深度轴的米制坐标。第七步变体的 `filter.enabled` 为真时，主体输出仍执行其截止波长定义的低频修正投影；只有在该开关为假且 `loss_weights.lfm_anchor` 为零时，主体输出才只使用高斯平滑。井控、低频模型和地震的深度轴、线网及空间几何需要一致。深度轴上的损失和局部统计使用当前米制坐标，平面验证间隔也以米表示。微调完成后生成所选权重、井曲线和局部剖面质检，全体积预测另行执行。

---

## 与时间域工作流的关系

前三步为两种域共享的井数据准备。时间域第四、第五步依次完成全井标定和共识子波生成；深度域第四、第五步依次完成固定子波提取和批量深度平移。两条路径的滤波测井成果分别进入第六步井控集，再用于第七步真实工区低频模型和第八步主体反演。

岩石物理分析从第三步测井出发，与深度域第四步的固定子波共同组成正演输入。深度域第六步质检、合成基准与主体反演共享这份输入。
