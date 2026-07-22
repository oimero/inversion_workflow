# 深度域工作流

深度域工作流定位为**一次性处理**，复用概率低。第1步–3（井资产盘点、LAS 筛选、测井预处理）和旁路岩石物理分析与时间域共享，本文不再重复。

---

## 总览

深度域特有的步骤只有两步：**固定子波提取**和**批量合成与深度平移**。深度域第4步/5 与时间域第4步/5 **互不依赖**、**平行存在**：

- 时间域第4步做全井自动标定，输出每井候选子波；
- 深度域第4步只跑一口指定井，输出一条固定子波；
- 时间域第5步做全井交叉评测和共识子波优化；
- 深度域第5步用固定子波做全井时移扫描和深度平移 LAS 导出。

---

## 第 4 步：固定子波提取

`vertical_well_auto_tie_depth.py` 是一个单井专用子波提取脚本，只跑一口井、输出一条固定时间子波，供后续批量合成使用。

### 快速开始

```powershell
python scripts/vertical_well_auto_tie_depth.py --config <config-yaml> --well <source-well>
python scripts/vertical_well_auto_tie_depth.py --config <config-yaml> --well <source-well> --output-dir scripts/output/vertical_well_auto_tie_depth_<source-well>
```

### 配置参考

所有配置在 `vertical_well_auto_tie_depth` 段下：

```yaml
vertical_well_auto_tie_depth:
  well_name: <source-well>          # 子波来源井
  source_runs:
    well_preprocess_dir:            # 留空时自动发现最新的第3步产物
  las_vp_unit: us/m                 # DT 单位
  las_rho_unit: g/cm3               # 密度单位
  target_crop_ms: 201.0             # 最终子波目标长度 (ms)

  tutorial_model: <path>            # wtie 预训练模型文件
  tutorial_params: <path>           # wtie 训练参数 YAML

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

默认自动接上最新一次测井预处理结果。复现实验时可按需填写
`source_runs.well_preprocess_dir` 固定整套输入。

此外还需顶层 `seismic` 段声明深度域：

```yaml
seismic:
  domain: depth
  depth_basis: tvdss
  file: <path-to-zgy>
  type: zgy
```

脚本入口会校验 `seismic.domain == "depth"` 且 `seismic.depth_basis == "tvdss"`，不满足直接失败。

### 脚本在做什么

深度域地震的纵轴是海拔深度（米），测井曲线的纵轴是钻井测量深度（米），二者虽然单位相同，但无法直接比对——因为波阻抗界面产生的地震反射是按时间先后到达的，而深度域地震道本身没有直接记录每个深度点对应的波传播时间。要知道井曲线在哪个深度产生哪个时间的地震响应，必须先知道速度随深度的分布。

脚本的做法分四步：

**1) 由速度曲线计算局部时深关系。** 从井的声波时差曲线得到速度，沿深度方向逐段累加声波的往返时间，得到一张该井的深度—时间对照表。同时读取井口坐标处的深度域地震道。

**2) 将深度域地震道转换到时间域。** 用上一步得到的对照表，把深度域地震道从深度轴插值到规则的时间轴上。至此，测井曲线和地震道都在时间域了，可以做常规的井震标定。

**3) 自动井震标定。** 用已有的标定模块在时间域内寻找最佳子波——这条子波描述了从波阻抗变化到地震振幅的转换关系。深度域和时间域共用同一个时间子波（子波本身定义在时间上），所以时间域提取的子波可以直接用于深度域正演。

**4) 裁剪和归一化。** 原始标定出的子波通常有几万个采样点，实际有用的能量集中在中心零点附近。脚本截取中心约 200 毫秒的片段，将其能量归一化到 1，输出为固定长度、可直接复用的子波文件。

另外生成五张质量检查图：深度与时间的对应关系、标定收敛曲线、原始标定窗口的波形对比、裁剪子波的合成记录与地震对比、子波的波形和频谱。

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

`wavelet_batch_synthetic_depth.py` 用第4步产出的固定子波，对全部井做批量合成记录，并导出按各井时移策略处理后的两套 LAS。

### 快速开始

```powershell
python scripts/wavelet_batch_synthetic_depth.py --config <config-yaml>
python scripts/wavelet_batch_synthetic_depth.py --config <config-yaml> --well <well-name>
python scripts/wavelet_batch_synthetic_depth.py --config <config-yaml> --output-dir scripts/output/wavelet_batch_synthetic_depth_<run-tag>
```

用 `--well` 可以只跑一口井调试。

### 配置参考

所有配置在 `wavelet_batch_synthetic_depth` 段下：

```yaml
wavelet_batch_synthetic_depth:
  source_runs:
    well_preprocess_dir:                 # 留空时自动发现最新的第3步产物
    vertical_well_auto_tie_depth_dir:    # 留空时自动发现最新的深度域第4步产物
  las_vp_unit: us/m
  las_rho_unit: g/cm3

  source_well_name: <source-well>          # 子波来源井名
  skip_shift_scan_well_names:             # 保持输入深度、不扫描时移的井
    - <well-with-untrusted-shift-a>
    - <well-with-untrusted-shift-b>

  shift_min_ms: -20.0                    # 时移扫描下限
  shift_max_ms: 20.0                     # 时移扫描上限
```

同样需要顶层 `seismic.domain: depth` + `seismic.depth_basis: tvdss`。

默认自动接上最新一次测井预处理结果和最新一次包含指定来源井产物的深度域第4步结果。复现实验时可按需填写 `source_runs` 下对应的运行目录固定输入。

脚本从发现的深度域第4步目录读取子波和 `run_summary_<source-well>.json`，并使用该次标定选出的测井中值滤波窗口、去尖峰阈值和高斯平滑标准差。摘要缺少任一参数时脚本直接报错。

### 脚本在做什么

第4步只跑了一口井、产出一条子波。第5步用这条子波跑全工区所有有测井数据的井，做两件事：评估子波在每口井上的适用性，以及把每口井的测井曲线在深度上平移以对齐地震。

**1) 每口井各自计算时深关系。** 和第4步一样，每口井用自身的速度曲线沿深度积分得到局部的深度—时间对照表，然后把井旁的深度域地震道换算到时间域。

**2) 时移扫描。** 固定子波不变。用测井曲线计算反射系数，与子波卷积得到合成记录，在 ±40 毫秒范围内逐档平移反射系数，每次平移后计算合成记录与实际地震道的相似度，选出相关系数最高的平移量。这一步的目的是找出测井和地震之间的整体时间偏差。

名单中的井只在零时移位置计算一次合成记录、相关系数和 NMAE，不生成扫描明细与扫描图。它们的深度平移曲线为零，两套 LAS 保持输入深度坐标；滤波版仍使用第4步继承的测井滤波参数。名单留空或写成空列表时，所有井都执行正常扫描。子波来源井不能放入该名单，未知井名和重复井名会使脚本直接报错。

**3) 时间偏差转换为深度偏差。** 因为每口井的速度随深度变化，时间偏差和深度偏差之间不是固定比例。脚本通过该井的深度—时间对照表，把每个时间采样点的最佳时间偏移反查为对应的深度偏移，得到一条深度平移曲线。

**4) 导出两套深度平移后的 LAS 文件。** 这是深度域工作流最重要的产出，供后续合成数据生成使用：

- **全曲线版**（`shifted_preprocessed_las`）：把第3步原始预处理 LAS 的全部曲线按深度平移曲线搬移到新位置，保留原始的数据空缺不填补。下游用它提取波阻抗的真实变化幅度。
- **滤波版**（`shifted_filtered_las`）：只包含声波时差、密度、波阻抗三条曲线，且经过了平滑滤波。下游用它构建背景趋势，避免个别尖刺干扰背景拟合。

**5) 质量检查。** 每口井都生成合成记录与地震道的波形对比图；执行扫描的井另外生成时移扫描相似度图。全工区汇总图使用实际采用的时移量，跳过扫描的井显示为零。脚本末尾还会检查子波来源井在批量合成中的时移量是否与第4步的标定结果一致。

跳过扫描只改变该井的深度平移策略，不改变井的发布资格。成功产出的井会继续进入第六步井控集、全部低频模型基线和 Synthoseis-lite。

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

`depth_forward_model_inputs.py` 将岩石物理分析产出的 AI–Vp 关系与深度域第 4 步产出的固定子波组装为统一的 `forward_model_inputs.json`，供 Synthoseis-lite 深度域、ablation 和 R1 正演诊断使用。

岩石物理分析和子波提取各自独立重跑，本旁路只在子波或关系发生变化时才需要重跑，避免更新子波时必须重跑整个岩石物理拟合。

### 快速开始

```powershell
python scripts/depth_forward_model_inputs.py
python scripts/depth_forward_model_inputs.py --config experiments/common/common.yaml
python scripts/depth_forward_model_inputs.py --output-dir scripts/output/depth_forward_model_inputs_test
```

不带参数时，脚本自动发现最新的岩石物理分析和深度域第 4 步产物。

### 配置参考

所有配置在 `depth_forward_model_inputs` 段下：

```yaml
depth_forward_model_inputs:
  source_runs:
    rock_physics_analysis_dir:           # 留空时自动发现最新的 rock_physics_analysis 产物
    vertical_well_auto_tie_depth_dir:     # 留空时自动发现最新的 vertical_well_auto_tie_depth 产物
  source_well_name: <well-name>           # 子波来源井名
```

同时需要顶层 `seismic.domain: depth` + `seismic.depth_basis: tvdss`。

### 脚本在做什么

1. 校验来源。确认岩石物理分析运行成功且 `ai_vp_linear` 模块拟合通过，确认深度域第 4 步运行成功且来源井名匹配。
2. 校验子波。读取子波 CSV，检查奇数长度、零时间居中，通过一次正向卷积验证子波可用。
3. 校验岩石物理关系。确认 `rock_physics_relation.json` 的 schema、公式、单位和系数合法，合格井清单非空无重复。
4. 冻结契约。将岩石物理关系路径、子波路径及参数、来源运行指纹写入 `forward_model_inputs.json`。

### 核心输出文件

| 文件 | 内容 |
|------|------|
| `forward_model_inputs.json` | 冻结的正演输入：子波路径与参数、AI–Vp 关系路径与系数、来源运行契约指纹 |
| `run_summary.json` | 来源运行发现方式、`source_well_name`、契约指纹 |

---

## 深度域与时间域的关键差异

前五步的差异已在上面各节说明——深度域第 4 步只跑一口井输出固定子波，第 5 步用固定子波做全井深度平移。从第 6 步开始，两个域的数据流汇入同一套 `cup` 模块，差异体现在合同参数上。

### 第 6 步：真实工区井控数据集

| | 时间域 | 深度域 |
|---|---|---|
| 上游步骤 | `well_auto_tie` | `wavelet_batch_synthetic_depth` |
| 输入 LAS | 时间对齐后的标定 LAS | 深度平移后的 LAS |
| `sample_domain` | `time` | `depth` |
| `sample_unit` | `s` | `m` |
| 轴间距 | ~2 ms | 5.0 m |
| `depth_basis` | 无 | `tvdss` |

其余逻辑完全一致：每口井从对齐后的曲线提取 AI，在 5 m 规则轴上重采样，产出 `canonical_background_log_ai` 和采样点掩码。

### 第 7 步：真实工区低频模型

关键差异：

| | 时间域 | 深度域 |
|---|---|---|
| 上游步骤 | 第 6 步 (`well_auto_tie`) | 第 6 步 (`wavelet_batch_synthetic_depth`) |
| `sample_domain` | `time` | `depth` |
| `sample_unit` | `s` | `m` |
| 轴间距 | ~2 ms | 5.0 m |
| `depth_basis` | 无 | `tvdss` |
| 滤波 cutoff 参数 | `cutoff_hz` (cycles/s) | `cutoff_wavelength_m` (m/cycle) |
| 滤波内部表示 | `cutoff_hz` 原值 | `1.0 / cutoff_wavelength_m` (cycles/m) |
| Log basis 标签 | `twt` | `tvdss` |

低通滤波是差异最集中的地方。`parse_lowpass_spec()` 根据 `sample_axis.domain` 校验配置中的 cutoff 参数名：时间域配置必须用 `cutoff_hz`，深度域配置必须用 `cutoff_wavelength_m`，错域参数直接失败。`filter.enabled: false` 时不接受任何 filter 参数。滤波实现 `apply_lfm_lowpass()` 内部统一转为 cycles-per-axis-unit 后走同一个 Butterworth 滤波，区别只在 Log 的 basis 标签。

`LfmContext` 初始化时强制校验：深度域必须 `depth_basis='tvdss'`，时间域 `depth_basis` 必须为 None。

其余逻辑完全一致：两种基线模型方法（trend 和 proportional_kriging）的构建流程、XY ordinary kriging 的空间插值、framework 修饰器的概率场叠加、变体图和 comparison 的生成，以及 volume 模式的 SEG-Y/ZGY 导出，都不区分域。

---

## 与时间域工作流的关系

```text
时间域主链：Step 1 → 2 → 3 → 4(well_auto_tie) → 5(wavelet_generation) → ...
                                          ↓
深度域旁路：Step 1 → 2 → 3 → 4(vawt_depth) → 5(wbs_depth)
                         ↓                        ↓
                    rock_physics ──────→ depth_forward_model_inputs
                                              ↓
                                              synthoseis_lite depth v4 → ablation
                                                                          ↓
                                                   统一井控 → LFM v3 → R0 → R1(TVDSS)
```
