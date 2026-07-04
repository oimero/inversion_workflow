# 深度域工作流

深度域工作流定位为**一次性处理**，复用概率低。第1步–3（井资产盘点、LAS 筛选、测井预处理）和旁路岩石物理分析与时间域共享，本文不再重复。

深度域特有的步骤只有两步：**固定子波提取**和**批量合成与深度平移**。二者是当前工区的遗留一次性路径，不属于标准工作流 API，也不作为后续统一正演重构的依赖。

---

## 总览

| 步骤 | 脚本 | 做什么 |
|------|------|--------|
| Step 4（深度域） | `scripts/vertical_well_auto_tie_depth.py` | 从指定井提取一条固定时间子波 |
| Step 5（深度域） | `scripts/wavelet_batch_synthetic_depth.py` | 用该子波做全井批量合成、时移扫描、导出深度平移后的两套 LAS |

产物关系：

```text
vertical_well_auto_tie_depth.py
  → wavelet_201ms_<source-well>.csv        （固定时间子波，供 Step 5 使用）

wavelet_batch_synthetic_depth.py
  → wavelet_batch_metrics.csv     （全井指标汇总）
  → shifted_preprocessed_las/     （深度平移后的 Step 3 全曲线 LAS）
  → shifted_filtered_las/         （深度平移后的 filtered DT_USM/RHO_GCC/AI）
  → shift_scans/                  （每井时移扫描明细）
  → depth_shift_curves/           （每井深度平移曲线）
  → synthetic_qc/                 （每井合成记录 QC）
  → figures/                      （每井 QC 图 + 批量汇总图）
```

两套 shifted LAS 的下游消费：

- `shifted_preprocessed_las/AI` → Synthoseis-lite depth v2 的 `full_log_ai` 和 truth 统计；
- `shifted_filtered_las/AI` → Synthoseis-lite depth v2 的 background fit，避免背景被尖刺支配。

---

## 第4步（深度域）：固定子波提取

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
  las_dir: <path-to-las-dir>        # Step 3 预处理 LAS 所在目录
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

## 第5步（深度域）：批量合成与深度平移

`wavelet_batch_synthetic_depth.py` 用第4步产出的固定子波，对全部井做批量合成记录、时移扫描，并导出深度平移后的两套 LAS。

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
  las_dir: <path-to-las-dir>            # Step 3 预处理 LAS 目录
  las_vp_unit: us/m
  las_rho_unit: g/cm3

  source_auto_tie_dir: scripts/output/vertical_well_auto_tie_depth_<timestamp>
  source_well_name: <source-well>          # 子波来源井名

  shift_min_ms: -40.0                    # 时移扫描下限
  shift_max_ms: 40.0                     # 时移扫描上限

  excluded_well_names: []                # 排除的井名列表

  fallback_log_filter:                   # 当日志过滤参数从 source run 读取失败时的回退
    logs_median_size: 7
    logs_median_threshold: 0.15
    logs_std: 0.15
```

同样需要顶层 `seismic.domain: depth` + `seismic.depth_basis: tvdss`。

脚本从 `source_auto_tie_dir` 读取子波和 `run_summary_<source-well>.json`，日志过滤参数优先从 autotie 的 `best_parameters` 继承（`logs_median_size`、`logs_median_threshold`、`logs_std`），读取失败时回退到 `fallback_log_filter`。

### 脚本在做什么

第4步只跑了一口井、产出一条子波。第5步用这条子波跑全工区所有有测井数据的井，做两件事：评估子波在每口井上的适用性，以及把每口井的测井曲线在深度上平移以对齐地震。

**1) 每口井各自计算时深关系。** 和第4步一样，每口井用自身的速度曲线沿深度积分得到局部的深度—时间对照表，然后把井旁的深度域地震道换算到时间域。

**2) 时移扫描。** 固定子波不变。用测井曲线计算反射系数，与子波卷积得到合成记录，在 ±40 毫秒范围内逐档平移反射系数，每次平移后计算合成记录与实际地震道的相似度，选出相关系数最高的平移量。这一步的目的是找出测井和地震之间的整体时间偏差。

**3) 时间偏差转换为深度偏差。** 因为每口井的速度随深度变化，时间偏差和深度偏差之间不是固定比例。脚本通过该井的深度—时间对照表，把每个时间采样点的最佳时间偏移反查为对应的深度偏移，得到一条深度平移曲线。

**4) 导出两套深度平移后的 LAS 文件。** 这是深度域工作流最重要的产出，供后续合成数据生成使用：

- **全曲线版**（`shifted_preprocessed_las`）：把第3步原始预处理 LAS 的全部曲线按深度平移曲线搬移到新位置，保留原始的数据空缺不填补。下游用它提取波阻抗的真实变化幅度。
- **滤波版**（`shifted_filtered_las`）：只包含声波时差、密度、波阻抗三条曲线，且经过了平滑滤波。下游用它构建背景趋势，避免个别尖刺干扰背景拟合。

**5) 质量检查。** 每口井生成两张图（合成记录与地震道的波形对比、时移扫描的相似度曲线），另外生成两张全工区汇总图（各井的相关性和时移量、深度平移量统计）。脚本末尾还会检查子波来源井在批量合成中的时移量是否与第4步的标定结果一致。

**7) 生成 QC 图。** 每井两张图：合成记录 vs 地震道波形对比（R1 风格六 panel）、时移扫描 corr 曲线。批量汇总两张图：全井指标柱状图、深度平移统计。

### 核心输出文件

| 文件 | 内容 |
|------|------|
| `wavelet_batch_metrics.csv` | 全井汇总：best shift、corr、NMAE、深度平移统计、产物路径 |
| `run_summary.json` | 输入路径、LAS 契约说明、成功/失败计数 |
| `shifted_preprocessed_las/<well>.las` | 深度平移后的 Step 3 全曲线 LAS |
| `shifted_filtered_las/<well>.las` | 深度平移后的 filtered DT_USM/RHO_GCC/AI |
| `shift_scans/shift_scan_<well>.csv` | 每井时移扫描明细（shift_s, corr, nmae, scale） |
| `depth_shift_curves/depth_shift_curve_<well>.csv` | 每井深度平移曲线（twt_s, tvdss_m, depth_shift_m） |
| `synthetic_qc/synthetic_qc_<well>.csv` | 每井最佳时移下的合成记录 QC |
| `figures/qc_01_batch_metric_summary.png` | 批量指标汇总（corr, NMAE, best shift） |
| `figures/qc_02_batch_depth_shift_summary.png` | 深度平移汇总（median + P10-P90） |
| `figures/qc_<well>_synthetic_vs_seismic.png` | 每井 R1 风格六 panel 波形 QC 图 |
| `figures/qc_<well>_shift_scan.png` | 每井时移扫描 corr 曲线 |

### 来源井一致性检查

脚本末尾会检查子波来源井在批量合成中的 best shift 是否与 autotie 阶段的 `table_t_shift` 一致。差异过大会在终端醒目提示。

---

## 深度域与时间域的关键差异

### 合成旁路（Synthoseis-lite）

深度域 v3 与时间域 v3 共享同一入口脚本 `scripts/synthoseis_lite.py` 和 `synthoseis_lite_v3` 数据模式，通过 `sample_domain` 配置分派：

```yaml
synthoseis_lite:
  sample_domain: depth        # time | depth
  benchmark_schema: synthoseis_lite_v3
```

主要差异：

| 维度 | 时间域 v3 | 深度域 v3 |
|------|-----------|-----------|
| 采样轴 | TWT (ms) | TVDSS (m)，向下为正 |
| 井曲线来源 | Step 4 filtered LAS + Step 5 全局子波 | Step 5 `shifted_filtered_las/AI` + `shifted_preprocessed_las/AI`、Step 6 冻结子波和 AI–Vp 关系 |
| 可用套件 | canonical, field_conditioned, frequency_probe, seismic_variant | 仅 `field_conditioned`；canonical 和 probe 关闭 |
| 模型轴 | TWT 方向 | 工区原生 5 m + 8× 高分轴 |
| 空间路径 | inline/xline 索引 | 显式 inline/xline 折线路径 |
| 校准依赖 | Step 4/5/6 时间域来源 | Step 1 + Step 5（深度域）+ Step 6 |
| mismatch 深度静差 | 不支持（只有时间方向平移） | 独立的米制深度静差，与秒制子波平移相互独立 |
| mismatch 额外残余 | — | `residuals/residual_vs_lfm_ideal`、`residuals/residual_vs_lfm_controlled_degraded` |
| LFM over_smoothing | 秒制 | 米制 |
| Hz 低通字段 | 支持 | 禁止 |

深度域 mismatch 的完整扰动链：扰动时间子波 → 深度正演 → 米制深度静差 → gain → noise。时间子波相位旋转和秒制平移独立于米制深度静差，三个维度可交叉组合。

### 正演内核

| 维度 | 时间域 | 深度域 |
|------|--------|--------|
| 正演算子 | 平稳卷积（子波宽度恒定） | 非平稳矩阵（子波米制宽度随速度变化） |
| 正演函数 | `forward_time` | `forward_depth` |
| 反射率 | 下界面挂点 | 下界面挂点 |
| 输出 | N 点 | N 点 |
| 坐标 | TWT | TVDSS，通过梯形慢度积分转相对 TWT，再套用时间子波 |

核心实现在 `src/cup/physics/`。

### 工作流配置

深度域强制：

- `seismic.domain: depth`
- `seismic.depth_basis: tvdss`
- 米制边界使用 `_m` 后缀（`margin_top_m`、`margin_bottom_m`），不使用 `_ms`
- 错域字段直接报错，不静默忽略

### 体数据几何

当前深度域地震体 inline 步长 1、xline 步长 4。所有体数据抽取必须通过显式 iline/xline 轴和 `SurveyLineGeometry` 计算数组位置，不得假设步长为 1 或把线号差直接当下标。

---

## 与时间域工作流的关系

```text
时间域主链：Step 1 → 2 → 3 → 4(well_auto_tie) → 5(wavelet_generation) → ...
                                          ↓
深度域旁路：Step 1 → 2 → 3 → 4(vawt_depth) → 5(wbs_depth) → 旁路(rock_physics)
                                                              ↓
                                              synthoseis_lite depth v2 → GINN v2
```

深度域第4步/5 与时间域第4步/5 **互不依赖**、**平行存在**：

- 时间域第4步做全井自动标定，输出每井候选子波；
- 深度域第4步只跑一口指定井，输出一条固定子波；
- 时间域第5步做全井交叉评测和共识子波优化；
- 深度域第5步用固定子波做全井时移扫描和深度平移 LAS 导出。
