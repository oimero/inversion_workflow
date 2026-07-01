# 深度域工作流

深度域工作流定位为**一次性处理**，复用概率低。Step 1–3（井资产盘点、LAS 筛选、测井预处理）和 Step 6（岩石物理分析）与时间域共享，本文不再重复。

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

## Step 4（深度域）：固定子波提取

`vertical_well_auto_tie_depth.py` 从指定井提取一条固定时间子波，供后续批量合成使用。它**不是**通用井震标定——只跑一口井、输出一条子波。

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

脚本的核心流程（`run_auto_tie` 函数，[vertical_well_auto_tie_depth.py:319-598](scripts/vertical_well_auto_tie_depth.py#L319-L598)）：

**1) 加载井数据和井旁地震道。** 从来源井的 Step 3 预处理 LAS 中读取 Vp、Rho 曲线，对 NaN 做线性插值；通过 `survey.read_trace_at_xy(well_x, well_y, domain="depth")` 读取井口 XY 处的深度域地震道。

**2) 计算重叠窗口并建立局部时深关系。** 将 MD 减去 KB 得到 TVDSS，取井 TVDSS 与地震深度轴的重叠范围。在重叠窗口内，由 Vp 做梯形慢度积分构建局部时深表（TVDSS ↔ 相对 TWT），调用 `grid.build_local_tdt_from_vp`（`wtie.processing.grid`）。然后将深度域地震道通过局部 TDT 插值到规则 TWT 轴。

**3) 运行 autotie 提取子波。** 将 LogSet（含 Vp、Rho）、TWT 域地震道和局部 TDT 组装为 `wtie` 的 `InputSet`，调用 `autotie.tie_v1` 做自动井震标定，输出 raw wavelet。

**4) 裁剪并能量归一化。** 将 raw wavelet 裁剪到 `target_crop_ms`（默认 201 ms，奇数点），中心在零时刻，L2 能量归一化到 1。最终输出 `wavelet_201ms_<source-well>.csv`。

**5) 生成 QC 图。** 共 5 张图：深度-时间匹配、优化目标函数、raw tie window、裁剪后合成 vs 地震、子波时域/频域对比。

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

## Step 5（深度域）：批量合成与深度平移

`wavelet_batch_synthetic_depth.py` 用 Step 4 产出的固定子波，对全部井做批量合成记录、时移扫描，并导出深度平移后的两套 LAS。

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

主循环 `process_well`（[wavelet_batch_synthetic_depth.py:682-920](scripts/wavelet_batch_synthetic_depth.py#L682-L920)）对每口井执行以下步骤：

**1) 加载井数据并取地震道。** 读取该井的 Step 3 预处理 LAS，对 Vp、Rho 做 NaN 线性插值。读取井口 XY 处的深度域地震道。计算井 TVDSS 与地震深度轴的重叠窗口。

**2) 建立局部时深关系并转换到 TWT。** 由重叠窗口内的 Vp 积分构建局部时深表（`grid.build_local_tdt_from_vp`），将滤波后的 MD 域测井曲线通过 TDT 转到 TWT 域，计算反射系数。

**3) 时移扫描。** 在 `[shift_min_ms, shift_max_ms]` 范围内按子波采样间隔扫描整体时移。对每个时移量：平移反射系数 → 与固定子波卷积 → 计算与归一化地震道的 corr、NMAE。选出相关系数最高的时移量。

**4) 计算深度平移曲线。** 将最佳 TWT 时移通过局部 TDT 转换为 TVDSS 深度平移（`compute_depth_shift_curve`，[wavelet_batch_synthetic_depth.py:232-241](scripts/wavelet_batch_synthetic_depth.py#L232-L241)）。对 TWT 轴上的每个样点，查 TDT 得到当前位置的 TVDSS，再查 TDT 得到时移后位置的 TVDSS，差值即为深度平移量。输出 `depth_shift_curve_<well>.csv`。

**5) 导出 shifted_preprocessed_las。** `export_shifted_preprocessed_las`（[wavelet_batch_synthetic_depth.py:405-521](scripts/wavelet_batch_synthetic_depth.py#L405-L521)）读取原始 Step 3 LAS，对所有数值曲线按深度平移曲线做 TVDSS 方向的平移，重新插值到规则 MD 网格。保留原始间隙结构（不桥接 null gap）。当源 LAS 有 DT_USM 和 RHO_GCC 时，在输出 LAS 中重新计算 AI。

**6) 导出 shifted_filtered_las。** `build_shifted_filtered_logset_for_export`（[wavelet_batch_synthetic_depth.py:524-574](scripts/wavelet_batch_synthetic_depth.py#L524-L574)）对 Step 5 内部滤波后的 LogSet 做同样的深度平移，输出 DT_USM、RHO_GCC、AI 三条曲线。

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

深度域 v2 与时间域 v2 共享同一入口脚本 `scripts/synthoseis_lite.py` 和 `synthoseis_lite_v2` schema，通过 `sample_domain` 配置分派：

```yaml
synthoseis_lite:
  sample_domain: depth        # time | depth
  benchmark_schema: synthoseis_lite_v2
```

主要差异：

| 维度 | 时间域 v2 | 深度域 v2 |
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
深度域旁路：Step 1 → 2 → 3 → 4(vawt_depth) → 5(wbs_depth) → 6(rock_physics)
                                                              ↓
                                              synthoseis_lite depth v2 → GINN v2
```

深度域 Step 4/5 与时间域 Step 4/5 **互不依赖**、**平行存在**：
- 时间域 Step 4 做全井自动标定，输出每井候选子波；
- 深度域 Step 4 只跑一口指定井，输出一条固定子波；
- 时间域 Step 5 做全井交叉评测和共识子波优化；
- 深度域 Step 5 用固定子波做全井时移扫描和深度平移 LAS 导出。

Step 6（岩石物理分析）对两个域提供同一份 `forward_model_inputs.json`，消费的是 Step 3 的原始 MD LAS，不依赖 Step 4/5 的结果。
