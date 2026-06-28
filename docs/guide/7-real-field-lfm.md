# 07 实际工区 LFM 输入准备

`real_field_lfm.py` 是主链第七步。它把第四步的井震标定结果冻结为一份三维波阻抗低频模型，供第八步零样本推理使用。这一步不涉及训练或反演——它只做一件事：**把每口井的波阻抗趋势拟合成两个空间参数场，再用它们重建一张光滑的低频背景。**

---

## 快速开始

```bash
python scripts/real_field_lfm.py
python scripts/real_field_lfm.py --config experiments/common/common.yaml
python scripts/real_field_lfm.py --output-dir scripts/output/real_field_lfm_test
```

不带参数时，脚本自动发现最新的第四步和第一步产物，在 `scripts/output/real_field_lfm_<timestamp>/` 下写出结果。

---

## 运行前需要什么

第七步依赖以下上游产物：

| 来源 | 文件 | 用途 |
|------|------|------|
| 第四步 | `well_tie_metrics.csv` | 标定成功的井列表、优化时深表路径、滤波 LAS 路径 |
| 第四步 | `well_tie_plan.csv` | 井的路由信息 |
| 第四步 | `wavelet_inventory.csv` | 候选子波清单（用于来源校验） |
| 第一步 | `well_inventory.csv` | 井口坐标（inline/xline）和井型 |
| 数据目录 | 地震体 | 提供输出网格的几何参数（inline/xline/TWT 轴），不参与数值拟合 |

---

## 配置参考

```yaml
# 以下字段写在 experiments/common/common.yaml 的顶层或 real_field_lfm 段下。

# --- 必填 ---
target_interval:
  horizons:
    - {name: <top>, file: <horizon-file>}
    - {name: <bottom>, file: <horizon-file>}
    # 中间层位可选，只用于 QC

seismic:
  file: <seismic-volume>
  type: zgy
  domain: time

# --- 可选（有默认值）---
real_field_lfm:
  output_geometry:
    mode: volume                    # 默认 volume
    target_context_s: 0.05          # 默认 0.05

  trend_fit:
    min_valid_samples_per_well: 32  # 默认 32
    huber_f_scale_log_ai: 0.05      # 默认 0.05

  parameter_modeling:
    min_wells: 3                    # 默认 3
    allow_constant_fallback: false  # 默认 false
    variogram: spherical            # 默认 spherical，可选 exponential / gaussian
    nugget: 0.0                     # 默认 0.0

  lfm_qc:
    min_time_diff_rms: 1.0e-4       # 默认 1.0e-4
    min_trace_time_std_median: 1.0e-4  # 默认 1.0e-4
```

### `target_interval.horizons`

从浅到深排列的层位列表，长度至少为 2。首层位和末层位之间的范围是完整目标窗，中间层位只用于 QC 和连续性检查，不拆分拟合。

### `output_geometry`

| 参数 | 含义 |
|------|------|
| `mode` | `volume`（全工区三维）或 `section`（单剖面）。目前主要使用 volume |
| `target_context_s` | 在顶层位之上和底层位之下各额外包含多少秒的输出上下文 |

### `trend_fit`

控制每口井的趋势拟合行为。脚本用 Huber 回归对全目标窗拟合 `logAI(u) = a + b(2u-1)`，其中 `u` 是层位间的归一化比例坐标。`min_valid_samples_per_well` 是单井最少需要的有效 TWT 样点数；`huber_f_scale_log_ai` 控制 Huber 损失的离群点阈值。

### `parameter_modeling`

控制 a、b 两个系数的空间参数场建模。脚本用 ordinary kriging（基于真实 XY 米制坐标）插值两个系数场。`min_wells` 控制最少需要多少口趋势合格的井；不足时运行失败，除非显式打开 `allow_constant_fallback`。`variogram` 可选 `spherical`、`exponential` 或 `gaussian`。

### `lfm_qc`

输出 LFM 的基本结构检查阈值。`min_time_diff_rms` 和 `min_trace_time_std_median` 用于检测 LFM 是否过于平坦——如果低于阈值，标记 `lfm_time_flat_or_invalid`。

---

## 脚本在做什么

脚本分四个阶段：**趋势拟合 → 空间参数场 → LFM 重建 → QC 输出**。

### 第一阶段：逐井趋势拟合

对每口第四步标定成功的井：

1. 从第四步 `filtered_las_file` 读取波阻抗，经 `optimized_tdt_file` 投影到 TWT 轴。投影使用分段线性 cell-average，不跨 LAS 无效段或 TDT 长缺口。
2. 在顶底层位之间的完整目标窗内，用 Huber 回归拟合：
   ```
   logAI(u) = a + b(2u - 1)
   ```
   其中 `u` 是目标窗内的归一化比例坐标（0 到 1）。
3. 每口井只产出一组 `(a, b)` 系数。斜井使用优化后落道计划中的轨迹坐标，取 TWT-cell-length 加权平均位置作为该井的代表位置。
4. 趋势合格的井进入空间建模；不合格的井记录拒绝原因后跳过。

结果写入 `well_trend_controls.csv`——每口井一行，记录 `a`、`b`、代表位置（真实 XY 米制坐标）、拟合样点数和状态。

### 第二阶段：空间参数场建模

对 trend fit 得到的 `a` 和 `b` 分别做 ordinary kriging 空间插值：

1. 使用真实 XY 米制坐标（不是 inline/xline 线号），从控制井的最近邻距离中位数估计 range hint。
2. 对 `a` 和 `b` 使用同一个 range hint，避免两个场有不同的平滑尺度。
3. 输出 `a_field` 和 `b_field`——两张覆盖全工区网格的二维参数场，以及对应的 kriging variance。

结果写入 `parameter_field_qc.csv`，记录控制井数、参数范围、range hint、kriging variance 的 P50/P95、离最近控制井的距离 P50/P95、网格在控制点凸包外的比例。

### 第三阶段：LFM 重建

用参数场重建三维低频模型：

```
lfm_log_ai(inline, xline, twt) = a_field(inline, xline) + b_field(inline, xline) * (2u - 1)
```

其中 `u` 在每个空间位置由顶底层位计算。LFM 只在目标窗内有效——窗外为 NaN，对应 `valid_mask_model = false`。

同时输出 `lfm_support_mask`（该位置离控制井是否足够近）和 `distance_to_control`（到最近控制井的距离），作为空间外推风险 QC。中间层位处的 LFM 连续性也会被检查并写入 `internal_horizon_continuity_qc.csv`。

### 第四阶段：QC 与导出

输出 LFM 的基础统计（全局 RMS、每道时间方向标准差、时间差分 RMS、横向标准差），并与合成训练集的 LFM 输入域做对照。生成参数场 QC 图、井旁趋势对照图、距离-控制图等。可选导出 ZGY 格式。

---

## 核心输出文件

所有文件在 `<output_root>/real_field_lfm_<timestamp>/` 下：

### 主数据

| 文件 | 内容 |
|------|------|
| `real_field_lfm.npz` | 三维 LFM，包含 `log_ai`、`valid_mask_model`、`lfm_support_mask`、`distance_to_control`、`a_field`、`b_field`、坐标轴和 metadata |
| `real_field_lfm_ai.zgy` | 供解释软件使用的线性 AI 体，数值为 `exp(log_ai)`；内部 NPZ 仍保持 log(AI) |
| `real_field_lfm_summary.json` | 来源路径、控制井统计、LFM 统计、输出文件清单和 SHA-256 |

### 井趋势

| 文件 | 内容 |
|------|------|
| `well_trend_controls.csv` | 每口井一行：`a`/`b` 系数、代表位置、拟合样点数、残差 RMS、状态和拒绝原因 |

### 参数场 QC

| 文件 | 内容 |
|------|------|
| `parameter_field_qc.csv` | a/b 参数场的空间建模状态：控制井数、参数范围、range hint、kriging variance、distance-to-control、凸包外比例 |

### 层位与连续性 QC

| 文件 | 内容 |
|------|------|
| `horizon_qc.csv` | 目标窗与中间层位的有效率、厚度统计、交叉道数、超出 TDT 支持的井数 |
| `internal_horizon_continuity_qc.csv` | 每个中间层位上下相邻样点的 LFM 差异，检查是否存在人工跃变 |

### 图件

| 文件 | 内容 |
|------|------|
| `figures/well_trend_controls.png` | 每口井的滤波 LAS、拟合趋势和采样 LFM 对照 |
| `figures/parameter_field_qc.png` | `a_field` / `b_field` / kriging variance / distance-to-control 四面板 |
| `figures/lfm_reconstruction_qc.png` | 代表 inline/xline 剖面的 LFM 和有效掩码 |

---

## 如何阅读结果

### 第一步：看终端输出

```
=== Real-field LFM ===
Output: scripts/output/real_field_lfm_<timestamp>
Status: ok
Accepted controls: 15 / 16
```

分母是第四步标定成功的井数，分子是通过了趋势拟合和参数场建模的井数。如果分子比分母少，看 `well_trend_controls.csv` 了解每口井的具体拒绝原因。

### 第二步：看 `real_field_lfm_summary.json`

- `status` 为 `ok` 表示 LFM 通过了基本结构检查。
- `lfm_stats` 中的 `trace_time_std_median` 和 `time_diff_rms` 应显著大于零。如果这些值异常低，说明 LFM 过于平坦——可能是控制井太少或 `a`/`b` 系数几乎没有空间变化。

### 第三步：看 `parameter_field_qc.csv`

- `outside_control_hull_fraction`：网格在控制点凸包外的比例。如果这个比例很高（如 > 0.3），说明有大面积区域没有任何附近的控制井，LFM 在这些区域是纯外推。
- `distance_to_control_p95_m`：95% 的网格点到最近控制井的距离。如果很大，空间可信度低。

### 第四步：看 `well_trend_controls.csv`

按 `status` 过滤，关注被拒绝的井。常见拒绝原因：目标窗内有效样点不足、趋势拟合失败、缺少滤波 LAS 或 TDT。

### 第五步：看图

- `figures/parameter_field_qc.png` — a/b 参数场的空间分布是否平滑合理，kriging variance 在远离井的区域是否急剧升高。
- `figures/well_trend_controls.png` — 抽查几口井，拟合的直线趋势是否合理穿过滤波 LAS 的 TWT 域投影。

---

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| `well_auto_tie_dir` 无法自动发现 | 第四步产物不存在或不符合格式 | 确认第四步已成功运行，或在 `source_runs` 中显式指定 |
| `well_inventory_file` 无法自动发现 | 第一步产物不存在 | 确认第一步已运行 |
| `insufficient_control_wells` | 趋势拟合合格的井数小于 `min_wells` | 检查第四步标定成功率和 `well_trend_controls.csv` 中的拒绝原因 |
| `lfm_time_flat_or_invalid` | LFM 在时间方向过于平坦 | 控制井的 b 系数可能都接近零，或控制井太少导致参数场过于平滑 |
| 所有井趋势拟合失败 | `filtered_las_file` 或 `optimized_tdt_file` 缺失，或目标窗内无有效样点 | 检查第四步的输出完整性 |

---

## 留到第二轮

- 支持剖面模式（`section`）作为体积模式的替代。
- 支持多个低频模型假设（如不同变差函数模型、不同插值半径）的敏感性分析。
- 支持 `a`/`a-b`/`a+b` 三个端点的显式值域校验（当前只做基本范围检查）。
