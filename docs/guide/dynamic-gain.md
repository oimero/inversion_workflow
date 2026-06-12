# Dynamic Gain：让合成记录回到观测地震的尺度

GINN 训练端用能量归一化的子波做正演，合成记录的振幅和真实地震不在一个量级上。`dynamic_gain.py` 负责生成一个随时间和空间变化的增益体，把单位子波正演结果拉回到观测地震的尺度。它不是主链步骤，跑在第七步 LFM 之后、第八步 GINN 训练之前，由人根据产出证据决定第八步用 fixed gain 还是 dynamic gain。

---

## 快速开始

```bash
python scripts/dynamic_gain.py
python scripts/dynamic_gain.py --train-config experiments/ginn/train.yaml --config experiments/common.yaml
python scripts/dynamic_gain.py --output-dir scripts/output/dynamic_gain_test
```

不带参数时，脚本自动发现最新的第六、七步产物和第五步全局子波，按第八步训练配置重建归一化口径，在 `scripts/output/dynamic_gain_<timestamp>/` 下写出结果。

---

## 运行前需要什么

| 来源 | 文件/配置 | 用途 |
|------|-----------|------|
| 地震数据 | 与第八步相同的地震体和 SEG-Y 几何配置 | 读取原始地震、确定时间轴和工区几何 |
| 第四步 | `filtered_las/filtered_logs_<well>.las`、`time_depth/optimized_tdt_<well>.csv` | 井上波形 QC 的波阻抗曲线来源 |
| 第五步 | `selected_wavelet.csv` | 单位子波正演 |
| 第六步 | `log_ai_anchor_time.npz` | 井上 gain 样本来源 |
| 第七步 | `ai_lfm_time.npz` | LFM 体、目标层元数据、层位路径 |
| 训练配置 | 与第八步一致的 mask、验证划分、子波、LFM 和 anchor 配置 | 复现第八步的训练 mask 和归一化 RMS |

---

## 配置参考

脚本配置放在 `dynamic_gain` 段下：

```yaml
spatial_debias:
  cluster_radius_m: 600.0

dynamic_gain:
  segments:
    min_segment_valid_samples: 8
    max_segment_count_per_trace: 20
    min_segments_per_well: 1
    gain_eps: 1e-12

  spatial_debias:
    enabled: true

  attributes:
    candidate_attributes: [seismic_rms, seismic_abs_mean, seismic_abs_p90]
    attr_tie_threshold: 0.05
    attribute_floor_fraction: 0.10
    window_s: null

  prediction:
    clip_percentiles: [5.0, 95.0]
    gain_smoothing_samples: 1

  runtime:
    forward_batch_traces: 256
    volume_batch_traces: 512
```

### `source_runs`

默认自动接上最新的第四步、第五步、第六步和第七步产物。第四步只在井上波形 QC 时使用；gain 估计逻辑不依赖第四步。复现实验时，可以按需加入 `source_runs` 并填写对应步骤目录。

### `segments`

控制从锚点道上切出 gain 估计段的粒度。`min_segment_valid_samples` 是每段最少有效样点数；`max_segment_count_per_trace` 限制单道最多切多少段。`gain_eps` 是正则化小量，防止合成记录能量为零时除零。

### `spatial_debias`

密井平台上的十几口井如果各算一票，gain 推荐值会被这个平台主导。空间去偏先把井口 XY 近的井聚成空间簇，再按簇聚合。半径来自顶层 `spatial_debias.cluster_radius_m`；`enabled: false` 时每口井各自为簇。

### `attributes`

`candidate_attributes` 是候选地震属性列表，当前支持三种。`attr_tie_threshold` 控制属性选择的保守程度：如果另一个属性的 Pearson |r| 比 RMS 高出这个阈值以上，脚本才会选它，否则优先用 RMS。`attribute_floor_fraction` 是属性下限的安全因子——在预测 gain 体时，任何属性值低于全局 P01 的这个倍数时会被 clamp 到安全下限。`window_s` 是滑动窗口秒数，不填则用井样本段长的中位值。

### `prediction`

`clip_percentiles` 是 gain 预测值的 clip 上下百分位，防止极端预测值失控。`gain_smoothing_samples` 是生成 gain 体后的沿时间轴轻量平滑窗口（奇数样点，1 表示不做平滑）。

### `runtime`

`forward_batch_traces` 控制单位子波正演时的批大小；`volume_batch_traces` 控制生成 gain 体时的批大小。减少它们可以降低显存占用。

---

## 脚本在做什么

脚本分五个阶段：**归一化与正演 → 井上 gain 样本 → 空间去偏与固定 gain → 属性选择与拟合 → gain 体生成与导出**。主要计算逻辑在 `src/cup/seismic/gain.py` 中，脚本负责 CLI、上下文组装和 QC 图绘制。

### 第一阶段：归一化与正演

按第八步训练配置重建目标层 mask 和 loss mask，计算 `train_mask_rms`——目标层训练 mask 内全部有限地震样点的 RMS。然后把原始地震按 `seismic_raw / train_mask_rms` 归一到单位量级。

归一化之后，用第五步全局子波（不乘任何增益）对 LFM 做一次完整正演，得到单位子波合成记录。后续所有 gain 估计都在这个归一化域里进行。

### 第二阶段：井上 gain 样本估计

在第六步 anchor 的控制井道上，对每个有效段用最小二乘估计局部 gain——给定一段归一化观测地震和一段单位合成记录，求一个正数增益使残差平方和最小。

一口井可能切出多段，每段一个 gain。段内的有效样点必须同时满足三个条件：在 anchor mask 内、在 loss mask 内、地震和合成记录都是有限值。太短的连续有效段会被跳过。

### 第三阶段：空间去偏与固定增益推荐

对井 gain 样本按井口 XY 做空间聚类，然后给出一个 recommended fixed gain：先取每段的中位 gain，再取每口井的中位，再取每个空间簇的中位，最后取全局中位。这个值是人做决策时的基线——如果不使用 dynamic gain，第八步填这个数。

### 第四阶段：属性选择与对数拟合

核心是用一条简单的对数-对数关系从地震振幅属性预测增益：`ln(gain) = a + b * ln(attribute)`。

属性是地震归一化振幅的局部移动窗统计量，本质上回答"这个位置的观测地震强不强"。三种候选属性各有侧重：RMS 最常用，绝对均值更稳健，P90 对强反射更敏感。脚本在井样本上比较它们和 gain 的相关性，RMS 有轻微偏好加权。

选定属性后做一次最小二乘对数拟合。同时从井样本的 gain 分布中确定上下 clip 百分位，作为后续生成 gain 体的安全边界。

### 第五阶段：gain 体生成与导出

对全体地震道计算同一属性、套用对数关系、clip 到安全范围、做轻量平滑，生成完整的三维增益体。输出的 gain 体必须是正值有限值；任何非正值会被替换为 clip 上下界的几何平均。

同时输出 `recommended_fixed_gain.json`、一系列 QC 图和井上波形 QC，供人判断第八步用 fixed 还是 dynamic。

---

## 核心输出文件

所有文件在 `<output_root>/dynamic_gain_<timestamp>/` 下：

### 主输出

| 文件 | 内容 |
|------|------|
| `dynamic_gain.npz` | 正值 gain 体，供第八步 `gain_source: dynamic_gain_model` 读取 |
| `recommended_fixed_gain.json` | 空间去偏后的推荐固定增益 |
| `dynamic_gain_samples.csv` | 井上 gain 样本明细，含属性和拟合用字段 |
| `dynamic_gain_summary.json` | 完整输入、归一化参数、拟合结果和输出路径 |

### QC 输出

| 文件 | 内容 |
|------|------|
| `figures/qc_01_gain_distribution.png` | 井上 gain 段分布直方图 |
| `figures/qc_02_attribute_fit.png` | 选定属性的对数散点图与拟合线 |
| `figures/qc_03_attribute_metrics.png` | 候选属性的 Pearson 相关系数柱状图 |
| `figures/qc_04_spatial_debias.png` | 空间簇的井增益与簇增益对比 |
| `figures/qc_05_dynamic_gain_volume.png` | 增益体的 inline、xline 和时间切片 |
| `well_qc/figures/anchor_trace_*.png` | 每道 anchor 的井上波形 QC 图（amplitude 模式） |
| `well_qc/traces/anchor_trace_*.csv` | 地震、合成、gain、波阻抗和反射系数的逐样点明细 |
| `well_qc/well_qc_metrics.csv` | 逐 anchor 道波形 QC 指标（corr、MAE、RMSE、bias、RMS ratio） |

### `dynamic_gain.npz`

使用 `dynamic_gain_v1` schema，包含：

| 键 | 形状 | 语义 |
|----|------|------|
| `volume` | `(n_inline, n_xline, n_sample)` | 正值 dynamic gain |
| `samples` / `inline` / `xline` | 一维轴 | TWT 采样轴和线号轴 |
| `geometry_json` | 标量 | 与第八步地震几何一致 |
| `metadata_json` | 标量 | 归一化口径、拟合参数、上游路径 |

### `recommended_fixed_gain.json`

| 字段 | 含义 |
|------|------|
| `recommended_fixed_gain` | 空间去偏后的井上 gain 中位数 |
| `n_wells` / `n_segments` / `n_spatial_clusters` | 参与估计的井数、段数和空间簇数 |
| `normalization` | 固定为 `seismic_raw_divided_by_train_mask_rms` |
| `gain_reference` | 固定为 `unit_wavelet_synthetic_to_normalized_observation` |
| `train_mask_rms` | 与第八步训练端一致的地震 RMS |

---

## 如何阅读结果

### 第一步：看图说话

不用看任何数字，先看三张图：

- `qc_02_attribute_fit.png`：散点是否沿拟合线集中。如果对数散点是一团散沙、看不出线性趋势，dynamic gain 就不可信——直接用 fixed。
- `qc_01_gain_distribution.png`：井之间的 gain 差异有多大。如果分布集中在一个窄峰附近，fixed 够用；如果明显分散甚至双峰，dynamic 值得试。
- `qc_04_spatial_debias.png`：空间簇之间的 gain 是否有系统性差异。如果某个区域的井增益整体偏高或偏低，说明振幅空间变化大，dynamic 可能更有价值。

### 第二步：看 `recommended_fixed_gain.json`

`recommended_fixed_gain` 就是如果不用 dynamic 时该填的值。同时关注 `n_spatial_clusters`——如果只有 1-2 个簇，说明空间去偏意义不大（井分布太集中或工区太小）；如果簇数很多但各簇增益接近，说明振幅空间变化不显著。

### 第三步：看体切片

`qc_05_dynamic_gain_volume.png` 三张切片用于排查异常。关注：是否有明显的条带（可能是采集脚印）、边界突变（检查 clip 比例）、孤立的高增益或低增益区（检查对应位置的原始地震是否是噪声）。

### 第四步：必要时看 samples 表

`dynamic_gain_samples.csv` 是井上估计的所有 gain 段明细。可以按 well_name 分组，看同一口井不同深度的 gain 是否一致、跨井的 gain 范围和井位空间分布有没有关系。

---

## 第八步接入

使用 dynamic gain：

```yaml
gain_source: dynamic_gain_model
dynamic_gain_model: scripts/output/dynamic_gain_<timestamp>/dynamic_gain.npz
include_dynamic_gain_input: true
in_channels: 4
fixed_gain: null
```

使用推荐固定增益：

```yaml
gain_source: fixed_gain
fixed_gain: <recommended_fixed_gain>
include_dynamic_gain_input: false
in_channels: 3
dynamic_gain_model: null
```

---

## 留到第二轮

- 属性候选池扩展到更多地震属性（瞬时振幅、相对波阻抗等）。
- 增益体在井控之外区域的验证——当前没有盲井测试，完全依赖属性→gain 关系的外推可信度。
- dynamic gain 和 fixed gain 的量化对比——目前只提供了图件证据，没有一个自动化的 A/B 对比指标。
- 按层段分别拟合 gain 关系的可能性——有些目标层振幅空间变化由沉积相控制，一个全局关系可能不够。
