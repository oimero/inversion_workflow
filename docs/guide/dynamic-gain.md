# 时间域 Dynamic Gain 旁路设计

本文是时间域 `dynamic_gain.py` 的实现规格。它不是主链步骤，也不自动决定第八步一定使用 dynamic gain；它只负责在第七步 LFM、第五步全局子波和 GINN 同口径地震归一化下，生成一个可选的随样点变化 gain 体，并顺手给出一个井上推荐 fixed gain。

---

## 位置与目标

运行位置：第七步 `lfm_precomputed.py` 之后，第八步 `ginn_train.py` 之前。

核心目标：

```text
单位子波正演 LFM × gain(t, inline, xline)
  ≈ GINN 训练端看到的归一化观测地震
```

这里的“归一化观测地震”必须和第八步一致：

```text
seismic_norm = seismic_raw / train_mask_rms
```

`train_mask_rms` 只在目标层训练 mask 内计算，不做逐道 zscore，不做逐道去均值，也不使用深度域旧脚本里的 per-trace 标准化口径。

---

## 输入

| 来源 | 文件/配置 | 用途 |
|------|-----------|------|
| 地震数据 | 与第八步相同的时间域地震体和 SEG-Y 几何配置 | 读取 raw seismic、inline/xline/time 轴 |
| 第五步 | `selected_wavelet.csv` | 单位子波正演 |
| 第六步 | `log_ai_anchor_time.npz` | 第一版井上 fixed gain 样本来源 |
| 第七步 | `ai_lfm_time.npz` | LFM 体、目标层 metadata、层位路径和目标层 QC 参数 |
| 配置 | 与第八步一致的 validation / mask 口径 | 复现第八步训练 mask 和 `train_mask_rms` |

脚本不得直接从 LAS、TDT 或井轨迹重建井约束。第一版只读取第六步已经生成的 `log_ai_anchor_time.npz`；若后续需要从 `well_constraint_points.csv` 取更细的点级事实，应作为新的输入契约显式加入。

---

## 输出

主输出：

| 文件 | 内容 |
|------|------|
| `dynamic_gain.npz` | 正值 gain 体，供第八步 `gain_source: dynamic_gain_model` 读取 |
| `recommended_fixed_gain.json` | 用井上样本估计的推荐 fixed gain |
| `dynamic_gain_summary.json` | 输入路径、归一化参数、拟合参数、样本数量和输出路径 |
| `dynamic_gain_samples.csv` | 井上或控制点上的 gain 样本、地震属性和拟合用字段 |
| `figures/*.png` | gain 曲线、属性关系、体切片等人工 QC 图 |

不输出常数 gain volume。fixed gain 只写在 `recommended_fixed_gain.json` 中，由人工决定是否填入第八步配置。

脚本也不自动判定 dynamic gain 好坏。可以输出相关系数、残差、样本数、clip 比例等 QC 指标，但不应自动切换为 fixed gain，也不应在配置中替用户选择 `gain_source`。

---

## Dynamic Gain NPZ Schema

`dynamic_gain.npz` 建议包含：

| 键 | 形状 | 语义 |
|----|------|------|
| `volume` | `(n_inline, n_xline, n_sample)` | 正值 dynamic gain |
| `samples` | `(n_sample,)` | 正秒 TWT 采样轴 |
| `inline` / `xline` | `(n_inline,)` / `(n_xline,)` | 线号轴 |
| `geometry_json` | 标量 | 与第八步地震几何一致 |
| `metadata_json` | 标量 | 输入路径、归一化口径、拟合参数和输出说明 |

`metadata_json` 至少写入：

| 字段 | 含义 |
|------|------|
| `schema_version` | 例如 `dynamic_gain_v1` |
| `sample_domain` / `sample_unit` | 必须为 `time` / `s` |
| `gain_reference` | `unit_wavelet_synthetic_to_normalized_observation` |
| `normalization` | `seismic_raw_divided_by_train_mask_rms` |
| `train_mask_rms` | 与第八步训练端一致的地震 RMS |
| `gain_model_is_relative_to_fixed_gain` | 固定为 `false` |
| `unit_wavelet_file` | 第五步 `selected_wavelet.csv` |
| `ai_lfm_file` | 第七步 `ai_lfm_time.npz` |
| `target_layer` / `horizons` | 从第七步 LFM metadata 继承 |

第八步读取该 NPZ 时，应校验 shape、采样轴、时间域单位和正值约束。

---

## 计算流程

### 1. 重建 GINN 数据口径

读取地震体、LFM NPZ 和 LFM metadata 中的层位信息，按第八步相同逻辑重建 target mask、loss mask 和训练 trace 集合。

然后计算：

```text
train_mask_rms = RMS(seismic_raw[train_mask])
seismic_norm = seismic_raw / train_mask_rms
```

这一步是整个旁路的尺度基准。后续所有 gain 样本和 gain 体都在 `seismic_norm` 域里解释。

### 2. 单位子波正演 LFM

读取第五步 `selected_wavelet.csv`，不乘 fixed gain，得到单位子波正演：

```text
syn_unit = ForwardModel(unit_wavelet)(LFM)
```

若第八步未来使用 dynamic gain，则训练端应保持 `fixed_gain = 1.0`，由 `dynamic_gain.npz` 直接缩放 `syn_unit`。

### 3. 井上 fixed gain 样本

在第六步入选井或 anchor 控制点上，按同一时间窗计算局部最小二乘 gain：

```text
gain_i = sum(seismic_norm_i * syn_unit_i) / sum(syn_unit_i^2)
```

只保留正值、有限值、有效样点数足够的样本。密井平台应做空间去偏聚合，避免一组近井控制 fixed gain 的全局推荐值。

`recommended_fixed_gain.json` 建议包含：

| 字段 | 含义 |
|------|------|
| `recommended_fixed_gain` | 空间去偏后的井上 gain 中位数 |
| `n_wells` / `n_segments` | 参与估计的井数和片段数 |
| `normalization` | `seismic_raw_divided_by_train_mask_rms` |
| `train_mask_rms` | 地震归一化 RMS |
| `gain_reference` | `unit_wavelet_synthetic_to_normalized_observation` |

这个 fixed gain 是人工决策时的 baseline。若不使用 dynamic gain，第八步配置可填：

```yaml
gain_source: fixed_gain
fixed_gain: <recommended_fixed_gain>
include_dynamic_gain_input: false
```

### 4. 拟合 dynamic gain

第一版建议使用地震属性驱动的简洁模型，不把网络引进这个旁路：

```text
ln(gain) = intercept + slope * ln(attribute)
```

可选属性包括：

| 属性 | 说明 |
|------|------|
| `moving_rms(seismic_norm)` | 首选，表达局部振幅背景 |
| `moving_abs_mean(seismic_norm)` | RMS 的稳健替代 |
| `moving_abs_p90(seismic_norm)` | 对强反射更敏感 |

所有属性都从 `seismic_norm` 计算，不能从 raw seismic 或逐道 zscore 地震计算。拟合结果写入 `dynamic_gain_summary.json` 和 `dynamic_gain_samples.csv`。

### 5. 生成 gain 体

对全体地震道计算同一属性，套用拟合关系生成 gain：

```text
attribute = moving_rms(seismic_norm)
log_gain = intercept + slope * log(attribute)
gain = exp(log_gain)
```

gain 必须为正有限值。clip、平滑和边界处理都应写入 metadata；clip 是数值保护，不是自动 QC 决策。

---

## 第八步接入

使用 dynamic gain 时：

```yaml
gain_source: dynamic_gain_model
fixed_gain: null
dynamic_gain_model: scripts/output/dynamic_gain_<timestamp>/dynamic_gain.npz
include_dynamic_gain_input: true
in_channels: 4
```

不使用 dynamic gain，只使用推荐 fixed gain 时：

```yaml
gain_source: fixed_gain
fixed_gain: <recommended_fixed_gain>
dynamic_gain_model: null
include_dynamic_gain_input: false
in_channels: 3
```

第八步训练端已有两个不同用途：

1. `dynamic_gain` 作为正演乘子，直接乘到合成地震上。
2. `dynamic_gain_log_ratio` 作为网络输入通道，表达局部振幅补偿上下文。

因此 dynamic gain 体的绝对尺度必须和 `seismic_norm` 对齐，不能再额外乘 fixed gain。

---

## 人工 QC 建议

脚本只提供证据，不自动选择 dynamic 或 fixed。建议至少输出：

| QC | 目的 |
|----|------|
| 井上 `gain_i` 分布 | 判断 fixed baseline 是否稳定 |
| `ln(attribute)` vs `ln(gain_i)` 散点图 | 判断 dynamic 关系是否可信 |
| dynamic gain 体切片 | 排查异常条带、边界突变和非地质性强振幅 |
| 井上 fixed / dynamic 合成对比 | 人工判断 dynamic 是否真的改善波形 |
| clip 比例 | 判断拟合是否大量落在保护边界 |

最终第八步用 `fixed_gain` 还是 `dynamic_gain_model`，由人工根据这些图和指标决定。

---

## 与深度域旧旁路的区别

时间域实现不要直接复刻深度域两个脚本的拆分方式。建议脚本层面合并为一个 `dynamic_gain.py`，内部拆函数即可：

```text
load_context()
build_ginn_normalization()
estimate_well_gain_samples()
fit_gain_relationship()
build_gain_volume()
write_outputs()
```

深度域旧实现中逐道 zscore 后计算属性的做法，不适合作为时间域默认口径。时间域 dynamic gain 的尺度必须服务第八步 GINN 的 waveform loss，因此以 `seismic_raw / train_mask_rms` 为唯一归一化口径。
