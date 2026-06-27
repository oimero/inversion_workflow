# 08 R1 正演闭环诊断

`real_field_forward_diagnostic.py` 是第八步的第二轮，也是当前流程的最终闭环。它回答一个核心问题：**R0 零样本预测真的比低频背景模型更接近观测地震数据吗？** 脚本把每个模型的预测波阻抗用全局子波正演合成地震记录，与观测地震做系统比较，从四个维度给出判断：地震波形残差、波阻抗与井曲线的一致性、子波不确定性下的稳键性、以及横向信息的副作用。

这一步不是可选的 QC——它是判定 R0 预测是否具有物理可信度的必要环节。

---

## 快速开始

```bash
python scripts/real_field_forward_diagnostic.py
python scripts/real_field_forward_diagnostic.py --config experiments/common/common.yaml
python scripts/real_field_forward_diagnostic.py --zero-shot-dir scripts/output/real_field_zero_shot_<timestamp>
python scripts/real_field_forward_diagnostic.py --output-dir scripts/output/real_field_forward_diagnostic_test
```

不带参数时，脚本自动发现最新的 R0 产物，在 `scripts/output/real_field_forward_diagnostic_<timestamp>/` 下写出诊断结果。

---

## 运行前需要什么

| 来源 | 文件 | 用途 |
|------|------|------|
| 第八步 R0 | `real_field_zero_shot_summary.json` | R0 推理的完整元数据和来源回溯 |
| 第八步 R0 | `predictions.npz`（每个模型子目录） | 预测波阻抗和地震输入 |
| 第五步 | `selected_wavelet.csv` | 全局子波，用于正演合成地震记录 |
| 第五步 | `selected_wavelet_summary.json` | 子波来源校验 |
| 第四步 | `well_tie_metrics.csv` | 井的标定状态和坐标，用于井 QC |
| 第四步 | `wavelet_inventory.csv` | 候选子波清单，用于子波不确定性扫描 |
| 第一步 | `well_inventory.csv` | 井位坐标和井型信息 |

---

## 配置参考

```yaml
# 以下字段写在 experiments/common/common.yaml 的 real_field_forward_diagnostic 段下。

# --- 必填 ---
# real_field_forward_diagnostic 段本身必须存在，但内部参数均可选。
# zero_shot_dir 缺失时自动发现最新 R0 产物。
real_field_forward_diagnostic: {}

# --- 可选（有默认值）---
real_field_forward_diagnostic:
  diagnostic_max_hz: 80.0             # 默认 80.0

  boundary:
    forward_diagnostic_erosion_s: 0.0 # 默认 0.0（不侵蚀）

  well_qc:
    enabled: true                     # 体积模式默认 false，剖面模式默认 true
    max_xy_distance_m: 300.0          # 井 QC 启用时必填
    include_deviated_wells: false     # 体积模式默认 true，剖面模式默认 false

  red_flag_thresholds:
    seismic_ood_fraction_abs_gt5: 0.05    # 默认 0.05
    lateral_nullspace_energy_ratio: 0.5   # 默认 0.5

  spectral_qc:                        # 不填时自动按 diagnostic_max_hz 三等分
    # bands 和 manual_override_reason 只在需要自定义频带时才配置

  diagnostic_scan:
    phase_deg: [-20, -10, 0, 10, 20]           # 默认值
    fractional_shift_samples: [-1.0, -0.5, 0.0, 0.5, 1.0]  # 默认值
    candidate_wavelet_limit: 0                  # 默认 0（不限制）
```

### `boundary`

正演诊断时是否从连续有效 mask 段内侧做侵蚀，避免卷积边界伪影污染诊断统计：

```yaml
  boundary:
    forward_diagnostic_erosion_s: 0.0
```

默认 `0.0`（不做侵蚀）。该侵蚀只作用于诊断统计区，不改写 R0 预测。

### `well_qc`

控制井数据闭环检查。开启后从第四步读取所有标定成功井：用滤波 LAS 做正演基准、用低频模型做正演底线、用模型预测做正演对比。

```yaml
  well_qc:
    enabled: true
    max_xy_distance_m: 300.0
    include_deviated_wells: false
```

| 参数 | 含义 |
|------|------|
| `enabled` | 开关。体积模式默认 `false`，剖面模式默认 `true` |
| `max_xy_distance_m` | 井到推理网格迹的最大距离，井 QC 启用时必填 |
| `include_deviated_wells` | 是否包含斜井。体积模式默认 `true`，剖面模式默认 `false` |

### `spectral_qc`

不填时按 `diagnostic_max_hz` 自动三等分。R1 的频带诊断比 R0 多一层：同时分析合成记录的频带拟合残差。需要自定义频带时：

```yaml
  spectral_qc:
    bands:
      - name: lowfreq
        low_hz: 0.0
        high_hz: 16.0
      - name: observable_band
        low_hz: 16.0
        high_hz: 32.0
      - name: highfreq_or_nullspace
        low_hz: 32.0
        high_hz: 80.0
    manual_override_reason: "说明为什么不用默认三等分"
```

`bands` 每条必须有 `name`、`low_hz`、`high_hz`。`manual_override_reason` 在手动配置时必填。

### `diagnostic_scan`

残差分解扫描参数。对每个阻抗输入做相位扫描和分数采样偏移扫描——如果残差能通过调整子波来显著降低，说明主要问题在子波或时深关系不确定性，而非波阻抗本身有误。

```yaml
  diagnostic_scan:
    phase_deg: [-20, -10, 0, 10, 20]
    fractional_shift_samples: [-1.0, -0.5, 0.0, 0.5, 1.0]
    candidate_wavelet_limit: 0
```

| 参数 | 默认 | 含义 |
|------|------|------|
| `phase_deg` | `[-20, -10, 0, 10, 20]` | 常相位扫描角度列表 |
| `fractional_shift_samples` | `[-1.0, -0.5, 0.0, 0.5, 1.0]` | 分数采样偏移列表 |
| `candidate_wavelet_limit` | `0` | 子波敏感性检查的候选子波数量上限。`0` 不限制 |

### `red_flag_thresholds`

| 参数 | 含义 |
|------|------|
| `seismic_ood_fraction_abs_gt5` | R0 输入地震标准化后超过 5σ 的样本比例上限 |
| `lateral_nullspace_energy_ratio` | 两个模型差异在高频零空间带的能量占比上限 |

---

### 第一阶段：加载与校验

1. 从 R0 的摘要中加载所有信息和来源路径。
2. 加载全局子波和候选子波，准备多场景正演。
3. 校验来源闭环：R0 → LFM → 第四步 → 第一步的链路必须完整且路径一致。

### 第二阶段：批量正演

对每个阻抗输入（低频模型、每个 R0 模型预测）执行 Robinson 正演：

```text
r[j] = tanh((logAI[j] - logAI[j-1]) / 2)
synthetic = convolve(r, wavelet)  # 挂在下部样本上
```

卷积约定与第四、五步一致：反射系数挂在下部样本上，合成记录对齐到 `observed[:, 1:]`（丢弃观测地震的第一个样点以保证维度匹配）。

这一步同时用全局子波和所有候选子波做多场景正演，用于测试子波不确定性的影响。

### 第三阶段：残差诊断

对每个阻抗 × 每个子波场景的合成-观测配对，计算：

- **基本波形指标**：相关系数（无缩放和尺度归一化两种）、标准化残差 RMS、最优最小二乘缩放因子。
- **相位和偏移扫描**：在 ±20° 相位范围和 ±1.0 样本时间偏移范围内，找到使残差最小的相位和偏移。如果最优相位远离 0° 或最优偏移远离 0，说明残差可以通过调整子波来部分消除。
- **频带分解**：在每个频带内单独计算观测 RMS、合成 RMS 和残差 RMS，判断各频带的拟合质量。
- **空间残差模式**：逐 inline / xline 计算残差的 RMS，检查是否存在系统性的空间偏差（如工区边缘残差系统偏高）。

所有指标汇总到 `forward_diagnostic_metrics.csv`。

### 第四阶段：井数据闭环

这是 R1 最关键的诊断维度。对每口标定成功的井：

1. 从推理网格中提取井位处的预测波阻抗。
2. 从第四步获取该井的滤波 LAS 波阻抗（以优化 TDT 投影到 TWT 域）。
3. 计算预测 vs 井曲线的 RMSE、偏差和相关系数。
4. 用滤波 LAS 做正演，作为“完美波阻抗的正演匹配”基准。
5. 用低频模型做正演，作为“模型必须击败的基线”。
6. 对每个模型，比较四个指标：
   - 波阻抗 RMSE（比 LFM 低吗？）
   - 波阻抗相关系数（比 LFM 高吗？）
   - 正演合成与观测地震的波形相关系数（高于 0.9 吗？）
   - 频带级别的 RMSE 和相关系数

根据 LFM vs 模型的指标对比，每口井被分类为：

| 分类 | 含义 |
|------|------|
| `model_improves_ai` | 模型在 RMSE 和相关系数上均优于 LFM |
| `shape_improves_bias_worse` | 相关系数改善但全频带 RMSE 变差 |
| `bias_improves_shape_worse` | RMSE 改善但相关系数未改善 |
| `waveform_good_ai_worse` | 波形匹配好但波阻抗指标不如 LFM |
| `filtered_las_weak_reference` | 滤波 LAS 的正演匹配本身就很弱，无法作为有效基准 |
| `mixed_or_insufficient` | 混合信号或有效样本不足 |

### 第五阶段：综合判定

脚本汇总所有维度的诊断结果，生成红色告警列表和推荐下一步动作：

- 有任何红色告警 → `return_to_input_preparation_or_synthetic_diagnostic`：建议回溯输入准备或合成基准诊断。
- 无红色告警 → `future_sparse_well_adapter_candidate`：可以考虑进入稀疏井适配等下一步研究。

---

## 核心输出文件

所有文件在 `<output_root>/real_field_forward_diagnostic_<timestamp>/` 下：

### 主诊断输出

| 文件 | 内容 |
|------|------|
| `real_field_forward_diagnostic_summary.json` | 诊断摘要：来源路径、正演约定、红色告警、推荐下一步 |
| `forward_diagnostic_metrics.csv` | 每个阻抗 × 子波场景的基本波形指标和缩放信息 |
| `residual_decomposition.csv` | 相位扫描和分数偏移扫描的残差分解明细 |
| `wavelet_sensitivity.csv` | 各候选子波场景下的波形指标，用于评估子波不确定性的影响 |
| `spatial_residual_qc.csv` | 逐 inline/xline 的空间残差模式 |

### 频带分析

| 文件 | 内容 |
|------|------|
| `forward_band_residual_qc.csv` | 各频带的观测 RMS、合成 RMS、残差 RMS 和残差/观测比值 |
| `ai_plausibility_qc.csv` | 预测波阻抗和预测差值的统计分布、频带能量、与训练分布的对比 |

### 井闭环

| 文件 | 内容 |
|------|------|
| `well_forward_diagnostic.csv` | 每口井每个角色的完整诊断：波阻抗指标、频带指标、波形指标 |
| `well_ai_comparison_summary.csv` | LFM vs 模型对比及逐井分类 |
| `well_ai_band_comparison.csv` | 逐井逐角色逐频带的 RMSE 和相关系数 |

### 正演合成记录

| 目录 | 内容 |
|------|------|
| `synthetic/` | 每个阻抗输入的正演合成地震记录（`.npz` 格式），包含合成记录、观测地震和有效掩码 |

### 图表

| 文件 | 内容 |
|------|------|
| `figures/<role>_observed_synthetic_residual.png` | 三面板：观测 / 合成 / 残差 |
| `figures/phase_shift_gain_scan.png` | 相位和分数偏移扫描曲线 |
| `figures/spatial_residual_qc.png` | 空间残差模式（逐线） |
| `figures/forward_band_residual_qc.png` | 各频带残差/观测比值柱状图 |
| `figures/ai_band_energy_qc.png` | 预测差值频带能量分布 |
| `figures/well_ai_comparison_summary.png` | 井分类统计柱状图 |
| `figures/well_ai_band_comparison.png` | 逐频带井曲线匹配中位数 RMSE |
| `figures/wells/well_forward_qc_<well>_<role>.png` | 单井六面板波形 QC 图（波阻抗、反射系数、合成、观测、互相关、动态互相关） |

---

## 如何阅读结果

### 第一步：看 `real_field_forward_diagnostic_summary.json`

直接跳到 `red_flags` 字段：

- 空列表：没有致命问题，R0 预测在物理上是自洽的。
- 有红色告警：按 severity 逐个查看。`seismic_ood` 类型的告警通常意味着需要回溯 R0 的输入标准化配置；`scale_status_not_ok` 意味着正演合成记录的振幅尺度异常；`lateral_difference_concentrated_in_nullspace` 意味着横向混合器的效果不可靠。

同时看 `recommended_next_state`：它告诉你脚本认为下一步应该做什么。

### 第二步：看 `forward_diagnostic_metrics.csv`

按 `model_role` 分组，比较各角色的 `residual_corr_scaled`（带尺度优化的波形相关系数）和 `residual_rms_scaled`（标准化后的残差 RMS）：

- `lfm_only`：低频模型的正演匹配——这是基线。如果它已经很差（如相关系数 < 0.3），说明正演所用的子波本身就和地震数据不匹配。
- `zero_shot_lateral` / `zero_shot_no_lateral`：模型的匹配应该优于 `lfm_only`。如果更差，说明模型的波阻抗偏离了物理约束。

### 第三步：看 `residual_decomposition.csv`

筛选 `scan_type == phase`，观察最优相位是否接近 0°：

- 最优相位接近 0° → 残差不来自子波相位误差。
- 最优相位偏离 0° 超过 10° 且残差显著降低 → 残差可能来自全局子波与真实子波的系统相位差异。

筛选 `scan_type == fractional_shift`，观察最优偏移是否接近 0：

- 最优偏移接近 0 → 合成和观测在时间上对齐。
- 最优偏移偏离 0 超过 0.5 样本 → 可能存在系统的时深关系偏移。

### 第四步：看 `well_ai_comparison_summary.csv`

按 `classification` 分组统计：

- `model_improves_ai` 的井数应该占多数。如果大部分井是 `model_improves_ai`，说明模型在井位处确实比低频模型更准确。
- 如果 `waveform_good_ai_worse` 占多数，说明模型的波形虽然和地震匹配，但波阻抗本身还不如低频模型——这可能是因为模型在拟合噪声。
- 如果 `filtered_las_weak_reference` 频繁出现，说明这些井的滤波 LAS 本身正演匹配就很差——井的标定可能有问题，不应作为基准。

### 第五步：看频带分析

`forward_band_residual_qc.csv` 中 `residual_to_observed_ratio` 越低的频带，拟合越好：

- 如果高频带的比值接近 1.0，说明该频带几乎没有被拟合——这是正常的，通常高频带本身就是噪声为主的零空间。
- 对比 `no_lateral` 和 `lateral` 在各频带的残差/观测比值：如果 `lateral` 只在低频带有改善，说明横向混合器并没有引入虚假的高频信息。

### 第六步：抽查井 QC 图

打开 `figures/wells/well_forward_qc_<well>_<role>.png`：

- 六个面板从左上到右下：预测 AI + 滤波 AI、反射系数、合成地震 wiggle、观测地震 wiggle、静态互相关、动态互相关。
- 合成和观测的 wiggle 应该在目标窗口内形态相似。
- 互相关热力图的对角线应该是亮的——如果不是，说明合成和观测在时间上有系统错位。

---

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| R0 摘要缺失或 schema 不匹配 | R0 输出不完整或版本不对 | 重新运行 R0，确认 `real_field_zero_shot_summary.json` 存在且状态为 `needs_forward_diagnostic` |
| `well_qc.well_auto_tie_dir` 缺失 | 井 QC 启用但第四步目录未指定 | 在配置中显式填写，或确认脚本能从 R0 → LFM 链路中自动发现 |
| 所有井被 `skipped_outside_section_support` 跳过 | `max_xy_distance_m` 太小 | 增大距离阈值，或确认井坐标和推理网格的对齐关系 |
| 正演合成振幅接近零 | 子波加载失败或预测波阻抗全为常数 | 检查子波文件和预测波阻抗的数值范围 |
| `red_flag: real_input_seismic_ood` | 真实地震数据严重偏离训练分布 | 检查地震值域变换配置；可能需要在更接近真实工区分布的数据上重新训练 |
| `red_flag: lateral_difference_concentrated_in_nullspace` | 横向混合器的效果集中在不可靠的高频带 | 横向模型的可信度存疑；考虑只用不含横向混合器的模型 |

---

## 留到第二轮

- 自动化的统计假设检验，而不只是描述性指标比较。
- 逐层段（而非仅全窗口）的正演残差分析。
- 三维体积级别的井轨迹正演合成对比（当前体积模式的井 QC 是沿单一 inline/xline 读取，未做双线性空间插值到实际井轨迹）。
- 与合成基准评估结果的双向反馈：如果合成基准上模型表现优异但真实工区上正演残差大，自动诊断差异来源。
- 模型预测的不确定性区间在正演域的传播和可视化。
