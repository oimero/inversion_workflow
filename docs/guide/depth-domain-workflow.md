# 深度域工作流

深度域使用与时间域相同的步骤入口。本页说明深度坐标、单位换算和正演方式；通用流程见[工作流总览](../index.md)，各步骤的完整配置见对应主指南。

---

## 总览

在公共配置中声明地震采样域和深度基准：

```yaml
seismic:
  file: <seismic-volume-file>
  type: <seismic-file-type>
  domain: depth
  depth_basis: tvdss
```

可信井与背景井的用途划分沿用[第四步的井用途配置](4-well-auto-tie.md)。

| 量 | 单位 |
|---|---|
| 地震深度轴、层位、主体平滑宽度和低频滤波波长 | 米 |
| 标定内部的双程旅行时、子波和学习卷积核时间轴 | 秒 |
| 短缺口阈值 | 双程旅行时的秒数，通过本井时深关系换算 |

---

## 第 4 步：井震自动标定

### 快速开始

```powershell
python scripts/well_auto_tie.py --config <config-yaml>
```

### 配置参考

通用配置见[井震自动标定](4-well-auto-tie.md)。深度域的标定边界和上下冗余按米填写：

```yaml
well_auto_tie:
  target_interval:
    top_horizon: <top-horizon-file>
    bottom_horizon: <bottom-horizon-file>
    margin_top_m: <top-margin-m>
    margin_bottom_m: <bottom-margin-m>
```

也可以用 `top_depth_m` 和 `bottom_depth_m` 直接指定深度边界。脚本通过本井局部时深关系，将深度窗转换为标定所用的时间窗。

### 脚本在做什么

1. **建立局部时深关系。** 将测井深度转换到地震深度基准，沿纵波慢度积分，得到深度到双程旅行时的映射。
2. **转换到时间轴。** 在井旁或沿轨迹读取深度地震，将地震和井曲线插值到同一规则时间轴。
3. **执行细标定。** 联合搜索测井去尖峰、平滑、子波尺度和整体时移，以相关系数和归一化误差评价结果。
4. **回写深度结果。** 将优化时深关系和滤波曲线映射回观测深度。标定合成记录和图件使用局部时间轴。

### 核心输出

通用产物见主指南，深度域另保存以下坐标转换结果：

| 产物 | 用途 |
|---|---|
| `time_depth/` | 保存标定前后的局部时深关系 |
| `depth_mapping/source_depth_mapping_<well>.csv` | 裁窗前的原始深度到双程旅行时映射 |
| `depth_mapping/aligned_depth_<well>.csv` | 保存优化时深关系回写后的深度位置 |
| `seismic_trace/depth_seismic_trace_<well>.csv` | 保存原始深度地震道 |

第六步利用完整的原始时深映射和标定总时移，恢复目标层外仍有测井覆盖的曲线；超出原始映射范围的位置保持无效。

人工粗偏移通过 `coarse_correction.manual_shift` 按毫秒填写，正值使井曲线对应到较晚时间、较深位置。总时移为人工粗偏移与自动细调之和，实际深度位移由本井时深关系换算。

结果检查结合局部时深关系、回写深度位置和原生深度地震道；第五步的批量波形质检使用原始深度网格。

---

## 第 5 步：全局共识子波生成

### 快速开始

```powershell
python scripts/wavelet_generation.py --config <config-yaml>
```

第五步从第四步成功可信井读取候选子波，在规则时间轴上卷积正演，再通过固定的观测时深关系将合成记录采样回原始深度地震网格。在这个深度网格上交叉评测候选并搜索共识子波；子波本身保持时间采样轴。

配置与候选筛选规则见[全局共识子波生成](5-wavelet-generation.md)。

### 脚本在做什么

1. **准备正演轴。** 读取第四步滤波曲线和候选子波，按优化时深关系在规则时间轴上计算反射系数与卷积合成。
2. **在深度网格评分。** 利用固定观测时深关系，将合成记录采样到原始深度地震网格，在第四步的标定窗口内计算相关性和误差。
3. **搜索与输出。** 按主指南的共识搜索方法选择全局子波，保存深度网格上的逐井合成记录和指标。

### 核心输出

| 产物 | 用途 |
|---|---|
| `selected_wavelet.csv` | 第六步正演质检和后续步骤读取的共识子波 |
| `batch_synthetic_metrics.csv` | 每口评测井的合成记录指标 |
| `synthetic_qc/`、`figures/` | 逐井波形、残差和子波质量控制 |

深度域正演输入使用第五步的 `selected_wavelet.csv`。正演输入旁路见本页末尾。

---

## 第 6 步：真实工区井控数据集

入口、来源目录与曲线选择见[真实工区井控数据集](6-real-field-well-controls.md)。深度域的平滑尺度和相关窗口按米填写：

```yaml
real_field_well_controls:
  supervision_curve: body  # filtered 或 body
  body_smoothing:
    fwhm_axis_units: <body-smoothing-m>
  qc:
    forward_model_inputs_run_dir: <forward-input-run-dir> # 留空自动发现
    dynamic_correlation_window_m: <correlation-window-m>
```

正演输入用于井曲线质检，留空时自动发现与当前深度域匹配的运行。监督曲线的选择沿用主指南。

### 脚本在做什么

1. **恢复井曲线。** 可信井使用第四步的完整时深映射与标定总时移恢复可用深度曲线；背景井读取第三步曲线。
2. **转换深度与位置。** 将测深和井位转换到地震深度基准，按连续有效段插值到地震轴。短缺口阈值通过深度和速度换算为双程旅行时。
3. **生成主体曲线。** 配置主体平滑时，先在可信井的原生深度轴上分段平滑对数阻抗，再投影到地震轴。
4. **执行正演质检。** 使用共识子波和阻抗—速度关系比较井曲线的合成地震与真实地震。缺口及评价支撑的通用规则见[数据与单位约定](../concepts/data-and-coordinate-conventions.md)。

---

## 第 7 步：真实工区低频模型

第七步的通用配置见[真实工区低频模型](7-real-field-lfm.md)。深度域配置要点如下：

```yaml
real_field_lfm:
  context_extension:
    enabled: true
    baseline_id: <baseline-id>
    top_axis_units: <top-context-m>
    bottom_axis_units: <bottom-context-m>
    transition_axis_units: <transition-m>
  baselines:
    <baseline-id>:
      method: trend
      filter:
        enabled: true
        cutoff_wavelength_m: <cutoff-wavelength-m>
        buffer_axis_units: <depth-buffer-m>
```

截止波长、缓冲长度和上下文扩展长度均按米解释。扩展背景与目标层模型的用途见主指南。

### 脚本在做什么

1. **按深度滤波。** 将截止波长换算为每米的空间频率，对井曲线的连续有效段低通，再按主指南的方法建立空间基线。
2. **扩展背景。** 按指定米数向目标层上下延伸，并在衔接区平滑连接，供第八步建立网络输入和正演支撑。

---

## 第 8 步：物理约束神经网络反演

第八步的通用配置见[物理约束神经网络反演](8-ginn.md)。深度域训练配置至少包含：

```yaml
ginn_v3_body_inversion:
  inputs:
    lfm_run_dir: <lfm-run-dir> # 留空自动发现
    variant_id: <lfm-variant-id>
    well_control_run_dir: <well-control-run-dir> # 留空自动发现
    ai_velocity_relation:
      formula: "AI = a * Vp + b"
      a: <ai-vp-coefficient-a>
      b: <ai-vp-intercept-b>
      ai_unit: m/s*g/cm3
      vp_unit: m/s
      a_unit: g/cm3
      b_unit: m/s*g/cm3
  kernel:
    dt_s: <kernel-sampling-interval-s>
    duration_s: <kernel-duration-s>
```

深度域输入必须在 `velocity_volume` 和 `ai_velocity_relation` 之间选择一个。上例使用单位明确的 AI–Vp 关系；如果使用速度体，则将其路径填入 `inputs.velocity_volume`，并删除 `ai_velocity_relation`。

### 脚本在做什么

1. **换算反射界面。** 利用速度体或阻抗—速度关系得到固定正演速度，将相邻阻抗样点之间的界面位置换算为双程旅行时。
2. **计算深度域波形。** 在扩展背景范围内，用预测阻抗和时间采样的学习核正演，再回到地震深度网格计算目的层内的波形损失。
3. **训练和输出。** 沿用主指南的井监督与卷积核交换约束，保存权重、学习核和井旁质检。体预测另行执行。

---

## 深度域正演输入旁路

当第六步需要使用固定 AI–Vp 关系和共识子波进行井控质检时，运行正演输入旁路：

```powershell
python scripts/depth_forward_model_inputs.py --config <config-yaml>
```

配置使用第五步的选定子波：

```yaml
depth_forward_model_inputs:
  source_runs:
    rock_physics_analysis_dir: <rock-physics-run-dir>
    wavelet_generation_dir: <wavelet-generation-run-dir>
```

旁路将岩石物理关系与共识子波汇总到 `forward_model_inputs.json`，供第六步井控质检读取。
