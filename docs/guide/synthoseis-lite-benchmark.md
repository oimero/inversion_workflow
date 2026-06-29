# 合成基准生成与评估

`synthoseis_lite.py` 冻结一个已知真值的二维声阻抗合成基准，用它判断逆问题的实际可恢复性。时间域 v1 可以读取正演可观测性旁路的结果来设计 probe，但该旁路不占用步骤编号，也不改变合成基准的物理校准。脚本先用井数据冻结阻抗校准（calibrate），再生成规范几何或场条件几何的真值剖面（generate），最后用模型无关的基线评估器（`evaluate_synthoseis_lite.py`）读出报告卡。

---

## 快速开始

```bash
# 一键运行（推荐）：通过 PowerShell runner
cd experiments/synthoseis_lite
.\run_synthoseis_lite.ps1 <ExperimentName>
# 自动顺序执行 calibrate → generate canonical → generate field_conditioned
# 结果落在 experiments/synthoseis_lite/results/<ExperimentName>/

# 手动分步运行：
# 第一步：从井数据冻结阻抗校准
python scripts/synthoseis_lite.py calibrate
python scripts/synthoseis_lite.py calibrate --config experiments/synthoseis_lite/synthoseis_lite.yaml

# 第二步：生成规范几何基准
python scripts/synthoseis_lite.py generate \
  --suite canonical \
  --impedance-calibration experiments/synthoseis_lite/results/<name>/calibrate/impedance_calibration.json

# 第三步：生成场条件几何基准
python scripts/synthoseis_lite.py generate \
  --suite field_conditioned \
  --impedance-calibration experiments/synthoseis_lite/results/<name>/calibrate/impedance_calibration.json

# 场条件调试选项
python scripts/synthoseis_lite.py generate \
  --suite field_conditioned \
  --impedance-calibration <calibration.json> \
  --debug-attempt-limit 5          # 限制尝试次数，跳过接受率检查
python scripts/synthoseis_lite.py generate \
  --suite field_conditioned \
  --impedance-calibration <calibration.json> \
  --geometry-family wedge           # 只生成楔体几何族
python scripts/synthoseis_lite.py generate \
  --suite field_conditioned \
  --impedance-calibration <calibration.json> \
  --qc-only                        # 运行完整生成+QC，但不落盘实现数组

# 第四步：评估基线
python scripts/evaluate_synthoseis_lite.py \
  --benchmark-dir experiments/synthoseis_lite/results/<name>/generate_field_conditioned
python scripts/evaluate_synthoseis_lite.py \
  --benchmark-dir <path> \
  --sample-kind base --sample-kind frequency_probe  # 只评估特定样本类
python scripts/evaluate_synthoseis_lite.py \
  --benchmark-dir <path> \
  --baseline lfm_ideal --baseline lfm_controlled_degraded  # 只跑特定基线
```

`generate` 的 `--suite` 参数必须二选一：`canonical` 或 `field_conditioned`，一次运行只生成一套。

---

## 运行前需要什么

### calibrate（校准阶段）

| 输入 | 来源 | 用途 |
|------|------|------|
| `well_preprocess_status.csv` | 第三步 | 预处理 LAS 路径 |
| `well_tie_metrics.csv` | 第四步 | 滤波后 LAS 路径、优化 TDT 路径、成功井标识 |
| `evaluation_well_spatial_clusters.csv` | 第五步 | 空间簇分配 |
| `selected_wavelet.csv` | 第五步 | 提取子波 dt，验证与输出 dt 匹配 |
| 井分层文件 | Petrel | 构建区域边界 |
| `experiments/synthoseis_lite/synthoseis_lite.yaml` | 合成旁路配置 | `synthoseis_lite` 段 |

### generate（生成阶段）

| 输入 | 来源 | 用途 |
|------|------|------|
| `impedance_calibration.json` | calibrate 输出 | 冻结的区域模型、父先验、转移矩阵 |
| `run_summary.json` | 正演可观测性分析 | 验证来源闭环 + 实验区间 |
| `frequency_evidence_bands.csv` | 正演可观测性分析 | 探针频率选择 |
| `well_frequency_sensitivity.csv` | 正演可观测性分析 | 噪声等效 log(AI) 参考 |
| `well_preprocess_status.csv` | 第三步 | 来源闭环校验 |
| `well_tie_metrics.csv` | 第四步 | 来源闭环校验 |
| `selected_wavelet.csv` + `selected_wavelet_summary.json` | 第五步 | 子波 |
| `evaluation_well_spatial_clusters.csv` | 第五步 | 来源闭环校验 |
| 解释层位文件 | Petrel | 场条件几何构建（canonical 套件不需要） |

### evaluate（评估阶段）

| 输入 | 用途 |
|------|------|
| `synthetic_benchmark.h5` | 冻结的基准数据（真值、地震、先验、探针） |
| `sample_index.csv` | 样本索引（ID、类型、状态、元数据） |
| `benchmark_manifest.json` | 文件清单和 SHA-256 校验和 |

---

## 配置参考

```yaml
# 配置文件：experiments/synthoseis_lite/synthoseis_lite.yaml

# --- 必填 ---
synthoseis_lite:
  global_seed: <integer>

  sections:
    - section_id: section_A
      path:
        - {inline: <il>, xline: <xl>}
        - {inline: <il>, xline: <xl>}
      resample_interval_m: 25.0

target_interval:
  horizons:
    - {name: <top-id>, well_top: <well-top-surface>, file: <horizon-file>}
    - {name: <middle-id>, well_top: <well-top-surface>, file: <horizon-file>}
    - {name: <base-id>, well_top: <well-top-surface>, file: <horizon-file>}

# --- 可选（source_runs 缺失时自动发现最新第六步产物）---
synthoseis_lite:
  source_runs:
    forward_observability_dir: scripts/output/forward_observability_<timestamp>

# --- 可选（有默认值）---
synthoseis_lite:
  sampling:
    expected_output_dt_s: 0.002
    vertical_oversampling_factor: 8

  geometry:
    lateral_sample_interval_m: 25.0
    field_conditioned:
      target_zone:
        mode: filled_target_zone
        min_thickness_s: 0.050
    canonical:
      enabled: true
      lateral_samples: 128
      center_twt_s: 1.5
      vertical_extent_periods: 6.0

  impedance_attribute_generator:
    family: object_coefficients_v1
    state_threshold_sigma: 1.0
    lateral:
      correlation_length_section_fractions: [0.1, 0.3, 1.0]
      coefficient_sigma_multipliers: [0.25, 0.5]
      thickness_log_sigma_values: [0.10, 0.25]
    qc:
      max_global_reversal_fraction: 0.10
      max_object_reversal_fraction: 0.25
      max_global_clipping_fraction: 0.005
      max_object_clipping_fraction: 0.02
      minimum_attempts_per_scenario: 20

  generation:
    attempts_per_scenario: 20
    duration_modes: [standard]
    geometry_families: [none, wedge, pinchout]
    geometry_directions: [left_to_right, right_to_left]

  lfm:
    enabled: true
    ideal:
      cutoff_hz: 10.0
      numtaps: 129
      kaiser_beta: 8.6
    controlled_degraded:
      constant_bias_sigma_log_ai: 0.02
      linear_twt_trend_sigma_log_ai: 0.02
      zonewise_bias_sigma_log_ai: 0.03
      over_smoothing:
        cutoff_hz: 6.0
        blend: 1.0

  seismic_mismatch:
    enabled: true
    noise:
      white_noise_rms_fraction: 0.05
      colored_noise_rms_fraction: 0.05
    gain:
      global_log_sigma: 0.15
      tracewise_log_sigma: 0.15
    wavelet:
      phase_rotation_degrees: [-10.0, 10.0]
      time_shift_samples: [-0.5, 0.5]
    combined:
      enabled: true

  probe_selection:
    enabled: true
    amplitude_multipliers: [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    phases: [sin, cos]
    lateral_shapes:
      - section_coherent
      - {name: localized_tukey, centered_fraction: 0.40, alpha: 0.5}
```

### `global_seed`

全局随机种子，所有随机流都从它派生。每个随机流通过 `named_seed(global_seed, benchmark_version, generator_family, stream_purpose, ...)` 生成独立的 PCG64DXSM 生成器。命名是顺序无关的（通过 JSON 序列化有序键做 SHA-256），保证跨运行复现。

### `source_runs`

可留空——脚本自动从 `output_root` 发现最新第六步（forward_observability）产物，并从其 `run_summary.json` 反查第三/四/五步来源。显式填写时仅需提供 `forward_observability_dir`，其余目录从第六步 summary 中读取并做闭环校验。

### `sampling`

- `expected_output_dt_s`：模型网格时间采样间隔，必须与子波 dt 匹配。
- `vertical_oversampling_factor`：高分辨率真值网格相对于模型网格的过采样倍数。默认 8× 意味着高分辨率 dt 是模型网格 dt 的 1/8。

### `geometry.field_conditioned`

场条件几何的核心配置。层位从顶层 `target_interval.horizons` 读取，剖面定义从 `synthoseis_lite.sections` 读取。`target_zone.mode` 控制目标区构建方式（当前仅 `filled_target_zone`）。

每个层位条目显式区分三种语义：`name` 是校准产物、区域 ID 和下游输出使用的稳定内部层位 ID；`well_top` 是井分层文件 `Surface` 列中的名称；`file` 是解释层面文件。三项均为必填，代码不会用 `name` 猜测 `well_top`，也不会从文件名猜测层位语义。`generate` 还会验证当前配置中的内部层位 ID 及顺序与冻结校准完全一致。

### `impedance_attribute_generator`

控制从井数据校准阻抗统计模型的参数。

#### `state_threshold_sigma`

区分低阻抗/背景/高阻抗三态的 log(AI) 残差阈值（以背景拟合残差的标准差为单位）。默认 1.0，意味着残差超过 1σ 才被划为非背景态。太小会产生大量碎片态，太大会漏掉真实阻抗异常。

#### `lateral` 参数

控制场条件生成中对象属性的横向变异性：`correlation_length_section_fractions` 是横向相关长度占剖面长度的比例（用于 AR(1) 过程），`coefficient_sigma_multipliers` 控制系数横向波动幅度，`thickness_log_sigma_values` 控制对象厚度对数标准差。

#### `qc` 参数

控制生成拒绝的硬门槛：反转比例（AI 对比度方向与校准不一致的像素比例）和裁剪比例（高分辨率真值超出区域 AI P01/P99 边界的比例）分别有全局和单对象阈值。`minimum_attempts_per_scenario` 是每个场景最少尝试次数。

### `generation`

场条件套件的生成网格：`duration_modes` × `geometry_families` × `geometry_directions` × 横向参数组合。每个场景尝试 `attempts_per_scenario` 次直到成功。`geometry_families` 控制几何事件（楔体尖灭、楔体张开），`none` 表示无几何事件的平层。

### `lfm`

控制低频模型先验的推导。

- `ideal`：从真值做零相位 Kaiser FIR 低通滤波，截止频率 `cutoff_hz`。
- `controlled_degraded`：在理想 LFM 上叠加受控退化——常数偏差、线性 TWT 趋势、逐区域偏差、横向平滑偏差（AR(1)）、振幅尺度缩放、过度平滑混合（更低截止频率的混合）、局部缺乏控制偏差（余弦凸块窗口）。

两个先验都写入基准供消费：`lfm_ideal` 和 `lfm_controlled_degraded`。

### `seismic_mismatch`

控制为每个真值生成命名不匹配地震场景：白噪声、有色噪声、全局标量增益、逐道横向平滑增益、时变横向平滑增益、相位旋转、分数时间偏移，以及 `combined_moderate`（相移 + 增益 + 噪声的组合）。噪声在有效前向掩码内归一化到目标 RMS。增益始终严格为正（`exp(Gaussian)`）。

### `probe_selection`

控制频率-振幅探针矩阵的构建。从第一闸门的频率证据目录中选择频率，对每个频率生成振幅梯度（0× 到 4×）和相位对（sin/cos）的探针。每个探针有横向形状选择（剖面一致或局部 Tukey 窗口）。

---

## 脚本在做什么

### calibrate（校准）阶段

校准的目标是从井数据中提取一个冻结的、可复现的随机地质模型参数化。

1. **投影井曲线到 TWT 轴：** 通过第四步的优化 TDT 将滤波后和完整的 log(AI) 从 MD 域分段积分平均到高分辨率 TWT 网格单元。每个网格单元的值是该单元内所有 MD 样本的积分平均。

2. **逐区域背景拟合：** 对每个层位定义的区域，用滤波后的 log(AI) 做 OLS 线性背景拟合 `a + b*(2*zeta - 1)`，其中 zeta 是归一化 TWT 位置（0 到 1）。残差的标准差作为后续状态阈值的基础。

3. **三态识别：** 用 `state_threshold_sigma * sigma` 做阈值，将完整 log(AI) 的残差划分为低阻抗、背景和高阻抗三态。状态短于特定长度的会被合并到相邻的支配态。

4. **对象轮廓拟合：** 对每个非背景对象，用参数化轮廓 `c0 + c1*(2*xi-1) + c2*sin(pi*xi)` 拟合。2-3 点对象只用前两个参数做 OLS，4+ 点对象做 Huber 回归。c0 刻画平均阻抗对比度，c1 刻画线性趋势，c2 刻画曲率。

5. **证据加权与收缩：** 用层级化证据权重（簇等权 → 井内等权 → 单元内等权），计算每个参数在井内的稳健统计，然后收缩到父先验。父先验的方差用 `coefficient_sigma_parent_floor` 和 `coefficient_sigma_parent_cap` 限制。

6. **转移矩阵：** 计算态到态的转移概率，零对角线（不允许自转移），包含直接跳转的支持规则和 `forbidden` 标记。父先验中罕见转移会被混合收缩。

7. **写出冻结校准：** `impedance_calibration.json` 包含模式版本、生成器系列、truth_dt、层位列表、区域模型和父先验。所有 SHA-256 校验和锁定输入数据的版本。

### generate（生成）阶段

生成阶段有两种截然不同的套件：

#### canonical 套件（规范几何）

生成固定的、确定性的解析几何剖面，不涉及随机地质。6 个几何族：

| 族 | 描述 |
|---|------|
| `horizontal_thin_beds` | 不同厚度比（1/16 到 1 倍峰值周期）的水平薄层 |
| `wedge` | 线性楔体，从峰值周期厚度过渡到 0 |
| `pinchout` | 侧向尖灭，终止于剖面 75% 位置 |
| `dipping_layers` | 倾斜层，倾角从 0.25 到 1.0 倍周期落差 |
| `lateral_impedance_change` | 侧向阻抗对比度变化（0.25× 到 2×） |
| `frequency_probe` | 频率探针父样本（振幅 = 0） |

每个规范场景产生一个 HDF5 文件，包含几何实现 + 探针父项。规范几何的作用是提供一个"可以先看懂的"基准切片——没有随机性，方便定位模型行为的具体模式。

#### field_conditioned 套件（场条件随机地质）

这是主基准。对每个场景（duration_mode × correlation_length × sigma_multiplier × geometry_family × direction × 剖面）：

1. **对象序列采样：** 用 Semi-Markov 过程从校准的持续时间和转移概率中采样对象序列。每个对象从父先验中抽取系数（c0, c1, c2）和厚度。

2. **横向 AR(1) 调制：** 每个对象的系数和厚度沿剖面横向用 AR(1) 过程调制，相关长度由 `correlation_length_section_fractions` 控制。

3. **几何事件：** 如果 `geometry_family` 非 `none`，在剖面特定位置施加楔体张开或尖灭事件——对象厚度向楔体顶点线性减小到最小厚度。

4. **背景采样：** 每个横向位置的背景值从区域模型中采样。

5. **逐点对象值：** 对每个横向位置，将所有对象的轮廓投影到高分辨率 TWT 网格，叠加上背景。

6. **条件化 c0：** 将 c0 限制在系数边界、轮廓均值边界、状态方向约束和区域 AI P01/P99 的可行交集内。通过二分搜索将系数投影到有效轮廓边界，不随机重新采样。

7. **QC 拒绝：** 检查高分辨率真值的 AI 反转载剪比例。如果任一 QC 指标超标（全局反转 > 10%、单对象反转 > 25%、全局裁剪 > 0.5%、单对象裁剪 > 2%），抛出 `GenerationRejected` 并重试。

8. **抗混叠下采样：** 用 Kaiser FIR/polyphase 方案将高分辨率真值下采样到模型网格。

9. **正演闭合：** 验证 `F_model_grid(model_target_log_ai) == seismic_model_consistent` 精确闭合，计算高分辨率正演与模型网格正演的 RMS、相关性和频谱形状误差。

10. **探针生成：** 对每个父样本，生成完整的探针矩阵——校准频率 × 振幅梯度 × 相位对。每个探针在垂直方向用 Tukey 窗口形状，横向有权重选择。

11. **地震变体生成：** 对基础和探针样本生成命名不匹配场景。

12. **LFM 推导：** 从真值推导理想和退化低频模型先验。

**接受率检查：** 所有场景生成完毕后，如果任何场景的接受率低于 50% 或尝试次数少于 20（`--debug-attempt-limit` 模式除外），生成以 `field_conditioned_acceptance_qc_failed` 失败。

### evaluate（评估）阶段

`evaluate_synthoseis_lite.py` 是模型无关的基线评估器：

1. **加载基准：** `SynthoseisBenchmark` 读取 `synthetic_benchmark.h5` + `sample_index.csv` + `benchmark_manifest.json`。

2. **三类基线预测：**
   - `oracle_target`：直接返回真值——这是 pipeline 自检，验证基准本身没有数据泄漏。
   - `lfm_ideal`：理想低通先验。
   - `lfm_controlled_degraded`：场条件退化 LFM 先验。

3. **样本指标：** 对每个样本和基线，计算 bias、MAE、RMSE、NRMSE、相关系数。写入 `model_sample_metrics.csv`。

4. **探针指标：** 对每个探针样本，除了绝对误差（vs 真值），还计算配对增量误差——将探针样本的真值增量和预测增量做回归。振幅为 0 的探针对计算绝对误差。写入 `model_probe_metrics.csv`。

5. **几何指标：** 按基线/套件/几何族聚合 base 样本的 RMSE、NRMSE 和相关系数。写入 `model_geometry_metrics.csv`。

6. **报告卡：** `model_report_card.json` 包含基线聚合和探针聚合的摘要。

---

## 核心输出文件

### calibrate 输出

所有文件在 `<output_root>/synthoseis_lite_calibrate_<timestamp>/` 下：

| 文件 | 内容 |
|------|------|
| `impedance_calibration.json` | 冻结的区域模型、父先验、转移矩阵、来源哈希 |
| `zone_models.csv` | 每个区域的三态统计（均值、标准差、样本数） |
| `object_catalog.csv` | 每口井每个对象的系数和轮廓指标 |
| `transfer_matrix.csv` | 态到态转移概率和支持规则 |
| `calibration_qc.csv` | 每次背景拟合和轮廓拟合的 QC 指标 |
| `zone_backgrounds.csv` | 每口井每个区域的背景拟合参数 |
| `profile_sample_statistics.csv` | 每个参数在井内的稳健统计和收缩后值 |
| `figures/` | 背景拟合图、残差直方图、c0 分布图、对象轮廓拟合示例 |
| `run_summary.json` | 输入路径、哈希、配置摘要 |

### generate 输出

所有文件在 `<output_root>/synthoseis_lite_generate_<timestamp>/` 下：

#### 主数据

| 文件 | 内容 |
|------|------|
| `synthetic_benchmark.h5` | 所有实现数组：真值 log(AI)、地震、掩码、LFM、探针 |
| `sample_index.csv` | 每行一个样本的索引、类型、状态和元数据 |
| `benchmark_manifest.json` | 文件清单、SHA-256、配置快照 |

#### 目录级 CSV

| 文件 | 内容 |
|------|------|
| `scenario_catalog.csv` | 所有生成场景的定义和接受状态 |
| `object_catalog.csv` | 所有生成对象的系数和厚度 |
| `object_lateral_coefficients.csv` | 对象系数的横向变化 |
| `generation_qc.csv` | 每次生成尝试的 QC 指标和拒绝原因 |
| `generation_rejection_details.csv` | 拒绝的详细诊断 |
| `frequency_probe_results.csv` | 所有探针变体的定义和 RMS |
| `probe_frequency_catalog.csv` | 选择的探针频率和选择理由 |
| `seismic_variant_results.csv` | 所有地震变体的定义和参数 |
| `section_geometry_qc.csv` | 场条件截面几何的横向支撑 QC |
| `figures/` | 截面 log(AI) 和地震图像、状态带、LFM 对比、接受率条形图 |
| `run_summary.json` | 接受/拒绝计数、各阶段统计 |

### evaluate 输出

所有文件在 `<benchmark_dir>/../synthoseis_lite_evaluate_<timestamp>/` 下：

| 文件 | 内容 |
|------|------|
| `model_sample_metrics.csv` | 每样本每基线的回归指标 |
| `model_probe_metrics.csv` | 每探针每基线的绝对和配对增量误差 |
| `model_geometry_metrics.csv` | 按几何族聚合的指标 |
| `model_report_card.json` | 基线聚合和探针聚合摘要 |
| `evaluation_summary.json` | 输入文件 SHA-256、输出文件列表和哈希 |

---

## 如何阅读结果

### calibrate 后

#### 第一步：看 `zone_models.csv`

确认每个区域的三态统计是否合理——低阻抗和高阻抗态的均值应该分别在背景两侧。如果某一态的样本数极少（< 5），说明该区域几乎没有显著的阻抗异常，后续生成可能缺乏变化。

#### 第二步：看 `figures/background_fit_*.png`

确认背景线（线性趋势）合理穿过滤波后 log(AI) 的中心。如果背景拟合明显偏斜或残差有系统性结构，可能是区域划分需要调整。

#### 第三步：看 `object_catalog.csv`

关注 `profile_mean`（c0）的分布——这是阻抗对比度的核心参数。如果绝大多数对象的 c0 绝对值极小，说明井数据本身的阻抗变化微弱，后续合成基准的 AI 对比度也会很低。

### generate 后

#### 第四步：看 `run_summary.json`

关键计数：

```json
{
  "accepted_realizations": 420,
  "rejected_realizations": 85,
  "failed_scenario_count": 0,
  "probe_variant_count": 1200,
  "seismic_variant_count": 45
}
```

如果 `failed_scenario_count > 0`，说明某些场景的接受率 < 50%——这本身是重要信息：这些参数组合在当前校准下"很难合理地生成"。检查 `scenario_catalog.csv` 中 `acceptance_status` 列的具体场景。

#### 第五步：看 `section_geometry_qc.csv`（仅场条件）

确认每个剖面的横向支撑状态。`support_status` 列标记每个横向样本是原始拾取、线性插值、填充还是越界。大量 `filled` 或 `out_of_bounds` 的样本说明剖面位置可能不够好。

#### 第六步：看 `figures/` 中的截面图

抽查 1-2 个剖面的 `log_ai_section.png` 和 `seismic_section.png`：

- log(AI) 图应该有可见的横向连续性和纵向分层。
- 地震图应该与 log(AI) 图有视觉上的对应——高阻抗层应对应明显的反射轴。
- 如果有几何事件（wedge/pinchout），确认尖灭位置在剖面内是可见的。

#### 第七步：抽样看 `generation_qc.csv`

关注 `reversal_fraction` 和 `clipping_fraction` 列——这些是拒绝的主要原因。如果某个场景的拒绝几乎全是同一种 QC 失败，可能需要调整对应的 `qc` 阈值或检查校准参数。

### evaluate 后

#### 第八步：看 `model_report_card.json`

核心字段：

```json
{
  "baseline_aggregate": [
    {"baseline_id": "oracle_target", "mean_rmse": 0.0, "mean_nrmse": 0.0, ...},
    {"baseline_id": "lfm_ideal", "mean_rmse": 0.15, "mean_nrmse": 0.35, ...},
    {"baseline_id": "lfm_controlled_degraded", "mean_rmse": 0.28, ...}
  ],
  "probe_aggregate": [...]
}
```

- `oracle_target` 的 RMSE 必须为 0（或机器精度级别接近 0）——这是 pipeline 自检，不为 0 说明基准存在数据泄漏或加载错误。
- `lfm_ideal` 的指标是"你能不靠反演就做到的最好水平"——低频先验的上限。
- `lfm_controlled_degraded` 是"场条件退化先验的水平"——更接近实际工作条件。

#### 第九步：看 `model_probe_metrics.csv`

筛选 `probe_metric_semantics == paired_probe_increment_error`，按 `probe_frequency_hz` 和 `probe_amplitude_multiplier` 分组看配对的增量恢复能力。如果某些频率在低振幅（0.25×）时增量误差就很大，说明该频率在实际反演中可能很难恢复精细变化。

#### 第十步：看 `model_geometry_metrics.csv`

对比不同几何族的 RMSE 和相关系数。规范几何族的对比特别有用——如果 `horizontal_thin_beds` 的薄层比例越小误差越大，说明模型在薄层分辨率上有系统性问题。这比场条件几何的聚合指标更容易定位。

---

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| `source_run_mismatch` | 配置的来源目录与正演可观测性分析记录不一致 | 检查配置的四个 `source_runs` 是否与 forward_observability 的 `run_summary.json` 中记录的闭环 |
| `GenerationRejected: global_reversal_fraction` | 高分辨率真值中 AI 对比度方向与校准方向不一致的像素太多 | 增大 `max_global_reversal_fraction`，或检查校准中态的方向约束是否正确 |
| `GenerationRejected: object_reversal_fraction` | 单个对象内部的 AI 剖面出现方向反转 | 增大 `max_object_reversal_fraction`，或检查该对象的 c0 是否被过度条件化 |
| `GenerationRejected: global_clipping_fraction` | 高分辨率真值超出区域 AI P01/P99 边界 | 增大 `max_global_clipping_fraction`，或检查校准的 P01/P99 是否过于保守 |
| `field_conditioned_acceptance_qc_failed` | 某些场景的接受率 < 50% 或尝试次数不足 | 查看 `scenario_catalog.csv` 定位失败的场景，调整对应的 `qc` 阈值或检查校准质量 |
| `impedance_calibration.json` 加载时 schema 不匹配 | 校准文件版本与当前代码不一致 | 重新运行 `calibrate` |
| 评估时 `oracle_target` 的 RMSE 不为 0 | 基准数据存在泄漏或加载路径错误 | 检查 `synthetic_benchmark.h5` 的 SHA-256 与 `benchmark_manifest.json` 是否一致 |

---

## 留到第二轮

- 通过/失败阈值：当前只冻结数据和报告卡，不设硬性指标门槛。后续根据实际反演模型的表现校准阈值。
- 多全局子波支持：当前基准只用第五步选出的一个标称子波。
- 三维基准扩展：当前只支持二维剖面。三维体积基准需要完全不同的几何构建和更大的存储/计算预算。
- 探针矩阵与实测模型表现的双向反馈：用 GINN 或类似模型在基准上的实际恢复能力，反向校准探针振幅梯度范围、频率选择策略和 LFM 退化参数。
- 模型评估指标标准化：定义一套跨模型可比的报告卡格式，让不同反演架构可以在同一基准上直接对比。
