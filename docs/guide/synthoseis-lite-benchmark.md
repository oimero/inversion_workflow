# 合成基准生成与评估

`synthoseis_lite.py` 是工作流的研究旁路。它生成带真实工区统计约束的二维合成波阻抗剖面和配套地震数据，供模型训练和评估使用。它分为两个子命令：`calibrate`（从真实井数据冻结统计模型）和 `generate`（按统计模型产出合成样本）。

时间域和深度域共用同一入口脚本，通过配置中的 `sample_domain` 字段分派到不同分支。

---

## 快速开始

时间域：

```powershell
python scripts/synthoseis_lite.py --config <config-yaml> calibrate
python scripts/synthoseis_lite.py --config <config-yaml> generate \
    --suite field_conditioned \
    --impedance-calibration scripts/output/synthoseis_lite_calibrate_<timestamp>/impedance_calibration.json
```

深度域：

```powershell
python scripts/synthoseis_lite.py --config <config-yaml> calibrate
python scripts/synthoseis_lite.py --config <config-yaml> generate \
    --suite field_conditioned \
    --impedance-calibration scripts/output/synthoseis_lite_calibrate_<timestamp>/impedance_calibration.json
```

两条命令完全一致，差异由配置文件控制。

调试时常用的额外参数：

- `--well <well-name>`：只跑一口井（仅 calibrate 支持）
- `--debug-attempt-limit <N>`：限制每场景最大尝试数，快速验证流程
- `--geometry-family wedge`：只生成特定几何类型的剖面
- `--qc-only`：完整运行但不写入数据数组，仅验证接受率
- `--output-dir <path>`：指定输出目录

---

## 运行前需要什么

### 时间域

| 来源 | 文件 | 用途 |
|------|------|------|
| 第四步 | `well_tie_metrics.csv`、`filtered_las/` | 标定成功的井、滤波后的测井曲线 |
| 第五步 | `selected_wavelet.csv`、`selected_wavelet_summary.json` | 全局子波，用于正演和 mismatch 扰动 |
| 独立旁路 | `forward_observability/` | 只保存正演可观测性分析结果，不作为 v4 benchmark 的样本选择器 |
| 数据目录 | 地震体、解释层位、井分层 | 几何约束、目标窗口定义 |

### 深度域

与时间域的差异：

| 来源 | 文件 | 用途 |
|------|------|------|
| 第一步 | `well_inventory.csv` | 井口坐标 |
| 第五步（深度域）| `wavelet_batch_metrics.csv`、`shifted_preprocessed_las/`、`shifted_filtered_las/` | 深度平移后的两套 LAS |
| 深度域正演输入冻结旁路 | `forward_model_inputs.json` | 冻结子波和 AI–Vp 关系 |

深度域的井曲线来自深度域第5步产出的两套平移 LAS：滤波版用于背景拟合，全曲线版用于提取波阻抗变化幅度。

两种域都需要在配置中声明 `sample_domain` 和 `benchmark_schema`：

```yaml
# 时间域
synthoseis_lite:
  sample_domain: time
  benchmark_schema: synthoseis_lite_v4
  seismic_input:
    policy: observed_highres_forward
  forward_qc:
    highres_forward:
      enabled: true
      required: true

# 深度域
synthoseis_lite:
  sample_domain: depth
  benchmark_schema: synthoseis_lite_v4
  seismic_input:
    policy: observed_highres_forward
  seismic_forward:
    backend: auto        # auto | numpy | torch_cuda
    dtype: float64
```

v4 两个域都把 `seismic/seismic_observed` 作为网络输入；它位于 model axis，来自该
域的高分辨率正演和抗混叠路径。`seismic/seismic_model_consistent` 只用于 physics
或 closure，不能由 reader 作为输入回退。两个域的公共有效区域是
`masks/valid_mask`；高分辨率 forward support 只作为生成 QC。时间域的
`forward_qc.highres_forward` 必须启用且设为 required；深度域的等价高分辨率正演
由 depth 分支固定执行，`seismic_forward.backend=auto` 会自动选择 CUDA 或 NumPy。

---

## 配置参考

Synthoseis-lite 的配置分为两部分：公共工作流配置（`workflow_config` 或实验 YAML 自身）和 Synthoseis 专有配置（`synthoseis_lite` 段）。时间域支持两种写法：把工作流字段和 Synthoseis 字段写在同一个文件里，或者用 `workflow_config` 指向公共配置再叠加 Synthoseis 专有段。深度域要求使用 `workflow_config`。

### 顶层结构

```yaml
# 时间域（单文件写法）
data_root: <path>
output_root: <path>
seismic:
  domain: time
  file: <path-to-seismic>
  type: zgy
target_interval:
  horizons:
    - {name: top, well_top: <marker>, file: <horizon-file>}
    - {name: base, well_top: <marker>, file: <horizon-file>}

synthoseis_lite:
  sample_domain: time
  benchmark_schema: synthoseis_lite_v4
  global_seed: 20260615
  source_runs: ...
  sampling: ...
  geometry: ...
  sections: ...
  ...
```

```yaml
# 时间域（叠加写法）或深度域
workflow_config: <path-to-common-yaml>

synthoseis_lite:
  sample_domain: time          # 或 depth
  benchmark_schema: synthoseis_lite_v4
  seismic_input:
    policy: observed_highres_forward
  # 时间域还必须声明：
  # forward_qc.highres_forward.enabled=true
  # forward_qc.highres_forward.required=true
  global_seed: 20260615
  source_runs: ...
  ...
```

### `source_runs`

指定上游产物目录。留空则自动从输出目录下发现最新的对应步骤产物，填入则固定使用指定目录。

### `sampling`

```yaml
sampling:
  expected_output_dt_s: 0.002         # 模型网格的时间采样间隔（秒）
  vertical_oversampling_factor: 8     # 高分辨率真值网格相对于模型网格的过采样倍数
```

深度域模型网格的纵轴是 TVDSS（米，真垂深，海平面以下），分辨率由工区原生采样决定。

### `geometry`

控制剖面的空间属性：

```yaml
geometry:
  lateral_sample_interval_m: 25.0     # 横向采样间隔（米）
  field_conditioned:
    enabled: true
    target_zone:
      mode: filled_target_zone        # 目标区域模式
      nearest_distance_limit: null    # 可选，限制离最近控制井的距离
      ...
  canonical:                          # 仅时间域
    enabled: true
    lateral_sample_interval_m: 25.0
    lateral_samples: 128
    center_twt_s: 1.5
    ...
```

深度域仅开放 `field_conditioned`，不支持 `canonical`。

### `sections`

定义需要生成剖面的空间路径，每条路径是若干线号/道号点连成的折线：

```yaml
sections:
  - section_id: "section_a"
    path:
      - {inline: 100, xline: 200}
      - {inline: 150, xline: 250}
  - section_id: "section_b"
    path:
      - {inline: 300, xline: 400}
      - {inline: 350, xline: 450}
```

时间域的 section 路径使用线号/道号索引；深度域使用显式线号/道号坐标对序列，线号步长由相邻点的实际坐标差值决定。

### `impedance_attribute_generator`

控制随机波阻抗场的生成方式：

```yaml
impedance_attribute_generator:
  family: object_coefficients_v1     # 生成器族（时间域 v1，深度域 v2）
  state_threshold_sigma: 2.0         # 高/低阻抗态的划分阈值（σ 倍数）
  lateral:
    correlation_length_section_fractions: [0.1, 0.3, 1.0]
    coefficient_sigma_multipliers: [0.25, 0.50]
    thickness_log_sigma_values: [0.10, 0.25]
  qc:
    max_global_reversal_fraction: 0.10
    max_object_reversal_fraction: 0.25
    max_global_clipping_fraction: 0.005
    max_object_clipping_fraction: 0.02
  robust_scale:
    huber_delta_parent_sigma_floor: 0.05
    ...
  duration_modes:
    standard:
      minimum_highres_cells: 8
```

时间域使用 `object_coefficients_v1`，深度域使用 `object_coefficients_v2`。校准阶段自动写入正确的族标识，生成阶段读取校准产物后自动选择对应族。

### `generation`

控制生成量和接受率门禁：

```yaml
generation:
  attempts_per_scenario: 50          # 每场景最大尝试数
  duration_modes: [standard]
  geometry_families: [none, wedge, pinchout]
  geometry_directions: [left_to_right, right_to_left]
  acceptance_qc:
    minimum_attempts_per_scenario: 20
    warning_fraction: 0.80
    failure_fraction: 0.50
    enforcement: warn                 # warn 或 fail_fast
```

`enforcement` 为 `warn` 时，接受率不足会标记告警但正常完成；为 `fail_fast` 时，preflight 确认无法通过门禁后立即停止，不创建数据文件。

### `seismic_mismatch`

控制合成地震数据中引入的各类失配：

```yaml
seismic_mismatch:
  enabled: true
  noise:
    white_noise_rms_fraction: 0.05
    colored_noise_rms_fraction: 0.05
    absolute_noise_rms_floor: 0.01
    ...
  gain:
    global_log_sigma: 0.15
    tracewise_log_sigma: 0.15
    time_lateral_log_sigma: 0.15
    ...
  wavelet:
    phase_rotation_degrees: [-10.0, 10.0]
    time_shift_s: [-0.001, 0.001]
  combined:
    enabled: true
    phase_rotation_degrees: 10.0
    time_shift_s: 0.001
    gain_log_sigma: 0.10
    noise_rms_fraction: 0.05
```

深度域额外支持独立的米制深度静差（与秒制子波平移分开扫描），禁止 Hz 低通字段。

### `lfm`

`lfm` 控制低频模型的生成（理想低通和受控退化两种）。深度域的
`lfm.controlled_degraded.over_smoothing` 使用米制截止参数。v4 benchmark 不读取
`forward_observability`，也不接受 `probe_selection`。

```yaml
lfm:
  controlled_degraded:
    over_smoothing:
      enabled: true
      cutoff_hz: 6.0
      numtaps: 129
      kaiser_beta: 8.6
      blend: 1.0

```

频率探针和对应的 sample row、HDF5 group、报告字段不属于 v4 canonical benchmark；
需要观测性证据时，单独运行 `forward_observability.py` 并阅读其分析报告。

---

## 脚本在做什么

脚本的运行逻辑分为两个独立的阶段：**校准**和**生成**。必须先校准、再生成，校准产物是生成的必传入参。

### 校准阶段

校准的目标是：从真实工区的测井数据中提取波阻抗的统计规律，冻结为一个可复现的统计模型。

**1) 收集井数据。** 从上游产物中读取每口标定成功井的测井曲线（时间域读第四步的滤波 LAS，深度域读第5步的平移 LAS），结合井分层和解释层位，提取每口井在目标层段内的波阻抗曲线。

**2) 背景趋势拟合。** 对每口井的波阻抗做层位约束的背景趋势拟合，得到该井的"正常"波阻抗随深度变化的曲线。实际波阻抗减去背景趋势，得到残差——残差反映了波阻抗在正常值附近的起伏。

**3) 三态划分。** 按残差的幅度将每个深度点分为三类：高阻抗异常、正常背景、低阻抗异常。划分阈值由 `state_threshold_sigma` 控制（默认 2 倍标准差）。

**4) 对象提取。** 在连续的高阻抗或低阻抗异常段上，提取每个异常体的轮廓（厚度、幅度、纵向形状），统计所有井中异常体的尺寸分布、形状特征和空间转移概率。

**5) 冻结输出。** 将所有统计量写入 `impedance_calibration.json`，包含区域定义、各态的高斯参数、对象目录和转移矩阵。这个文件是后续生成阶段的唯一统计输入。

### 生成阶段

生成阶段用校准好的统计模型，创造出大量"像真实工区一样"的合成剖面和地震数据。

**1) 制定生成计划。** 根据配置中的剖面路径、几何类型、持续时间模式，排列出所有要生成的场景组合，每个场景是一个 `(section, geometry_family, geometry_direction, duration_mode)` 四元组。为目标接受率门禁计算每个场景至少需要的尝试次数，写入 `attempt_plan.csv`。

**2) Preflight 结构检查。** 对每个尝试，先快速检查剖面是否能容纳所需的层位约束（层位是否存在、间距是否足够），不执行昂贵的正演。通过检查的尝试才进入下一步。此阶段的统计写入 `preflight_attempts.csv` 和 `preflight_scenario_catalog.csv`。

**3) 生成波阻抗真值。** 对每个通过 preflight 的尝试：先从校准模型中随机采样对象参数（位置、厚度、幅度、横向展布），在剖面上放置高/低阻抗异常体，叠加背景趋势和层位约束，生成高分辨率（通常 8 倍过采样）的二维波阻抗场。

**4) 质量门控。** 检查生成的波阻抗场是否存在不合理的情况——比如整道反转、大面积限幅、异常体比例过高等。不合理的样本被拒绝并记录原因。每个场景的成功样本数和拒绝原因分布持续写入 `attempt_progress.csv`，可在另一个终端实时读取。

**5) 正演与失配。** 对通过质量检查的波阻抗场做两件事：一是抗混叠降采样到模型网格，计算理想地震响应（通过子波正演）；二是引入受控失配——子波相位旋转或时移、噪声（白噪/有色噪）、增益（全局/逐道/时空变化）、以及组合失配。每个失配变体都重新执行正演。

**6) 低频模型推导。** 从波阻抗真值出发，用理想低通滤波得到理想低频模型，再叠加受控退化（幅值误差、趋势倾斜、层段偏差、空间平滑偏差、局部缺失控制井偏差）得到退化低频模型。这些低频模型是训练时模型输入的一部分。

**7) 写入 HDF5。** 将每个成功样本的波阻抗真值、各种地震数据（理想、噪声、增益、组合失配）、低频模型、有效掩码等写入 `synthetic_benchmark.h5`。每个数据集附带单位、采样域、轴、形状和数据类型等元数据属性。

**8) 输出清单与报告。** 写 `sample_index.csv`（每个样本的元数据一行）、`benchmark_manifest.json`（全局元数据、直接上游契约、唯一 benchmark 指纹、接受率统计）、`scenario_catalog.csv`（场景级统计）。

### 深度域生成的特殊之处

深度域生成的整体流程与时间域相同，差异集中在几个关键点：

- **坐标体系**：纵轴是 TVDSS（米）而非双程旅行时（秒），模型网格分辨率由工区原生采样决定。
- **正演**：调 `cup.physics.forward_depth`（而非 `forward_time`），使用非平稳深度正演矩阵。子波米制宽度随速度变化。
- **低频模型**：退化参数使用米制（如平滑截止波长而非截止频率），禁止 Hz 低通字段。
- **失配**：额外支持独立的米制深度静差，与秒制子波相位和时移相互独立，可交叉组合。
- **可用套件**：仅 `field_conditioned`，不支持 `canonical` 和 `frequency_probe`。

---

## 核心输出文件

所有文件在 `<output_root>/synthoseis_lite_<calibrate|generate>_<timestamp>/` 下。

### 校准阶段

| 文件 | 内容 |
|------|------|
| `impedance_calibration.json` | 冻结的统计模型：区域定义、三态高斯参数、对象目录、转移矩阵和直接上游契约 |
| `zone_models.csv` | 每个区域每种态的样本数和分布参数 |
| `object_catalog.csv` | 所有提取到的异常体：来源井、区域、态、轮廓系数、厚度 |
| `transfer_matrix.csv` | 态之间的转移概率 |
| `figures/` | 校准图：分区域波阻抗剖面、三态划分结果、对象尺寸分布 |
| `run_summary.json` | 来源路径、井数统计、配置快照 |

### 生成阶段

| 文件 | 内容 |
|------|------|
| `synthetic_benchmark.h5` | 所有样本的数据数组（波阻抗、地震、低频模型、掩码），附带完整元数据属性 |
| `sample_index.csv` | 每个样本一行：ID、所属剖面、几何类型、状态、observed 输入/physics 参照/有效掩码 HDF5 路径；变体另有 family、operator source、参数和 QC |
| `benchmark_manifest.json` | 全局元数据：域、schema、状态、直接上游契约、唯一 benchmark 指纹和接受率统计 |
| `scenario_catalog.csv` | 每个场景的尝试数、成功数、接受率、门禁状态 |
| `attempt_plan.csv` | 全部计划的尝试列表 |
| `attempt_progress.csv` | 增量进度日志，运行时持续刷新 |
| `preflight_attempts.csv` | preflight 阶段的逐尝试结果 |
| `preflight_scenario_catalog.csv` | preflight 阶段的场景级统计 |
| `preflight_summary.json` | preflight 汇总 |
| `rejection_reason_summary.csv` | 拒绝原因的分布统计 |
| `section_geometry_qc.csv` | 各剖面在各层位上的横向支撑状态 |
| `generation_qc.csv` | 正式生成阶段逐尝试的质量指标和拒绝原因 |
| `generation.log` | 带时间戳的生成日志 |
| `well_horizon_consistency.csv` | 井分层与地震解释层位的一致性检查（时间域） |
| `run_summary.json` | 配置、来源、计数汇总 |

---

## 如何阅读结果

### 第一步：看终端输出

```
=== synthoseis-lite ===
Command: generate
Output: scripts/output/synthoseis_lite_generate_<timestamp>
Status: success
```

正常结束只有这三行。如果状态是 `completed_with_warnings`，说明某些场景的接受率低于告警线但高于失败线；如果是 `development_limited`，说明使用了 `--debug-attempt-limit`，产物不能用于正式训练。

### 第二步：看 `benchmark_manifest.json`

关注顶层字段：

- `sample_domain`：确认是 `time` 还是 `depth`
- `status`：`success` / `completed_with_warnings` / `development_limited`
- `qc_only` / `training_consumable`：QC-only 运行不能用于训练
- `accepted_parent_realizations` / `rejected_parent_realizations`：全局接受率
- `input_contracts`：只列 calibration 等直接上游契约
- `contract_fingerprint_sha256`：benchmark 唯一契约指纹

### 第三步：看 `scenario_catalog.csv`

按 `acceptance_rate` 排序，关注低于 `warning_fraction` 的场景。如果某个场景的接受率特别低（比如 wedge + standard 组合只有 30%），说明当前的几何约束和对象参数很难在该场景下生成合理的波阻抗场——可能需要调整 `impedance_attribute_generator.qc` 的门控阈值或 `geometry.field_conditioned.target_zone` 的约束。

### 第四步：看 `rejection_reason_summary.csv`

如果某个拒绝原因占比异常高（如 `global_reversal_fraction_exceeded` 超过 30%），说明校准出的对象参数与几何约束不匹配，可能的原因包括背景趋势拟合不稳定、层位约束过紧、或对象尺寸分布与实际工区偏离较大。

### 第五步：抽查样本

打开 `sample_index.csv`，筛出 `status == ok` 的行，随机挑几个 `hdf5_group`，用 HDF5 工具查看对应的波阻抗剖面和地震剖面。重点关注：异常体的形态是否自然、层位约束是否正确施加、地震响应是否与波阻抗变化在视觉上一致。

抽查地震字段时，网络输入读取 `seismic_input_dataset` 指向的
`seismic_observed`；`seismic_model_consistent_dataset` 是 physics/closure 参照，
两者应分别查看，不能用 sample kind 推断路径。

---

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| `sample_domain/branch mismatch` | 配置的域或 schema 与脚本期望不一致 | 检查 `sample_domain` 和 `benchmark_schema` 是否正确配对 |
| `source_run mismatch` | 上游产物与配置记录的来源不一致 | 重新校准，确保使用同一批上游产物 |
| `No finite well curves in any zone` | 所有井在目标层段内都没有有效曲线 | 检查井分层与 LAS 覆盖范围是否匹配 |
| `preflight: insufficient horizon support` | 剖面路径上的层位约束不足 | 检查解释层位覆盖面，或更换剖面路径 |
| `scenario acceptance below failure_fraction` | 某场景接受率低于失败线且 enforcement 为 fail_fast | 放宽 QC 门控、调整剖面路径、或增加 attempts_per_scenario |
| `scenario acceptance below warning_fraction` | 接受率在告警线和失败线之间 | 检查 `rejection_reason_summary.csv` 定位主要拒绝原因 |
| `contract fingerprint/schema mismatch` | 输入仍是旧 schema，或所选直接上游契约与校准记录不一致 | 重建相关上游、校准和 benchmark；要求上游契约的哈希字段与校准记录匹配，不使用兼容回退 |
| `calibration schema mismatch` | 校准产物版本与生成阶段期望不一致 | 重新运行校准 |

---

## 深度域补充说明

### 与时间域的关键差异速查

| 维度 | 时间域 | 深度域 |
|------|--------|--------|
| 纵轴 | TWT（秒） | TVDSS（米） |
| 井曲线来源 | Step 4 filtered LAS | Step 5 shifted LAS |
| 正演 | `forward_time` | `forward_depth` |
| 可用套件 | canonical, field_conditioned, seismic_variant | 仅 field_conditioned |
| 生成器族 | `object_coefficients_v1` | `object_coefficients_v2` |
| 低频模型截止 | Hz | 米制 |
| 深度静差 mismatch | 无 | 有，与子波平移独立 |
| 正演可观测性 | 独立 observability 分析旁路 | 独立 observability 分析旁路 |

### 深度域两套 LAS 的分工

校准阶段的背景拟合使用 `shifted_filtered_las`（滤波后的 AI），避免背景趋势被测井曲线中的尖刺支配；三态划分和对象残差统计使用 `shifted_preprocessed_las`（全曲线 AI），保证异常体的幅度统计来自未经滤波的真实数据。

---

## 留到第二轮

- 深度域 canonical 套件的支持，以及是否需要为特定窗口重新组织 observability 分析。
- 是否允许按区块或层段分组校准，产出多组统计模型。
- 训练数据拆分比例的自动推演（当前由 GINN 训练端按 `parent_realization_id` 哈希派生）。
- 校准图的交互式版本。
