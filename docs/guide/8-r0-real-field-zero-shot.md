# 08 R0 真实工区零样本预测

`real_field_zero_shot.py` 是第八步的第一轮。它把第七步冻结的低频模型作为输入，用已训练的模型在真实工区数据上做直接推理——不做任何微调、不接触任何井标签。这是模型从合成基准走向真实工区的第一步：**它能从未见过的真实地震数据里看到什么。**

每个模型由其训练实验的唯一标识（`experiment_id`）管理。可以一次加载多个实验的模型，通过显式配置的 comparison 来诊断不同架构或训练策略的差异。

---

## 快速开始

```bash
python scripts/real_field_zero_shot.py
python scripts/real_field_zero_shot.py --config experiments/common/common.yaml
python scripts/real_field_zero_shot.py --device cuda
python scripts/real_field_zero_shot.py --output-dir scripts/output/real_field_zero_shot_test
```

脚本会发现全局子波，但不会自动选择低频模型。调用者必须显式绑定一个已发布的第七步变体及其第六步井控集，再读取 `models` 列表中的每个实验依次推理。

旧版 `--stitch-strategy center_crop` 已停用。生产拼接固定使用均匀策略，确保任何包含有效样点的窗口都参与推理。

---

## 运行前需要什么

| 来源 | 文件 | 用途 |
|------|------|------|
| 第七步 | `lfm_run_summary.json` / `variant_manifest.csv` | 成功 run、显式 variant 和直接契约身份 |
| 第七步 | `variants/<variant_id>/lfm.npz` | 所选低频模型，作为推理的结构性输入 |
| 第六步 | `run_summary.json` / `well_control_manifest.csv` | 与 variant 直接契约一致的 canonical 井控 |
| 第五步（时间域） | `selected_wavelet.csv` | 时间域模型的冻结子波来源 |
| 深度域正演输入冻结旁路 | `forward_model_inputs.json` | 深度域冻结子波、速度关系和直接上游契约 |
| 旁路 | 模型 checkpoint + `model_run_manifest.json` | 冻结的模型权重和标准化参数，manifest 中必须包含 `experiment_id` |
| 旁路 | `input_reference_stats.json` | 地震数据值域变换的参考统计量 |
| 数据目录 | 地震体 | 真实工区地震数据 |

### 模型要求

`models` 列表中每个条目只需要 `experiment_dir`，指向一个包含 `model_run_manifest.json` 的实验目录。manifest 必须声明 `experiment_id`，模型按此标识管理输出子目录和 comparison 引用。

模型 manifest 和检查点必须声明采样域、采样单位与深度基准。它们必须和所选低频模型、井控及地震体完全一致；时间域模型不能读取深度域输入。

### 地震值域变换

当 `seismic_value_transform` 不是 `identity` 时，脚本需要模型的 `input_reference_stats.json` 来将真实地震数据标准化到模型训练时的分布。这个文件在模型训练时生成，放在 checkpoint 和 `normalization.json` 旁边。

---

## 配置参考

```yaml
# 以下字段写在 experiments/common/common.yaml 的 real_field_zero_shot 段下。

# --- 必填 ---
real_field_zero_shot:
  models:
    - experiment_dir: experiments/ablation/results/<experiment_id_a>
    - experiment_dir: experiments/ablation/results/<experiment_id_b>

# --- 可选（有默认值）---
real_field_zero_shot:
  mode: volume                    # 默认 volume；可选 section
  device: auto                    # 默认 auto；cuda / cpu / auto
  diagnostic_max_hz: 80.0         # 默认 80.0
  sections_file: experiments/common/real_field_sections.yaml  # section 模式下的一个或多个剖面

  real_field_inputs:
    lfm_run_dir: scripts/output/real_field_lfm_<timestamp>
    variant_id: <descriptive-variant-id>
    well_control_run_dir: scripts/output/real_field_well_controls_<timestamp>
    seismic_value_transform: identity    # 默认 identity；常见 p99_abs_matched
    lfm_value_transform: identity        # v3 log(AI) 固定 identity

  boundary:
    loss_or_eval_erosion_s: 0.0          # 默认 0.0
    prediction_taper_halo_s: 0.0         # 默认 0.0

  volume: {}                              # 默认 {}，留空即全工区
    # inline_start / inline_stop / xline_start / xline_stop
    # sample_start_s / sample_stop_s —— 可选裁剪子体积

  spectral_qc:                            # 不填时自动按 diagnostic_max_hz 三等分
    # bands 和 manual_override_reason 只在需要自定义频带时才配置

  well_qc:                                # 只在剖面模式下生效，体积模式自动禁用
    max_xy_distance_m: 300.0              # 默认不限制（但建议显式填写）
    include_deviated_wells: false         # 剖面模式默认 false

  volume_export:
    enabled: true                         # 默认 true
    inline_chunk_size: 16                 # 默认 16

  comparisons:                            # 可选，显式声明模型间对比
    - comparison_id: <descriptive-comparison-name>
      left: <experiment_id_a>
      right: <experiment_id_b>
```

上例是时间域配置。深度域使用米制字段，并且不配置频率诊断：

```yaml
real_field_zero_shot:
  boundary:
    loss_or_eval_erosion_m: 0.0
    prediction_taper_halo_m: 0.0
  volume:
    sample_start_m: 1000.0
    sample_end_m: 2500.0
```

### `models`

必填。每个条目仅接受 `experiment_dir`，指向一个包含 `model_run_manifest.json` 的实验目录。其他字段会被拒绝。模型按 manifest 中的 `experiment_id` 标识——脚本不再从架构名称或目录名推断模型角色。

通常配置两个实验：一个不含横向混合器，一个含横向混合器，用于诊断横向信息的利用程度。但配置本身不做此假设——任何合法 experiment_id 都可以加载。

### `comparisons`

可选。显式声明模型间的对比对。`left` 和 `right` 必须引用已在 `models` 中加载的 experiment_id，且两者的输出形状和采样轴必须一致。

```yaml
  comparisons:
    - comparison_id: lateral_vs_trace
      left: <experiment_id_with_lateral>
      right: <experiment_id_without_lateral>
```

不填时仍会生成每个模型的独立图表，但不会生成模型间差值图。comparison 不存在隐式的 `no_lateral`/`lateral` 推断。

### `mode`

支持 `volume`（全工区三维推理）和 `section`（一个或多个剖面批处理）。默认 `volume`。section 模式从 `sections_file` 读取 `sections` 列表；每个剖面独立写入输出子目录，并生成自己的预测、图表、井 QC 和摘要，同时根目录生成跨剖面的汇总摘要。

section 文件中的每个条目需要唯一的 `section_id` 和 `path`。对于 xline 步长不为 1 的工区，显式填写 `n_traces`，使路径坐标落在实际 survey 轴上：

```yaml
sections:
  - section_id: well_a
    path:
      - {inline: 1684.0, xline: 6111.0}
      - {inline: 1684.0, xline: 6591.0}
    n_traces: 121
```

批处理输出目录下的 `<section_id>/real_field_zero_shot_summary.json` 可以直接作为该剖面的 R1 输入；根目录摘要记录所有剖面的索引和汇总 QC。

### `real_field_inputs`

- `seismic_value_transform`：地震数据值域变换方式。默认 `identity`。若设为 `p99_abs_matched` 等非 identity 值，需要模型的 `input_reference_stats.json`——脚本自动从第一个模型目录中读取。
- `lfm_run_dir` / `variant_id` / `well_control_run_dir`：三者均必填且禁止 `auto`；R0 会核对直接契约身份，并校验数据模式、轴、形状、数据类型和掩码等显式语义，不重算文件 SHA。
- `lfm_value_transform`：固定 `identity`；v3 主数组已经是波阻抗对数。
- `lfm_file` 和外部 `target_mask_file` 不属于当前入口；掩码只能来自所选变体。

### `volume`

控制输入体积的裁剪范围。留空 `{}` 时使用全工区。支持以下可选键：

```yaml
  volume:
    inline_start: 100
    inline_stop: 500
    xline_start: 200
    xline_stop: 800
    sample_start_s: 1.0
    sample_stop_s: 2.5
```

四个范围键均为可选，只填需要裁剪的维度即可。常用于在小范围子体积上快速验证。

### `boundary`

控制推理边界的侵蚀和锥度参数，避免卷积边界伪影污染拼接结果：

| 参数 | 默认 | 含义 |
|------|------|------|
| `loss_or_eval_erosion_s` | `0.0` | 从 patch 边缘向内侵蚀多少秒，不参与评估 |
| `prediction_taper_halo_s` | `0.0` | 拼接时在 patch 边缘做锥度衰减的宽度 |

深度域边缘字段如下：

| 参数 | 含义 |
|------|------|
| `loss_or_eval_erosion_m` | 从块边缘向内侵蚀的米数 |
| `prediction_taper_halo_m` | 拼接锥度的米制宽度 |

秒制与米制字段不能同时出现，错域配置会立即失败。

### `spectral_qc`

不填时按 `diagnostic_max_hz` 自动三等分。需要自定义频带时：

该诊断只在时间域运行。深度域不会把米制采样间隔解释为秒，也不会发布赫兹频带结果。

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

`bands` 是频带列表，每条必须有 `name`、`low_hz`、`high_hz`。`manual_override_reason` 在手动配置时必填。

### `well_qc`

只在剖面模式下生效，体积模式下自动跳过。控制井位预测与滤波 LAS 的对比 QC。

```yaml
  well_qc:
    enabled: true
    max_xy_distance_m: 300.0
    include_deviated_wells: false
```

| 参数 | 含义 |
|------|------|
| `enabled` | 开关，默认 `true`（但体积模式下忽略） |
| `max_xy_distance_m` | 井到剖面迹的最大允许距离，必须显式填写 |
| `include_deviated_wells` | 是否包含斜井，剖面模式默认 `false` |

### `volume_export`

控制是否导出 ZGY 格式预测体，默认开启。

```yaml
  volume_export:
    enabled: true
    inline_chunk_size: 16
```

| 参数 | 默认 | 含义 |
|------|------|------|
| `enabled` | `true` | 是否导出 |
| `inline_chunk_size` | `16` | ZGY 写入时的 inline 分块大小 |

---

## 脚本在做什么

脚本分四个阶段：**加载 → 推理 → 质量控制 → 摘要**。

### 第一阶段：加载输入

1. 从成功发布的 `variant_manifest.csv` 解析显式 `variant_id`，读取对应 `variants/<variant_id>/lfm.npz`，核对变体与第六步的直接契约身份并做语义校验。
2. 按 `mode` 决定是加载整个地震体（体积模式）还是逐个加载 `sections_file` 中的剖面（剖面模式）。
3. 应用 `seismic_value_transform`，把原始地震振幅变换到模型训练时的值域。
4. 从每个模型目录加载 manifest，读取 `experiment_id`（不再从架构名称推断角色）、checkpoint 权重和标准化参数。

### 第二阶段：逐个模型推理

对 `models` 列表中的每个实验：

1. 从 manifest 读取 `experiment_id`，创建同名输出子目录。
2. 从 manifest 读取 patch 几何（训练时冻结的窗口大小和步长）。
3. 对大体积进行分块推理。窗口内只要存在一个有效样点就执行推理——不再因全局有效比例阈值丢弃窗口。无效位置和 padding 位置的输入通道填零，但不参与拼接累计。
4. 拼接固定使用均匀策略：每个有效样点取覆盖它的所有 patch 预测值的等权平均。
5. 推理完成后强制校验：每个有效样点必须至少被一个 patch 覆盖（即拼接权重大于零），否则整次运行失败并报告缺口坐标。
6. 输出 `predictions.npz`，包含：
   - `predicted_log_ai`：拼接后的最终预测对数波阻抗
   - `predicted_increment_log_ai`：预测增量
   - `input_lfm_log_ai`：输入的外部 LFM
   - `seismic_input`：变换后的地震输入
   - `valid_mask`：有效掩码
   - `stitching_weight`：拼接权重
   - `prediction_support_count`：每个有效样点被多少个 patch 覆盖
   - 坐标轴（`ilines`、`xlines`、`samples`）及采样域、单位、深度基准

### 第三阶段：质量控制图表

1. 生成每个模型的三面板图（低频模型 / 差值 / 预测），检查推断结果的全局分布。
2. 如果配置了 `comparisons`，生成每对 comparison 的模型间差值图，按频带拆分以判断差异在哪些频率起作用。
3. 在剖面模式下，对每个剖面抽取井位置处的预测值与井上波阻抗做对比，生成剖面级井位 QC 图，并在根目录汇总。
4. 写出一系列频带能量 QC 表格。

### 第四阶段：摘要

汇总所有模型的推理元数据、来源校验、输入分布异常告警和输出文件清单，写入 `real_field_zero_shot_summary.json`。每个模型的输入契约按 `model:<experiment_id>` 记录。状态固定为 `needs_forward_diagnostic`，表示这一步的输出尚未经过 R1 的正演闭环验证。

---

## 核心输出文件

所有文件在 `<output_root>/real_field_zero_shot_<timestamp>/` 下：

### 模型推理输出

体积模式下每个模型一个以 `experiment_id` 命名的子目录；剖面模式下先按 `<section_id>/` 分目录，再在其中按 `experiment_id` 分模型目录。每个模型目录内部包含：

| 文件 | 内容 |
|------|------|
| `predictions.npz` | 拼接后的预测数组，包含坐标轴、掩码和支持度 |
| `real_field_zero_shot_model_summary.json` | 该模型的推理参数、标准化参数、体积维度 |
| `<experiment_id>_pred_ai.zgy` | 线性 AI 预测体，数值为 `exp(predicted_log_ai)`；内部 NPZ 仍保持 log(AI) |

### 全局 QC 和摘要

| 文件 | 内容 |
|------|------|
| `real_field_zero_shot_summary.json` | 来源路径、模型清单、坐标轴约定、掩码统计、输出文件清单 |
| `model_input_qc.csv` | 每个输入通道在推理前的标准化分布检查 |
| `real_field_spectral_qc.csv` | 时间域每个模型每个频带的预测差值能量分布和可观测性证据联表 |
| `lateral_difference_band_qc.csv` | 模型间预测差值的逐频带能量分析 |
| `well_prediction_qc.csv` | 剖面模式下，每口井的预测 vs 滤波 LAS 指标（体积模式不生成） |

### 图表

| 文件 | 内容 |
|------|------|
| `<section_id>/figures/<experiment_id>_lfm_delta_pred.png` | 剖面模式下的三面板对比：低频模型 / 预测差值 / 预测 |
| `figures/<comparison_id>_difference.png` | comparison 中左右模型的预测差异 |
| `figures/<comparison_id>_difference_band_split.png` | comparison 预测差异的逐频带拆分 |
| `figures/spectral_band_energy_qc.png` | 各频带预测差值能量柱状图 |
| `<section_id>/figures/wells/r0_well_prediction_qc_<well>.png` | 剖面模式下，井位预测与统一井控在当前采样域的对比 |

---

## 如何阅读结果

### 第一步：看终端输出

```
=== Real Field Zero-Shot ===
Output: scripts/output/real_field_zero_shot_<timestamp>
Sections: 10
Models per section: 2
Status: needs_forward_diagnostic
```

确认所有模型都完成了推理。状态 `needs_forward_diagnostic` 是正常的——这意味着 R0 没有做正演验证，那是 R1 的职责。

### 第二步：看 `real_field_zero_shot_summary.json`

- 确认 `source_runs` 指向正确的第七步和第五步产物。
- 检查 `mask_contract.valid_fraction`：如果有效掩码占比很低，说明地震体中有大面积的无效区域。
- 看 `input_distribution_qc.warnings`：如果出现 `input_distribution_ood` 告警，说明真实地震数据在标准化后偏离了训练分布——超过 5% 的样本绝对值大于 5 个标准差是一个红色信号。
- 确认每个模型的 `experiment_id` 与预期一致。

### 第三步：看预测三面板图

体积模式打开 `figures/<experiment_id>_lfm_delta_pred.png`；section 模式打开对应的 `<section_id>/figures/<experiment_id>_lfm_delta_pred.png`：

- **左图（低频模型）**：低频背景。应该展现光滑的大尺度结构。
- **中图（Pred - 低频模型）**：预测差值。这部分是模型从地震数据里推断出的高频修正——如果几乎全为零，可能说明模型没有利用地震信息。
- **右图（Pred）**：完整的预测波阻抗。结构和数值范围是否地质合理。

### 第四步：看模型间差异

如果配置了 comparison，打开 `figures/<comparison_id>_difference.png` 和对应的频带拆分图：

- 整体差异的幅度和空间分布——如果差异集中在特定层段或构造位置，说明两个模型的差异只在部分地区起作用。
- 逐频带拆分——差异如果集中在高频带，可能反映了其中一个模型主要在高频引入信息。

### 第五步：看频带能量

`real_field_spectral_qc.csv` 的每一行是一个模型在一个频带的预测差值能量。关注：

- `energy_ratio`：该频带占整体预测差值的比例。高频带比例过高可能意味着模型在高频产生了噪声。
- `pred_delta_spectrum_vs_synthetic_train`：真实工区预测差值的频带能量与训练时合成数据的预测差值标准差的比值。远大于 1 的频带说明模型在真实工区上产生的修正远大于训练数据中见过的幅度——需要谨慎对待。

### 第六步（仅剖面模式）：看井位 QC

`well_prediction_qc.csv` 逐井列出每个模型的 RMSE、偏差和相关系数。打开对应井的 `figures/wells/r0_well_prediction_qc_<well>.png`，观察预测的趋势是否与滤波 LAS 一致。

---

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| variant 不存在或契约身份不一致 | Step 7 未完整发布，或 Step 6/7 配错 | 检查 `variant_manifest.csv` 并显式绑定同一 WellControlSet；旧 run 需重建 |
| manifest 缺少 `experiment_id` | 模型使用了旧版训练产物 | 用当前版 `ablation.py train` 重新训练 |
| `input_reference_stats.json` 缺失 | 模型训练时未生成参考统计文件 | 用当前版重新训练一次 |
| `model_run_manifest.json` 缺少 experiment_id | 旧 manifest schema | 重新训练当前实验 |
| `input_distribution_ood` 告警 | 真实地震数据分布显著偏离训练分布 | 检查 `seismic_value_transform` 是否正确；考虑是否需要重新训练以匹配真实数据分布 |
| section 条目缺少或重复 `section_id` | 批处理无法稳定组织输出 | 为每个剖面指定唯一的 `section_id` |
| WellControlSet 不匹配 | 剖面井 QC 使用的 Step 6 run 与 variant 不同 | 修正 `well_control_run_dir`；不得回退读取 Step 4 |
| R0 有效点没有预测支持 | 有效样点未被任何推理窗口覆盖 | 检查模型的 patch 几何合同与推理体积是否匹配；不要用数值回填掩盖 |
| comparison 左右模型输出形状不一致 | 两个实验使用了不同的 patch 几何或输出范围 | 确认两个实验的 patching 配置一致 |
| `models` 条目包含了 `experiment_dir` 之外的键 | 旧教程残留了其他字段 | 每个条目只保留 `experiment_dir` |

---

## 后续工作

- 推理时的不确定性量化（如使用模型集成或多轮随机丢弃）。
- 在不重新训练的前提下对真实工区做自适应微调（如 AdaBN 或浅层适配）。
- 体积模式下沿井轨迹的自动对比（当前只在剖面模式下支持井 QC）。
- 跨工区 adapted checkpoint 的输入分布 OOD QC 自动化。
