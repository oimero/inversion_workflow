# 08 R0 实际工区零样本预测

`real_field_zero_shot.py` 是第八步的第一轮。它把第七步冻结的低频模型作为输入，用已训练的模型在真实工区数据上做直接推理——不做任何微调、不接触任何井标签。这是模型从合成基准走向真实工区的第一步：**它能从未见过的真实地震数据里看到什么。**

这一步同时输出两个模型的预测（含和不含横向混合器），用它们的差异来诊断横向信息的利用程度。

---

## 快速开始

```bash
python scripts/real_field_zero_shot.py
python scripts/real_field_zero_shot.py --config experiments/common/common.yaml
python scripts/real_field_zero_shot.py --device cuda
python scripts/real_field_zero_shot.py --stitch-strategy center_crop
python scripts/real_field_zero_shot.py --output-dir scripts/output/real_field_zero_shot_test
```

脚本会发现全局子波，但不会自动选择低频模型。调用者必须显式绑定一个已发布的第7步变体及其第6步井控集，再读取 `models` 列表中的每个模型依次推理。

`--stitch-strategy` 控制大体积分块推理后如何拼接重叠区域：`uniform` 用等权平均，`center_crop` 只用块中心的无边界效应区域。默认使用 `uniform`。

---

## 运行前需要什么

| 来源 | 文件 | 用途 |
|------|------|------|
| 第七步 | `lfm_run_summary.json` / `variant_manifest.csv` | 成功 run、显式 variant 和直接契约身份 |
| 第七步 | `variants/<variant_id>/lfm.npz` | 所选低频模型，作为推理的结构性输入 |
| 第六步 | `run_summary.json` / `well_control_manifest.csv` | 与 variant 直接契约一致的 canonical 井控 |
| 第五步 | `selected_wavelet.csv` | 全局子波，作为标准化参数之一 |
| 旁路 | 模型 checkpoint + `model_run_manifest.json` | 冻结的模型权重和标准化参数 |
| 旁路 | `input_reference_stats.json` | 地震数据值域变换的参考统计量 |
| 数据目录 | 地震体 | 真实工区地震数据 |

### 模型要求

`models` 列表中每个条目只需要 `model_run_dir`，指向一个包含 `model_run_manifest.json` 的目录。清单中必须包含 `synthetic_gate_evidence`，证明该模型已经通过了合成基准的闸门校验。

### 地震值域变换

当 `seismic_value_transform` 不是 `identity` 时，脚本需要模型的 `input_reference_stats.json` 来将真实地震数据标准化到模型训练时的分布。这个文件在模型训练时由 `ginn_v2.py train` 命令生成，放在 checkpoint 和 `normalization.json` 旁边。

---

## 配置参考

```yaml
# 以下字段写在 experiments/common/common.yaml 的 real_field_zero_shot 段下。

# --- 必填 ---
real_field_zero_shot:
  models:
    - model_run_dir: experiments/ginn_v2/results/<no_lateral_run>
    - model_run_dir: experiments/ginn_v2/results/<lateral_run>

# --- 可选（有默认值）---
real_field_zero_shot:
  mode: volume                    # 默认 volume；可选 section
  device: auto                    # 默认 auto；cuda / cpu / auto
  stitch_strategy: uniform        # 默认 uniform；可选 center_crop
  diagnostic_max_hz: 80.0         # 默认 80.0

  real_field_inputs:
    lfm_run_dir: scripts/output/real_field_lfm_<timestamp>
    variant_id: <descriptive-variant-id>
    well_control_run_dir: scripts/output/real_field_well_controls_<timestamp>
    seismic_value_transform: identity    # 默认 identity；常见 p99_abs_matched
    lfm_value_transform: identity        # v2 log(AI) 固定 identity

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
```

### `models`

必填。模型按 `model_run_dir` 逐一加载，每个条目只需 `model_run_dir`。通常包含两个：一个不含横向混合器（推断为 `no_lateral`），一个含横向混合器（推断为 `lateral`）。

### `mode`

支持 `volume`（全工区三维推理）和 `section`（单剖面快速验证）。默认 `volume`。剖面模式要求提供恰好一个剖面的定义文件（通过 `sections_file` 指定），且井 QC 自动禁用。

### `real_field_inputs`

- `seismic_value_transform`：地震数据值域变换方式。默认 `identity`。若设为 `p99_abs_matched` 等非 identity 值，需要模型的 `input_reference_stats.json`——脚本自动从第一个模型目录中读取。
- `lfm_run_dir` / `variant_id` / `well_control_run_dir`：三者均必填且禁止 `auto`；R0 会核对直接契约身份，并校验数据模式、轴、形状、数据类型和掩码等显式语义，不重算文件 SHA。
- `lfm_value_transform`：固定 `identity`；v2 主数组已经是波阻抗对数。
- `lfm_file` 和外部 `target_mask_file` 已退役；掩码只能来自所选变体。

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

### `spectral_qc`

不填时按 `diagnostic_max_hz` 自动三等分。需要自定义频带时：

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

1. 从成功发布的 `variant_manifest.csv` 解析显式 `variant_id`，读取对应 `variants/<variant_id>/lfm.npz`，核对变体与第6步的直接契约身份并做语义校验。
2. 按 `mode` 决定是加载整个地震体（体积模式）还是单个剖面（剖面模式）。
3. 应用 `seismic_value_transform`，把原始地震振幅变换到模型训练时的值域。
4. 从每个模型目录加载 checkpoint 权重和标准化参数。

### 第二阶段：逐个模型推理

对 `models` 列表中的每个模型：

1. 推断 `model_role`（`lateral` 或 `no_lateral`），创建同名输出子目录。
2. 对大体积进行分块推理（每个块的大小由模型训练时的 patch 大小决定）。
3. 按 `stitch_strategy` 拼接块边缘：`uniform` 对重叠区域做等权平均，`center_crop` 只取块中心的无边界效应区域。
4. 输出 `predictions.npz`，包含：
   - `stitched_pred_log_ai`：拼接后的最终预测对数波阻抗
   - `pred_delta_vs_lfm`：预测与低频模型的差值
   - `lfm_input`：输入的低频模型
   - `seismic_input`：变换后的地震输入
   - `valid_mask_model`：有效掩码
   - `stitching_weight`：拼接权重
   - 坐标轴（`ilines`、`xlines`、`twt_s`）

### 第三阶段：质量控制图表

1. 生成每个模型的三面板图（低频模型 / 差值 / 预测），检查推断结果的全局分布。
2. 当两个模型的 `model_role` 分别为 `lateral` 和 `no_lateral` 时，额外生成两者的差值图，按频带拆分以判断横向信息在哪些频率起作用。
3. 在剖面模式下，抽取井位置处的预测值与井上波阻抗做对比，生成井位 QC 图。
4. 写出一系列频带能量 QC 表格。

### 第四阶段：摘要

汇总所有模型的推理元数据、来源校验、输入分布异常告警和输出文件清单，写入 `real_field_zero_shot_summary.json`。状态固定为 `needs_forward_diagnostic`，表示这一步的输出尚未经过 R1 的正演闭环验证。

---

## 核心输出文件

所有文件在 `<output_root>/real_field_zero_shot_<timestamp>/` 下：

### 模型推理输出

每个模型一个子目录（`no_lateral/` 和 `lateral/`），内部包含：

| 文件 | 内容 |
|------|------|
| `predictions.npz` | 拼接后的预测数组，包含坐标轴和掩码 |
| `real_field_zero_shot_model_summary.json` | 该模型的推理参数、标准化参数、体积维度 |
| `<role>_pred_ai.zgy` | 线性 AI 预测体，数值为 `exp(stitched_pred_log_ai)`；内部 NPZ 仍保持 log(AI) |

### 全局 QC 和摘要

| 文件 | 内容 |
|------|------|
| `real_field_zero_shot_summary.json` | 来源路径、模型清单、坐标轴约定、掩码统计、输出文件清单 |
| `model_input_qc.csv` | 每个输入通道在推理前的标准化分布检查 |
| `real_field_spectral_qc.csv` | 每个模型每个频带的预测差值能量分布和可观测性证据联表 |
| `lateral_difference_band_qc.csv` | 两个模型预测差值的逐频带能量分析 |
| `well_prediction_qc.csv` | 剖面模式下，每口井的预测 vs 滤波 LAS 指标（体积模式不生成） |

### 图表

| 文件 | 内容 |
|------|------|
| `figures/<role>_lfm_delta_pred.png` | 三面板对比：低频模型 / 预测差值 / 预测 |
| `figures/lateral_minus_no_lateral.png` | 两个模型的预测差异 |
| `figures/lateral_minus_no_lateral_band_split.png` | 预测差异的逐频带拆分 |
| `figures/spectral_band_energy_qc.png` | 各频带预测差值能量柱状图 |
| `figures/wells/r0_well_prediction_qc_<well>.png` | 剖面模式下，井位预测与滤波 LAS 的 TWT 域对比 |

---

## 如何阅读结果

### 第一步：看终端输出

```
=== Real Field Zero-Shot ===
Output: scripts/output/real_field_zero_shot_<timestamp>
Models: 2
Status: needs_forward_diagnostic
```

确认两个模型都完成了推理。状态 `needs_forward_diagnostic` 是正常的——这意味着 R0 没有做正演验证，那是 R1 的职责。

### 第二步：看 `real_field_zero_shot_summary.json`

- 确认 `source_runs` 指向正确的第七步和第五步产物。
- 检查 `mask_contract.valid_fraction`：如果有效掩码占比很低，说明地震体中有大面积的无效区域。
- 看 `input_distribution_qc.warnings`：如果出现 `input_distribution_ood` 告警，说明真实地震数据在标准化后偏离了训练分布——超过 5% 的样本绝对值大于 5 个标准差是一个红色信号。

### 第三步：看预测三面板图

打开 `figures/<role>_lfm_delta_pred.png`：

- **左图（低频模型）**：低频背景。应该展现光滑的大尺度结构。
- **中图（Pred - 低频模型）**：预测差值。这部分是模型从地震数据里推断出的高频修正——如果几乎全为零，可能说明模型没有利用地震信息。
- **右图（Pred）**：完整的预测波阻抗。结构和数值范围是否地质合理。

### 第四步：看两模型差异

打开 `figures/lateral_minus_no_lateral.png` 和 `lateral_minus_no_lateral_band_split.png`：

- 整体差异的幅度和空间分布——如果差异集中在特定层段或构造位置，说明横向信息只在部分地区起作用。
- 逐频带拆分——差异如果集中在高频带，可能反映了模型的横向混合器主要在高频引入信息。

### 第五步：看频带能量

`real_field_spectral_qc.csv` 的每一行是一个模型在一个频带的预测差值能量。关注：

- `energy_ratio`：该频带占整体预测差值的比例。高频带比例过高可能意味着模型在高频产生了噪声。
- `pred_delta_spectrum_vs_synthetic_train`：真实工区预测差值的频带能量与训练时合成数据的预测差值标准差的比值。远大于 1 的频带说明模型在真实工区上产生的修正远大于训练数据中见过的幅度——需要谨慎对待。

### 第六步（仅剖面模式）：看井位 QC

`well_prediction_qc.csv` 逐井列出每个模型的 RMSE、偏差和相关系数。打开对应井的 `figures/wells/r0_well_prediction_qc_<well>.png`，观察预测的双程旅行时域趋势是否与滤波 LAS 一致。

---

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| variant 不存在或契约身份不一致 | Step 7 未完整发布，或 Step 6/7 配错 | 检查 `variant_manifest.csv` 并显式绑定同一 WellControlSet；旧 run 需重建 |
| `input_reference_stats.json` 缺失 | 模型训练时未生成参考统计文件 | 用当前版 `ginn_v2.py train` 重新训练一次 |
| `model_run_manifest.json` 缺少 `synthetic_gate_evidence` | 模型发布时未冻结合成基准闸门证据 | 重新训练并在发布时提供 gate 报告；成功 run 不允许事后盖章 |
| `input_distribution_ood` 告警 | 真实地震数据分布显著偏离训练分布 | 检查 `seismic_value_transform` 是否正确；考虑是否需要重新训练以匹配真实数据分布 |
| 剖面模式下 `sections_file` 包含多个剖面 | R0 剖面模式只支持单个剖面 | 只保留一个剖面定义，或改用体积模式 |
| WellControlSet 不匹配 | 剖面井 QC 使用的 Step 6 run 与 variant 不同 | 修正 `well_control_run_dir`；不得回退读取 Step 4 |

---

## 留到第二轮

- 多剖面批量推理（当前剖面模式只支持单个剖面）。
- 推理时的不确定性量化（如使用模型集成或多轮随机丢弃）。
- 在不重新训练的前提下对真实工区做自适应微调（如 AdaBN 或浅层适配）。
- 体积模式下沿井轨迹的自动对比（当前只在剖面模式下支持井 QC）。