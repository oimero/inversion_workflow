# 08 GINN v2 主体反演

`body_train.py` 是工作流的第八步。它读取第六步井控、第七步选定的低频模型和第五步选定子波，先进行地震自监督预训练，再用可信井曲线微调，输出模型权重、逐轮评价和局部质检结果。

本文按时间域工区说明。主体输出由平滑后的初始模型与受低频约束的网络修正共同组成，主体平滑、井目标导数和波形质检窗口沿 TWT 轴计算，尺度以秒表达。深度域配置见[深度域工作流](depth-domain-workflow.md)。

---

## 快速开始

```powershell
python scripts/body_train.py
python scripts/body_train.py --config "<body-config-yaml>" --output-dir "<training-output-dir>"
```

默认读取 `experiments/ginn_v2/ginn_v2.yaml`，依次完成预训练和微调。运行前将配置中的资源路径、模型变体和可信井名单填写完整。未指定输出目录时，结果写入配置输出根目录下的 `ginn_v2_body_inversion_<timestamp>/`。

### 分阶段运行

```powershell
python scripts/body_train.py --config "<body-config-yaml>" --stage pretrain --output-dir "<pretraining-output-dir>"
python scripts/body_train.py --config "<body-config-yaml>" --stage finetune --pretrain-checkpoint "<pretrain-checkpoint-file>" --output-dir "<finetuning-output-dir>"
```

| 阶段 | 用途 | 权重来源 |
|------|------|----------|
| `all` | 连续执行预训练和微调，默认选项 | 本次预训练最后一轮 |
| `pretrain` | 只运行地震自监督预训练 | 新建网络 |
| `finetune` | 复用预训练权重进行井约束微调 | 显式指定的预训练权重 |

复用预训练时，网络结构、主体尺度、补丁半径、采样方向、地震特征和预训练损失设置必须与权重中保存的配置一致。

训练入口支持用命令行覆盖上游输入：`--lfm-run-dir`、`--variant-id`、`--well-control-run-dir` 和 `--wavelet-generation-run-dir`。分阶段微调还需要 `--pretrain-checkpoint`。

### 可选：全体积预测

训练结束后会生成井旁与局部剖面质检。全体积预测由单独的入口执行：

```powershell
python scripts/body_infer.py --config "<body-config-yaml>" --checkpoint "<selected-checkpoint-file>" --output-dir "<volume-output-dir>"
```

预测需要使用与所选权重一致的上游输入及模型变体。可以先预测局部方形区域：

```powershell
python scripts/body_infer.py --config "<body-config-yaml>" --checkpoint "<selected-checkpoint-file>" --output-dir "<tile-output-dir>" --smoke-tile-size <tile-size> --skip-segy-export
```

局部区域需同时使用 `--skip-segy-export`，因为地震格式导出要求完整工区体。未完成的预测运行可以用相同输出目录和 `--resume` 接续。

预测入口同样支持 `--lfm-run-dir`、`--variant-id`、`--well-control-run-dir` 和 `--wavelet-generation-run-dir` 覆盖时间域上游输入。

---

## 运行前需要什么

| 来源 | 文件或数据 | 用途 |
|------|------------|------|
| 第六步 | `run_summary.json`、`well_control_manifest.csv` 和逐井 NPZ | 可信井的阻抗曲线、有效样点与逐样点位置 |
| 第七步 | `lfm_run_summary.json`、`variant_manifest.csv` 和选定变体的 `lfm.npz` | 初始对数阻抗、有效掩码、输出轴和建模配置 |
| 第五步子波目录 | `selected_wavelet.csv` | 固定子波的时间和振幅，直接由第八步读取 |
| 工区配置 | 时间域地震体 | 网络输入与地震形态监督 |

第六步与第七步都需要完成，第八步同时读取两者。选定的低频模型必须是三维体，并完整覆盖当前地震体的时间轴和线网，与井控集采用相同的空间几何。通常使用第七步的全工区输出模式。

可信井名单需要显式填写，名单中的每口井都应存在于第六步成功井中，并具有可用的训练目标。

---

## 配置参考

工作流公共配置继续提供数据根目录、输出根目录、时间域地震和井资产。第八步的配置位于 `ginn_v2_body_inversion`。下面的尖括号需要替换为本地资源、井名或尺度，其余数值为示例训练设置。

```yaml
workflow_config: "<workflow-config-yaml>"

ginn_v2_body_inversion:
  inputs:
    lfm_run_dir: "<lfm-run-dir>"
    variant_id: "<lfm-variant-id>"
    well_control_run_dir: "<well-control-run-dir>"
    wavelet_generation_run_dir: "<wavelet-generation-run-dir>"

  training:
    trusted_well_names:
      - "<trusted-well-a>"
      - "<trusted-well-b>"
    body_smoothing_fwhm_s: "<body-fwhm-s>"
    waveform_qc_dynamic_window_s: "<qc-window-s>"
    patch_radius: 8
    orientations: [inline, xline]
    batch_size: 8
    pretrain_epochs: 1
    finetune_epochs: 3
    pretrain_learning_rate: 0.001
    finetune_learning_rate: 0.0002
    max_train_centers: 512
    max_validation_centers: 128
    review_fraction: 0.04
    validation_gap_m: 300.0
    validation_anchor: maxmin
    well_batch_multiplier: 2
    seismic_feature_mode: global_trace_normalized
    seismic_balance_window_samples: 61
    seismic_balance_floor_fraction: 0.10
    device: cpu
    log_every_batches: 20

    loss_weights:
      seismic_shape: 1.0
      trusted_well_body: 1.0
      trusted_well_derivative: 0.5
      trusted_well_seismic_shape: 0.0
      lfm_anchor: 1.0
      lambda_shape: 0.25

    selection_weights:
      well_rmse: 1.0
      amplitude_mapping: 0.0

    warnings:
      pretrain_masked_corr_improvement: 0.01
      pretrain_masked_shape_ratio: 0.99
      masked_corr_drop_tolerance: 0.01
      well_pooled_rmse_ratio_max: 1.0
      seismic_body_amplitude_spearman_max: 1.0
```

### 上游输入

| 参数 | 含义 |
|------|------|
| `lfm_run_dir` | 第七步运行目录 |
| `variant_id` | 该运行中实际存在的模型变体名称 |
| `well_control_run_dir` | 第六步运行目录 |
| `wavelet_generation_run_dir` | 第五步运行目录，直接读取其中的 `selected_wavelet.csv` |

四项上游输入必须显式填写，可以用对应的命令行参数覆盖。相对路径均从仓库根目录解析，目录与变体按配置明确解析。

第五步目录中的 `selected_wavelet.csv` 直接提供子波时间和振幅。子波有自己的规则时间轴，子波坐标可独立于地震体轴，正演根据地震采样步长处理子波采样。

### 主体尺度与补丁

| 参数 | 含义 |
|------|------|
| `body_smoothing_fwhm_s` | 高斯平滑的半高全宽，沿 TWT 轴表达，单位为秒，必须为正 |
| `waveform_qc_dynamic_window_s` | 波形质检局部相关窗口，沿 TWT 轴表达，单位为秒，必须为正 |
| `patch_radius` | 中心道两侧读取的道数，完整补丁宽度为两倍半径加一 |
| `orientations` | 采样方向，可以包含 `inline`、`xline` 或两者 |
| `seismic_feature_mode` | 地震输入的振幅处理方式 |
| `seismic_balance_window_samples` | 局部振幅平衡窗口，至少为 3 的奇数样点数 |
| `seismic_balance_floor_fraction` | 局部振幅尺度的下限比例，取值在 0 与 1 之间 |

两种地震特征模式分别为 `global_trace_normalized` 和 `local_amplitude_balanced`；后者通过局部振幅平衡降低强振幅位置对输入尺度的影响。该设置控制网络特征，地震形态损失仍使用真实地震观测。

网络默认使用六个输入通道、32 个隐藏通道和四个残差块。可在 `training.network` 中调整 `hidden_channels`、`residual_blocks`、`lateral_kernel` 和 `sample_kernel`；两个卷积核长度均需为奇数，输入通道数需要与补丁读取器一致。

### 训练与空间划分

| 参数 | 含义 |
|------|------|
| `pretrain_epochs` / `finetune_epochs` | 两阶段的训练轮数，均需为正整数 |
| `pretrain_learning_rate` / `finetune_learning_rate` | 两阶段学习率，均需为正 |
| `batch_size` | 每批补丁数量 |
| `max_train_centers` / `max_validation_centers` | 训练与评价使用的中心位置数量上限 |
| `review_fraction` | 保留空间验证块的比例，取值在 0 与 1 之间 |
| `validation_anchor` | 验证块位置：四角 `maxmax`、`maxmin`、`minmax`、`minmin`，或 `center` |
| `validation_gap_m` | 训练区与验证块之间保留的米制间隔 |
| `well_batch_multiplier` | 每轮可信井批次的重复倍数 |
| `trusted_well_names` | 参与井约束的井名，必须非空且唯一 |
| `device` | 计算设备，例如 `cpu` 或 `cuda` |

空间划分依据真实平面距离，补丁半径依据道数；线号的实际间隔由工区几何提供。可信井的全部有效目标样点用于井约束训练，井误差反映这些井的拟合效果；空间验证块用于地震补丁评价。

### 损失与模型选择

| 参数 | 作用 |
|------|------|
| `loss_weights.seismic_shape` | 合成地震与真实地震的波形形态匹配 |
| `loss_weights.trusted_well_body` | 主体输出与平滑井曲线的数值匹配 |
| `loss_weights.trusted_well_derivative` | 主体输出与井目标的 TWT 导数匹配 |
| `loss_weights.trusted_well_seismic_shape` | 可信井批次上的地震形态匹配 |
| `loss_weights.lfm_anchor` | 约束网络修正中的低频部分 |
| `loss_weights.lambda_shape` | 波形形态损失中的归一化振幅误差权重 |
| `selection_weights.well_rmse` | 模型选择时的井误差权重 |
| `selection_weights.amplitude_mapping` | 模型选择时的地震振幅与主体局部振幅关联惩罚权重 |

当第七步变体的 `filter.enabled` 为真时，主体输出仍执行该低通设置定义的低频修正投影；只有在 `filter.enabled` 为假且 `loss_weights.lfm_anchor` 为零时，主体输出才只使用高斯平滑。选择权重必须非负，且至少一项大于零。

微调跑完配置的全部轮次后，按“井误差相对预训练的比例”与“振幅关联绝对值”的加权和选择最低分轮次；分数相同时选择较早轮次。

### 警告阈值

| 参数 | 当前作用 |
|------|----------|
| `masked_corr_drop_tolerance` | 允许遮挡中心道评价的相关系数相对预训练下降多少 |
| `well_pooled_rmse_ratio_max` | 允许井汇总误差相对预训练的最大比例 |
| `seismic_body_amplitude_spearman_max` | 地震振幅与主体局部振幅关联绝对值上限 |
| `pretrain_masked_corr_improvement` | 预训练后遮挡中心道的地震相关性相对零修正基线应提高的下限 |
| `pretrain_masked_shape_ratio` | 预训练后地震形态损失相对零修正基线的比例上限 |

警告写入阶段状态、所选权重摘要和微调结果，模型选择仍按选择权重执行。

### 全体积预测配置

```yaml
ginn_v2_volume_inference:
  checkpoint: "<selected-checkpoint-file>"
  device: cpu
  batch_size: 128
  log_every_sections: 10
  min_lfm_support: 8
  orientations: [inline, xline]
  exports:
    body_increment_log_ai: true
    direction_disagreement: false
```

预测批量大小可以通过 `--batch-size` 覆盖。启用两个方向时，同一位置的有效结果取平均，并记录方向差异。导出选项分别控制主体相对低频模型的增量体和方向差异体。

---

## 脚本在做什么

### 第一阶段：准备输入与训练目标

1. **建立共同采样空间。** 读取低频模型、井控和时间域地震，核对 TWT 采样轴及平面几何，直接读取第五步的选定子波。
2. **构造地震补丁。** 沿配置方向读取邻道地震、初始阻抗和有效性信息，建立训练区、空间验证块及两者之间的隔离区。
3. **建立可信井目标。** 将可信井的滤波阻抗曲线插值到共同 TWT 轴，按秒制尺度平滑，生成主体曲线、有效样点与导数监督，并将井目标关联到对应补丁。

### 第二阶段：形成主体输出

1. **平滑初始模型。** 直接沿 TWT 采样坐标对初始对数阻抗执行以秒为尺度的高斯平滑。
2. **计算网络修正。** 卷积网络读取邻道补丁，输出中心道修正；对修正后的初始模型使用同一平滑，再减去平滑初模，得到主体尺度的修正。
3. **限制低频变化。** 使用第七步的低通设置提取修正中的低频部分，将其从修正中减去，再叠加到平滑初模上。该过程同时用于训练和推理。

### 第三阶段：地震自监督预训练

1. **遮挡中心输入。** 遮挡中心道的地震特征，使用邻道地震和初始模型预测中心道主体阻抗。
2. **执行时间域正演。** 从主体阻抗计算反射系数，与固定子波卷积，生成中心道合成地震；计算限定在连续有效支撑段内。
3. **匹配形态与低频。** 将合成和真实地震逐道去均值并按均方根振幅归一化，联合优化相关性、归一化振幅误差和低频锚定。每轮评价并保存权重，最后一轮作为微调起点。

### 第四阶段：可信井约束微调

1. **交替使用三类批次。** 依次处理中心道遮挡的地震补丁、中心道可见的地震补丁和可信井补丁。可信井按井均衡采样，并按配置重复批次。
2. **约束主体与导数。** 在地震形态和低频锚定之外，增加井主体曲线及其 TWT 导数的匹配；需要时加入可信井位置的地震形态约束。
3. **逐轮评价并选择。** 记录空间验证结果、井曲线误差、波形相关性和局部振幅关联。全部微调轮次完成后，选出综合分数最低的权重。

### 第五阶段：生成质检与可选预测

1. **写出局部质检。** 使用所选权重生成井旁主体曲线、井处正演波形和保留区域剖面，汇总评价与警告。
2. **按需预测全体积。** 单独运行预测入口，逐方向计算中心道结果，合并有效预测，保存主体阻抗、方向支持和边界填补记录，并按配置导出地震格式体。

---

## 核心输出文件

### 训练运行

```text
ginn_v2_body_inversion_<timestamp>/
├── training.log
├── baseline_metrics.json
├── input_contract.json
├── split.json
├── well_roles.json
├── pretraining/
│   ├── checkpoints/epoch_<n>.pt
│   ├── shared_checkpoint.pt
│   └── validation/epoch_<n>/
│       ├── metrics.json
│       ├── fixed_validation_<orientation>.png
│       ├── fixed_amplitude_<orientation>.png
│       └── well_waveform_qc/
├── pretrain_metrics.json
├── finetuning/
│   ├── checkpoints/epoch_<n>.pt
│   ├── selected_checkpoint.pt
│   └── validation/epoch_<n>/
├── finetune_result.json
├── selected_checkpoint.pt
├── selected_checkpoint.json
├── review_package/
│   ├── fixed_well_profiles/
│   ├── blind_sections/
│   ├── well_waveform_qc/
│   └── review_manifest.json
└── body_inversion_status.json
```

预训练单独运行时只生成该阶段成果；完成微调后才生成所选权重与对应质检。逐轮权重、指标及摘要也保存在本次运行目录中。

| 文件或目录 | 用途 |
|------------|------|
| `body_inversion_status.json` | 本次运行阶段、权重路径、质检清单和警告 |
| `input_contract.json` | 上游输入路径、模型变体和输入设置 |
| `baseline_metrics.json` | 零修正基线，即平滑初始模型的评价指标 |
| `split.json` / `well_roles.json` | 空间划分、补丁身份与可信井样点角色 |
| `pretraining/shared_checkpoint.pt` | 最后一轮预训练权重，供微调复用 |
| `pretrain_metrics.json` | 最后一轮预训练指标 |
| `finetune_result.json` / `selected_checkpoint.json` | 候选权重、所选轮次、指标和警告 |
| `pretraining/validation/epoch_<n>/` / `finetuning/validation/epoch_<n>/` | 每轮评价指标、固定剖面、局部振幅图和逐井波形质检 |
| `selected_checkpoint.pt` | 所选微调权重，供下游推理读取 |
| `review_package/fixed_well_profiles/` | 固定井位的主体曲线与初模、井目标对比 |
| `review_package/blind_sections/` | 保留空间区域的主体与初模剖面 |
| `review_package/well_waveform_qc/` | 所选模型的逐井正演波形、局部相关和指标 |

### 全体积预测

预测运行与训练目录分开。未指定预测输出目录时，写入 `experiments/ginn_v2/results/volume_<timestamp>/`。主导出体为 `ginn_v2_body_linear_ai`，格式与源地震一致；线性声阻抗单位为 `m/s*g/cm3`。

| 文件或目录 | 用途 |
|------------|------|
| `volume_inference_summary.json` | 预测模式、权重、轴范围、覆盖统计及导出路径 |
| `section_predictions/` | 分方向的预测缓存，供结果合并和未完成运行接续 |
| `figures/center_inline.png` / `figures/center_xline.png` | 两个代表性剖面的主体、初模、增量和方向差异，有效支撑满足要求时生成 |
| `figures/coverage.png` | 预测支持与填补情况 |
| `ginn_v2_body_linear_ai.<segy_or_zgy>` | 主体线性阻抗体 |
| `ginn_v2_body_increment_log_ai.<segy_or_zgy>` | 主体减去原始低频模型的对数阻抗增量体，按配置导出 |
| `ginn_v2_direction_disagreement_log_ai.<segy_or_zgy>` | 两方向主体预测差异体，按配置导出 |

---

## 如何阅读结果

### 第一步：看运行摘要与逐轮指标

从 `input_contract.json` 和 `well_roles.json` 确认上游目录、低频模型变体与可信井角色，再看 `body_inversion_status.json` 中的阶段状态和警告。`selected_checkpoint.json` 与 `finetune_result.json` 保存所选轮次及其指标；选择规则见本页“损失与模型选择”。预训练最后一轮是微调比较的基准。

### 第二步：看井旁主体曲线

对照初始模型、平滑初模、井目标和主体预测，判断主体是否贴合可信井的整体结构。曲线有效范围应与上游井控覆盖一致。

### 第三步：看井处波形质检

对照真实地震、主体正演地震和局部相关曲线。波形相似度与井主体误差需要一起阅读，局部振幅关联指标可辅助识别主体变化过度跟随地震振幅的情况。

### 第四步：看保留区域剖面

查看保留地震补丁对应剖面的主体结构与初模差异，关注横向连续性、层段内的变化和两个采样方向的表现。井曲线指标反映参与约束井的拟合效果，空间验证结果反映保留地震补丁的表现。

### 第五步：需要时检查预测体

查看预测支持与填补记录，再读取主体体和方向差异图。双方向均有有效结果的位置可以直接比较两方向的预测；边界或有效支撑不足处需结合填补方式理解。

---

## 常见失败原因

| 问题 | 原因 | 处理方式 |
|------|------|----------|
| 输入目录或变体未填写 | 第八步要求显式选择上游成果 | 填写四项上游输入，或通过对应命令行参数覆盖 |
| 低频模型为剖面或覆盖不完整 | 训练入口要求三维体及完整工区轴与几何 | 在第七步生成与当前地震轴一致的全工区变体 |
| 时间轴、线网或几何不一致 | 井控、低频模型和地震来自不同采样空间 | 使用同一工区配置重建对应上游步骤 |
| 第五步子波无法加载 | 运行目录缺少 `selected_wavelet.csv`，或子波列不满足规则采样和归一化约束 | 核对第五步运行目录、`time_s` 与 `amplitude` 列及子波质量检查结果 |
| 可信井不存在或没有可用目标 | 名单与第六步成功井不一致，或曲线支撑不足 | 核对井名、有效曲线段与目标层覆盖 |
| 低通未启用或曲线段过短 | 主体分解和低频锚定需要可用的上游低通设置 | 检查第七步滤波设置与有效段长度 |
| 训练或评价集合为空 | 验证块、隔离距离或补丁范围排除了全部可用位置 | 调整空间划分或补丁半径 |
| 地震段零方差或支持点不足 | 波形形态损失无法计算 | 检查地震数据和目标有效支撑 |
| 预训练权重配置不匹配 | 网络或预训练语义设置与权重不一致 | 使用匹配的配置，或重新预训练 |
| 设备不可用 | 配置的计算设备无法使用 | 改用可用设备，或检查本机计算环境 |
