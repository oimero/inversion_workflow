# 08 物理约束神经网络反演

`piai_train.py` 是工作流的第八步。它读取地震体、第六步井控和第七步低频模型，联合学习阻抗修正与卷积核，输出反演网络、训练指标和井旁质量检查结果。

本文按时间域工区说明。每条地震道和对应低频模型组成一组输入，预测的对数阻抗等于低频模型与网络修正之和。第五步子波作为参考波形，用于确定采样设置和比较正演结果。

---

## 快速开始

```powershell
python scripts/piai_train.py --config "<inversion-config-yaml>"
python scripts/piai_train.py --config "<inversion-config-yaml>" --output-dir "<training-output-dir>"
```

运行前按本页配置参考填写工区配置、上游目录、模型变体和可信井名单，并显式传入配置文件。未指定输出目录时，脚本在配置的输出根目录下创建 `ginn_v3_piai_<timestamp>/`。

训练采用固定更新次数，每次更新同时使用井批次和未标注地震道批次。更新次数和设备可以从命令行覆盖：

```powershell
python scripts/piai_train.py --config "<inversion-config-yaml>" --updates <update-count> --device <device>
```

### 可选：体预测

```powershell
python scripts/piai_infer.py --config "<inversion-config-yaml>" --checkpoint "<training-output-dir>" --output-dir "<prediction-output-dir>"
```

检查点参数可以填写训练目录，也可以填写该目录中的 `selected_checkpoint.pt`。预测按批次读取地震道并写入磁盘数组。

使用局部区域检查输出时，可以指定中心区域边长，并保存数组结果：

```powershell
python scripts/piai_infer.py --config "<inversion-config-yaml>" --checkpoint "<selected-checkpoint-file>" --output-dir "<local-prediction-output-dir>" --smoke-tile-size <tile-size> --skip-segy-export
```

---

## 运行前需要什么

| 来源 | 文件或数据 | 用途 |
|------|------------|------|
| 工区配置 | 时间域地震体 | 网络输入及正演误差计算 |
| 第六步 | 井控运行目录、逐井 NPZ | 已标定滤波阻抗曲线、位置与有效样点 |
| 第六步 | `qc/evaluation_support.json` | 每口井固定的评价区间和样点索引 |
| 第七步 | 低频模型运行目录和选定三维变体的 `lfm.npz` | 初始对数阻抗、有效范围和空间网格 |
| 第五步 | `selected_wavelet.csv` | 参考子波的完整时间轴与振幅 |

井控、低频模型和地震体需要具有相同的时间采样轴，低频模型的线轴和形状需要与当前地震体一致。时间单位为秒，线号对应关系由工区几何提供。

当前训练井采用直井和固定平面位置。可信井名单中的井需要存在于井控集中，且第六步固定评价区间完整落在可用井曲线、地震和初模的共同支撑内。井位在地震道之间时，使用四邻道双线性权重采样。

参考子波需为有限值、奇数样点、零时刻居中的规则波形，其采样间隔需要与时间域地震体一致。子波时间轴表示相对时间，独立于地震体的绝对双程旅行时轴。

---

## 配置参考

配置位于 `ginn_v3_body_inversion` 段。公共工区配置继续提供地震文件、数据根目录和输出根目录。尖括号内容替换为本地路径、井名、模型变体或所需物理时长；其余数值展示软件默认设置。

```yaml
workflow_config: "<workflow-config-yaml>"

ginn_v3_body_inversion:
  inputs:
    lfm_run_dir: "<lfm-run-dir>"
    variant_id: "<lfm-variant-id>"
    well_control_run_dir: "<well-control-run-dir>"
    wavelet_generation_run_dir: "<wavelet-generation-run-dir>"

  trusted_well_names:
    - "<trusted-well-a>"
    - "<trusted-well-b>"

  network:
    wavelet_duration_s: "<learned-kernel-duration-s>"
    tcn_channels: [16, 16, 16]
    hidden_channels: 32
    kernel_size: 3
    dilation: 2
    gru_layers: 3
    dropout: 0.0

  training:
    updates: 1000
    labeled_batch_size: 6
    unlabeled_batch_size: 32
    learning_rate: 0.004
    weight_decay: 0.01
    validate_every: 100
    log_every: 20
    max_train_traces: 4096
    validation_traces: 128
    validation_gap_m: 300.0
    min_support_samples: 8
    device: cuda
    loss_weights:
      independent: 1.0
      physics: 1.0
      cross: 1.0

  inference:
    batch_size: 128
    min_support_samples: 8
```

### 上游目录与可信井

| 参数 | 含义 |
|------|------|
| `lfm_run_dir` | 第七步低频模型运行目录 |
| `variant_id` | 该运行中实际存在的变体名称 |
| `well_control_run_dir` | 第六步井控运行目录 |
| `wavelet_generation_run_dir` | 第五步运行目录，读取其中的参考子波 |
| `trusted_well_names` | 参与井约束训练的井名，必须非空且唯一 |

上游目录和变体需要显式填写，相对路径从仓库根目录解析。命令行的 `--lfm-run-dir`、`--variant-id`、`--well-control-run-dir` 和 `--wavelet-generation-run-dir` 可以覆盖对应配置。重复使用 `--trusted-well-name` 可以指定多口可信井。

### 网络与学习卷积核窗口

| 参数 | 含义 |
|------|------|
| `wavelet_duration_s` | 学习卷积核的总时间跨度，以秒表示 |
| `wavelet_samples` | 学习卷积核的样点数，需为不小于 3 的奇数 |
| `tcn_channels` | 各残差时序卷积块的通道数 |
| `hidden_channels` | 循环分支的隐藏宽度，需为正偶数 |
| `kernel_size` | 时序卷积核长度，需为正奇数 |
| `dilation` | 时序卷积的膨胀率 |
| `gru_layers` | 每个双向循环分支的层数 |
| `dropout` | 丢弃比例，取值在 0 与 1 之间 |

卷积核窗口二选一填写总时长或样点数。填写总时长时，脚本根据地震采样间隔向内取整为奇数样点；两者都留空时采用参考子波的总时长。窗口需至少覆盖两个采样间隔，并结合目标层段时长与需要解释的波形设置。

参考子波保持完整波形用于质量检查。学习卷积核由网络自由预测，其时间轴由窗口配置确定，物理振幅随训练更新。网络输入长度由地震采样轴确定。

### 训练与空间划分

| 参数 | 含义 |
|------|------|
| `updates` | 优化器更新次数 |
| `labeled_batch_size` | 每次更新使用的井样本道数 |
| `unlabeled_batch_size` | 每次更新使用的未标注地震道数 |
| `learning_rate` / `weight_decay` | 学习率与权重衰减 |
| `max_train_traces` | 参与训练的地震道数量上限 |
| `validation_traces` | 空间验证使用的地震道数量 |
| `validation_gap_m` | 训练道与验证道之间的平面距离间隔，单位为米 |
| `min_support_samples` | 可用道的最少有效样点数 |
| `validate_every` / `log_every` | 验证和日志输出的更新间隔 |
| `device` | 计算设备，例如 `cuda` 或 `cpu` |

空间划分使用真实平面距离。井目标沿用标定滤波曲线的分段重采样值，井曲线误差反映参与训练井的拟合效果，空间验证地震误差反映保留道的波形解释效果。

归一化量由训练数据一次计算：地震使用全局均值和标准差，低频模型使用对数阻抗统计量，井监督误差使用训练井的线性阻抗标准差。

### 联合损失

| 配置项 | 约束内容 |
|--------|----------|
| `independent` | 井预测阻抗与井目标的误差，以及井目标用井批平均卷积核正演后的地震误差 |
| `physics` | 井位置预测阻抗用井批平均卷积核正演后的地震误差 |
| `cross` | 井批与未标注批交换平均卷积核后的两侧正演误差 |

三项权重均需非负，且至少一项大于零。各项从第一次更新开始联合优化。默认选定权重为最后一次更新的模型；最佳验证权重作为单独文件保存。

### 推理配置

`inference.batch_size` 控制每批预测道数，`inference.min_support_samples` 控制可用道的最小支撑。批大小可通过 `--batch-size` 覆盖。预测以低频模型的有效范围为输出支撑，无效位置在阻抗数组中保持空值，并在掩码中标记为无效。

---

## 脚本在做什么

### 第一阶段：准备数据与网络

1. **建立共同采样空间。** 读取地震、低频模型和井控，按时间轴、线轴与井位几何建立单道样本。
2. **准备井目标与验证道。** 在第六步固定评价区间内，对原生滤波井曲线进行分段重采样；按平面距离选取训练道和验证道。
3. **确定尺度与窗口。** 汇总训练数据的归一化量，读取参考子波并建立学习卷积核的相对时间轴。

### 第二阶段：预测阻抗与卷积核

1. **提取道内特征。** 将地震和低频阻抗输入共享残差时序卷积网络，提取局部波形与纵向结构特征。
2. **预测两个输出。** 两个双向循环分支分别产生逐样点阻抗修正和整条道的卷积核。阻抗修正直接叠加到低频模型，得到对数阻抗预测。

### 第三阶段：联合更新

1. **计算井独立约束。** 比较井位置的预测线性阻抗与井目标，并用井批平均卷积核对井目标正演，比较其与井旁地震的差异。
2. **计算预测正演约束。** 将井位置的预测阻抗转为反射系数，与井批平均卷积核卷积，计算地震重建误差。
3. **交换卷积核进行交叉学习。** 井批平均卷积核用于未标注道预测的正演，未标注批平均卷积核用于井目标的正演。前一项中井批卷积核停止梯度，使该项主要更新未标注道的阻抗预测。
4. **共同反向更新。** 按配置权重合并三项损失，更新共享特征网络、阻抗分支和卷积核分支。

### 第四阶段：评价与输出

1. **定期评价。** 在固定验证道和训练井上计算地震重建误差与阻抗误差，记录各项损失和更新次数。
2. **保存权重与平均核。** 完成指定更新次数后保存最终模型，并在全部训练道上确定性计算平均学习卷积核。
3. **生成井旁质检。** 在固定评价区间内对比井曲线、初模和预测阻抗，分别计算参考子波与学习卷积核的正演结果。
4. **按需预测体。** 单独运行预测入口，逐批写出对数阻抗和有效掩码，对预测范围内的有效道汇总平均卷积核，再按配置导出线性阻抗体。

---

## 核心输出文件

### 训练运行

```text
ginn_v3_piai_<timestamp>/
├── training.log
├── input_contract.json
├── history.csv
├── selected_checkpoint.pt
├── last_checkpoint.pt
├── best_validation_checkpoint.pt
├── training_summary.json
└── well_qc/
    ├── learned_wavelet.csv
    ├── well_metrics.csv
    ├── metrics.json
    └── <well>/
        ├── curves.csv
        └── waveform_qc.png
```

| 文件 | 内容 |
|------|------|
| `input_contract.json` | 采样轴、网络与训练设置、归一化量、参考子波和学习卷积核时间轴 |
| `history.csv` | 逐次更新损失及定期验证指标 |
| `selected_checkpoint.pt` / `last_checkpoint.pt` | 最后一次更新的模型，用于后续推理 |
| `best_validation_checkpoint.pt` | 最佳验证地震误差对应的模型 |
| `training_summary.json` | 完成更新次数、最终与最佳验证指标、配置及权重路径 |
| `well_qc/learned_wavelet.csv` | 所有训练道平均学习卷积核，包含标准化系数和恢复物理尺度后的振幅 |
| `well_qc/well_metrics.csv` / `well_qc/metrics.json` | 每井训练或评价角色、阻抗误差、参考与学习卷积核的正演指标 |
| `well_qc/<well>/curves.csv` | 固定评价区间内的逐样点曲线与正演结果 |
| `well_qc/<well>/waveform_qc.png` | 井曲线、初模、预测及波形对比 |

### 体预测

```text
ginn_v3_piai_infer_<timestamp>/
├── piai_log_ai.npy
├── piai_log_ai_valid_mask.npy
├── learned_wavelet.csv
├── inference_summary.json
├── piai_ai.segy / piai_ai.zgy / piai_ai.npz
└── well_qc/
```

预测默认写入 `scripts/output/ginn_v3_piai_infer_<timestamp>/`，可以用 `--output-dir` 指定目录。

| 文件 | 内容 |
|------|------|
| `piai_log_ai.npy` | 磁盘映射的对数阻抗预测 |
| `piai_log_ai_valid_mask.npy` | 与预测同形状的有效样点掩码 |
| `learned_wavelet.csv` | 本次预测有效道的平均学习卷积核 |
| `inference_summary.json` | 权重来源、预测范围、有效道数、归一化量和输出路径 |
| `piai_ai.segy` / `piai_ai.zgy` / `piai_ai.npz` | 按源地震格式选择其中一种导出线性声阻抗；启用体导出时生成 |
| `well_qc/` | 使用本次平均学习卷积核生成的井旁曲线与波形质检 |

线性声阻抗由对数阻抗取指数得到，单位为 `m/s*g/cm3`。卷积核的物理振幅由训练时的全局地震尺度恢复，正演地震还需加回对应的全局均值。

---

## 如何阅读结果

### 第一步：看训练摘要与历史

确认完成更新次数、可信井、输入变体和配置与预期一致。选定模型对应最后一次更新，最佳验证模型的更新次数单独记录。结合三项训练损失与验证地震误差查看训练过程。

### 第二步：看井曲线与固定支撑

查看逐井曲线表和图件，对照标定滤波井曲线、低频模型与预测阻抗，并按表中记录的训练或评价角色阅读指标。图表区间来自第六步固定评价支撑。

### 第三步：比较两套正演结果

同时阅读参考子波和平均学习卷积核的正演指标。两套结果分别反映参考算子与训练所得算子对预测阻抗的波形解释效果。

### 第四步：看学习卷积核

检查卷积核的时间跨度、振幅和波形，并核对平均值所对应的数据范围。训练质检使用训练道平均核，体预测质检使用本次预测有效道的平均核。

### 第五步：需要时查看预测体

先查看有效掩码和实际预测范围，再查看阻抗结构与初模差异。空间验证地震指标与井曲线指标分别对应不同的数据范围，应结合各自用途阅读。

---

## 常见失败原因

| 问题 | 原因 | 处理方式 |
|------|------|----------|
| 上游输入或可信井缺失 | 必需目录、模型变体或井名单未填写 | 对照配置参考补全 |
| 第五步参考子波无法使用 | 波形不规则、未居中、样点非奇数或采样间隔不匹配 | 核对第五步产物及当前地震采样设置 |
| 卷积核窗口无效 | 同时设置时长与样点数，或时长过短、样点数为偶数 | 选择一种窗口设置并满足最小长度要求 |
| 地震、初模与井控轴不一致 | 上游产物来自不同采样空间 | 使用同一工区配置生成对应产物 |
| 低频模型维数不符合要求 | 选用的变体为二维剖面 | 使用与地震体网格对应的三维低频模型 |
| 井目标无法建立 | 井为斜井、平面位置变化或固定评价支撑不完整 | 使用满足当前直井要求的井控，核对评价支撑 |
| 可用训练或验证道不足 | 目标支撑、空间间隔或样本数量设置不满足要求 | 核对有效范围并调整采样数量或空间间隔 |
| 数据方差为零或预测非有限 | 归一化数据或正演支撑不可用 | 检查输入数据及有效曲线段 |
| 检查点与配置不匹配 | 网络长度、采样轴或学习卷积核窗口不同 | 使用对应训练配置与上游输入 |
| 输出目录非空 | 目录已有产物 | 指定新的运行目录 |
| 设备不可用 | 指定计算设备无法使用 | 选择本机可用设备 |
