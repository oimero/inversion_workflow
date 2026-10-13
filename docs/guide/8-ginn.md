# 08 物理约束神经网络反演

`piai_train.py` 是工作流的第八步。它读取地震体、第六步井控和第七步低频模型，联合学习阻抗修正与卷积核，输出反演网络、训练指标和井旁质量检查结果。

本文说明时间域工区。每条地震道和对应低频模型组成一组输入，预测的对数阻抗等于低频模型与网络修正之和。学习卷积核的采样方式和时间跨度由本步配置显式给出。

---

## 快速开始

```powershell
python scripts/piai_train.py --config "<inversion-config-yaml>"
python scripts/piai_train.py --config "<inversion-config-yaml>" --output-dir "<training-output-dir>"
```

运行前按本页配置参考填写工区配置、上游目录、模型变体和可信井名单，并显式传入配置文件。未指定输出目录时，脚本在配置的输出根目录下创建带时间标记的训练目录。

训练采用固定更新次数，每次更新同时使用井批次和未标注地震道批次。更新次数和设备可以从命令行覆盖：

```powershell
python scripts/piai_train.py --config "<inversion-config-yaml>" --updates <update-count> --device <device>
```

### 可选：体预测

```powershell
python scripts/piai_infer.py --config "<inversion-config-yaml>" --checkpoint "<training-output-dir>" --output-dir "<prediction-output-dir>"
```

检查点参数可以填写训练目录，也可以填写该目录中的 `selected_checkpoint.pt`。预测按批次读取地震道并写入磁盘数组。推理直接采用 checkpoint 中保存的学习卷积核时间轴；配置中的 `kernel` 用于训练时建立卷积核。

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
| 第七步 | 低频模型运行目录和选定变体的 `lfm.npz` | 初始对数阻抗、有效范围和空间网格 |

井控、低频模型和地震体需要具有相同的时间采样轴，低频模型的线轴和形状需要与当前地震体一致。时间单位为秒，线号对应关系由工区几何提供。

当前训练井采用直井和固定平面位置。可信井名单中的井需要存在于井控集中，且第六步固定评价区间完整落在可用井曲线、地震和初模的共同支撑内。井位在地震道之间时，使用四邻道双线性权重采样。

学习卷积核使用以零时刻为中心的规则时间轴。卷积核采样间隔可以不同于地震体的采样间隔；正演时按秒进行线性插值，并在相邻阻抗样点的中点界面上放置反射事件。

---

## 配置参考

配置位于 `ginn_v3_body_inversion` 段。公共工区配置继续提供地震文件、数据根目录和输出根目录。尖括号内容替换为本地路径、井名、模型变体或所需物理参数；其余数值展示一组可运行的配置形式。

```yaml
workflow_config: "<workflow-config-yaml>"

ginn_v3_body_inversion:
  kernel:
    dt_s: 0.002
    duration_s: 0.2

  inputs:
    lfm_run_dir: "<lfm-run-dir>"
    variant_id: "<lfm-variant-id>"
    well_control_run_dir: "<well-control-run-dir>"

  trusted_well_names:
    - "<trusted-well-a>"
    - "<trusted-well-b>"

  network:
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

如果按样点数指定卷积核窗口，将 `kernel` 改为以下形式；`duration_s` 和 `samples` 只能填写其中一个：

```yaml
kernel:
  dt_s: <kernel-sampling-interval-s>
  samples: <odd-kernel-sample-count>
```

### 上游目录与可信井

| 参数 | 含义 |
|------|------|
| `lfm_run_dir` | 第七步低频模型运行目录 |
| `variant_id` | 该运行中实际存在的变体名称 |
| `well_control_run_dir` | 第六步井控运行目录 |
| `trusted_well_names` | 参与井约束训练的井名，必须非空且唯一 |

上游目录和变体需要显式填写，相对路径从仓库根目录解析。命令行的 `--lfm-run-dir`、`--variant-id` 和 `--well-control-run-dir` 可以覆盖对应配置。重复使用 `--trusted-well-name` 可以指定多口可信井。

### 学习卷积核窗口

| 参数 | 含义 |
|------|------|
| `kernel.dt_s` | 学习卷积核的采样间隔，单位为秒 |
| `kernel.duration_s` | 学习卷积核的总时间跨度，和 `kernel.samples` 二选一 |
| `kernel.samples` | 学习卷积核的样点数，需为不小于 3 的奇数，和 `kernel.duration_s` 二选一 |

`duration_s` 按 `dt_s` 向内确定居中的奇数样点；窗口至少覆盖两个采样间隔。学习卷积核只由本步网络训练得到，正演使用其实际时间轴。

### 网络结构

| 参数 | 含义 |
|------|------|
| `tcn_channels` | 各残差时序卷积块的通道数 |
| `hidden_channels` | 循环分支的隐藏宽度，需为正偶数 |
| `kernel_size` | 时序卷积核长度，需为正奇数 |
| `dilation` | 时序卷积的膨胀率 |
| `gru_layers` | 每个双向循环分支的层数 |
| `dropout` | 丢弃比例，取值在 0 与 1 之间 |

网络结构只描述特征提取和输出分支。地震采样长度与学习卷积核长度由运行数据和 `kernel` 配置确定。

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
|----------|----------|
| `independent` | 井位置预测阻抗与井目标的误差，以及井目标用井批平均卷积核正演后的地震误差 |
| `physics` | 井位置预测阻抗用井批平均卷积核正演后的地震误差 |
| `cross` | 井批与未标注批交换平均卷积核后的两侧正演误差 |

三项权重均需非负，且至少一项大于零。各项从第一次更新开始联合优化。默认选定权重为最后一次更新的模型；验证地震误差与训练井阻抗误差之和最小的模型作为单独文件保存。

### 推理配置

`inference.batch_size` 控制每批预测道数，`inference.min_support_samples` 控制可用道的最小支撑。批大小可通过 `--batch-size` 覆盖。预测以低频模型的有效范围为输出支撑，无效位置在阻抗数组中保持空值，并在掩码中标记为无效。

---

## 脚本在做什么

### 第一阶段：准备数据与网络

1. **建立共同采样空间。** 读取地震、低频模型和井控，按时间轴、线轴与井位几何建立单道样本。
2. **准备井目标与验证道。** 将原生滤波井曲线按有效段重采样到共同时间轴，并读取第六步固定评价区间；按平面距离选取训练道和验证道。
3. **确定尺度与卷积核轴。** 汇总训练数据的归一化量，按配置的时间间隔和窗口长度建立以零时刻为中心的学习卷积核时间轴。

### 第二阶段：预测阻抗与卷积核

1. **提取道内特征。** 将地震和低频阻抗输入共享残差时序卷积网络，提取局部波形与纵向结构特征。
2. **预测两个输出。** 两个双向循环分支分别产生逐样点阻抗修正和整条道的卷积核。阻抗修正直接叠加到低频模型，得到对数阻抗预测。

### 第三阶段：联合更新

1. **计算井独立约束。** 比较井位置的预测线性阻抗与井目标，并用井批平均卷积核对井目标正演，比较其与井旁地震的差异。
2. **计算预测正演约束。** 将井位置的预测阻抗转为反射系数，在相邻样点中点放置反射界面，按时间差对井批平均卷积核插值并叠加界面响应，计算地震重建误差。
3. **交换卷积核进行交叉学习。** 井批平均卷积核用于未标注道预测的正演，未标注批平均卷积核用于井目标的正演。前一项中井批卷积核停止梯度，使该项主要更新未标注道的阻抗预测。
4. **共同反向更新。** 按配置权重合并三项损失，更新共享特征网络、阻抗分支和卷积核分支。

### 第四阶段：评价与输出

1. **定期评价。** 在固定验证道和训练井上计算地震重建误差与阻抗误差，记录各项损失和更新次数。
2. **保存权重与平均核。** 完成指定更新次数后保存最终模型，并在全部训练道上确定性计算平均学习卷积核。
3. **生成井旁质检。** 在固定评价区间内对比井曲线、初模和预测阻抗，计算学习卷积核正演结果的波形指标。
4. **按需预测体。** 单独运行预测入口，逐批写出对数阻抗和有效掩码，对预测范围内的有效道汇总平均学习卷积核，再按配置导出线性阻抗体。

---

## 核心输出文件

### 训练运行

```text
<training-output-dir>/
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
        └── curves.csv
```

| 文件 | 内容 |
|------|------|
| `input_contract.json` | 采样轴、卷积核设置、网络与训练设置、归一化量 |
| `history.csv` | 逐次更新损失及定期验证指标 |
| `selected_checkpoint.pt` / `last_checkpoint.pt` | 最后一次更新的模型，用于后续推理 |
| `best_validation_checkpoint.pt` | 验证地震误差与训练井阻抗误差之和最小的模型 |
| `training_summary.json` | 完成更新次数、最终与最佳验证指标、配置及权重路径 |
| `well_qc/learned_wavelet.csv` | 训练道平均学习卷积核，包含标准化系数和恢复物理尺度后的振幅 |
| `well_qc/well_metrics.csv` / `well_qc/metrics.json` | 每井角色、阻抗误差、`learned_global_correlation` 和学习卷积核正演指标 |
| `well_qc/<well>/curves.csv` | 固定评价区间内的逐样点曲线与学习卷积核正演结果 |

检查点保存模型结构、归一化量、学习卷积核时间轴及中点界面对齐方式。推理直接使用检查点中的核时间轴。

### 体预测

```text
<prediction-output-dir>/
├── piai_log_ai.npy
├── piai_log_ai_valid_mask.npy
├── learned_wavelet.csv
├── inference_summary.json
├── piai_ai.segy / piai_ai.zgy / piai_ai.npz
└── well_qc/
```

预测默认写入配置指定的输出根目录，可以用 `--output-dir` 指定目录。

| 文件 | 内容 |
|------|------|
| `piai_log_ai.npy` | 磁盘映射的对数阻抗预测 |
| `piai_log_ai_valid_mask.npy` | 与预测同形状的有效样点掩码 |
| `learned_wavelet.csv` | 本次预测有效道的平均学习卷积核 |
| `inference_summary.json` | 权重来源、预测范围、有效道数、归一化量和输出路径 |
| `piai_ai.segy` / `piai_ai.zgy` / `piai_ai.npz` | 按源地震格式选择其中一种导出线性声阻抗；启用体导出时生成 |
| `well_qc/` | 使用本次平均学习卷积核生成的井旁曲线与波形指标质检 |

线性声阻抗由对数阻抗取指数得到，单位为 `m/s*g/cm3`。卷积核的物理振幅由训练时的全局地震尺度恢复，正演地震还需加回对应的全局均值。

---

## 如何阅读结果

### 第一步：看训练摘要与历史

确认完成更新次数、可信井、输入变体和配置与预期一致。选定模型对应最后一次更新，最佳验证模型的更新次数单独记录。结合三项训练损失与验证地震误差查看训练过程。

### 第二步：看井曲线与固定支撑

查看逐井曲线表和指标，对照标定滤波井曲线、低频模型与预测阻抗，并按表中记录的训练或评价角色阅读指标。曲线区间来自第六步固定评价支撑。

### 第三步：看学习卷积核

检查卷积核的时间跨度、采样间隔、振幅和波形，并核对平均值所对应的数据范围。训练质检使用训练道平均核，体预测质检使用本次预测有效道的平均核。

### 第四步：需要时查看预测体

先查看有效掩码和实际预测范围，再查看阻抗结构与初模差异。空间验证地震指标与井曲线指标分别对应不同的数据范围，应结合各自用途阅读。

---

## 常见失败原因

| 问题 | 原因 | 处理方式 |
|------|------|----------|
| 上游输入或可信井缺失 | 必需目录、模型变体或井名单未填写 | 对照配置参考补全 |
| 卷积核窗口无效 | `duration_s` 与 `samples` 同时设置，或时长过短、样点数为偶数 | 选择一种窗口设置并满足最小长度要求 |
| 地震、初模与井控轴不一致 | 上游产物来自不同采样空间 | 使用同一工区配置生成对应产物 |
| 低频模型维数不符合要求 | 选用的变体为二维剖面 | 使用与地震体网格对应的三维低频模型 |
| 井目标无法建立 | 井为斜井、平面位置变化或固定评价支撑不完整 | 使用满足当前直井要求的井控，核对评价支撑 |
| 可用训练或验证道不足 | 目标支撑、空间间隔或样本数量设置不满足要求 | 核对有效范围并调整采样数量或空间间隔 |
| 数据方差为零或预测非有限 | 归一化数据或正演支撑不可用 | 检查输入数据及有效曲线段 |
| 检查点与输入轴不匹配 | checkpoint 记录的地震采样轴与当前输入不同 | 使用与 checkpoint 对应的上游输入 |
| 输出目录非空 | 目录已有产物 | 指定新的运行目录 |
| 设备不可用 | 指定计算设备无法使用 | 选择本机可用设备 |
