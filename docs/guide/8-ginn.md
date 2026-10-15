# 08 物理约束神经网络反演

`piai_train.py` 是工作流的第八步。它读取地震体、第六步井控和第七步低频模型，联合学习阻抗修正与卷积核，输出反演网络、训练指标和井旁质量检查结果。第七步启用上下文扩展时，同时读取选定变体的扩展背景。

本文按时间域工区说明第八步。每条地震道和对应低频模型组成一组输入，预测的对数阻抗等于低频模型与网络修正之和。学习卷积核的采样方式和时间跨度由本步配置显式给出。启用第七步上下文扩展时，扩展背景用于网络输入和正演支撑，目标层模型用于预测波形损失、低频归一化参考和体输出。

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

检查点参数可以填写训练目录，也可以填写该目录中的 `selected_checkpoint.pt`。预测按批次读取地震道并写入磁盘数组。推理直接采用 checkpoint 中保存的学习卷积核时间轴；配置中的 `kernel` 用于训练时建立卷积核。预测使用与训练相同的第七步变体及其上下文设置。

使用局部区域检查输出时，可以指定中心区域边长，并保存数组结果：

```powershell
python scripts/piai_infer.py --config "<inversion-config-yaml>" --checkpoint "<selected-checkpoint-file>" --output-dir "<local-prediction-output-dir>" --smoke-tile-size <tile-size> --skip-segy-export
```

预测时可用 `--qc-inline <line>`、`--qc-xline <line>` 选择需要绘制的剖面；加上 `--qc-physical-closure` 可生成正演地震、真实地震与目的层残差对比图。图件写入预测目录的 `qc/`。

---

## 运行前需要什么

| 来源 | 文件或数据 | 用途 |
|------|------------|------|
| 工区配置 | 时间域地震体 | 网络输入及正演误差计算 |
| 第六步 | 井控运行目录、逐井 NPZ | 已标定监督曲线、位置与有效样点；监督曲线类型和主体平滑参数由第六步记录 |
| 第六步 | `qc/evaluation_support.json` | 每口井固定的评价区间和样点索引 |
| 第七步 | 低频模型运行目录和选定变体的 `lfm.npz` | 目标层初始对数阻抗、预测波形损失范围、低频归一化参考和体输出范围 |
| 第七步 | 启用上下文扩展时的 `context.npz` | 目标层外的低频背景、网络输入和正演支撑 |

井控、低频模型和地震体需要具有相同的时间采样轴，低频模型的线轴和形状需要与当前地震体一致。时间单位为秒，线号对应关系由工区几何提供。第六步记录的监督曲线类型、有效掩码和评价支撑随井控运行目录传入；短缺口修复值可用于井目标，长缺口保持分段。可信井用于第八步井监督，背景井只参与第七步低频模型；井位落在地震道之间时按四邻道双线性权重采样。

当前井监督使用固定平面位置的直井。

学习卷积核使用以零时刻为中心的规则时间轴。卷积核采样间隔可以不同于地震体的采样间隔；正演时按秒进行线性插值，并在相邻阻抗样点的中点界面上放置反射事件。

---

## 配置参考

配置位于 `ginn_v3_body_inversion` 段。公共工区配置继续提供地震文件、数据根目录和输出根目录。尖括号内容替换为本地路径、井名、模型变体或所需物理参数；其余数值展示一组可运行的配置形式。

```yaml
workflow_config: "<workflow-config-yaml>"

well_usage:
  trusted_well_names:
    - <trusted-well-a>
    - <trusted-well-b>
  background_only_well_names:
    - <background-well-a>

ginn_v3_body_inversion:
  kernel:
    dt_s: 0.002
    duration_s: 0.2

  inputs:
    lfm_run_dir: "<lfm-run-dir>"        # 留空自动发现匹配的第七步运行
    variant_id: "<lfm-variant-id>"
    well_control_run_dir: "<well-control-run-dir>" # 留空自动发现匹配的第六步运行

  # 可选：只从全局可信井中取训练子集
  trusted_well_names:
    - <training-subset-well>

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

### 上游目录与井监督

| 参数 | 含义 |
|------|------|
| `lfm_run_dir` | 第七步低频模型运行目录 |
| `variant_id` | 该运行中实际存在的变体名称 |
| `well_control_run_dir` | 第六步井控运行目录 |
| `well_usage.trusted_well_names` | 默认参与井约束训练的井名 |
| `well_usage.background_only_well_names` | 参与低频背景、但不进入井监督的井名 |
| `ginn_v3_body_inversion.trusted_well_names` | 可选的训练子集；填写后覆盖全局可信井名单 |

第八步沿用第六步记录的监督曲线。`ginn_v3_body_inversion.trusted_well_names` 可以从全局可信井中取训练子集，不能加入背景井。

低频模型目录和井控目录留空时，脚本自动发现与当前地震域匹配的最近成功运行；填写目录后固定使用该次产物。变体名称必须存在于该运行。命令行参数可以覆盖对应配置，训练子集必须是全局可信井的子集；全局名单为 `null` 时，未列入背景井的合格井默认作为可信井。

### 第七步模型范围

第七步启用上下文扩展时，选定变体目录同时包含两个模型范围：

| 范围 | 用途 |
|------|------|
| 目标层模型（`lfm.npz`） | 预测波形损失、低频归一化参考和体输出 |
| 扩展背景（`context.npz`） | 网络输入、预测阻抗正演和上下文反射响应 |

两者在目标层内部保持一致。网络在扩展背景上建立输入和正演支撑，预测波形损失只在目的层内且响应完整的位置计算。未启用上下文扩展时，目标层模型同时承担两种输入范围。

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

归一化量由训练数据一次计算：地震使用选定训练道有限样点的均值和标准差，低频背景使用目的层范围的对数阻抗统计量，井监督误差使用目的层内有效可信井样点的线性阻抗标准差。

### 联合损失

| 配置项 | 约束内容 |
|----------|----------|
| `independent` | 井位置预测阻抗与井目标的误差，以及已有井曲线用井批平均卷积核正演后的地震误差 |
| `physics` | 井位置预测阻抗用井批平均卷积核正演后的地震误差 |
| `cross` | 井批与未标注批交换平均卷积核后的两侧正演误差 |

三项权重均需非负，且至少一项大于零。默认选用最后一次更新的模型，另存验证地震误差与训练井阻抗误差之和最小的模型。

各项使用的范围如下：

| 约束或评价 | 使用范围 |
|------------|----------|
| 井阻抗监督 | 可信井全部可用曲线，可覆盖目的层外 |
| 真实井曲线的正演损失 | 目的层内有效井样点，区间端点允许截断响应 |
| 预测阻抗的正演损失 | 目的层内响应完整的样点 |
| 井旁质检 | 第六步固定的评价区间 |

长缺口分段处理，波形计算排除内部缺口附近受截断影响的样点。

### 推理配置

`inference.batch_size` 控制每批预测道数，`inference.min_support_samples` 控制可用道的最小支撑。批大小可通过 `--batch-size` 覆盖。预测以低频模型的有效范围为输出支撑，无效位置在阻抗数组中保持空值，并在掩码中标记为无效。

---

## 脚本在做什么

### 第一阶段：准备数据与网络

1. **建立共同采样空间。** 读取地震、低频模型、可选扩展背景和井控，按采样轴、线轴与井位几何建立单道样本。
2. **准备井目标与验证道。** 沿用第六步记录的监督曲线，在连续有效段内重采样到共同时间轴，并按平面距离选取训练道和验证道。
3. **确定尺度与卷积核轴。** 根据训练数据计算归一化量，按配置的时间间隔和窗口长度建立以零时刻为中心的学习卷积核时间轴。

### 第二阶段：预测阻抗与卷积核

1. **提取道内特征。** 将地震和低频阻抗输入共享残差时序卷积网络，提取局部波形与纵向结构特征。
2. **预测两个输出。** 两个双向循环分支分别产生逐样点阻抗修正和整条道的卷积核。阻抗修正直接叠加到低频模型，得到对数阻抗预测。

### 第三阶段：联合更新

1. **计算井独立约束。** 比较可信井位置的预测阻抗与选定井目标，并对连续有效段正演后计算井旁波形误差。
2. **计算预测正演约束。** 将预测阻抗转为反射系数，在相邻样点中点放置反射界面，按时间差插值学习卷积核并叠加界面响应，在目标范围内计算地震重建误差。
3. **交换卷积核进行交叉学习。** 井批平均卷积核用于未标注道预测的正演，未标注批平均卷积核用于井目标的正演。前一项中井批卷积核停止梯度，使该项主要更新未标注道的阻抗预测。
4. **共同反向更新。** 按配置权重合并三项损失，更新共享特征网络、阻抗分支和卷积核分支。

### 第四阶段：评价与输出

1. **定期评价。** 在验证道和训练井的目标范围内计算地震重建误差与阻抗误差，记录各项损失和更新次数。
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
        ├── curves.csv
        ├── waveform_qc.png
        └── waveform_qc.svg
```

| 文件 | 内容 |
|------|------|
| `input_contract.json` | 采样轴、卷积核设置、网络与训练设置、归一化量，以及第六步记录的监督曲线来源和平滑半高全宽 |
| `history.csv` | 逐次更新损失及定期验证指标 |
| `selected_checkpoint.pt` / `last_checkpoint.pt` | 最后一次更新的模型，用于后续推理 |
| `best_validation_checkpoint.pt` | 验证地震误差与训练井阻抗误差之和最小的模型 |
| `training_summary.json` | 完成更新次数、最终与最佳验证指标、监督曲线来源、平滑半高全宽、配置及权重路径 |
| `well_qc/learned_wavelet.csv` | 训练道平均学习卷积核，包含标准化系数和恢复物理尺度后的振幅 |
| `well_qc/well_metrics.csv` / `well_qc/metrics.json` | 每井角色、阻抗误差、`learned_global_correlation`、学习卷积核正演指标，以及 `support_samples` 和实际完整波形样点数 `waveform_support_samples` |
| `well_qc/<well>/curves.csv` | 固定评价区间内的逐样点曲线与学习卷积核正演结果 |
| `well_qc/<well>/waveform_qc.png` / `waveform_qc.svg` | 六联井旁质检图：阻抗、反射系数、合成地震、观测地震、残差与局部相关性 |

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
| `qc/` | 按指定线号生成的阻抗剖面，可选正演对比图及逐样点数据，提供 PNG 和 SVG |

线性声阻抗由对数阻抗取指数得到，单位为 `m/s*g/cm3`。卷积核的物理振幅由训练时的全局地震尺度恢复，正演地震还需加回对应的全局均值。

---

## 如何阅读结果

### 第一步：看训练摘要与历史

确认完成更新次数、可信井、输入变体和配置与预期一致。结合三项训练损失与验证地震误差查看训练过程，并按用途选择最后一次更新的模型或验证指标最优模型。

### 第二步：看井曲线与固定支撑

查看逐井曲线表和指标，对照第六步选定的监督曲线、低频模型与预测阻抗，并按训练或评价角色阅读指标。井旁质检使用第六步固定评价支撑，训练监督和波形指标只统计有效样点。

### 第三步：看学习卷积核

检查卷积核的时间跨度、采样间隔、振幅和波形，并核对平均值所对应的数据范围。

### 第四步：需要时查看预测体

先查看有效掩码和实际预测范围，再查看阻抗结构与初模差异。

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
