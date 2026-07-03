# 模型消融训练与评估

`ginn_v2.py` 是工作流的研究旁路。它做一件事：**在冻结的合成基准上训练并评估轻量反演模型，通过受控消融实验揭示不同架构选择的实际增益。** 训练完成后，自动在合成基准的三个验证范围上做推理评估，最终汇总成一份消融报告卡，为后续真实工区的模型选择提供定量依据。

---

## 快速开始

```bash
# 方式一（推荐）：PowerShell 一键运行
cd experiments/ginn_v2
.\train_network.ps1 <实验名称>                    # 训练 + 三组预测 + 三组报告
.\run_synthetic_gate.ps1 <闸门名> <实验名1> ...     # 汇总 + 盖章

# 方式二：手动分步运行
# 第一步：训练
python scripts/ginn_v2.py train --config experiments/ginn_v2/train.yaml

# 第二步：预测（三组）
python scripts/ginn_v2.py predict --model-run-dir <训练输出目录> \
  --sample-kind base --split validation
python scripts/ginn_v2.py predict --model-run-dir <训练输出目录> \
  --sample-kind base --split test
python scripts/ginn_v2.py predict --model-run-dir <训练输出目录> \
  --sample-kind base frequency_probe seismic_variant --split validation

# 第三步：报告（三组，每组预测各跑一次）
python scripts/ginn_v2.py report --prediction-dir <预测输出目录>

# 第四步：汇总
python scripts/ginn_v2.py summarize \
  --report <模型角色>:<范围>:<报告目录> \
  --report ...

# 第五步：盖章
python scripts/ginn_v2.py stamp-gate \
  --model-run-dir <训练输出目录> \
  --synthetic-gate-report-dir <闸门目录> \
  --synthetic-gate-report-card <闸门目录>/ablation_report_card.json
```

`train_network.ps1` 封装了第一到第三步，`run_synthetic_gate.ps1` 封装了第四到第五步。日常迭代用 runner，调试或自定义评估范围时用手动命令。

---

## 运行前需要什么

| 输入 | 来源 | 用途 |
|------|------|------|
| `synthetic_benchmark.h5` | 合成基准生成 | 冻结的训练数据和真值 |
| `sample_index.csv` | 合成基准生成 | 样本切分索引 |
| `benchmark_manifest.json` | 合成基准生成 | 数据完整性校验 |
| `experiments/ginn_v2/train.yaml` | 用户配置 | 模型架构、训练超参 |

`train.yaml` 中 `benchmark_dir` 设为 `auto` 时，脚本自动从合成基准的实验结果目录中找最新合格的场条件基准。

---

## 配置参考

```yaml
# 配置文件：experiments/ginn_v2/train.yaml
train:
  sources:
    # auto 自动选择最新的场条件合成基准
    benchmark_dir: auto
    # 留空则每次运行确定性派生 patch_index 和 normalization
    patch_index:
    normalization:

  model:
    model_id: trace_1d_dilated_tcn_mismatch_training
    model_role: no_lateral
    hidden_channels: 32
    depth: 5

  patching:
    patch_lateral: 32
    patch_twt: 128
    lateral_stride: 16
    twt_stride: 64
    min_valid_fraction: 0.50

  split:
    split_policy: derive
    validation_fraction: 0.15
    test_fraction: 0.15

  optimization:
    epochs: 5
    batch_size: 8
    learning_rate: 0.001
    seed: 20260617

  losses:
    lambda_physics: 0.0
    # 0.1 = PH5 holdout anchor 实验；设为 0.0 则为 QC-enabled control
    lambda_real_delta: 0.1

  # anchor 和 QC-enabled control 均应保留此段
  real_delta:
    lfm_run_dir: scripts/output/real_field_lfm_<timestamp>
    variant_id: <descriptive-variant-id>
    well_control_run_dir: scripts/output/real_field_well_controls_<timestamp>
    held_out_well: PH5
    exclude_same_cluster: false
    clusters_per_step: 4
    cluster_radius_m: 600.0
    diagnostic_max_hz: 80.0
    reconstruction_tolerance_log_ai: 1.0e-5
    seismic_value_transform: p99_abs_matched
    lfm_value_transform: identity

  runtime:
    device: auto
    log_interval_batches: 10

  smoke_test:
    # 留空即不限制。设为整数则只在对应数量的块上训练
    max_patches:
    overfit_tiny: false

  synthetic_gate:
    # 训练阶段通常留空；在 stamp-gate 子命令中填充
    synthetic_gate_report_dir:
    synthetic_gate_report_card:
    synthetic_gate_frozen_candidate: false
```

配置按参数块分组：`sources`、`model`、`patching`、`split`、`optimization`、`losses`、`real_delta`、`runtime`、`smoke_test`、`synthetic_gate`。解析器会拒绝未知分组和未知字段，避免拼错参数后静默使用默认值。上例已经是 PH5 holdout 的 real-delta anchor 实验；control 只需把 `lambda_real_delta` 改为 `0.0` 并保留 `real_delta` 段即可。

---

### `train.sources`

控制训练数据的来源和已派生产物的复用。

```yaml
  sources:
    benchmark_dir: auto
    patch_index:
    normalization:
```

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `benchmark_dir` | 路径或 `auto` | 必填 | 合成基准目录。设为 `auto` 时从 `experiments/synthoseis_lite/results/` 下搜索含 `generate_field_conditioned` 子目录且 `synthetic_benchmark.h5`、`sample_index.csv`、`benchmark_manifest.json` 均存在的最新结果 |
| `patch_index` | 路径或空 | 空（自动派生） | 预先生成的块索引 CSV。留空时脚本根据 patching 和 split 参数从基准确定性构建 |
| `normalization` | 路径或空 | 空（自动计算） | 预先生成的归一化 JSON。留空时脚本在训练集上计算均值/标准差 |

三个参数均为可选（除 `benchmark_dir` 必填外），但有以下约束：

- `patch_index` 和 `normalization` 留空时，每次运行基于同一 benchmark、切块配置和 seed 独立地确定性生成。这是 control 与 anchor 复现性锁定的基础——不需要把 anchor 绑定到 control 的文件。
- 显式指定 `patch_index` 时，脚本会校验其 `sample_kind` 列是否匹配当前 `model_id` 的允许训练样本类型。不能与 `--overfit-tiny` 或 `--max-patches` 命令行参数同时使用。
- `benchmark_dir` 用于定位；benchmark 的 `contract_fingerprint_sha256` 写入模型 manifest 的 `input_contracts`。训练端不重算三个 benchmark 文件的 SHA。

---

### `train.model`

控制模型架构、容量和消融角色标签。

```yaml
  model:
    model_id: trace_1d_dilated_tcn_mismatch_training
    model_role: no_lateral
    hidden_channels: 32
    depth: 5
```

#### `model_id`

必填，共 10 个注册架构，按族分组：

| 族 | 架构 ID | 训练样本 | 横向感受野 |
|------|------|------|------|
| 一维基础 | `trace_1d` | 仅 base | 1 |
| | `trace_1d_mismatch_training` | base + seismic_variant | 1 |
| 一维膨胀 | `trace_1d_dilated_tcn` | 仅 base | 1 |
| | `trace_1d_dilated_tcn_mismatch_training` | base + seismic_variant | 1 |
| 横向混合 | `trace_1d_tcn_lateral_mixer_k1_mismatch_training` | base + seismic_variant | 1 |
| | `trace_1d_tcn_lateral_mixer_mismatch_training` | base + seismic_variant | 3 |
| | `trace_1d_tcn_lateral_mixer_k5_mismatch_training` | base + seismic_variant | 5 |
| 二维 | `patch_2d_supervised` | 仅 base | `1 + 2*depth` |
| | `patch_2d_mismatch_training` | base + seismic_variant | `1 + 2*depth` |
| | `patch_2d_with_physics_loss` | 仅 base | `1 + 2*depth` |

各族的内部实现：

- **`Trace1DNet`**（`trace_1d` 系列）：五层 Conv1d（kernel_size=5, padding=2），逐道独立处理。时间感受野 = `1 + 4 * depth`。
- **`Trace1DDilatedTCN`**（`trace_1d_dilated_tcn` 系列）：首层 1×1 conv + `depth-1` 层膨胀 Conv1d（kernel_size=5, dilation=2^block, padding=2·dilation） + 输出 1×1 conv。时间感受野 = `1 + 4 * (2^(depth-1) - 1)`。
- **`Trace1DTCNShallowLateralMixer`**（`lateral_mixer` 系列）：在上述膨胀 TCN 的编码特征之上加一层 Conv2d 横向混合（kernel_size=(lateral_kernel, 1)），残差连接后输出。横向感受野 = `lateral_kernel`。
- **`Patch2DNet`**（`patch_2d` 系列）：`depth` 层 Conv2d（kernel_size=3, padding=1），横向和时间感受野均为 `1 + 2*depth`。

`model_id` 同时决定了可用的训练样本类型：

- 含 `mismatch_training` 后缀的 ID 使用 `{"base", "seismic_variant"}` 两类样本，WeightedRandomSampler 按样本类别均匀采样。
- 不含此后缀的 ID 只用 `{"base"}`。

real-delta 是附加损失，不引入新的 `model_id`。control 和 anchor 应使用相同的 `model_id`。

#### `model_role`

模型在消融实验中的角色标签，决定了后续真实工区预测时结果存放的子目录名。默认从 `model_id` 自动推断：

| `model_id` 特征 | 推断角色 |
|------|------|
| 含 `lateral_mixer` | `lateral` |
| 含 `trace_1d` 或 `trace1d` | `no_lateral` |
| 其他 | `model_id` 本身（`-` 替换为 `_`） |

可在 YAML 中显式填写来覆盖自动推断。首版非零 `lambda_real_delta` 只在 `model_role` 为 `no_lateral` 且 `receptive_field_lateral == 1` 时通过校验。

#### `hidden_channels`

网络中所有隐藏层的通道数。默认 32。增大可提升模型容量但也增加过拟合风险和显存占用。

#### `depth`

网络深度（总层数）。默认 5，必须 ≥ 2。一维膨胀架构的实际感受野随 depth 指数增长（`1 + 4 * (2^(depth-1) - 1)`），depth=5 时约 125 个时间采样点。

---

### `train.patching`

控制合成基准剖面的滑动窗口切块参数。

```yaml
  patching:
    patch_lateral: 32
    patch_twt: 128
    lateral_stride: 16
    twt_stride: 64
    min_valid_fraction: 0.50
```

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `patch_lateral` | 正整数 | 32 | 每个块的横向道数 |
| `patch_twt` | 正整数 | 128 | 每个块的时间采样点数 |
| `lateral_stride` | 正整数 | 16 | 横向滑动步长（道数） |
| `twt_stride` | 正整数 | 64 | 时间方向滑动步长（采样点数） |
| `min_valid_fraction` | (0, 1] | 0.50 | 块内有效像素比例下限，低于此值的块被丢弃 |

切块逻辑：对每个基准样本的有效区域，横向从 0 到 `n_lateral - patch_lateral`、时间方向从 0 到 `n_twt - patch_twt`，以指定 stride 生成窗口起始位置。最后一个窗口会对齐到边界（覆盖到末尾）。步长越小块数越多——训练更慢但覆盖更密集。

`min_valid_fraction` 过滤掉目标窗口内有效像素（`valid_mask_model` 为 True 且 target/seismic/lfm 均有限）比例不足的块。如果训练块数过少（< 100），检查是否设得过高。

这些参数同时影响 real-delta 的井轨迹推理——`DifferentiableWellPredictor` 用相同的切块规格在真实工区体积上构建支撑块。

---

### `train.split`

控制基准样本到训练/校验/测试集的划分。

```yaml
  split:
    split_policy: derive
    validation_fraction: 0.15
    test_fraction: 0.15
```

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `split_policy` | `derive` 或 `strict` | `derive` | 划分策略 |
| `validation_fraction` | (0, 1) | 0.15 | 校验集比例 |
| `test_fraction` | (0, 1) | 0.15 | 测试集比例 |

**`split_policy=derive`**：基准样本索引中的 `split` 列被忽略。对每个父实现（`parent_realization_id`）做 SHA-256 哈希，取前 8 字节转为 [0, 1) 均匀值。该值 < `test_fraction` 的划为 test，< `test_fraction + validation_fraction` 的划为 validation，其余为 train。关键特性：同一剖面的所有块继承相同的 split，因此训练集和校验集中不会出现同一剖面的不同块——避免了空间泄漏。

**`split_policy=strict`**：直接使用样本索引中的 `split` 列。如果某样本已有非 `train`/`validation`/`test`/`benchmark` 的 split 值，抛出错误。

---

### `train.optimization`

控制训练循环的超参数。

```yaml
  optimization:
    epochs: 5
    batch_size: 8
    learning_rate: 0.001
    seed: 20260617
```

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `epochs` | 正整数 | 5 | 完整遍历训练集的次数 |
| `batch_size` | 正整数 | 8 | 每批次的块数 |
| `learning_rate` | 正浮点数 | 0.001 | AdamW 优化器的学习率 |
| `seed` | 整数 | 20260617 | 全局随机种子 |

种子同时控制：

1. PyTorch 全局种子（`torch.manual_seed` / `torch.cuda.manual_seed_all`）
2. `WeightedRandomSampler` 的生成器种子
3. real-delta sampler 的簇/井洗牌种子（派生为 `seed + 1_000_003`）

所有随机流均从该种子确定性派生，保证相同配置下的完全复现。

训练过程中的核心逻辑：

- 每个 batch：前向 → 计算 loss（synthetic + physics + real_delta）→ 反向 → AdamW step
- 每个 epoch 后：在 validation 集上计算 pure synthetic MSE，选择最低的 epoch 保存 `checkpoint_best.pt`
- 最终 epoch 后：额外保存 `checkpoint_final.pt`
- 所有 loss 分量均写入 `training_history.csv`

---

### `train.losses`

控制多任务损失的各分量权重。

```yaml
  losses:
    lambda_physics: 0.0
    lambda_real_delta: 0.1
```

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `lambda_physics` | ≥0 浮点数 | 0.0 | 物理正演损失权重 |
| `lambda_real_delta` | ≥0 浮点数 | 0.0 | 真实井 delta 监督损失权重 |

总损失公式：

```
total_loss = synthetic_mse
           + lambda_physics * physics_mse
           + lambda_real_delta * real_delta_mse
```

**`lambda_physics`**：非零时在 base 样本上施加 Robinson 正演一致性损失。对模型预测的 log AI 做反射系数正演，与观测地震比较 MSE。目前只支持 `patch_2d_with_physics_loss` 和 `trace_1d_dilated_tcn_mismatch_training` 两个 model_id。

**`lambda_real_delta`**：非零时在每个 training step 中额外采样显式 variant 的 `well_log_ai - lfm_log_ai` normalized-delta MSE。权重大于 0 时必须配置 `train.real_delta` 段，且当前只支持 `no_lateral` 角色。设为 0.0 但保留 `real_delta` 段时，不参与训练但仍生成 best/final 全井 QC（QC-enabled control）。

校验 loss（用于选择 best checkpoint）始终只用 pure synthetic validation MSE，不受 `lambda_physics` 或 `lambda_real_delta` 影响。

---

### `train.real_delta`

控制真实工区井数据的稀疏 delta 监督。

```yaml
  real_delta:
    lfm_run_dir: scripts/output/real_field_lfm_<timestamp>
    variant_id: <descriptive-variant-id>
    well_control_run_dir: scripts/output/real_field_well_controls_<timestamp>
    held_out_well: PH5
    exclude_same_cluster: false
    clusters_per_step: 4
    cluster_radius_m: 600.0
    diagnostic_max_hz: 80.0
    reconstruction_tolerance_log_ai: 1.0e-5
    seismic_value_transform: p99_abs_matched
    lfm_value_transform: identity
```

#### `lfm_run_dir` / `variant_id` / `well_control_run_dir`

三者必须显式提供，不接受 `auto`。解析时依次校验成功的 `real_field_lfm_run_v3`、`variant_manifest.csv`、所选 `real_field_lfm_variant_v3` 主 NPZ，以及 Step 6 的显式 schema/轴/shape/mask 语义。所选 variant 和 WellControlSet 的直接契约指纹写入模型 manifest 的 `input_contracts`。

#### 井监督分配

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `held_out_well` | 井名（字符串） | 必填 | 始终不参与监督的 holdout 井 |
| `exclude_same_cluster` | 布尔 | `false` | 是否同时排除与 holdout 井同空间簇的所有井 |
| `clusters_per_step` | 正整数 | 必填 | 每个 training step 随机选择多少个空间簇 |
| `cluster_radius_m` | 正浮点数 | 必填 | 空间聚类的半径（米），用于将井分组到簇 |

井的监督角色分三类：

| 角色 | 含义 |
|------|------|
| `training` | 参与 real-delta 损失计算 |
| `held_out` | 配置指定的 holdout 井，既不参与训练也不参与同簇排除逻辑 |
| `same_cluster_excluded` | 与 holdout 井同簇、被 `exclude_same_cluster` 排除的井 |

当 `lambda_real_delta > 0` 且 `require_training=True` 时，training 井数必须 ≥ 1，否则运行失败。

#### 均衡采样器

`BalancedRealDeltaSampler` 负责每个 step 选择哪些井参与 real-delta loss：

1. 从所有空间簇中随机不重复地选择 `clusters_per_step` 个簇
2. 在每个选中的簇内，随机不重复地选择一口井
3. 簇内所有井被均匀轮换后才重新洗牌
4. 所有随机流从 `seed + 1_000_003` 确定性派生

每个选中井的 `well_log_ai - lfm_log_ai`（来自 Step 6 canonical 控制和所选 Step 7 variant 三线性插值）作为 normalized-delta target。每口井的选择次数记录在 `real_delta_sampling_qc.csv` 中；variant ID、LFM variant 契约指纹和 WellControlSet 契约指纹写入样点契约。

#### 可微井预测器

`DifferentiableWellPredictor` 在不含横向混合的模型（`receptive_field_lateral == 1`）上使用 sparse 模式：只对井轨迹穿过的 inline/xline 节点做逐道逐块推理，再通过三线性插值得到井轨迹上的预测值。

训练开始前，predictor 先做 reconstruction 校验：在初始化模型上比较 sparse 模式和 full-patch 模式（在整个块上做二维推理）的预测差异。最大绝对差异必须 ≤ `reconstruction_tolerance_log_ai`（默认 1e-5），确保单道推理与完整块推理等价。校验失败抛出 `real_field_well_reconstruction_mismatch`。

如果模型有横向混合（`receptive_field_lateral > 1`），predictor 切换到 `canonical_full_patch` 模式（在整个块上做完整二维推理），reconstruction 校验被跳过。

#### 值域变换

| 参数 | 默认 | 含义 |
|------|------|------|
| `seismic_value_transform` | `p99_abs_matched` | 真实地震数据的值域变换方式。非 `identity` 时需要 `input_reference_stats.json` |
| `lfm_value_transform` | `identity` | 真实 LFM 的值域变换方式 |

这些变换在加载真实工区地震体和 LFM 体时应用，确保输入分布与训练时的合成数据一致。`input_reference_stats.json` 由训练阶段自动生成，记录训练集上地震数据的 P01/P05/P50/P95/P99/abs_p95/abs_p99 等参考统计量。

#### 质量控制

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `diagnostic_max_hz` | 正浮点数 | 80.0 | 井 QC 频带分析的最高频率（Hz）。实际使用 `min(diagnostic_max_hz, 0.45 * Nyquist)` |
| `reconstruction_tolerance_log_ai` | 正浮点数 | 1e-5 | sparse vs full-patch 重建误差的容忍上限（log AI 单位） |

---

### `train.runtime`

控制计算设备和日志频率。

```yaml
  runtime:
    device: auto
    log_interval_batches: 10
```

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `device` | `auto` / `cuda` / `cpu` | `auto` | 计算设备。`auto` 在有 CUDA 时使用 CUDA，否则回退 CPU。显式 `cuda` 在 CUDA 不可用时直接失败 |
| `log_interval_batches` | 正整数 | 10 | 每隔多少个 batch 输出一行训练日志（含各 loss 分量、耗时、ETA） |

日志输出到 stdout 和 `training.log` 文件。日志格式：
```
epoch=1/5 batch=10/240 synthetic=0.0423 physics=0 real_delta=0.0156 ...
batch_s=0.12 ema_batch_s=0.11 epoch_eta_s=25.3 run_eta_s=126.5 ...
```

---

### `train.smoke_test`

用于快速冒烟测试，验证代码和配置是否正确，而非正式训练。

```yaml
  smoke_test:
    max_patches:
    overfit_tiny: false
```

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `max_patches` | 正整数或空 | 空（不限制） | 限制构建的块总数。达到此数量后停止切块。不能与 probe 类样本同时使用 |
| `overfit_tiny` | 布尔 | `false` | 极简过拟合测试。只使用 4 个块训练，同时复制为 validation 集 |

`overfit_tiny` 的行为：
1. 取前 4 个块（如不足 4 个则取全部）
2. 复制这些块的 `patch_id`（追加 `__overfit_validation` 后缀），设 `split=validation`
3. 所有块的原始 split 强制设为 `train`
4. 此时 train 和 validation 完全相同，模型应能快速过拟合到近乎零 loss——验证 pipeline 端到端畅通

---

### `train.synthetic_gate`

控制合成闸门盖章信息。训练阶段通常留空——这些字段在 `stamp-gate` 子命令或 `run_synthetic_gate.ps1` Runner 中填充。

```yaml
  synthetic_gate:
    synthetic_gate_report_dir:
    synthetic_gate_report_card:
    synthetic_gate_frozen_candidate: false
```

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `synthetic_gate_report_dir` | 路径或空 | 空 | 消融报告卡所在目录。与 `synthetic_gate_report_card` 必须同时提供或同时为空 |
| `synthetic_gate_report_card` | 路径或空 | 空 | 消融报告卡 JSON 文件。提供后写入 `model_run_manifest.json` 的 `synthetic_gate_evidence` 段 |
| `synthetic_gate_frozen_candidate` | 布尔 | `false` | 标记此 run 属于当前冻结的合成闸门候选集 |

训练发布时可同时写入 `synthetic_gate_evidence_status: "ok"` 和报告路径，并将评估 run 的契约指纹记录到 `input_contracts.synthetic_gate_evaluation`。旧的事后 `stamp-gate` 会被拒绝，因为成功 run 不允许原地改写。这是后续真实工区 R0 推理的准入门槛。

---

### Control 与 anchor 的复现关系

Control 和 anchor 是两个独立实验，不互相读取输出。复现保证来自以下机制：

1. **`patch_index` 与 `normalization` 留空**：每次运行基于同一 benchmark、切块配置和 seed 独立确定性生成。不需要把 anchor 显式绑定到 control 的文件。
2. **所有随机流从 `seed` 派生**：PyTorch 全局种子、WeightedRandomSampler 生成器、real-delta sampler 的簇/井洗牌种子均从 `seed` 确定性派生。
3. **synthetic batch sequence 指纹**：每个 batch 的 `patch_id` 序列被 SHA-256 哈希，记录在 manifest 的 `training.synthetic_sequence_sha256` 中。

比较两次实验前，应核对 manifest 中的以下契约和条件一致：

- `input_contracts.benchmark.contract_fingerprint_sha256`
- `patch_spec` 和 `split` 配置
- `training.synthetic_sequence_sha256`

其中 batch 序列 SHA 属于随机流审计，不是文件完整性校验。这样锁定的是实验条件，不是路径依赖；模型 run 自身只发布一个 `contract_fingerprint_sha256`。

---

## 脚本在做什么

### 训练阶段

1. **加载基准**：读取 benchmark 的直接契约指纹，并校验 schema、domain、shape、dtype、采样轴和 mask 等显式语义；不重算 benchmark 文件 SHA。
2. **构建块索引**：按切块参数和划分策略，将所有可用剖面切分为训练块、校验块和测试块。`split_policy=derive` 时，同一剖面的块全部分到同一集合，避免训练集和校验集中出现同一剖面的不同块。
3. **计算归一化与参考统计量**：在训练集上计算地震和低频模型的均值、标准差。
4. **训练循环**：组合 synthetic、physics 和可选 real-delta loss；每个 epoch 后只用 synthetic validation loss 选择 best。
5. **双 checkpoint**：同时保存 synthetic validation 最优的 best 和固定最终 epoch 的 final。
6. **真实井 QC**：配置 `real_delta` 时，best/final 都对全部有效井输出 AI、delta、分频和正演 QC。

### 预测与报告阶段

训练完成后，`train_network.ps1` 自动跑三组评估：

| 范围 | 数据划分 | 样本类别 | 测什么 |
|------|---------|---------|--------|
| `validation_base` | 校验集 | 只有基础样本 | 核心精度——干净数据上学进去没有 |
| `test_base` | 测试集 | 只有基础样本 | 泛化能力——未见剖面上的真实水平 |
| `validation_mismatch` | 校验集 | 基础 + 频率探针 + 地震变体 | 鲁棒性——噪声、子波失配等干扰下扛不扛得住 |

每组评估先运行预测（加载 checkpoint 做前向推理），再运行报告（与真值对比计算指标）。

### 汇总与盖章阶段

`run_synthetic_gate.ps1` 消费所有模型的全部报告：

1. **汇总**：把分散的模型报告合并成一份消融报告卡，并排对比各模型在各范围下的指标，自动选出各维度上的最优模型。
2. **盖章**：将消融报告卡的路径和内容指纹写回每个模型的运行清单。这一步是后续真实工区入口的准入门槛——运行清单中缺少合成闸门证据的模型不会被真实工区接受。

---

## 核心输出文件

### 训练输出

所有文件在 `experiments/ginn_v2/results/<实验名称>/` 下：

| 文件 | 内容 |
|------|------|
| `checkpoint_best.pt` | synthetic validation loss 最优权重 |
| `checkpoint_final.pt` | 最终 epoch 权重 |
| `training.log` | 终端同款阶段、batch 耗时、loss 与 ETA 日志 |
| `training_history.csv` | 每个 epoch 的各项训练损失、校验损失、耗时和 checkpoint 标记 |
| `patch_index.csv` | 全部块的索引、划分和元数据 |
| `normalization.json` | 输入数据的归一化参数 |
| `input_reference_stats.json` | 地震和低频模型的参考统计量 |
| `model_run_manifest.json` | 运行清单：架构、超参、直接上游契约、切块参数、训练结果摘要和唯一模型契约指纹 |

Manifest schema 为不兼容旧版本的 `ginn_v2_model_run_v2`。下游默认读取 manifest 中
`checkpoints.primary=best`；旧 v1 run 必须重训。手工预测可用
`--checkpoint primary|best|final` 显式选择权重。

配置真实井数据后还会生成 `real_delta_well_samples.csv`、`real_delta_sampling_qc.csv`、
`real_well_metrics.csv`、`real_well_band_metrics.csv`、`real_well_waveform_metrics.csv` 和
best/final 井 QC 图。它们只陈述当前 run 的绝对证据，不自动比较 control，也不产生通过判定。

### 预测与报告输出

在训练输出目录下：

| 子目录 | 内容 |
|--------|------|
| `predict_validation_base/` | 校验基础集的预测张量 |
| `report_validation_base/` | 校验基础集的指标报告卡 |
| `predict_test_base/` | 测试基础集的预测张量 |
| `report_test_base/` | 测试基础集的指标报告卡 |
| `predict_validation_mismatch/` | 校验不匹配集的预测张量 |
| `report_validation_mismatch/` | 校验不匹配集的指标报告卡 |

每份报告卡包含：回归指标（RMSE、NRMSE、相关系数）、低频模型基线和理想低频基线的对照指标、几何指标（边界误差、事件误差、横向梯度误差）、块级缝合评估指标。

### 闸门输出

所有文件在 `experiments/ginn_v2/results/synthetic_gate_<闸门名称>/` 下：

| 文件 | 内容 |
|------|------|
| `ablation_report_card.json` | 消融报告卡：各模型各范围指标并排对比，最优模型推荐 |
| `synthetic_gate_manifest.json` | 闸门清单：参与模型列表、报告来源、盖章记录 |

---

## 如何阅读结果

### 第一步：看训练输出

```
Patches: 192390
Best validation loss: 0.39
```

块数太少说明基准规模小或 `min_valid_fraction` 设得太高。训练损失和校验损失差距过大说明过拟合——增加 `max_patches` 或减小 `batch_size`。

### 第二步：看消融报告卡

打开 `ablation_report_card.json`，关注三个关键字段：

- **`best.test_base_by_rmse`**：在未见剖面上的最小 RMSE 模型。这是消融实验的核心结论——谁泛化最好。
- **`best.mismatch_by_rmse`**：在不匹配数据上的最小 RMSE 模型。反映谁的鲁棒性最强。
- **`conclusion`**：自动生成的推荐文字。如果 recommend 为 `continue_ablation`，说明横向混合模型仍未显著超过单道模型，可以继续增加消融维度。

### 第三步：对比低频模型基线

每次报告都包含 `lfm_rmse`（退化低频模型基线）和 `lfm_ideal_rmse`（理想低频基线）。关键技术指标是 `rmse_improvement_pct_vs_lfm`——模型相对于退化低频基线提升了多少。如果这个值接近零，说明模型几乎没有学到超越先验的信息。

### 第四步：检查自检值

`oracle_rmse` 必须为 0（或机器精度级别接近 0）。不为 0 说明基准数据存在泄漏或加载路径错误。

### 第五步：看几何分解指标

`geometry_boundary_rmse`、`geometry_event_rmse`、`geometry_lateral_gradient_rmse` 分别衡量模型在层位边界、地质事件和横向梯度上的表现。如果边界误差远高于事件误差，说明模型在阻抗突变处有模糊化倾向。

---

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| `No synthoseis_lite result found` | `benchmark_dir: auto` 找不到合格合成基准 | 确认 `experiments/synthoseis_lite/results/` 下有含 `generate_field_conditioned` 的完整结果 |
| 训练损失不下降 | 学习率或架构不适配当前基准 | 调低学习率、增加 `epochs`、或换模型架构 |
| 校验损失远高于训练损失 | 过拟合 | 增加 `max_patches` 限制数据量、减小模型容量（`hidden_channels`、`depth`） |
| `best_validation_loss` 极大 | 模型几乎没有学到有效映射 | 检查归一化是否正确、基准数据是否合理 |
| `oracle_rmse` 不为 0 | 基准或评估逻辑异常 | 重新生成合成基准并检查 schema、轴、shape、dtype 与 mask 语义 |
| stamp-gate 报 `missing` | 运行清单缺少某些报告 | 确认 `train_network.ps1` 的三组评估全部完成 |
| 预测时显存溢出 | 评估块数或块尺寸过大 | 增大 `lateral_stride` 和 `twt_stride`，或设置 `max_patches` |
| `model_role` 自动推断错误 | `model_id` 命名不规范 | 在 `train.yaml` 中显式填写 `model_role` |

---

## 留到第二轮

- 模型架构快速扩展机制：当前添加新架构需要改多处代码和注册表，后续应支持通过配置文件注册自定义模型。
- 探针增量评估与频率选择性诊断：当前探针相关指标在核心评估中未完全覆盖，后续应将探针对比和频率假能量检查纳入默认评估范围。
- 训练超参自动搜索：`hidden_channels` 和 `depth` 等关键超参目前靠手调，后续应支持网格搜索或贝叶斯优化。
- 与真实工区诊断结果的双向反馈：真实工区 R1 的正演诊断结论反向校准合成基准的退化参数，使两个闸门的难度对齐。
