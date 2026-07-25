# Structured GINN V2 实施规格

## 1. 第一版目标

第一版只回答一个问题：

> 网络能否从 clean seismic 和 LFM 中恢复三参数半马尔科夫 latent？

三参数对象表示为：

```text
profile = c0 + c1 * (2ξ - 1) + c2 * sin(πξ)
```

zone 背景表示为：

```text
background = a + b * (2ζ - 1)
log_ai = background + profile
```

第一版使用：

- 深度域；
- 单 trace；
- 单 zone；
- clean seismic 和 LFM 两个输入通道；
- 三状态 emission；
- 固定 transition 和 duration prior；
- exact HSMM；
- zone `a/b` 和 segment `c0/c1/c2` 监督。

第一版的结果是参数恢复、状态恢复和持续时间恢复。真实工区只使用冻结模型推理。

波形 augmentation 属于模型训练侧。canonical dataset 保存 clean observation；训练时按显式随机 identity 在线生成 phase、shift、gain、static 和 noise 扰动。

## 2. 固定数据与坐标约定

time/depth workflow 使用相同的 in-memory record、writer、HDF5 layout 和 reader interface。

domain 差异只进入：

- sample domain；
- sample unit；
- sample axis；
- depth basis；
- forward context；
- domain-specific forward extras。

深度域约定：

```text
sample_domain = depth
sample_unit = m
depth_basis = tvdss
positive_direction = down
```

时间域约定：

```text
sample_domain = time
sample_unit = s
depth_basis = null
positive_direction = increasing_time
```

`inline` 和 `xline` 是几何身份。物理横向距离使用 `lateral_m`。工区 `xline_step=4` 来自 survey geometry，并随 parent identity 保存；它不是 writer 或模型中的常量，也不参与单道垂向 forward。

带采样轴的数据使用 `SampleAxis` 或 `wtie.processing.grid` 中的带轴对象。裸数组只存在于模块内部实现。

三种 mask 分离：

- `observed_valid`：模型网格 observation、LFM 和 forward 的合法支持；
- `truth_valid_highres`：高分辨率有限 truth；
- `segment_supervision_valid`：整个 segment 的参数监督合法性。

## 3. 阶段 0：canonical dataset 重构

状态：代码已落地，depth smoke 已通过。

预计生产代码量：700–1100 行。

### 3.1 单一产物

一个 generation run 的 canonical 数据产物为：

```text
synthetic_benchmark.h5
```

CSV、JSON 和 figures 是索引与报告。

HDF5 root 声明：

```text
artifact_type = structured_synthetic_benchmark
artifact_version = 1
schema = structured_synthetic_benchmark_v1
sample_domain = time | depth
sample_unit = s | m
depth_basis = null | tvdss
```

### 3.2 Parent layout

```text
/realizations/<realization_id>/
├── identity/
├── axes/
│   ├── lateral_m
│   ├── inline_float / xline_float
│   ├── x_m / y_m
│   ├── highres axis
│   └── model axis
├── observed/
│   ├── seismic
│   ├── lfm
│   └── valid
├── truth/
│   ├── log_ai_highres
│   ├── state_id_highres
│   ├── object_id_highres
│   ├── object_xi_highres
│   ├── zone_id_highres
│   ├── truth_valid_highres
│   ├── clipping_mask_highres
│   ├── model_log_ai
│   ├── projection/categorical arrays
│   ├── zones/
│   └── segments/
├── forward/
│   ├── model_consistent_seismic
│   ├── subgrid_residual
│   ├── support/
│   ├── context/
│   └── domain_extras/
└── qc/
```

`zones/` 和 `segments/` 是 columnar datasets。zone 主键为 `(lateral_index, zone_id)`；segment 主键由 `(lateral_index, zone_id, object_id)` 给出。

segment 保存：

- state、state id；
- top、bottom；
- duration fraction、duration samples；
- raw/projected/effective 三套 `c0/c1/c2`；
- `segment_supervision_valid`。

### 3.3 In-memory seam

generation 的唯一 sample interface 是：

```text
StructuredSampleRecord
├── SyntheticTruth
├── ProjectedTruth
├── ForwardResult
├── LfmObservation
├── valid_mask
├── qc
└── domain_metadata
```

time/depth builder 直接构造该 record。pipeline 每个 parent 只调用：

```text
write_structured_sample(h5, record) -> ArtifactReference
```

writer 在 HDF5 staging group 中完成字段、shape、axis、mask、zone/segment 主键和端点校验，随后将 complete parent 移入 `/realizations`。

### 3.4 Reader seam

`StructuredSyntheticBenchmark` 暴露：

```text
list_parents(split) -> ParentIdentity[]
read_parent(parent_id) -> StructuredParent
```

`StructuredTruthAdapter.from_structured_parent()` 将一个 parent 的 `(lateral, zone)` slice 转换为模型和 Oracle 使用的 `StructuredSample`。

reader 根据 root artifact type/version 判断合同。HDF5 group 名和 columnar table 实现不进入模型代码。

### 3.5 已通过 smoke

depth smoke 使用一个短 section、一个 wedge parent，并执行：

```text
generation
→ parent transaction
→ close HDF5
→ reopen reader
→ disk Oracle
→ figures
→ final publication gate
```

结果：

- parent count：1；
- Oracle trace count：14；
- Oracle passed：true；
- parent group 只有 `identity/axes/observed/truth/forward/qc`；
- HDF5 中没有 materialized seismic views；
- run 中没有第二套 truth tree。

迁移 smoke 另外覆盖了普通 parent 和 pinchout parent。pinchout smoke 暴露并修正了单 high-resolution sample segment 的 decoder 约定：profile 求值分母为 `max(segment thickness, highres sample interval)`，与 producer 一致。

## 4. 一次性迁移：20260725

迁移源：

```text
experiments/synthoseis_lite/results/20260725/generate_field_conditioned
```

迁移输出建议：

```text
experiments/synthoseis_lite/results/20260725/generate_field_conditioned_structured
```

迁移器：

```text
scripts/migrate_synthoseis_structured_v1.py
```

它按 parent 流式执行：

1. 从旧 HDF5 读取 axes、clean seismic、LFM、dense truth 和 forward arrays；
2. 从 realization manifest 和 trace manifest 读取 zone/segment truth；
3. 每个 lateral 读取一份 NPZ，校验 seismic、LFM、truth、mask parity，并取得 clipping mask；
4. 组装 `StructuredSampleRecord`；
5. 调用正式 `write_structured_sample()`；
6. flush complete parent；
7. 更新 realization index；
8. 完成后用正式 reader 校验 parent 集合并抽样运行 Oracle。

sidecar 读取按 parent 内最多 8 个 worker 并行。迁移不会复制 materialized seismic views。

完整迁移命令：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python scripts\migrate_synthoseis_structured_v1.py `
  --source-run experiments\synthoseis_lite\results\20260725\generate_field_conditioned `
  --output-dir experiments\synthoseis_lite\results\20260725\generate_field_conditioned_structured `
  --oracle-parent-count 3
```

中断后恢复：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python scripts\migrate_synthoseis_structured_v1.py `
  --source-run experiments\synthoseis_lite\results\20260725\generate_field_conditioned `
  --output-dir experiments\synthoseis_lite\results\20260725\generate_field_conditioned_structured `
  --resume `
  --oracle-parent-count 3
```

迁移门禁：

- parent 数为 1360；
- source HDF5 与 sidecar 的字段级 parity 全部通过；
- realization index 与 HDF5 parent 集合一致；
- 每个 parent 标记 complete；
- 新 HDF5 不含 `seismic_views`；
- 抽样 Oracle 通过；
- reader 只读取迁移后的 HDF5 和 realization index。

2026-07-25 迁移结果：1360 个 parent 全部提交，realization index 与
HDF5 parent 集合一致，staging group 为空，抽样 3 个 parent 的 Oracle
通过。阶段 1 以
`experiments/synthoseis_lite/results/20260725/generate_field_conditioned_structured`
为训练 artifact。

## 5. 阶段 1：单 zone 结构化监督模型

状态：阶段 0 和完整迁移通过后开始。

预计生产代码量：1500–2400 行。

模型输入：

```text
[clean seismic, LFM]
```

模型输出：

- 三状态 emission；
- exact HSMM 的 MAP segmentation；
- zone `a/b`；
- 每个 segment 的 `c0/c1/c2`；
- decoded high-resolution log AI。

训练顺序：

1. true segmentation teacher forcing；
2. 验证 zone `a/b` 与 segment `c0/c1/c2` 恢复；
3. 加入小范围 boundary jitter；
4. 使用 predicted MAP segmentation 验证参数 head；
5. 加入在线有界 waveform augmentation；
6. 冻结 clean holdout 和固定随机种子的 dirty holdout。

在线 augmentation 位于 reader 与模型输入之间：

```text
StructuredSample
→ SeismicAugmentationPipeline(operator spec, random identity)
→ augmented seismic + unchanged LFM/truth
→ model
```

augmentation 只改变 seismic observation。可支持：

- bounded phase rotation；
- bounded wavelet shift；
- depth-domain static；
- global/tracewise positive gain；
- white/colored noise。

扰动强度必须有上限。扰动后可辨识信息或 valid support 不足的样本直接拒绝。

至少报告：

- state accuracy 和 segment IoU；
- duration error；
- `a/b/c0/c1/c2` MAE、相关性和符号一致率；
- decoded high-resolution log-AI error；
- projected model-grid log-AI error；
- clean/dirty 配对性能差；
- no-seismic 和 parent-shuffle 对照。

进入真实阶段的条件：

- 主模型相对 no-seismic 有配对改善；
- parent shuffle 后改善明显减弱；
- 参数恢复不能仅由 LFM 解释；
- augmentation 没有造成不可接受的 clean 性能退化。

## 6. 阶段 2：真实工区冻结模型推理

预计生产代码量：400–700 行。

输入使用真实深度域 seismic、真实工区 LFM、TVDSS axis 和显式 geometry。模型权重冻结。

输出：

- state/segment 结果；
- zone `a/b`；
- segment `c0/c1/c2`；
- decoded/projected AI；
- 不确定性与井震诊断。

真实阶段先判断合成模型是否具有可迁移信号。无标签联合微调、复杂 posterior adaptation、横向 topology 和可微物理约束根据阶段 1 与真实诊断结果另行设计。

## 7. 当前停止条件

当前工作顺序固定为：

```text
canonical generation smoke
→ 20260725 full migration
→ migration parity + sampled Oracle
→ 阶段 1 模型设计与实现
```

完整迁移通过前不开始模型训练。
