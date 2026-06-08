# 09 时间域 GINN 反演

`ginn_inversion.py` 是时间域工作流的第九步。它读取第八步训练好的 checkpoint，在全目标层范围内推理出 stage-1 波阻抗体。反演不重新训练、不重新选择子波、不重建 LFM——它完全复用 checkpoint 中保存的训练配置和数据口径，保证推理和训练在同一个尺度世界里进行。

第一版只做 stage-1 反演：不接 enhance，不输出 stage-2 产物。如果第八步训练时使用了动态增益，反演会沿用 checkpoint 里记录的增益口径和输入配置。

---

## 快速开始

```bash
python scripts/ginn_inversion.py
python scripts/ginn_inversion.py --config experiments/common.yaml
python scripts/ginn_inversion.py --checkpoint experiments/ginn/results/.../checkpoints/best.pt
python scripts/ginn_inversion.py --output-dir scripts/output/ginn_inversion_test
python scripts/ginn_inversion.py --skip-zgy
```

不带参数时，脚本从配置的 `ginn_inversion.checkpoint_path` 读取 checkpoint；`--checkpoint` 可以临时覆盖这个路径。输出写入 `<output_root>/ginn_inversion_<timestamp>/`。

---

## 运行前需要什么

| 输入 | 用途 |
|------|------|
| 第八步 checkpoint（`best.pt` 或 `final.pt`） | 提取训练配置、模型权重和完整数据事实链 |
| checkpoint 内记录的子波、LFM、地震路径 | 复用训练时的数据口径，确保推理和训练一致 |
| 可选输出配置 | 控制剖面方向、抽样比例和 ZGY 导出 |

反演不需要单独配置地震路径、子波路径或 LFM 路径——这些全部从 checkpoint 中读取。这样做的好处是：即使第八步的训练目录被移动或重跑了，只要 checkpoint 是同一个文件，反演就能精准复现训练时的数据口径。

---

## 配置参考

脚本配置放在 `ginn_inversion` 段下：

```yaml
ginn_inversion:
  checkpoint_path: experiments/ginn/results/.../checkpoints/best.pt

  slice_mode: inline
  slice_index: null
  clip_percentiles: [1.0, 99.0]

  export_zgy: true
  zgy_inline_chunk_size: 16
  write_qc_context: false
  crossplot_max_samples: 200000
```

### `checkpoint_path`

指向第八步训练产出的 checkpoint 文件。脚本会自动从中读取训练配置并重建数据加载管线。同一个 checkpoint 在不同机器上跑反演，只要路径可解析，输出应一致。

### `slice_mode` / `slice_index`

控制 QC 剖面的方向：`inline` 表示沿 inline 方向切，`xline` 表示沿 xline 方向切。`slice_index` 为 null 时取工区中央位置。

### `clip_percentiles`

预测体和 LFM 体在 QC 图上的颜色范围。默认的 1% 到 99% 去掉了最极端的异常值，让多数区域的颜色对比度更好。这个参数只影响显示，不改变输出的体数据。

### `export_zgy` / `zgy_inline_chunk_size`

控制是否将预测体导出为 ZGY 格式。只在原始地震体是 ZGY 时尝试导出；SEG-Y 工区不会导出 ZGY。`zgy_inline_chunk_size` 是写入时的批大小，减少它可以降低内存峰值。

### `write_qc_context`

开启时额外写出一个 NPZ 包，包含 LFM 体和目标层 mask。这个包体积较大（和主输出同级），默认关闭。只在需要离线对比 LFM 和预测体时才打开。

### `crossplot_max_samples`

控制交会图的抽样点数。工区体素量极大时，全部画进 hexbin 会很慢且看不出更多信息。默认 20 万点足够覆盖分布。

---

## 脚本在做什么

脚本分四步：**加载 checkpoint → 重建训练上下文 → 全工区推理 → 导出与 QC**。

### 第一步：加载 checkpoint

从磁盘读取第八步保存的 checkpoint 文件，取出三样东西：训练配置字典（`config`）、模型权重（`model_state_dict`）、以及训练元信息（epoch、best loss 等）。校验 checkpoint 包含必需的 key 和类型。

### 第二步：重建训练上下文

用 checkpoint 内的配置构建一个新的训练上下文——和训练时一样的子波、一样的 LFM、一样的目标层 mask、一样的归一化口径。模型权重加载到新构建的网络上，切换到推理模式。

关键的是：反演不做任何新的配置决策。子波是不是 ricker、LFM 是不是某个特定路径、fixed gain 是不是某个数——这些全部由 checkpoint 决定。反演只负责"照章执行"。

### 第三步：全工区推理

对整个工区的所有 inline 和 xline 逐道推理。网络在同一时间轴上看到地震道、LFM 道和 mask 道，输出每条道上的残差。残差通过 `AI = LFM * exp(residual)` 合成波阻抗。目标层之外的残差被 taper 平滑压回零——所以层外波阻抗基本等于 LFM。

### 第四步：导出与 QC

将推理出的波阻抗体保存为 NPZ，写出几何轴和元数据。元数据记录 checkpoint 来源、训练配置摘要、预测统计和输出文件清单。可选地导出 ZGY 格式和 QC 上下文包。

同时生成两张 QC 图：一张剖面四联图（预测、LFM、差异、mask），一张预测 vs LFM 的二维交会图。

---

## 核心输出文件

所有文件在 `<output_root>/ginn_inversion_<timestamp>/` 下：

| 文件 | 内容 |
|------|------|
| `stage1_ginn_base_ai_time.npz` | 时间域 stage-1 波阻抗预测体 |
| `stage1_ginn_base_ai_time.zgy` | 可选 ZGY 导出，仅在 checkpoint 的地震类型为 ZGY 且未跳过时写出 |
| `metadata/run_summary.json` | checkpoint、输出路径、几何和预测统计 |
| `figures/<slice>_prediction_vs_lfm.png` | 预测、LFM、差异和 mask 四联剖面对比 |
| `figures/prediction_vs_lfm_crossplot.png` | 预测波阻抗 vs LFM 抽样交会图 |
| `qc/prediction_context_time.npz` | 可选 QC 包，包含 LFM 体和 mask 体（默认不写） |
| `trainer_context/` | 训练上下文目录（Trainer 初始化时自动生成，不删） |

### `stage1_ginn_base_ai_time.npz`

| 键 | 含义 |
|----|------|
| `volume` | stage-1 波阻抗体，`(n_inline, n_xline, n_sample)` |
| `ilines` / `xlines` / `samples` | inline、xline 和 TWT 秒轴 |
| `geometry_json` | 时间域地震几何 |
| `metadata_json` | checkpoint 来源、训练配置摘要、预测统计和输出清单 |

metadata 中记录了 `checkpoint_path`、`checkpoint_epoch`、`checkpoint_best_epoch`、`checkpoint_best_loss`、`ai_lfm_file`、`wavelet_file`、`gain_source` 和 `prediction_stats`。这些信息让下游步骤（如 enhance 或人工抽查）在不需要第八步源文件的情况下也能追溯反演的完整事实链。

LFM 体和 mask 体不塞进主 NPZ——大工区下它们的体积和预测体同级，会让文件翻倍。需要调试时打开 `write_qc_context` 单独导出。

---

## 如何阅读结果

### 第一步：看 `run_summary.json` 的预测统计

先看 `prediction_stats` 里的三项：

- `prediction_ai`：预测波阻抗的分布。中位数应该在合理波阻抗量级（通常 5000-15000），min 和 max 不应出现物理上不可能的极端值。
- `lfm_ai`：LFM 波阻抗的同口径统计，作为背景基线。
- `prediction_minus_lfm`：残差分布。它的中位数应接近零——如果整体明显偏正或偏负，说明网络在系统性补偿振幅偏差，需要回到第六步或第七步检查 LFM 和增益。

### 第二步：看交会图

`prediction_vs_lfm_crossplot.png` 是预测值和 LFM 值的二维直方图。红斜线是 1:1 参考线。健康的分布应该沿 1:1 线对称展开，高频细节表现为 LFM 值附近的垂直散布。如果整体偏离 1:1 线、或者存在明显的系统偏斜，说明网络学到了 LFM 之外的大尺度偏差。

### 第三步：看剖面图

剖面四联图从左到右依次是：预测体、LFM 体、差异体、目标层 mask。重点检查三处：

- 目标层内部：预测体是否相对 LFM 增加了合理的层内细节，而不是生成了垂直条带或随机噪声。
- 目标层边界：差异体在层位上下是否平滑过渡到零，而不是突然截断或反向跳变——这验证了 taper 是否生效。
- 目标层外部：预测体是否基本还原为 LFM。如果层外出现明显的异常值，说明 mask 的边界效应超过了预期。

默认剖面取工区中央。正式检查时应至少多看两条：一条过井密集区，一条过目标层厚度变化剧烈区。

---

### 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| checkpoint 不存在 | 配置或 CLI 指向的 checkpoint 路径无效 | 检查第八步输出目录和 `checkpoint_path` |
| `model_state_dict` 缺失 | checkpoint 文件不是训练脚本保存的模型 | 确认文件来自第八步的 `best.pt` 或 `final.pt` |
| LFM 与地震 shape 不匹配 | checkpoint 配置指向的 LFM 不在同一个工区 | 回到第七步检查 `ai_lfm_time.npz` 和地震几何 |
| ZGY 导出失败 | ZGY 写入库不可用或几何不一致 | 先用 `--skip-zgy` 只输出 NPZ，再排查写入端 |
| GPU OOM | 全工区推理显存不足 | 减小 `zgy_inline_chunk_size`，或在 CPU 上跑 |

---

## 留到第二轮

- 井旁波阻抗 vs 预测波阻抗的逐道对比，作为井控验证。
- 输出给 enhance stage-2 训练的输入契约——预测体、子波和 mask 的标准化格式。
- SEG-Y 导出支持（当前只支持 ZGY 导出，SEG-Y 工区不写体格式）。
- 多 checkpoint 集成预测——比如取 best 和 last 的均值作为更稳健的预测。
