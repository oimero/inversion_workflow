# 09 GINN 反演

`ginn_inversion.py` 是时间域工作流的第九步。它读取第八步训练好的 checkpoint，在全目标层范围内推理出 stage-1 波阻抗体。反演完全复用 checkpoint 中保存的训练配置和数据口径，保证推理和训练在同一个尺度世界里进行。如果第八步训练时使用了动态增益，反演会沿用 checkpoint 里记录的增益口径和输入配置。

---

## 快速开始

```bash
python scripts/ginn_inversion.py
python scripts/ginn_inversion.py --config experiments/common.yaml
python scripts/ginn_inversion.py --checkpoint experiments/ginn/results/.../checkpoints/best.pt
python scripts/ginn_inversion.py --slice inline=400
python scripts/ginn_inversion.py --output-dir scripts/output/ginn_inversion_test
python scripts/ginn_inversion.py --skip-volume
python scripts/ginn_inversion.py --write-qc-context
```

不带参数时，脚本自动选择 `experiments/ginn/results/*/checkpoints/best.pt` 中最新的一次；`--checkpoint` 用于固定具体实验。输出写入 `<output_root>/ginn_inversion_<timestamp>/`。

---

## 运行前需要什么

| 输入 | 用途 |
|------|------|
| 第八步 checkpoint（`best.pt` 或 `final.pt`） | 提取训练配置、模型权重和完整数据事实链 |
| checkpoint 内记录的子波、LFM、地震路径 | 复用训练时的数据口径，确保推理和训练一致 |
| 第六步 `log_ai_anchor_time.npz` 和 `well_constraint_points.csv` | 井上波阻抗 QC 的井曲线和全频 AI 来源 |
| 可选输出配置 | 控制剖面方向、抽样比例、井 QC 开关和 ZGY 导出 |

---

## 运行参数

GINN 推理不再拥有常用 YAML 段。checkpoint、QC 切片和调试输出都是本次运行选择：

- `--checkpoint PATH`：固定 checkpoint；缺省自动发现最新 `best.pt`。
- `--slice inline` / `--slice xline=250`：选择剖面方向及可选零基索引；不写索引时取中央剖面。
- `--skip-volume`：只写 NPZ，不写工区原生格式体。
- `--write-qc-context`：额外写出 LFM 和 mask 调试 NPZ。

剖面裁剪固定为 1%–99%，交会图固定最多抽样 20 万点。体格式跟随 checkpoint 的地震类型，ZGY 分块大小读取顶层 `seismic.zgy_inline_chunk_size`。

### 井上波阻抗 QC

反演结束后自动生成与第七步一致的六联井 QC：reference/GINN target/LFM/预测 AI、预测反射系数、GINN 正演、归一化地震、残差和动态互相关。图中显示第六步 point facts 恢复的井分层；标题统一报告 `corr`、`nmae` 和科学计数法 AI RMSE。

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
| `stage1_ginn_base_ai_time.zgy` / `.segy` | 可选工区原生格式体，格式跟随 checkpoint 地震类型 |
| `metadata/run_summary.json` | checkpoint、输出路径、几何和预测统计 |
| `figures/<slice>_prediction_vs_lfm.png` | 预测、LFM、差异和 mask 四联剖面对比 |
| `figures/prediction_vs_lfm_crossplot.png` | 预测波阻抗 vs LFM 抽样交会图 |
| `qc/prediction_context_time.npz` | 可选 QC 包，包含 LFM 体和 mask 体（默认不写） |
| `trainer_context/` | 训练上下文目录（Trainer 初始化时自动生成，不删） |
| `well_qc/figures/well_qc_*.png` | 每口 anchor 井的预测 AI vs GINN target/reference AI 对比图 |
| `well_qc/traces/well_qc_*.csv` | reference AI、GINN target AI、预测 AI 的逐样点明细 |
| `well_qc/well_qc_metrics.csv` | 逐 anchor 井阻抗 QC 指标 |

### `stage1_ginn_base_ai_time.npz`

| 键 | 含义 |
|----|------|
| `volume` | stage-1 波阻抗体，`(n_inline, n_xline, n_sample)` |
| `ilines` / `xlines` / `samples` | inline、xline 和 TWT 秒轴 |
| `geometry_json` | 时间域地震几何 |
| `metadata_json` | checkpoint 来源、训练配置摘要、预测统计和输出清单 |

metadata 中记录了 `checkpoint_path`、`checkpoint_epoch`、`checkpoint_best_epoch`、`checkpoint_best_loss`、`ai_lfm_file`、`wavelet_file`、`gain_source` 和 `prediction_stats`。这些信息让下游步骤（如 enhance 或人工抽查）在不需要第八步源文件的情况下也能追溯反演的完整事实链。

LFM 体和 mask 体不塞进主 NPZ——大工区下它们的体积和预测体同级，会让文件翻倍。需要调试时使用 `--write-qc-context` 单独导出。

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
| checkpoint 不存在 | 未发现最新 `best.pt` 或 CLI 路径无效 | 检查第八步结果目录或显式传 `--checkpoint` |
| `model_state_dict` 缺失 | checkpoint 文件不是训练脚本保存的模型 | 确认文件来自第八步的 `best.pt` 或 `final.pt` |
| LFM 与地震 shape 不匹配 | checkpoint 配置指向的 LFM 不在同一个工区 | 回到第七步检查 `ai_lfm_time.npz` 和地震几何 |
| 体导出失败 | ZGY/SEG-Y 写入库不可用或几何不一致 | 先用 `--skip-volume` 只输出 NPZ，再排查写入端 |
| GPU OOM | 全工区推理显存不足 | 调整训练/推理批大小，或在 CPU 上跑 |

---

## 留到第二轮

- 井旁波阻抗 vs 预测波阻抗的逐道对比，作为井控验证。
- 输出给 enhance stage-2 训练的输入契约——预测体、子波和 mask 的标准化格式。
- SEG-Y 导出支持（当前只支持 ZGY 导出，SEG-Y 工区不写体格式）。
- 多 checkpoint 集成预测——比如取 best 和 last 的均值作为更稳健的预测。
