# 08 时间域 GINN 反演

`ginn_inversion.py` 是时间域工作流的第八步。它读取第七步训练好的 checkpoint，复用 checkpoint 内保存的训练配置，在全目标层范围内推理并输出 stage-1 AI 体。

第一版只做时间域 stage-1 反演：不接 enhance，不接 dynamic gain，不重新选择子波，也不重建 LFM。

---

## 快速开始

```bash
python scripts/ginn_inversion.py
python scripts/ginn_inversion.py --config experiments/common.yaml
python scripts/ginn_inversion.py --checkpoint scripts/output/ginn_train/checkpoints/best.pt
python scripts/ginn_inversion.py --output-dir scripts/output/ginn_inversion_test
python scripts/ginn_inversion.py --skip-segy
```

脚本应默认读取 `experiments/common.yaml` 里的 `ginn_inversion` 段；`--checkpoint` 用于临时覆盖 checkpoint 路径。

---

## 运行前需要什么

| 输入 | 用途 |
|------|------|
| 第七步 checkpoint | 重建训练配置、加载模型权重和事实链 |
| checkpoint 内记录的地震/LFM/子波路径 | 复用训练时的数据口径 |
| 可选输出配置 | 控制剖面方向、抽样剖面和 SEG-Y 导出 |

---

## 配置参考

```yaml
ginn_inversion:
  checkpoint_path: scripts/output/ginn_train/checkpoints/best.pt

  slice_mode: inline
  slice_index: null
  clip_percentiles: [1.0, 99.0]

  export_segy: true
```

checkpoint 内已经包含第七步训练配置，因此第八步不应该重复配置 `ai_lfm_file`、`wavelet_file` 或地震路径。除非用户显式传 `--checkpoint`，脚本只使用配置里的 checkpoint。

---

## 脚本在做什么

业务逻辑对齐深度域 `ginn_inversion_depth.py`，但采样域为 TWT 秒：

1. 读取 checkpoint，取出保存的 `config`、`model_state_dict`、epoch 和 best loss。
2. 用 checkpoint 内配置重新构建 `GINNConfig`，并把设备解析为当前机器可用设备。
3. 初始化 `src.ginn.trainer.Trainer`，复用第七步的数据加载、mask、LFM、子波和 fixed gain 口径。
4. 加载模型权重，调用 `predict_volume()` 得到全目标层 stage-1 AI。
5. 未进入 inference mask 的样点保持 LFM，不静默留 0。
6. 输出 NPZ、可选 SEG-Y、QC 图和统计摘要。

第八步不是新的训练实验入口。它只消费第七步 checkpoint 的事实链，保证反演使用的 seismic、LFM、wavelet、mask 和 fixed gain 与训练一致。

---

## 核心输出文件

所有文件在 `<output_root>/ginn_inversion_<timestamp>/` 下：

| 文件 | 内容 |
|------|------|
| `stage1_ginn_base_ai_time.npz` | 时间域 stage-1 AI 预测体 |
| `stage1_ginn_base_ai_time.segy` | 可选 SEG-Y 导出 |
| `qc/prediction_context_time.npz` | 可选 QC 包，保存 LFM 和 mask 等调试上下文 |
| `figures/<slice>_prediction_vs_lfm.png` | 预测 AI、LFM 和差异剖面对比 |
| `figures/prediction_vs_lfm_crossplot.png` | 预测 AI 与 LFM 抽样交会图 |
| `metadata/run_summary.json` | checkpoint、输出、几何和预测统计 |

### `stage1_ginn_base_ai_time.npz`

NPZ 应与第六步 LFM 的轴和 geometry 对齐：

| 键 | 含义 |
|----|------|
| `volume` | stage-1 AI 体，shape 为 `(n_inline, n_xline, n_sample)` |
| `ilines` / `xlines` / `samples` | inline、xline、TWT 秒轴 |
| `geometry_json` | 时间域地震几何 |
| `metadata_json` | checkpoint、训练配置摘要、统计信息和上游路径 |

metadata 中应至少记录 `checkpoint_path`、`checkpoint_epoch`、`checkpoint_best_epoch`、`checkpoint_best_loss`、`ai_lfm_file`、`wavelet_file` 和 `prediction_stats`。

`lfm_volume` 和 `mask` 适合写入可选 QC 包，而不是默认塞进主预测 NPZ。大工区下这两个数组会显著增加文件体积；主 NPZ 保持为 stage-1 AI 体和轴信息即可。

---

## 如何阅读结果

### 预测体统计

`run_summary.json` 应同时记录：

- `prediction_ai` 的 min / p01 / median / p99 / max。
- `lfm_ai` 的同类统计。
- `prediction_minus_lfm` 的分布。

如果 `prediction_minus_lfm` 明显整体偏正或偏负，说明网络可能在用 residual 弥补振幅或低频偏差，需要回到第六、七步检查 LFM 和 fixed gain。

### 剖面图

默认输出中间 inline 或 xline。正式检查时至少抽三类剖面：

- 中央剖面。
- 井密集区域剖面。
- 目标层边界变化剧烈区域剖面。

预测 AI 应在目标层内相对 LFM 增加合理高频细节；目标层外 residual 应被 taper 压回 0。

---

### 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| `Checkpoint not found` | 配置或 CLI 指向的 checkpoint 不存在 | 检查第七步输出目录和 `checkpoint_path` |
| `model_state_dict` 缺失 | checkpoint 不是训练脚本保存的模型文件 | 使用 `best.pt` 或 `last.pt` |
| `LFM shape does not match seismic shape` | checkpoint 配置指向的 LFM 与地震不一致 | 回到第七步确认训练配置是否是当前工区 |
| SEG-Y 导出失败 | 原始地震文件或头字节配置不可用 | 先用 `--skip-segy` 输出 NPZ，再排查 SEG-Y |

---

## 留到第二轮

- 井旁 LAS-vs-prediction QC。
- 输出给 enhance 的 stage-2 输入契约。
- 多 checkpoint 集成或模型不确定性。
- dynamic gain 版本的反演入口。



