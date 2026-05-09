# GINN 阶段脚本流程

## 数据流

```
vertical_well_auto_tie_depth.py
        ↓  子波 CSV
wavelet_batch_synthetic_depth.py
        ↓  shifted LAS + 批量标定指标 CSV
lfm_depth.py  ───────────────────────────────┐
        ↓  AI LFM + Vp LFM                  │
dynamic_gain_attr_fitting_depth.py  ←───────┤  共用 shifted LAS 的井旁地震道
        ↓  增益拟合参数 CSV                 │
dynamic_gain_model_depth.py                  │
        ↓  动态增益体 NPZ                   │
well_resolution_prior_depth.py  ←───────────┘  共用 shifted LAS + AI LFM
        ↓  井先验 NPZ
ginn_train_depth.py
        ↓  模型 checkpoint
ginn_inversion_depth.py
        ↓  base AI 体 (NPZ + SEG-Y)
```

---

## 各脚本详细说明

### 1. vertical_well_auto_tie_depth.py

**单井自动井震标定，提取子波。**

| 配置来源 | 说明 |
|---|---|
| `common_depth.yaml` → `vertical_well_auto_tie_depth` | 井名、LAS 参数、搜索空间、子波后处理 |
| `common_depth.yaml` 共享节 | `data_root`、`output_root`、`segy`、`well.well_heads_file`、`seismic_depth.file` |

| 外部输入 | 说明 |
|---|---|
| `las_dir` 中的原始 LAS 文件 | 深度域 LAS（Vp 单位为 us/m） |
| `tutorial_model` | 预训练子波提取网络权重 `.pt` |
| `tutorial_params` | 网络参数 YAML |

输出：裁剪到 201ms 的标定子波 CSV。

---

### 2. wavelet_batch_synthetic_depth.py

**批量井震标定，生成深度偏移 LAS。**

| 配置来源 | 说明 |
|---|---|
| `common_depth.yaml` → `wavelet_batch_synthetic_depth` | 子波来源、LAS 参数、排除井、shift 扫描范围 |
| `common_depth.yaml` 共享节 | `data_root`、`output_root`、`segy`、`well.well_heads_file`、`seismic_depth.file` |

| 外部输入 | 说明 |
|---|---|
| 上一步的子波 CSV | `source_auto_tie_dir` 指向 auto-tie 输出 |
| `las_dir` 中的原始 LAS 文件 | 与 auto-tie 共享同一批 LAS |

输出：每口井的 `shifted_las/*.las`、批量标定指标 CSV。

---

### 3. lfm_depth.py

**构建深度域低频模型。**

| 配置来源 | 说明 |
|---|---|
| `common_depth.yaml` → `lfm_depth` | LAS 参数、LFM 构建参数（变差函数、滤波等） |
| `common_depth.yaml` 共享节 | `data_root`、`output_root`、`seismic_depth.file`、`horizons` |

| 外部输入 | 说明 |
|---|---|
| 上一步的 `shifted_las/*.las` | 深度偏移后的 LAS（Vp 单位已变为 m/s） |
| `train_config` | `experiments/ginn_depth/train.yaml`，仅读取其中的目的层 QC 参数 |

输出：`ai_lfm_depth.npz` + `vp_lfm_depth.npz`。

---

### 4. dynamic_gain_attr_fitting_depth.py

**拟合地震属性与振幅增益的关系。**

| 配置来源 | 说明 |
|---|---|
| `common_depth.yaml` → `dynamic_gain_attr_fitting_depth` | 批量标定输出目录、分段参数、属性选择阈值 |
| `common_depth.yaml` 共享节 | `data_root`、`output_root`、`segy`、`well.well_heads_file`、`seismic_depth.file` |

| 外部输入 | 说明 |
|---|---|
| 上一步的批量标定输出 | `source_batch_dir` 指向 script 2 的输出，读取其中的 batch_metrics.csv 和 QCA CSV |

输出：`gain_attr_fit_metrics.csv`（`ln(gain) = intercept + slope * ln(seismic_rms)` 拟合参数）。

---

### 5. dynamic_gain_model_depth.py

**生成全工区动态增益体。**

| 配置来源 | 说明 |
|---|---|
| `common_depth.yaml` → `dynamic_gain_model_depth` | 拟合参数来源、volume batch size |
| `common_depth.yaml` 共享节 | `data_root`、`output_root`、`segy`、`seismic_depth.file` |

| 外部输入 | 说明 |
|---|---|
| 上一步的 `gain_attr_fit_metrics.csv` | `source_fit_dir` 指向 script 4 的输出 |

输出：`dynamic_gain_depth.npz` + `.segy`。

---

### 6. well_resolution_prior_depth.py

**构建井分辨率先验。**

| 配置来源 | 说明 |
|---|---|
| `common_depth.yaml` → `well_resolution_prior_depth` | shifted LAS 来源、AI LFM 路径、置信度映射参数 |
| `common_depth.yaml` 共享节 | `data_root`、`output_root`、`segy`、`well.well_heads_file`、`seismic_depth.file` |

| 外部输入 | 说明 |
|---|---|
| script 2 的 `shifted_las/*.las` + 批量标定指标 CSV | `source_batch_dir` 指向 script 2 的输出 |
| script 3 的 `ai_lfm_depth.npz` | `ai_lfm_file` 指向 script 3 的输出 |

输出：`well_resolution_prior_depth.npz`。

---

### 7. ginn_train_depth.py

**Stage-1 GINN 深度域训练。**

| 配置来源 | 说明 |
|---|---|
| `experiments/ginn_depth/train.yaml` | 独立的 `DepthGINNConfig` YAML，包含所有训练参数 |

`DepthGINNConfig` 所需的外部输入（在 train.yaml 中以路径形式配置）：

| 外部输入 | 配置字段 | 来源 |
|---|---|---|
| 深度域地震体 | `seismic_file` | 原始数据 |
| 目的层顶/底解释面 | `top_horizon_file`、`bot_horizon_file` | 解释成果 |
| AI 低频模型 | `ai_lfm_file` | script 3 输出 |
| Vp 低频模型 | `vp_lfm_file` | script 3 输出 |
| 子波 CSV | `wavelet_file` | script 1 输出（或使用 Ricker 替代） |
| 动态增益体 | `dynamic_gain_model` | script 5 输出（可选，配合 `gain_source=dynamic_gain_model`） |
| 井先验 NPZ | `well_anchor_prior_file` | script 6 输出（可选，配合 `lambda_well_log_ai > 0`） |

输出：模型 checkpoint（`best.pt`、`metrics.csv`、`run_summary.json`）。

---

### 8. ginn_inversion_depth.py

**GINN 深度域推理，产出 base AI 体。**

| 配置来源 | 说明 |
|---|---|
| `common_depth.yaml` → `ginn_inversion_depth` | checkpoint 路径、SEG-Y 导出、井旁 QC 参数 |
| `common_depth.yaml` 共享节 | `data_root`、`output_root`、`segy`、`well.well_heads_file`、`seismic_depth.file` |

| 外部输入 | 说明 |
|---|---|
| 训练好的 checkpoint `.pt` | 脚本优先使用 checkpoint 内保存的训练配置来恢复推理环境 |
| script 2 的 `shifted_las/*.las` | `source_batch_dir` 指向 script 2 的输出，井旁 QC 用（可选） |

输出：`base_ai_depth.npz` + `.segy`。
