# 时间域工作流总览

时间域主链由九个顺序步骤 + 一个旁路脚本组成。

## 主链

| 步骤 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| 01 | `well_inventory.py` | Petrel 井头、LAS 目录、地震体 | `well_inventory.csv` |
| 02 | `well_screen.py` | `well_inventory.csv`、LAS 目录 | `well_screen.csv`、`selected_las/` |
| 03 | `well_preprocess.py` | `well_screen.csv`、`selected_las/` | `well_preprocess_status.csv`、`preprocessed_las/` |
| 04 | `well_auto_tie.py` | 03 产物 + 时深表 + 井分层 + 轨迹 QC | `well_tie_metrics.csv`、优化后 TDT、子波 |
| 05 | `wavelet_generation.py` | 04 子波 + 标定产物 | `selected_wavelet.csv`、`batch_synthetic_metrics.csv` |
| 06 | `well_constraints.py` | 04 标定结果 + 05 全局子波评测 + 层位 | `lfm_control_points.csv`、`log_ai_anchor_time.npz`、高频监督和统计 |
| 07 | `lfm_precomputed.py` | 06 `lfm_control_points.csv` + 层位 | `ai_lfm_time.npz` |
| 08 | `ginn_train.py` | 时间域地震体 + 05 `selected_wavelet.csv` + 07 LFM + 可选 06 anchor | GINN checkpoint |
| 09 | `ginn_inversion.py` | 08 checkpoint | stage-1 波阻抗预测体 |

## 旁路

| 脚本 | 说明 | 运行时机 |
|------|------|----------|
| `well_trajectory.py` | 解析轨迹文件，复核井型（直/斜），输出轨迹几何事实 | 01 之后，04 之前 |

旁路不改变主链编号，但 04 的路由决策依赖轨迹 QC 的输出。

## 数据流

```mermaid
flowchart LR
  A["01 well_inventory"] --> B["02 well_screen"]
  B --> C["03 well_preprocess"]
  C --> D["04 well_auto_tie"]
  D --> E["05 wavelet_generation"]
  E --> F["06 well_constraints"]
  F --> G["07 lfm_precomputed"]
  G --> H["08 ginn_train"]
  H --> I["09 ginn_inversion"]
  A --> T["well_trajectory (旁路)"]
  T --> D
```

## 深度域 Legacy 工作流

深度域脚本（`scripts/*_depth.py`）独立于上述主链，互不依赖。
两者共享 `src/cup/` 库函数与 `src/wtie/` 核心，但脚本层面的
输入/输出产物不交叉。
