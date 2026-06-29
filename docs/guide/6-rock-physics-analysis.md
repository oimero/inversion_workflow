# 06 岩石物理分析

`rock_physics_analysis.py` 是深度域工作流的第六步。它自动发现最新合格的第三步预处理产物，读取全部通过预处理的井的 LAS 曲线，通过可开关的分析模块拟合全工区统一的岩石物理关系，并冻结为正演输入清单供后续合成、训练和诊断使用。

本步骤本身不执行地震正演，也不依赖井分层、井震标定结果或时移校正后的 LAS。

---

## 快速开始

```bash
python scripts/rock_physics_analysis.py
python scripts/rock_physics_analysis.py --config experiments/common/common.yaml
python scripts/rock_physics_analysis.py --output-dir scripts/output/rock_physics_test
```

不带参数时，脚本自动从输出目录发现最新的 `well_preprocess_*` 产物，在 `<output_root>/rock_physics_analysis_<timestamp>/` 下写出结果。

---

## 运行前需要什么

第六步只依赖第三步产物：

| 来源 | 文件 | 用途 |
|------|------|------|
| 第三步 | `well_preprocess_status.csv` | 权威井清单；只读取 `preprocess_status=passed` 的井 |
| 第三步 | `preprocessed_las/*.las` | 每口通过预处理的井的规则 MD 网格 LAS |

脚本不读取井分层、第四步标定指标、第五步时移结果或 shifted LAS。

### 子波依赖

仅当 `ai_vp_linear` 模块启用且拟合成功时，脚本才会读取配置中显式指定的时间子波文件（来自第四步的 NW11 子波产物），用于装配正演输入清单。子波不参与拟合过程本身。

---

## 配置参考

```yaml
rock_physics_analysis:
  source_runs:
    well_preprocess_dir:        # 留空则自动发现；填入路径则固定输入
  modules:
    ai_vp_linear:
      enabled: true             # 必填；false 时只做输入审计，不拟合
      min_valid_samples_per_well: 100
      min_valid_wells: 3
      huber_delta_sigma: 1.345
  forward_model:
    wavelet_file: <path-to-wavelet-csv>
    source_well: NW11
```

### `source_runs`

`well_preprocess_dir` 为空时，脚本自动在输出目录下寻找最新的 `well_preprocess_*` 运行。填入具体路径则固定使用该目录，适合复现实验或排除自动发现带来的井群变化。

无论自动发现还是显式指定，第三步的 `well_preprocess_status.csv` 都必须存在且包含 `well_name`、`preprocess_status` 和 `preprocessed_las` 三列。

### `modules`

当前只有 `ai_vp_linear` 一个模块。`enabled` 必须显式填写；设为 `false` 时脚本仍完整读取所有输入并写出审计结果，但不生成关系或正演清单。

未来新增模块（如多孔介质替换、流体替代等）只需在 `modules` 下增加同名配置段，脚本会按白名单校验并顺序执行。

### `ai_vp_linear`

#### `min_valid_samples_per_well`

每口井至少需要这么多对有限且为正的 `(Vp, AI)` 样点，不足的井被模块拒绝。默认 100 点，对应 10 m 左右的测井段（在 0.1 m 步长下）。

#### `min_valid_wells`

模块至少需要这么多口井通过数据校验才会启动全局拟合。少于该数量时模块和整步失败。默认 3 口。

#### `huber_delta_sigma`

Huber 损失函数中区分二次损失和线性损失的阈值，以稳健尺度的倍数为单位。`1.345` 是标准值——当残差服从正态分布时，约 95% 的样点落在二次区。增大阈值让拟合更接近普通最小二乘，减小阈值让拟合对异常点更不敏感。

### `forward_model`

仅在 `ai_vp_linear` 启用时必填。`wavelet_file` 指向第四步产出的 NW11 时间子波 CSV，`source_well` 记录子波来源井名。模块关闭时这一段可以省略，脚本也不会去检查子波文件是否存在。

---

## 脚本在做什么

脚本分三层：**输入审计 → 模块分析 → 正演清单装配**。输入审计始终执行，后两层由模块开关控制。

### 第一层：输入审计（始终执行）

1. 读取第三步 `well_preprocess_status.csv`，按 `preprocess_status=passed` 筛选候选井。
2. 逐一检查每口井的 `preprocessed_las` 路径是否存在且可读取；记录每口井的 LAS 哈希和曲线清单。
3. 写出 `well_input_inventory.csv`，审计实际读到了哪些井和曲线。

任一口状态为 passed 的井的 LAS 缺失、损坏或路径重复，整个第六步立即失败——这一层不受模块开关影响。

### 第二层：模块分析（按配置启停）

当前只有 `ai_vp_linear` 模块。关闭时脚本直接跳到摘要写出，以成功状态结束。

#### 逐井数据校验

对每口候选井：

1. 从 LAS 中严格读取三条曲线：`DT_USM [us/m]`、`RHO_GCC [g/cm3]`、`AI [m/s*g/cm3]`。曲线缺失或单位与预期不符即拒绝该井。
2. 由慢度计算速度：`Vp = 1,000,000 / DT_USM`，再由速度和密度重算波阻抗：`AI_recomputed = Vp × RHO_GCC`。
3. 在 DT、Rho、Vp、LAS AI 和重算 AI 全部有限且为正的样点上比较 AI 一致性。不一致的井以 `ai_consistency_mismatch` 拒绝。
4. 剩余有效样点少于 `min_valid_samples_per_well` 的井以 `insufficient_valid_samples` 拒绝。

整个过程不填值、不裁异常、不平滑、不重采样——样点要么原样通过，要么整口井被拒绝。

#### 全局拟合

对通过校验的井做等井权 Huber 回归，拟合全工区唯一关系：

```text
AI [m/s*g/cm³] = a [g/cm³] × Vp [m/s] + b [m/s*g/cm³]
```

拟合要点：

- **等井权**：每口井基础权重相同，与井深和采样密度无关。井内每个样点均分该井权重。
- **稳健尺度**：用加权残差的中位数绝对偏差（MAD）估计数据离散度，不被少量极端样点绑架。
- **Huber 迭代**：以等井权最小二乘为初值，在 `1.345σ` 阈值下迭代重加权，逐步降低异常点影响力。
- **硬约束**：斜率 `a` 必须大于零，且全部拟合样点的 AI 反算回速度后必须全部有限且为正。任一条件不满足，模块失败且不生成正演清单。

拟合完成后生成逐井 QC 指标和全局散点图，但这些逐井系数仅供诊断，不写入正演输入清单。

### 第三层：正演清单装配（条件执行）

模块成功后才触发。脚本读取配置中指定的 NW11 时间子波，校验采样规则性和中心零点，然后装配 `forward_model_inputs.json`。该文件固定记录：

- 第三步来源目录和每口预处理 LAS 的 SHA-256；
- 岩石物理关系的 a/b、单位和自身文件 SHA-256；
- NW11 子波路径、采样间隔和 SHA-256。

任何来源数据变化都会导致哈希改变，下游可通过校验 `forward_model_inputs_sha256` 发现输入漂移。

---

## 核心输出文件

所有文件在 `<output_root>/rock_physics_analysis_<timestamp>/` 下：

### 始终输出

| 文件 | 内容 |
|------|------|
| `well_input_inventory.csv` | 全部第三步井的入选状态、LAS 路径与哈希、曲线清单和读取结果 |
| `run_summary.json` | 输入发现方式、模块启停状态、产物清单和拒绝统计 |

模块全部关闭时，脚本只输出这两份文件，`run_summary.json` 中标记 `no_analysis_modules_enabled`。

### 模块成功时额外输出

| 文件 | 内容 |
|------|------|
| `forward_model_inputs.json` | 冻结的正演输入：子波引用、岩石物理关系、来源哈希 |
| `modules/ai_vp_linear/rock_physics_relation.json` | 全局 a/b、公式、单位、合格井清单、Huber 参数、收敛信息和汇总指标 |
| `modules/ai_vp_linear/well_fit_qc.csv` | 全部候选井的校验状态、样点数、值域、AI 一致性偏差、R²、RMSE、MAE、偏差和权重 |
| `modules/ai_vp_linear/figures/ai_vp_fit.png` | 分井散点图 + 全局拟合直线 + 残差图 |

---

## 如何阅读结果

### 第一步：看终端输出

```
Wrote rock-physics analysis to scripts/output/rock_physics_analysis_<timestamp>
```

正常结束只有这一行。如果有井被拒绝或拟合失败，脚本会在终端打印具体原因后退出。

### 第二步：看 `run_summary.json`

关注：

- `source_run.discovery_mode` — `auto_discovered` 还是 `explicit`，确认输入来源是否符合预期。
- `modules.ai_vp_linear.status` — `success` / `failed` / `disabled`。
- `modules.ai_vp_linear.rejection_counts` — 如果所有井都被拒绝，看拒绝原因的分布。

### 第三步：看 `well_fit_qc.csv`

按 `module_status` 分组：

- **accepted 井**：重点看 `r2` 和 `well_effective_weight`。某口井 R² 明显低于其他井时，它的 Huber 有效权重会被压低——说明该井不服从全局线性关系。
- **rejected 井**：看 `reasons` 列。`ai_consistency_mismatch` 表示第三步导出的 AI 与从 DT/Rho 重算的 AI 不一致，上游数据可能有问题。`insufficient_valid_samples` 表示有效样点太少，通常是因为曲线覆盖不足。

### 第四步：看图

`figures/ai_vp_fit.png` 左侧是 Vp–AI 散点图，右侧是残差图。全局关系应该穿过大多数井的主体点云。如果有整口井的系统性偏离（散点在线的同一侧），说明该井可能有独立的岩石物理趋势，被 Huber 降权是合理的。

### 第五步：确认正演输入

如果后续 Synthoseis 或训练报哈希不匹配，打开 `forward_model_inputs.json` 对比下游 manifest 中记录的 `forward_model_inputs_sha256`——任一来源 LAS、关系文件或子波变化都会导致哈希改变。

---

## 留到第二轮

- 是否支持多模块并行执行（如同时做 AI–Vp 和 Vp–Vs 关系拟合）。
- 是否允许按区块或井型分组建模，产出多组岩石物理关系。
- 是否在拟合前对样点做地质层段筛选（如只拟合储层段）。
- 拟合图的交互式版本。
