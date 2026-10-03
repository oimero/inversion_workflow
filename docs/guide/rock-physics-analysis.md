# 岩石物理分析

`rock_physics_analysis.py` 是工作流的研究旁路。它在第三步输出中查找最近一次包含所需文件的运行，读取其中状态为通过的井的 LAS 曲线，并按配置决定是否拟合全工区统一的 AI–纵波速度关系。

---

## 快速开始

```bash
python scripts/rock_physics_analysis.py
python scripts/rock_physics_analysis.py --config experiments/<project>.yaml
python scripts/rock_physics_analysis.py --output-dir <output_dir>
```

不带参数时，脚本从输出根目录发现最近一次包含 `well_preprocess_status.csv` 和 `run_summary.json` 的 `well_preprocess_*` 运行，在 `<output_root>/rock_physics_analysis_<run_timestamp>/` 下写出结果。

---

## 运行前需要什么

| 来源 | 文件 | 用途 |
|------|------|------|
| 第三步 | `well_preprocess_status.csv` | 权威井清单；只读取 `preprocess_status=passed` 的井 |
| 第三步 | `<preprocessed_las_dir>/*.las` | 每口通过预处理的井的规则 MD 网格 LAS |

### AI–Vp 关系

拟合成功后产出 `rock_physics_relation.json`，供下游正演输入装配使用。

---

## 配置参考

```yaml
rock_physics_analysis:
  source_runs:
    well_preprocess_dir: <well_preprocess_run_dir>  # 留空自动发现
  modules:
    ai_vp_linear:
      enabled: true             # 必填；false 时只读取输入，不拟合
      min_valid_samples_per_well: 100
      min_valid_wells: 3
      huber_delta_sigma: 1.345
      excluded_well_names: [<excluded_well_name_a>, <excluded_well_name_b>]
```

### `source_runs`

`well_preprocess_dir` 为空时，脚本在输出根目录下寻找最近一次包含所需文件的 `well_preprocess_*` 运行。填入具体路径则固定使用该目录。

无论自动发现还是显式指定，第三步的 `well_preprocess_status.csv` 都必须存在且包含 `well_name`、`preprocess_status` 和 `preprocessed_las` 三列。

### `modules`

当前只有 `ai_vp_linear` 一个模块。`enabled` 必须显式填写；设为 `false` 时脚本仍读取所有通过预处理的输入并写出输入清单，但不进行关系拟合。模块启用时，`excluded_well_names` 中的井仍会被读取并记录，但不会进入本模块的候选井集合。

### `ai_vp_linear`

#### `min_valid_samples_per_well`

每口井需要达到配置数量的有限且为正的速度—阻抗样点，不足的井被模块拒绝。示例设置为 100 对；实际对应的测深长度取决于输入曲线的采样步长。

#### `min_valid_wells`

模块至少需要这么多口井通过数据校验才会启动全局拟合。该值必须显式配置且不得小于 3；少于该数量时模块和整步失败。

#### `huber_delta_sigma`

Huber 损失函数中区分二次损失和线性损失的阈值，以稳健尺度的倍数为单位。示例使用 `1.345`。增大阈值让拟合更接近普通最小二乘，减小阈值让拟合对异常点更不敏感。

#### `excluded_well_names`

可选的排除井名列表。模块启用时，名称必须存在于当前第三步运行的通过井清单中；列入该列表的井保留在输入清单和质控结果中，但不参与本模块拟合。

---

## 脚本在做什么

脚本先读取和检查第三步输入，再按模块开关决定是否进行关系拟合。输入检查始终执行，关系拟合只在模块启用时执行。

### 第一阶段：读取和检查输入

1. 读取第三步的井状态表，只选择预处理状态为通过的井。
2. 逐一读取这些井的规则 MD 网格 LAS，检查路径、文件可读性和路径唯一性，并记录曲线名称与单位。
3. 写出输入清单。任一通过井的 LAS 缺失、损坏或路径重复，整次运行失败；没有通过井时也失败。

### 第二阶段：模块分析

当前只有 AI–纵波速度线性关系模块。模块关闭时脚本直接写出输入清单和摘要，并以成功状态结束；模块开启时，排除井之外的候选井进入逐井检查和全局拟合。

#### 逐井数据校验

对每口候选井执行以下处理：

1. 读取慢度、密度和 LAS 中的 AI 曲线，要求单位分别为 us/m、g/cm³ 和 m/s·g/cm³。
2. 由慢度计算纵波速度，再由速度和密度重算波阻抗。
3. 只保留所有输入均有限且为正的样点，比较 LAS AI 与重算 AI；最大相对误差超过 `1e-5` 或最大绝对误差超过 `1e-3 m/s*g/cm3` 时拒绝该井。
4. 有效样点少于配置下限时拒绝该井。

#### 全局拟合

对通过逐井检查的井执行等井权 Huber 回归，拟合全工区唯一关系：

```text
AI [m/s*g/cm³] = a [g/cm³] × Vp [m/s] + b [m/s*g/cm³]
```

拟合要点：

- **等井权**：每口井的总基础权重相同，井内样点均分本井权重，因此井深和采样密度不会改变井间基础权重。
- **稳健尺度**：用加权残差的中位数绝对偏差估计尺度。
- **Huber 迭代**：以等井权线性拟合为初值，按配置的稳健阈值迭代更新样点权重，降低大残差样点的影响。
- **物理约束**：拟合斜率必须为正，且关系反算速度时必须得到有限正值；任一条件不满足，模块失败。

拟合完成后生成逐井质控指标、全局关系文件和散点图。逐井输出记录的是固定全局关系下的拟合误差与有效权重。

---

## 核心输出文件

所有文件在 `<output_root>/rock_physics_analysis_<run_timestamp>/` 下：

### 始终输出

| 文件 | 内容 |
|------|------|
| `well_input_inventory.csv` | 全部第三步井的入选状态、LAS 路径、曲线清单和读取结果 |
| `run_summary.json` | 输入发现方式、模块启停状态、产物清单和拒绝统计 |

模块全部关闭时，脚本只输出这两份文件，`run_summary.json` 中标记 `no_analysis_modules_enabled`。

### 模块启用后的输出

| 文件 | 内容 |
|------|------|
| `modules/ai_vp_linear/rock_physics_relation.json` | 全局关系系数、公式、单位、候选井/配置排除井/合格井/拒绝井清单、Huber 参数、收敛信息和汇总指标；全局拟合成功时写出 |
| `modules/ai_vp_linear/well_fit_qc.csv` | 全部输入井（含配置排除井）的校验状态、样点数、值域、AI 一致性偏差、R²、RMSE、MAE、偏差和权重 |
| `modules/ai_vp_linear/figures/ai_vp_fit.png` | 分井散点图、全局拟合直线和残差图；全局拟合成功时写出 |

---

## 如何阅读结果

### 第一步：看终端输出

```
Wrote rock-physics analysis to scripts/output/rock_physics_analysis_<run_timestamp>
```

正常结束只有这一行。如果有井被拒绝或拟合失败，脚本会在终端打印具体原因后退出。

### 第二步：看 `run_summary.json`

关注：

- `source_run.discovery_mode` — `auto_discovered` 还是 `explicit`，确认输入来源是否符合预期。
- `modules.ai_vp_linear.status` — `success` / `failed` / `disabled`。
- `modules.ai_vp_linear.rejection_counts` — 如果所有井都被拒绝，看拒绝原因的分布。

### 第三步：看 `well_fit_qc.csv`

按 `module_status` 分组：

- **accepted 井**：重点看 `r2` 和 `well_effective_weight`。有效权重受 Huber 残差权重影响，井的基础权重仍按等井权规则确定。
- **rejected 井**：看 `reasons` 列。`ai_consistency_mismatch` 表示 LAS AI 与由慢度、密度重算的 AI 不一致；`insufficient_valid_samples` 表示有效样点少于配置下限。
- **configured_excluded 井**：表示井名出现在排除名单中，井仍在输入清单中，但未参与本模块拟合。

### 第四步：看图

`figures/ai_vp_fit.png` 左侧是纵波速度–AI 散点图，右侧是残差图。图中同时显示各井样点、全局拟合直线和残差分布；整口井的系统性偏离会在残差图中呈现一致方向。

### 第五步：确认下游输入

如果后续步骤报输入不匹配，核对本旁路 `run_summary.json` 记录的来源运行与下游清单中的输入来源。上游 LAS 发生变化时，重建本旁路及所有下游产物。
