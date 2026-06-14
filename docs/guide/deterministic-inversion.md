# 确定性波阻抗反演

`deterministic_inversion.py` 是时间域工作流里第七步和第八步之间的旁路实验。它在低频模型生成后、GINN 训练前，运行一次纯地震驱动的确定性叠后波阻抗反演，输出一个不含神经网络的对照基线。

它不是新的主线步骤，不会修改第七步低频模型产物，也不会自动触发第八步训练。产物写入独立输出目录，并在元数据和运行摘要中标注 `baseline/bypass`，与主线结果明确区分。

---

## 快速开始

```bash
python scripts/deterministic_inversion.py
python scripts/deterministic_inversion.py --config experiments/common.yaml
python scripts/deterministic_inversion.py --slice xline=250
python scripts/deterministic_inversion.py --output-dir scripts/output/deterministic_inversion_test
python scripts/deterministic_inversion.py --skip-volume
```

不带参数时，脚本从配置的 `deterministic_inversion` 段读取参数，输出写入 `<output_root>/deterministic_inversion_<timestamp>/`。当低频模型、子波和井锚点路径在配置中留空时，脚本自动从对应步骤的最新输出中寻找。

---

## 运行前需要什么

| 来源 | 内容 | 用途 |
|------|------|------|
| 地震数据 | 当前工区时间域地震体 | 观测记录和 inline/xline/time 几何 |
| 第五步 | `selected_wavelet.csv` | 叠后正演褶积子波，必须和 GINN 使用同一个 |
| 第七步 | `ai_lfm_time.npz` | 低频模型、采样轴、目标层元数据和层位路径 |
| 第六步 | `log_ai_anchor_time.npz` | 可选，仅用于井验收，不进入反演目标函数 |

输入缺省从对应步骤最新输出自动发现；复现实验时仍可填写具体上游目录或产物路径。地震始终读取顶层 `seismic`，不允许步骤级覆盖。

---

## 配置参考

```yaml
deterministic_inversion:
  boundary_extension_samples: 30
  epsR: 0.20
  damp: 0.03
  iter_lim: 100
  export_volume: true
```

日常 YAML 只保留科学求解决策。显式上游目录和文件路径仍可用于固定实验，但空 `source_runs`、空路径和 `null` override 不写入常用配置。低频模型必须包含层位和目标层元数据；井锚点缺失时反演仍可运行，但井 QC 会记录跳过原因。

### `boundary_extension_samples`

目标层上下额外保留的样点数。作用是给子波褶积留出缓冲——窗口太窄会导致边界附近的正演不准。这不是 GINN 的 `boundary_effect_samples`，但含义相近。

### `epsR`

空间拉普拉斯正则化权重。只作用于 inline 和 xline 方向，不沿时间方向平滑。值越大，空间上越平滑；值越小，越贴近地震但更容易受噪声影响。

### `damp`

解偏离低频模型的阻尼系数，是确定性反演抑制不适定性的核心手段。值越大，结果越接近低频模型；值越小，地震拟合自由度越高，但出现极端值的风险也越大。求解后若出现非正数或极端值，优先调大本参数。

### `iter_lim`

求解器的最大迭代轮数。默认 100 轮对大多数情况足够；终端求解过程固定显示，不再作为配置参数。

### 输出与 QC

`export_volume` 控制是否默认导出三维体；格式跟随顶层 `seismic.type`，ZGY 分块大小读取顶层配置。`--skip-volume` 可临时关闭。`--slice inline` / `--slice xline=250` 控制本次 QC 剖面，裁剪固定为 1%–99%。有井锚点时井 QC 固定生成。

---

## 脚本在做什么

脚本分五个阶段：**加载与校验 → 裁剪目标窗口 → 建立算子 → 求解反演 → 导出与验收**。

### 第一阶段：加载与校验

1. 打开地震工区，获取时间域几何和完整地震数据体。
2. 读取第七步低频模型产物，提取阻抗体、采样轴、几何和元数据。
3. 读取第五步子波文件。
4. 校验：地震与低频模型 shape 一致、低频模型采样轴与地震时间轴差异不超过 1e-6 秒、子波采样间隔与地震一致、低频模型全部为有限正数。

### 第二阶段：裁剪目标窗口

1. 从低频模型元数据中读取层位路径和目标层质量控制参数。
2. 用工区几何重建目标层三维掩码。
3. 找出所有有效样点覆盖的全局时间范围。
4. 向上下各扩展 `boundary_extension_samples` 形成反演窗口。
5. 裁剪地震、低频模型、掩码和采样轴。

窗口外不做求解，最终输出保持原始低频模型。

### 第三阶段：建立算子

正演算子使用叠后线性建模（`PoststackLinearModelling`）：输入对数波阻抗，通过反射系数近似和子波褶积输出合成地震记录。正则化算子使用只沿 inline 和 xline 方向作用的空间拉普拉斯——时间方向不加平滑，避免先验抹平垂向地质变化。

### 第四阶段：求解反演

先做归一化：计算目标层掩码内地震数据的均方根值，将观测地震除以该值，归一化因子写入运行摘要。然后求解最小化问题：合成地震与观测的拟合误差、空间拉普拉斯正则项、解偏离低频模型的阻尼项，三项加权求和。迭代收敛后，将解从对数域转回波阻抗域，填充回窗口位置，并检查全部为有限正数。

### 第五阶段：导出与验收

写出完整产物包：压缩格式的确定性波阻抗数据体、可选的三维体导出、运行摘要、代表性剖面质量控制图、井质量控制报告。

---

## 核心输出文件

所有文件在 `<output_root>/deterministic_inversion_<timestamp>/` 下：

| 文件 | 内容 |
|------|------|
| `deterministic_ai_full.npz` | 完整的确定性波阻抗数据体，含三维数组、规则轴、几何和完整元数据 |
| `deterministic_ai_full.zgy` / `.segy` | 可选工区原生格式体 |
| `metadata/run_summary.json` | 输入路径、核心参数、窗口范围、归一化因子、迭代状态和残差范数 |
| `figures/<slice>_deterministic_vs_lfm.png` | 四栏剖面图：确定性反演、低频模型、差值、掩码，窗口边界以虚线标注 |
| `well_qc/well_qc_metrics.csv` | 逐井质量控制指标（平均绝对误差、均方根误差、偏差、相关系数、归一化平均绝对误差） |
| `well_qc/figures/well_qc_*.png` | 六联图：井 AI、反射系数、确定性正演、地震、残差和动态互相关，并标注井分层 |
| `well_qc/traces/well_qc_*.csv` | 井 AI、确定性 AI、正演、地震、残差和 QC mask 的逐样点明细 |

### `deterministic_ai_full.npz`

| 键 | 含义 |
|----|------|
| `volume` | 确定性波阻抗体，shape `(n_inline, n_xline, n_sample)` |
| `ilines` / `xlines` / `samples` | 三个规则轴，samples 为正秒 TWT |
| `geometry_json` | 时间域地震几何 |
| `metadata_json` | 输入来源、配置、求解状态、预测统计和输出清单 |

`metadata_json` 中记录了 `experiment_type: baseline/bypass`、`ai_lfm_file`、`wavelet_file`、`seismic_file`、求解参数（`epsR`、`damp`、`iter_lim`、`istop`、`niter`、残差范数）、归一化因子、窗口范围和预测统计。相同配置重复运行时，除时间戳外数值结果可复现。

---

## 如何阅读结果

### 第一步：看运行摘要

先看 `run_summary.json` 中的 `prediction_stats`：

- `deterministic_ai`：确定性反演波阻抗的分布。中位数应在合理波阻抗量级，不应出现物理不可能的极端值。
- `lfm_ai`：低频模型的同口径统计，作为背景基线。
- `deterministic_minus_lfm`：残差分布。中位数应接近零——如果整体明显偏正或偏负，说明归一化或阻尼可能需要调整。

再看 `solver` 中的 `istop` 和 `niter`：`istop` 的取值含义遵循 PyLops 约定（1 为相对残差收敛，2 为达到迭代上限）。如果是 2，说明未在限内收敛，可考虑增大 `iter_lim` 或检查 `epsR`/`damp` 是否设置合理。

### 第二步：看剖面图

剖面四联图从左到右依次是：确定性反演结果、低频模型、差值、目标层掩码。虚线标注反演窗口的上下界。

- 目标层内部：是否相对低频模型呈现了合理的层内细节，而不是散点状噪声。
- 窗口边界：差值在窗口上下界附近是否平滑，边界上是否出现假异常。
- 窗口外部：是否完全等于低频模型（差值为零）。
- 掩码覆盖率：目标层掩码是否覆盖了预期层段，是否存在大块空洞。

### 第三步：看井质量控制

`well_qc_metrics.csv` 每行一口井，按井检查 `vs_anchor_mae`（与锚点的平均绝对误差）、`vs_anchor_rmse`（均方根误差）和 `vs_anchor_corr`（相关系数）。锚点数据是第六步的低频井波阻抗，不是全频井 AI——指标反映的是确定性反演结果在多大程度上保持了井低频趋势。

结合 `well_qc/figures/`：标题中的 `corr/nmae` 衡量求解器正演记录与归一化地震，AI RMSE 用科学计数法报告确定性 AI 与 GINN target 井曲线的差异。层位横线用于确认偏差集中在哪个层段；确定性反演只受地震和正则化驱动，不追求对井曲线的完美贴合。

### 第四步：和 GINN 对比

确定性反演的核心价值在对比中体现。在同一口井上把全频井波阻抗、低频模型、确定性反演和 GINN 第一阶段输出画在一起：

- 井上曲线振荡：如果 GINN 出现比全频井波阻抗更高频的锯齿，而确定性反演在同一位置平滑，振荡更可能来自网络自由度而非地震分辨率。
- 子波外频段能量：计算对数波阻抗残差的频谱。GINN 残差在子波有效频带外出现强能量而确定性反演没有时，这部分信号物理不可信。
- 垂向粗糙度：确定性反演为"合理粗糙度"提供了一个上界参考。GINN 结果远高于这个上界时需要警惕。
- 合成地震误差：GINN 的波形拟合误差通常略低于确定性反演（因为自由度更大），但如果误差只降了一点点、波阻抗却明显更锯齿，应优先相信更稳定的解释。

---

### 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| LFM NPZ 无 `metadata_json.horizons` | 第七步 NPZ 缺少层位信息 | 回到第七步确认 `metadata_json` 写入了 `horizons` 和目标层参数 |
| seismic shape 与 LFM shape 不匹配 | 第七步和本步使用的地震体或采样轴不一致 | 检查地震文件和配置 |
| LFM `samples` 轴与地震采样轴不对齐 | 最大差异超过 1e-6 秒 | 确认第七步使用的时间域地震几何 |
| 子波采样间隔与地震不一致 | 第五步子波和地震 dt 不同 | 确认子波导出时使用的是正确的采样间隔 |
| 目标层掩码为空 | 层位无法覆盖有效时间范围 | 检查层位文件路径和 TWT 数值，适当增大 `boundary_extension_samples` |
| 非正数或非有限波阻抗值 | 反演求解失稳，出现负值或无穷 | 优先增大 `damp`；若仍不可解，再尝试增大 `epsR` 或检查地震归一化 |
| 反演结果几乎等于低频模型 | `damp` 或 `epsR` 过大 | 降低相应参数，但保持求解稳定 |
| 体导出失败 | 写入库不可用或几何不一致 | 用 `--skip-volume` 保留 NPZ，再检查对应格式写入库 |

---

## 留到第二轮

- 独立井弱约束确定性反演实验，必须使用单独实验名称和输出目录，与纯地震基线区分。
- 确定性反演结果低通滤波后替代低频模型作为 GINN 种子，验证对振荡的改善效果。
- 自动化对比报告：把低频模型、确定性基线和 GINN 的井质量控制、频谱、粗糙度和正演误差汇总为一份报告。
