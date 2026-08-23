# GINN v2 与 cup.well 重构 Handoff

状态：实现完成，关键链路验收完成。

本 Handoff 描述 2026-08-23 重构后的当前结构、运行语义、入口和验收产物。GINN v2 的主体/残差分界为 25 m。

## 1. 当前语义

- GINN v2 输出 25 m 主体尺度 log-AI；25 m 以下细节由 Enhance v2 承接。
- 自监督预训练与半监督井控微调是两个可独立调用的阶段。预训练 checkpoint 可以在相同输入语义和相同 25 m 主体定义下复用。
- 训练质量指标产生告警；checkpoint 选择在全部微调轮次中按记录分数完成。告警不阻止训练、选择或体推理。
- 体推理覆盖整个目标层。直接预测之外，空间上未覆盖的位置使用最近预测增量；某深度没有任何预测增量时使用 LFM。完整体积的目标层样点必须都有值。
- inline/xline 的 line number 与数组索引分开处理。当前地震体 xline line number 步长为 4；物理距离来自测线几何，不把 line number 差值当成数组步数或米制距离。
- FWHM sweep 属于 GINN 尺度诊断，由 `ginn_v2.diagnose` 负责。

## 2. 模块结构

`src/ginn_v2` 包含 9 个 Python 文件：

| 文件 | 职责 |
| --- | --- |
| `__init__.py` | 六个工作流级公开入口 |
| `workflow.py` | 配置、输入装配、阶段编排和已训练模型 facade |
| `data.py` | 地震 patch、空间切分和井目标 |
| `model.py` | 中心道网络和 25 m 主体投影 |
| `physics.py` | 时域/深度域正演适配 |
| `loss.py` | 波形、LFM、可见度目标和诊断量 |
| `train.py` | 训练配置、checkpoint、评估告警和两阶段训练 |
| `infer.py` | 道、剖面、体推理和完整覆盖 |
| `diagnose.py` | 井波形 QC、FWHM sweep 和图表产物 |

`src/cup/well` 包含 9 个 Python 文件：

| 文件 | 职责 |
| --- | --- |
| `__init__.py` | 井基础工具包说明 |
| `inventory.py` | 井名、资产、井口和空间盘点 |
| `curves.py` | mnemonic 规则、分类和主曲线选择 |
| `las.py` | LAS 读取、标准 Vp/Rho 与导出 |
| `preprocess.py` | MD 规则化、单位、异常值和 AI 派生 |
| `trajectory.py` | 井轨迹、TDT 和标定窗口变换 |
| `tie.py` | 连续标定曲线、标定路由、子波评价 |
| `controls.py` | Step 6 井控构建、加载和深度域 QC |
| `scale.py` | 按物理距离进行井曲线尺度分离 |

两个目录合计 18 个 Python 文件。聚合后同一工作流职责位于同一深模块内，跨模块调用通过少量数据对象或 facade 完成。

## 3. GINN v2 公开接口

包级 API：

```python
from ginn_v2 import (
    BodyRun,
    LoadedBody,
    finetune_body,
    load_body,
    pretrain_body,
    train_body,
)
```

典型阶段调用：

```python
pretrained = pretrain_body(output_dir="...")
finetuned = finetune_body(pretrained=pretrained, output_dir="...")
loaded = load_body(checkpoint=finetuned.selected_checkpoint)
```

`LoadedBody` 负责装配网络、尺度投影、物理适配、patch reader 和体推理器。Enhance v2 与命令行脚本通过它取得道预测或体预测。

命令行入口：

- `scripts/body_train.py`：`all`、`pretrain`、`finetune` 三种阶段；
- `scripts/body_infer.py`：小体积或完整体积推理及 SEG-Y 导出；
- `scripts/body_sweep.py`：25 m 附近的主体边界诊断。

默认配置位于 `experiments/ginn_v2/ginn_v2.yaml`。其中训练输入、25 m checkpoint、体推理 batch size 和 sweep 输入均指向本次关键链路产物。

## 4. 关键链路验收

本次在当前机器上从原始输入重新执行了关键链路：

| 阶段 | 验收产物 | 结果 |
| --- | --- | --- |
| Step 1 井资产盘点 | `scripts/output/well_inventory_20260823_refactor` | 12 口井；9 口进入 LAS 链路 |
| Step 2 LAS 筛选 | `scripts/output/well_screen_20260823_refactor` | 9/9 通过并导出 |
| Step 3 测井预处理 | `scripts/output/well_preprocess_20260823_refactor` | 9/9 通过并导出 |
| 深度域井震标定 | `scripts/output/vertical_well_auto_tie_depth_20260823_refactor` | NW11 完成；裁剪合成相关系数 0.764 |
| 批量深度域合成 | `scripts/output/wavelet_batch_synthetic_depth_20260823_refactor` | 9/9 成功 |
| 岩石物理输入 | `scripts/output/rock_physics_analysis_20260823_refactor` | 完成 |
| 深度域正演输入 | `scripts/output/depth_forward_model_inputs_20260823_refactor` | 完成 |
| Step 6 真实工区井控 | `scripts/output/real_field_well_controls_20260823_refactor` | 9/9 成功；QC 使用 25 m |
| Step 7 真实工区 LFM | `scripts/output/real_field_lfm_20260823_refactor` | 两个变体完成 |
| GINN v2 训练 | `experiments/ginn_v2/results/body_20260823_refactor` | 1 轮预训练、3 轮微调完成 |
| GINN v2 完整体推理 | `experiments/ginn_v2/results/volume_20260823_refactor_final_vector` | 完成并导出 SEG-Y |

训练实际耗时：初始化与基线约 56 秒，自监督阶段约 72 秒，三轮半监督阶段约 6 分 27 秒，训练与 review package 合计约 8 分 51 秒。

最终选择第 3 轮微调 checkpoint：

- masked correlation：0.8464；
- visible correlation：0.8371；
- trusted-well pooled RMSE：0.03060；
- well fraction improved from pretrain：1.0；
- warning 列表：空。

完整体推理的模型计算约 44 分 45 秒，含统计、图件和两个 SEG-Y 写盘约 45 分 55 秒。结果体尺寸为 601 × 801 × 601：

- 目标层样点 49,642,449 个，NaN 数为 0；
- 直接预测占 99.9523%，最近增量补齐占 0.0477%，LFM-only 补齐占 0；
- 双方向预测占 95.2111%，单方向预测占 4.7413%；
- inline/xline 方向差异 RMS 为 0.01815 log-AI；
- `ginn_v2_body_linear_ai.segy` 与 `ginn_v2_body_increment_log_ai.segy` 均已写出，每个 1,272,827,844 字节。

完整摘要位于输出目录中的 `volume_inference_summary.json`，图件位于同目录的 `figures`。

## 5. 验收边界

- `src`、`scripts` 与 `tests` 已通过字节码编译；目标模块已通过导入检查。
- 重构后源码以 24 × 24 × 601 smoke tile 回归通过，目标层 41,065 个样点的 NaN 数为 0。
- 两个正式 SEG-Y 均已由底层读取器回读首尾道。
- 本次执行的是当前小规模 GINN 配置，便于研究迭代。
- `tests/` 中的导入和语义测试已按当前模块路径更新；pytest 由工作区所有者执行。
- Enhance v2 需要以本次 25 m GINN 结果重新训练和评价，才能与当前主体/残差定义一致。
