# 06 岩石物理分析

入口：`scripts/rock_physics_analysis.py`  
配置：`experiments/common/common.yaml` 的 `rock_physics_analysis` 段

本步骤自动发现最新合格的 Step 3 `well_preprocess_*` 运行，也可用
`source_runs.well_preprocess_dir` 固定输入。`well_preprocess_status.csv` 是
权威井清单；所有 `preprocess_status=passed` 的 LAS 都必须存在且可读取。
井分层、Step 4 标定和 Step 5 shifted LAS 不参与本步骤。

```powershell
python .\scripts\rock_physics_analysis.py
python .\scripts\rock_physics_analysis.py --config .\experiments\common\common.yaml
```

## 模块开关

当前只有 `ai_vp_linear` 模块。`enabled` 必须显式配置；未知模块立即失败。
关闭模块时，脚本仍审计全部 Step 3 输入并成功结束，只写：

- `well_input_inventory.csv`
- `run_summary.json`

关闭状态不会读取或要求子波，也不会生成关系文件或
`forward_model_inputs.json`。

## AI—Vp 线性模块

模块严格读取 `DT_USM [us/m]`、`RHO_GCC [g/cm3]` 和
`AI [m/s*g/cm3]`。它由 `DT_USM` 计算 `Vp`，由 `Vp·Rho` 重算 AI；LAS
中的 AI 只用于一致性 QC。模块不填值、不重采样、不裁异常。

拟合关系为：

```text
AI [m/s*g/cm3] = a [g/cm3] * Vp [m/s] + b [m/s*g/cm3]
```

每口合格井具有相同基础总权重，拟合使用加权 MAD 尺度和 Huber IRLS。
关系必须满足 `a>0`，且全部拟合 AI 反算出的速度有限、为正。

成功产物：

- `modules/ai_vp_linear/rock_physics_relation.json`
- `modules/ai_vp_linear/well_fit_qc.csv`
- `modules/ai_vp_linear/figures/ai_vp_fit.png`
- `forward_model_inputs.json`
- `well_input_inventory.csv`
- `run_summary.json`

`forward_model_inputs.json` 仅在模块成功后生成；它固定记录 Step 3 LAS、
AI—Vp 关系和显式时间子波的 SHA-256，供 `cup.physics` 及后续深度正演使用。
