# 旁路：Synthoseis-lite 合成基准

当前工区使用 `synthoseis_lite_v2` 深度域分支。它从最新合格的 Step 1 和
Step 6 产物冻结校准，随后生成 field-conditioned 二维合成剖面。详细数据契约
以 [`depth-synthoseis-lite-v2.md`](../spec/depth-synthoseis-lite-v2.md) 为准。

这是一条旁路，不占用步骤编号，也不依赖正演可观测性、Step 4 或 Step 5
运行目录。旧时间域 v1 入口仍保留用于历史工区，但 v1 benchmark 不能被 v2
reader 读取，必须重新生成。

## 运行顺序

先冻结深度校准：

```powershell
python scripts/synthoseis_lite.py `
  --config experiments/synthoseis_lite/synthoseis_lite.yaml `
  calibrate
```

确认校准产物后，再生成 field-conditioned 数据：

```powershell
python scripts/synthoseis_lite.py `
  --config experiments/synthoseis_lite/synthoseis_lite.yaml `
  generate `
  --suite field_conditioned `
  --impedance-calibration scripts/output/synthoseis_lite_calibrate_<timestamp>/impedance_calibration.json
```

初次验证可加 `--debug-attempt-limit 1`。该参数只缩小开发运行，不执行正式
接受率门禁，因此产物状态为 `development_limited`，不得用于正式训练。

## 配置与来源

实验配置必须通过 `workflow_config` 继承
`experiments/common/common.yaml`。工区资产、地震、层位及坐标配置只在 common
文件中维护。

`source_runs.well_inventory_dir` 和
`source_runs.rock_physics_analysis_dir` 留空时，各自自动发现最新合格 Step 1
和 Step 6 run；填写目录时则固定来源。两者独立解析，并在产物中记录路径与
SHA-256。Step 3 LAS、AI–Vp 关系和 NW11 时间子波由 Step 6 的
`forward_model_inputs.json` 锁定，不会另行猜测来源。

深度 v2 只接受：

- `sample_domain: depth`；
- `depth_basis: tvdss`，向下为正；
- 工区原生 5 m 模型轴和 8 倍高分轴；
- 显式 inline/xline 折线路径；
- `field_conditioned` 套件，canonical 和 probe 均关闭。

inline/xline 步长不写死。生成前从工区显式线号轴读取真实步长，验证每个路径
节点能够通过 `SurveyLineGeometry` 完成“线号 → XY → 线号”往返，并检查沿线
采样仍落在解释支撑内。因此 xline 步长 4 是当前数据的属性，而不是代码常量。

## calibrate 产物

校准阶段读取 Step 3 AI，在 `TVDSS = MD - KB` 上按连续有限段做 0.625 m
cell average，不跨缺口。井顶用于曲线分层，解释层面用于剖面几何；两者的
数值差异只审计，不作为门禁，但缺映射、层序交叉或缺解释支撑会失败。

主要产物包括：

- `impedance_calibration.json`：背景 Huber 拟合、对象统计、生成 logAI 上界及
  最大允许 Vp；
- `well_status.csv` 与 `well_zone_status.csv`；
- `well_horizon_consistency.csv`；
- 背景和对象校准 QC 表及图件；
- `run_summary.json`：来源、哈希、拒绝原因和输出清单。

## generate 产物

每个父样本都保存 N 点模型尺度数组：

- `model_target_log_ai`、`vp_model_mps`、LFM 与有效掩码；
- `seismic_observed`：0.625 m 高分正演后抗混叠到 5 m；
- `seismic_model_consistent`：5 m logAI/Vp 直接正演；
- `subgrid_forward_residual`：两条地震的差；
- `tvdss_model_m`，以及嵌套的 `tvdss_highres_m` 和高分 logAI/Vp。

完整高分地震不写入 HDF5，只记录哈希与 QC。时间子波相位/时间平移和最终
米制深度静差是互相独立的错配；静差不外推，并同步收缩有效掩码。

顶层主要文件为：

- `synthetic_benchmark.h5`；
- `sample_index.csv`、`attempt_plan.csv`、`scenario_catalog.csv`；
- `generation_qc.csv`、`subgrid_forward_qc.csv`、`highres_forward_qc.csv`；
- `seismic_variant_results.csv`、对象与剖面几何 QC；
- `benchmark_manifest.json` 与 `run_summary.json`。

正式生成在任一场景未达到尝试数/接受率门禁，或 train/validation/test 任一
split 为空时失败。split 在生成端按父 realization 固定；同一父样本的所有
错配变体继承同一 split，held-out 几何族只进入 test。

## reader 与 GINN v2 接缝

`SynthoseisBenchmark` 严格验证 v2 schema、domain、TVDSS 轴、文件及数组哈希、
来源 `forward_model_inputs_sha256` 和 split 继承关系。它会明确拒绝 v1、错域、
错轴或来源已变化的产物。

GINN v2 的最小消费语义是：网络输入使用 `seismic_observed`，physics loss
参考 `seismic_model_consistent`；深度 target、Vp、两条地震和掩码全部为 N 点，
不执行时间域旧实现的“丢弃首样点”对齐。当前训练端尚未实现完整深度 physics
loss，若对深度数据设置非零 physics 权重会直接报错。

## 验证边界

本仓库只提交测试，测试由用户运行。建议按以下顺序验证：

1. 运行 calibrate，检查井/层审计、背景拟合和最大 Vp；
2. 用 `--debug-attempt-limit 1` 生成最小开发数据，检查轴嵌套、N 点闭合与哈希；
3. 用 reader 打开开发产物，检查 split、mask 和 observed/consistent 语义；
4. 去掉 debug 限制运行正式生成并检查接受率门禁。
