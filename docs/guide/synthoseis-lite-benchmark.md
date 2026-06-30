# 旁路：Synthoseis-lite 合成基准

Synthoseis-lite 当前有两条受支持分支，均使用 `synthoseis_lite_v2` 产物 schema：

- 时间域 `sample_domain: time`：保留 canonical、field-conditioned、Hz probe、seismic variant 和 evaluate。
- 深度域 `synthoseis_lite_v2`：TVDSS 米制 field-conditioned 流程，契约见本文“深度域 v2”。

入口脚本保持单一：

```powershell
python scripts/synthoseis_lite.py --config <yaml> calibrate
python scripts/synthoseis_lite.py --config <yaml> generate --suite <suite> --impedance-calibration <json>
```

配置必须显式声明分支，不再根据是否存在 `workflow_config` 猜测：

```yaml
synthoseis_lite:
  sample_domain: time | depth
  benchmark_schema: synthoseis_lite_v2
```

`SynthoseisBenchmark` 先读取 manifest，再按 schema/domain 分派 reader：

- `sample_domain: time` 分派到 time-v2 reader；
- `sample_domain: depth` 分派到 depth-v2 reader；
- `synthoseis_lite_v1` 是 unsupported legacy schema；
- 错 schema/domain 组合直接失败，不静默 fallback；
- 不提供旧产物升级或坏产物修复迁移。

## 时间域 v2

时间域配置通过 `workflow_config` 继承 common workflow，并显式声明：

```yaml
workflow_config: experiments/common/common.yaml

synthoseis_lite:
  sample_domain: time
  benchmark_schema: synthoseis_lite_v2
```

`seismic.domain: time` 属于 workflow 真相，应写在 `experiments/common/common.yaml`
的 `seismic` 段里；time 实验配置不应覆盖地震文件、资产、层位或井曲线等 workflow 字段。

时间域 v2 依赖 Step 4/5/6 的时间域来源，保留：

- `canonical`；
- `field_conditioned`；
- `frequency_probe`；
- `seismic_variant`；
- `frequency_probe_seismic_variant`；
- `scripts/evaluate_synthoseis_lite.py` 基础评估。

运行顺序：

```powershell
python scripts/synthoseis_lite.py `
  --config experiments/synthoseis_lite/synthoseis_lite.yaml `
  calibrate
```

```powershell
python scripts/synthoseis_lite.py `
  --config experiments/synthoseis_lite/synthoseis_lite.yaml `
  generate `
  --suite field_conditioned `
  --impedance-calibration scripts/output/synthoseis_lite_calibrate_<timestamp>/impedance_calibration.json
```

评估：

```powershell
python scripts/evaluate_synthoseis_lite.py `
  --benchmark-dir scripts/output/synthoseis_lite_generate_<timestamp>
```

时间域 v2 reader 对训练端返回 N 点对齐数组。HDF5 中保留正演原始 N−1 数组时，
reader 在 adapter 边界补齐到模型轴；GINN 不再把 time-v2 当 legacy Robinson `[1:]`
契约处理。

时间域 v2 的 generator 只冻结父样本身份和评估角色：

- `parent_realization_id`；
- `evaluation_role=development_pool|geometry_holdout`；
- mismatch variant 必须继承 parent 的 `evaluation_role`。

生成端不写 train/validation/test 比例。GINN 训练端在 `split_policy=derive` 下按
`parent_realization_id` 哈希派生 train/validation/test；`geometry_holdout` 永远进入
test。改变训练比例只重建 patch index 和 normalization，不重新生成 HDF5 benchmark。

时间域 v2 额外写：

- `attempt_plan.csv`；
- `section_geometry_feasibility_qc.csv`；
- `rejection_reason_summary.csv`；
- `well_horizon_consistency.csv`；
- HDF5 文件级 provenance attrs；
- dataset 级 `unit/sample_domain/axis_path/axis_order/shape_json/dtype/sha256`；
- `masks/physics_valid_mask` 与 `seismic/subgrid_forward_residual`。

## 深度域 v2

深度域配置通过 `workflow_config` 继承 common workflow，并显式声明：

```yaml
workflow_config: experiments/common/common.yaml

synthoseis_lite:
  sample_domain: depth
  benchmark_schema: synthoseis_lite_v2
```

深度 v2 只接受：

- `sample_domain: depth`；
- `depth_basis: tvdss`，向下为正；
- 工区原生 5 m 模型轴和 8 倍高分轴；
- 显式 inline/xline 折线路径；
- `field_conditioned` 套件，canonical 和 probe 均关闭。

深度 v2 校准依赖最新合格 Step 1、Step 5 和 Step 6；复现实验时可显式固定：

```yaml
synthoseis_lite:
  source_runs:
    well_inventory_dir:
    rock_physics_analysis_dir:
    wavelet_batch_synthetic_depth_dir:
```

其中 Step 6 继续提供 `forward_model_inputs.json`、子波和 AI–Vp 关系；井曲线来源改为
Step 5 的两套深度平移 LAS：

- `shifted_filtered_las/AI` 只用于 background fit，避免背景图件和趋势被尖刺支配；
- `shifted_preprocessed_las/AI` 用于 full logAI、对象残差和后续 truth 统计。

运行顺序：

```powershell
python scripts/synthoseis_lite.py `
  --config <depth-v2-yaml> `
  calibrate
```

```powershell
python scripts/synthoseis_lite.py `
  --config <depth-v2-yaml> `
  generate `
  --suite field_conditioned `
  --impedance-calibration scripts/output/synthoseis_lite_calibrate_<timestamp>/impedance_calibration.json
```

初次验证可加 `--debug-attempt-limit 1`。该参数只缩小开发运行，不执行正式接受率门禁，
因此产物状态为 `development_limited`，不得用于正式训练。

time/depth 的 `field_conditioned` 共享两阶段生成契约。第一阶段先对全部 attempt
执行不包含昂贵高分正演和 mismatch 生成的结构 preflight；第二阶段只正演
preflight 通过的 parent。运行过程中终端和 `generation.log` 会逐 attempt 输出状态，
同时持续刷新可在另一个终端读取的 `attempt_progress.csv`。因此不需要等 HDF5 关闭后
才能知道当前 scenario 的成功数、拒绝原因和理论最高接受率。

接受率配置为：

```yaml
generation:
  acceptance_qc:
    minimum_attempts_per_scenario: 20
    warning_fraction: 0.80
    failure_fraction: 0.50
    enforcement: warn  # warn | fail_fast
```

时间域和深度域共享的 object-core 门控配置位于
`synthoseis_lite.impedance_attribute_generator`，不是 `generation`：

```yaml
impedance_attribute_generator:
  lateral:
    correlation_length_section_fractions: [0.1, 0.3, 1.0]
    coefficient_sigma_multipliers: [0.25, 0.50]
    thickness_log_sigma_values: [0.10, 0.25]
  qc:
    max_global_reversal_fraction: 0.10
    max_object_reversal_fraction: 0.25
    max_global_clipping_fraction: 0.005
    max_object_clipping_fraction: 0.02
  robust_scale:
    huber_delta_parent_sigma_floor: 0.05
    coefficient_sigma_parent_floor: 0.05
    coefficient_sigma_parent_cap: 3.0
```

- `warn` 是默认行为：接受率不足时保留全部成功样本，run/manifest 标记
  `completed_with_warnings`，终端明确告警，但脚本正常完成。
- `fail_fast`：preflight 确认任一 scenario 无法通过正式门禁后，在创建昂贵 HDF5
  之前停止。
- 没有任何 preflight 成功样本、输入损坏或正演契约错误仍属于运行失败，不降级为告警。

preflight 固定输出 `preflight_attempts.csv`、`preflight_scenario_catalog.csv` 和
`preflight_summary.json`。最终 `scenario_catalog.csv` 的分母始终是原始
`attempt_plan.csv`，不会因为第二阶段只处理成功 parent 而虚假变成 100%。

time/depth 的 `field_conditioned` 都支持三个诊断型 CLI 开关：

- `--geometry-family <none|wedge|pinchout>`：临时过滤本次生成的几何家族，不改配置文件；
- `--qc-only`：完整执行生成和接受率统计，但不持久化 realization HDF5 数组。
- `--debug-attempt-limit <正整数>`：逐 section/scenario 限制开发运行的 attempt 数。

`--qc-only` 产物会写 `sample_index.csv`、`generation_qc.csv`、`scenario_catalog.csv`、
`benchmark_manifest.json` 等 QC 文件，并在 manifest 中标记 `qc_only=true` 与
`training_consumable=false`。两个 domain reader 和训练端都拒绝把它当正式 benchmark 使用。

深度 v2 的 generator 只冻结父样本身份和评估角色：

- `parent_realization_id`；
- `evaluation_role=development_pool|geometry_holdout`；
- mismatch variant 必须继承 parent 的 `evaluation_role`。

生成端不写 train/validation/test 比例。GINN 训练端在 `split_policy=derive` 下按
`parent_realization_id` 哈希派生 train/validation/test；`geometry_holdout` 永远进入
test。改变训练比例只重建 patch index 和 normalization，不重新生成 HDF5 benchmark。

深度 v2 的 mismatch 覆盖与时间域 v2 对齐到可比范围：

- 独立时间子波相位旋转和秒制子波平移，均重新执行深度正演；
- 米制深度静差；
- white/colored noise、global/tracewise gain；
- `vertical_lateral_smooth_gain`，即横向与 TVDSS 方向共同变化的二维平滑增益场；
- `combined_moderate`，按“扰动时间子波 → 深度正演 → 深度静差 → gain → noise”顺序执行。

每个 depth seismic variant 在 HDF5 中写 `seismic_observed`、`observed_valid_mask`、
`positive_gain` 和 `additive_noise`。depth LFM 在 `controlled_degraded` 后可配置米制
`over_smoothing`；深度域禁止 Hz 低通字段，并额外写
`residuals/residual_vs_lfm_ideal` 与
`residuals/residual_vs_lfm_controlled_degraded`。

## Reader 与 GINN v2 接缝

`SynthoseisBenchmark` facade 保留：

- `sample_ids()`；
- `row()`；
- `load_sample()`。

domain reader 返回明确采样轴：

- time-v2：TWT/time 轴；
- depth-v2：TVDSS 轴。

GINN 通过训练 adapter 获取已对齐数组：

- time adapter 使用 reader 返回的 N 点 time-v2 数组；
- depth adapter 使用 N/N 和 physics mask；
- `PatchDataset` 不无条件读取 depth-only 字段；
- 深度 physics loss 在专项实现前继续拒绝非零权重。

## 验证边界

本仓库只提交测试，测试由用户运行。建议覆盖：

1. time-v2 与 depth-v2 reader 分派；
2. 错 schema/domain 明确失败；
3. `time + synthoseis_lite_v1` 明确失败；
4. time-v2 canonical/probe/variant 可加载；
5. GINN patch smoke 读取 time/depth v2 的 N 点数组和 physics mask；
6. depth-v2 样本严格要求 N 点 observed/model-consistent/mask；
7. `split_policy=derive` 忽略 benchmark 普通 split，held-out geometry 永远 test；
8. 时间和深度 CLI 的合法/非法 suite 与参数组合。
