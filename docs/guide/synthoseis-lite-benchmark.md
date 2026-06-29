# 旁路：Synthoseis-lite 合成基准

Synthoseis-lite 当前有两条受支持分支：

- 时间域 `synthoseis_lite_v1`：历史时间工区流程，保留 canonical、Hz probe、seismic variant 和 evaluate。
- 深度域 `synthoseis_lite_v2`：TVDSS 米制 field-conditioned 流程，详细契约见 [`depth-synthoseis-lite-v2.md`](../spec/depth-synthoseis-lite-v2.md)。

入口脚本保持单一：

```powershell
python scripts/synthoseis_lite.py --config <yaml> calibrate
python scripts/synthoseis_lite.py --config <yaml> generate --suite <suite> --impedance-calibration <json>
```

配置必须显式声明分支，不再根据是否存在 `workflow_config` 猜测：

```yaml
synthoseis_lite:
  sample_domain: time | depth
  benchmark_schema: synthoseis_lite_v1 | synthoseis_lite_v2
```

`SynthoseisBenchmark` 先读取 manifest，再按 schema/domain 分派 reader：

- `synthoseis_lite_v1` 只作为时间域 reader；
- `synthoseis_lite_v2` 当前只作为 depth/TVDSS reader；
- 错 schema/domain 组合直接失败，不静默 fallback；
- 不提供旧产物升级或坏产物修复迁移。

## 时间域 v1

时间域 v1 支持两种配置形态：历史直接 workflow 配置，以及推荐的 common-overlay 形态。

推荐形态：

```yaml
workflow_config: experiments/common/common.yaml

synthoseis_lite:
  sample_domain: time
  benchmark_schema: synthoseis_lite_v1
```

`seismic.domain: time` 属于 workflow 真相，应写在 `experiments/common/common.yaml`
的 `seismic` 段里；time 实验配置不应覆盖地震文件、资产、层位或井曲线等 workflow 字段。

历史直接配置示例见：

```text
experiments/synthoseis_lite/synthoseis_lite.yaml
```

无论哪种形态，都必须包含：

```yaml
synthoseis_lite:
  sample_domain: time
  benchmark_schema: synthoseis_lite_v1
```

时间域 v1 仍依赖 Step 4/5/6 的时间域来源，保留：

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

时间域 reader 保留旧 N/N−1 对齐语义：目标、LFM 和 mask 是模型 N 点，Robinson
正演地震是 N−1 点；GINN adapter 显式丢弃首个模型样点完成对齐。统一
`forward_time` 仍保持 N 点契约，时间 v1 兼容层在需要旧 Robinson 语义的位置显式取
`[1:]`。

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

深度 v2 的 generator 只冻结父样本身份和评估角色：

- `parent_realization_id`；
- `evaluation_role=development_pool|geometry_holdout`；
- mismatch variant 必须继承 parent 的 `evaluation_role`。

生成端不写 train/validation/test 比例。GINN 训练端在 `split_policy=derive` 下按
`parent_realization_id` 哈希派生 train/validation/test；`geometry_holdout` 永远进入
test。改变训练比例只重建 patch index 和 normalization，不重新生成 HDF5 benchmark。

## Reader 与 GINN v2 接缝

`SynthoseisBenchmark` facade 保留：

- `sample_ids()`；
- `row()`；
- `load_sample()`。

domain reader 返回明确采样轴：

- time-v1：TWT/time 轴；
- depth-v2：TVDSS 轴。

GINN 通过训练 adapter 获取已对齐数组：

- time adapter 保留 N/N−1 对齐；
- depth adapter 使用 N/N 和 physics mask；
- `PatchDataset` 不无条件读取 depth-only 字段；
- 深度 physics loss 在专项实现前继续拒绝非零权重。

## 验证边界

本仓库只提交测试，测试由用户运行。建议覆盖：

1. time-v1 与 depth-v2 reader 分派；
2. 错 schema/domain 明确失败；
3. time-v1 canonical/probe/variant 可加载；
4. GINN patch smoke 不要求 time 样本具有 depth physics 字段；
5. depth-v2 样本严格要求 N 点 observed/model-consistent/mask；
6. `split_policy=derive` 忽略 benchmark 普通 split，held-out geometry 永远 test；
7. 时间和深度 CLI 的合法/非法 suite 与参数组合。
