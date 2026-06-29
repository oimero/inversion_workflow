# Synthoseis 时间/深度链条修复文档

## Summary

`eba7179 synthoseis depth` 基于 `docs/spec/depth-synthoseis-lite-v2.md` 引入了深度域 Synthoseis-lite v2，但同时把原本的时间域 Synthoseis-lite v1 读数、评估和 GINN 数据入口挤成了 depth-only 路径，导致时间域合成数据阶段链条断裂。

本文件记录兼容性断点、架构决策、分阶段修复方案和当前落地状态。

核心决策：

- 时间域 v1 流程作为正式分支继续保留，未来另行设计 time-v2。
- `SynthoseisBenchmark` 按明确 schema/domain 分派 time-v1/depth-v2 reader。
- 保留统一 `forward_time` 的 N 点契约；时间 v1 adapter 显式取 `[1:]` 保持旧 Robinson 语义。
- 删除生成端 train/validation/test 比例；生成端只冻结 parent 和 held-out geometry 角色，训练端派生 split。
- `v2_*` 重组为 `time/`、`depth/` 和 domain-neutral `core/`，版本号只属于产物 schema。

## Implementation status

截至 2026-06-30，已落地：

- `SynthoseisBenchmark` 已改为 schema/domain facade，分派到 `readers/time_v1.py` 与 `readers/depth_v2.py`。
- time-v1 reader 已恢复，现有 `20260625` time benchmark 可读取 base、frequency probe、seismic variant 和 frequency-probe seismic variant。
- GINN patch adapter 已恢复 time-v1 N/N−1 对齐，`PatchDataset` 不再无条件访问 depth-only physics 字段。
- CLI 已改为显式 `sample_domain` / `benchmark_schema` 分派；默认 time 配置已补 `seismic.domain=time`。
- depth-v2 generator 不再写 train/validation/test split，改写 `evaluation_role=development_pool|geometry_holdout`。
- GINN `split_policy=derive` 已改为按 parent hash 派生，并让 geometry holdout 永远进入 test。
- `v2_config.py`、`v2_calibration.py`、`v2_generation.py` 已迁入 `cup.synthetic.depth`。
- depth 对旧 object core 的 time 字段适配已集中到 `cup.synthetic.depth.object_core_adapter`；depth 主流程不再散落 `twt_model_s`、`truth_dt_s` 等桥接名。
- 已新增 `cup.synthetic.time` 与 `cup.synthetic.core` facade，作为后续物理迁移的稳定 import seam。
- guide 与 depth-v2 spec 已更新为当前 time/depth 双分支和 split 职责。
- 已补本地测试 `tests/test_synthoseis_time_depth_repair.py`；按项目约定由用户运行。

仍属后续演进：

- `cup.synthetic.time` 目前是兼容 facade；time-v1 生成/校准实现仍保留在既有 top-level time 模块，后续可在不改调用方的前提下物理迁移。
- `cup.synthetic.core` 目前是 object-model facade；真正 domain-neutral 的实现抽取仍是下一步演进。

## Compatibility breakpoints

当前断点集中在 reader、训练入口、CLI 分派和文档契约四处：

1. `src/cup/synthetic/dataset.py` 从 time-v1 reader 被替换成 depth-v2-only reader。
   - 旧 reader 读取 `synthetic_benchmark.h5` 和 `sample_index.csv`，支持 `base`、`frequency_probe`、`seismic_variant`、`frequency_probe_seismic_variant`。
   - 新 reader 要求 `benchmark_manifest.json`、`sample_domain=depth`、`depth_basis=tvdss`、`schema=synthoseis_lite_v2`，并拒绝旧 `synthoseis_lite_v1`。

2. `scripts/evaluate_synthoseis_lite.py` 和 `src/ginn_v2/data.py` 仍通过 `SynthoseisBenchmark` 进入数据，但现在会遇到 depth-only reader。

3. `src/ginn_v2/data.py` 已部分加入 depth 分支，但 `PatchDataset` 仍无条件访问深度字段，例如 `seismic_model_consistent` 和 `physics_valid_mask`；时间样本不应要求这些字段。

4. `scripts/synthoseis_lite.py` 当前用 `workflow_config` 是否存在来推断时间/深度分支。这个分派条件是配置形态耦合，不是领域契约。

5. `docs/guide/synthoseis-lite-benchmark.md` 被改成深度 v2 说明后，时间域 calibrate、canonical、probe、evaluate 的运行说明缺失。

6. `src/cup/synthetic/v2_config.py`、`src/cup/synthetic/v2_calibration.py`、`src/cup/synthetic/v2_generation.py` 以版本号组织实现，并在若干路径里把米制字段伪装成时间域字段再改名回来，暴露出 domain seam 不清晰。

## Repair principles

- 不回退整个 depth commit；深度 v2 是要保留的新增能力。
- 不把时间 v1 当作临时遗留路径；它是受支持流程，直到单独设计并迁移到 time-v2。
- 不继续用“是否存在某个配置字段”猜测 domain；domain 和 schema 必须显式声明。
- 不在生成端冻结训练比例；生成数据集只描述样本身份、变体关系和几何留出角色。
- 不靠 fallback 静默兼容错误；schema/domain 不匹配必须明确失败。

## Target public interfaces

配置分派键：

```yaml
synthoseis_lite:
  sample_domain: time | depth
  benchmark_schema: synthoseis_lite_v1 | synthoseis_lite_v2
```

Reader facade 保留稳定入口：

- `sample_ids()`
- `row()`
- `load_sample()`

Facade 负责先读 manifest/index，再按 schema/domain 选择 domain reader。domain reader 返回明确的采样轴：

- time-v1 reader 返回 TWT/time 轴与时间域样本字段。
- depth-v2 reader 返回 TVDSS 轴与深度域样本字段。

GINN 不直接猜 reader 的内部字段，而是通过训练 adapter 获取已对齐数组：

- time adapter 保留旧的 N/N−1 对齐语义。
- depth adapter 使用 N/N 对齐和 physics mask。
- 深度 physics loss 在专项实现完成前继续明确拒绝非零权重。

## Target module layout

目标结构：

```text
src/cup/synthetic/
  dataset.py              # facade: schema/domain dispatch only
  readers/
    time_v1.py            # time-v1 artifact reader
    depth_v2.py           # depth-v2 artifact reader
  time/
    calibration.py
    generation.py
    evaluation.py
  depth/
    config.py
    calibration.py
    generation.py
  core/
    random_streams.py
    statistical_models.py
    scenarios.py
```

迁移策略可以分两步：

1. 先把 depth-only reader 挪进 `readers/depth_v2.py`，恢复 time-v1 reader，并让 `dataset.py` 成为薄 facade。
2. 再把 `v2_config.py`、`v2_calibration.py`、`v2_generation.py` 迁入 `depth/`，把真正无量纲、domain-neutral 的对象生成、随机流和统计模型逐步下沉到 `core/`。

完成迁移后删除：

- `src/cup/synthetic/v2_config.py`
- `src/cup/synthetic/v2_calibration.py`
- `src/cup/synthetic/v2_generation.py`

## Implementation phases

### 1. 紧急恢复时间链

- 将当前 depth-only reader 移入 depth adapter，恢复 time-v1 reader。
- `SynthoseisBenchmark` 先读 manifest/index，再按 schema/domain 选择 reader。
- 修复 GINN 数据入口：
  - time adapter 保留 N/N−1 对齐；
  - depth adapter 使用 N/N 和 physics mask；
  - `PatchDataset` 不再无条件访问深度字段。
- 恢复时间域 calibrate、canonical、probe、evaluate 文档和 CLI 行为。
- 保留统一 `forward_time` 的 N 点契约；时间 v1 adapter 显式使用 `forward_time(...)[1:]` 保持旧 Robinson 语义。

### 2. 重组模块

- 保留单一 `scripts/synthoseis_lite.py` 入口。
- 配置显式声明 `sample_domain` 与 `benchmark_schema`，不再根据是否存在 `workflow_config` 猜分支。
- 深度实现迁入 `cup.synthetic.depth`。
- 时间实现迁入 `cup.synthetic.time`。
- 无量纲对象生成、随机流和统计模型逐步下沉到 `cup.synthetic.core`。
- 消除将米制字段伪装成 `truth_dt_s`、`twt_model_s` 再改名回来的实现。
- 完成迁移后删除 `v2_config.py`、`v2_calibration.py`、`v2_generation.py`。

### 3. 调整 split 职责

- generator 输出 `parent_realization_id` 和 `evaluation_role=development_pool|geometry_holdout`。
- mismatch variant 必须继承 parent 的 `evaluation_role`。
- GINN `derive` 按 parent 哈希和训练配置派生 train/validation/test 比例，忽略 benchmark 的普通 split。
- `geometry_holdout` 永远进入 test；改变 train/validation 比例只重建 patch index 和 normalization，不改变 HDF5。
- generator 不再因 split 为空失败，也不再写比例和 split assignment hash。

### 4. 文档收敛

- 将当前 `docs/guide/synthoseis-lite-benchmark.md` 改为时间/深度入口页，并拆分两套运行说明。
- 修订 `docs/spec/depth-synthoseis-lite-v2.md` 中“生成端冻结 split”和“只接受 depth-v2 reader”的条款。
- 明确 time-v1 是受支持流程，但不提供旧产物升级或修复迁移。
- 高级评估指标继续搁置，不纳入这份兼容性修复。

## Split contract

生成端只负责样本来源和评估角色，不负责训练比例。

生成端字段：

- `parent_realization_id`：同一个 parent 的 canonical、probe、variant 必须共享该 id。
- `evaluation_role`：
  - `development_pool`：训练端可按配置派生 train/validation/test。
  - `geometry_holdout`：永远作为 test，不进入训练。

训练端规则：

- `split_policy=derive` 时，按 `parent_realization_id` 哈希派生 split。
- 同一 parent 的所有 variant 不允许跨 split。
- `geometry_holdout` 无条件覆盖为 test。
- 调整 train/validation/test 比例只影响 patch index 和 normalization，不要求重新生成 benchmark。

## Test and acceptance plan

仅编写测试，由用户运行。

Reader 与 schema：

- time-v1 与 depth-v2 reader 可按 schema/domain 正确分派。
- schema/domain 不匹配时明确失败，不静默 fallback。
- time-v1 canonical/probe/variant 均可加载。
- 深度样本严格要求深度 physics 字段；时间样本不需要深度 physics 字段。

时间链烟测：

- `20260625` 时间 benchmark 可完成 reader smoke。
- `20260625` 时间 benchmark 可完成 evaluator smoke。
- `20260625` 时间 benchmark 可完成 GINN patch smoke。

Forward contract：

- `forward_time(...)[1:]` 与旧 Robinson fixture 逐点一致。
- 时间 v1 adapter 使用 `[1:]` 的位置有测试覆盖。

Split：

- 调整训练比例不改变 HDF5，只改变 patch index。
- 同一 `parent_realization_id` 的所有 variant 不跨 split。
- `geometry_holdout` 不进入训练。
- `split_policy=derive` 忽略 benchmark 的普通 split。

CLI：

- 时间 CLI 的合法/非法 suite、参数组合有覆盖。
- 深度 CLI 的合法/非法 suite、参数组合有覆盖。
- 缺失或冲突的 `sample_domain` / `benchmark_schema` 明确报错。

## Assumptions

- 修复目标仓库为当前 workspace `xihu_workflow_standardlize`。
- 当前 workspace 与 `mero_workflow_standardlize` 均位于 `eba7179`。
- 不回退整个 depth commit。
- time-v1 正式保留到另行迁移；本修复不承诺旧产物升级或历史坏产物修复。
- 高级评估指标不属于这份兼容性修复文档。
