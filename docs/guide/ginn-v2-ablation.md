# GINN v2 模型训练与消融

GINN v2 将架构、数据源、损失块、阶段和归一化参考拆成可组合模块。合成基准使用 Synthoseis-lite v5 的父实现与 seismic view；真实工区和真实井监督保持各自的 Adapter。

## 合成数据源

```yaml
sources:
  synthetic:
    kind: synthoseis_lite
    benchmark_dir: auto
```

合成 source 只声明基准目录。具体输入视图属于 loss block 的采样合同，不属于 source。

## 父实现均衡视图抽样

每个训练 item 的选择顺序固定为：父实现、base/variant 类型、view、patch。配置使用显式概率：

```yaml
sampling:
  kind: parent_balanced_seismic_view
  parent_weights: {base: 0.5, variant: 0.5}
  view_weights:
    gain: 0.5
    noise: 0.5
```

父实现权重必须包含 base 和 variant 并归一化；variant 权重大于零时，view 权重必须只包含已物化视图并归一化。clean-only 使用 variant 为零且 view 权重为空。抽样器按父实现均衡，不按索引行数推导概率；增加未被引用的视图不会改变已有抽样序列。

## 实验 split 与 normalization

benchmark 只发布 `evaluation_role`。GINN 在构建 patch catalog 前按 synthetic source 写出 `split_assignment_<source_id>.csv`，使用固定的父实现 hash 合同：

```yaml
split_contract:
  version: parent_hash_split_v1
  owner: ginn_v2_experiment_suite
  seed: 20260714
  hash_algorithm: sha256
  validation_fraction: 0.15
  test_fraction: 0.15
  geometry_holdout_role: test
```

patch catalog 一行表示父实现上的一个窗口，不因视图数量复制。normalization 只从 train 父实现的 base 地震和 canonical background 计算，并保存 identity。所有归因组复用同一 normalization identity。

## 固定验证合同

验证配置与训练配置分开：

```yaml
validation:
  selection_metric: synthetic_increment.weighted_mse
  seismic_views:
    parent_weights: {base: 0.5, variant: 0.5}
    view_weights:
      gain: 0.5
      noise: 0.5
```

验证先在相同父实现、相同 patch 上计算 base 与逐视图指标，再按显式权重聚合。训练 history 和 checkpoint provenance 保存 base、逐视图、paired 变化、计数和权重；checkpoint 选择使用 weighted metric，不用索引行平均。

## 真实井监督角色

真实井 source 可以声明监督排除名单。名单中的井保留逐井预测、曲线、井指标和 sampling QC，但不进入 sampler、训练损失、held-out MSE 或 checkpoint 选择。`held_out_well` 必须是名单外的另一口存在井；名称不存在、重复或重叠时配置直接失败。同簇排除是空间隔离角色，与 configured supervision exclusion 和 held-out 角色分开。

## 训练阶段与消融

一个阶段包含 optimizer、loss blocks、采样合同和验证合同。合成监督 block 的 target 始终是父实现的 canonical increment；physics block 的 target 始终是 nominal model-consistent seismic。真实工区 physics 不改变合成父实现的标签语义。

深度域归因可固定架构、父实现集合、split、seed、预算和 normalization，分别运行 clean-only、amplitude、noise、operator 和 all 五组。五组共享统一全视图验证分布；最终保留哪些活动视图由人工根据逐视图指标、paired 变化和物理合理性裁决。

## 主要产物

实验目录包含 `experiment_manifest.json`、`normalization.json`、每个 synthetic source 的 split assignment、每阶段 patch index、training history、best/final checkpoint 和验证诊断。checkpoint 至少记录 benchmark v5 identity、science/view/operator/random 合同、双索引、split identity、normalization identity、训练/验证权重和实际采样计数。

## 检查清单

- 配置使用 v5 benchmark 与父实现视图采样；
- split assignment 在 patch 构建前生成且没有缺失父实现；
- normalization 只来自 base/canonical background；
- 训练和验证 view 权重均显式归一化；
- history 中的 weighted metric 可以由逐视图指标和权重复算；
- checkpoint 恢复时 benchmark、视图、split 和权重 identity 一致；
- 深度域纵轴保持 TVDSS 米制，横向真实坐标保持 xline 步长语义。
