# 合成基准生成与评估

Synthoseis-lite 是时间域、深度域共用的合成基准旁路。入口根据配置选择域 Adapter；校准、场景计划、父实现事务、视图编排、接受率、索引和 manifest 由同一共享 Pipeline 完成。

当前产物合同为 `synthoseis_lite_v5`，科学合同为 `synthoseis_lite_science_v3`。配置和产物只接受这一版合同。

## 快速开始

```powershell
python scripts/synthoseis_lite.py --config <config-yaml> calibrate
python scripts/synthoseis_lite.py --config <config-yaml> generate `
  --impedance-calibration <calibration-run>/impedance_calibration.json
```

深度域和时间域使用相同命令；域、单位和正演执行由配置与 Adapter 决定。

调试参数可使用 `--debug-attempt-limit <N>`、`--geometry-family <name>`、`--qc-only` 和 `--output-dir <path>`。

## 上游与域差异

两种域都需要校准产物、测井统计来源、解释层位和剖面路径。时间域使用秒制采样轴和时间正演；深度域使用 TVDSS 米制采样轴、冻结的 AI–Vp 关系和深度正演。深度域的 xline 坐标按真实线号差值解释，不能把数组下标当作一个线号步长。

配置的最小版本链如下：

```yaml
synthoseis_lite:
  sample_domain: time       # 或 depth
  benchmark_schema: synthoseis_lite_v5
  science_revision: synthoseis_lite_science_v3
  seismic_input:
    policy: observed_highres_forward
  seismic_views:
    operators: {}
    views: []
```

时间域还要求高分辨率正演 QC；深度域还要求深度正演输入旁路。公共配置使用占位符路径，不在教程中记录具体工区或用户机器路径。

## 原子地震视图

一个父实现只拥有一份 truth、target、canonical background、target increment、base seismic 和 mask。base 是未施加算子的理想正演，不是一个空视图。失配输入是父实现上的有序 seismic view。

配置由算子目录和有序视图列表组成：

```yaml
seismic_views:
  operators:
    gain:
      kind: global_gain
      log_sigma: 0.15
    noise:
      kind: additive_white_noise
      rms_fraction: 0.05
  views:
    - view_id: gain
      operator_ids: [gain]
    - view_id: gain_then_noise
      operator_ids: [gain, noise]
```

允许的算子包括相位旋转、子波时移、深度静校正、全局/逐道/横向增益、白噪声和有色噪声。前向参数算子必须连续位于列表开头；随后才是采样地震算子。前向参数汇总后只重新正演一次，再按列表顺序施加采样算子。

空算子目录和空视图列表表示只生成 base。未声明的视图不计算，也不写入 HDF5。视图身份由规范化配置、科学合同和随机流合同共同计算 `view_spec_sha256`；随机系数不依赖视图列表顺序或其他视图是否存在。

## LFM 与 HDF5

每个父实现只写入 canonical background。训练 reader 直接把它作为 `input_lfm_log_ai`，并校验：

```text
model_target_log_ai = canonical_background_log_ai + target_increment_log_ai
```

低频背景采用唯一的 canonical background。视图只物化自身的地震、算子 trace 和 QC；父实现集中保存 truth、target、canonical background 和 mask。

HDF5 的主要结构为：

```text
/realizations/<realization_id>/
  truth/model_target_log_ai
  priors/canonical_background_log_ai
  targets/target_increment_log_ai
  seismic/seismic_observed
  seismic/seismic_model_consistent
  masks/valid_mask
  seismic_views/<view_id>/seismic_observed
```

## 产物索引与事务

`realization_index.csv` 一行一个成功父实现；`seismic_view_index.csv` 一行一个成功物化视图，base 不进入视图索引。索引只记录完整发布的行，失败 attempt 进入拒绝记录。

父实现的 base 和全部声明视图在一个事务中提交。任一视图失败，当前父实现整体拒绝且不保留半成品；其他 attempt 可以继续。reader 按 v5 合同提供父实现和视图两层读取接口。

## 评估与检查

生成完成后，先检查 `benchmark_manifest.json` 的版本链、域、单位、接受率和 fingerprint；再检查两个索引的唯一键和 HDF5 路径。`realization_index.csv` 的父实现数与 HDF5 父目录一致，视图索引的每个联合键都应指向对应父目录下的视图。

场景接受率仍用于发现几何与统计模型之间的冲突。它是 QC 证据，不是自动调整算子强度或训练权重的机制。合成地震输入的失配数值是当前实验配置，不应表述为真实工区反演得到的物理分布。

## 与 GINN v2 的连接

GINN v2 读取父实现和视图双索引。训练配置在具体 synthetic loss block 内声明父实现权重、base/variant 权重和 view 权重；新增未被引用的视图不会改变既有父实现概率。训练与验证的 split、normalization 和 checkpoint provenance 由 GINN 实验套件单独拥有。
