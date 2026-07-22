# 合成基准生成与评估

Synthoseis-lite 是时间域、深度域共用的合成基准旁路。入口根据配置选择域 Adapter；校准、场景计划、父实现事务、视图编排、接受率、索引和 manifest 由同一共享 Pipeline 完成。

当前产物合同为第五版，科学合同为第四版。配置和产物只接受当前合同。

## 快速开始

```powershell
python scripts/synthoseis_lite.py --config <config-yaml> calibrate
python scripts/synthoseis_lite.py --config <config-yaml> generate-amplitude-pilot `
  --impedance-calibration <calibration-run>/impedance_calibration.json
python scripts/synthoseis_lite.py --config <config-yaml> calibrate-amplitude `
  --impedance-calibration <calibration-run>/impedance_calibration.json `
  --pilot-benchmark <pilot-run>
python scripts/synthoseis_lite.py --config <config-yaml> generate `
  --impedance-calibration <calibration-run>/impedance_calibration.json `
  --seismic-amplitude-prior <amplitude-run>/seismic_amplitude_prior.json
```

标定振幅视图启用时执行完整的四段流程；未配置该视图时仍可只运行阻抗标定和正式生成。深度域和时间域使用相同命令；域、单位和正演执行由配置与 Adapter 决定。

调试参数可使用 `--debug-attempt-limit <N>`、`--geometry-family <name>`、`--qc-only` 和 `--output-dir <path>`。

## 上游与域差异

两种域都需要校准产物、测井统计来源、解释层位和剖面路径。时间域使用秒制采样轴和时间正演；深度域使用 TVDSS 米制采样轴、冻结的 AI–Vp 关系和深度正演。深度域的 xline 坐标按真实线号差值解释，不能把数组下标当作一个线号步长。

配置的最小版本链如下：

```yaml
synthoseis_lite:
  sample_domain: time       # 或 depth
  benchmark_schema: synthoseis_lite_v5
  science_revision: synthoseis_lite_science_v4
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

允许的算子包括相位旋转、子波时移、轴向静校正、全局增益、逐道增益、标定层序增益、白噪声和有色噪声。沿原始垂向轴变化的增益实现只保留为非地质坐标压力测试，不进入默认 benchmark。前向参数算子必须连续位于列表开头；随后才是采样地震算子。前向参数汇总后只重新正演一次，再按列表顺序施加采样算子。

## 地震振幅先验

时间域和深度域共用同一个二维粗振幅场估计、残差分解、先验发布、加载校验和视图解析实现。域适配层只负责读取本域真实剖面、提供横向距离、把层位映射到无量纲层序坐标，以及调用本域正演生成无视图 pilot。

真实侧和合成侧都先对剖面等权；同一剖面内的场景等权，同一场景内成功生成的样本再等分该场景权重。因此，不同场景的接受率不会改变标定目标。每张剖面先形成横向距离与层序坐标上的粗尺度对数均方根场，并去掉整体中位数，因此先验不恢复绝对地震增益。

粗网格中有足够原始样本的单元才参与均值、残差和协方差估计；空白单元的插值仅用于形成连续可执行的模板。产物同时记录每张场的支持比例，以及真实侧与合成侧的共同覆盖比例。

共享估计把粗振幅残差分为层序分量、横向分量和二者的交互分量。前两项的随机强度由真实方差减去 pilot 方差后取非负平方根；交互项对真实与 pilot 的协方差差做正半定截断，并由解释方差确定模式数。相关长度估计不稳定或额外方差为零的分量会在先验中禁用，禁用分量的公开随机强度恒为零。均值模板与三倍残差标准差之和受统一的总对数增益上限约束；超出时按同一比例收缩全部随机残差，并保留收缩前统计量。默认 benchmark 包含均值、随机残差和完整先验三个可归因视图；三者通过同一个算子完成一次规范校正、裁剪和指数映射。

pilot 保存完整兼容合同，覆盖域和轴、工区地震身份、层位内容、剖面路径、采样、场景、正演输入及阻抗标定身份。振幅先验拟合和正式 benchmark 生成都会重新计算该合同；只有摘要一致的 pilot、先验和正式生成配置能够连接。先验使用临时目录完成自校验后原子发布。

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

单个实现无法获得完整正演上下文、随机层序无法成立、有效正演样点不足，或者某个视图无法覆盖该实现的公共有效区时，均作为当前实现的拒绝原因记录。此类情况不终止预检或生成循环。

## 评估与检查

生成完成后，先检查 `benchmark_manifest.json` 的版本链、域、单位、接受率和 fingerprint；再检查两个索引的唯一键和 HDF5 路径。`realization_index.csv` 的父实现数与 HDF5 父目录一致，视图索引的每个联合键都应指向对应父目录下的视图。

场景接受率仍用于发现几何与统计模型之间的冲突。它是 QC 证据，不是自动调整算子强度或训练权重的机制。合成地震输入的失配数值是当前实验配置，不应表述为真实工区反演得到的物理分布。

接受率、覆盖率、裁剪比例和经验指标只产生质量告警。低接受率或某个场景没有成功父实现时，其他完整父实现仍然发布；带告警完成的正式产物可供后续标定、训练和评估消费。只有配置、数据结构、域与单位、版本身份、文件引用、有限性或数学可计算性不成立时，整次运行才失败。意外失败会保留临时运行目录和失败记录，已完成父实现不会因末尾质量判断被删除。

## 与 ablation 的连接

ablation 读取父实现和视图双索引。训练配置在具体 synthetic loss block 内声明父实现权重、base/variant 权重和 view 权重；新增未被引用的视图不会改变既有父实现概率。训练与验证的 split、normalization 和 checkpoint provenance 由 GINN 实验套件单独拥有。
