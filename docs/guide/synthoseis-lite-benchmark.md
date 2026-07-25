# 合成基准生成与检查

Synthoseis-lite 为时间域和深度域提供同一套结构化合成基准。两个域共享场景计划、真值生成、父实现事务、索引和发布流程；域适配层负责采样轴、正演输入和正演实现。

每次正式生成只发布一个数据产物：

```text
synthetic_benchmark.h5
```

表格、清单和图片是该产物的索引与报告。

## 快速开始

先冻结阻抗校准：

```powershell
python scripts\synthoseis_lite.py `
  --config <config-yaml> `
  --output-dir <calibration-run> `
  calibrate
```

再生成结构化基准：

```powershell
python scripts\synthoseis_lite.py `
  --config <config-yaml> `
  --output-dir <generation-run> `
  generate `
  --impedance-calibration <calibration-run>\impedance_calibration.json `
  --run-structured-oracle
```

调试时可限制 attempt 和几何类型：

```powershell
python scripts\synthoseis_lite.py `
  --config <config-yaml> `
  --output-dir <smoke-run> `
  generate `
  --impedance-calibration <calibration-run>\impedance_calibration.json `
  --structured-smoke
```

## 配置合同

最小版本链如下：

```yaml
synthoseis_lite:
  sample_domain: depth
  benchmark_schema: structured_synthetic_benchmark_v1
  science_revision: synthoseis_lite_science_v4
  seismic_input:
    policy: observed_highres_forward
```

时间域使用秒制采样轴和时间正演。深度域使用 TVDSS 米制采样轴、冻结的阻抗—速度关系和深度正演。

inline 和 xline 表示几何身份。横向物理距离使用米制的 lateral axis。工区 xline 步长从 survey geometry 读取。

## 数据内容

每个父实现保存：

- clean seismic；
- LFM；
- model-grid valid mask；
- high-resolution impedance truth；
- state、object、zone 和 boundary 网格；
- zone 背景参数；
- segment 的三阶段对象系数；
- 投影结果；
- model-consistent seismic；
- forward support 和 forward context。

对象 profile 为：

```text
profile = c0 + c1 * (2ξ - 1) + c2 * sin(πξ)
```

zone 背景为：

```text
background = a + b * (2ζ - 1)
```

LFM 是带采样轴、合法 mask 和来源身份的 observation。

## HDF5 结构

```text
/realizations/<realization_id>/
  identity/
  axes/
  observed/
    seismic
    lfm
    valid
  truth/
    log_ai_highres
    model_log_ai
    zones/
    segments/
  forward/
    model_consistent_seismic
    support/
    context/
    domain_extras/
  qc/
```

zone 和 segment 使用列式数据集。高分辨率 truth、模型网格 observation 和 segment supervision 各自使用独立合法性语义。

## 事务与索引

父实现先写入 HDF5 staging group。字段、shape、axis、mask、主键和 segment 端点校验通过后，完整父实现移动到正式 realization group。

```text
realization_index.csv
```

每行索引一个成功父实现，并给出 observation、LFM、truth、forward 和 mask 的 HDF5 路径。

生成结束后的发布检查确认：

- 索引与 HDF5 parent 集合一致；
- 每个 parent 标记 complete；
- staging group 为空；
- root artifact identity 正确；
- disk Oracle 按请求通过。

## 波形扰动

canonical benchmark 保存 clean seismic。训练时的相位、时移、静校正、增益和噪声扰动由模型实验配置在线生成。

在线扰动必须记录算子规格和随机身份，并保持 LFM、truth、采样轴和 mask 语义不变。

## 20260725 数据迁移

一次性迁移使用：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python scripts\migrate_synthoseis_structured_v1.py `
  --source-run experiments\synthoseis_lite\results\20260725\generate_field_conditioned `
  --output-dir experiments\synthoseis_lite\results\20260725\generate_field_conditioned_structured `
  --oracle-parent-count 3
```

中断后在相同命令中增加：

```text
--resume
```

迁移按 parent 提交，并对旧 HDF5 与结构化 sidecar 做字段级一致性检查。
