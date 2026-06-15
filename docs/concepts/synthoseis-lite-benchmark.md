# Truth-First `synthoseis-lite` 基准

## 文档地位

本文是[前向可观测性闸门](forward-observability-gate.md)之后的第二个研究闸门，也是当前
第五步以后研究工作的活动入口。

本闸门以评测为先：先建立冻结、可复现、已知真值的二维合成基准，再用它判断逆问题的
实际可恢复性。它不选择神经网络架构，不立即建设大规模训练集，也不把某次工区诊断得到
的频率固化为模型边界。

未来实现使用不带步骤编号的入口：

- CLI：`scripts/synthoseis_lite.py`
- 核心包：`src/cup/synthetic/`
- schema：`synthoseis_lite_v1`

稳定生产链仍终止于第五步。本研究入口不是“第六步”。

## 1. 闸门回答的问题

本基准需要区分三个问题：

1. 在当前离散正演、子波和噪声条件下，给定频率和振幅的阻抗扰动是否可恢复。
2. 模型是否能恢复薄层、楔状体、尖灭和倾斜层，而不是只会拟合规则正弦探针。
3. 模型在低频先验、子波、gain、相位、时移和噪声失配下是否仍保持可信。

前向可观测性闸门只分析“阻抗扰动能否产生可辨识地震响应”。本闸门进一步要求模型从
地震和先验中恢复已知阻抗真值。两者都不能单独解释为永久的反演 cutoff。

## 2. 输入与来源约束

### 2.1 显式运行目录

每次运行必须显式指定：

- `forward_observability_dir`：第一研究闸门运行目录。
- `wavelet_generation_dir`：第五步运行目录。
- `well_auto_tie_dir`：第四步运行目录。
- `well_preprocess_dir`：第三步运行目录。

禁止搜索 `latest`、按修改时间猜测目录、扫描子波目录或在缺失时回退到其他运行。
第一闸门 `run_summary.json.source_runs` 中记录的第三、四、五步目录必须与本次显式输入
逐项指向同一路径，否则整次运行失败。

最小配置形态为：

```yaml
synthoseis_lite:
  source_runs:
    forward_observability_dir: <observability-run>
    wavelet_generation_dir: <step-5-run>
    well_auto_tie_dir: <step-4-run>
    well_preprocess_dir: <step-3-run>
  sampling:
    output_dt_s: 0.002
    vertical_oversampling_factor: 8
  geometry:
    canonical:
      enabled: true
    field_conditioned:
      enabled: true
      horizons:
        - name: top_a
          file: interpre/top_a
        - name: marker_b
          file: interpre/marker_b
        - name: base_c
          file: interpre/base_c
      sections:
        - section_id: example_section
          path:
            - {inline: 300.0, xline: 900.0}
            - {inline: 500.0, xline: 1200.0}
  splits:
    held_out_geometry_family: pinchout
```

示例名称和路径不属于默认值。实现必须接受任意 `N >= 2` 个有序解释层位，不得在变量名、
schema、场景生成或测试中内置当前工区的层位名称和数量。

### 2.2 井曲线的角色

第三步全频曲线和第四步 filtered LAS 只用于估计以下稳健范围：

- 目标窗内 `log(AI)` 背景及趋势。
- 相邻层或状态之间的阻抗对比度。
- 可见层厚和持续长度。
- 第四步条件化前后的幅度差异。
- 井震残差的 RMS、频谱形态和相关长度。

它们不是无噪地质真值。生成器不得复制某口井的完整曲线、残差波形或局部 patch 作为
synthetic truth。校准统计应按空间簇先聚合，再跨簇取稳健分位数，避免密井平台重复加权。

井统计只限定 field-like 随机场景的合理范围。可控频率探针仍须按第 6 节的振幅阶梯主动
覆盖低于、接近和高于现实失配底的情况，不能被当前小样本井分布截断。

## 3. 双套二维场景

所有样本均为 `lateral x TWT` 二维剖面。横向轴保存物理距离和其来源坐标；时间轴使用
正秒 TWT。首版不生成三维小体。

### 3.1 Canonical suite

Canonical suite 是固定、公开、完全参数化的评测集，包括：

- `horizontal_thin_beds`：水平薄互层。
- `wedge`：厚度单调变化并跨越调谐尺度的楔状体。
- `pinchout`：具有已知终止位置的尖灭。
- `dipping_layers`：连续倾斜层。
- `lateral_impedance_change`：几何不变、阻抗对比横向变化。
- `frequency_probe`：Tukey 加窗的正弦和余弦 `log(AI)` 扰动。

每个场景都有解析或构造时已知的界面位置、层厚、终止位置、扰动频率、相位和振幅。
Canonical 样本不得进入后续随机训练抽样，只用于基准验证与模型评测。

### 3.2 Field-conditioned suite

Field-conditioned suite 从显式配置的解释层位和二维路径构造大尺度地层框架：

1. 读取每个 `HorizonSurface`，校验 domain、单位和支持范围。
2. 沿配置路径按物理横向距离规则采样所有层位。
3. 校验每个横向位置的层位顺序严格从浅到深。
4. 在相邻层位之间建立归一化 RGT/层序坐标。
5. 在该坐标中生成新的层序、层厚和阻抗属性。

路径可以是 inline、xline 或任意折线，不限定方向。生成器只使用解释层位的大尺度几何，
不读取或复制路径上的真实地震纹理。

层位缺失、交叉、局部无支持或路径越界时，应拒绝对应 section 或 realization，并写出
明确状态，不能使用最近层位、其他路径或默认平层替代。

### 3.3 随机层序与 Semi-Markov

Semi-Markov 只用于随机地质套件，沿 RGT 方向生成：

- `low_impedance`
- `background`
- `high_impedance`

状态只表示抽象阻抗关系，不宣称砂、泥或具体岩相。Semi-Markov 负责：

- 初始状态。
- 状态转移。
- 每个状态的持续层厚。
- 层序中不同阻抗状态的组合。

它不直接生成带通残差、波峰、波谷或地震纹理。横向相关随机场负责缓慢调制层厚、阻抗
对比度和尖灭位置；同一父 realization 内的相邻道共享地层对象，不能逐道独立抽样状态。

高频残差只能在真值生成后按以下形式派生：

```text
derived_residual = truth_log_ai - degraded_target_log_ai
```

它不是生成器的原始随机变量。

## 4. 真值网格与抗混叠

### 4.1 两套垂向网格

默认配置：

```text
output_dt = 0.002 s
vertical_oversampling_factor = 8
truth_dt = output_dt / 8 = 0.00025 s
```

所有层位、界面、薄层和尖灭先在 `truth_dt` 网格构造。实现必须同时保存高分辨率真值和
2 ms 工程目标：

- `truth_log_ai_highres`：高分辨率构造真值。
- `target_log_ai_2ms`：抗混叠后落到 2 ms 轴的主模型目标。

首版模型评测以 `target_log_ai_2ms` 为准，不要求模型从 2 ms 地震直接输出 0.25 ms
超分辨率曲线。高分辨率真值用于界面、层厚、部分体素和正演 QC。

### 4.2 降采样

高分辨率阻抗、RGT 和连续属性在降采样前必须使用显式低通抗混叠滤波，不允许简单
`array[..., ::factor]` 抽取。离散状态和边界 mask 使用适合其语义的占比、最近状态或
覆盖率表达，不把类别编号做普通线性低通。

2 ms 反射系数不能由高分辨率反射系数直接抽样获得。必须分别从对应采样轴上的
`log(AI)` 重新计算，以保持离散正演定义清楚。

### 4.3 上下文

每个目标窗上下至少额外生成半个 nominal 子波长度的高分辨率上下文。所有子波场景均在
完整上下文上卷积，之后才裁取目标窗。若扰动子波比 nominal 更长，则按场景中的最大
半长度建立上下文。

## 5. 统一真值派生

一个 realization 先生成完整 `log(AI)` 真值，再从同一真值派生全部输入与标签：

```text
high-resolution geometry and log(AI) truth
  -> exact reflectivity
  -> wavelet convolution
  -> anti-aliased 2 ms seismic
  -> ideal and field-like LFM
  -> RGT, zone, validity and boundary masks
  -> derived residual targets
```

### 5.1 正演约定

正演与前向可观测性闸门保持一致：

```text
r[j] = tanh((logAI[j] - logAI[j-1]) / 2)
```

反射系数挂在下部样点，子波为奇数长度且零时刻位于中心样点，使用
`numpy.convolve(wavelet, reflectivity, mode="same")`。未来实现应复用同一个公开正演
核心，不能在两个研究闸门中维护数值上略有不同的副本。

保存 nominal 子波场景及第五步准入候选。人工失配场景至少包括：

- 白噪声。
- 具有可配置相关长度或谱形的有色噪声。
- 正 gain 扰动。
- 常相位旋转。
- 正负亚采样时移。

第四步现实残差仅用于校准噪声 RMS、谱包络和相关长度。不得把真实残差 patch 直接叠加
到合成地震，因为其中混合了标定误差、未建模地质和噪声。

### 5.2 两类低频先验

每个 realization 同时派生：

- `lfm_ideal`：由真值通过明确低通和降采样得到。
- `lfm_field_like`：在 ideal LFM 上注入可审计的趋势、幅度和横向平滑偏差。

两者共享同一基础 cutoff 配置，但 field-like 误差参数单独记录。评测必须分别报告二者，
不能只在理想低频模型上评价。

## 6. 第一闸门驱动的探针矩阵

### 6.1 频率选择

只读取第一闸门 `whole_target` 的逐频结果。选择：

- 所有 `conditional` 频率。
- 所有 conservative operator 为 `core` 且经验状态为 `not_detectable` 的频率。
- conservative operator 为 `weak` 或 `unsupported` 的频率，作为压力测试或负对照。

实现不得硬编码 20 Hz、35 Hz、55 Hz 或任何当前工区结果。所选频率及其来源状态完整写入
`scenario_catalog.csv` 和 manifest。

### 6.2 Noise-equivalent 基准

每个频率的参考振幅从第一闸门 `well_frequency_sensitivity.csv` 计算：

1. 只使用 `whole_target`、nominal 子波且 `status=ok` 的记录。
2. 在同一空间簇内对 `noise_equivalent_log_ai` 取中位数。
3. 跨空间簇再取中位数，得到 `reference_noise_equivalent_log_ai`。
4. 至少需要三个有效空间簇；不足时该频率仅生成 `0x` 负对照，并标记
   `insufficient_noise_equivalent_calibration`。

每个可校准频率使用：

```text
0, 0.25, 0.5, 1, 2, 4
```

倍的 reference noise-equivalent 振幅，并分别生成正弦和余弦两组正交相位。振幅是
`log(AI)` 的加权 RMS，不是峰值或 AI 绝对值。

`0x` 样本用于测量模型凭先验或架构虚构目标频率能量的倾向。weak/unsupported 样本主要
用于负对照，不得因模型在这些频率偶然得到低误差就将其自动升级为生产频带。

## 7. 数据拆分与防泄漏

### 7.1 父 realization

随机地质样本先生成完整父 realization，再裁取 patch。一个父 realization 只能属于一个
split；其相邻 patch、噪声变体、LFM 变体和子波变体不得跨 split。

训练、验证和测试使用独立 seed 空间。seed 由 manifest 明确记录，不能依赖 Python
进程级随机状态或生成顺序。

### 7.2 几何家族留出

除 realization 隔离外，默认把 `pinchout` 作为未见几何测试家族。该选择必须来自配置并
写入 manifest，代码中不得假定被留出的永远是尖灭。

留出家族的所有 realization 只能进入测试集。其他随机场景按父 realization 分配到
train/validation/test。Canonical suite 独立标记为 `benchmark`，不属于上述随机比例。

## 8. 输出契约

本节定义未来 schema，但本轮不把它加入稳定工作流的
[核心 CSV 契约](csv-contracts.md)。

### 8.1 `synthetic_benchmark.h5`

每个父 realization 使用独立 group。至少保存：

```text
/realizations/<realization_id>/
  axes/
    lateral_m
    twt_highres_s
    twt_2ms_s
  truth/
    log_ai_highres
    log_ai_2ms
    reflectivity_highres
    reflectivity_2ms
    rgt_highres
    rgt_2ms
    zone_id_2ms
    boundary_mask_2ms
    valid_mask_2ms
  priors/
    lfm_ideal_2ms
    lfm_field_like_2ms
  seismic/
    nominal_2ms
    <scenario_id>_2ms
```

数组必须带 shape、dtype、单位、domain、采样率、生成参数和 schema 属性。禁止依靠数组
名称猜测单位或轴顺序。

### 8.2 `sample_index.csv`

一行对应一个可消费样本或 patch，至少记录：

- `sample_id`、`realization_id`、`parent_realization_id`。
- `suite`、`geometry_family`、`split`、`hdf5_group`。
- patch 的横向和 TWT 范围。
- 子波与失配场景。
- 探针频率、相位、noise-equivalent 倍数和实际 `log(AI)` RMS。
- LFM 版本。
- seed、状态和拒绝原因。

### 8.3 其他正式输出

- `scenario_catalog.csv`：冻结的场景、参数网格和预期用途。
- `frequency_probe_results.csv`：生成阶段对探针真值、正演响应和理论参数的自检，不是
  模型评测结果。
- `generation_qc.csv`：逐 realization 的范围、能量、层厚、连续性和异常状态。
- `benchmark_manifest.json`：schema、输入来源、场景版本、split、seed、held-out 家族、
  HDF5 数据集定义及文件校验值。
- `run_summary.json`：运行参数、接受/拒绝数量、拒绝原因、频率矩阵和 warning。

图件至少包括：

- 真值、反射系数、nominal 地震、两类 LFM 和 mask 总览。
- 楔状体层厚与调谐响应。
- 尖灭真值位置和地震响应。
- 频率-振幅正演响应矩阵。
- Semi-Markov 状态、层厚和横向连续性 QC。

无效 realization 必须保留状态和拒绝原因，不能静默丢弃或重采样到“刚好成功”为止。

## 9. 评测报告卡

首版冻结指标和数据拆分，不设置单一总分，也不预设绝对通过阈值。完成 1D、2D 和空间
约束基线后，再以新的 benchmark 版本记录相对门槛。

报告卡至少包含：

- 分频 `log(AI)` 幅度误差和相位误差。
- 全频及分频 NRMSE、相关性和均值偏差。
- 楔状体最小可分辨厚度。
- 尖灭位置误差。
- 层边界定位误差。
- 横向连续性和虚假事件率。
- `0x` 与 unsupported 探针中的虚假目标频率能量。
- nominal 到各失配场景的性能退化。
- ideal LFM 与 field-like LFM 的性能差异。
- seen geometry 与 held-out geometry 的泛化差距。

正演地震误差只用于物理一致性检查，不能替代阻抗真值指标，也不能凭 waveform
correlation 单独选出模型。

## 10. 状态与失败策略

未来实现至少使用：

- `ok`
- `source_run_mismatch`
- `missing_input`
- `unsupported_schema`
- `invalid_wavelet`
- `sampling_mismatch`
- `missing_horizon`
- `crossing_horizons`
- `outside_horizon_support`
- `section_outside_support`
- `invalid_geometry`
- `invalid_impedance`
- `invalid_antialias_result`
- `insufficient_noise_equivalent_calibration`
- `split_leakage`
- `hdf5_contract_error`
- `generation_rejected`

运行级来源矛盾、schema 不支持、nominal 子波无效、split 泄漏和 HDF5/manifest 不一致应
使整次运行失败。单个 section 或 realization 的地质生成问题保留拒绝记录后继续。

## 11. 实现约束与测试

`.ref/synthoseis/` 只提供“先层序、后属性、再正演”的生成哲学。首版不得复制其完整工程、
HDF 管理、断层、盐体、圈闭、AVO 或 Linux 专用流程。

未来实现至少覆盖：

1. 任意 `N >= 2` 的层位数量、任意名称和显式文件路径。
2. 层位缺失、交叉、无支持及 section 路径越界。
3. 8 倍超采样、抗混叠和 2 ms 轴严格对齐。
4. 高分和 2 ms 轴分别计算精确 `tanh` 反射系数。
5. 与前向可观测性闸门相同的卷积、挂点和子波中心约定。
6. 已知频率、相位与振幅探针的构造值和数值投影一致。
7. `0x` 探针不含目标频率真值能量。
8. Semi-Markov 状态占比、持续长度和横向相关性符合配置。
9. 高频 residual 只能由 truth 与退化目标相减派生。
10. 父 realization 及其所有变体不跨 split。
11. held-out geometry family 不出现在 train 或 validation。
12. 固定配置和 seed 产生完全一致的 manifest、参数和数组。
13. HDF5 数组、`sample_index.csv` 和 manifest 的 ID、shape、单位及校验值一致。
14. 无效 realization 产生拒绝记录，不静默重采样。
15. 文档、导航和内部链接通过 MkDocs 严格构建。

## 12. 首版边界

首版只做二维叠后声阻抗基准，不做：

- 断层、盐体、河道或复杂沉积相。
- 叠前 AVO。
- Vp、Vs、Rho 联合岩石物理。
- 三维小体。
- 神经网络、训练器或生产反演入口。
- 永久频率 cutoff 或单一模型排行榜分数。

当前工区的运行结果可以作为配置和契约实例，但不得成为默认路径、固定层位、固定频率或
固定井数。完成本基准后，下一项工作才是在同一冻结数据与报告卡上实现最小 1D、2D 和
空间约束反演基线。
