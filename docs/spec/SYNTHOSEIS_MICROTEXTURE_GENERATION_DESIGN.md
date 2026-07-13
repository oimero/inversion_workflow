# Synthoseis-lite 微纹理生成方法论规格

## 1. 目标与范围

本文是 [GINN v2 Canonical Increment 语义重构规格](GINN_V2_CANONICAL_INCREMENT_SEMANTICS.md) 的合成数据补充。它定义如何在半马尔科夫对象序列和对象内部 profile 上生成高分辨率真值，并保持 canonical decomposition 与 GINN v2 训练接口一致。

本合同的公共字段是 `model_target_log_ai`、`canonical_background_log_ai`、`target_increment_log_ai`、`input_lfm_log_ai` 和 `predicted_increment_log_ai`。历史 `enhance` 的 Stage-1/Stage-2 资产、旧 `base_ai` 接口和无对象限定的 residual/delta 字段不属于本合成接口。

首版把对象内部表达固定为三参数宏观 profile 加一个可替换的微纹理层。首轮使用三个清晰的合成实验模式：

| 实验 | `macro_profile` | `microtexture` | 目的 |
| --- | --- | --- | --- |
| A | `three_parameter` | `none` | 当前宏观基线 |
| B | `three_parameter` | `thin_bed_cluster` | 参数化薄层先验 |
| C | `three_parameter` | `canonical_well_texture` | 去宏观后的井纹理先验 |

`compound_cycle` 和对象内概率混合属于后续独立消融。本首版不要求网络输出随机 realization，也不改变 GINN v2 的网络输入、输出或损失名称。

## 2. 生成语义

### 2.1 HSMM、宏观 profile 与微纹理

半马尔科夫生成器负责地层宏观语法：

- zone 和 state 的序列；
- 对象的连续厚度和边界；
- 几何事件、尖灭和楔形；
- 横向相关和对象间状态转移。

对象局部坐标 `xi` 从 0 到 1。宏观 profile 使用当前三参数表达：

```text
q_macro(xi) = c0 + c1 * (2 * xi - 1) + c2 * sin(pi * xi)
```

对象真值为：

```text
target_log_ai_highres
    = background_log_ai_highres
    + q_macro_highres
    + q_texture_highres
```

微纹理模式为：

```text
q_texture = 0                         # none
q_texture = q_thin_bed_cluster        # thin_bed_cluster
q_texture = q_canonical_well_texture  # canonical_well_texture
```

`canonical_well_texture` 是从井 canonical increment 中移除宏观基函数后的局部纹理，可以叠加到三参数 profile 上。

### 2.2 井纹理的宏观基函数去重

对一个归一化井 patch `w(xi)`，定义宏观基函数空间：

```text
M = span { 1, 2 * xi - 1, sin(pi * xi) }
```

拟合：

```text
beta = argmin_beta || w - [1, 2*xi-1, sin(pi*xi)] beta ||²
w_macro = [1, 2*xi-1, sin(pi*xi)] beta
w_texture_raw = w - w_macro
```

`canonical_well_texture` 使用带端点 taper 的 `w_texture_raw`。这一操作只服务于生成器内部的职责分离，不改变最终 canonical 标签：

```text
canonical_background_log_ai = P(target_log_ai)
target_increment_log_ai     = target_log_ai - canonical_background_log_ai
```

不在输出端引入第二个低通算子，也不把 `w_texture_raw` 宣称为严格频带投影。其低频响应、端点跳变和幅度只进入 QC。

### 2.3 可观测性边界

微纹理是生成先验，不是地震唯一可观测性的声明：

1. 生成器创建高分辨率真值结构；
2. 正演模型决定该结构对当前地震的响应；
3. GINN v2 学习在训练分布和物理约束下给出 canonical increment 解释。

当高分辨率结构的正演响应低于噪声或波子有效频带时，报告称其为 prior-supported 或 ambiguous detail，不称为地震直接恢复的唯一细节。

## 3. 微纹理发射器合同

### 3.1 微纹理接口

```text
emit_microtexture(
    mode,                     # none / thin_bed_cluster / canonical_well_texture
    sample_domain,
    sample_axis_highres,
    object_top,
    object_bottom,
    zone_id,
    state_id,
    macro_profile,
    calibration,
    random_seed,
) -> MicrotextureEmission
```

返回对象至少包含：

```text
values_log_ai               # 对象内部高分辨率 log(AI) 扰动
mode
source_id
physical_thickness
amplitude_rms
interface_count
endpoint_jump_top
endpoint_jump_bottom
```

发射器使用时间域秒轴或深度域米轴，不使用固定采样点数表达跨域厚度。输出必须与对象高分辨率轴等长、对象外为零、对象内有限，并在固定输入和 seed 下完全确定。生成失败时返回明确原因，不静默切换到其他模式。

### 3.2 `none`

`none` 返回全零微纹理，并记录对象 mode、zone、state 和厚度。它用于测量无纹理真值上的 false texture。

### 3.3 `thin_bed_cluster`

`thin_bed_cluster` 在对象内部生成有限数量的交替阻抗层：

```yaml
thin_bed_cluster:
  interface_count: [2, 5]
  layer_thickness_axis_units: [min, max]
  contrast_log_ai: [min, max]
  alternating_sign: true
  state_conditioned: true
```

所有范围按采样域显式配置：时间域为秒，深度域为米。实现检查对象实际物理厚度、最小可表达厚度、state 条件和对象边界。交替层在高分辨率轴生成，随后与宏观 profile 一起经过既有抗混叠和降采样。

### 3.4 `canonical_well_texture`

`canonical_well_texture` 从训练井 bank 取 canonical increment 片段，拟合并移除 `M` 中的宏观成分，施加显式端点 taper，再重采样到目标对象轴。

首轮振幅倍率固定为 1。长度伸缩范围、taper 长度和失败条件必须显式配置；不根据采样点个数隐式缩放。patch 两端记录：

```text
endpoint_jump_top
endpoint_jump_bottom
taper_axis_units
```

端点跳变过大时，发射器失败或按 acceptance policy 拒绝，不用 `none` 或 `thin_bed_cluster` 替代。

## 4. Canonical well bank

### 4.1 井曲线处理顺序

bank 的井曲线处理顺序与 canonical increment 主规格一致。

深度域：

```text
预处理井 logAI（MD）
-> 井轨迹映射到 TVDSS
-> 重采样到最终等间隔 TVDSS 轴
-> 按连续有限段应用 depth canonical P
-> well_target_increment_log_ai = well_log_ai - P(well_log_ai)
-> 由训练井校准流程产生 state label
-> 提取 canonical increment entry
```

时间域：

```text
预处理井 logAI（MD）
-> 井震标定映射到 TWT
-> 重采样到最终等间隔 TWT 轴
-> 按连续有限段应用 time canonical P
-> well_target_increment_log_ai = well_log_ai - P(well_log_ai)
-> 由训练井校准流程产生 state label
-> 提取 canonical increment entry
```

低通不在 MD 轴或中间转换轴上执行。任何 entry 不跨越 NaN、无效间隙、zone 边界或 state 对象边界。

### 4.2 State label 合同

真实井没有天然的 HSMM `state_id`。首版使用训练井校准流程产生的 `calibrated_background_difference_threshold` 标签：

```text
well_background_difference = well_log_ai - calibrated_zone_background
low      if well_background_difference < center - threshold_sigma * sigma
middle   if |well_background_difference - center| <= threshold_sigma * sigma
high     if well_background_difference > center + threshold_sigma * sigma
```

短 state run 按校准流程的最小长度规则合并。bank manifest 必须记录：

```text
state_labeling_method: calibrated_background_difference_threshold
state_labeling_config
state_thresholds
state_calibration_version
state_boundary_minimum_length
```

阈值、校准参数和 state 合并规则只由训练井拟合。留出井不参与阈值、分类器或 bank 的构建。

### 4.3 Bank entry

一个 bank entry 保存 canonical increment 及其去宏观成分后的纹理表示：

| 字段 | 含义 |
| --- | --- |
| `source_id` | bank 内稳定标识 |
| `source_well_id` | 来源井标识 |
| `sample_domain` | `time` 或 `depth` |
| `axis_unit` | `s` 或 `m` |
| `zone_id` | 来源 zone |
| `state_id` | 校准 state |
| `physical_length` | entry 物理长度 |
| `increment_values_log_ai` | canonical increment 片段 |
| `increment_texture_values_log_ai` | 去宏观基函数后的井纹理 |
| `texture_amplitude_rms` | texture RMS |
| `peak_to_peak` | entry 峰峰值 |
| `n_valid_samples` | 有效样点数 |

`increment_texture_values_log_ai` 用于 `canonical_well_texture`。`increment_values_log_ai` 保留完整 canonical increment，便于检查去宏观基函数的计算；二者不共享同一个未处理的数组语义。

### 4.4 井控隔离与覆盖预检

bank 只使用训练井。真实井监督验证、真实工区评估和独立 QC 使用的井不得进入 bank。synthetic geometry holdout 与 real-well holdout 分别记录。

在生成任何 paired group 前，先对所有候选对象建立 coverage matrix：

```text
(zone_id, state_id, physical_thickness)
    -> canonical_well_texture candidates
```

预检必须检查：

- texture entry 是否存在；
- source/target 物理长度比是否在配置范围；
- source state、zone 和采样域是否匹配；
- entry 的连续有效长度是否满足 taper 和高分辨率要求。

覆盖不足时，在 benchmark 生成前明确标为 unsupported 或终止运行，不等到生成一半才改变宏观分布。

### 4.5 Bank manifest

manifest 记录：

- increment contract；
- state labeling contract；
- 训练井与留出井集合；
- zone/state/厚度 coverage matrix；
- `bank_builder_git_commit`；
- `bank_entry_count`；
- `bank_total_sample_count`；
- increment 和 texture 的长度与幅度分布。

本项目不增加逐文件 SHA-256、递归 provenance 或整体内容 digest。运行目录、配置、builder commit 和数量统计用于区分 bank 构建。

### 4.6 Entry 采样

1. 按 zone、state 和物理长度筛选候选；
2. 用显式 seed 选择 entry；
3. 在物理坐标上重采样到目标对象轴；
4. 对 `canonical_well_texture` 执行声明的 taper；
5. 输出 source、长度伸缩和端点 QC。

长度伸缩超出范围时发射器失败，不自动改用其他 mode。

## 5. 生成顺序、配对与数值合同

### 5.1 生成一次 macro parent

A/B/C 必须共享同一个宏观父样本。流程为：

```text
生成一次 HSMM 状态、对象厚度、几何和宏观随机流
-> 保存 macro_parent
-> 为同一个 parent 分别实例化 A / B / C
-> 分别运行 texture emitter、composite 和 QC
```

宏观随机流与 mode-specific 随机流分离。改变 mode 不得重新抽取 HSMM 序列、对象厚度、几何事件或 lateral 坐标。

### 5.2 Paired-group acceptance

三个 mode 采用组级接受合同：

```text
三个 mode 全部成功 -> paired_group_status = accepted
任一 mode 失败     -> paired_group_status = rejected
```

任一失败原因包括：

- bank coverage 或 entry 查询失败；
- 薄层簇在对象厚度内不可行；
- composite clipping/reversal/QC 失败；
- forward 或降采样失败；
- canonical 标签无法在完整道上计算。

每个 group 记录：

```text
paired_group_id
paired_group_status
macro_parent_id
mode_a_status
mode_b_status
mode_c_status
paired_rejection_reason
```

只有 `accepted` group 进入任一 A/B/C benchmark。三个 benchmark 的最终 `macro_parent_id` 集合必须完全相同。

### 5.3 对象内生成与组件 QC

每个对象按以下顺序生成：

```text
读取 macro parent 对象
-> 计算 background_log_ai_highres
-> 计算 three-parameter macro profile
-> 调用 microtexture emitter
-> 组合 background + q_macro + q_texture
-> 在 composite 上执行 AI bounds / reversal / clipping QC
-> 记录 object catalog
-> 对整段高分辨率道执行既有 anti-alias 和降采样
-> 生成 model_target_log_ai
```

`generator_macro_log_ai` 和 `generator_microtexture_log_ai` 是诊断组件。组件使用相同高分辨率轴和抗混叠流程；高分辨率组件只在小型 fixture 或显式诊断运行保存。

如果 composite 发生 clipping：

- `model_target_log_ai` 保存最终真值；
- 组件诊断保存 clipping 前的组件；
- `composite_clipping_fraction` 进入 object 和 realization QC；
- 组件相加检查针对 clipping 前 composite；
- acceptance policy 决定 group 是否接受，不静默改变 mode。

### 5.4 Canonical 标签

对完整 model 轴道计算：

```text
canonical_background_log_ai = P(model_target_log_ai)
target_increment_log_ai
    = model_target_log_ai - canonical_background_log_ai
```

NaN 和 invalid 间隙两侧独立处理。训练 patch 只裁剪完整道数组；patch 内不调用 `P`，也不从 `input_lfm_log_ai` 反推 target。

## 6. synthoseis_lite_v4 产物扩展

### 6.1 HDF5 结构

沿用 `synthoseis_lite_v4`，增加以下 v4 内部字段：

```text
/realizations/<id>/truth/model_target_log_ai
/realizations/<id>/truth/generator_macro_log_ai
/realizations/<id>/truth/generator_microtexture_log_ai
/realizations/<id>/truth/microtexture_mode_model
/realizations/<id>/targets/target_increment_log_ai
/realizations/<id>/priors/canonical_background_log_ai
/realizations/<id>/priors/input_lfm_variants/<variant_id>/log_ai
```

`microtexture_mode_model` 是类别场，不经过普通抗混叠滤波。它使用 object ID/类别的最近邻或模型采样点中心所属对象赋值。模式场不产生中间数值。

### 6.2 Object catalog

`object_catalog.csv` 保存生成参数和边界 QC：

```text
paired_group_id
macro_parent_id
microtexture_mode
microtexture_source_id
physical_thickness
macro_amplitude_rms
microtexture_amplitude_rms
microtexture_peak_to_peak
microtexture_interface_count
length_stretch_ratio
taper_axis_units
endpoint_jump_top
endpoint_jump_bottom
composite_clipping_fraction
```

object catalog 不复制 realization 级 forward response 或 visibility。对象级 visibility 只有在显式反事实正演时才计算；首版默认只报告 realization 级聚合。

### 6.3 Manifest

`benchmark_manifest.json` 记录：

- `benchmark_schema: synthoseis_lite_v4`；
- `microtexture_schema: microtexture_emission_v1`；
- 当前 `microtexture_mode`；
- `paired_group_id` 规则和全组接受统计；
- bank split、state labeling contract 和 coverage matrix；
- 采样域、轴单位、高分辨率间隔和抗混叠参数；
- 所有薄层、taper 和长度范围；
- common-random group 与 `macro_parent_id` 规则；
- canonical increment contract；
- realization 级 forward visibility、LFM 和接受率 QC 摘要。

## 7. 可比消融协议

### 7.1 三个训练模式

首轮建立三个独立 benchmark：

| 实验 | `macro_profile` | `microtexture` | 目的 |
| --- | --- | --- | --- |
| A | `three_parameter` | `none` | 宏观基线 |
| B | `three_parameter` | `thin_bed_cluster` | 参数化薄层先验 |
| C | `three_parameter` | `canonical_well_texture` | 井局部纹理 |

三个运行固定：

- `macro_parent_id` 集合；
- HSMM 状态、对象厚度、几何和横向坐标；
- LFM 退化参数与随机数；
- seismic mismatch 参数与随机数规则；
- split、attempt budget、paired acceptance；
- GINN v2 架构、训练预算、seed 和报告流程。

clean forward seismic 随最终 target 改变。相同 seismic 条件指相同 wavelet、噪声、相位、静差、增益和随机数规则；每个真值仍独立计算 `seismic_model_consistent`。

### 7.2 跨模式共享诊断测试集

三个训练模式都评估同一组跨模式测试集：

```text
shared_none_test
shared_thin_bed_cluster_test
shared_canonical_well_texture_test
```

形成训练模式 × 测试模式矩阵：

| 训练 \ 测试 | none | thin-bed | well-texture |
| --- | ---: | ---: | ---: |
| A | ✓ | ✓ | ✓ |
| B | ✓ | ✓ | ✓ |
| C | ✓ | ✓ | ✓ |

关键报告：

- B、C 在 `shared_none_test` 上的 false texture；
- A 在各微纹理测试集上的漏检；
- B 与 C 的先验迁移；
- 各模式跨测试集的 closure 和井上指标。

分别训练只避免混合三种差异较大的先验，不消除同一 mode 内多个高分辨率真值对应近似地震的条件平均问题。报告不得把确定性 MSE 解释为解决了反演多解性。

### 7.3 LFM 对照

生产 v4 benchmark 按每个 target 的 canonical background 生成 `input_lfm_log_ai`。为了隔离 profile/texture 影响，可额外生成 paired diagnostic fixture：

- 三个 mode 共用同一 `macro_parent_id` 和 macro-anchor LFM；
- 每个 target 仍保存自己的 canonical background 与 target increment；
- manifest 标明 `lfm_comparison_mode: shared_macro_anchor`；
- fixture 用于 common-LFM 诊断，不替代标准 v4 LFM 合同。

### 7.4 GINN v2 训练

三个 mode 分别训练：

```text
synthoseis_lite_v4(A/B/C)
-> canonical increment reader
-> synthetic_supervised
-> ginn_v2_checkpoint_v5
-> increment / closure / forward report
```

GINN v2 继续使用：

```text
input:  seismic, input_lfm_log_ai, valid_mask
output: predicted_increment_log_ai
final:  predicted_log_ai = input_lfm_log_ai + predicted_increment_log_ai
```

physics、真实井监督、R0 和 R1 使用同一 canonical increment 语义。profile/texture mode 是 benchmark 生成条件，不是网络额外输入。

## 8. Forward visibility 与 QC

### 8.1 Realization 级响应

对每个 realization，使用同一 wavelet、采样轴和 forward 参数：

```text
seismic_macro_only       = F(generator_macro_log_ai)
seismic_model_consistent = F(model_target_log_ai)
microtexture_response    = seismic_model_consistent - seismic_macro_only
```

`microtexture_response` 是 realization 级聚合量，不复制到每个 object。多个对象的响应可能重叠、增强或抵消。

首版报告：

```text
microtexture_response_rms
response_to_noise_ratio
response_to_macro_signal_ratio
microtexture_visible_band_response
```

对 `none` mode，`microtexture_response` 记为 `not_applicable` 或零。对象级 visibility 只有在执行明确的反事实正演时才生成：

```text
response_j = F(composite) - F(composite with texture_j removed)
```

这属于后续高成本诊断，不把 realization 级结果填入 object catalog。

### 8.2 Visibility class

manifest 显式记录分类阈值，推荐使用 `response_to_noise_ratio`：

| 类别 | 含义 |
| --- | --- |
| `visible` | response 明显高于当前 mismatch/noise 基线 |
| `ambiguous` | response 与噪声或有效频带处于同一量级 |
| `invisible` | response 低于当前观测条件的可分辨范围 |
| `not_applicable` | 当前模式没有 microtexture，或 clean 数据没有噪声基线 |

clean/base 样本没有噪声时，`response_to_noise_ratio` 使用显式 floor 或记为 `not_applicable`。阈值只作诊断，不作为模型质量硬门禁。

### 8.3 振幅公平性

首轮不强制 B、C 的微纹理振幅完全匹配，但报告必须分离形态和能量影响：

```text
macro_amplitude_rms distribution by mode
microtexture_amplitude_rms distribution by mode
microtexture_peak_to_peak distribution by mode
forward_response_rms distribution by mode
```

模型比较同时展示 raw 指标和按 amplitude/visibility 分层的指标。

### 8.4 必须报告的模型指标

每个训练模式和共享测试集至少报告：

- canonical increment masked MSE 和频带统计；
- canonical closure AI 指标；
- deployment closure AI 指标；
- LFM-only 井上指标和网络净增益；
- thin-bed boundary precision/recall；
- `shared_none_test` 上的 false texture rate；
- canonical well texture 的纹理统计覆盖率；
- 预测和目标的频谱响应；
- realization visibility 分层结果；
- geometry holdout 和 real-well holdout 指标。

## 9. 深度域与横向坐标

深度域所有纵向参数使用 TVDSS 米制轴、对象物理厚度、局部波长或调谐尺度以及高分辨率生成间隔。禁止用固定采样点数表达跨域厚度。

横向生成使用实际 `inline_float`、`xline_float`、`x_m` 和 `y_m`。当前 xline 步长为 4 时，横向相关长度按真实坐标差值计算；微纹理是纵向对象内结构，不因 xline 索引步长改变厚度或频带语义。

## 10. 实施顺序

### 10.1 定义微纹理发射接口

建立领域无关的 `MicrotextureEmission` 值对象：

- 输入字段；
- 输出字段；
- seed 规则；
- 物理单位；
- 对象边界；
- 失败原因；
- source、端点和幅度 metadata。

### 10.2 构建训练井 bank

在最终规则轴上完成 canonical 井处理、state labeling、increment/texture entry、训练井隔离和 coverage matrix。小型时间域与深度域 fixture 先检查轴、单位、NaN 间隙、状态边界、taper 和物理长度。

### 10.3 实现三种组合

按统一接口实现 A/B/C：

1. three-parameter + none；
2. three-parameter + thin-bed cluster；
3. three-parameter + canonical well texture。

### 10.4 接入时间域和深度域生成器

在 macro parent 固定后，先生成 macro profile，再生成 texture，最后执行 composite QC、抗混叠和降采样。时间域和深度域共享模式语义；轴、厚度、波子和 anti-alias 参数由 domain adapter 提供。

### 10.5 实现 paired-group 生成

增加 bank coverage preflight、三模式全组接受/全组拒绝、paired status、拒绝原因和相同 parent 集合检查。任何单模式失败都不进入三个 benchmark。

### 10.6 写入 v4 benchmark 与报告

写入宏观/纹理组件、类别场、object catalog、paired metadata、state contract、coverage 和 realization visibility。类别场使用 nearest/center assignment，不经过 anti-alias。

### 10.7 运行共享测试集

生成三组 shared test set，运行三个训练模式的交叉评估矩阵。先验证生成器、标签、forward 和 report 可复算，再运行完整 GINN v2 训练。

### 10.8 外部 enhance 对照

历史 `enhance` 保持独立，只作为冻结的外部参考方法：

```text
ginn_v1 base_ai -> enhance -> enhanced_ai
```

它用于同一工区上的频谱、井上指标和 false texture 对照，不进入新合成器、GINN v2 输入或 canonical increment 训练目标。只有在合成先验消融完成后，才评估是否需要重新设计 canonical detail refiner。

## 11. 配置示例

下面示例展示单个 mode-specific benchmark 的配置形状。三个实验只替换 `microtexture.mode`，其余宏观、几何、LFM、mismatch 和随机数保持一致。

```yaml
synthoseis_lite:
  benchmark_schema: synthoseis_lite_v4
  sample_domain: depth
  global_seed: 20260720

  impedance_attribute_generator:
    family: object_coefficients_v2
    state_threshold_sigma: 1.0
    lateral:
      correlation_length_section_fractions: [0.1, 0.3, 1.0]
      coefficient_sigma_multipliers: [0.25, 0.50]
      thickness_log_sigma_values: [0.10, 0.25]

    microtexture:
      schema: microtexture_emission_v1
      mode: none              # none / thin_bed_cluster / canonical_well_texture
      selection: realization_fixed
      common_random_group: depth_microtexture_ablation_20260720

      thin_bed_cluster:
        interface_count: [2, 5]
        layer_thickness_m: [10.0, 40.0]
        contrast_log_ai: [0.03, 0.12]
        alternating_sign: true
        state_conditioned: true

      canonical_well_texture:
        bank_manifest: path/to/training_well_bank.json
        allowed_length_stretch_ratio: [0.5, 2.0]
        taper_axis_units: 20.0
        amplitude_multiplier: 1.0
        require_same_zone: true
        require_same_state: true

  canonical:
    enabled: true

  splits:
    assignment_unit: parent_realization
    held_out_geometry_family: pinchout

  paired_ablation:
    group_id: depth_microtexture_ablation_20260720
    required_modes:
      - three_parameter_none
      - three_parameter_thin_bed_cluster
      - three_parameter_canonical_well_texture
    require_all_modes: true

  microtexture_comparison:
    shared_cross_mode_tests: true
    shared_lfm_diagnostic_fixture: true
    visibility_thresholds:
      response_to_noise_visible: 2.0
      response_to_noise_invisible: 0.5
```

时间域将深度专用字段替换为秒制轴单位，并使用时间域 canonical increment contract。`microtexture_mode_model` 使用类别/对象 ID 的最近邻赋值。

## 12. 验收矩阵

### 12.1 Bank 与状态

- bank 只含训练井；留出井查询失败；
- state label 来自训练井 calibrated background-difference threshold；
- manifest 记录 state method、阈值、校准版本和最小状态长度；
- increment 和 texture entry 的 zone/state/长度 coverage matrix 完整；
- texture entry 的 source/target 物理长度比符合配置；
- builder commit、entry count 和 sample count 可追溯。

### 12.2 生成器与 paired group

- 固定 seed 产生完全相同的 macro parent、组件、catalog 和 target；
- A/B/C 共享完全相同的 `macro_parent_id` 集合；
- 任一模式失败时 paired group 整组拒绝；
- `paired_rejection_reason` 能区分 bank、厚度、边界、clipping、forward 和标签失败；
- `none` 的微纹理数组严格为零；
- thin-bed 的层数、厚度、对比度和交替方向符合配置；
- 三参数 profile 固定，`canonical_well_texture` 只叠加到三参数；
- 端点 taper、端点跳变和对象边界检查通过；
- 深度域 xline 步长 4 不改变纵向单位和 lateral 坐标计算。

### 12.3 Canonical 标签和 v4 产物

- `target_log_ai` 等于最终 composite；
- `canonical_background_log_ai = P(target_log_ai)` 可由完整道复算；
- `target_increment_log_ai = target - background` 可由完整道复算；
- NaN 间隙两侧不互相影响；
- 训练 patch 的 target 等于完整道 target 的直接切片；
- 类别场不经过普通抗混叠，不产生中间类别值；
- v4 manifest 的 mode、bank、state、paired、contract 和 common-random 元数据完整；
- 组件诊断和 clipping QC 可独立复算。

### 12.4 Forward 与跨模式报告

- macro-only 和 composite forward 使用相同 wavelet、轴和 mismatch 条件；
- realization 级 `microtexture_response`、visibility class 和 noise ratio 可复算；
- object catalog 不复制未经反事实正演的 realization visibility；
- 三个训练模式都评估三个 shared test set；
- B/C 在 shared none 上的 false texture 单独报告；
- A 在薄层和井纹理测试集上的漏检单独报告；
- amplitude、peak-to-peak 和 forward response 按 mode 分层报告；
- canonical increment、两类 closure、LFM-only 和最终 AI 指标均有报告；
- geometry holdout 与 real-well holdout 不混用。

### 12.5 GINN v2 与历史对照

- reader 只读取 v4 canonical 字段；
- 网络仍接收 seismic、LFM、valid mask 三通道；
- 网络输出仍为 `predicted_increment_log_ai`；
- A/B/C 分别建立 checkpoint 和 report；
- physics、真实井、R0 和 R1 使用相同 canonical increment 语义；
- 微纹理 mode 不作为网络额外输入；
- 新 v4 产物不承载历史 `enhance` Stage-1/Stage-2 字段、旧 `base_ai` 接口或无对象限定的 residual/delta 字段；
- `enhance` 只作为独立外部对照，不被新生产包导入。

## 13. 首版边界与后续消融

首版固定单一 canonical increment 目标和单一 GINN v2 输出。后续可以独立研究：

- `compound_cycle` 发射器；
- 对象内按 state/thickness 概率混合 mode；
- 同一 seismic/LFM 条件下的多真值训练；
- latent 或 ensemble 微纹理 realization；
- paired view consistency loss；
- resolved increment 与 prior-supported detail 的多尺度报告；
- 由 patch bank 压缩得到的状态条件稀疏字典；
- 对象级反事实 forward visibility；
- canonical 语义下的独立 detail refiner。

这些扩展以 A/B/C 的生成合同、paired acceptance 和 forward visibility 结果为前提，不改变本首版 canonical increment 的定义。
