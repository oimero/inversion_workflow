# 合成基准生成与评估

## 文档地位

本闸门以评测为先：先建立冻结、可复现、已知真值的二维合成基准，再用它判断逆问题的
实际可恢复性。它不选择神经网络架构，不立即建设大规模训练集，也不把某次工区诊断得到
的频率固化为模型边界。

首版只冻结基准数据、拆分和报告卡，还不设置阻断模型的通过阈值。完成首批 1D、2D 和
空间约束基线后，才在新的 benchmark 版本中建立相对门槛；在此之前它是“闸门建设阶段”，
不是已经启用的模型准入门。

未来校准、生成与评测使用不带步骤编号的入口：

- 属性校准：`scripts/synthoseis_lite.py calibrate`
- Canonical 生成：`scripts/synthoseis_lite.py generate --suite canonical --impedance-calibration <file>`
- Field-conditioned 生成：`scripts/synthoseis_lite.py generate --suite field_conditioned --impedance-calibration <file>`
- 评测 CLI：`scripts/evaluate_synthoseis_lite.py`
- 核心包：`src/cup/synthetic/`
- 数据 schema：`synthoseis_lite_v1`
- 阻抗校准 schema：`synthoseis_lite_impedance_calibration_v1`
- 阻抗生成器 family：`object_coefficients_v1`
- 报告 schema：`synthoseis_lite_report_v1`

稳定生产链仍终止于第五步。本研究入口不是“第六步”。

## 1. 闸门回答的问题

本基准需要区分三个问题：

1. 在当前离散正演、子波和噪声条件下，给定频率和振幅的阻抗扰动是否可恢复。
2. 模型是否能恢复薄层、楔状体、尖灭和倾斜层，而不是只会拟合规则正弦探针。
3. 模型在低频先验、子波、gain、相位、时移和噪声失配下是否仍保持可信。

正演可观测性旁路只分析“阻抗扰动能否产生可辨识地震响应”。本合成基准进一步要求模型从
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
target_interval:
  horizons:
    - name: top_a
      well_top: Petrel Well Top A
      file: interpre/seismic_top_a
    - name: marker_b
      well_top: Petrel Well Top B
      file: interpre/seismic_marker_b
    - name: base_c
      well_top: Petrel Well Top C
      file: interpre/seismic_base_c

synthoseis_lite:
  global_seed: 20260615
  source_runs:
    forward_observability_dir: <observability-run>
    wavelet_generation_dir: <step-5-run>
    well_auto_tie_dir: <step-4-run>
    well_preprocess_dir: <step-3-run>
  sampling:
    expected_output_dt_s: 0.002
    vertical_oversampling_factor: 8
    antialias:
      fir_taps_per_factor: 32
      cutoff_output_nyquist_fraction: 0.9
      kaiser_beta: 8.6
  geometry:
    lateral_sample_interval_m: 25.0
    patch:
      lateral_samples: 128
      twt_samples: 256
      lateral_stride: 64
      twt_stride: 128
    canonical:
      enabled: true
      lateral_samples: 128
      center_twt_s: 1.5
      vertical_extent_periods: 6.0
      thin_bed_period_ratios: [0.0625, 0.125, 0.25, 0.5, 1.0]
      wedge_transition_fraction: 0.80
      pinchout_termination_fraction: 0.75
      dip_drop_period_ratios: [0.25, 0.5, 1.0]
      lateral_contrast_multipliers: [0.25, 0.5, 1.0, 2.0]
    field_conditioned:
      enabled: true
      geometry_families: [none, wedge, pinchout]
  sections:
    - section_id: example_section
      path:
        - {inline: 300.0, xline: 900.0}
        - {inline: 500.0, xline: 1200.0}
  impedance_attribute_generator:
    family: object_coefficients_v1
    state_threshold_sigma: 1.0
    sensitivity_threshold_sigmas: [0.75, 1.25]
    minimum_calibration_duration_truth_samples: 2
    robust_scale:
      huber_delta_parent_sigma_floor: 0.05
      coefficient_sigma_parent_floor: 0.05
      coefficient_sigma_parent_cap: 3.0
    duration_modes:
      standard:
        minimum_truth_samples: 4
        allowed_splits: [train, validation, test]
      ultra_thin_stress:
        minimum_truth_samples: 2
        allowed_splits: [benchmark, test]
    lateral:
      correlation_length_section_fractions: [0.1, 0.3, 1.0]
      minimum_correlation_sample_intervals: 4
      correlation_length_warning_relative_error: 0.35
      coefficient_sigma_multipliers: [0.25, 0.50]
      thickness_log_sigma_values: [0.10, 0.25]
    qc:
      max_global_reversal_fraction: 0.10
      max_object_reversal_fraction: 0.25
      max_global_clipping_fraction: 0.005
      max_object_clipping_fraction: 0.02
      scenario_acceptance_warning_fraction: 0.80
      scenario_acceptance_failure_fraction: 0.50
      minimum_attempts_per_scenario: 20
  forward_qc:
    highres_mismatch:
      enabled: true
      required: false
  lfm:
    enabled: true
    ideal:
      cutoff_hz: 10.0
      numtaps: 129
      kaiser_beta: 8.6
    controlled_degraded:
      constant_bias_sigma_log_ai: 0.02
      linear_twt_trend_sigma_log_ai: 0.02
      zonewise_bias_sigma_log_ai: 0.03
      lateral_smooth_bias_sigma_log_ai: 0.02
      lateral_correlation_fraction: 0.30
      amplitude_scale_sigma: 0.05
      over_smoothing:
        cutoff_hz: 6.0
        numtaps: 129
        kaiser_beta: 8.6
        blend: 1.0
      local_missing_control_bias:
        enabled: true
        max_abs_log_ai: 0.04
        lateral_width_fraction: 0.30
        twt_width_fraction: 0.30
  seismic_mismatch:
    enabled: true
    noise:
      white_noise_rms_fraction: 0.05
      colored_noise_rms_fraction: 0.05
      absolute_noise_rms_floor: 0.01
      colored_time_correlation_samples: 5.0
    gain:
      global_log_sigma: 0.15
      tracewise_log_sigma: 0.15
      time_lateral_log_sigma: 0.15
      lateral_correlation_fraction: 0.30
      time_correlation_fraction: 0.25
    wavelet:
      phase_rotation_degrees: [-10.0, 10.0]
      time_shift_samples: [-0.5, 0.5]
    combined:
      enabled: true
      phase_rotation_degrees: 10.0
      time_shift_samples: 0.5
      gain_log_sigma: 0.10
      noise_rms_fraction: 0.05
  splits:
    held_out_geometry_family: pinchout
  probe_selection:
    enabled: true
    weak_representatives_per_band: 3
    unsupported_representatives_per_band: 3
    minimum_noise_equivalent_clusters: 3
    low_probe_energy_warning_fraction: 0.01
    conservative_to_nominal_warning_ratio: 1.5
    vertical_tukey_alpha: 0.5
    amplitude_multipliers: [0, 0.25, 0.5, 1, 2, 4]
    phases: [sin, cos]
    lateral_shapes:
      - {name: section_coherent}
      - {name: localized_tukey, centered_fraction: 0.40, alpha: 0.5}
    field_parent_geometry_family: none
    field_parents_per_section: 1
```

示例名称和路径不属于默认值。实现必须接受任意 `N >= 2` 个有序解释层位，不得在变量名、
schema、场景生成或测试中内置当前工区的层位名称和数量。

`expected_output_dt_s` 不是独立的项目默认值。实现从第五步 `selected_wavelet.csv` 推导 nominal
子波采样间隔，并将它作为唯一输出采样间隔。`expected_output_dt_s` 只作显式审计断言；
若填写后与子波采样间隔不一致，则整次运行失败。首版不支持隐式子波重采样来改变主输出
采样率。

### 2.2 井曲线的角色

第三步全频曲线和第四步 filtered LAS 用于冻结 `object_coefficients_v1` 的校准产物：

- 第四步 filtered LAS 估计每井、每 zone 的 `log(AI)` 背景和趋势。
- 第三步全频 LAS 在扣除该背景后校准状态、持续长度和对象内部形态。
- 第四步条件化前后的幅度差异。
- 井震残差的 RMS、频谱形态和相关长度。

它们不是无噪地质真值。生成器不得复制某口井的完整曲线、残差波形或局部 patch 作为
synthetic truth。属性校准按“空间簇等权、簇内井等权、井内样点或对象等权”聚合，避免
密井平台和长井段重复加权。当前首版使用全部有效井校准，因此 manifest 必须明确记录：
这些井参与了 synthetic 分布设计，后续不能被称为属性层面的独立真实盲测井。

井统计只限定 field-conditioned 随机场景的合理范围。可控频率探针仍须按第 6 节的振幅阶梯主动
覆盖低于、接近和高于现实失配底的情况，不能被当前小样本井分布截断。

### 2.3 冻结校准产物

`calibrate` 与 `generate` 是两个显式阶段。`calibrate` 读取第三、四、五步及第一闸门的
显式来源，输出 `synthoseis_lite_impedance_calibration_v1`；`generate` 只消费该冻结产物，
不得在生成过程中重新估计井统计。校准产物必须记录：

- 全部来源运行目录及输入文件 SHA-256。
- 参与和拒绝的井、空间簇、zone、有效样点和拒绝原因。
- `truth_dt`、状态阈值、持续长度模式和所有稳健统计定义。
- 每个 zone/state 的证据等级、父先验、收缩权重和最终参数。
- 第 2.4 节对象目录及 `0.75/1.0/1.25 sigma` 阈值敏感性结果。

`generate --impedance-calibration` 必须核对 schema、generator family、来源目录、
层位顺序、`truth_dt` 和文件校验值。任一不一致均拒绝整次运行，不自动重新校准或回退到
YAML 中的人工范围。

### 2.4 TWT 投影、背景与状态识别

第三、四步 LAS 均通过同井 optimized TDT 映射到 TWT。每个连续有限区间使用分段线性函数
在每个 `truth_dt` cell 上做积分平均；不得用最近邻抽样，不得跨长缺口连接。cell 未被有限
输入完整覆盖时保持无效；允许填补的短缺口必须沿用第一闸门的显式时长和 provenance 规则。

对每井、每相邻层位 zone，将第四步 filtered LAS 写成 zone 坐标 `zeta in [0, 1]`，拟合：

```text
zone_background(zeta) = a + b * (2*zeta - 1)
```

本文固定使用 `zeta` 表示 zone 内归一化坐标，使用 `xi` 表示单个对象内部归一化坐标，
二者不得在 schema、公式或实现变量中混用。

背景使用有限样点普通最小二乘。第三步全频 `log(AI)` 减去该背景后形成状态识别残差。
若共有 `C` 个空间簇，簇 `c` 有 `W_c` 口有效井，井 `w` 有 `N_cw` 个有效样点，则该井
每个样点权重固定为 `1 / (C * W_c * N_cw)`。用这些权重计算 weighted median `center` 和：

```text
sigma = 1.4826 * weighted_median(abs(residual - center))
```

主 benchmark 状态固定为：

```text
low_impedance:  residual < center - 1.0*sigma
background:     center - 1.0*sigma <= residual <= center + 1.0*sigma
high_impedance: residual > center + 1.0*sigma
```

`1.0 sigma` 是冻结 benchmark 参数，不是地质类别真值。校准 QC 必须用 `0.75` 和
`1.25 sigma` 重复分段，比较对象数、状态占比、持续长度分布和转移矩阵，但不得把这两套
阈值扩展成额外生成场景。

连续同状态样点构成一个校准对象。短于 2 个 truth 样点的对象按以下确定性规则合并：

1. 内部短对象分别尝试并入上、下邻对象。
2. 计算改标后相对各候选状态中心的残差平方误差增量。
3. 并入增量较小的一侧；完全相等时并入上侧。
4. zone 边缘短对象并入唯一邻对象。

zone 边界始终是对象边界；相邻 zone 即使状态相同也不得合并。

### 2.5 对象参数与稀疏收缩

对每个校准对象，在其局部坐标 `xi` 上拟合：

```text
profile(xi) = c0 + c1 * (2*xi - 1) + c2 * sin(pi*xi)
```

- 两样点对象固定 `c2=0`，用两点精确求 `c0/c1`。
- 三样点对象使用普通最小二乘。
- 四样点及以上使用 Huber 回归。令 `sigma_zs` 为该 zone/state 残差 robust sigma，
  `sigma_parent` 为全目标窗同状态父先验尺度，阈值固定为
  `max(1.345*sigma_zs, 0.05*sigma_parent)`。
- `c0` 已包含状态条件的对象阻抗偏移，不再另加 `state_offset`。

对象统计使用与样点相同的三级等权原则，将 `N_cw` 换成该井的有效对象数。按 zone/state
分别冻结 `c0/c1/c2` 的 weighted median、`1.4826 * weighted MAD` 和 weighted P01/P99
截断边界。生成时三个系数独立抽样，但最终轮廓还必须检查对象均值、端点差、峰峰值及
位于 `(0,1)` 内的极值；每项必须落入相同 zone/state 校准指标的
`median +/- 3 robust sigma`。缺少内部极值的对象只检查前三项。

独立抽样得到的是候选系数组合，不保证天然满足四项联合轮廓约束。首版不得靠反复抽新
seed 寻找可行组合，而是沿“候选系数向同 zone/state 系数中心”的线段做确定性最大尺度
投影，保留尽可能大的候选偏离；投影尺度、受影响横向列比例和最小投影尺度必须进入
`object_catalog.csv` 与 realization QC。若校准中心本身不满足轮廓边界，标记
`invalid_impedance_calibration`。

原始 zone/state robust sigma 为零、非有限或样本过少时，不得直接产生退化分布。位置、
尺度和上下截断边界均按本节证据权重向父先验收缩；最终系数尺度再限制到对应父尺度的
`[0.05, 3.0]` 倍。校准产物必须同时保存 raw、parent、shrink weight 和 final 值。
父尺度本身非有限或不大于零时，标记 `invalid_impedance_calibration`，不得用任意 epsilon
继续生成。

持续长度在 log 域按 zone/state 拟合截断正态：

```text
mu_log_duration = weighted_median(log(duration))
sigma_log_duration =
    (weighted_Q75(log(duration)) - weighted_Q25(log(duration))) / 1.349
```

截断下界来自生成模式。raw weighted P99 在 log-duration 域按证据权重向全目标窗父
P99 收缩，最终截断上界为该收缩值与对应 zone 长度的较小值。

Semi-Markov 转移矩阵对角线固定为零。zone 内某条有向转移只有在至少出现 2 次、来自至少
2 口井且覆盖至少 2 个空间簇时，才标记为 `zone_supported`。未达到门槛的
`low_impedance <-> high_impedance` 直接跳转不得由 zone 原始统计单独启用；它只能通过
父先验混合保留为 `parent_prior_only`，若父先验也无该边则标记 `forbidden`。校准产物
必须保存每条边的转移次数、井数、空间簇数和支持来源。

每个 zone/state 的校准证据分为：

- `field_calibrated`：至少 5 口井、3 个空间簇和 20 个该状态对象。
- `shrunk`：至少 3 口井和 2 个空间簇，但未达到完整门槛。
- `generic_prior`：低于上述门槛或该状态没有对象，使用全目标窗父先验。

收缩权重固定为：

```text
w = min(
    1,
    n_wells / 5,
    n_clusters / 3,
    n_state_objects / 20
)
```

转移矩阵每行再将 `n_transitions / 10` 纳入同一最小值。位置、尺度、持续长度参数、
P01/P99 截断边界和转移概率均按 `w` 与全目标窗父先验混合；概率混合后重新归一化。
校准产物必须同时保存原始 zone 统计、父先验、`w` 和最终参数，不能只保存收缩后的结果。

## 3. 双套二维场景

所有样本均为 `lateral x TWT` 二维剖面。横向轴保存物理距离和其来源坐标；时间轴使用
正秒 TWT。首版不生成三维小体。

### 3.1 Canonical suite

Canonical suite 是固定、公开、完全参数化的评测集，包括：

- `horizontal_thin_beds`：水平薄互层。
- `wedge`：厚度单调变化并跨越 nominal 子波主周期相关薄层尺度的楔状体。
- `pinchout`：具有已知终止位置的尖灭。
- `dipping_layers`：连续倾斜层。
- `lateral_impedance_change`：几何不变、阻抗对比横向变化。
- `frequency_probe`：Tukey 加窗的正弦和余弦 `log(AI)` 扰动。

每个场景都有解析或构造时已知的界面位置、层厚、终止位置、扰动频率、相位和振幅。
Canonical 样本不得进入后续随机训练抽样，只用于基准验证与模型评测。

Canonical 场景使用 nominal 子波峰值频率对应的主周期
`T_peak = 1 / peak_frequency_hz` 参数化，避免写死某个工区的毫秒数。`T_peak` 只是冻结
几何网格的尺度参考，不等同于由具体子波、相位和层型决定的严格调谐厚度：

| 场景 | 首版参数网格 |
|------|--------------|
| `horizontal_thin_beds` | 单层 TWT 厚度为 `T_peak * [1/16, 1/8, 1/4, 1/2, 1]` |
| `wedge` | 层厚沿横向从 `0` 线性增加到 `T_peak`，变化区占 section 中部 80% |
| `pinchout` | 起始厚度 `T_peak/4`，在 section 75% 位置减至 0，并保持为 0 |
| `dipping_layers` | 跨 section 总 TWT 落差为 `T_peak * [1/4, 1/2, 1]` |
| `lateral_impedance_change` | 对比度为井统计参考对比度的 `[0.25, 0.5, 1, 2]` 倍 |
| `frequency_probe` | 使用第 6 节由第一闸门派生的频率、相位和振幅矩阵 |

manifest 必须保存展开后的具体 TWT、横向位置和阻抗数值。Canonical 版本变更任何参数网格
都需要升级 benchmark 版本，不能在同一版本下悄悄调整难度。

首个实现切片固定生成表中前五个解析几何家族，共 14 个几何 realization；探针实现再
增加一个 `frequency_probe__smooth_background` 父 realization，因此当前 Canonical
父样本总数为 15。每个几何 realization
在校准得到的稳健背景 `log(AI)` 中嵌入一个 high-impedance 目标层；参考对比度取各
zone high/low 对象 `|c0|` 中位数，并限制在背景至校准 P01/P99 中位安全边距的 80%。
`lateral_impedance_change` 只改变目标层对比度，其他四类保持参考对比度。该构造用于
隔离几何和调谐效应，不声称是随机地质分布。

解析界面先以连续 TWT 定义，再投影到 `truth_dt` 网格。`canonical_geometry_qc.csv`
同时记录解析层厚和离散网格占据层厚；两者的最大差不得超过一个 truth sample。
pinchout 的解析终止位置必须精确等于配置值，最后一个非零离散 cell 则按局部层厚斜率、
`truth_dt` 和横向采样间隔计算可解释容差，不能把亚 cell 尖灭误报为几何错误。

`frequency_probe` 使用纯平滑背景，不包含薄层或对象边界，因此其 `0x` 是绝对负对照。
频率、振幅和证据标签严格由第 6 节的第一闸门联接生成，不得用手工固定频率补齐。

### 3.2 Field-conditioned suite

Field-conditioned suite 从显式配置的解释层位和二维路径构造大尺度地层框架：

1. 读取每个解释层位，并使用与低频模型一致的 TargetZone filled horizon grid 构造可建模
   层位面。
2. 沿配置折线按 `lateral_sample_interval_m` 等物理距离重采样最终层位面；折点处保持累计
   路径距离连续。
3. 校验每个横向位置的层位顺序严格从浅到深，并输出 raw/linear/nearest/thickness-fill
   支撑状态 QC。
4. 在相邻层位之间建立归一化 RGT/层序坐标。
5. 在该坐标中生成新的层序、层厚和阻抗属性。

路径可以是 inline、xline 或任意折线，不限定方向。生成器只使用解释层位的大尺度几何，
不读取或复制路径上的真实地震纹理。

section 必须记录累计横向距离、浮点 inline/xline 和 XY 坐标。配置可设置最大 section
长度；超过时显式拒绝，不自动截短。所有二维数组固定使用
`axis_order=["lateral", "twt"]`。patch 的尺寸和 stride 来自显式配置，
`sample_index.csv` 保存每个 patch 的实际横向和 TWT 范围。

最终 filled 层位仍缺失、交叉或路径越界时，应拒绝对应 section 或 realization，并写出
明确状态，不能使用其他路径或默认平层替代。原始解释支撑不足不得静默隐藏，必须通过
`section_geometry_qc.csv` 和几何支撑图暴露。

### 3.3 `object_coefficients_v1` 对象序列

Semi-Markov 只用于 field-conditioned suite 的随机地质部分，Canonical 场景不消费这些
统计。状态固定为：

- `low_impedance`
- `background`
- `high_impedance`

状态只表示相对当地 zone 背景的阻抗关系，不宣称砂、泥或具体岩相。本文中的“层厚”均指
TWT 持续时间或 normalized RGT 持续长度，不表示米制真厚度。

每个 zone 独立生成一条参考柱对象序列，再将同一组 `object_id` 横向延拓到整个 section。
相邻道不得各自运行 Semi-Markov。对象序列生成规则为：

1. 从冻结初始状态分布抽取首状态。
2. 从该 zone/state 的截断对数正态抽取持续长度。
3. 按零对角转移矩阵抽取下一状态，直到填满 zone。
4. 末端余量短于当前模式的最短持续长度时，并入前一对象并记录右删失。
5. 首个对象已超过整个 zone 时截断到 zone 长度并记录右删失。
6. 不为获得“更漂亮”的序列动态重抽对象或追加 seed。

生成包含两个冻结持续长度模式：

- `standard`：最短 4 个 truth 样点，用于常规随机 train/validation/test。
- `ultra_thin_stress`：最短 2 个 truth 样点，只进入 `benchmark` 或 `test`，不得进入训练抽样。

校准对象目录仍保留所有不少于 2 个 truth 样点的对象。这样既保留井上亚毫秒持续长度
证据，又不让常规随机训练套件被工程采样地震难以表达的超薄对象主导。

### 3.4 对象阻抗函数

令 `zeta_top_k(x)` 和 `zeta_bottom_k(x)` 为对象 `k` 在横向位置 `x` 的 zone 坐标边界，
对象局部坐标为：

```text
xi = (zeta - zeta_top_k(x)) / (zeta_bottom_k(x) - zeta_top_k(x))
```

高分辨率真值固定使用：

```text
logAI(x,zeta) =
    zone_background(zeta)
  + c0_k(x)
  + c1_k(x) * (2*xi - 1)
  + c2_k(x) * sin(pi*xi)
```

`zone_background` 的截距和梯度按冻结校准分布为每个 realization、每个 zone 抽取一次，
沿横向保持不变。对象边界允许阻抗跳变；对象内部只允许线性项和半正弦曲率项产生连续
变化。首版禁止额外层内 OU、二维 Gaussian random field、Perlin noise 或其他
`micro_texture`，并固定 `micro_texture=0`。

三个系数从各自截断分布独立抽样。独立抽样不表示假定地质上完全独立；它是首版低自由度
benchmark 选择，必须经过第 3.7 节的最终对象轮廓 QC。

### 3.5 对象级横向 AR(1) 调制

横向变化只调制同一地层对象的系数和厚度，不在完整 `lateral x TWT` 剖面上叠加二维
平滑场，因此不会跨对象边界模糊阻抗对比。对不规则横向采样位置：

```text
rho_i = exp(-delta_x_i / Lx)
g_i = rho_i * g_(i-1) + sqrt(1-rho_i^2) * epsilon_i
```

每条 latent 场生成后先裁剪到 `[-3, 3]`，再去均值并归一化到单位 RMS。系数与厚度必须
使用彼此独立的命名随机流。去均值和单位 RMS 只固定有限 section 上的幅度约定，不宣称
保持理论 AR(1) 协方差完全不变。

相关长度是 benchmark 难度轴，不是井网估计结果：

```text
requested_Lx = section_length * [0.1, 0.3, 1.0]
effective_Lx = max(requested_Lx, 4*lateral_sample_interval)
```

requested/effective 米制值均写入 manifest；下限抬升时记录 warning。属性调制幅度为对应
`c0/c1/c2` 校准 robust sigma 的 `0.25` 或 `0.50` 倍：

```text
cj_k(x) = cj_k_base + amplitude_multiplier * sigma_j * gj_k(x)
```

厚度先从截断对数正态抽取参考柱基础值，再使用：

```text
thickness_multiplier_k(x) = exp(sigma_thickness * g_k(x))
sigma_thickness in [0.10, 0.25]
```

三档相关长度与 weak/strong 两档幅度正交组合，形成 6 个横向场景。每道的正厚度权重在
zone 内重新归一化，严格保持对象顺序并填满解释层位之间的空间。普通对象的归一化使用
带下界的 simplex 分配：先为每个对象保留当前持续长度模式要求的最短厚度，再按 AR(1)
正权重分配剩余厚度；不得先普通归一化、再把过薄对象静默裁厚。

每条 latent 场必须在裁剪前、裁剪后和归一化后分别记录均值、RMS，以及按物理横向距离
估计的经验 e-folding 相关长度。对 `requested_Lx < section_length` 的场景，若归一化后
经验相关长度相对 `effective_Lx` 的偏差超过 35%，记录
`lateral_correlation_length_warning`，但不重抽随机流。`requested_Lx = section_length`
档位定位为 section-scale trend；有限区间无法稳定估计相关长度时只记录
`section_scale_correlation_unresolved`，不据此拒绝 realization。

### 3.6 显式 Field-conditioned 几何事件

Field-conditioned geometry family 固定包含：

- `none`
- `wedge`
- `pinchout`

事件目标只能是 `low_impedance` 或 `high_impedance` 对象。不存在合格目标对象时拒绝该
attempt，不能改选 background 或重新生成对象序列。

`wedge` 的目标对象厚度倍率固定从 `0.25x` 线性变化到 `1.75x`，左右方向分别形成场景。
`pinchout` 使用 smoothstep 渐消，尖灭位置为 `0.35L` 或 `0.65L`，渐消区宽度为
`0.25L`，并分别生成左右方向。

目标对象厚度曲线为精确真值，不再叠加普通厚度 AR(1) 调制。其余对象按各自 AR(1) 正
权重分享剩余 zone 厚度。若任何剩余对象低于当前持续长度模式的最短厚度，则拒绝该
attempt，不裁剪对象、改变尖灭位置或动态补 seed。

“合格目标对象”还必须满足事件后的离散表达约束：

- wedge 的 `0.25x` 薄端至少保留 2 个 truth 样点，不能退化为离散网格上的消失对象。
- pinchout 目标允许为 0 个 truth 样点；其轮廓 QC 在连续对象局部坐标上计算，不能因
  尖灭列缺少离散样点而误判失败。
- 两类事件都必须在选目标前确认增厚端仍能为全部非目标对象保留当前模式的最短厚度。
- 在 high/low 候选中按距对象序列中心由近到远确定性选择首个可行对象；无可行对象时
  记录 `missing_geometry_event_target`。

### 3.7 阻抗语义与生成 QC

状态分布允许有限重叠，但状态标签不能失去相对背景的意义。对每个有效
`object_id x lateral_position`，用对象内部平均 `log(AI)` 与当地
`zone_background` 比较：

- high 对象均值不高于背景，记为 reversal。
- low 对象均值不低于背景，记为 reversal。
- background 对象不进入 reversal 统计。

全 realization 比例以全部有效 high/low `object_id x lateral_position` 为分母；单对象
比例以该对象仍存在的有效横向位置为分母。全 realization reversal 比例不得超过 10%，
单对象比例不得超过 25%；任一超限均拒绝。

每个 zone 先从第三步曲线计算簇去偏 raw weighted P01/P99，再使用第 2.5 节的 zone
证据权重分别向全目标窗父 P01/P99 收缩，得到最终 `log(AI)` 边界。raw、parent、权重和
final 边界必须全部保存；小样本 raw 分位数不能直接充当硬边界。实现先生成未裁剪真值，
再将超界样点裁到最终边界；裁剪只作为安全阀：

- 全 realization clipping 比例以全部有效 high-resolution truth 样点为分母，不得超过
  0.5%。
- 单对象 clipping 比例以该对象全部有效 high-resolution 样点为分母，不得超过 2%。
- 任一超限即拒绝。

必须保存上下边界裁剪的数量、比例和位置。对象轮廓还必须满足第 2.5 节的四项稳健边界；
不得仅凭最终绝对 AI 位于 P01/P99 内就接受异常的 slope/curve 组合。

绝对 AI 生成采用条件位置系数：给定已抽样的 zone background、`c1/c2` 和对象局部坐标
后，`c0` 候选被限制在系数截断边界、轮廓均值边界、状态方向约束及 zone AI P01/P99 的
联合可行区间内。该条件化只调整 `c0`，并记录受影响列比例及最大调整量；最终 clipping
仍保留为数值安全阀和独立拒绝规则，不能用条件化统计替代 clipping QC。

每个场景使用预先冻结的 attempt/seed 表，不能因拒绝而追加 seed。接受率始终统计，但
只有 attempt 数不少于 20 时才执行阈值判断：

- 低于 80%：写入 warning。
- 低于 50%：该场景无效，并使整次 benchmark 生成失败。
- 少于 20 个 attempt：标记 `insufficient_attempts_for_acceptance_qc`，不计算通过/失败
  结论；该冻结场景无效，并使整次 benchmark 生成失败。

拒绝原因至少分别统计对象轮廓、持续长度、reversal、clipping 和事件不可行。

### 3.8 派生残差

不得输出语义含糊的单一 `derived_residual`。首版只输出名称明确的工程采样轴派生标签：

```text
residual_vs_lfm_ideal =
    model_target_log_ai - lfm_ideal
residual_vs_lfm_controlled_degraded =
    model_target_log_ai - lfm_controlled_degraded
```

若以后加入其他 degraded base，必须使用 `residual_vs_<base_semantics>` 新字段并记录
base 的生成参数，不得复用上述名称。残差不是生成器的原始随机变量。

## 4. 真值网格与抗混叠

### 4.1 两套垂向网格

输出采样间隔来自第五步 nominal 子波。以当前常见的 2 ms 子波为例：

```text
output_dt = selected_wavelet_dt = 0.002 s
vertical_oversampling_factor = 8
truth_dt = output_dt / 8 = 0.00025 s
```

所有层位、界面、薄层和尖灭先在 `truth_dt` 网格构造。实现必须同时保存高分辨率真值和
工程采样目标：

- `truth_log_ai_highres`：高分辨率构造真值。
- `model_target_log_ai`：高分辨率真值经过固定抗混叠与工程网格化后的主模型监督
  目标；它不是无损地质真值。

首版模型评测以 `model_target_log_ai` 为准，不要求模型从工程采样地震直接输出
`truth_dt` 超分辨率曲线，也不以该目标评价高分辨率细节恢复。高分辨率真值用于界面、
层厚、部分体素和正演 QC。

### 4.2 降采样

高分辨率阻抗、RGT 和连续属性使用固定的线性相位 Kaiser FIR/polyphase 方案降采样。
令 oversampling factor 为 `q`：

```text
numtaps = 32*q + 1
cutoff = 0.9/q              # 相对于 high-resolution Nyquist
window = ("kaiser", 8.6)
taps = scipy.signal.firwin(numtaps, cutoff, window=window, scale=True)
output = scipy.signal.resample_poly(
    input, up=1, down=q, window=taps, padtype="line"
)
```

FIR 系数、SciPy 版本和系数 SHA-256 必须写入 manifest。滤波器为奇数长度、线性相位，
`resample_poly` 补偿群延迟；上下文在滤波后才裁剪。不得使用简单
`array[..., ::factor]`、IIR 或未记录参数的库默认滤波器。

离散状态和边界 mask 使用适合其语义的占比、主状态或覆盖率表达，不把类别编号做普通
线性低通。

工程采样轴反射系数不能由高分辨率反射系数直接抽样获得。必须分别从对应采样轴上的
`log(AI)` 重新计算，以保持离散正演定义清楚。

### 4.3 上下文

每个目标窗上下至少额外生成半个 nominal 子波长度的高分辨率上下文。所有子波场景均在
完整上下文上卷积，之后才裁取目标窗。若扰动子波比 nominal 更长，则按场景中的最大
半长度建立上下文。

## 5. 统一真值派生与正演闭合

一个 realization 先生成完整 `log(AI)` 真值，再从同一真值派生全部输入与标签：

```text
high-resolution geometry and log(AI) truth
  -> fixed anti-alias/grid operation
  -> model_target_log_ai
  -> exact model-grid reflectivity
  -> nominal model-grid wavelet convolution
  -> seismic_model_consistent
  -> ideal and controlled-degraded LFM
  -> RGT, zone, validity and boundary masks
  -> explicitly named residual targets
```

这是首版主闭合链路。无噪声 nominal 场景必须严格满足：

```text
seismic_model_consistent =
    F_model_grid(model_target_log_ai, selected_wavelet)
```

模型即使完美预测 `model_target_log_ai`，也应能用与第一闸门相同的工程采样正演核心
严格复现主输入地震。首版不得把高分辨率正演后降采样的地震冒充该闭合输入。

### 5.1 正演约定

正演与正演可观测性分析保持一致：

```text
r[j] = tanh((logAI[j] - logAI[j-1]) / 2), j = 1,...,N-1
```

反射系数数组长度为 `N-1`，值挂在 `logAI[j]` 所在的下部样点，不在首部补零，也不改挂
到中点。子波为奇数长度且零时刻位于中心样点，使用
`numpy.convolve(wavelet, reflectivity, mode="same")`。未来实现应复用同一个公开正演
核心，不能在两个研究闸门中维护数值上略有不同的副本。

因此阻抗与正演数组使用不同长度的显式轴：

```text
twt_forward_highres_s = twt_highres_s[1:]
twt_forward_model_s = twt_model_s[1:]
```

`reflectivity_highres` 和 high-resolution 卷积地震使用 `twt_forward_highres_s`；
`reflectivity_model`、`seismic_model_consistent` 和全部工程采样地震场景使用
`twt_forward_model_s`。实现、训练器和评测器不得把长度为 `N-1` 的地震与长度为 `N`
的阻抗按裸数组下标直接对齐，也不得在内部悄悄补零。正演有效 mask 固定由相邻阻抗
样点共同决定：

```text
forward_valid_mask[..., j-1] =
    valid_mask[..., j-1] and valid_mask[..., j]
```

当前生成器会在目标窗上下保留卷积上下文，并把目标窗外的阻抗延拓为边界值以避免卷积
边界伪影。`valid_mask_model` 表示训练/评测的阻抗目标窗；`forward_valid_mask_*`
只表示相邻阻抗样点足以正演地震。GINN、baseline evaluator 和模型报告卡的阻抗误差
必须使用 `valid_mask_model`，不能把上下文延拓区域纳入训练目标或阻抗评分。

可选保存 `seismic_from_highres_truth_model_grid` 作为亚采样调谐和 forward mismatch QC：

1. 用同一显式 Kaiser polyphase FIR 将工程采样子波上采样到 `truth_dt`。
2. 裁剪或补齐到相同物理时长、保持奇数长度和零时刻中心样点，并重新做离散 L2 归一化。
3. 在 high-resolution 反射系数上卷积。
4. 用第 4.2 节同一抗混叠方案降采样到输出轴。

该分支由 `forward_qc.highres_mismatch.enabled` 控制，不作为首版主模型输入或主评分
对象。成功时记录它与 `seismic_model_consistent` 的 RMS、相关性和频谱差异；失败时
记录 `highres_forward_qc_failed`。默认 `required=false`，因此辅助分支失败不阻断主闭合
链路；只有显式配置 `required=true` 时才使整次生成失败。该分支不得替代或修改
`seismic_model_consistent`。

当前实现对每个父 realization 执行主链闭合复算，并记录
`model_grid_closure_max_abs/RMS`。高分辨率分支同时报告原始 RMS/NRMSE、相关性、
频谱形状误差，以及用一个正标量对齐到主链后的 RMS/NRMSE。manifest 保存高分辨率
子波和 Kaiser FIR 的长度、采样率、离散 L2、SciPy 版本与 SHA-256。

两个采样率上的子波都按各自离散 L2 重新归一化，因此原始振幅差不自动解释为挂点错误；
相关性、频谱形状和正尺度对齐误差用于区分振幅口径与真正的亚采样形状差异。即使辅助
分支失配较大，主模型输入仍是严格闭合的 `seismic_model_consistent`。

保存 nominal 子波场景及第五步准入候选。人工失配场景至少包括：

- 白噪声。
- 具有可配置相关长度或谱形的有色噪声。
- `global_scalar_gain`：整个 section 一个正标量。
- `tracewise_lateral_smooth_gain`：每道一个横向平滑的正 gain。
- `time_lateral_smooth_gain`：TWT 与横向均平滑变化的正 gain 场。
- 常相位旋转。
- 正负亚采样时移。

gain 一律作用于无噪声卷积结果之后、加性噪声之前：

```text
seismic_observed = positive_gain * seismic_convolved + additive_noise
```

gain 由 Gaussian latent field 经 `exp()` 转换，保证严格为正；类型、RMS、范围及横向/TWT
相关长度写入样本索引和 manifest。

第四步现实残差仅用于校准噪声 RMS、谱包络和相关长度。不得把真实残差 patch 直接叠加
到合成地震，因为其中混合了标定误差、未建模地质和噪声。

当前实现为每个 base 样本和每个 frequency probe 样本生成有限的命名失配场景，而不是
做全笛卡尔积：`white_noise`、`colored_noise`、`global_scalar_gain`、
`tracewise_lateral_smooth_gain`、`time_lateral_smooth_gain`、两档常相位旋转、两档
亚采样时移，以及 `combined_moderate`。每个场景输出 `seismic_observed`、
`positive_gain` 和 `additive_noise`；其中 gain 场始终严格为正，噪声在有效 forward mask
内按配置 RMS 归一化。`combined_moderate` 按 phase rotation、fractional shift、positive
gain、colored additive noise 的顺序构造。所有随机流使用父 realization 与 source
variant id 命名，因此新增场景不会改变已有样本。

### 5.2 两类低频先验

每个 realization 同时派生：

- `lfm_ideal`：由真值通过明确低通和降采样得到。
- `lfm_controlled_degraded`：在 ideal LFM 上注入可审计的人为退化。

controlled-degraded LFM 的首版误差族包括：

- `constant_log_ai_bias`
- `linear_twt_trend_bias`
- `zonewise_bias`
- `lateral_smooth_bias_field`
- `amplitude_scale_bias`
- `over_smoothing`
- `local_missing_control_bias`

每个 realization 记录误差类型、RMS、频谱和横向/TWT 相关长度。它是理想低频模型的
可控退化，**不代表** kriging 或其他井控空间插值 LFM 的真实误差结构。真实 LFM 鲁棒性
需要在未来获得新 LFM 实现后单独评测，不能把本结果外推为已验证。

当前实现已经生成这两类 LFM 及对应 residual。`lfm_ideal` 使用 model-grid TWT 轴上的
zero-phase Kaiser FIR 低通，参数来自 `synthoseis_lite.lfm.ideal`。`lfm_controlled_degraded`
在 ideal LFM 上叠加固定命名随机流产生的常数偏差、TWT 线性趋势、zonewise bias、横向
AR(1) 平滑 bias、振幅尺度偏差、额外过平滑和局部缺控制 bias。每个父 realization 与其
probe 变体共享同一个 degradation variant，因此 paired `0x` 与非零 probe 的差异只来自
显式 probe，而不是 LFM 随机性。`sample_index.csv` 和 `generation_qc.csv` 记录 LFM RMS、
degradation RMS、residual RMS、滤波参数和 HDF dataset 路径。

## 6. 第一闸门驱动的探针矩阵

### 6.1 频率选择

只读取第一闸门 `whole_target` 的逐频结果。选择：

- 所有 `robust_detectable` 频率；若为空是合法结果。
- 所有 `conditional` 频率。
- 所有 conservative operator 为 `core` 且经验状态为 `not_detectable` 的频率。
- conservative operator 为 `weak` 或 `unsupported` 的连续频带代表点，作为压力测试或
  负对照。每个连续频带至多选择 low edge、center、high edge 三点；不足三点则全部保留。

实现不得硬编码 20 Hz、35 Hz、55 Hz 或任何当前工区结果。所选频率及其来源状态完整写入
`scenario_catalog.csv` 和 manifest。

### 6.2 Noise-equivalent 基准

每个频率从第一闸门 `well_frequency_sensitivity.csv` 计算两套参考振幅：

1. `reference_noise_equivalent_nominal`：只使用 `whole_target`、nominal 子波且
   `status=ok` 的记录。
2. `reference_noise_equivalent_conservative`：同井、同频率内对全部有效子波场景取
   upper empirical P75，再进入空间聚合。P75 使用与第一闸门一致的离散保守定义
   `quantile(..., method="inverted_cdf")`。
3. 两套参考都先在同一空间簇内取井中位数，再跨空间簇取中位数。
4. 至少需要三个有效空间簇；不足时该频率仅生成 `0x` 负对照，并标记
   `insufficient_noise_equivalent_calibration`。

探针矩阵默认使用 nominal 参考振幅，同时记录
`conservative_to_nominal_noise_equivalent_ratio`；该比值大于配置 warning 阈值时必须
提示子波不确定性显著。每个可校准频率使用：

```text
0, 0.25, 0.5, 1, 2, 4
```

倍的 reference noise-equivalent 振幅，并分别生成正弦和余弦两组正交相位。振幅是
`log(AI)` 的加权 RMS，不是峰值或 AI 绝对值。

最终矩阵为：

```text
calibrated frequencies
  x [0, 0.25, 0.5, 1, 2, 4]
  x [sin, cos]
```

不可校准频率只生成 `0x x [sin, cos]`。每个非零探针同时生成两种二维横向形态：

- `section_coherent`：整个 section 同相、同幅。
- `localized_tukey`：位于 section 中央 40%，横向 Tukey `alpha=0.5`，其余位置为零。

探针始终沿物理 TWT 按 Hz 构造，不沿 normalized RGT 拉伸。

探针在 high-resolution 网格构造后，必须记录：

- `probe_rms_requested_highres`
- `probe_rms_actual_highres`
- `probe_rms_after_antialias`
- `probe_rms_actual_model_grid`
- `probe_rms_fraction_of_total_model_grid`
- `probe_energy_fraction_of_total_model_grid`

两个占比在探针与背景的相同有效 mask 内计算。分别去均值后：

```text
probe_rms_fraction_of_total_model_grid =
    rms(probe_model_grid) / rms(model_target_log_ai)

probe_energy_fraction_of_total_model_grid =
    probe_rms_fraction_of_total_model_grid ** 2
```

若背景 RMS 为零或有效样点不足，占比标记无效并记录原因。

模型评测和 noise-equivalent 倍数解释以 `probe_rms_actual_model_grid` 为准。若 4x 探针的
`probe_energy_fraction_of_total_model_grid` 仍低于配置阈值，记录 `low_probe_energy_warning`，
但不自动放大探针或拒绝样本。

`0x` 的语义按 suite 区分：

- Canonical `0x` 使用目标频带能量可忽略的平滑背景，作为绝对虚假恢复负对照。
- Field-conditioned `0x` 保留对象边界和天然地质频谱。对应非零探针必须共享同一父
  realization、对象序列、几何、属性系数、LFM、子波和失配随机流，仅增加已知探针。
- Field-conditioned 评测比较“非零探针预测减 `0x` 预测”与已知注入增量，不得把
  `0x` 中原有的目标频带能量解释为模型虚构。

当前实现不把完整探针矩阵附加到每个随机 realization。Canonical 只在专用平滑父样本上
展开；Field-conditioned 按 `field_parent_geometry_family`，为每个 section 选择前
`field_parents_per_section` 个通过完整地质 QC 的父 realization。默认是每个 section
首个合格 `none` realization。父样本 ID 和数量写入 manifest，不能在运行后按结果挑选。

同一父样本下，所有非零探针与对应 `0x` 共享基础真值、对象、几何和子波。HDF5 只保存
一次父真值；probe group 保存 high-resolution/model-grid 增量及对应闭合地震：

```text
probe_target = parent_base_target + probe_increment
```

`sample_index.csv.paired_zero_sample_id` 是配对键。Field-conditioned 评测必须先按该键
做预测差分，再与已知增量比较。

weak/unsupported 样本主要用于负对照，不得因模型在这些频率偶然得到低误差就将其自动
升级为生产频带。

第一闸门没有 `robust_detectable` 频率是合法结论。此时 benchmark 仍按上述矩阵运行；
若后续模型也没有达到工程可用恢复水平，应报告“算子支持、现实可检测性和逆问题恢复性”
的完整证据链，而不是回调第一闸门阈值或反复调参追求伪高分。

## 7. 数据拆分与防泄漏

### 7.1 父 realization

随机地质样本先生成完整父 realization，再裁取 patch。一个父 realization 只能属于一个
split；其相邻 patch、噪声变体、LFM 变体和子波变体不得跨 split。

训练、验证和测试使用独立 seed 空间。seed 由 manifest 明确记录，不能依赖 Python
进程级随机状态或生成顺序。

随机生成允许使用 manifest 中预先列出的固定 `attempt_id/seed` 候选表。无效候选保留
拒绝记录，数据集只消费 `status=ok` 的 realization；运行过程中不得根据失败数量动态
追加新 seed，以免形成不可审计的漂亮样本筛选。

所有随机流由以下命名键的 UTF-8 规范串经 SHA-256 派生 128-bit seed，并使用 NumPy
`PCG64DXSM`：

```text
global_seed
benchmark_version
generator_family
stream_purpose
realization_id
zone_id
object_id
coefficient_name
variant_id
```

键使用固定顺序、长度前缀和小写十六进制摘要，禁止依赖 Python `hash()`。新增对象、
QC、图件或场景不得改变既有命名流。Field `0x` 与非零探针除探针增量流外必须复用相同
键值。`stream_purpose` 必须来自冻结枚举，例如 `state_sequence`、`duration`、
`coefficient_c0`、`thickness_lateral`、`probe_increment` 或 `additive_noise`，不能用
调用栈位置或循环序号代替。

### 7.2 几何家族留出

除 realization 隔离外，默认把 `pinchout` 作为未见几何测试家族。该选择必须来自配置并
写入 manifest，代码中不得假定被留出的永远是尖灭。

留出家族的所有 realization 只能进入测试集。其他随机场景按父 realization 分配到
train/validation/test。Canonical suite 独立标记为 `benchmark`，不属于上述随机比例。

当前生成器实现仍将非留出 field-conditioned base realization 标记为 `unassigned`，
仅供生成端和 baseline consumer 验证。正式训练入口落地前必须实现父 realization 级
train/validation/test 分配，并使该检查成为 hard failure。

## 8. 输出契约

本节定义未来 schema，但本轮不把它加入稳定工作流的
[核心 CSV 契约](../concepts/csv-contracts.md)。

### 8.1 冻结阻抗校准产物

`calibrate` 至少输出：

- `impedance_calibration.json`：schema、generator family、来源校验值、层位/zone 定义、
  状态阈值、背景、发射、持续长度、转移矩阵、收缩证据和最终参数。
- `well_object_catalog.csv`：逐井、zone、对象记录状态、TWT/RGT 边界、持续长度、
  `c0/c1/c2`、轮廓指标、空间簇和有效性。
- `calibration_qc.csv`：逐 zone/state 记录井数、空间簇数、对象数、证据等级、收缩权重、
  `0.75/1.0/1.25 sigma` 敏感性和拒绝原因。
- `well_calibration_samples.csv`：逐井、zone、样点记录 filtered/full `log(AI)`、背景线、
  residual、状态阈值和初始状态，用于校准解释图。
- `well_background_fits.csv`：逐井、zone 记录背景线 `a/b` 和有效时窗。
- `well_object_profile_samples.csv`：逐对象拟合样点、观测 residual、拟合 residual 和误差。

校准 JSON 必须包含上述 CSV 的 SHA-256。`generate` 不得修改这些文件。

### 8.2 `synthetic_benchmark.h5`

每个父 realization 使用独立 group。至少保存：

```text
/realizations/<realization_id>/
  axes/
    lateral_m                       # shape (n_lateral,)
    twt_highres_s                   # shape (n_twt_highres,)
    twt_model_s                     # shape (n_twt_model,)
    twt_forward_highres_s           # twt_highres_s[1:], shape (n_twt_highres-1,)
    twt_forward_model_s             # twt_model_s[1:], shape (n_twt_model-1,)
  truth/
    truth_log_ai_highres
    model_target_log_ai
    reflectivity_highres
    reflectivity_model
    rgt_highres
    rgt_model
    state_id_highres
    object_id_highres
    object_xi_highres
    geometry_event_mask_highres
    boundary_mask_highres
    state_fraction_model             # shape (n_lateral, n_twt_model, 3)
    dominant_object_id_model
    zone_id_model
    boundary_fraction_model
    boundary_mask_model
    valid_mask_model
    forward_valid_mask_highres
    forward_valid_mask_model
  priors/
    lfm_ideal
    lfm_controlled_degraded
  residuals/
    residual_vs_lfm_ideal
    residual_vs_lfm_controlled_degraded
  seismic/
    seismic_model_consistent
    seismic_from_highres_truth_model_grid  # optional forward QC
  seismic_variants/<scenario_id>/
    seismic_observed
    positive_gain
    additive_noise
    qc/
  probes/<probe_variant_id>/
    truth/
      probe_log_ai_highres
      probe_log_ai_model_grid
      reflectivity_model
    priors/
      lfm_ideal
      lfm_controlled_degraded
    residuals/
      residual_vs_lfm_ideal
      residual_vs_lfm_controlled_degraded
    seismic/
      seismic_model_consistent
      seismic_from_highres_truth_model_grid
    seismic_variants/<scenario_id>/
      seismic_observed
      positive_gain
      additive_noise
      qc/
    qc/
```

除显式多通道数组外，二维数组统一为 `[lateral, twt]`。每个 dataset 必须带
`axis_order`、shape、dtype、单位、domain、采样率、生成参数和 SHA-256；时间轴是上述
shape 的一维 float64 数组，不是标量。禁止依靠名称猜测单位或轴顺序。

阻抗、RGT、状态和 model-grid mask 绑定长度为 `N` 的 `twt_*` 轴；反射系数、地震和
`forward_valid_mask_*` 绑定长度为 `N-1` 的 `twt_forward_*` 轴。每个 dataset 必须用
显式 `axis_dataset` 属性引用对应 HDF5 轴路径。

`state_fraction_model` 的最后一轴固定按
`[low_impedance, background, high_impedance]` 排列，表示每个 model-grid cell 被各状态
覆盖的 high-resolution 比例，其 `axis_order` 固定为 `["lateral", "twt", "state"]`。
`dominant_object_id_model` 取覆盖率最大的对象；完全无效时使用 schema 中定义的无效
整数，不得用 0 同时表示合法对象和缺失。

`boundary_mask_highres` 在高分辨率对象或 zone 边界 cell 上为真。
`boundary_fraction_model` 是每个 model-grid cell 内该 mask 的 high-resolution 覆盖率，
`boundary_mask_model = boundary_fraction_model > 0`。不得对布尔边界 mask 做普通线性
低通后再任意设阈值。

probe group 不复制完整父真值，但保存针对 `base_model_target + probe_increment` 派生的
LFM 和 residual。其属性必须保存 `base_truth_dataset`、
`base_model_target_dataset`、`target_semantics` 和 `paired_zero_variant_id`；消费者通过
这些路径相加恢复目标，不得把 probe increment 单独误读为绝对阻抗。probe group 内的
`priors/` 与 `residuals/` 是 full-target 派生结果，不是增量。

`seismic_variants/<scenario_id>/` 保存同一阻抗目标下的 observed-seismic 输入变体。
`seismic_observed = positive_gain * seismic_convolved + additive_noise`；base 样本的
`seismic_convolved` 是父 group 的 `seismic/seismic_model_consistent`，probe 样本的
`seismic_convolved` 是对应 probe group 的 `seismic/seismic_model_consistent`。variant
group 不复制阻抗、LFM 或 residual，消费者通过 `sample_index.csv` 中的 source/sample
路径回到目标标签。

### 8.3 `sample_index.csv`

一行对应一个可消费样本或 patch，至少记录：

- `sample_id`、`realization_id`、`parent_realization_id`。
- `suite`、`geometry_family`、`split`、`hdf5_group`。
- patch 的横向和 TWT 范围。
- generator family、持续长度模式和六档横向场景标识。
- requested/effective 横向相关长度、属性幅度和厚度 log sigma。
- 子波与失配场景。
- 探针频率、相位、两套 noise-equivalent 基准、倍数、各采样阶段的实际 `log(AI)` RMS
  和能量占比；probe 行必须带 `sample_kind=frequency_probe` 和
  `paired_zero_sample_id`。
- LFM 版本。
- seed、状态和拒绝原因。

### 8.4 其他正式输出

- `scenario_catalog.csv`：冻结的场景、参数网格和预期用途。
- `probe_frequency_catalog.csv`：第一闸门频率选择原因、证据状态、解析支持、nominal/
  conservative noise-equivalent、有效空间簇数和子波不确定性 warning。
- `object_catalog.csv`：逐 realization、zone、对象和横向场景记录状态、持续长度、
  系数、事件角色、reversal、clipping 和轮廓验收状态。
- `frequency_probe_results.csv`：生成阶段对探针请求/实际 RMS、能量占比、两套
  noise-equivalent 基准、横向形态、配对 `0x`、正演响应和理论参数的自检，不是模型
  评测结果。
- `generation_qc.csv`：逐 realization 的范围、能量、层厚、连续性、轮廓、reversal、
  clipping、场景接受率和异常状态。
- `seismic_variant_results.csv`：逐 base/probe source 样本的 observed-seismic variant、
  gain、noise、phase/shift 参数和 RMS 统计。
- `canonical_geometry_qc.csv`：Canonical 解析几何、truth-grid 离散误差、对比度和
  pinchout 终止位置闭合记录；Field-conditioned 运行不生成该文件。
- `section_geometry_qc.csv`：Field-conditioned section 的最终层位采样、原始解释支撑和
  filled/厚度插值状态；Canonical 运行不生成该文件。
- `object_lateral_coefficients.csv`：Field-conditioned accepted realization 中每个对象、
  每个横向位置最终进入 truth 的 `c0/c1/c2`、厚度权重和条件化指标。
- `generation_rejection_details.csv`：逐拒绝规则、zone、对象记录分子、分母、阈值、
  轮廓指标和超限量；即使没有拒绝记录也必须输出稳定表头。
- `benchmark_manifest.json`：schema、输入来源、场景版本、split、seed、held-out 家族、
  阻抗校准校验值、HDF5 数据集定义及文件校验值。
- `run_summary.json`：运行参数、逐场景 attempt/接受/拒绝数量、拒绝原因、频率矩阵和
  warning。

图件至少包括：

- 真值、反射系数、nominal 地震、两类 LFM 和 mask 总览。
- 楔状体层厚与调谐响应。
- 尖灭真值位置和地震响应。
- 频率-振幅正演响应矩阵。
- Semi-Markov 状态、对象轮廓、层厚、横向相关长度和几何事件 QC。
- `0.75/1.0/1.25 sigma` 校准敏感性及 standard/ultra-thin 持续长度对比。

生成 CLI 到此为止，不加载模型预测，也不计算第 9 节模型报告卡。无效 realization 必须
保留状态和拒绝原因，不能静默丢弃或动态重采样到“刚好成功”为止。

实现诊断允许使用 `generate --qc-only`：它必须执行完整真值生成和接受率判定，但不把
accepted realization 数组持久化到 HDF5。该产物只用于场景可行性研究，manifest 必须记录
`qc_only=true`，不得作为模型训练或正式 benchmark 输入。

## 9. 评测报告卡

本节属于独立 `evaluate_synthoseis_lite.py` 的评测协议。它消费冻结 benchmark 与模型
预测，输出 schema `synthoseis_lite_report_v1`；生成 CLI 不实现这些指标。

当前首版先实现模型无关的 baseline evaluator，用于验证消费端契约：`lfm_controlled_degraded`、
`lfm_ideal` 和 `oracle_target` 三个 baseline 直接从 benchmark 产物派生。`oracle_target`
只作为管线自检，不代表模型能力。
消费端必须同时支持 `base`、`frequency_probe`、`seismic_variant` 和
`frequency_probe_seismic_variant`；probe 的 HDF5 `probe_log_ai_model_grid` 是增量，
评测目标语义为父样本 target 加 probe 增量。

首版冻结指标和数据拆分，不设置单一总分，也不预设绝对通过阈值。完成 1D、2D 和空间
约束基线后，再以新的 benchmark 版本记录相对门槛。

评测器至少输出：

- `model_sample_metrics.csv`：逐样本、逐场景的全频和分频指标。
- `model_geometry_metrics.csv`：楔体、尖灭、层边界和横向连续性指标。
- `model_probe_metrics.csv`：逐频率、振幅和相位的探针指标。
- `model_report_card.json`：按 benchmark manifest 与模型运行聚合的报告卡。
- `evaluation_summary.json`：输入模型、预测文件、schema、状态和拒绝原因。

这些文件属于 `synthoseis_lite_report_v1`，不得写回或改写生成器产物。评测器必须校验
预测数组与 `sample_index.csv`、HDF5 group、轴顺序和有效 mask 完全一致。

报告卡至少包含：

- 分频 `log(AI)` 幅度误差和相位误差。
- 全频及分频 NRMSE、相关性和均值偏差。
- 楔状体最小可分辨厚度。
- 尖灭位置误差。
- 层边界定位误差。
- 横向连续性和虚假事件率。
- Canonical `0x` 与 unsupported 探针中的绝对虚假目标频率能量。
- Field-conditioned 配对 `0x` 到非零探针的增量恢复误差。
- nominal 到各失配场景的性能退化。
- ideal LFM 与 controlled-degraded LFM 的性能差异。
- seen geometry 与 held-out geometry 的泛化差距。

正演地震误差只用于物理一致性检查，不能替代阻抗真值指标，也不能凭 waveform
correlation 单独选出模型。

## 10. 状态与失败策略

未来实现至少使用：

- `ok`
- `source_run_mismatch`
- `missing_input`
- `unsupported_schema`
- `invalid_impedance_calibration`
- `impedance_calibration_source_mismatch`
- `invalid_wavelet`
- `sampling_mismatch`
- `missing_horizon`
- `crossing_horizons`
- `outside_horizon_support`
- `section_outside_support`
- `invalid_geometry`
- `invalid_impedance`
- `invalid_layer_duration`
- `invalid_state_sequence`
- `invalid_object_profile`
- `invalid_impedance_contrast`
- `excessive_reversal_fraction`
- `excessive_clipping_fraction`
- `missing_geometry_event_target`
- `insufficient_scenario_acceptance`
- `insufficient_attempts_for_acceptance_qc`
- `lateral_correlation_length_warning`
- `section_scale_correlation_unresolved`
- `invalid_seed_contract`
- `invalid_probe_frequency`
- `invalid_probe_amplitude`
- `invalid_lfm_degradation`
- `invalid_noise_model`
- `invalid_gain_model`
- `invalid_wavelet_scenario`
- `highres_forward_qc_failed`
- `invalid_antialias_result`
- `invalid_downsample_alignment`
- `invalid_patch_window`
- `insufficient_noise_equivalent_calibration`
- `split_leakage`
- `hdf5_contract_error`
- `generation_rejected`

运行级来源矛盾、阻抗校准不匹配、schema 不支持、nominal 子波无效、split 泄漏、
命名 seed 契约失败和 HDF5/manifest 不一致应使整次运行失败。单个 section 或
realization 的地质生成问题保留拒绝记录后继续；任一冻结场景 attempt 少于 20 或接受率
低于 50% 时，整次 benchmark 生成在汇总完成后失败，不得删除该场景或动态补充 attempt。
`lateral_correlation_length_warning` 和默认可选的 `highres_forward_qc_failed` 只影响
QC 状态；后者仅在配置 `required=true` 时升级为运行失败。

## 11. 实现约束与测试

`.ref/synthoseis/` 只提供“先层序、后属性、再正演”的生成哲学。首版不得复用其 Perlin
noise、Markov、随机属性扰动或 HDF 管理代码作为数值核心，也不得复制其断层、盐体、
圈闭、AVO 或 Linux 专用流程。

未来实现至少覆盖：

1. 任意 `N >= 2` 的层位数量、任意名称和显式文件路径。
2. 层位缺失、交叉、无支持及 section 路径越界。
3. LAS 经 optimized TDT 投影后使用 truth-cell 分段线性积分平均，且长缺口严格传播。
4. `1.0 sigma` 状态分段和短对象确定性合并。
5. `0.75/1.25 sigma` 只产生校准敏感性 QC，不改变主 benchmark。
6. 两、三、四样点及以上对象分别遵守固定的 `c0/c1/c2` 拟合契约；零或近零 robust
   scale 使用父尺度下限，不产生退化 Huber 阈值或退化抽样分布。
7. 独立系数抽样后的对象均值、端点差、峰峰值和内部极值通过稳健轮廓验收。
8. zone/state 稀疏证据的系数位置、尺度、P01/P99 和持续长度上界正确收缩到全目标窗
   父先验，并完整记录原始和最终参数。
9. 转移矩阵零对角；low/high 直接跳转遵守“2 次、2 井、2 簇”支持门槛，并区分
   `zone_supported`、`parent_prior_only` 和 `forbidden`。
10. standard 与 ultra-thin stress 的最短持续长度及 split 限制严格分离。
11. 不规则横向采样下 AR(1) 在裁剪、去均值和归一化前后记录尺度与经验相关长度；
    非 section-scale 场景超过 35% 偏差时产生 warning 而不重抽。
12. requested `Lx` 小于四个横向间隔时被抬升并记录 warning。
13. 对象级调制不跨层位或对象边界污染其他状态。
14. wedge/pinchout 的方向、倍率、尖灭位置和渐消宽度与冻结真值一致。
15. reversal 的全局/单对象比例按对象均值定义，并执行 10%/25% 门槛。
16. clipping 在未裁剪真值之后计算，并执行 0.5%/2% 门槛。
17. 固定 attempt 表下，场景至少有 20 个 attempt 后才执行 80% warning 和 50% failure；
    样本不足时整次 benchmark 失败且不动态补 seed。
18. 包含 benchmark 版本、generator family 和 stream purpose 的 SHA-256 命名流与
    `PCG64DXSM` 不受循环顺序、额外 QC 或新增场景影响。
19. 8 倍超采样、抗混叠和模型网格轴严格对齐。
20. 高分辨率轴和模型网格轴分别计算精确 `tanh` 反射系数，并绑定各自长度为 `N-1`
    的 `twt_forward_*` 轴；不允许补零或与长度为 `N` 的阻抗轴隐式对齐。
21. 与正演可观测性分析相同的卷积、挂点和子波中心约定。
22. `F_model_grid(model_target_log_ai)` 与无噪声
   `seismic_model_consistent` 数值一致。
23. 可选 high-resolution 正演降采样结果成功时记录与工程采样闭合正演的差异，失败时
    遵守 `required` 开关，且任何情况下都不会替代主输入。
24. nominal 子波 dt 与输出 dt 一致；不一致时拒绝运行。
25. 子波高分重采样保持物理时长、奇数长度、中心和 L2 契约。
26. 已知频率、相位与振幅探针的请求值、high-resolution 实际值和工程采样实际值一致可审计。
27. Canonical `0x` 目标频带能量可忽略；Field `0x` 与非零探针只相差已知增量。
28. section-coherent 与中央 40% localized Tukey 探针均满足冻结横向形态。
29. 低探针能量只产生 warning，不被静默放大。
30. 高频 residual 只能由有明确名称的 truth/base 对相减派生。
31. controlled-degraded LFM 的误差 RMS、频谱和相关长度符合配置。
32. 三类正 gain 模型严格为正，且作用顺序为卷积、gain、加性噪声。
33. 父 realization 及其所有 probe/wavelet/noise/LFM/gain 变体不跨 split。
34. held-out geometry family 不出现在 train 或 validation。
35. `boundary_fraction_model` 由高分辨率边界覆盖率得到，
    `boundary_mask_model = boundary_fraction_model > 0`。
36. HDF5、对象目录、`sample_index.csv` 和 manifest 的 ID、shape、单位、axis dataset、
    axis order 及校验值一致。
37. 无效 realization 产生拒绝记录，且不会动态追加 seed。
38. `calibrate` 不生成 realization；`generate` 不重新估计井统计；评测入口不修改冻结
    benchmark。
39. 文档、导航和内部链接通过 MkDocs 严格构建。

## 12. 首版边界

首版只做二维叠后声阻抗基准，不做：

- 断层、盐体、河道或复杂沉积相。
- 叠前 AVO。
- Vp、Vs、Rho 联合岩石物理。
- 三维小体。
- 神经网络、训练器或生产反演入口。
- 永久频率 cutoff 或单一模型排行榜分数。
- 二维 GRF、层内 OU 或其他随机 micro-texture。

当前工区的运行结果可以作为配置和契约实例，但不得成为默认路径、固定层位、固定频率或
固定井数。完成本基准后，下一项工作才是在同一冻结数据与报告卡上实现最小 1D、2D 和
空间约束反演基线。

`state_threshold_sigma`、持续长度模式、横向相关长度/幅度档、wedge/pinchout 参数、
reversal/clipping 门槛或场景接受率门槛的任何变化，都必须升级 generator family 或
benchmark 版本，不能在 `object_coefficients_v1` 的同一版本下静默调整。
