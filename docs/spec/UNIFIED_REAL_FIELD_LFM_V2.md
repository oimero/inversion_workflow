# 统一真实工区 LFM v2 架构规范

> 状态：已实施（测试由用户本地执行）
> 范围：Step 6 真实井控制准备、Step 7 时间/深度域 LFM variant 构建，以及 R0/GINN-v2 real-delta 接缝
> 迁移策略：直接切换 v2，不保留 `real_field_lfm_v1` 双读兼容
> 本文是新的权威实施规范。时间域和深度域不得再各自建设独立 LFM 生产管线。

## 1. 目标与基本判断

真实工区 LFM 不是一种算法，而是两类背景构建方法和一类场景修饰能力：

- **trend baseline**：高偏差、低方差，以少量纵向参数表达稳定背景；
- **proportional-kriging baseline**：降低纵向结构偏差，但更依赖井分布、变差函数和切片控制；
- **framework modifier**：显式注入框架场景。框架正确时可能改善初始模型，框架错误时会主动注入系统误差，因此不能被称为天然更优的 baseline。

统一架构固定为：

```text
Step 6 real_field_well_controls
    └─ canonical WellControlSet
            ├─ Step 7 trend baseline
            └─ Step 7 proportional_kriging baseline
                         ↓
                  zero or more modifiers
                         └─ framework scenario
                                  ↓
                       explicit LFM variants
                                  ↓
                 variant-specific label preparation
                                  ↓
                     R0 / GINN-v2 real-delta
```

v2 必须满足：

- 同一套 builder 和 modifier 同时支持 TWT 与 TVDSS；
- 时间/深度差异只出现在 source adapter、采样轴单位和正演/导出 adapter；
- 同一次 LFM 运行的所有 variant 共享同一个 WellControlSet、TargetZone 和输出几何；
- framework 是 modifier，不是第三种 baseline；
- 每个 variant 使用描述性 `variant_id`，禁止继续使用 `M0`、`M1` 作为模型名称；
- LFM 主数组统一为自然对数 `log(AI)`，线性 AI 单位为 `m/s*g/cm3`；
- 所有来源、配置、控制集、层位、地震、方法链和产物哈希可追溯；
- 不通过路径扫描、字段猜测或错域兼容选择算法或 source adapter。

## 2. 非目标

本规范不处理：

- 合成训练集的 LFM variant 生成与 split 策略；
- 自动判断 trend、proportional kriging 或 framework 谁“最好”；
- 自动识别礁滩、自动生成 framework body 或自动估计 AI 倍率；
- 修改 GINN-v2 模型输入通道或 residual 定义；
- Petrel 插件、多边形编辑器或解释软件交互 UI；
- 继续维护 `real_field_lfm_v1`、`lfm_time.py` 或旧 LFM modeling API 的长期兼容门面；
- 把岩石物理分析重新放回主链。岩石物理分析继续是 Step 3 下游的旁路。

## 3. 模块边界

规划的新生产边界为：

```text
scripts/real_field_well_controls.py   # Step 6 唯一入口
scripts/real_field_lfm.py             # Step 7 唯一入口
src/cup/well/real_field_controls.py   # source adapter 与 WellControlSet 契约
src/cup/seismic/lfm/                  # builder、modifier、公共数学和产物编排
```

`cup.seismic.lfm` 公开且仅公开下列核心概念：

- `LfmContext`：SurveyLineGeometry、SampleAxis、TargetZone、输出几何和公共哈希；
- `LfmBuilder`：从 WellControlSet 构建 baseline 的协议；
- `LfmModifier`：从已有 LFM result 派生场景 variant 的协议；
- `LfmVariantResult`：公共主数组、有效掩码、method/modifier sidecar 和 QC；
- `build_lfm_variants()`：验证显式 variant 图并原子构建全部请求产物；
- 共享的 `apply_lfm_lowpass()` 与 XY ordinary kriging 原语。

builder 和 modifier 不读取 source run，不决定输出目录，也不写主 summary。source 发现、CLI、配置合并和原子发布由脚本/编排层负责。

现有能力的迁移原则：

- `src/cup/seismic/real_field_lfm.py` 中 trend 拟合、参数场和 QC 的可用逻辑迁入新包；
- `src/cup/seismic/modeling.py` 中比例切片、逐道映射和 completion 数学迁入或改造成新包的公共实现；
- `src/cup/seismic/lfm_time.py` 的时间控制点适配由 Step 6 canonical control 取代；
- 迁移完成并切换消费者后删除上述旧生产模块/API，不保留转发门面；
- `TargetZone`、`SampleAxis`、`SurveyLineGeometry`、`grid.Log` 和 `volume_export` 继续作为公共基础设施。

## 4. Step 6：Canonical WellControlSet

### 4.1 职责

Step 6 只负责把不同上游产物转换成相同的真实井控制事实：

```text
source run
  -> log(AI) 域转换
  -> 目标地震 SampleAxis 对齐
  -> 逐样点井轨迹坐标
  -> 有效性与来源审计
  -> canonical WellControlSet
```

Step 6 不读取任何 LFM，不计算 delta，不生成 `valid_for_fit`，也不执行训练空间聚类。否则 WellControlSet 会与某个 LFM variant 形成循环依赖。

### 4.2 Source adapter

配置必须显式声明以下一种 `source_run_type`：

| `source_run_type` | 目标域 | 必需上游 |
|---|---|---|
| `well_auto_tie` | `time + s` | `well_tie_metrics.csv`、`filtered_las_file`、`optimized_tdt_file`、斜井的 `optimized_trace_sample_plan_file` |
| `wavelet_batch_synthetic_depth` | `depth + tvdss + m` | `wavelet_batch_metrics.csv`、`shifted_filtered_las_path`、well inventory 和项目井轨迹 |

adapter ID、上游 summary 的 schema/domain 以及顶层 `seismic.domain/depth_basis` 必须一致。禁止根据目录名、CSV 某个偶然字段或 LAS 文件名自动猜 adapter。

时间 adapter：

1. 只接受 Step 4 标定成功且路径/summary 可消费的井；
2. 用优化 TDT 将 MD 域 AI 映射到目标 TWT SampleAxis；
3. 斜井使用 trace sample plan 的逐样点 inline/xline/XY；
4. 直井把 inventory 位置广播到全部有效 TWT 样点；
5. 不跨 LAS、TDT 或轨迹无效段插值。

深度 adapter：

1. 只接受 depth source run 中状态成功且路径/summary 可消费的井；
2. shifted filtered LAS 的 basis 仍按 MD 解释；
3. 通过 `cup.well.trajectory` 的项目级轨迹语义映射到 TVDSS 和逐样点 XY；
4. 当前直井只是 TVDSS 轨迹的常横向位置特例，不在新模块内散写通用斜井的 `MD-KB` 替代算法；
5. 输出严格对齐目标地震的 TVDSS SampleAxis，缺少轨迹或 LAS 支撑的样点为无效。

### 4.3 内存契约

每口井的 canonical control 至少包含：

```text
well_name
sample_axis: SampleAxis
log_ai: grid.Log                 # basis 与 sample_axis 完全一致
inline_by_sample: float64[N]
xline_by_sample: float64[N]
x_m_by_sample: float64[N]
y_m_by_sample: float64[N]
valid_mask: bool[N]
wellbore_class
sampling_mode
source_run_type
provenance
```

`grid.Log` 的 basis type 必须与 `SampleAxis.domain` 一致：时间域为 TWT，深度域为 TVDSS。`valid_mask=true` 时 logAI 和四个位置数组必须有限；AI 原值必须为正，之后才可取自然对数。

### 4.4 磁盘契约

Step 6 输出：

```text
well_control_manifest.csv
wells/<normalized_well_name>.npz
run_summary.json
```

每井 NPZ 固定包含：

```text
samples             float64 [N]
log_ai              float32 [N]
inline              float64 [N]
xline               float64 [N]
x_m                 float64 [N]
y_m                 float64 [N]
valid_mask          bool    [N]
metadata_json       scalar UTF-8 JSON
```

无效样点的 `log_ai/inline/xline/x_m/y_m` 必须为 NaN，不得端点延伸。NPZ 禁止 pickle/object array。

`well_control_manifest.csv` 一行一口候选井，至少包含：

```text
well_name
status
reason
source_run_type
source_run_path
source_summary_path
source_summary_sha256
source_las_path
source_las_sha256
source_transform_path
source_transform_sha256
wellbore_class
sample_domain
sample_unit
depth_basis
sampling_mode
n_samples
n_valid_samples
sample_min
sample_max
well_npz_path
well_npz_sha256
```

不适用字段写空字符串，不用臆造路径。失败井保留 manifest 行但不写可消费的 well NPZ。成功 run 至少需要一口有效井；具体 baseline 能否建模由 Step 7 决定。

`run_summary.json` schema 固定为 `real_field_well_controls_v2`，记录配置、source adapter、输入/输出哈希、SampleAxis、候选/成功/失败井统计和产物清单。

## 5. Step 7 配置与 variant 图

### 5.1 单一入口

唯一生产命令规划为：

```powershell
python scripts/real_field_lfm.py `
  --config <experiment-yaml> `
  --output-dir <optional-output-dir>
```

脚本从顶层 workflow 读取地震 domain、depth basis、层位和 survey geometry，从 `real_field_lfm.source_runs.well_control_run_dir` 读取 Step 6。禁止新增 `real_field_lfm_depth.py`。

### 5.2 配置结构

统一实验配置最小结构：

```yaml
workflow_config: experiments/common/common.yaml

real_field_well_controls:
  source_run_type: <well_auto_tie|wavelet_batch_synthetic_depth>
  source_run_dir: <explicit-path-or-empty-for-schema-aware-discovery>
  well_inventory_file: <path>

real_field_lfm:
  source_runs:
    well_control_run_dir: <step-6-run>

  output_geometry:
    mode: <volume|window|section>
    # window/section 参数按 mode 显式配置

  baselines:
    trend_main:
      method: trend
      filter:
        enabled: <true|false>
        cutoff_hz: <time-only>
        cutoff_wavelength_m: <depth-only>
        order: <positive-int>
        buffer_mode: <reflect|edge|none>
        buffer_axis_units: <non-negative-number>
      fit:
        min_valid_samples_per_well: <positive-int>
        huber_f_scale_log_ai: <positive-number>
      spatial:
        variogram: <spherical|exponential|gaussian>
        exact: <true|false>
        nugget: <non-negative-number>

    slice_main:
      method: proportional_kriging
      filter:
        enabled: <true|false>
        cutoff_hz: <time-only>
        cutoff_wavelength_m: <depth-only>
        order: <positive-int>
        buffer_mode: <reflect|edge|none>
        buffer_axis_units: <non-negative-number>
      n_slices: <integer>=2
      spatial:
        variogram: <spherical|exponential|gaussian>
        exact: <true|false>
        nugget: <non-negative-number>

  modifiers:
    reef_case_a:
      method: framework
      bodies_file: <framework-body-csv>
      classes:
        reef:
          top_horizon: <name>
          bottom_horizon: <name>
          linear_ai_multiplier: <positive-number>
          edge_taper_m: <positive-number>
          top_taper_fraction: <number-in-(0,0.5)>
          bottom_taper_fraction: <number-in-(0,0.5)>

  variants:
    - variant_id: trend
      baseline_id: trend_main
      modifier_ids: []
    - variant_id: trend_reef_case_a
      baseline_id: trend_main
      modifier_ids: [reef_case_a]
    - variant_id: proportional_slice
      baseline_id: slice_main
      modifier_ids: []

  comparisons:
    - comparison_id: trend_vs_slice
      left_variant_id: trend
      right_variant_id: proportional_slice
    - comparison_id: trend_framework_effect
      left_variant_id: trend
      right_variant_id: trend_reef_case_a
```

尖括号是在其配置分支活动时的必填占位，不是默认值；例如 `filter.enabled=false` 时 cutoff/order/buffer 字段必须整体不存在。配置不得自动生成 baseline×modifier 笛卡尔积。modifier 按 `modifier_ids` 顺序应用并写入 metadata；当前只注册 framework，但协议允许未来新增显式 modifier。

`variant_id`、baseline ID、modifier ID 和 comparison ID 必须全局唯一、非空且可安全用作目录名。`M0`、`M1` 及仅由这类编号派生的名称不满足描述性命名要求，应拒绝。

### 5.3 输出几何

统一支持：

- `volume`：完整 survey 规则体；
- `window`：落在显式 inline/xline 轴上的矩形子体；
- `section`：由配置路径定义的二维剖面。

同一 run 的全部 variant 必须共享完全相同的输出几何和 `valid_mask_model`。window/section 只裁输出，不减少 baseline 可使用的控制井集合。inline/xline 始终是真实线号；xline 步长 4 不能被解释为数组步长 1。

## 6. 公共低通与空间建模

### 6.1 Builder-owned filter

低通属于 baseline 方法配置，不属于 Step 6 井控事实。每个 builder 可配置不同滤波参数，但必须调用同一个 `apply_lfm_lowpass(grid.Log, LowpassSpec)` 实现。

域自然单位固定为：

```text
time:  cutoff_cycles_per_second = cutoff_hz
depth: cutoff_cycles_per_metre  = 1 / cutoff_wavelength_m
```

规则：

- 时间域只允许 `cutoff_hz`，出现 `cutoff_wavelength_m` 失败；
- 深度域只允许 `cutoff_wavelength_m`，出现 `cutoff_hz` 失败；
- `enabled=false` 时禁止同时给 cutoff/order/buffer 参数；
- 每个连续有限 run 独立滤波，不跨 NaN gap；
- 输出仍是同 basis、同 domain、同单位的 `grid.Log`；
- 不修改或覆盖 Step 6 NPZ。

### 6.2 共享 XY ordinary kriging

trend 参数场和 proportional slice 场必须调用同一个 XY kriging 原语：

- 控制点和输出网格都通过 `SurveyLineGeometry` 使用真实 XY 米制坐标；
- 不以 inline/xline 数值差或数组 index 作为物理距离；
- variogram、nugget、exact 和实际 range/sill 全部写入 QC；
- sill 固定为当前切片或参数的有限控制值总体方差；
- range 固定为 `max(median(nearest_neighbor_distance_xy_m), nominal_bin_spacing_m)`；
- 二者不从时间/深度域分别派生，不设置隐式 anisotropy、搜索半径、象限或最大控制点数；
- 至少两个控制值存在但全部相同/近似相同，或计算得到非正 sill 时失败；禁止把 sill 修补为 1 或任意常数。

近似常值判定必须使用实现中冻结并记录的确定性 `rtol/atol`，不得依赖库版本的隐式默认值。

## 7. Baseline builder

### 7.1 Trend

trend 保留当前高偏差、低方差语义。每口有效井在首层位到末层位的整个目标窗内计算：

```text
u_total = (sample - top_horizon_at_well) /
          (bottom_horizon_at_well - top_horizon_at_well)
x = 2*u_total - 1
logAI = a + b*x
```

每井使用 Huber 回归得到 `a/b`；中间层位只参与 TargetZone 几何和连续性 QC，不增加趋势参数。随后分别在 XY 中插值 `a_field/b_field`，逐道按首末层位重建体。

空间控制语义：

- 无有效井：variant 失败；
- 恰好一口有效井：a/b 分别生成 `single_control_constant` 场并记录零方差语义；
- 至少两口井且某参数控制值退化为常值：variant 失败；
- 其他情况执行共享 XY kriging。

method sidecar 至少保存 `a_field`、`b_field`、对应 variance、`distance_to_control_m` 和控制/拟合 QC。

### 7.2 Proportional kriging

对 TargetZone 的每个相邻层位段定义：

```text
u_k = k / (n_slices - 1),  k=0..n_slices-1
sample_well(k) = (1-u_k)*top_well + u_k*bottom_well
```

每个切片从 builder 自己滤波后的 canonical logAI 取值，并使用该样点处的逐样点 XY 作为空间控制位置。斜井不能压缩成一个固定井口位置。

completion 行为固定保留旧数学并强制审计：

| 有限控制数 | 行为 | mode |
|---:|---|---|
| 0 | 若同层段存在其他有效切片，则由最近上下有效切片线性补齐；仅有单侧时复制最近切片 | `neighbor_slice_fill` |
| 1 | 全平面使用该控制值 | `single_control_constant` |
| >=2 且非退化 | 共享 XY ordinary kriging | `kriging` |
| >=2 且控制值近似常值，或 sill 非正 | 失败 | 无产物 |

若整个层段没有任何有效切片，variant 失败。补齐不得跨越层段边界。每个切片必须记录原始控制数、控制井、原始/最终 mode、上下来源切片、插值权重和实际 variogram 参数。

逐道映射只写入该层段的 TargetZone 有效样点。取消 `_extend_volume_constant_outside_modeled_range` 类型的目标层外首尾延拓。

method sidecar 至少保存 kriging variance、slice 轴和可机器读取的 slice mode/QC；复杂表格可放入 CSV，NPZ 不嵌套对象。

## 8. Framework modifier

framework modifier 可作用于任一 baseline，输出父 variant 的显式派生体。它不改变 `valid_mask_model`，也不声称严格保持井值。

### 8.1 Body CSV

CSV 固定字段：

```text
body_id,framework_class,u_top,u_bottom,vertex_order,inline,xline
```

一个 `body_id` 定义一个平面 polygon 和一个母层段内的局部纵向窗。必须满足：

- 同一 body 的 class、`u_top/u_bottom` 逐行一致；
- `0 <= u_top < u_bottom <= 1`；缺失时不得默认成 `[0,1]`；
- polygon 至少三个互异顶点，顺序连续，不自交、不退化、不越出 survey；
- 顶点是实际线号，不是数组 index；
- 同一平面位置允许多个具有不同纵向窗的 body。

### 8.2 概率与修改公式

每个 body 先独立计算：

```text
u_zone = (sample - zone_top) / (zone_bottom - zone_top)
v_body = (u_zone - u_top) / (u_bottom - u_top)

P_body = P_map_body * P_vertical_body(v_body)
P_class = max(P_body for bodies in class)
```

横向 edge taper 使用真实 XY 米制距离并仅向 polygon 内渐变；圈外严格为 0。垂向 raised-cosine taper 相对于 body 自身厚度计算。必须先形成三维 body，再在同 class 内取最大值；禁止先合并二维 map、概率求和或相加截断。

modifier 在 logAI 域应用：

```text
logAI_modified = logAI_parent
               + sum(P_class * log(linear_ai_multiplier_class))
```

同一 class 的 body 共用该 class 的 multiplier 和 taper。framework sidecar 保存 class probability、二维 QC map、body/class QC 和修改统计；不要求保存每个 body 的全体积三维概率。

## 9. Variant 与产物契约

### 9.1 原子发布

所有显式请求的 baseline、modifier、variant 和 comparison 必须在临时运行目录中全部成功后，才发布可消费的 `variant_manifest.csv` 与成功 summary。任一请求失败时：

- 整次 run 状态为 `failed`；
- 不发布可被 R0 自动发现的 manifest；
- 可保留唯一失败原因和诊断日志；
- 不把已完成的部分 variant 宣称为成功运行。

### 9.2 目录结构

```text
lfm_run_summary.json
variant_manifest.csv
comparisons/
  <comparison_id>/metrics.csv
  <comparison_id>/figures/
variants/
  <variant_id>/
    lfm.npz
    method_fields.npz
    modifier_fields.npz       # 仅含 modifier 时
    variant_summary.json
    qc/
```

`lfm_run_summary.json` schema 固定为 `real_field_lfm_run_v2`，记录原子运行状态、完整解析配置、WellControlSet/地震/层位哈希、输出几何、请求的 baseline/modifier/variant/comparison、产物清单和哈希。成功状态只允许在 manifest 已发布后写为 `ok`。

### 9.3 公共主 NPZ

每个 `lfm.npz` 只包含跨方法公共字段：

```text
log_ai              float32 [geometry-dependent]
valid_mask_model    bool    [same shape]
ilines              float64 [n_inline or n_trace]
xlines              float64 [n_xline or n_trace]
samples             float64 [n_sample]
metadata_json       scalar UTF-8 JSON
```

volume/window 的 `log_ai` shape 为 `[n_inline,n_xline,n_sample]`；section 为 `[n_trace,n_sample]`。主 NPZ 禁止出现 a/b、kriging variance、framework probability 等可选方法字段。

`metadata_json` schema 固定为 `real_field_lfm_variant_v2`，至少包含：

```text
variant_id
baseline_id
baseline_method
modifier_chain
sample_domain
sample_unit
depth_basis
value_key=log_ai
value_domain=log(AI)
linear_ai_unit=m/s*g/cm3
valid_mask_key=valid_mask_model
output_geometry
well_control_run_path_and_hash
seismic_path_and_hash
horizon_paths_and_hashes
resolved_baseline_config_and_hash
resolved_modifier_configs_and_hashes
method_sidecar_path_and_hash
modifier_sidecar_path_and_hash
```

规范 `log_ai` 在 `valid_mask_model=true` 处必须有限，在 false 处必须为 NaN。所有 variant 的 axes、shape 和 mask 必须逐点一致。

### 9.4 Manifest 与 comparison

`variant_manifest.csv` 一行一 variant，至少记录 ID、baseline method、modifier chain、状态、主 NPZ/sidecar/summary 路径和哈希。不存在隐式 `primary` variant。

comparison 只按配置中的显式 pair 生成。每一对要求同网格同 mask，输出线性 AI、logAI、差值、百分比差、井旁指标和配置剖面图。comparison 不输出 winner/best 字段，也不自动把左侧解释为基准。

### 9.5 体导出

只有 `volume` 模式允许导出解释软件体。导出值为 `exp(log_ai)` 的线性 AI，文件名包含 `variant_id`，textual/metadata header 写出 variant、domain、单位、主 NPZ 路径和哈希。

导出复用 `volume_export` 的格式 adapter，并要求输出全轴与源地震严格一致。`nan_fill=None`；mask 外 NaN 不静默改成 0 或端点值。若具体解释软件不接受 NaN，应另立 display-only 产物规范，不能改变 v2 主产物。

## 10. R0 与 real-delta 接缝

### 10.1 显式选择

R0 v2 配置必须同时提供：

```yaml
real_field_inputs:
  lfm_run_dir: <step-7-run>
  variant_id: <explicit-variant-id>
  well_control_run_dir: <step-6-run>
```

R0 不接受默认 variant、目录中第一个 variant 或文件名推断。指定 variant 必须存在于已成功发布的 manifest，且其 WellControlSet 哈希必须与配置指定的 Step 6 run 一致。

### 10.2 Variant-specific label preparation

`build_well_anchor_samples` 的职责拆分为：

1. 读取 canonical WellControlSet；
2. 在逐样点 inline/xline/sample 位置三线性采样所选 variant 的 `log_ai` 和 mask；
3. 计算：

   ```text
   target_delta = well_log_ai - sampled_lfm_log_ai
   ```

4. 根据井控制、LFM、mask 和 survey 支撑生成 `valid_for_fit`；
5. 按 real-delta 配置执行空间聚类和 balanced sampling 所需统计；
6. 写入 variant ID、LFM 哈希和 WellControlSet 哈希。

不同 variant 必须各自生成 label artifact；禁止复用另一 variant 的 `lfm_log_ai`、delta、有效性或缓存。一次 R0 运行只使用一个 variant。

## 11. v1 直接迁移

本规范不设置双读期。实施时必须在同一迁移阶段完成：

1. Step 6 canonical control producer；
2. Step 7 v2 variant producer；
3. R0 resolver、NPZ validator 和 label preparation 切换 v2；
4. CSV/JSON 契约、guide 和配置示例更新；
5. 删除 `real_field_lfm_v1` 自动发现和 schema 接受逻辑；
6. 删除旧 `cup.seismic.real_field_lfm`、`cup.seismic.lfm_time` 和旧 modeling LFM 生产 API；
7. 全仓搜索确认无生产代码继续导入旧入口。

旧 v1 产物只能作为明确的历史数据存在；新生产代码遇到时必须报错要求重建，不能静默升级或猜测 variant。

## 12. 严格失败规则

以下情况必须失败：

- source adapter ID 与 source summary/domain 不一致；
- Step 6 或 Step 7 来源缺文件、状态不可消费、路径不一致或哈希不匹配；
- AI 单位错误、非正值、basis/domain/depth basis 不一致；
- SampleAxis 非有限、非严格递增、不规则或与 survey 不一致；
- 斜井缺逐样点轨迹，或轨迹/AI 无共同支撑；
- xline 步长 4 被按 1 或数组 index 解释；
- filter 使用错域 cutoff、跨 NaN gap 或修改 canonical controls；
- variant 引用不存在的 baseline/modifier，ID 重复，或形成隐式/自动组合；
- requested comparison 引用不存在或不同网格的 variant；
- 整层无有效 proportional slice；
- 至少两个控制却退化为常值场，或 sill 非正；
- framework body 字段、拓扑、层段映射、倍率或 taper 不合法；
- 任一 variant 在 mask 内非有限、在 mask 外有限，或 variant 间 mask/axes 不一致；
- 仅部分请求 variant 成功却尝试发布 manifest；
- R0 未显式指定 variant，或 WellControlSet/LFM 哈希链不一致；
- 尝试消费 `real_field_lfm_v1` 或旧无 schema 产物。

禁止：

- 按目录名或 CSV 猜 `source_run_type`；
- 把 filtered LFM、sampled LFM、delta 或 cluster 写回 WellControlSet；
- 为比较方便自动改变某个 baseline 的 filter 或控制井集合；
- 用线号差代替 XY 米制距离；
- 修补非正 sill；
- 把 framework 描述为更高等级或默认更优的 LFM；
- 给 comparison 自动赋予 winner；
- 在目标层外做首尾常值延拓；
- 为兼容旧 R0 写 v1 镜像产物。

## 13. 实施顺序

### 阶段 A：Step 6

1. 抽取时间域 anchor 中可复用的 MD→TWT 与逐样点轨迹逻辑；
2. 实现 time/depth source adapter 和 canonical 内存对象；
3. 写 manifest、逐井 NPZ、summary 与来源哈希；
4. 冻结 `real_field_well_controls_v2` 契约。

门禁：Step 6 不导入、读取或采样任何 LFM。

### 阶段 B：公共数学和 baseline

1. 建立 `cup.seismic.lfm` 包与公共 result/context；
2. 实现 builder-owned、shared-primitive lowpass；
3. 统一 XY kriging；
4. 迁移 trend；
5. 迁移 proportional slicing 与已锁定 completion modes。

门禁：两种 builder 在 TWT/TVDSS 均通过同一公共接口；无 domain 专用 builder 副本。

### 阶段 C：modifier、variant 和产物

1. 迁移 framework body 契约；
2. 实现显式 variant 图、sidecar 和 comparisons；
3. 实现原子发布、统一主 NPZ 和体导出；
4. 验证所有 variant 同轴同 mask。

### 阶段 D：R0 切换与清理

1. R0 改为 `lfm_run_dir + variant_id + well_control_run_dir`；
2. label preparation 改为 Step 6 + selected variant；
3. 更新 guide/CSV 契约；
4. 删除 v1 consumer 和旧模块/API。

门禁：全仓不存在活跃的 v1 生产或自动消费路径。

## 14. 测试规范

测试由实现方写入 `tests/`，用户本地运行。至少覆盖：

1. time/depth source adapter 输出相同 WellControlSet schema；
2. time TDT 投影、depth TVDSS 轨迹、直井和斜井逐样点位置；
3. inline step=1、xline step=4 的线号、XY 和窗口裁取；
4. Step 6 产物不含 LFM、delta、`valid_for_fit` 或 cluster；
5. manifest/逐井 NPZ 的 dtype、shape、NaN、hash 和 provenance；
6. 时间 Hz、深度 wavelength cutoff 经过同一低通实现，错域参数失败；
7. NaN gap 分 run 滤波且不跨缺口；
8. trend 在 TWT/TVDSS 上共享公式和 XY kriging；
9. proportional kriging 在 TWT/TVDSS 上共享切片和逐道映射；
10. 0 控制 neighbor fill、1 控制 constant、>=2 kriging 的 mode 与来源审计；
11. 整层无控制、>=2 常值控制和非正 sill 失败；
12. framework 可分别修饰 trend 和 proportional baseline；
13. 同位置多个 framework body、同类 max 聚合和圈外零概率；
14. 显式 variant、显式 comparison、禁止自动笛卡尔积；
15. 任一 requested variant 失败时不发布可消费 manifest；
16. 各 variant 主 NPZ 公共键一致，方法字段只在 sidecar；
17. variant axes/mask 逐点一致，mask 外 logAI 为 NaN；
18. window/section/volume 的公共算法一致，只有 volume 可导出体；
19. R0 缺 variant ID、错误 ID 或哈希链不一致时失败；
20. 同一井在不同 variant 下产生不同且可逐点重建的 delta label；
21. label preparation 的 `valid_for_fit` 和空间聚类不回写 Step 6；
22. v1 schema 明确拒绝，旧模块/API 删除后无残留 import。

## 15. 验收标准

v2 完成需同时满足：

- Step 6 是时间/深度真实井控制的唯一规范化入口；
- Step 7 只有 `scripts/real_field_lfm.py` 一个生产入口；
- trend 与 proportional kriging 是 domain-neutral builder；
- framework 是可作用于任一 baseline 的 modifier；
- 一次 run 可从同一控制集构建多个显式 variant；
- 所有空间建模使用真实 XY 米制距离；
- builder filter 参数可不同，但调用同一个实现；
- completion mode 全部可追溯，退化 sill 不被修补；
- 主 NPZ 跨方法同构，专属字段进入 sidecar；
- mask 外 NaN，不再生成目的层外常值延拓；
- R0 每次显式绑定一个 variant，并重新生成该 variant 的 real-delta label；
- v1 consumer、旧 LFM 模块和旧生产 API 已删除；
- 合成 LFM variant 未被本规范伪装为已经接入。
