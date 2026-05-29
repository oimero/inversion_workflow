# 06 时间域点云低频模型

`lfm_precomputed.py` 是时间域工作流的第六步。它把第四步标定成功的井在目标层内的曲线样点转成空间控制点，再通过层位约束插值生成 AI 低频模型，供第七步 GINN 训练直接读取。

本步的核心不再是"一口井一个控制点"，而是"目标层内的样点控制"——每口井在目的层内贡献数十到上百个带空间坐标的 AI 样点，斜井样点随轨迹分布在不同的 inline/xline 上。

第一版只输出 **AI LFM**。Vp、Rho、well constraints、dynamic gain 和 enhance 都不属于本步主线。

---

## 快速开始

```bash
python scripts/lfm_precomputed.py
python scripts/lfm_precomputed.py --config experiments/common.yaml
python scripts/lfm_precomputed.py --output-dir scripts/output/lfm_precomputed_test
```

不带参数时，脚本自动发现最新的第四步和第五步产物，按时间戳创建输出目录。

---

## 运行前需要什么

第六步消费第四步的井震标定事实和第五步的全局子波批量合成 QC，不重新做曲线滤波、时深转换或子波扫描。

| 来源 | 内容 | 用途 |
|------|------|------|
| 第四步 | 标定成功/失败、路由、优化后时深表、滤波 LAS | 筛选候选井、读取 AI 曲线、获取井位坐标 |
| 第四步 | 斜井样点 TWT/MD/XY/inline/xline 映射 | 斜井控制点的空间事实来源 |
| 第五步 | 全局子波批量合成指标 | 逐井合成质量，用于筛选控制井 |
| 地震数据 | 时间域地震体 + 顶底解释层位 | 提供时间轴、工区几何和目标层 mask |

直井用井口坐标配合优化后时深表将 MD 域曲线映射到 TWT 域后生成控制点；斜井优先复用第四步写出的样点级空间映射，不重新写一套轨迹反查逻辑。斜井缺少空间映射时脚本会跳过该井并记录原因，不会将它降级成井口直井控制。

---

## 配置参考

```yaml
lfm_precomputed:
  source_runs:
    mode: latest
    well_auto_tie_dir: null
    global_wavelet_generation_dir: null

  seismic:
    file: null
    type: segy

  target_interval:
    top_horizon: null
    bottom_horizon: null
    twt_unit: auto

  control_wells:
    min_batch_corr: 0.35
    max_batch_nmae: null
    include_wells: null
    exclude_wells: []

  controls:
    sample_step_s: null
    min_control_samples_per_well: 16
    vertical_source: well_head_trace
    deviated_source: trace_sample_plan

  modeling:
    boundary_extension_samples: 50
    n_slices: 20
    variogram: spherical
    exact: true
    nugget: 0.0
    filter_cutoff_hz: 10.0
    filter_order: 6
    filter_buffer_seconds: null
    filter_buffer_mode: reflect
    post_slice_smoothing: false

  export:
    write_segy: true
    write_zgy: true
```

### `source_runs`

`mode: latest` 使脚本在 `output_root` 下按前缀和必需文件自动发现上一次的第四步和第五步产出。显式指定目录时可设为非 `latest` 模式。

### `control_wells`

控制井必须先通过第五步的全局子波批量合成 QC：

- `min_batch_corr`：批量合成相关系数低于此值的井被排除。
- `max_batch_nmae`：可选 NMAE 上限，为空时不按 NMAE 过滤。
- `include_wells`：白名单模式，只使用指定井。
- `exclude_wells`：人工排除可疑井。

此外，井还必须同时具备第四步产出的滤波 LAS 和优化后时深表；斜井还需要对应的空间映射文件。

### `controls`

`sample_step_s: null` 表示使用地震体的时间采样间隔，不另行重采样。设为正值时按给定间隔沿 TWT 轴抽取控制点。

本步产出的控制点以 `inline_float`、`xline_float` 和 `twt_s` 为规范坐标。`flat_idx` 和 `sample_index` 是派生字段，只在当前地震几何和采样轴下有效，不应被下游脚本当作规范坐标依赖。

直井的 `source` 标记为 `vertical_trace`，所有控制点共用井口 XY。斜井的 `source` 标记为 `deviated_trajectory`，每个控制点的 XY 来自第四步的空间映射。

### `target_interval`

`twt_unit: auto` 表示按解释层位值的量级自动判断单位：值量级像毫秒则转为正秒，否则按正秒使用。解析结果必须与地震时间轴同单位。也可显式指定为 `s` 或 `ms`。

### `modeling`

`filter_cutoff_hz` 是时间域低通截止频率。它对控制点曲线在 TWT 域做 Butterworth 零相位低通滤波，去除高频噪声后再进入插值。`filter_buffer_seconds` 为 `null` 时自动按截止频率估算缓冲长度，`filter_buffer_mode` 控制缓冲区的延拓方式。

---

## 脚本在做什么

脚本分四个阶段：**前置发现 → 控制点生成 → 层位约束建模 → 导出**。

### 第一阶段：前置发现

1. 从配置或自动发现中定位第四步和第五步的产出目录。
2. 打开地震体，校验采样域为时间域且单位为秒。
3. 读取顶底解释层位，构建目标层并输出层位 QC（mask 有效性、层厚、交叉、薄层）。

### 第二阶段：控制点生成

对第四步标定成功的每口井，先过第五步 QC 门槛，再过资产完整性检查，然后按井型分两路生成控制点：

**直井路径**

1. 从滤波 LAS 读取 AI 曲线。
2. 用优化后时深表将 MD 域 AI 转成 TWT 域。
3. 对 TWT 域曲线做低通滤波。
4. 在时间轴的每个样点处，判断是否落入目标层内；若是，记录该点的 TWT、AI、MD、zone 和层内比例位置 `u_in_zone`。

**斜井路径**

1. 从滤波 LAS 读取 AI 曲线。
2. 读取第四步的空间映射文件，只保留工区内样点。
3. 按 TWT 排序后可选重采样。
4. 在空间映射给出的每个轨迹样点处，插值 AI 值并做低通滤波。
5. 判断每个样点是否落入目标层内，记录其空间坐标、TWT、AI、zone 和 `u_in_zone`。

两路共用同一条 AI 低通滤波管线：先检查数据量是否足够（至少 4 个以上有效样点），然后规整化到均匀网格、应用 Butterworth 零相位低通滤波、再插值回原始 TWT 位置。

生成的控制点写入 `lfm_layer_control_points.csv`；每口井的过滤结果和统计写入 `lfm_control_qc.csv`。

### 第三阶段：层位约束建模

建模核心是将三维目标层分解为沿顶底界面按比例离散的二维切片，在每张切片上做空间插值，再将切片序列重构成三维体。

1. **比例切片离散。** 将目标层沿顶底界面之间的方向均匀划分为若干层段，每个层段再沿层内比例方向（`u_in_zone` = 0 到 1）等间隔划分为切片。切片数量由 `n_slices` 控制，越多则垂向分辨率越高。

2. **控制点分配。** 每个控制点按其 `u_in_zone` 值分配到最近的切片。切片之间的分界取相邻两个切片中心的中点。

3. **切片插值。** 对每张切片，收集所有分配到该切片的控制点的 `inline_float`、`xline_float` 和 AI 值。若该切片只有一个控制点，整张切片填为该常数；若有多个控制点，用普通克里金在 inline/xline 平面上插值出完整切片。

   切片内的重复坐标（密井网下多个控制点落在同一个浮点线号）先按控制点权重做加权平均聚合，再进入插值。

4. **缺失切片填充。** 某些切片可能没有任何控制点（特别是目标层顶部或底部），此时沿比例方向向上、向下搜索最近的有效切片，用其值填充。

5. **边界扩展。** 按 `boundary_extension_samples` 在目标层最顶层之上和最底层之下各扩展一段。扩展区若没有直接控制值，则从相邻原始层段的边界切片做常值延拓。

6. **体重建。** 将所有切片的插值结果按比例位置映射回三维采样空间，对每条 trace 的已建模范围外做首尾常值延拓，确保全工区无 NaN。

### 第四阶段：导出

1. 将建模结果保存为 `ai_lfm_time.npz`，内含体积、方差体、三个规则轴、几何元数据、建模元数据和覆盖统计。
2. 可选导出 SEG-Y 或 ZGY 格式，方便在地质软件中检查。
3. 输出 QC 图：控制点平面分布图和 LFM 剖面图。

---

## 输出文件

所有文件在 `<output_root>/lfm_precomputed_<timestamp>/` 下：

| 文件 | 内容 |
|------|------|
| `ai_lfm_time.npz` | GINN 训练可直接读取的 AI 低频模型 |
| `ai_lfm_time.segy` | 可选 SEG-Y 导出 |
| `ai_lfm_time.zgy` | 可选 ZGY 导出 |
| `lfm_layer_control_points.csv` | 点级控制样本，每行一个目标层内控制点 |
| `lfm_control_qc.csv` | 逐井筛选结果、控制点数量和无效比例 |
| `target_layer_qc/*` | 目标层 mask、层厚、层位有效性 QC |
| `figures/*.png` | 控制点分布图和 LFM 剖面图 |
| `run_summary.json` | 输入路径、筛选统计、建模参数和输出路径 |

### `ai_lfm_time.npz`

供 `src.ginn.data` 训练端读取，包含以下键：

| 键 | 含义 |
|----|------|
| `volume` | AI LFM 体，shape `(n_inline, n_xline, n_sample)` |
| `variance_volume` | 切片插值方差体，同 shape |
| `ilines` / `xlines` / `samples` | 三个规则轴；samples 为正秒 TWT |
| `geometry_json` | 地震几何，包含 `sample_domain: time` 和 `sample_unit: s` |
| `metadata_json` | 训练端重建 mask 所需的元数据 |
| `coverage_stats_json` | 井和层段覆盖统计 |

`metadata_json` 中必须包含 `horizons`（顶底解释层位的路径和平均 TWT）、`target_layer`（目标层 QC 参数）和 `path_style`。第七步会从中读取层位文件重建训练 mask，因此这些信息不能只写进 `run_summary.json`。

第七步会校验 `samples` 轴与训练地震体的采样轴完全对齐。shape 相同但 TWT 轴不一致时必须失败，不能静默训练。

### `lfm_layer_control_points.csv`

每行一个控制点，以 `inline_float`、`xline_float` 和 `twt_s` 为规范坐标。关键字段：

| 字段 | 含义 |
|------|------|
| `well_name` / `route` | 来源井和第四步标定路径 |
| `source` | `vertical_trace`（直井井口）或 `deviated_trajectory`（斜井沿轨迹） |
| `twt_s` / `md_m` | 控制点的时间域和深度域位置 |
| `x_m` / `y_m` | 控制点平面坐标 |
| `inline_float` / `xline_float` | 投影到工区后的浮点线号 |
| `zone_name` / `u_in_zone` | 所属层段和层内比例位置（0 到 1） |
| `ai` / `weight` | 控制点 AI 值及其插值权重 |
| `flat_idx` / `sample_index` | 派生字段，仅供调试；依赖地震几何 |

---

## 如何阅读结果

### 第一步：看 `lfm_control_qc.csv`

优先确认入选井的 `status` 分布。`rejected` 井的数量和原因（`batch_corr_below_threshold`、`too_few_control_samples`、`missing_trace_sample_plan` 等）直接反映第四步和第五步的衔接质量。

对入选井，看 `control_point_count` 和 `invalid_point_fraction`。无效比例过高通常说明时深表、井轨迹或目标层解释没有充分重叠。

斜井的 `unique_trace_count` 应大于 1。如果斜井所有控制点落在同一 trace 上，要回头检查第四步空间映射。

### 第二步：看 `lfm_layer_control_points.csv`

抽查几口斜井的 `inline_float` / `xline_float` 是否随 `twt_s` 变化。直井的这两个字段应恒定。同时检查 `u_in_zone` 覆盖是否均匀——集中在某个窄范围说明该井只有部分曲线段落落入目标层。

### 第三步：看图

`figures/` 中的控制点分布图用于判断是否存在空间偏差：某个平台并群的密度是否远超其他区域。密井网导致同一 `(inline, xline)` 坐标出现多个控制点时，当前版本在切片内做加权平均后进入插值，并在覆盖统计中记录聚合数量。更复杂的冲突策略留到第二轮。

---

## 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| `No LFM control points selected` | 所有井都被过滤或控制点生成失败 | 检查 `lfm_control_qc.csv` 的拒绝原因；多数情况是第四步成功井太少或第五步门槛过高 |
| `target_layer geometry domain is not time` | 地震数据不在时间域 | 确认地震体路径和类型正确 |
| 某口斜井 `missing_trace_sample_plan` | 第四步未为该斜井写出空间映射 | 回到第四步检查斜井路径是否执行成功 |
| `too_few_control_samples` | 落入目标层的有效样点不足 | 检查时深表范围是否覆盖目标层；LAS 曲线在目标层深度内是否有值 |
| ZGY 导出失败 (zgy export skipped/failed) | 源地震非 ZGY 格式 | ZGY 导出只对 ZGY 源地震开启；对 SEG-Y 源使用 SEG-Y 导出 |
| SEG-Y 导出失败 | 缺少 SEG-Y 头字节配置或 `cigsegy` 不可用 | 检查配置中的 `iline_byte`/`xline_byte`；确认 `cigsegy` 已安装 |

---

## 下游消费

第七步 `ginn_train.py` 读取 `ai_lfm_time.npz` 的 `volume` 作为低频模型输入通道，读取 `metadata_json.horizons` 重建训练 mask，读取 `metadata_json.target_layer` 复用目标层 QC 参数。

因此第六步产出的 NPZ 不只是数据体，还承载了目标层的层位选择和 QC 口径。第七步的 train config 不再单独配置顶底层位——这一事实从第六步继承。

---

## 留到第二轮

- 输出 Vp 和 Rho LFM。
- 控制点冲突的 weighted average / highest confidence / fail-on-conflict 策略，取代当前的简单加权平均。
- 从点级控制直接生成 GINN 井约束锚点文件。
- dynamic gain 或 frequency split 诊断与 LFM 的联动。
- 斜井 AI 低通滤波的采样率自适应（当前使用中位步长，对极不均匀采样的适应性有限）。
