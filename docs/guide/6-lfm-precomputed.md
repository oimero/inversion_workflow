# 06 时间域点云低频模型

`lfm_precomputed.py` 是时间域工作流的第六步。它消费第四步的井震标定事实和第五步的全局子波批量合成 QC，把每口井在目标层内的曲线样点转成空间控制点，再插值生成 GINN 训练可直接读取的 AI 低频模型。

第一版只输出 **AI LFM**。Vp、Rho、well constraints、dynamic gain 和 enhance 都不属于本步主线。

---

## 快速开始

```bash
python scripts/lfm_precomputed.py
python scripts/lfm_precomputed.py --config experiments/common.yaml
python scripts/lfm_precomputed.py --output-dir scripts/output/lfm_precomputed_test
```

不带参数时，脚本应自动发现最新的第四步和第五步产物，在 `scripts/output/lfm_precomputed_<timestamp>/` 下写出结果。

---

## 运行前需要什么

| 来源 | 文件 | 用途 |
|------|------|------|
| 第四步 | `well_tie_metrics.csv` | 标定成功/失败、路由、优化后 TDT、滤波 LAS、轨迹采样文件 |
| 第四步 | `filtered_las/filtered_logs_<well>.las` | 读取 `DT_USM` 和 `RHO_GCC`，构造 AI 控制值 |
| 第四步 | `time_depth/optimized_tdt_<well>.csv` | 将 MD 域曲线样点映射到 TWT |
| 第四步 | `trace_sample_plan/trace_sample_plan_<well>.csv` | 斜井样点的 MD/TWT/XY/inline/xline/trace 映射 |
| 第五步 | `batch_synthetic_metrics.csv` | 用全局子波后的逐井合成质量，用于筛选控制井 |
| 地震数据 | ZGY 或 SEG-Y 体、解释层位 | 提供时间轴、inline/xline 几何和目标层 mask |

直井没有 `trace_sample_plan` 时，脚本用井口 XY 作为该井所有控制点的空间位置；斜井优先复用第四步写出的样点级空间事实，不重新写一套轨迹映射逻辑。

---

## 配置参考

```yaml
lfm_precomputed:
  source_runs:
    mode: latest
    well_auto_tie_dir: null
    global_wavelet_generation_dir: null

  seismic:
    file: raw/obn-clipped-240-912-872-1544.zgy
    type: zgy

  target_interval:
    top_horizon: interpre/H3-1
    bottom_horizon: interpre/H7-1
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
```

### `control_wells`

第六步默认不把所有第四步成功井都放进 LFM。控制井必须先通过第五步全局子波批量合成 QC：

- `min_batch_corr`：低于该相关系数的井不参与 LFM。
- `max_batch_nmae`：可选 NMAE 上限；为空时不按 NMAE 过滤。
- `include_wells`：白名单模式，只使用指定井。
- `exclude_wells`：人工排除可疑井。

筛选后的井必须同时存在 `filtered_las_file`、`optimized_tdt_file`。斜井还应有对应的 `trace_sample_plan`，否则不能作为斜井点级控制参与。

斜井缺少 `trace_sample_plan` 时，脚本应跳过该井，并在 `lfm_control_qc.csv` 中记录 `status = rejected` 和原因 `missing_trace_sample_plan`。不要把斜井降级成井口直井控制点；那会把空间事实写错。

### `controls`

本步的核心不是“一口井一个控制点”，而是“目标层内的样点控制”：

```text
filtered LAS + optimized TDT + WellSpatialSampleSet
  -> AI(TWT)
  -> layer control points
  -> AI LFM
```

控制点的规范坐标是 `inline_float`、`xline_float` 和 `twt_s`。CSV 至少包含 `well_name`、`route`、`twt_s`、`md_m`、`x_m`、`y_m`、`inline_float`、`xline_float`、`zone_name`、`u_in_zone`、`ai`、`weight` 和 `source`。

`flat_idx` 和 `sample_index` 可以作为方便排查的派生字段写出，但不能替代规范坐标；它们只在当前地震几何和采样轴下有效。对直井，`source = vertical_trace`，XY 固定为井口坐标。对斜井，`source = deviated_trajectory`，XY 来自第四步 `trace_sample_plan_<well>.csv`。

`sample_step_s: null` 表示使用地震体时间采样间隔，不另行重采样控制点。`target_interval.twt_unit: auto` 表示按项目现有解释层位读取口径解析 TWT：若解释层位值量级像毫秒则转为正秒，否则按正秒使用；解析结果必须和地震时间轴同单位。

### `modeling`

建模逻辑对齐深度域 `lfm_precomputed_depth.py`，但采样轴是 TWT 秒：

1. 用解释层位和地震几何构建 `TargetZone`。
2. 按层段比例 `u_in_zone` 把控制点分配到各个 slice。
3. 在 inline/xline 平面上对每个 slice 做插值。
4. 沿时间轴重建 AI LFM，并对 TWT 轴做低通滤波。
5. 将结果保存为 `ai_lfm_time.npz`，供 `src.ginn.data` 读取。

`filter_cutoff_hz` 是时间域低通截止频率，不能写成深度域的 wavelength 参数。

---

## 输出文件

所有文件在 `<output_root>/lfm_precomputed_<timestamp>/` 下：

| 文件 | 内容 |
|------|------|
| `ai_lfm_time.npz` | GINN 时间域训练读取的 AI 低频模型 |
| `ai_lfm_time.segy` | 可选 SEG-Y 导出，便于地质软件检查 |
| `lfm_layer_control_points.csv` | 点级控制样本，每行一个目标层内控制点 |
| `lfm_control_qc.csv` | 每口井的控制点数量、无效点数量、跨 trace 情况和入选状态 |
| `target_layer_qc/*` | 目标层 mask、层厚、层位有效性 QC |
| `figures/*.png` | LFM 剖面、目标层 mask 和控制点分布图 |
| `run_summary.json` | 输入路径、控制井过滤统计、建模参数和输出路径 |

### `ai_lfm_time.npz`

为兼容 `src.ginn.data.load_lowfreq_model()`，NPZ 至少包含：

| 键 | 含义 |
|----|------|
| `volume` | AI LFM，shape 为 `(n_inline, n_xline, n_sample)` |
| `variance_volume` | 插值方差体，shape 与 `volume` 相同 |
| `ilines` / `xlines` / `samples` | 三个规则轴；`samples` 为正秒 TWT |
| `geometry_json` | 地震几何，必须包含 `sample_domain: time` 和 `sample_unit: s` |
| `metadata_json` | 训练端重建 mask 所需的元数据 |
| `coverage_stats_json` | 井和层段覆盖统计 |

`metadata_json` 必须包含：

```json
{
  "property_name": "AI",
  "target_layer": {
    "min_thickness": null,
    "nearest_distance_limit": null,
    "outlier_threshold": null,
    "outlier_min_neighbor_count": 2
  },
  "horizons": [
    {"file": "data/interpre/H3-1", "mean_twt_s": 1.23},
    {"file": "data/interpre/H7-1", "mean_twt_s": 1.56}
  ],
  "path_style": "repo_relative"
}
```

`src.ginn.data` 会从 `metadata_json.horizons` 重新读取顶底解释层位，并用 `metadata_json.target_layer` 里的 QC 参数重建训练 mask；这些信息不能只写进 `run_summary.json`。

第七步会校验 `samples` 与训练地震体的 `sample_min/sample_step/n_sample` 完全对齐。即使 `volume.shape` 相同，只要 TWT 轴不一致，也必须失败，不能静默训练。

---

## 如何阅读结果

### 第一步：看 `lfm_control_qc.csv`

优先确认每口井的 `status`、`control_point_count` 和 `invalid_point_fraction`。如果某口斜井跨了很多 trace，这是预期现象；如果它只有极少数有效控制点，通常说明 TDT、轨迹或目标层解释没有重叠好。

### 第二步：看 `lfm_layer_control_points.csv`

抽查斜井的 `inline_float` / `xline_float` 是否随 TWT 变化。斜井如果整口井都落在同一个 trace 上，要回头检查第四步的 `trace_sample_plan`。

### 第三步：看图

`figures/` 里的控制点分布和 LFM 剖面用于判断插值是否被单个平台井群主导。密井网导致的控制点冲突第一版只做 QC 报告，不进入 well constraints 聚合。

---

## 留到第二轮

- 输出 Vp/Rho LFM。
- 从点级控制生成 GINN `log_ai_anchor_file`。
- 控制点冲突的 weighted average / highest confidence / fail-on-conflict 策略。
- dynamic gain 或 frequency split 诊断与 LFM 的联动。
