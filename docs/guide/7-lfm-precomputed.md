# 07 构建低频模型

`lfm_precomputed.py` 是工作流的第七步。它读取第六步 `well_constraints.py` 已经统一生成的低频控制点，再通过层位约束插值生成波阻抗低频模型，供第八步 GINN 训练直接读取。

本步的核心不再是"一口井一个控制点"，而是"目标层内的样点控制"——每口井在目的层内贡献数十到上百个带空间坐标的波阻抗样点，斜井样点随轨迹分布在不同的 inline/xline 上。

第二轮重构后，第六步 `well_constraints.py` 已经前置生成共享井空间事实和分频事实；本步消费点级 `lfm_control_points.csv`，并在第七步内部完成顺层切片、必要的重复点聚合和插值建模，不再从 LAS、TDT 或井轨迹临时拼井约束。

---

## 快速开始

```bash
python scripts/lfm_precomputed.py
python scripts/lfm_precomputed.py --config experiments/common.yaml
python scripts/lfm_precomputed.py --output-dir scripts/output/lfm_precomputed_test
```

不带参数时，脚本自动发现最新的第六步 `well_constraints_*` 产物，按时间戳创建输出目录。

---

## 运行前需要什么

| 来源 | 内容 | 用途 |
|------|------|------|
| 第六步 | `lfm_control_points.csv` | 点级低频 AI 控制事实 |
| 第六步 | `well_constraint_qc.csv` / `run_summary.json` | 控制井 QC 和分频配置审计 |
| 地震数据 | 地震体 + 顶底解释层位 | 提供时间轴、工区几何和目标层 mask |

---

## 配置参考

```yaml
seismic:
  file: <seismic-volume-file>
  type: zgy
  zgy_inline_chunk_size: 16

lfm_precomputed:
  target_interval:
    horizons:
      - <top-horizon-file>
      - <middle-horizon-file>
      - <bottom-horizon-file>
    twt_unit: auto

  modeling:
    boundary_extension_samples: 50
    n_slices: 20
    variogram: spherical
    exact: true
    nugget: 0.0
    post_slice_smoothing: false

  export:
    export_volume: true
```

### `source_runs`

默认自动接上最新一次井约束结果。复现实验时可按需加入 `source_runs.well_constraints_dir` 固定输入。

`export_volume: true` 时，输出格式跟随顶层 `seismic.type`：ZGY 输入写 ZGY，SEG-Y 输入写 SEG-Y。ZGY 分块大小来自顶层 `seismic.zgy_inline_chunk_size`。

每个控制点本质上是“一口井在目标层内某个时间样点上的低频波阻抗值”。第六步只负责把这个点的 TWT、浮点 inline/xline、层段、层内比例和低频 AI 写清楚；第七步再按 `modeling.n_slices` 把这些点分配到顺层切片。那些整数道号、数组索引只用于排查，不能当作跨工区稳定坐标。

直井的控制点落在井口对应的同一个 trace 上；斜井控制点来自轨迹样点，因此一口斜井的控制点会分布在多个不同的 `inline_float`、`xline_float` 处。

切片聚合不再由第六步提前完成。第七步在建模时按切片收集控制点，并只对同一切片内完全重复的 inline/xline 坐标做权重聚合，避免密井或重复轨迹点在克里金里被当成多个独立空间位置。

### `target_interval`

`horizons` 是目标层位列表，至少包含两个文件。脚本会读取所有层位并按平均 TWT 从浅到深排序，相邻层位组成建模层段。给出三个层位时，会形成两个相邻 zone；给出更多层位时，依此类推。

`twt_unit: auto` 表示按解释层位值的量级自动判断单位：值量级像毫秒则转为正秒，否则按正秒使用。解析结果必须与地震时间轴同单位。也可显式指定为 `s` 或 `ms`。

### `modeling`

这一块只控制顺层插值和三维 LFM 重建。井曲线低通、cutoff 诊断和高低频拆分都在第六步完成；第七步消费的 `ai` 已经是第六步确定的低频控制值。

#### `n_slices`

控制顺层切片数量。相邻层位组成一个 zone，每个 zone 会按层内比例切成 `n_slices` 张薄片；如果目标窗由多个层位分成多个 zone，每个 zone 都会独立切片。切片数越多，层内垂向变化表达越细，但每张切片能分到的控制点也更少；如果井少或控制点稀疏，过大的 `n_slices` 反而会让很多切片靠补值支撑。

#### `variogram` / `exact` / `nugget`

控制每张切片上的空间插值。多于一个控制点的切片会在 inline/xline 平面上做普通克里金；`variogram` 描述空间相关性形态，默认 `spherical` 是比较稳健的常用选择。`exact: true` 表示插值结果尽量精确穿过控制点；`nugget: 0.0` 表示暂不显式加入测量噪声项。如果后续发现井点噪声或同平台控制点冲突明显，可以再考虑放松 exact 或引入 nugget。

#### `boundary_extension_samples`

控制目标层外的上下缓冲。第八步训练和物理正演会受到子波长度、mask 边界和卷积边界影响；如果 LFM 只在目标层内有值，边界附近容易出现突变。第七步会在目标层上方和下方各扩展一段样点，用相邻层段的边界趋势延拓，让第八步读取到的体在边界附近更平稳。

#### `post_slice_smoothing`

切片之间的额外平滑，默认关闭。启用后对每个层段的比例切片沿 slice 维做轻度加权平均（核 `[0.1, 0.2, 0.4, 0.2, 0.1]`），减弱相邻切片面的突变和尖刺。代价是进一步偏离控制点——平滑核覆盖 5 个切片，会把邻切片的克里金值混入当前切片。如果井上抽出的 LFM 曲线有明显锯齿感，可以尝试开启。

---

## 脚本在做什么

脚本分四个阶段：**前置发现 → 控制点读取 → 层位约束建模 → 导出**。

### 第一阶段：前置发现

1. 从配置或自动发现中定位第六步井约束产出目录。
2. 打开地震体，校验采样域为时间域且单位为秒。
3. 读取顶底解释层位，构建目标层并输出层位 QC（mask 有效性、层厚、交叉、薄层）。

### 第二阶段：控制点读取

第七步读取第六步写出的 `lfm_control_points.csv`，校验字段、正值 AI、有限坐标和 `u_in_zone` 范围，然后把控制点复制到本次第七步输出目录。第六步不再提供 `n_slices`，切片数量只由本步骤的 `modeling.n_slices` 决定。

### 第三阶段：层位约束建模

建模核心可以理解为“顺层切片插值”：先把目标层按层内比例切成多张薄片，每张薄片独立做平面插值，最后再把这些薄片拼回三维体。这样比直接按固定 TWT 切片更尊重层位形态。

1. **比例切片离散。** 相邻层位组成一个层段，每个层段再按层内比例切成若干薄片。`n_slices` 越大，层内变化表达越细，但插值也更依赖控制点覆盖。

2. **控制点分配。** 每个控制点按其 `u_in_zone` 值分配到最近的切片。

3. **切片插值。** 每张切片收集落入其中的控制点。有多个控制点时做平面插值；只有一个控制点时，该切片只能退化为常值片。

4. **缺失切片填充。** 某些切片可能没有任何控制点（特别是目标层顶部或底部），此时沿比例方向向上、向下搜索最近的有效切片，用其值填充。

5. **边界扩展。** 按 `boundary_extension_samples` 在目标层最顶层之上和最底层之下各扩展一段。扩展区若没有直接控制值，则从相邻原始层段的边界切片做常值延拓。

6. **体重建。** 将所有切片按层位几何映射回三维采样空间，并对建模范围外做温和延拓，确保第八步读取时不会遇到空洞。

### 第四阶段：导出

1. 将建模结果保存为 `ai_lfm_time.npz`，内含体积、方差体、三个规则轴、几何元数据、建模元数据和覆盖统计。
2. 可选导出地震体格式（SEG-Y 进则 SEG-Y 出，ZGY 进则 ZGY 出），方便在地质软件中检查。
3. 输出 QC 图：控制点平面分布图、LFM 剖面图和井上波阻抗 QC 图。

---

## 核心输出文件

所有文件在 `<output_root>/lfm_precomputed_<timestamp>/` 下：

| 文件 | 内容 |
|------|------|
| `ai_lfm_time.npz` | GINN 训练可直接读取的波阻抗低频模型 |
| `ai_lfm_time.segy` 或 `.zgy` | 可选地震体导出，格式跟随源地震 |
| `lfm_control_points.csv` | 点级控制样本，每行一个目标层内低频 AI 控制点 |
| `target_layer_qc/*` | 目标层 mask、层厚、层位有效性 QC |
| `figures/*.png` | 控制点分布图和 LFM 剖面图 |
| `well_qc/figures/well_qc_<well>.png` | 每口控制点井的低频 AI vs LFM 采样 AI 对比 |
| `well_qc/traces/well_qc_<well>.csv` | 控制点 AI、LFM 采样 AI 和残差的逐样点明细 |
| `well_qc/well_qc_metrics.csv` | 逐井 LFM QC 指标（相关系数、MAE、RMSE、bias、NMAE） |
| `run_summary.json` | 输入路径、筛选统计、建模参数和输出路径 |

### `ai_lfm_time.npz`

这是第八步真正读取的低频模型包，包含：

| 键 | 含义 |
|----|------|
| `volume` | 波阻抗 LFM 体，shape `(n_inline, n_xline, n_sample)` |
| `variance_volume` | 切片插值方差体，同 shape |
| `ilines` / `xlines` / `samples` | 三个规则轴；samples 为正秒 TWT |
| `geometry_json` | 地震几何，包含 `sample_domain: time` 和 `sample_unit: s` |
| `metadata_json` | 训练端重建 mask 所需的元数据 |
| `coverage_stats_json` | 井和层段覆盖统计 |

`metadata_json` 中必须包含 `horizons`（顶底解释层位的路径和平均 TWT）、`target_layer`（目标层 QC 参数）和 `path_style`。第八步会从中读取层位文件重建训练 mask，因此这些信息不能只写进 `run_summary.json`。

第八步会严格校验时间轴。即使体大小相同，只要 TWT 采样轴不一致，也应该回到第七步或地震配置排查；这种情况不能被当成正常训练继续下去。

### `lfm_control_points.csv`

每行一个控制点，以 `inline_float`、`xline_float` 和 `twt_s` 为规范坐标。关键字段：

| 字段 | 含义 |
|------|------|
| `well_name` / `route` | 来源井和第四步标定路径 |
| `source` | `vertical_trace`（直井井口）或 `deviated_trajectory`（斜井沿轨迹） |
| `twt_s` / `md_m` | 控制点的时间域和深度域位置 |
| `x_m` / `y_m` | 控制点平面坐标 |
| `inline_float` / `xline_float` | 投影到工区后的浮点线号 |
| `zone_name` / `u_in_zone` | 所属层段和层内比例位置（0 到 1） |
| `ai` / `weight` | 第六步分频后的低频 AI 控制值及权重 |
| `flat_idx` / `seismic_sample_index` | 派生字段，仅供调试和交叉校验；依赖地震几何与全局采样轴 |

---

## 如何阅读结果

### 第一步：看 `lfm_control_points.csv`

抽查几口斜井的 `inline_float` / `xline_float` 是否随 `twt_s` 变化。直井的这两个字段应恒定。同时检查 `u_in_zone` 覆盖是否均匀——集中在某个窄范围说明该井只有部分曲线段落落入目标层。

若入选井或点数不符合预期，回到第六步的 `well_constraint_qc.csv` 看拒绝原因和无效样点比例。

### 第二步：看图

`figures/` 中的控制点分布图用于判断是否存在空间偏差：某个平台井群的密度是否远超其他区域。同一张切片里如果多口井落到重复 `(inline, xline)` 坐标，第七步会在切片建模时按权重聚合这些重复坐标，再交给克里金插值。

---

### 常见失败原因

| 原因 | 含义 | 怎么处理 |
|------|------|---------|
| `No LFM control points selected` | 第六步没有生成可用低频控制点 | 检查第六步 `well_constraint_qc.csv` 和 `run_summary.json`；上游第四步标定、第五步批量合成门槛或第六步目标层覆盖都可能是原因 |
| `target_layer geometry domain is not time` | 地震数据不在时间域 | 确认地震体路径和类型正确 |
| 某口斜井 `missing_optimized_trace_sample_plan` | 第四步未为该斜井写出细标定后的空间映射 | 回到第四步检查斜井路径是否执行成功，以及 `well_tie_metrics.csv` 是否有 `optimized_trace_sample_plan_file` |
| `too_few_control_samples` | 落入目标层的有效样点不足 | 检查时深表范围是否覆盖目标层；LAS 曲线在目标层深度内是否有值 |
| 地震体导出失败 | 缺少 SEG-Y 头字节配置或 ZGY 写入库不可用 | 检查顶层 `seismic` 的 SEG-Y 几何配置，或确认 `pyzgy` 已安装；也可设置 `export.export_volume: false` 跳过导出 |

---

## 留到第二轮

- highest confidence / fail-on-conflict 等更激进的控制点冲突策略。
- 按平台或空间簇做 LFM 控制点去偏，避免密井平台在克里金里权重过高。
- 斜井波阻抗低通滤波的采样率自适应（当前使用中位步长，对极不均匀采样的适应性有限）。
