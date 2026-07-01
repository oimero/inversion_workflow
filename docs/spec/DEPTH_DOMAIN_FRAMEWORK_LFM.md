# 深度域框架约束 LFM 实施规范

> 状态：待实施
> 范围：深度域实际工区 Step 7；M0 层状插值 LFM 与 M1 平面框架约束 LFM
> 当前工区：TVDSS，向下为正；inline 步长 1，xline 步长 4，深度采样 5 m；当前控制井均为直井
> 本文是实施规范。若实现与本文冲突，应先修改本文并记录原因，不得通过静默兼容、字段猜测或数值兜底绕过契约。

## 1. 目标

导师要求把“框架约束反演”首先体现在初始波阻抗模型中，并与原来的层状插值模型对比。v1 不修改 GINN-v2 的输入通道或输出语义，而是构建两套同网格、可直接比较的深度域 LFM：

- **M0**：由深度校正后的井 AI 曲线、解释层位和比例切片克里金生成的层状插值 LFM；
- **M1**：在 M0 上叠加平面礁/滩展布框架的场景 LFM。

平面框架只回答目标体“横向在哪里”。其纵向范围由解释层位和相对地层坐标定义：

```text
reef  = reef_polygon(inline, xline) × interval(base_of_salt, base_of_bve)
shoal = shoal_polygon(inline, xline) × interval(base_of_bve, base_of_itp)
```

因此，M1 的准确表述是“平面展布框架 + 层位控制的层间初始模型”，不是由剖面精细勾勒并插值得到的三维礁丘几何体。

v1 必须具备以下性质：

- M0、M1 使用相同的 inline、xline、TVDSS 轴和有效区掩码；
- 规范值域为 `log(AI)`，AI 单位固定为 `m/s*g/cm3`；
- M0 的 400 m 截止波长只约束井曲线背景；
- M1 的线性 AI 倍率和过渡尺度全部由实验配置显式给出；
- 窗口模式用于快速迭代和多剖面 QC，全体积模式用于解释软件导出；
- 所有来源、参数、轴和文件哈希可追溯；
- 控制不足或语义不明时立即失败，不自动降低标准。

命名约定：

- **M0** 表示 baseline model，即常规层状插值低频初始模型；
- **M1** 表示 framework scenario model，即在 M0 上叠加一个显式框架场景后的初始模型；
- M0/M1 不是 schema 或神经网络版本号，二者均为真实工区 LFM 体。

## 2. 非目标

v1 不处理：

- 从地震属性或神经网络自动识别礁滩；
- 在垂直剖面逐条勾勒并重建精细三维礁丘形态；
- 多边形编辑 UI、Petrel 插件或解释文件交互；
- 斜井的 MD—TVDSS 轨迹变换；
- 根据井自动判定礁、滩或背景相；
- 自动估计 reef/shoal `linear_ai_multiplier`；
- 构建多套弱/中/强框架模型；v1 每次运行只有一个显式命名的 M1 场景；
- 修改 Synthoseis-lite 的 LFM 生成、GINN-v2 训练、R0 或 R1；
- 恢复或依赖 `src/ginn/`、`src/ginn_depth/` 以及 `.ref/` 中的旧生产接口。

合成训练将来需要见到与真实工区一致的 LFM 变体，但不属于本规范的实施和验收范围。

## 3. 现状与设计依据

### 3.1 当前工区事实

`experiments/common/common.yaml` 已冻结：

```text
sample_domain = depth
depth_basis   = tvdss
inline        = 1501 .. 2101, step 1
xline         = 3799 .. 6999, step 4
TVDSS         = 4750 .. 7500 m, step 5 m
```

相邻 inline 和相邻 xline 道的实际平面间距均约为 25 m。线号步长不是物理距离；所有米制距离必须通过 `SurveyLineGeometry` 计算。

目标层位从浅到深为：

1. `base_of_salt`；
2. `base_of_bve`；
3. `base_of_itp`。

地质语义固定为：

- `base_of_salt -> base_of_bve`：礁，class id 为 `reef`；
- `base_of_bve -> base_of_itp`：滩，class id 为 `shoal`。

### 3.2 井曲线来源

M0 只消费 `wavelet_batch_synthetic_depth` source run 生成的：

- `wavelet_batch_metrics.csv`；
- `shifted_filtered_las/*.las`；
- `run_summary.json`。

`shifted_filtered_las/AI` 是该 source run 生成的深度平移后 filtered AI，LAS 深度轴仍是 MD。`wavelet_batch_synthetic_depth_dir` 只用于定位和校验 `wavelet_batch_metrics.csv`、`shifted_filtered_las` 与 `run_summary.json`。本规范不再使用 Step 5/Step 6 编号描述这组来源，避免与时间域主流程编号混淆。

当前七口控制井均在 inventory 中判定为 `vertical`，因此 v1 显式使用：

```text
TVDSS = shifted_MD - KB
```

该等式不是通用斜井算法。任何进入候选集合的非直井必须使运行失败。

### 3.3 可复用与不可复用能力

可复用当前代码：

- `cup.config.workflow.WorkflowConfig`：工区事实；
- `cup.config.sources`：来源运行发现与校验；
- `cup.seismic.survey`：深度域地震几何和线号坐标；
- `cup.seismic.target_zone.TargetZone`：层位建面、厚度 QC 和目标窗掩码；
- `cup.seismic.modeling`：比例切片和 ordinary kriging 数学；
- `cup.seismic.volume_export`：全体积 SEG-Y 导出；
- `wtie.processing.grid.Log`：携带 TVDSS 轴与单位的井曲线。

`.ref/scripts_depth/lfm_precomputed_depth.py` 和旧 `lfm_depth.py` 只用于审计数学顺序：先转 TVDSS、再低通井曲线、最后比例切片建模。旧相控代码采用局部控制点和半径，不等于本规范的平面框架，不能成为新接口或数据契约。

时间域 `scripts/real_field_lfm.py` 固定使用 TWT 语义，不得通过替换字段名改造成深度域入口。

## 4. 代码边界

规划新增：

```text
scripts/real_field_lfm_depth.py
src/cup/seismic/lfm_depth.py
experiments/real_field_lfm_depth/lfm.yaml
experiments/real_field_lfm_depth/framework_polygons.csv
```

其中：

- `scripts/real_field_lfm_depth.py` 只负责 CLI、配置合并、来源解析、产物落盘和终端摘要；
- `cup.seismic.lfm_depth` 负责配置值对象、井控制准备、米制低通、严格比例切片、平面框架栅格化、M1 合成、QC 统计和绘图数据准备；
- 实验 YAML 存放可变场景假设；
- `common.yaml` 继续只存工区地震、层位和资产事实，不写 `linear_ai_multiplier` 或 polygon 路径。

CLI 固定为：

```powershell
python scripts/real_field_lfm_depth.py `
  --config experiments/real_field_lfm_depth/lfm.yaml `
  --output-dir scripts/output/real_field_lfm_depth_test
```

`--output-dir` 可省略；省略时写入 `scripts/output/real_field_lfm_depth_<timestamp>/`。CLI 不暴露 `linear_ai_multiplier`、taper、窗口或建模参数，这些必须进入可追溯的实验 YAML。

## 5. 配置契约

实验文件最小结构如下。尖括号表示必须由使用者填写，不是实现默认值。

```yaml
workflow_config: experiments/common/common.yaml

real_field_lfm_depth:
  source_runs:
    wavelet_batch_synthetic_depth_dir:
    well_inventory_dir:

  output_geometry:
    mode: window
    window:
      inline_min: <actual-line-number>
      inline_max: <actual-line-number>
      xline_min: <actual-line-number>
      xline_max: <actual-line-number>

  baseline:
    cutoff_wavelength_m: 400.0
    filter_order: 6
    filter_buffer_mode: reflect
    filter_buffer_m: 800.0
    n_slices: 32
    min_controls_per_slice: 3
    variogram: spherical
    exact: true
    nugget: 0.0

  framework:
    scenario_id: M1_framework
    polygons_file: experiments/real_field_lfm_depth/framework_polygons.csv
    classes:
      reef:
        top_horizon: base_of_salt
        bottom_horizon: base_of_bve
        linear_ai_multiplier: <positive-number>
        edge_taper_m: <positive-number>
        top_taper_fraction: <number-in-(0,0.5)>
        bottom_taper_fraction: <number-in-(0,0.5)>
      # shoal 可替代 reef，或与 reef 同时出现。
      # shoal:
      #   top_horizon: base_of_bve
      #   bottom_horizon: base_of_itp
      #   linear_ai_multiplier: <positive-number>
      #   edge_taper_m: <positive-number>
      #   top_taper_fraction: <number-in-(0,0.5)>
      #   bottom_taper_fraction: <number-in-(0,0.5)>

  qc:
    sections:
      - {axis: inline, line: <actual-inline>}
      - {axis: xline, line: <actual-xline>}
```

### 5.1 配置校验

- `workflow_config` 必须解析为当前深度域 TVDSS 工区；
- `output_geometry.mode` 只允许 `window` 或 `volume`；
- `window` 模式必须给出四个边界，且边界严格落在显式线号轴上；
- `volume` 模式禁止出现 `window`；
- baseline 参数必须全部显式存在；
- v1 baseline 固定要求 400 m、六阶、`reflect`、32 slices、spherical ordinary kriging、`exact=true`、`nugget=0`；配置出现其他值时明确失败，而不是暗中覆盖；
- `framework.classes` 必须为非空映射，只允许 `reef`、`shoal`；允许仅启用一类；
- `reef` 和 `shoal` 的层位对必须分别精确匹配 §3.1；
- `linear_ai_multiplier` 表示线性 AI 域倍率，必须有限且大于 0；所有活动类别倍率均为 1 时失败，因为 M1 将与 M0 相同；
- `edge_taper_m` 必须有限且大于 0；
- top/bottom taper fraction 必须分别位于 `(0, 0.5)`，且二者之和小于 1；
- `qc.sections` 必须非空，每条线必须落在当前输出轴上；xline 必须遵守步长 4，不能按数组下标解释。

来源目录可以显式给出，也可按 `cup.config.sources` 发现最新满足契约的运行；发现是路径解析策略，不是数值或语义兜底。最终解析路径和文件哈希必须写入 summary。

## 6. 平面框架 CSV 契约

文件固定包含以下五列，不接受别名：

| 字段 | 类型 | 含义 |
|---|---|---|
| `polygon_id` | 非空字符串 | 多边形唯一标识；全文件唯一 |
| `framework_class` | 枚举 | `reef` 或 `shoal` |
| `vertex_order` | 整数 | 同一多边形内从 0 开始连续编号 |
| `inline` | 浮点线号 | 多边形顶点 inline；不是数组下标 |
| `xline` | 浮点线号 | 多边形顶点 xline；不是数组下标 |

示例：

```csv
polygon_id,framework_class,vertex_order,inline,xline
reef_001,reef,0,1630.0,5903.0
reef_001,reef,1,1660.0,5903.0
reef_001,reef,2,1660.0,6103.0
reef_001,reef,3,1630.0,6103.0
```

规则：

- 每个多边形至少三个互异顶点；
- 最后一个顶点不得重复第一个顶点，闭合由程序完成；
- `vertex_order` 必须唯一、连续且从 0 开始；
- 顶点必须有限并位于完整地震工区线号范围内；
- 顶点可以是浮点线号，不要求落在道中心；
- 多边形不得自交、退化或面积为 0；
- CSV 中出现的 class 必须在当前 M1 配置中活动，活动 class 也必须至少有一个多边形；
- 同一 class 允许多个多边形；不同多边形重叠时取概率最大值，不做概率相加；
- window 模式允许多边形仅部分落入窗口，但至少一个活动多边形必须与窗口相交；
- 不允许把越出完整工区的多边形静默裁剪回工区。

## 7. M0 层状插值模型

### 7.1 控制井准备

1. 读取 `wavelet_batch_synthetic_depth` source run 的 `wavelet_batch_metrics.csv`，只接受 `status=ok` 的记录；
2. 按规范化井名与 `well_inventory.csv` 做 1:1 联接；
3. 要求 `wellbore_class=vertical`、KB、inline、xline 均有限；
4. `shifted_filtered_las_path` 必须存在并与同一 source run summary 记录的输出目录一致；
5. 以 exact mnemonic 读取 `AI`，单位必须精确规范化为 `m/s*g/cm3`；
6. AI 有效值必须有限且大于 0，之后显式取自然对数；
7. 用 `TVDSS = MD - KB` 构造 `grid.Log(..., basis_type="tvdss")`。

井曲线缺口不得跨越插值。低通按连续有限 run 分别执行；某比例切片若不落在具有有效滤波支撑的 run 内，该井不贡献该切片。不得用前后填充、线性跨缺口或常值延伸制造控制值。

### 7.2 米制低通

对每口 `log(AI)` TVDSS 曲线，在进入比例切片之前执行：

```text
cutoff_cycles_per_m = 1 / 400 m
filter_order        = 6
buffer_mode         = reflect
buffer_m            = 800 m
zero_phase          = true
```

滤波结果仍是携带 TVDSS 轴和 `ln(m/s*g/cm3)` 值域语义的 `grid.Log`。400 m 是 M0 井曲线背景的截止波长，不是对 M1 最终体的频谱保证。

### 7.3 相对地层比例切片

对每个层段分别定义：

```text
u_k = k / 31,  k = 0 .. 31
z_well(k) = (1-u_k) * z_top_well + u_k * z_bottom_well
```

在每个 `u_k`：

1. 从每口低通 TVDSS log 采样 `log(AI)`；
2. 丢弃没有有限、连续滤波支撑的井控制；
3. 要求不同控制井数不少于 3；
4. 以 explicit inline/xline 轴归一化为网格 index 坐标；该转换必须除以各自真实线号步长；
5. 使用 spherical ordinary kriging，`exact=true`、`nugget=0`；
6. 按 §7.4 的固定规则逐切片派生 sill/range，并写出切片值、kriging variance 与实际参数；
7. 再按每一道自身的 top/bottom 层位，把 32 个切片线性映射回 TVDSS 样点。

任何层段的任何切片少于 3 口控制井时，整次运行失败。禁止：

- 用邻近切片填充空切片；
- 单井常值场；
- 自动降低 `n_slices`；
- 自动切换变差函数；
- 在控制不足时使用全区均值。

调用 `cup.seismic.modeling` 前必须完成控制数预检；调用后再次检查 `slice_control_counts` 和 `slice_modes`，确认没有进入历史 fallback 分支。

### 7.4 Kriging 参数派生

v1 不把 range、sill、anisotropy 或搜索邻域开放为场景调参项，而是冻结当前 `cup.seismic.modeling` 的确定性规则：

```text
normalized_inline = (inline - inline_min) / inline_step
normalized_xline  = (xline - xline_min) / xline_step

sill  = variance(control_log_ai_values)
range = max(median(nearest_neighbor_distance_in_normalized_grid), 1.0)
```

其他约定：

- 二维 isotropic spherical covariance；
- `nugget=0`、`exact=true`；
- 不设置 anisotropy；
- 不设置搜索半径、邻域象限或最大控制点数；每个切片使用全部有效控制井；
- 控制值方差为 0 会进入 constant-field 分支，按本规范必须失败，不能把 sill 改成任意常数继续运行。

每个切片必须在 `slice_control_qc.csv` 中记录实际 `control_count`、`control_log_ai_variance`、`sill`、`range_normalized_grid`、控制井名和 kriging mode。summary 必须写明上述派生公式、isotropy 和全控制点策略，保证 M0 可复现。

### 7.5 M0 掩码和值域

`valid_mask_model` 使用 `TargetZone` 的 filled target zone：首层位到末层位之间为有效，窗外无效；层位交叉、厚度不足或无支持位置遵循 `TargetZone` 的显式 QC。

M0 在无效样点必须为 NaN，不得以 0、均值或边界值填充。规范数组为：

```text
log_ai            float32 [n_inline, n_xline, n_tvdss]
valid_mask_model  bool    [n_inline, n_xline, n_tvdss]
kriging_variance  float32 [n_inline, n_xline, n_tvdss]
```

## 8. M1 平面框架模型

### 8.1 横向概率

多边形顶点和输出道中心均通过 `SurveyLineGeometry.line_to_coord()` 转为真实 XY 米制坐标。点是否在多边形内由完整几何判断；边界距离使用点到所有多边形线段的最短 XY 距离。

对一个多边形，令内部点到边界的距离为 `d_xy`，配置宽度为 `t=edge_taper_m`：

```text
P_polygon = 0                                      outside or on boundary
P_polygon = 0.5 * (1 - cos(pi * d_xy / t))        0 < d_xy < t
P_polygon = 1                                      d_xy >= t
```

这是仅向内渐变：解释圈外严格为 0，不因软边界修改圈外 LFM。同一 class 多个多边形的平面概率为逐道最大值：

```text
P_map_class = max(P_polygon_1, P_polygon_2, ...)
```

### 8.2 垂向概率

对每个输出道和 class 对应层段：

```text
u = (z - z_top) / (z_bottom - z_top)
```

仅 `0 <= u <= 1` 有效。设 `f_top`、`f_bottom` 为配置比例：

```text
P_vertical = 0.5 * (1 - cos(pi * u / f_top))
             when 0 <= u < f_top

P_vertical = 1
             when f_top <= u <= 1-f_bottom

P_vertical = 0.5 * (1 - cos(pi * (1-u) / f_bottom))
             when 1-f_bottom < u <= 1

P_vertical = 0
             outside the interval
```

最终 class 概率为：

```text
P_class(inline, xline, z) = P_map_class(inline, xline) * P_vertical(u)
```

层位起伏时 `u` 随道变化，因此框架跟随地层，不形成绝对深度水平盒子。若 reef 和 shoal 同时活动，两者分别在自己的层段计算；共享 BVE 层位处因上下 taper 均回到 0，M1 在该界面精确回到 M0。

### 8.3 M1 合成

M1 只按以下公式生成：

```text
logAI_M1 = logAI_M0
         + P_reef  * log(linear_ai_multiplier_reef)
         + P_shoal * log(linear_ai_multiplier_shoal)
```

未活动 class 的项不存在，不创建全零语义占位。线性域等价关系为：

```text
AI_M1 = AI_M0
      * linear_ai_multiplier_reef  ** P_reef
      * linear_ai_multiplier_shoal ** P_shoal
```

M1 不做第二次 400 m 低通。框架的横向/垂向 taper 是独立、显式的场景尺度，必须分别进入 metadata 和 QC。M1 是 `framework-scenario initial model`，不保证满足与 M0 相同的整体频谱低通定义；不能因为文件名含 LFM 就宣称其全部空间变化均满足 400 m 截止波长。

框架允许直接修改控制井位置，程序不得构造井保护半径或强制 M1 回到 M0。summary 和井旁 QC 必须明确记录每口井在各 class 下的最大概率、M0/M1 差值和倍率影响。因此 M1 必须标注为：

```text
framework_scenario_model; not strictly well-honoring after framework modification
```

## 9. 窗口、全体积和内存策略

### 9.1 Window 模式

window 的 inline/xline 边界必须落在地震显式轴上。输出仅包含该矩形子网格，但 M0 kriging 仍使用完整候选井集合，包括窗口外控制井。

层位应先在完整工区建立统一表面和井位采样，再裁取窗口输出网格，避免因窗口边界改变井位层位值或插值结果。window 与 full-volume 在相同子网格上的 M0、M1、概率和掩码必须逐点一致。

window 模式不导出 SEG-Y，只写 NPZ、CSV、JSON 和 PNG。

### 9.2 Volume 模式

当前全体积为 `601 × 801 × 551`，单个 float32 体约 1 GiB。实现不得同时保留不必要的 M0/M1/AI 临时副本：

- 以 float32 保存主数组；
- 框架概率按活动 class 分配；
- 线性 AI 仅在导出时分块或短生命周期转换；
- M0 kriging variance 独立保存，不复制为虚假的 M1 uncertainty；
- 不把完整 XY 点云为每个 polygon 重复实体化。

volume 模式同时输出规范 NPZ 和规范线性 AI SEG-Y。v1 固定要求源 SEG-Y 为格式码 5（4-byte IEEE float）；当前工区已满足该条件。M0/M1 SEG-Y 通过现有 `cup.seismic.volume_export.export_volume_like_source()` 共享源体头信息，并显式传入 `nan_fill=None`，目标窗外 NaN 不得填成 0、端点值或其他显示值。

SEG-Y 分支共享源体头信息，现有公共方法不会使用调用方传入的 `ilines/xlines/samples` 重建头。因此调用前必须严格校验 M0/M1 的 shape、inline、xline、TVDSS 轴与源体全体积逐项一致；不得用该路径导出任意窗口或错位子体积。写出后必须用 cigsegy 回读并校验 sample format、shape、显式轴和目标窗外 NaN。

v1 不输出 display-only filled SEG-Y，也不为 IBM float 预设兼容分支。若实际解释软件验证不能读取或正确显示规范 NaN SEG-Y，应另立显示产物规范，明确填充算法、文件角色和 mask；不得在本实现中静默改变规范体。

固定文件命名：

```text
m0_layered_lfm.segy
m1_framework_lfm.segy
```

textual header 必须写明 schema、M0/M1 角色、`domain=AI`、`unit=m/s*g/cm3`、`source_field=log_ai`、`transform=exp(log_ai)`、`outside_target=NaN` 及对应规范 NPZ 的路径与哈希。

## 10. 产物契约

### 10.1 `m0_layered_lfm.npz`

必需键：

```text
log_ai
valid_mask_model
kriging_variance
ilines
xlines
tvdss_m
metadata_json
```

### 10.2 `m1_framework_lfm.npz`

必需键：

```text
log_ai
valid_mask_model
ilines
xlines
tvdss_m
metadata_json
```

M1 文件不把 M0 variance 冒充成框架模型总不确定性。

### 10.3 `framework_fields.npz`

必需键：

```text
delta_log_ai
combined_framework_probability
ilines
xlines
tvdss_m
metadata_json
```

并为每个活动 class 写：

```text
<class>_map_probability      [n_inline, n_xline]
<class>_probability          [n_inline, n_xline, n_tvdss]
```

`combined_framework_probability` 仅用于 QC，定义为活动 class 三维概率的逐点最大值，不能用于重建 M1；M1 必须按各 class 自己的 `linear_ai_multiplier` 分项计算。

### 10.4 QC 表

至少输出：

| 文件 | 内容 |
|---|---|
| `well_control_qc.csv` | 井来源、KB、线号、TVDSS 支撑、单位、低通状态 |
| `slice_control_qc.csv` | zone、slice、u、控制井数、井名、控制值方差、实际 sill、归一化网格实际 range、kriging mode |
| `polygon_qc.csv` | polygon/class、顶点数、面积、范围、窗口交叠、栅格占比、`max/mean P_map`、`P_map>0.5/0.9` 面积 |
| `framework_class_qc.csv` | class 的线性 AI 倍率、taper、`max/mean P_map`、`P_map>0.5/0.9` 面积与占比、三维概率分布、修改样点数 |
| `well_framework_effect_qc.csv` | 每井 M0/M1、概率和 delta 的统计；不因井被修改而失败 |
| `section_metrics.csv` | 每条 QC 剖面的 M0/M1/delta/probability 统计 |

CSV 不嵌套 Python 对象。井名列表、参数字典等复杂字段必须使用确定性 JSON 字符串，并在 CSV 契约文档中明确。

`P_map` 指标只作事实记录，v1 不内置“约束过弱”告警阈值。使用者可直接看到窄 polygon 与大 `edge_taper_m` 是否导致概率从未接近 1；代码不得替地质解释自动缩小 taper。

### 10.5 图件

输出：

- `figures/framework_map_and_sections.png`：polygon、活动 class、控制井和 QC 线位置；
- 每条配置剖面一张图，固定面板为线性 AI 的 M0、线性 AI 的 M1、AI percent difference、class probability；
- M0/M1 默认显示 `AI=exp(logAI)`，单位为 `m/s*g/cm3`，并使用相同色标；
- 差值面板显示 `100 * (AI_M1 / AI_M0 - 1)`，使用以 0 为中心的发散色标；ΔlogAI 保留在 NPZ/CSV 中，不用无单位色标冒充线性 AI；
- 所有剖面叠加三个层位；
- 纵轴为 TVDSS m，向下增加；
- 图题写出 scenario id、活动 class、`linear_ai_multiplier` 和 taper，禁止生成无法追溯参数的“漂亮图”。

### 10.6 `run_summary.json`

schema 固定为 `depth_framework_lfm_v1`，至少包含：

```text
status
sample_domain=depth
depth_basis=tvdss
value_domain=log(AI)
ai_unit=m/s*g/cm3
scenario_id
active_framework_classes
resolved_config_and_hash
source_runs_and_hashes
seismic_and_horizon_hashes
polygon_file_and_hash
well_sources_and_hashes
axes_and_output_geometry
baseline_parameters
framework_parameters
source_segy_sample_format_code
segy_export_parameters
control_counts
output_files_and_hashes
warnings
```

成功状态只允许 `ok`。任何必需契约失败时不写伪成功 summary；若为审计写失败摘要，状态必须是 `failed` 并包含唯一失败原因，且不得留下可被下游误消费的主 NPZ。

## 11. 严格失败规则

以下情况必须失败：

- workflow 不是 `depth + tvdss`；
- 地震或层位轴不规则、层位交叉或目标窗无法建立；
- xline 步长被当作 1，或请求线号不在显式轴上；
- source run 缺文件、summary 状态不可消费、记录路径不一致或已有哈希不匹配；
- 候选井不是直井、KB/线号无效、井名联接不唯一；
- AI mnemonic、单位、正值或 TVDSS 轴不满足契约；
- 低通跨越缺口，或任何 slice 少于 3 口控制井；
- 建模结果使用了 single-well、neighbor-slice、constant-field 等 fallback mode；
- polygon CSV 多列/缺列、class 不匹配、顺序断裂、自交、退化或越出工区；
- `linear_ai_multiplier`/taper 缺失、非有限或越界；
- M1 与 M0 完全相同；
- M0/M1/probability/mask 的轴或 shape 不完全一致；
- 规范 M0/M1 `log_ai` 在有效样点出现 NaN/Inf，或在无效样点不是 NaN；
- window 与 volume 同一区域算法语义不同；
- window 模式尝试导出 SEG-Y；
- 源 SEG-Y sample format 不是 format code 5；
- SEG-Y 导出前的全体积 shape/轴与源体不完全一致，或回读后的 format/shape/轴/NaN 不满足契约；

禁止：

- 猜测 AI 单位或把线性 AI 当 logAI；
- 猜测 polygon class、层段或顶点顺序；
- 根据文件名扫描并拼接不在来源清单中的 LAS；
- 自动裁剪越界 polygon；
- 自动设置 `linear_ai_multiplier`、taper、切片数或窗口；
- 因控制不足而降低标准；
- 在规范 NPZ 或规范 AI SEG-Y 的目标窗外填 0、端点值或其他有限显示值；
- 在 v1 内生成 display-only filled SEG-Y；
- 用旧 GINN 相控点代码作新模块依赖。

## 12. 实施顺序

### 阶段 A：严格 M0

1. 实现深度 LFM 配置解析和来源审计；
2. 准备直井 TVDSS `grid.Log` 和连续 run 低通；
3. 构建完整 TargetZone 与 window/full 输出几何；
4. 在调用通用 modeling 前后执行 slice 控制严格门禁；
5. 写 M0 NPZ、控制 QC 和剖面基础图。

门禁：M0 在 xline step=4 下坐标正确；任何 slice fallback 均失败。

### 阶段 B：平面框架和 M1

1. 实现 CSV 解析、拓扑校验和 class 联接；
2. 在线号到 XY 的真实坐标中计算 inward-only taper；
3. 按逐道层位计算相对地层垂向概率；
4. 生成 M1、framework fields、井旁影响 QC 和多剖面图。

门禁：公式逐点可重建；圈外 delta 精确为 0；井位允许被修改并被记录。

### 阶段 C：全体积和导出

1. 验证 window/full 子区一致；
2. 控制峰值内存并生成全体积 NPZ；
3. 确认源 SEG-Y 为 format code 5，严格校验全体积轴后复用 `export_volume_like_source(..., nan_fill=None)` 导出规范 M0/M1 AI SEG-Y，并回读验证；
4. 冻结 schema、摘要和文件哈希。

门禁：cigsegy 回读后 TVDSS/inline/xline 头信息与源地震一致，目标窗外仍为 NaN。解释软件兼容性通过实际导入验收；若不兼容，本阶段只报告事实，不生成临时填充体。

## 13. 测试规范

测试由实现方写入 `tests/`，用户在本地环境运行。至少覆盖：

1. 直井 `MD-KB` 转 TVDSS，斜井明确失败；
2. 400 m、六阶、reflect 低通的轴、相位和单位；
3. NaN gap 被分段处理且不跨缺口；
4. 两个起伏层段的 32 个比例切片和逐道映射；
5. 每 slice 2 口井失败，3 口井可运行；
6. 历史 neighbor-fill、single-well 等 mode 被拒绝；
7. inline step=1、xline step=4 的坐标归一化和窗口裁取；
8. polygon 点内外判断、XY 米制距离和 inward-only raised cosine；
9. 自交、退化、越界、重复闭合顶点和断裂 vertex order 均失败；
10. 同 class 多 polygon 重叠时取最大概率；
11. 仅 reef、仅 shoal、reef+shoal 三种活动集合；
12. 起伏层位下 `u`、top/bottom taper 及窗外精确为 0；
13. log 域公式与线性 AI 幂乘公式逐点等价；
14. `linear_ai_multiplier` 大于 1、小于 1、等于 1 的组合校验；
15. 框架覆盖井位时 M1 确实改变井旁值并写入 QC；
16. window 与 volume 同一子区的 M0/M1/mask/probability 一致；
17. NPZ 必需键、dtype、shape、NaN 和 metadata schema；
18. `max/mean P_map` 与 `P_map>0.5/0.9` 面积统计正确，且不触发未配置的告警阈值；
19. window 禁止 SEG-Y，非 format code 5 源体明确失败；
20. format code 5 的规范 NaN SEG-Y 经 cigsegy 往返仍保留 NaN；
21. volume SEG-Y 的全体积 shape、轴、textual header、M0/M1 角色和数值正确；
22. 任意错位/窗口数组不能通过共享头路径导出；
23. 来源路径、配置、polygon、LAS、地震和层位哈希写入 summary；
24. 旧时间域 LFM、旧 GINN 和无 schema 产物不能被静默消费。

数值容差必须按 dtype 和计算路径显式设置。不得通过放宽全局容差掩盖线号偏移、边界概率、层位映射或 log/linear 值域错误。

## 14. 验收标准

v1 完成需同时满足：

- `scripts/real_field_lfm_depth.py` 是唯一的新深度域实际工区 LFM 入口；
- `cup.seismic.lfm_depth` 不依赖旧 GINN 或 `.ref` 代码；
- M0 严格来自同一个 `wavelet_batch_synthetic_depth` source run 的 shifted filtered AI、TVDSS、400 m 低通和 32-slice kriging；
- 每个 slice 至少 3 口井，无任何数值兜底；
- 平面 polygon 通过真实线号和 XY 米制距离生成框架概率；
- reef/shoal 纵向范围跟随三个解释层位；
- M1 可由 M0、各 class probability 和 `linear_ai_multiplier` 完整重建；
- M1 修改井位的事实在 metadata 和 QC 中明确可见；
- window/full 子区逐点一致；
- 规范 M0/M1 NPZ 和 AI SEG-Y 的目标窗外均为 NaN，不填 0 或显示值；
- 多剖面对比图、CSV、NPZ、summary 均可追溯到输入哈希和场景参数；
- SEG-Y 只在 full-volume 模式输出，固定为 format code 5 的 NaN 规范体；v1 不生成 display-only 产物；
- Synthoseis、GINN-v2、R0/R1 未在本阶段被修改。

## 15. 后续接缝

本规范完成后，下一份独立设计应处理“合成训练如何见到框架 LFM”。在 M1 被用于 GINN-v2 正式推理、训练对比或研究结论之前，必须先实现该合成 LFM variant 规范；只生成真实工区 M1 图件不构成正式模型接入。最低要求是：

- 合成 truth/seismic 不变；
- 同一 parent realization 派生 baseline/framework LFM variant；
- residual target 始终按当前输入 LFM 重新计算；
- split 仍按 parent realization，不得让同一真值的 LFM variants 泄漏到不同 split；
- 合成 framework LFM 使用与本文同构的 `log(linear_ai_multiplier)` 和概率场语义；
- GINN-v2 输入仍保持 `seismic + LFM + valid_mask`，除非另有独立规范改变。

这些接缝只用于界定未来方向，不授权本阶段实现合成训练或真实场反演。
