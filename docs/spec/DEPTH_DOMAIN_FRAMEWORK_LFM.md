# 深度域框架约束 LFM 实施规范

> 状态：已被 [`UNIFIED_REAL_FIELD_LFM_V2.md`](UNIFIED_REAL_FIELD_LFM_V2.md) 取代，禁止直接实施
> 范围：深度域实际工区 Step 7；M0 层状插值 LFM 与 M1 独立 framework body 约束 LFM
> 当前工区：TVDSS，向下为正；inline 步长 1，xline 步长 4，深度采样 5 m；当前控制井均为直井
> 本文仅保留 framework body、SEG-Y 和早期深度域方案的历史讨论。新的 Step 6/7、baseline、variant、framework modifier 和 R0 接缝一律以统一 v2 规范为准；不得据本文新增 `real_field_lfm_depth.py`、`lfm_depth.py` 或 M0/M1 生产接口。

## 1. 目标

导师要求把“框架约束反演”首先体现在初始波阻抗模型中，并与原来的层状插值模型对比。v1 不修改 GINN-v2 的输入通道或输出语义，而是构建两套同网格、可直接比较的深度域 LFM：

- **M0**：由深度校正后的井 AI 曲线、解释层位和比例切片克里金生成的层状插值 LFM；
- **M1**：在 M0 上叠加独立礁/滩 framework body 的场景 LFM。

每个 framework body 同时回答目标体“横向在哪里”和“位于母层段的哪个局部纵向窗”。平面 polygon 定义横向展布，`u_top/u_bottom` 定义该 body 在对应母层段中的固定相对地层范围：

```text
reef_body  = reef_polygon(inline, xline) × local_interval(u_top, u_bottom; base_of_salt, base_of_bve)
shoal_body = shoal_polygon(inline, xline) × local_interval(u_top, u_bottom; base_of_bve, base_of_itp)
```

同一平面位置可以由多个 `body_id` 分别定义上部、中部和下部目标体；它们之间的纵向空隙保持为 M0。body 的相对纵向位置和厚度在自身 polygon 内不随平面位置变化，v1 不从 polygon 自动推断透镜形态或横向变厚变薄。因此，M1 的准确表述是“独立平面 body + 层位控制的局部纵向窗初始模型”，不是由剖面精细勾勒并插值得到的三维礁丘几何体。

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
- 根据平面 polygon 自动推断 body 的透镜形态、横向变厚变薄或顶底曲面；
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

`.ref/scripts_depth/lfm_precomputed_depth.py` 和旧 `lfm_depth.py` 只用于审计数学顺序：先转 TVDSS、再低通井曲线、最后比例切片建模。旧相控代码采用局部控制点和半径，不等于本规范的独立 framework body，不能成为新接口或数据契约。

时间域 `scripts/real_field_lfm.py` 固定使用 TWT 语义，不得通过替换字段名改造成深度域入口。

## 4. 代码边界

规划新增：

```text
scripts/real_field_lfm_depth.py
src/cup/seismic/lfm_depth.py
experiments/real_field_lfm_depth/lfm.yaml
experiments/real_field_lfm_depth/framework_bodies.csv
```

其中：

- `scripts/real_field_lfm_depth.py` 只负责 CLI、配置合并、来源解析、产物落盘和终端摘要；
- `cup.seismic.lfm_depth` 负责配置值对象、井控制准备、米制低通、严格比例切片、framework body 栅格化、M1 合成、QC 统计和绘图数据准备；
- 实验 YAML 存放可变场景假设；
- `common.yaml` 继续只存工区地震、层位和资产事实，不写 `linear_ai_multiplier` 或 body 文件路径。

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
    bodies_file: experiments/real_field_lfm_depth/framework_bodies.csv
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
- `framework.bodies_file` 必须显式存在并指向满足 §6 的 CSV；解析路径和文件哈希必须进入 NPZ metadata 与 summary；
- `framework.classes` 必须为非空映射，只允许 `reef`、`shoal`；允许仅启用一类；
- `reef` 和 `shoal` 的层位对必须分别精确匹配 §3.1；
- `linear_ai_multiplier` 表示线性 AI 域倍率，必须有限且大于 0；所有活动类别倍率均为 1 时失败，因为 M1 将与 M0 相同；
- `edge_taper_m` 必须有限且大于 0；
- top/bottom taper fraction 必须分别位于 `(0, 0.5)`，且二者之和小于 1；二者相对于每个 body 自身厚度计算；
- `linear_ai_multiplier`、`edge_taper_m` 和 top/bottom taper fraction 均为 class 级参数，body 不得在 CSV 中单独覆盖；
- `qc.sections` 必须非空，每条线必须落在当前输出轴上；xline 必须遵守步长 4，不能按数组下标解释。

来源目录可以显式给出，也可按 `cup.config.sources` 发现最新满足契约的运行；发现是路径解析策略，不是数值或语义兜底。最终解析路径和文件哈希必须写入 summary。

## 6. Framework body CSV 契约

`framework_bodies.csv` 固定包含以下七列，不接受别名或额外列：

| 字段 | 类型 | 含义 |
|---|---|---|
| `body_id` | 非空字符串 | 独立 framework body 标识；可在自身顶点行中重复，但只能对应一个 body |
| `framework_class` | 枚举 | `reef` 或 `shoal` |
| `u_top` | 浮点数 | body 顶界在 class 母层段中的相对地层坐标 |
| `u_bottom` | 浮点数 | body 底界在 class 母层段中的相对地层坐标 |
| `vertex_order` | 整数 | 同一 body 的平面 polygon 内从 0 开始连续编号 |
| `inline` | 浮点线号 | polygon 顶点 inline；不是数组下标 |
| `xline` | 浮点线号 | polygon 顶点 xline；不是数组下标 |

一个 `body_id` 同时定义一个平面 polygon 和一个局部纵向窗。相同平面位置存在多个分离 body 时，使用不同 `body_id` 和不同 `u_top/u_bottom`；若平面轮廓相同，允许在各 body 下重复同一组顶点。例如：

```csv
body_id,framework_class,u_top,u_bottom,vertex_order,inline,xline
reef_upper_001,reef,0.10,0.25,0,1630.0,5903.0
reef_upper_001,reef,0.10,0.25,1,1660.0,5903.0
reef_upper_001,reef,0.10,0.25,2,1660.0,6103.0
reef_upper_001,reef,0.10,0.25,3,1630.0,6103.0
reef_lower_001,reef,0.65,0.82,0,1630.0,5903.0
reef_lower_001,reef,0.65,0.82,1,1660.0,5903.0
reef_lower_001,reef,0.65,0.82,2,1660.0,6103.0
reef_lower_001,reef,0.65,0.82,3,1630.0,6103.0
```

规则：

- 同一 `body_id` 的 `framework_class`、`u_top` 和 `u_bottom` 必须逐行完全一致；禁止按首行猜测或静默归并冲突值；
- `reef` 的 `u_top/u_bottom` 相对于 Salt–BVE，`shoal` 相对于 BVE–ITP；
- `u_top/u_bottom` 必须有限，并满足 `0 <= u_top < u_bottom <= 1`；`[0, 1]`、单侧接触母层顶底界和完全位于层内的区间均合法；
- `[0, 1]` 只是接口允许的显式输入，不得成为缺失 `u_top/u_bottom` 时的默认值；
- body 的 `u_top/u_bottom` 在其 polygon 内保持固定，不根据边界距离或其他属性自动改变厚度；
- 每个 body 的 polygon 至少包含三个互异顶点；
- 最后一个顶点不得重复第一个顶点，闭合由程序完成；
- `vertex_order` 必须唯一、连续且从 0 开始；
- 顶点必须有限并位于完整地震工区线号范围内；
- 顶点可以是浮点线号，不要求落在道中心；
- polygon 不得自交、退化或面积为 0；
- CSV 中出现的 class 必须在当前 M1 配置中活动，活动 class 也必须至少有一个 body；
- 同一 class 允许多个 body 在平面和纵向上重叠；三维概率按 §8.3 逐点取最大值，不做概率相加；
- window 模式允许 body 部分或全部位于窗口外，但至少一个活动 body 必须在窗口离散网格上产生正概率样点；
- 不允许把越出完整工区的 polygon 静默裁剪回工区。

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

## 8. M1 framework body 模型

### 8.1 Body 横向概率

每个 body 的 polygon 顶点和输出道中心均通过 `SurveyLineGeometry.line_to_coord()` 转为真实 XY 米制坐标。点是否在 polygon 内由完整几何判断；边界距离使用点到该 polygon 全部线段的最短 XY 距离。

对 body `b`，令内部点到边界的距离为 `d_xy`，其 class 配置宽度为 `t=edge_taper_m`：

```text
P_map_body_b = 0                                      outside or on boundary
P_map_body_b = 0.5 * (1 - cos(pi * d_xy / t))        0 < d_xy < t
P_map_body_b = 1                                      d_xy >= t
```

这是仅向内渐变：body polygon 外严格为 0，不因软边界修改圈外 LFM。不同 body 必须分别保留自己的二维概率直到完成三维 body 计算；不得先把同类 polygon 合并成一张 `P_map_class` 再进入垂向计算。

用于平面 QC 的 class map 可以定义为：

```text
P_map_class_qc = max(P_map_body_b for body_b in class)
```

该二维场只表示同类 body 平面展布的并集，不含 `u_top/u_bottom`，不得参与 M1 或 class 三维概率重建。

### 8.2 Body 垂向概率

对每个输出道，先按 body 对应 class 的母层段计算：

```text
u = (z - z_zone_top) / (z_zone_bottom - z_zone_top)
v_body = (u - u_top_body) / (u_bottom_body - u_top_body)
```

其中 reef 的 `z_zone_top/z_zone_bottom` 为 Salt/BVE，shoal 为 BVE/ITP。层位起伏时 `u` 逐道变化，因此 body 跟随地层，不形成绝对深度水平盒子。body 的有效纵向支撑只在 `0 <= v_body <= 1`；其外严格为 0。

设该 class 配置的 `f_top`、`f_bottom` 为 taper fraction。二者相对于 body 自身厚度 `u_bottom_body-u_top_body` 计算，而不是相对于整个母层厚度：

```text
P_vertical_body = 0.5 * (1 - cos(pi * v_body / f_top))
                  when 0 <= v_body < f_top

P_vertical_body = 1
                  when f_top <= v_body <= 1-f_bottom

P_vertical_body = 0.5 * (1 - cos(pi * (1-v_body) / f_bottom))
                  when 1-f_bottom < v_body <= 1

P_vertical_body = 0
                  outside the body interval
```

程序不得因 body 过薄、采样错位或 taper 过宽而自动扩大 `[u_top,u_bottom]`、吸附最近 TVDSS 样点或修改 taper。

### 8.3 Body 到 class 的三维聚合

每个 body 必须先独立形成完整三维概率：

```text
P_body_b(inline, xline, z)
  = P_map_body_b(inline, xline) * P_vertical_body_b(inline, xline, z)
```

之后同一 class 内逐点取最大值：

```text
P_class(inline, xline, z)
  = max(P_body_b(inline, xline, z) for body_b in class)
```

取最大值把同类 body 解释为几何并集，避免同一位置因 body 拆分或重叠而重复施加倍率。禁止概率相加、相加后截断，或先合并二维 map 再乘任意 class 级垂向窗。

同一平面位置的上、中、下 body 使用不同 `[u_top,u_bottom]` 时，会生成多个分离的纵向概率段；各 body 支撑之外及其间空隙的 `P_class` 必须精确为 0。reef 和 shoal 分别在自己的母层段聚合；共享 BVE 层位处因相应 body taper 回到 0，M1 在该界面精确回到 M0。

### 8.4 M1 合成

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

同一 class 的所有 body 共用该 class 的 `linear_ai_multiplier`、`edge_taper_m`、`top_taper_fraction` 和 `bottom_taper_fraction`；v1 不接受 body 级参数覆盖。

M1 不做第二次 400 m 低通。框架的横向/垂向 taper 是独立、显式的场景尺度，必须分别进入 metadata 和 QC。M1 是 `framework-scenario initial model`，不保证满足与 M0 相同的整体频谱低通定义；不能因为文件名含 LFM 就宣称其全部空间变化均满足 400 m 截止波长。

框架允许直接修改控制井位置，程序不得构造井保护半径或强制 M1 回到 M0。summary 和井旁 QC 必须明确记录每口井在各 class 下的最大概率、M0/M1 差值和倍率影响。因此 M1 必须标注为：

```text
framework_scenario_model; not strictly well-honoring after framework modification
```

## 9. 窗口、全体积和内存策略

### 9.1 Window 模式

window 的 inline/xline 边界必须落在地震显式轴上。输出仅包含该矩形子网格，但 M0 kriging 仍使用完整候选井集合，包括窗口外控制井。

层位应先在完整工区建立统一表面和井位采样，再裁取窗口输出网格，避免因窗口边界改变井位层位值或插值结果。window 与 full-volume 在相同子网格上的 M0、M1、概率和掩码必须逐点一致。

window 模式允许 body 部分或全部位于窗口外。与窗口相交的每个 body 必须在离散输出网格上至少产生一个 `P_body>0` 的有效样点；完全位于窗口外的 body 只记录 `window_intersects=false`，不因本次窗口未计算而失败。至少一个活动 body 必须在窗口中产生正概率并实际修改 M1，否则运行失败。

window 模式不导出 SEG-Y，只写 NPZ、CSV、JSON 和 PNG。

### 9.2 Volume 模式

当前全体积为 `601 × 801 × 551`，单个 float32 体约 1 GiB。实现不得同时保留不必要的 M0/M1/AI 临时副本：

- 以 float32 保存主数组；
- 框架概率按活动 class 分配；
- 不保存每个 body 的全体积三维概率；body 临时概率应逐体聚合到 class 最大值后释放；
- 线性 AI 仅在导出时分块或短生命周期转换；
- M0 kriging variance 独立保存，不复制为虚假的 M1 uncertainty；
- 不把完整 XY 点云为每个 body 重复实体化。

volume 模式要求 CSV 中每个 body 都在离散全体积网格上至少产生一个 `P_body>0` 的有效样点。任何 body 因 polygon 未覆盖道中心、纵向窗过薄、采样错位或 taper 而完全不可见时必须失败，不得忽略该 body 或自动改变几何。

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
<class>_map_probability      [n_inline, n_xline]              # 仅平面 QC
<class>_probability          [n_inline, n_xline, n_tvdss]
```

`<class>_map_probability` 是同类 body 二维 map 的逐点最大值，只用于平面 QC；它不包含各 body 的纵向窗，不得用于生成或重建 `<class>_probability`。`framework_fields.npz` 不保存每个 body 的独立三维概率。

`combined_framework_probability` 仅用于 QC，定义为活动 class 三维概率的逐点最大值，不能用于重建 M1；M1 必须按各 class 自己的 `linear_ai_multiplier` 分项计算。各 class 三维概率必须由 §8.3 的逐 body 计算与聚合得到。

`metadata_json` 必须记录 body CSV 的解析路径和哈希、按 `body_id` 排序的 class/`u_top`/`u_bottom` 清单、class 级倍率与 taper，以及 `body_then_class_max` 聚合规则；不得要求逐 body 三维体才能解释规范产物。

### 10.4 QC 表

至少输出：

| 文件 | 内容 |
|---|---|
| `well_control_qc.csv` | 井来源、KB、线号、TVDSS 支撑、单位、低通状态 |
| `slice_control_qc.csv` | zone、slice、u、控制井数、井名、控制值方差、实际 sill、归一化网格实际 range、kriging mode |
| `framework_body_qc.csv` | body/class、`u_top/u_bottom`、相对厚度、顶点数、polygon 面积/范围、窗口交叠、实际厚度 min/median/max、水平和三维概率统计、正概率体素数、修改样点数 |
| `framework_class_qc.csv` | class 的线性 AI 倍率、taper、`max/mean P_map`、`P_map>0.5/0.9` 面积与占比、三维概率分布、修改样点数 |
| `well_framework_effect_qc.csv` | 每井 M0/M1、概率和 delta 的统计；不因井被修改而失败 |
| `section_metrics.csv` | 每条 QC 剖面的 M0/M1/delta/probability 统计 |

CSV 不嵌套 Python 对象。井名列表、参数字典等复杂字段必须使用确定性 JSON 字符串，并在 CSV 契约文档中明确。

`framework_body_qc.csv` 中的实际物理厚度按逐道母层厚度计算：

```text
body_thickness_m
  = (u_bottom - u_top) * (z_zone_bottom - z_zone_top)
```

min/median/max 只统计该 body polygon 在当前输出网格内覆盖且层位有效的道。完全位于 window 外的 body 将这些窗口统计写为 NaN，并明确记录 `window_intersects=false`；volume 模式不存在该豁免。

`P_map` 指标只作事实记录，v1 不内置“约束过弱”告警阈值。使用者可直接看到窄 polygon 与大 `edge_taper_m` 是否导致概率从未接近 1；代码不得替地质解释自动缩小 taper。与输出网格相交但 `positive_probability_voxel_count=0` 不属于弱约束告警，而是 §11 的严格失败。

### 10.5 图件

输出：

- `figures/framework_map_and_sections.png`：每个 body 的 polygon、`body_id`、`[u_top,u_bottom]`、活动 class、控制井和 QC 线位置；
- 每条配置剖面一张图，固定面板为线性 AI 的 M0、线性 AI 的 M1、AI percent difference、class probability；
- class probability 剖面必须能够看出同一平面位置多个分离 body 及其间 `P_class=0` 的空隙；必要时叠加相交 body 的边界和 `body_id`，但不额外保存逐 body 全体积概率；
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
body_file_and_hash
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
- body CSV 多列/缺列、`body_id` 为空、同一 `body_id` 的 class/`u_top`/`u_bottom` 不一致、class 不匹配、相对坐标越界或顺序非法；
- body polygon 顶点顺序断裂、自交、退化或越出工区；
- window 模式没有任何 body 实际影响窗口，或与窗口相交的任一 body 在离散网格上没有正概率样点；
- volume 模式任一 body 在离散网格上没有正概率样点；
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
- 猜测 body class、母层段、`u_top/u_bottom` 或顶点顺序；
- 用 `[0,1]` 填补缺失的 `u_top/u_bottom`；
- 根据文件名扫描并拼接不在来源清单中的 LAS；
- 自动裁剪越界 polygon，或自动扩大/吸附不可见 body；
- 在进入垂向计算前把同类 body 合并为二维 class map；
- 对重叠 body 的概率求和或相加后截断；
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

### 阶段 B：Framework body 和 M1

1. 实现 body CSV 解析、逐 body 字段一致性校验、polygon 拓扑校验和 class 联接；
2. 在线号到 XY 的真实坐标中逐 body 计算 inward-only 横向 taper；
3. 按逐道母层位、`u_top/u_bottom` 和 body 局部坐标计算垂向概率；
4. 先形成独立三维 body，再按 class 逐点取最大值；
5. 生成 M1、framework fields、body/class/井旁影响 QC 和多剖面图。

门禁：同一道可出现多个分离 body，body 间隙和圈外 delta 精确为 0；class 概率及 M1 公式逐点可重建；不可见 body 严格失败；井位允许被修改并被记录。

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
8. 逐 body polygon 点内外判断、XY 米制距离和 inward-only raised cosine；
9. 自交、退化、越界、重复闭合顶点和断裂 vertex order 均失败；
10. 同一 body 各顶点行的 class、`u_top` 或 `u_bottom` 不一致时失败；
11. 相同 polygon、不同纵向窗可在同一道生成上、中、下三个分离 body；
12. 分离 body 之间及所有 body 支撑之外的 `P_class` 精确为 0；
13. 不同平面轮廓和不同纵向窗可独立组合；
14. 同 class 多 body 三维重叠时逐点取最大概率，不求和且不重复施加倍率；
15. 起伏层位下母层坐标 `u` 和局部坐标 `v_body` 正确，top/bottom taper 相对于 body 自身厚度；
16. `[0,1]`、单侧接触母层界面和完全位于层内的区间均合法，非法次序/越界失败，缺失值不默认成 `[0,1]`；
17. 与输出窗口相交但因过薄、采样错位或无道中心覆盖而完全不可见的 body 失败，不自动加厚或吸附；
18. window 外 body 可记录为不相交而不失败，volume 模式则要求所有 body 产生正概率样点；
19. 仅 reef、仅 shoal、reef+shoal 三种活动集合；
20. log 域公式与线性 AI 幂乘公式逐点等价，聚合后的 class 概率可逐点重建 M1；
21. `linear_ai_multiplier` 大于 1、小于 1、等于 1 的组合校验；
22. 框架覆盖井位时 M1 确实改变井旁值并写入 QC；
23. window 与 volume 同一子区的 M0/M1/mask/class probability 一致；
24. NPZ 必需键、dtype、shape、NaN 和 metadata schema，且不包含逐 body 全体积概率；
25. `max/mean P_map`、`P_map>0.5/0.9` 面积、body 物理厚度和正概率体素数统计正确；
26. window 禁止 SEG-Y，非 format code 5 源体明确失败；
27. format code 5 的规范 NaN SEG-Y 经 cigsegy 往返仍保留 NaN；
28. volume SEG-Y 的全体积 shape、轴、textual header、M0/M1 角色和数值正确；
29. 任意错位/窗口数组不能通过共享头路径导出；
30. 来源路径、配置、body CSV、LAS、地震和层位哈希写入 summary；
31. 旧时间域 LFM、旧 GINN 和无 schema 产物不能被静默消费。

数值容差必须按 dtype 和计算路径显式设置。不得通过放宽全局容差掩盖线号偏移、边界概率、层位映射或 log/linear 值域错误。

## 14. 验收标准

v1 完成需同时满足：

- `scripts/real_field_lfm_depth.py` 是唯一的新深度域实际工区 LFM 入口；
- `cup.seismic.lfm_depth` 不依赖旧 GINN 或 `.ref` 代码；
- M0 严格来自同一个 `wavelet_batch_synthetic_depth` source run 的 shifted filtered AI、TVDSS、400 m 低通和 32-slice kriging；
- 每个 slice 至少 3 口井，无任何数值兜底；
- 每个独立 body 的 polygon 通过真实线号和 XY 米制距离生成横向概率；
- reef/shoal body 分别跟随对应母层位，并由显式 `u_top/u_bottom` 限定局部纵向窗；
- 同一平面位置可表达多个分离 body，body 间空隙不修改 M0；
- 同类 body 先独立形成三维概率再逐点取最大值，不使用二维 class map 生成三维场；
- 与输出网格相交但完全不可见的 body 严格失败，不自动修改输入几何；
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
