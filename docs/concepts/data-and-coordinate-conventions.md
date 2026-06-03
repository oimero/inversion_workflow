# 数据与坐标约定

这份文件描述数据、入口函数、坐标、单位等内容的约定。脚本之间的 CSV 字段契约见[核心 CSV 契约](csv-contracts.md)。

## 井头、井分层、地震解释层位

- Petrel 井头、井分层、地震解释层位读取入口：`cup.petrel.load.import_well_heads_petrel()`、`import_well_tops_petrel()`、`import_interpretation_petrel()`。

## 井名

- 井名匹配键入口：`cup.well.assets.normalize_well_name(name)`。
    - 处理等价于 `str(name).strip().casefold()`：转字符串、去首尾空白、做大小写折叠。
    - 简单理解：`" A1 "`、`"A1"`、`"a1"` 会被当成同一口井。
- 文件查找、DataFrame join、lookup dict 都用规范化键；输出 CSV 保留原始显示井名。
- 同一资产目录或表格内若出现 `A1/a1` 这类冲突，应在导入阶段报错，不能静默覆盖。

## 井曲线

- LAS 单曲线读取入口：`cup.well.las.read_las_curve(path, mnemonic)`。
    - 只按用户指定 mnemonic 读曲线，不做单位转换、不做语义分类。
    - 匹配顺序是精确 mnemonic 优先，再用 LASIO 的 `:1` / `:2` 后缀规范化兜底；兜底命中多条时必须显式指定精确名。
- 标准预处理 LAS 读取入口：`cup.well.las.load_vp_rho_logset_from_standard_las(path)`。
    - 只读取含 `DT_USM` 和 `RHO_GCC` 的标准 LAS，并转换成 `Vp/Rho` 的 `grid.LogSet`。

## 时深表

- 读取入口：`cup.well.td.load_petrel_time_depth_table(path, domain="md" | "tvdss")`。
- Petrel 导出的时深表常见口径：
    - `TWT`：文件中为负毫秒。
    - `MD`：文件中为正米，向下为正。
    - `Z`：文件中为负米，表示地下深度。
- 工作流内部统一转换为：
    - `TimeDepthTable.twt`：正秒。
    - `TimeDepthTable.md`：正米，向下为正，仅在 `domain="md"` 时存在。
    - `TimeDepthTable.tvdss`：正米，向下为正，仅在 `domain="tvdss"` 时存在。

## 井轨迹

- 轨迹读取入口：`cup.well.trajectory.WellTrajectory.from_petrel_trace(path)`。
- 轨迹里的 `tvdss_m` 不是直接读 `Z` 字段，而是逐轨迹点计算：
    - `tvdss_m = tvd_kb_m - kb_m`
    - 含义：轨迹点相对海平面的垂向位置，海平面以下为正，海平面以上为负。
    - 因此，浅部轨迹点的 `tvdss_m` 可以为负。
- 易错点：
    - `TimeDepthTable.tvdss` 是时深表深度轴，当前通过 `abs(Z)` 进入项目，通常只服务地下目标层。
    - `WellTrajectory.tvdss_m` 是轨迹几何事实，保留海平面上下的符号。
    - 深部目标层通常二者同为正值；浅部不能直接混用。

## 地震工区几何

- 井旁道读取入口：`cup.seismic.survey.open_survey()`。
    - 返回 `SurveyContext`，用 `survey.read_trace_at_xy(x, y, domain="time" | "depth")` 读取井旁道。
    - 旧的 `cup.petrel.load.import_seismic()` 只读取整块 3D 体为数组，主要服务深度学习数据准备；井震标定和井位采样不要用它取井旁道。
- 工区几何入口：`survey.line_geometry`，类型为 `cup.seismic.geometry.SurveyLineGeometry`。
- `geometry["inline_step"]` / `geometry["xline_step"]` 是线号步长，不是 XY 米制间距。物理距离必须通过 `SurveyLineGeometry.line_to_coord()` 或已构建的 XY 网格计算。
- 最近道吸附公式：`line_min + round((line_float - line_min) / line_step) * line_step`。
    - 它用于 `nearest_inline`、`nearest_xline`、同道冲突检查、轨迹点落道报告。
- 井旁道双线性插值公式：先由 `SurveyLineGeometry.coord_to_index(x, y)` 得到浮点 index `(i, j)`，令 `wi = i - floor(i)`、`wj = j - floor(j)`，则：

```text
trace =
  (1 - wi) * (1 - wj) * trace00
  + (1 - wi) * wj       * trace01
  + wi       * (1 - wj) * trace10
  + wi       * wj       * trace11
```

## 第六步井约束：NPZ 数据包与步骤间契约

第六步 `well_constraints.py` 输出的 CSV 字段契约见[核心 CSV 契约](csv-contracts.md)。这里的重点是它产出的两个 NPZ 数据包和统计文件，它们是第七、八步和 enhance 的正式输入。

### `log_ai_anchor_time.npz` → 第八步 GINN

| NPZ 键 | 形状 | 语义 |
|--------|------|------|
| `samples` | `(n_sample,)` | 正秒 TWT 采样轴 |
| `flat_indices` | `(n_anchor,)` | 每条受控道的地震体内部编号 |
| `target_ai` | `(n_anchor, n_sample)` | 第六步分频后的低频波阻抗 |
| `target_log_ai` | `(n_anchor, n_sample)` | 低频波阻抗的对数，由 `target_ai` 自动推导 |
| `anchor_mask` | `(n_anchor, n_sample)` | 布尔掩码，标记哪些样点是有效控制点 |
| `anchor_weight` | `(n_anchor, n_sample)` | 每个控制点的训练权重 |
| `anchor_names` | `(n_anchor,)` | 每条道的来源井名 |
| `anchor_types` | `(n_anchor,)` | 锚点类型，当前均为 `"well"` |
| `inline` / `xline` | `(n_anchor,)` | 每条道的线号 |
| `summary_json` | 标量 | 锚点数、有效样点数、目标值统计、权重统计 |
| `metadata_json` | 标量 | 分频参数、冲突策略、上游路径 |

第一版默认只含直井锚点。第八步训练时，`anchor_mask` 为真的样点会参与井约束损失。

### `well_high_supervision_time.npz` → enhance 训练

遵循 `enhance.prior` 的 `WellResolutionPriorBundle` 格式（schema v3）。在第六步的语境下：

| 可用字段 | 语义 |
|----------|------|
| `well_high_log_ai` | 井上高频残余，enhance 训练的真实井监督目标 |
| `well_low_ai` / `well_low_log_ai` | 井上低频成分 |
| `well_ai` / `well_log_ai` | 井上全频波阻抗 |
| `well_mask` / `well_weight` | 有效样点掩码和训练权重 |
| `well_names` / `inline` / `xline` | 每口井的标识和位置 |
| `highres_depth` | 实际存储正秒 TWT（保留旧字段名以兼容 enhance.prior） |
| `highres_well_*` | 各字段的"高采样"副本，第六步与原始采样一致 |

不可用的占位字段（全零）：

| 字段 | 原因 |
|------|------|
| `lfm_log_ai` / `lfm_ai` / `highres_lfm_log_ai` | LFM 是第七步的产物，第六步执行时尚不存在 |

enhance 训练端若需要底阻抗输入，应从 GINN 或 LFM 主文件读取，不应从这份井监督包中读取。

### 统计文件 → enhance 合成器

| 文件 | 格式 | 消费者 |
|------|------|--------|
| `well_high_stats_global.json` | JSON，含振幅、事件密度、持续长度、转移矩阵、反射率和频谱的全局统计 | enhance 合成器，全窗预训练阶段 |
| `well_high_stats_by_layer.csv` | CSV，每层经验统计和可靠度 | enhance 合成器，分层微调阶段 |
| `well_high_stats_shrinkage.json` | JSON，每层向全窗收缩后的最终生成参数 | enhance 合成器的真实生成参数来源 |
| `well_high_motif_manifest.csv` | CSV，第一版为空占位 | 后续 motif bank 功能的接口 |

统计文件的核心约定：输出不写单一均值，保留分位数（p10/p50/p90），让合成器从分布中抽样而非永远取中位数。每层可靠度由井数、样点数、事件数三者综合决定。

### 跨步骤接口一览

```
第六步 well_constraints.py
  ├─ lfm_layer_control_points.csv  → 第七步 lfm_precomputed.py（顺层插值建模）
  ├─ lfm_control_qc.csv            → 第七步（控制井审计）
  ├─ log_ai_anchor_time.npz        → 第八步 ginn_train.py（井约束损失）
  ├─ well_high_supervision_time.npz → enhance 训练（真实井高频监督）
  ├─ well_high_stats_*.json/csv    → enhance 合成器（统计驱动生成）
  └─ well_high_motif_manifest.csv  → enhance 合成器（motif bank 占位）
```

每个下游步骤只读取第六步的输出，不应再直接访问第四步的 LAS、时深表或井轨迹。

## 单位约定

| 物理量 | 单位 | 说明 |
|--------|------|------|
| TWT | s | 内部正秒 |
| MD / TVDKB | m | 向下为正 |
| 内部 TDT 的 TVDSS | m | 向下为正 |
| 时间采样间隔 `dt_s` | s | 地震、子波、TWT 域曲线采样间隔 |
| 深度采样间隔 `dz_m` | m | 深度域地震或深度轴曲线采样间隔 |
| 声波时差 `DT_USM` | us/m | 第三步预处理后的纵波慢度曲线 |
| 密度 `RHO_GCC` | g/cm3 | 第三步预处理后的密度曲线 |
| 速度 `Vp/Vs` | m/s | 工作流内部属性 |
| 密度 `Rho` | g/cm3 | 工作流内部属性 |
