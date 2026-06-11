# 数据与单位约定

本文档分两部分：**Part A** 描述工作流用到的原始数据类型及其读取入口；**Part B** 描述所有物理量 & 坐标轴的单位约定。

脚本之间的 CSV 契约和中间产物（NPZ 等）见[核心 CSV 契约](csv-contracts.md)。

---

# Part A：数据

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

---

# Part B：物理量

## 物理量单位

| 物理量 | 单位 | 说明 |
|--------|------|------|
| 时间采样间隔 `dt_s` | s | 地震、子波、TWT 域曲线采样间隔 |
| 深度采样间隔 `dz_m` | m | 深度域地震或深度轴曲线采样间隔 |
| 声波时差 `DT_USM` | us/m | 第三步预处理后的纵波慢度曲线 |
| 密度 `RHO_GCC` | g/cm³ | 第三步预处理后的密度曲线 |
| 速度 `Vp/Vs` | m/s | 工作流内部属性 |
| 密度 `Rho` | g/cm³ | 工作流内部属性 |

## 坐标轴单位

| 坐标轴 | 单位/参照 | 说明 |
|------|-----------|------|
| TWT | s，双程旅行时 | 工作流内部为正秒 |
| MD | m，沿井轨迹向下 | 钻井测量深度 |
| TVDKB | m，从补心向下 | 垂深（以 Kelly Bushing 为基准） |
| TVDSS | m，从海平面向下 | 垂深（以 Mean Sea Level 为基准），海平面以下为正 |
| inline / xline | 线号（整数或浮点） | 地震工区网格坐标，非 XY 米制 |
| X / Y | m | 工区投影平面坐标 |
| `flat_idx` | 整数 | 地震体内部一维道编号，依赖当前几何 |
| `sample_index` | 整数 | 地震体内部采样点索引，依赖当前采样轴 |
