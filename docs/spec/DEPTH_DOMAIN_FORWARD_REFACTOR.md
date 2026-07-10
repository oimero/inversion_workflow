# 深度域正演能力重构设计

## 1. 目标

在 `src/cup/physics/` 建立统一物理能力模块：同时支持时间域和深度域的正演内核，并承载域无关的岩石物理关系，供 Synthoseis-lite、GINN v2 和 R1 共同使用。

核心性质：

- NumPy 与 PyTorch 后端提供同名、同语义 API。
- 时间域保留现有 Robinson 正演的数值语义。
- 深度域使用非平稳的纯深度域正演矩阵。
- 深度地震一律使用 TVDSS，单位为米，向下为正。
- 时间子波在时间域和深度域正演中都使用秒制时间轴。
- 核心内核只计算物理振幅，不填 NaN、不自动归一化、不施加 gain。
- 新实现不修改、不依赖 `src/ginn/`、`src/ginn_depth/` 中的遗留实现。

## 2. 非目标

- 叠前、角度道集、AVO/AVA 或各向异性；
- 斜井的 MD—TVDSS 轨迹变换；
- 上覆层绝对双程旅行时恢复；
- 深度域子波本身的定义或估计；
- 对遗留 `ginn`、`ginn_depth`、`wtie` 包进行迁移或清理；
- 读取旧 benchmark、checkpoint 或诊断产物并自动升级。

当前井均为直井，使用 `TVDSS = MD - KB`。未来支持斜井时，必须引入井轨迹并单独设计。

## 3. 核心域差异

深度域中，相同时间子波在不同速度和深度位置对应不同的米制宽度，正演算子随位置变化。因此时间域的平稳卷积、`twt_*` 命名和 `mode="same"` 裁剪契约不能直接搬到深度域。已通过统一正演内核分离两个域的算子构造。

## 4. 物理与离散约定

### 4.1 轴、单位和形状

- `log_ai[..., i] = ln(AI_i)`，最后一维长度为 `N`；
- `velocity_mps[..., i] = Vp_i`，单位 `m/s`，形状与 `log_ai` 相同；
- `depth_m[i] = z_i`，一维公共 TVDSS 轴，单位米，长度 `N`，严格递增；
- `wavelet_time_s[k]` 为规则采样的秒制子波时间轴，长度 `M`；
- `wavelet_amp[k]` 为子波振幅，长度 `M`。

批量维使用 `...` 表示。v1 中一批样本共享同一个一维深度轴和同一个时间子波。

### 4.2 反射率

反射率挂在下部界面：

```text
r[..., j] = tanh((log_ai[..., j+1] - log_ai[..., j]) / 2)
```

输入 `log_ai` 长度为 `N`，输出反射率长度为 `N-1`。不补零，不伪装成 N 点数据。

### 4.3 统一正演输出维度

```text
logAI            N
reflectivity     N-1
W_time           N × (N-1)
W_depth          N × (N-1)
seismic          N
```

**时间域。** 显式定义事件时间轴 `event_twt_s[j] = twt_s[j+1]`。`s_time[1:]` 是按显式挂点定义得到的常规居中 Robinson 结果；首点 `s_time[0]` 由有限子波支撑正常计算，不是补零。

**深度域。** 非平稳算子，输出为 TVDSS 上的 `N` 点。界面时间使用相邻样点累计 TWT 的中点 `t_interface_j = 0.5 * (t_sample_j + t_sample_{j+1})`。

统一后时间域和深度域的消费契约完全一致：logAI、地震和 TVDSS/TWT 轴均为 `N` 点；反射系数和 `forward_valid_mask` 为 `N-1`；不再按域区分丢弃首点或补零逻辑。

### 4.4 相对双程旅行时

使用梯形慢度积分：

```text
Δz_i       = z_{i+1} - z_i
Δtwt_i     = 2 * Δz_i * 0.5 * (1 / v_i + 1 / v_{i+1})
t_sample_0 = 0
t_sample_i = Σ_{k=0}^{i-1} Δtwt_k
t_interface_j = 0.5 * (t_sample_j + t_sample_{j+1})
```

速度必须是有限正数。深度必须有限且严格递增。

### 4.5 深度域非平稳正演

```text
W_depth[..., l, j] = w(t_sample[..., l] - t_interface[..., j])
d_depth[..., l]    = Σ_j W_depth[..., l, j] * r[..., j]
```

- `W_depth` 形状 `[..., N, N-1]`；输出 `d_depth` 形状 `[..., N]`；
- `w(τ)` 由 `wavelet_time_s`、`wavelet_amp` 做一维线性插值；
- 超出子波时间支撑范围时振幅为 0；
- 默认按 64 个输出深度样点分块计算；`return_operator=True` 时返回完整 `W_depth`。

### 4.6 时间子波契约

- `wavelet_time_s` 与 `wavelet_amp` 一维、有限、等长；
- 长度为奇数且至少为 3；
- 时间轴严格递增并规则采样；
- 中心样点的时间必须为 0；
- 不接受偶数长度后自动补零；
- 不在核心中自动重采样、归一化或移相。

### 4.7 AI—Vp 关系

```text
AI = a * Vp + b
Vp = (AI - b) / a
```

- AI：`m/s*g/cm3`；Vp：`m/s`；a：`g/cm3`；b：`m/s*g/cm3`。
- 必须 `a > 0`，所有派生速度有限且大于 0。
- 由 Step 6 在 Step 3 规则 MD 测井样点上拟合，不依赖地震采样域。

## 5. 公共接口

实现位于 `src/cup/physics/`。NumPy 与 PyTorch 后端提供同名、同语义 API。

### 包结构

```text
src/cup/physics/
├── __init__.py          # 导出 NumPy 后端函数 + AIVelocityRelation + rock_physics
├── numpy_backend.py     # NumPy 时间/深度内核
├── torch_backend.py     # PyTorch 同名内核，支持 autograd
├── adapters.py          # wtie.processing.grid ↔ 裸内核适配
├── calibration.py       # 冻结 AI–Vp 关系值对象
└── rock_physics.py      # 域无关等井权 Huber 拟合
```

### 公共函数

| 函数 | 输入形状 | 输出形状 | 后端 |
|------|---------|---------|------|
| `reflectivity_from_log_ai` | `[..., N]` | `[..., N-1]` | NumPy / PyTorch |
| `forward_time` | log_ai `[..., N]` + wavelet | `[..., N]` | NumPy / PyTorch |
| `build_depth_operator` | velocity `[..., N]` + depth `[N]` + wavelet | `[..., N, N-1]` | NumPy / PyTorch |
| `forward_depth` | log_ai `[..., N]` + velocity `[..., N]` + depth `[N]` + wavelet | `[..., N]`（或 + operator） | NumPy / PyTorch |
| `ai_from_velocity` | velocity `[...]` + `a, b` | `[...]` | NumPy / PyTorch |
| `velocity_from_ai` | ai `[...]` + `a, b` | `[...]` | NumPy / PyTorch |

关键契约：

- `forward_time`：显式下界面挂点，输出 N 点。反射率道不短于子波时 `[1:]` 与 Robinson 卷积逐点一致。
- `forward_depth`：默认 64 输出点分块；`return_operator=True` 时返回 `(seismic, operator)`。
- `log_ai` 与 `velocity_mps` 形状必须完全一致；`depth_m` 必须一维且长度为 N。
- 调用方必须先显式 `exp(log_ai)`，API 不猜测输入是 AI 还是 logAI。

### grid 适配器

`adapters.py` 提供 `forward_time_log` 和 `forward_depth_log`。时间域要求 TWT basis，深度域要求 TVDSS basis；basis 类型和单位在进入核心前校验。

## 6. 严格失败规则

核心与产物读取器禁止：插值/填充 NaN、猜测 CSV 字段语义、猜测单位、自动归一化/gain、自动把 MD 当 TVDSS、自动升级旧 schema、因形状"刚好可广播"接受错配输入。

## 7. 产物与 schema 契约

### `rock_physics_relation.json`

固定在 `modules/ai_vp_linear/`。记录等井权 Huber 拟合参数、a/b 系数及单位、合格井清单与拒绝原因、源 LAS 哈希。自身 SHA-256 写入 `forward_model_inputs.json`。

### `forward_model_inputs.json`

由 Step 6 生成（`ai_vp_linear` 启用且成功后），不允许人工拼装。记录 sample_domain、depth_basis、子波路径/哈希/采样参数、AI–Vp 关系引用及哈希。自身 SHA-256 作为 `forward_model_inputs_sha256` 写入下游产物。

### 模型产物

GINN v2 模型 manifest（`ginn_v2_model_run_v4`）和 checkpoint（`ginn_v2_checkpoint_v3`）记录 sample_domain、sample_unit、depth_basis、正演算子标识和 `forward_model_inputs_path`。R0 加载时校验模型域与 LFM/地震域一致。

### R0 摘要

`real_field_zero_shot_summary_v3`：包含 sample_domain、sample_unit、depth_basis、forward_model_inputs 引用、通用采样轴边界契约（米制或秒制）。深度域禁用 Hz 频谱 QC。

### R1 摘要

`real_field_forward_diagnostic_summary_v4`：包含 domain、轴契约、正演输入哈希、子波参数、相位/时间平移/深度静差独立扫描参数和闭环指标。

## 8. 兼容策略

读取到以下旧产物时必须报出 schema、实际版本、期望版本和重建入口：

- `synthoseis_lite_v1` 或无 schema 的合成数据；
- `ginn_v2_model_run_v2` 及更早模型；
- R1 v1/v2 或无 domain/正演输入哈希的摘要；
- 无明确 TWT/TVDSS 单位轴的中间产物。

禁止静默推断版本、为旧 checkpoint 注入默认 domain、为旧深度数据补虚构的正演输入哈希、保留旧 `forward_log_ai` 名称作为兼容包装。

## 9. 测试规范

测试由实现方编写，用户在本地环境运行。至少覆盖：

1. 新时间 N 点输出的 `[1:]` 与 Robinson 正演逐点一致；首点非补零。
2. NumPy/PyTorch 时间和深度结果分别一致。
3. 分块深度结果与完整 `W_depth` 一致。
4. PyTorch 对 logAI 和速度分别通过双精度 `gradcheck`。
5. 常速度模型满足解析预期。
6. 子波线性插值行为正确。
7. 非有限值、非正速度、非递增深度、错域和错单位输入均明确失败。
8. 数据集、模型、R1 对旧 schema 和错正演输入哈希明确失败。
9. R1 深度合成、观测与 TVDSS 轴逐点严格一致；相位/时间/深度扰动独立。
10. inline 步长 1、xline 步长 4 的体采样集成测试。

## 10. 验收标准

- `src/cup/physics/` 是新通用工作流唯一的正演和岩石物理实现来源；
- `forward_time` 输出 N 点；`forward_depth` 输出 TVDSS 上 N 点，分块/完整、NumPy/PyTorch 一致；
- 时间域和深度域共享同一套形状契约；
- Step 6/7 按统一真实工区 LFM 规范实施；
- xline 步长 4 不造成位置偏移或错误索引；
- 所有旧 schema 均以可操作错误要求重建；
- 核心没有 NaN 填充、自动归一化、gain 或单位猜测；
- 遗留 `wtie`、`ginn`、`ginn_depth` 未被修改或变成新核心依赖。

深度域 Jacobian/SVD 正演可观测性扩展不属于本轮验收条件。

## 11. 主要风险与控制

| 风险 | 控制 |
|---|---|
| 诊断结果污染训练 | probe 分库、独立 manifest；训练数组内容哈希不变测试 |
| 界面半样点错位 | 明确下部界面和界面 TWT 中点；解析测试 |
| PyTorch 卷积核方向错误 | 固化非对称子波 fixture |
| 深度矩阵内存过大 | 默认 64 输出点分块，仅显式请求完整算子 |
| 时间相位与深度静差混淆 | 秒制子波扰动和米制平移使用独立字段及扫描 |
| 旧产物混入新模型 | schema 和 SHA-256 严格校验 |
| 线号当数组下标 | 显式轴 + `SurveyLineGeometry` + 集成测试 |
