# 条件残差纹理迁移

## 1. 目标

在冻结的 GINN body-scale log-AI 上补充井中存在的高频地质纹理，形成确定性、横向连续的
高分辨率 log-AI：

```text
seismic + LFM
    -> frozen GINN
    -> body-scale log-AI
    -> conditional residual texture transfer
    -> enhanced log-AI
```

本方法的任务不是从带限地震中逐点回归唯一高频残差，而是：

> 以 GINN body 的局部地质形态为条件，在井残差字典中进行连续插值，并将所得纹理作为
> 一套确定性的条件实现铺展到工区。

地震通过 GINN body 参与主检索。冻结正演只比较增强前后的带限响应，不承担逐道选择 donor
的职责。

## 2. 尺度分工

设 `S_body` 为按物理垂向尺度定义的 Gaussian 平滑算子，当前深度域
`body_smoothing_fwhm_m = 15 m`：

```text
well_body     = S_body(filtered_full_well_log_ai)
well_residual = filtered_full_well_log_ai - well_body

enhanced_log_ai = ginn_body + predicted_residual
```

残差字典来自五口有效井的 native filtered log-AI：

```text
2-ANP-2A-RJS
L1-NW1
L5-NW5
L9-NW4A
NW11
```

字典中的 key 和 value 必须成对保存：

```text
DictionaryAtom
├── body_key                 # well_body 的局部形态
├── residual_value           # 同位置 well_residual
├── physical_axis
├── zone_id
├── normalized_zone_position
├── valid_support
├── source_well
└── source_interval
```

所有垂向窗口、伸缩和平移都以 `SampleAxis` 的物理坐标表达。当前生产域为 TVDSS/depth；
时间域使用相同 interface，由域 adapter 换算尺度。xline 步长只用于几何寻址，横向距离始终
使用米制坐标。

## 3. 检索键

检索主键只描述 GINN body 的局部形态，不直接使用原始地震绝对振幅：

```text
BodyKey
├── normalized local body profile
├── first physical derivative
├── second physical derivative
├── local body contrast
├── dominant body thickness
├── zone_id
└── normalized zone position
```

profile 在窗口内去除均值并按稳定尺度归一化；局部均值、对比度和物理厚度另作显式特征，
避免形态归一化丢失地质尺度。

字典和工区 query 使用完全相同的 `BodyKeyEncoder`。窗口中心按物理距离布置，相邻窗口重叠。
第一版不训练新的 key encoder，不把网络 hidden feature 当作不可解释的检索键。

## 4. 连续字典回归

### 4.1 软权重

对每个 query key `q` 和全部字典 key `k_i` 计算连续距离：

```text
d_i(q) = weighted_distance(q, k_i)
w_i(q) = exp(-d_i(q)^2 / temperature^2)
         / sum_j exp(-d_j(q)^2 / temperature^2)
```

预测残差窗口为：

```text
r_local(q) = sum_i w_i(q) * T_i(residual_value_i)
```

`T_i` 只包含有界的物理垂向平移、伸缩和正比例振幅变化。离线为每个 atom 建立同一套固定
变换 bank，并把 `(atom, transform)` 视为公共字典成员；query 对全部成员计算软权重。不能先
用离散相关峰或 `argmin` 选择平移、伸缩后再混合 residual，否则跳变会从 donor identity
转移到变换参数。

### 4.2 连续性硬合同

横向连续性首先由映射本身保证：

- 使用全字典 softmax 或固定的公共原型字典；
- 不使用逐道最近邻 `argmin`；
- 不使用随 query 改变成员集合的硬 `top-k`；
- 不随机抽取 donor；
- 同一个 query 始终产生同一组权重和同一 residual；
- 相近 key 的权重差异必须连续。

对任意两个有效 query `q1/q2`，实现需要报告：

```text
key_distance(q1, q2)
weight_distance(W(q1), W(q2))
residual_distance(r(q1), r(q2))
```

这些距离随 key 扰动平滑变化。temperature 在 YAML 中设置最小值，避免 softmax 退化成数值
上的 hard nearest。

字典较大时，先离线构建固定公共原型。所有 query 始终对同一组原型求权重，不能逐道运行
近邻搜索后只保留不同的候选集合。

## 5. 空间联合权重场

逐窗口的初始权重 `w0(x, z)` 只是 unary evidence。正式输出使用整个 section 上联合求得的
权重场：

```text
W* = argmin_W
       key_fit(W, W0)
     + lambda_lateral * lateral_metric_smoothness(W)
     + lambda_vertical * vertical_overlap_consistency(W)
```

约束为：

```text
W_i >= 0
sum_i W_i = 1
```

横向项根据真实米制间距加权。它作用于字典权重和变换参数，不直接平滑最终 log-AI。这样
相邻 query 即使位于两个原型的距离分界附近，也会平滑地改变混合比例，不会突然切换 donor。

空间平滑强度允许由 body key 梯度调制：

- body 连续且 zone identity 相同：使用正常横向耦合；
- body 出现明确横向界面、pinchout 或 zone 支持终止：降低或断开耦合；
- 不跨无效支持和不同 zone 传播纹理。

第一版分别沿 inline 和 xline 求权重场。两个方向在权重层等权融合后只生成一次 residual，
不能分别生成两套高频结果再平均。

## 6. 重叠相加生成

每个中心位置生成一个 residual window。完整道或剖面的 residual 使用固定物理窗口和
partition-of-unity 权重重叠相加：

```text
predicted_residual(z)
    = sum_c window_weight_c(z) * r_local_c(z)
      / sum_c window_weight_c(z)
```

窗口权重在轴上连续，重叠区权重和为一。该步骤避免垂向 patch 接缝。目的层外 residual
严格为零；目的层 mask 不派生 halo、侵蚀或过渡带。

最终 residual 再经过一次高频投影：

```text
predicted_residual
    <- predicted_residual - S_body(predicted_residual)
```

该投影只维持 body/residual 的尺度分账，不修正 GINN body。

## 7. 振幅合同

纹理振幅不能成为自由拟合参数。每个 atom 保留井上原始 residual 振幅，允许的缩放由 YAML
中的有限范围控制，并由 query/atom 的 body contrast ratio 连续确定。

正式输出至少报告：

- residual RMS 与五口井 residual RMS 分布的关系；
- residual 绝对振幅分位数；
- 不同 source well 的权重占比；
- 有效字典数 `1 / sum_i w_i^2`；
- 增强前后冻结正演的相关性与波形差异。

冻结正演诊断不能通过为每道自由拟合 residual scale 来改善。

## 8. 深模块 interface

条件残差迁移封装为一个深模块，调用方只接触：

```python
library = build_residual_library(well_controls, scale_contract)
result = transfer_residual_texture(ginn_body, geometry, library, policy)
```

`transfer_residual_texture()` 内部完成：

- body key 编码；
- 连续字典权重；
- 变换参数；
- section 权重场求解；
- inline/xline 权重融合；
- residual window 生成与重叠相加；
- 高频投影；
- summary 与诊断字段。

结果 interface：

```text
ResidualTransferResult
├── ginn_body
├── predicted_residual
├── enhanced_log_ai
├── dictionary_weight_summary
├── effective_dictionary_count
├── transform_summary
├── lateral_continuity_metrics
└── support
```

实现放在 `src/enhance`。第一版是解析的字典回归与空间权重求解，不训练 residual 回归网络。

## 9. 第一轮原型

第一轮只运行固定小剖面，不直接处理全体积：

1. 从五口井建立 body/residual 成对字典；
2. 生成每口井的 body、residual 和 atom atlas；
3. 对一条 inline 和一条 xline GINN body 剖面计算初始权重；
4. 求联合连续权重场；
5. 生成 residual 与 enhanced log-AI；
6. 输出冻结正演对比。

对照只需要三组：

```text
hard nearest atom                # 展示逐道跳变反例
soft dictionary only             # 检查 key-space 连续映射
soft dictionary + spatial field  # 正式方案
```

图件必须使用相同色标并包含：

- GINN body；
- hard-nearest residual；
- soft residual；
- spatially coupled residual；
- enhanced log-AI；
- source-well dominant weight；
- effective dictionary count；
- 横向放大窗口。

## 10. 第一轮判断标准

第一轮由图件和少量直接指标共同判断：

- 两个相近 body key 的 residual 不发生离散跳变；
- soft 方案相对 hard nearest 明显减少横向接缝；
- 空间耦合后 residual 仍保留垂向纹理，不退化成横向模糊带；
- 相邻道字典权重连续，明确 body 界面处允许变化；
- 对 query key 施加小扰动时，权重、变换和输出 residual 均连续变化；
- 不出现整段被单口井纹理统治或规则周期复制；
- residual 振幅落在井残差可见范围；
- 增强结果保持 GINN body 的主体结构和主要正演波瓣。

若 soft dictionary 本身仍产生明显跳变，先调整 key 标准化、公共原型和 temperature。若
soft dictionary 连续而 section 仍有接缝，调整空间权重场和重叠窗口。第一轮不新增神经网络、
随机 realization 或全体积训练。

## 11. 解释边界

交付结果解释为：

> 在 GINN body 条件下，由五口井高频纹理字典和空间连续性约束生成的一套确定性地质实现。

具体微层不是地震唯一分辨的真值。GINN body 提供主体尺度约束；井残差提供高频纹理；连续
字典权重和空间联合求解决定纹理如何在工区铺展。
