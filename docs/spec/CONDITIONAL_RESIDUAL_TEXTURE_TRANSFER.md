# 条件残差纹理迁移

## 1. 目标

在冻结的 GINN body-scale log-AI 上补充井中存在的亚主体尺度地质纹理，形成确定性、横向连续的
高分辨率 log-AI：

```text
seismic + LFM
    -> frozen GINN
    -> body-scale log-AI
    -> conditional residual texture transfer
    -> enhanced log-AI
```

本方法的任务不是从带限地震中逐点回归唯一亚主体尺度残差，而是：

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

这里的 residual 是 `15 m` body 以下、上游 filtered log-AI 有效频带以内的亚主体尺度纹理，
不是未经滤波的测井全频高频。字典的振幅和锐度均以 native filtered log-AI 为统计口径。

`S_body` 在每个 native 有效 finite run 内独立计算；卷积核只在 run 内归一化，不跨缺口或
目的层边界补零。随后使用与 GINN 井目标相同的 finite-run 物理平滑与轴插值合同，把
`well_body` 和 `well_residual` 映射到目标模型轴。key 和 value 共享完全相同的支持与重采样
位置。

zone 顶底来自解释层位在每口井轨迹上的交点；同名 zone 在不同井上拥有各自的 TVDSS
区间。工区 query 使用逐道层位网格生成逐样点 zone identity，不把任意一口井或全区包络
广播为统一垂向区间。

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

`zone_id` 只在同一 zone 的字典分区内匹配。检索距离先把六类特征各自压缩为一个与窗口
样点数无关的组距离：

```text
delta_profile   = RMS(profile_q - profile_i) / scale_profile
delta_d1        = RMS(d1_q - d1_i)           / scale_d1
delta_d2        = RMS(d2_q - d2_i)           / scale_d2
delta_contrast  = abs(contrast_q - contrast_i) / scale_contrast
delta_thickness = abs(thickness_q - thickness_i) / scale_thickness
delta_position  = abs(position_q - position_i) / scale_position

d_i(q)^2 = (
    delta_profile^2
  + delta_d1^2
  + delta_d2^2
  + delta_contrast^2
  + delta_thickness^2
  + delta_position^2
) / 6
```

profile 与导数先在公共归一化垂向坐标上采样。每个 `scale_*` 是全字典该组非零 pairwise
距离的中位数，并随字典产物发布；六个尺度都必须有限且大于零。六个特征组第一轮等权；
不能把向量逐点展开后与标量直接拼接，也不在 query 阶段重新估计尺度或权重。

temperature 由字典本身确定。对每个 atom 计算同 zone、排除自身后的第三近邻距离
`d_i,3`：

```text
temperature_base(zone) = median_{i in zone}(d_i,3)
temperature(zone)      = temperature_multiplier * temperature_base(zone)
```

每个 zone 必须至少有四个有效 atom，并且来自至少两口井；`temperature_base(zone)` 必须有限
且大于零。字典构建时发布每个 zone 的 atom 数、来源井数和 temperature；不满足合同则列出
对应 zone 并停止建库。第一轮固定输出 `temperature_multiplier = 0.75 / 1.0 / 1.5` 三组
结果，正式默认值为 `1.0`。同一 zone 的所有 query 使用同一个 temperature，不逐道拟合
bandwidth。

`T_i` 由 query body 与 atom body 的连续形态统计解析得到。定义：

```text
contrast(b) = RMS(b - mean(b))

mu(b) = sum_z z * abs(db/dz) / sum_z abs(db/dz)

thickness(b)
    = 2 * sqrt(
        sum_z abs(db/dz) * (z - mu(b))^2
        / sum_z abs(db/dz)
      )

shift_i   = mu(query_body) - mu(atom_body_i)
stretch_i = clip(
              thickness(query_body) / thickness(atom_body_i),
              0.75,
              1.33
            )
amplitude_i = clip(
                sqrt(contrast(query_body) / contrast(atom_body_i)),
                0.75,
                1.33
              )
```

所有分母使用由字典尺度确定的固定正数下限。`T_i` 用单调线性插值将 atom residual 对齐到
query 物理轴，再乘 `amplitude_i`。shift、stretch 和 amplitude 均由连续统计直接计算，不通过
相关峰、离散 transform bank 或逐道优化选择。

变换后的 donor 只在其真实 finite support 内参与混合。某个 donor 在 query 边缘或缺口处
无支持时，该样点对剩余 donor 权重重新归一化；无支持不能按数值零参与平均并压低振幅。

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

这些距离随 key 扰动平滑变化。实现同时报告 temperature multiplier、逐 zone atom/来源井
覆盖率和有效字典数，三档 temperature 使用同一字典、同一 query 和同一后续空间参数。

字典较大时，先离线构建固定公共原型。所有 query 始终对同一组原型求权重，不能逐道运行
近邻搜索后只保留不同的候选集合。

## 5. 空间联合权重场

逐窗口的初始权重 `w0(x, z)` 只是 unary evidence。正式输出在真实米制几何上建立统一邻接图
并联合求得权重场：

```text
W* = argmin_W
       sum_u ||W_u - W0_u||^2
     + lambda_lateral * sum_(u,v) g_uv ||W_u - W_v||^2
     + lambda_vertical * vertical_overlap_consistency(W)
```

约束为：

```text
W_i >= 0
sum_i W_i = 1
```

第一项中的 `W0` 在全部迭代中保持为固定 unary。邻接更新不得用上一轮 `W` 替换 `W0`，
否则多轮迭代会把条件检索偏好扩散成与 query 无关的均匀权重场。

其中图边权重为：

```text
g_uv
    = exp(-distance_m(u, v) / lateral_correlation_length_m)
    * exp(-key_distance(q_u, q_v)^2 / (2 * key_edge_scale^2))
```

不同 zone、无效支持和明确 pinchout 两侧不建立图边。横向项作用于字典权重，不直接平滑
最终 log-AI。变换参数由平滑后的字典权重对各 atom 解析参数求加权结果。这样
相邻 query 即使位于两个原型的距离分界附近，也会平滑地改变混合比例，不会突然切换 donor。

`vertical_overlap_consistency(W)` 对同一道上相邻垂向窗口的公共物理区间计算：先用各自权重
生成变换后的 residual window，再惩罚两者在重叠样点上的均方差。该项约束重叠区的预测
纹理一致，不直接惩罚两个窗口的权重向量必须相同。

空间平滑强度允许由 body key 梯度调制：

- body 连续且 zone identity 相同：使用正常横向耦合；
- body 出现明确横向界面、pinchout 或 zone 支持终止：降低或断开耦合；
- 不跨无效支持和不同 zone 传播纹理。

固定剖面使用相应米制一维图；全体积使用 inline/xline 最近合法邻道构成的稀疏二维物理
邻接图，只求一次权重场和一次 residual。线号步长只影响 trace 寻址；`distance_m` 来自
实际坐标，因此两个方向的不同采样密度自然进入图边权重。

## 6. 重叠相加生成

每个中心位置生成一个 residual window。完整道或剖面的 residual 使用固定物理窗口和
partition-of-unity 权重重叠相加：

```text
predicted_residual(z)
    = sum_c window_weight_c(z) * r_local_c(z)
      / sum_c window_weight_c(z)
```

每个窗的物理支撑与 `BodyKey` 窗一致。设半窗宽为 `H`，中心间距为 `H`，使用紧支撑的
raised-cosine 权重：

```text
window_weight_c(z)
    = 0.5 * (1 + cos(pi * abs(z - z_c) / H)),  abs(z - z_c) <= H
    = 0,                                       otherwise
```

每个 finite run 的首尾各放置一个边界中心，最后始终除以实际权重和。窗口权重在轴上连续，
重叠区构成 partition of unity，并保证 finite-run 边界也有非零支撑。该步骤避免垂向 patch
接缝。目的层外 residual 严格为零；目的层 mask 不派生 halo、侵蚀或过渡带。

最终 residual 再经过一次亚主体尺度投影：

```text
predicted_residual
    <- predicted_residual - S_body(predicted_residual)
```

`S_body` 在每个有效 finite run 内归一化计算，目的层外和缺口处不参与卷积。投影后再将
目的层外设为零。该投影只维持 body/residual 的尺度分账，不修正 GINN body，也不会因硬零
填充在目的层边缘制造高频环。

## 7. 振幅合同

每个 atom 保留井上原始 residual 振幅。振幅缩放固定采用 4.1 节的 contrast square-root ratio
和 `[0.75, 1.33]` 截断，不设置逐道拟合增益。

正式输出至少报告：

- residual RMS 与五口井 residual RMS 分布的关系；
- residual 绝对振幅分位数；
- 垂向一阶导数 RMS 相对加权 donor 的比值；
- 垂向自相关半宽相对加权 donor 的比值；
- 不同 source well 的权重占比；
- 有效字典数 `1 / sum_i w_i^2`；
- 初始权重场相对逐 zone 均匀权重的空间方差；
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
- 米制空间邻接图与权重场求解；
- residual window 生成与重叠相加；
- 亚主体尺度投影；
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

实现使用独立源码包 `src/enhance_v2`。该模块完整拥有残差字典、检索键、纹理迁移和产物
合同；以第六步 native filtered well controls、冻结的 GINN body 和物理几何为输入，发布
`ResidualTextureLibrary`。

建议初始结构控制为：

```text
src/enhance_v2/
├── __init__.py
├── contracts.py       # library、policy 和 result interface
├── library.py         # finite-run residual library
├── keys.py            # BodyKey 与连续 transform
├── transfer.py        # 软字典回归、空间权重场、重叠相加
├── artifacts.py       # 正式产物和图件
└── runtime.py         # 设备、日志和批处理
```

包根只暴露 `build_residual_library()` 和 `transfer_residual_texture()`。第一版是解析的字典回归
与空间权重求解，不训练 residual 回归网络。

## 9. 第一轮原型

第一轮只运行固定小剖面，不直接处理全体积：

1. 从五口井建立 body/residual 成对字典；
2. 生成每口井的 body、residual 和 atom atlas；
3. 对一条 inline 和一条 xline GINN body 剖面计算三档 temperature 的初始权重；
4. 求联合连续权重场；
5. 生成 residual 与 enhanced log-AI；
6. 输出冻结正演对比。

对照只需要四组：

```text
hard nearest atom                # 展示逐道跳变反例
uniform weights by zone          # 展示未使用 BodyKey 的平均纹理基线
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
- 初始权重相对逐 zone 均匀权重的空间差异；
- 三档 temperature 下的垂向锐度与权重分布；
- 横向放大窗口。

## 10. 第一轮判断标准

第一轮由图件和少量直接指标共同判断：

- 两个相近 body key 的 residual 不发生离散跳变；
- soft 方案相对 hard nearest 明显减少横向接缝；
- 空间耦合后 residual 仍保留垂向纹理，不退化成横向模糊带；
- 记录 predicted residual 相对加权 donor 的一阶导数 RMS 比值，首轮以 `0.70` 为参考线；
- 记录 predicted residual 相对加权 donor 的自相关半宽比值，首轮以 `1.50` 为参考线；
- 相邻道字典权重连续，明确 body 界面处允许变化；
- 对 query key 施加小扰动时，权重、变换和输出 residual 均连续变化；
- soft dictionary 相对逐 zone 均匀权重呈现与 BodyKey 对应的空间组织，而不是全区固定混合；
- 不出现整段被单口井纹理统治或规则周期复制；
- residual 振幅落在井残差可见范围；
- 增强结果保持 GINN body 的主体结构和主要正演波瓣。

若 soft dictionary 本身仍产生明显跳变，先检查 key 标准化、公共原型和三档 temperature。
若输出相对 donor 明显变糊，优先比较较低 temperature，并检查对应 zone 的 atom 覆盖和
解析变换对齐；第一轮不增加事后锐化。若 soft dictionary 连续而 section 仍有接缝，调整空间
权重场和重叠窗口。第一轮不新增神经网络、随机 realization 或全体积训练。

## 11. 解释边界

交付结果解释为：

> 在 GINN body 条件下，由五口井亚主体尺度纹理字典和空间连续性约束生成的一套确定性地质实现。

具体微层不是地震唯一分辨的真值。GINN body 提供主体尺度约束；井残差提供亚主体尺度
纹理；连续字典权重和空间联合求解决定纹理如何在工区铺展。

每个 atom 在一个窗口内只使用一组全局 shift、stretch 和 amplitude。窗口包含多个相互独立
界面时，解析变换不能分别对齐每个界面；这类局部错位属于第一版的已知表达限制。
