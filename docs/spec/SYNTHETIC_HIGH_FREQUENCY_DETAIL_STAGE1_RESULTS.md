# Synthetic 高频 detail 阶段 1 结果

> 运行日期：2026-07-11
> 域：深度域，TVDSS
> Calibration：`scripts/output/synthoseis_lite_calibrate_20260711_051812`
> 最终诊断：`scripts/output/synthoseis_detail_diagnostic_20260711_stage1_final`
> 结论：阶段 1 已完成；阶段 2 仍被尺度分解支撑问题阻塞。

## 1. 输入与扫描

正式 calibration 成功提取 411 个父对象。诊断使用冻结的 201 ms 时间子波和冻结 AI–Vp 关系，主周期为 50.5 ms。

扫描组合：

- 分辨率比例：0.75、1.00、1.25；
- 滞回进入阈值：1.5σ、2.0σ、2.5σ；
- 滞回退出阈值：0.75σ；
- 最小事件支撑对照：0.25、0.10、0.02、0.001 倍局部分辨率。

其中 71 个对象在尺度分解后为精确零 detail，作为有效零激活证据记录；没有非零但稳健尺度退化的对象。

## 2. 四项阻塞问题的结果

### 2.1 Post-H event 模板

留一井最近邻模板验证包含 981 个 event，归一化波形 RMSE 均值为 0.0227。post-H 波形模板在当前数据上具有可复用性，生成阶段不需要再次应用 H。

该结果只支持“模板表达可行”，不等于已经冻结模板池的聚类、抽样权重或外推规则。

### 2.2 Duration composition

Logistic-normal composition 在全部有效对象和横向探针上的最大长度闭合误差为 `1.42e-14`，可以严格保证：

```text
duration_k(x) > 0
sum_k duration_k(x) = parent_duration(x)
```

该构造可以进入后续设计。事件提取的最小支撑规则尚不能冻结，原因见第 3 节。

### 2.3 深度域 paired forward

self-consistent velocity 与 fixed reference velocity 的差值高度一致：

- paired-delta correlation 中位数：0.99993；
- correlation 第 10 百分位：0.99869；
- 两种可观测性比值绝对差中位数：0.00035；
- 绝对差第 90 百分位：0.01595。

当前工区中运动学污染整体较小。后续 benchmark 以 self-consistent velocity 作为物理自洽主产物，同时保留 fixed-reference velocity 作为诊断，不把两者混成同一字段。

### 2.4 Latent topology 与最终 active mask

latent boundary 与最终数值 active boundary 的平均不一致比例为 45.6%。因此 latent event boundary 不能作为最终波瓣边界或界面监督标签。

后续只允许：

- latent topology 作为生成过程元数据；
- final active mask 从最终 detail 数值独立计算；
- 需要最终波形 event 时重新分段，不能复用 latent boundary。

## 3. 新发现的阻塞问题

父对象物理长度与局部分辨率的比例为：

- 第 10 百分位：0.0205；
- 中位数：0.0441；
- 第 90 百分位：0.1380。

绝大多数父对象远短于尺度算子窗口。在单个父对象内部应用 L 时，L 近似退化为对象均值，导致 0.75–1.25 的分辨率比例对 event 统计完全无影响。

以局部分辨率的 0.02–0.25 倍作为最小 event 支撑还会把 latent active state 合并为 silent。使用近似“不合并”的 0.001 对照后，进入阈值才呈现预期敏感性：

| 进入阈值 | 有 active event 的对象比例 | 平均 active event 数 |
|---:|---:|---:|
| 1.5σ | 49.4% | 1.226 |
| 2.0σ | 32.6% | 0.667 |
| 2.5σ | 20.9% | 0.324 |

因此不能在父对象内部独立执行物理尺度分解，也不能把最小 event 支撑直接定义为局部分辨率比例。

阶段 2 前必须原型验证以下修正：先在连续 well-zone 轴上拼接对象 `fit_residual` 并执行一次 L/H，再将 post-H 结果映射回父对象；最小 event 支撑改为显式 truth-cell/测井采样支撑，并单独做噪声敏感性验证。

## 4. 冻结与未冻结决策

可以冻结：

- post-H 波形事件口径，生成阶段不二次高通；
- logistic-normal duration composition；
- self-consistent velocity 主正演与 fixed-reference 诊断；
- latent topology 不作为最终边界标签；
- 71 个零 detail 对象属于真实零激活证据，不进行统计回退。

暂不冻结：

- L/H 的对象支撑与边界处理；
- resolution ratio；
- 最小 event 支撑；
- 滞回进入/退出阈值；
- 模板聚类和抽样权重。

## 5. 阶段门禁

阶段 1 的代码、正式数据运行和结果审阅已完成。阶段 2 仍禁止开始，直到连续 well-zone 尺度分解原型证明：

- resolution ratio 对 detail 统计具有可解释敏感性；
- event 统计不由对象边界或滤波窗口退化主导；
- 留出井模板误差、频谱纯度和零激活证据保持稳定。
