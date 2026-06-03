# Enhance 合成器重构规划

本文记录时间域第二轮 enhance 合成方法的重构方向。它不是运行教程，而是给后续代码实现用的设计文档。

目标是把当前偏“真实井 patch 驱动”的合成方式，重构成“全窗-分层层级统计生成器”。真实井 patch 仍然有用，但它应从样本主体降级为统计校准和可选 motif bank。这样分层 enhance 不会因为某一层可截取的真实 patch 少而变得单调或过拟合。

---

## 当前原型的问题

深度域原型已经具备一个可工作的 enhance synthetic dataset：它读取 stage-1 base AI、井高频 prior、子波和目标 mask，生成 `target_delta_log_ai`、`target_ai` 和 `target_seismic`，再训练模型把 base 输入增强到目标高频。

当前思路的主要风险不在“不能合成”，而在分层化之后：

| 问题 | 影响 |
|------|------|
| 真实 patch 被当作主要样本来源 | 小样本层可用 patch 少，样本多样性受限 |
| 高频统计和逐点监督混在同一个 prior 中 | 训练监督、生成统计、QC 的职责不清 |
| Markov packet 参数主要来自全局经验值 | 分层后难以表达不同层的节律和振幅差异 |
| gallery/QC 只是消费 synthetic dataset | 如果合成器逻辑不清，图只能暴露问题，不能解释问题来源 |

重构后，合成器不再自己读井、拆频或处理井轨迹冲突。这些材料来自第六步 `well_constraints.py`。

---

## 目标生成器思想

推荐目标是：

```text
井高频 residual
  -> 第六步井约束统计
  -> 全窗统计 + 每层统计
  -> 每层向全窗统计收缩
  -> semi-Markov / run-length 高频 residual 生成
  -> base AI + delta log-AI
  -> 子波正演生成 target seismic
  -> 分层 enhance 训练与层间融合
```

这套设计借鉴的是程序化地质生成的思想：多样性主要来自统计参数空间和随机生成过程，而不是来自“能从真实井里截出多少段窗口”。

真实 patch 的新定位是：

| 用途 | 说明 |
|------|------|
| 统计校准 | 估计振幅、事件密度、run length、转移矩阵和频谱 |
| motif bank | 少量高质量真实片段可作为局部包络、极性和形态参考 |
| QC 对照 | 合成 residual 的统计分布要与真实井 residual 可比 |

---

## 从井约束读取什么

合成器消费第六步的高频统计材料，不消费原始 LAS、TDT 或井轨迹文件。

必须读取：

| 材料 | 用途 |
|------|------|
| `well_high_stats_global.json` | 全目标窗默认生成参数 |
| `well_high_stats_by_layer.csv` | 每层经验统计和可靠度 |
| `well_high_stats_shrinkage.json` | 每层最终生成参数 |
| `well_high_supervision_time.npz` | 只供训练监督项使用，不作为 synthetic 主体 |

可选读取：

| 材料 | 用途 |
|------|------|
| `well_high_motif_manifest.csv` | 选择可用 motif patch |
| `well_high_motif_bank.npz` | 为部分样本提供真实局部形态参考 |

注意：`well_high_supervision_time.npz` 和统计材料同源，但消费方式不同。监督项关心真实井位置上的逐点高频真值；合成器关心一层内应该生成什么样的高频扰动分布。

---

## 统计生成器设计

每条 synthetic trace 先从 stage-1 base dataset 中抽取一条真实位置的 base AI、地震输入、mask、层段信息和 taper。随后在目标层内生成高频 `delta_log_ai`。

推荐第一版生成流程：

1. 按 trace 的目标层 mask 找到每个层段的有效样点。
2. 对每个层段读取收缩后的 layer stats。
3. 采样该层的事件密度、正负 run length、振幅分位数和转移矩阵。
4. 用 semi-Markov 过程生成正负状态序列；状态持续长度来自 run-length 分布，而不是逐点独立转移。
5. 给每个状态或事件采样振幅，并按层内包络、taper 和 AI 物理边界约束调整。
6. 可选混入 motif bank：只让 motif 影响局部包络、极性或振幅，不让它完全替代统计生成主体。
7. 将各层生成的 `delta_log_ai` 拼接到全目标窗，并在层间边界做平滑过渡。
8. 用 `target_ai = base_ai * exp(delta_log_ai)` 得到目标 AI，再用当前全局子波正演生成 `target_seismic`。

关键配置建议：

```yaml
enhance_synthetic:
  generator_family: hierarchical_statistical
  layer_stats_source: well_constraints
  patch_guided_fraction: 0.2
  min_layer_reliability_for_patch: 0.5
  shrinkage_enabled: true
  semi_markov_enabled: true
  layer_boundary_blend_samples: null
```

`patch_guided_fraction` 不应成为主要多样性来源。小样本层或低可靠度层应降低 patch 使用比例，更多借用全窗统计。

---

## 分层 Enhance 与融合

时间域第二轮的大架构是全目标窗 GINN、分层 enhance、层际平滑融合。

合成器应支持两种训练组织方式：

| 模式 | 用途 |
|------|------|
| 全窗预训练 | 用全目标窗统计训练一个通用 enhance 网络 |
| 分层 fine-tune | 对每个层段用对应 layer stats 微调，学习层内高频风格 |

推荐默认是“全窗预训练 + 分层 fine-tune”。这样可以保留全局稳定性，又能让不同层的振幅、节律和薄互层尺度有所差异。

层间融合不应由 synthetic generator 临时决定。合成器只需要在训练样本里提供层边界附近的 taper/blend mask 和 QC 指标；最终 inference 阶段的层间融合应由 enhance 反演或融合脚本统一处理。

---

## QC、Gallery 与训练接口

需要明确四个职责：

| 组件 | 职责 |
|------|------|
| Synthetic generator | 生成训练样本和样本统计 |
| Enhance gallery | 抽样可视化 synthetic trace，不改变生成逻辑 |
| Enhance QC | 汇总真实/合成分布、质量门控、推荐动作 |
| Enhance trainer | 消费 synthetic loss 和井高频监督项 |

Gallery 图建议至少包含：base AI、target AI、delta log-AI、reflectivity、input seismic、target seismic、层段 mask、taper 和质量门控结果。

QC 指标建议覆盖：

| 指标 | 用途 |
|------|------|
| synthetic-to-real RMS / p95 / p99 | 检查振幅是否过强或过弱 |
| event density by layer | 检查层内节律是否符合井统计 |
| run-length distribution | 检查 Markov 生成是否过碎或过平 |
| base-target waveform corr | 检查目标地震是否离 base 过远 |
| target-observed waveform corr | 防止 synthetic 与真实地震统计完全脱节 |
| quality gate pass fraction | 评估样本拒绝率和重采样压力 |

训练端仍应同时保留 synthetic loss 和井高频监督项。synthetic loss 提供大量可控样本，井高频监督项把模型拉回真实井位置。

---

## 推荐迁移顺序

1. 将现有 synthetic dataset 中的 patch 抽样、Markov packet、正演和质量门控拆成可测试的小模块。
2. 让合成器只从第六步井约束输出读取高频统计，不再直接读取井文件。
3. 先实现全窗统计生成，确认 synthetic QC 与现有原型相当。
4. 加入分层统计和 shrinkage，支持按层生成不同高频风格。
5. 将真实 patch 改成可选 motif bank，并限制其使用比例。
6. 接入分层 fine-tune 和层间融合所需的 mask/QC 输出。
