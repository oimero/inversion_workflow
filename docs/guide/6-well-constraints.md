# 06 井约束与分频诊断

`well_constraints.py` 是时间域工作流第二轮计划中的第六步。它不直接建低频模型，也不训练网络；它负责把第四步、第五步之后可信的井曲线转换成统一的井约束事实，并把同一套事实分别交给 GINN、低频模型和后续 enhance 使用。

这一步的核心目标是：井的 TWT、MD、XY、inline/xline、trace、sample、层段位置和频率拆分只在一个地方确定。后续第七步 LFM、第八步 GINN 和 enhance 不再各自从 LAS、TDT、轨迹文件里临时拼一套井约束。

当前实现中，这部分井空间事实和分频事实已经前移到本步骤；第七步 `lfm_precomputed.py` 只消费本步骤写出的 `lfm_layer_control_points.csv` 并做顺层插值建模。

---

## 快速开始

```bash
python scripts/well_constraints.py
python scripts/well_constraints.py --config experiments/common.yaml
python scripts/well_constraints.py --output-dir scripts/output/well_constraints_test
```

不带参数时，脚本自动发现最新的第四步 `well_auto_tie_*` 和第五步 `wavelet_generation_*` 输出，在 `scripts/output/well_constraints_<timestamp>/` 下写出结果。

---

## 为什么需要这一前置步骤

时间域第二轮的链路会同时出现三类井约束：

| 使用方 | 需要什么 | 约束含义 |
|--------|----------|----------|
| 第八步 GINN | 低频 log-AI anchor | 约束 stage-1 预测不要偏离可信井上低频趋势 |
| Enhance 训练 | 井高频监督项 | 在真实井位置直接监督模型补出的高频 log-AI residual |
| Enhance 合成器 | 高频统计材料 | 估计每层高频生成规律，而不是依赖真实 patch 数量 |

这三类材料都来自同一批井、同一套标定结果和同一个目标层框架。如果分散在多个脚本里做，最容易出现的问题是：分频 cutoff 不一致、斜井轨迹采样不一致、密井冲突处理不一致，最后训练端看似有很多约束，实际口径已经漂移。

---

## 输入与共享空间事实

本步骤应读取第四步和第五步已经稳定输出的事实：

| 来源 | 内容 |
|------|------|
| 第四步 `well_auto_tie.py` | 标定状态、路由、优化后 TDT、滤波 LAS、井旁或轨迹地震道 |
| 第四步斜井样点计划 | 斜井沿 optimized TDT 重新生成的 TWT/MD/XY/inline/xline 映射 |
| 第五步 `wavelet_generation.py` | 全局子波和每井批量合成 QC 指标 |
| 地震与层位 | 时间轴、工区几何、目标层 mask、层段和层内比例位置 |

直井的空间事实来自井口 XY、优化后 TDT 和地震几何。斜井的空间事实必须来自第四步的 optimized 轨迹样点计划，不能退化成井口直井，也不能在本步骤重新用另一套 TDT 或轨迹插值口径。

本步骤建议先生成一张内部点级事实表，基本单元是“某口井在某个 TWT 样点上的空间和曲线事实”。规范坐标为 `inline_float`、`xline_float`、`twt_s`，整数 trace/sample 索引只作为当前地震几何下的派生调试字段。

---

## 分频诊断

分频诊断决定井曲线中哪部分交给低频约束，哪部分交给 enhance 高频材料。

推荐配置支持两种模式：

```yaml
well_constraints:
  frequency_split:
    mode: diagnose
    manual_cutoff_hz: null
    filter_order: 6
    candidate_cutoff_hz: [6.0, 8.0, 10.0, 12.0, 15.0]
    buffer_seconds: null
    buffer_mode: reflect
```

`mode: diagnose` 时，脚本在候选 cutoff 上比较井上低频平滑度、高频 residual 能量比例和边界稳定性，选出全窗默认分频；第五步批量合成质量进入井筛选和权重。`mode: manual` 时，跳过诊断，直接使用用户指定 cutoff。无论哪种模式，最终 cutoff、滤波阶数、缓冲策略和诊断证据都写入 `run_summary.json`。

诊断结果不是为了让每口井各用各的 cutoff。默认口径应是全目标窗共享一个分频；分层 enhance 可以在此基础上统计每层高频规律，但不应让同一条训练链路里出现多套互相不兼容的频率拆分。

---

## 低频井监督输出

低频井监督服务第八步 GINN。它应该产出训练端能直接读取的 log-AI anchor bundle，或足够清晰的中间材料来构建 `src.ginn.anchor.LogAIAnchorBundle`。

推荐主输出：

| 文件 | 内容 |
|------|------|
| `log_ai_anchor_time.npz` | GINN 低频井监督 bundle |
| `well_anchor_points.csv` | 聚合前点级低频约束，用于审计和冲突排查 |
| `well_anchor_conflicts.csv` | 同一 trace/sample 或近邻密井冲突明细 |
| `well_anchor_trace_summary.csv` | 每条受控 trace 的井数、样点数、覆盖层段和权重统计 |

低频 anchor 的目标值来自分频后的低频 log-AI，而不是原始全频井曲线。这样它不会和第八步网络学习的高频残差互相抢职责。

第一版默认只有直井进入 `log_ai_anchor_time.npz`；斜井进入点级事实、高频监督和高频统计，但不进入 GINN 低频 anchor，除非显式打开配置。

如果多口井落到同一个 `(flat_idx, sample_index)`，必须显式记录冲突。默认策略建议是按权重加权平均并写报告；可选策略包括保留最高置信度、丢弃冲突点或遇到冲突直接失败。密井网下不能静默覆盖。

---

## 高频井监督输出

高频井监督服务 enhance 训练里的直接井约束项。它不是 GINN anchor，而是让 enhance 模型在真实井位置上补出的高频 log-AI residual 与井上的高频 residual 对齐。

推荐主输出：

| 文件 | 内容 |
|------|------|
| `well_high_supervision_time.npz` | 逐井、逐样点高频监督 bundle |
| `well_high_supervision_qc.csv` | 每口井的有效样点、层段覆盖、高频能量和权重 |
| `well_high_supervision_conflicts.csv` | 高频监督在同一 trace/sample 被聚合前的冲突审计 |
| `frequency_split_qc/*` | 每口井分频前后曲线、低频曲线、高频 residual 和包络 QC |

这份材料必须保留井名、trace 位置、样点 mask、权重和层段归属。训练端可以按 batch 抽取这些真实井位置，计算模型增强后的高频 residual，再与 `well_high_log_ai` 做 SmoothL1 或同类损失。

第一版在第六步还没有 LFM，因此 `well_high_supervision_time.npz` 中的 `lfm_log_ai`、`lfm_ai` 和 `highres_lfm_log_ai` 是零值占位；metadata 会明确标记这些字段不可当作 base AI 使用。Enhance 训练若需要 base AI，应从 GINN/LFM 主输入读取，而不是从这份井高频监督包读取。

权重建议同时考虑第五步井震匹配质量、曲线有效性、目标层覆盖、密井冲突状态和人工 include/exclude。权重为 0 的样点应保留在 QC 中，但不参与训练损失。

---

## Enhance 生成器统计输出

Enhance 合成器需要的是统计规律，不是逐点监督真值。本步骤应从同一套高频 residual 中生成每层和全窗的统计材料，让后续合成器不依赖真实 patch 数量。

推荐输出：

| 文件 | 内容 |
|------|------|
| `well_high_stats_global.json` | 全目标窗高频 residual 统计 |
| `well_high_stats_by_layer.csv` | 每层振幅、事件密度、run length、转移概率、可靠度 |
| `well_high_stats_shrinkage.json` | 每层向全窗统计收缩后的最终生成参数 |
| `well_high_motif_manifest.csv` | 可选真实 motif patch 的索引、来源层段和质量标签 |
| `well_high_motif_bank.npz` | 可选 motif patch 数值包 |

第一版写出空的 `well_high_motif_manifest.csv` 作为后续 synthetic generator 的接口占位，不生成 `well_high_motif_bank.npz`。

统计项至少应覆盖：高频 residual RMS、绝对值分位数、正负尾部分布、事件密度、正负事件态持续长度、quiet 背景持续长度、二状态/三状态转移矩阵、垂向自相关长度、反射系数量级和频谱能量。

每层统计必须带可靠度。可靠度由有效井数、有效样点数、事件数和空间覆盖共同决定。小样本层不要完全相信层内经验统计，应向全窗统计收缩：

```text
final_layer_stats = alpha * layer_empirical_stats + (1 - alpha) * global_stats
```

`alpha` 写入输出，供 enhance 合成器和 QC 解释每层参数是“强层内统计”还是“主要借用全窗先验”。

这些统计不应只写成单个固定值。合成器需要的是可采样的生成分布，因此每层输出应尽量保留 `p10/p50/p90`、分位数、转移矩阵可靠度和可用样本量。比如 event density、run length、振幅倍率和频谱倾斜都应表达成“可抽样范围”，而不是一个孤立均值。

同时，本步骤要定义后验审计口径。后续 enhance synthetic 每生成一批样本，都应能把实际生成结果与这里的目标统计对齐检查：

| 审计项 | 用途 |
|--------|------|
| 目标 event density vs 实际 event density | 判断生成器是否过碎或过平 |
| 目标 run length vs 实际 run length | 判断薄互层持续长度是否符合井统计 |
| 目标振幅分位数 vs 实际振幅分位数 | 判断 synthetic residual 是否过强或过弱 |
| 收缩前统计 vs 收缩后统计 vs 生成后统计 | 解释小样本层为何更像全窗先验 |
| motif 使用比例与拒绝原因 | 判断真实 patch 是否重新主导了样本来源 |

---

## 冲突、斜井与密井网

密井网下，“离得近”不一定是坏事。同一平台多口井可能提供同一个局部地质体的重复观测，也可能在同一地震 trace/sample 上给出互相矛盾的 AI。脚本要把这两类情况分开：

| 情况 | 推荐处理 |
|------|----------|
| 同平台、曲线相近、分频后趋势一致 | 可聚合，提高局部置信度 |
| 同 trace/sample 上低频趋势差异过大 | 写入冲突报告，按配置聚合、丢弃或失败 |
| 斜井与平台直井空间上相交 | 默认不把斜井降级为直井；若启用斜井约束，必须沿轨迹逐样点处理 |
| 斜井约束风险过高 | 可只用于 QC 和统计，不进入 GINN 低频 anchor |

第一版可以保守：直井进入 GINN 低频 anchor；斜井默认进入空间事实 QC 和高频统计，是否进入低频 anchor 由显式配置控制。这样既不丢掉斜井对分层统计的价值，也避免把冲突密集的斜井强行塞进 stage-1 监督。

---

## 与下游步骤的接口关系

第七步 LFM 从本步骤读取共享井空间事实或低频控制事实，再做顺层切片插值。当前 `lfm_precomputed.py` 已经只消费 `lfm_layer_control_points.csv`，不再自己读取 LAS、TDT 或井轨迹生成井约束。

第八步 GINN 只读取 `log_ai_anchor_time.npz` 这类 anchor bundle，不再临时读 LAS、TDT 或井轨迹。

Enhance 直接井监督读取 `well_high_supervision_time.npz`。Enhance 合成器读取 `well_high_stats_*` 和可选 motif bank。两者同源，但用途不同：一个是逐点真值，一个是生成规律。

---

## 建议落地顺序

1. 抽出共享空间事实生成能力，保证直井和斜井都来自第四步标定后的口径。
2. 实现分频诊断和每井分频 QC。
3. 产出低频 `LogAIAnchorBundle`，先只支持直井或低风险井。
4. 产出高频监督 bundle，并让 enhance trainer 消费它。
5. 产出全窗和分层高频统计，支撑新的 enhance synthetic generator。
6. 回收第七步 LFM 中重复的井空间事实生成逻辑。
