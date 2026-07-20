# GINN 分阶段 MVP Handoff

## Summary

新增 `docs/spec/GINN_PHASED_MVP_HANDOFF.md`，作为当前唯一活跃实施路线：

- 阶段 0：在最新 Synthoseis-lite benchmark 上建立连续 GINN 基线，比较 clean、mismatch、真实井监督和横向上下文。
- 阶段 1：实现单 zone、纵向 trace-wise Structured MVP，验证地震是否真正改善高分辨率结构推断。
- `STRUCTURED_GINN_HANDOFF.md` 标记为延期的完整设计，`ENHANCE_V2_HANDOFF.md` 保持冻结替代路线；二者均链接到新 Handoff。

公共文档只使用角色和路径占位符，不记录真实井名、工区名或日期化产物路径。实际实验 YAML 必须冻结解析后的真实来源。

## 阶段 0：连续模型基线

### 前置工作

- 使用最新第五、六步结果重跑真实工区 LFM。
- R0 固定使用比例切片 LFM、当前振幅匹配地震变换和已有固定剖面清单。
- benchmark 不使用自动发现结果；实验配置记录明确目录、manifest 身份和 fingerprint。
- 扩展训练脚本，使其接受可选配置路径，并允许微调实验从外部不可变 checkpoint 初始化；记录 checkpoint 路径、哈希和来源实验，避免重复训练合成基线。

### 固定实验矩阵

共训练六个结果，使用相同 seed、父实现划分、训练集 normalization、patch 和训练预算：

1. `trace_dilated_tcn_clean`
   - `trace_dilated_tcn`，hidden 32，depth 5。
   - 只用 base seismic 训练。
   - 仍在完整 mismatch validation 分布上评估。

2. `trace_dilated_tcn_mismatch`
   - 同一网络。
   - base/variant 父样本权重各 0.5。
   - 当前 13 个 amplitude、noise、phase 和 registration views 等权采样。

3. `trace_dilated_tcn_mismatch_realwell_w001`
4. `trace_dilated_tcn_mismatch_realwell_w005`
5. `trace_dilated_tcn_mismatch_realwell_w010`
   - 三者共享第 2 个模型的最佳 checkpoint。
   - 真实井损失权重分别为 0.01、0.05、0.10。
   - 合成 anchor 每步保留，真实井监督不替代合成训练。
   - 微调 3 epochs、每 epoch 100 steps、学习率 `1e-5`。
   - 从当前第五步跳过时移扫描名单解析三口监督排除井。
   - 留出井按“非子波来源、非排除井、可靠候选中有效评分样本最多”规则解析一次并冻结。
   - 同簇井按 500 m 半径退出监督。
   - checkpoint 仍按合成 mismatch weighted MSE 选择；留出井 MSE 只作诊断。

6. `trace_lateral_mixer_mismatch`
   - hidden 32、depth 5、lateral kernel 3。
   - 使用与第 2 个模型完全相同的训练和验证分布。
   - 用于检验真实 25 m 邻道上下文是否优于纯纵向模型。

不纳入首轮的模型包括 `patch_conv2d`、physics loss、真实井与 lateral mixer 的组合，以及真实井单独训练。

### 统一评估

- 使用既有固定 parent hash split；geometry holdout 始终属于 test。
- 每个模型均输出 clean、逐 view、加权总体和按 scenario 分层指标。
- 报告三个单因素差值：
  - mismatch − clean；
  - 三个 real-well 权重 − mismatch；
  - lateral mixer − mismatch。
- 所有六个模型均运行固定真实剖面 R0；不运行全体积和 R1。
- 真实井扫参不自动选“赢家”，统一展示合成保持程度、留出井诊断及 R0 差异，由人工决定后续使用哪个权重。
- 科学指标差、某个 view 表现下降或真实井无改善只产生报告和 warning。

## 阶段 1：Structured MVP

### 数据合同

为新生成 benchmark 增加精简的 `structured_latent_supervision_v1`：

- 复用已有 high-resolution log AI、RGT、state、object ID、object ξ、zone、boundary 和三类 mask。
- producer 直接发布每个 realization/zone 的背景 `a/b`，禁止从 AI 反推。
- 每个 realization 保存紧凑对象参数：
  - 对象 ID、state 和 zone ordinal；
  - 每个 lateral trace、每个对象的最终有效 `c0/c1/c2`；
  - 对象顶底索引或坐标及有效 mask。
- 参数保存到 realization 自身的 HDF5 group，不在训练时扫描大型全局 CSV。
- 发布 active zone 使用的 initial、transition、duration、projection 和 forward 合同身份。
- 首轮 active zone 固定为 ordinal 1；只运行深度域数据，但 schema、decoder、HSMM 和 projection 接口保持时深域共享。

### 模型

代码放入 `src/ginn_v2/structured`，与连续 GINN checkpoint 和 prediction schema 隔离：

- 输入严格为两通道：seismic 和 LFM；zone、RGT、horizon、valid mask 不作为观测通道。
- 纵向 dilated TCN 在 model grid 编码，再按 artifact 中的 oversampling factor 映射到 high-resolution grid。
- 输出 high-resolution 三状态 emission。
- 使用 calibration 固定的 transition 和 duration 做 semi-Markov Viterbi；state、segment、boundary 和 duration 均由同一分段产生。
- 训练时对完整真实状态序列做 state CE，并使用真实 segment teacher forcing：
  - active zone pooling 输出 `a/b` 点估计；
  - 每个真实 segment pooling 输出最终有效 `c0/c1/c2` 点估计。
- 推断时先取得 MAP segmentation，再按预测 segment 输出参数。
- 三种 state 的每个 segment 都有 `c0/c1/c2`；“background”仍是生成器状态，不等于强制零增量。
- decoder 实现：
  `a + b(2ζ−1) + c0 + c1(2ξ−1) + c2 sin(πξ)`，
  并保持生成器的最终 AI clipping 语义。
- NumPy 与 Torch decoder 共享合同并要求数值一致。
- high-resolution AI 经现有 projection 和 forward model 得到 model-grid AI、reflectivity 和 synthetic seismic。

### MVP 对照

只使用 clean base seismic，固定同一 parent split 和一个 seed：

1. Oracle decoder：真 segmentation、真 `a/b/c0/c1/c2`，不训练。
2. Direct：阶段 0 的 `trace_dilated_tcn_mismatch`。
3. Structured LFM-only：独立训练，seismic 通道恒零。
4. Structured seismic+LFM：主模型。

Direct 只在共同的 model-grid AI 和 seismic 指标上公平比较；插值后的高分辨率输出只作参考，不解释为结构化预测。

### 指标和敏感性

报告：

- state accuracy 和 macro-F1；
- segment/object count error；
- boundary precision、recall、F1 和位置误差；
- teacher-forced segment 下的 `a/b/c0/c1/c2` MAE；
- predicted-segment 下的参数与重建误差；
- high-resolution AI、projected AI 和 seismic closure；
- 按 segment 厚度及调谐尺度分箱的指标；
- seismic 置零和父实现间 shuffle 后的性能变化。

阶段 1 的人工继续门禁为：

- 配对 parent bootstrap 显示 seismic+LFM 相对独立训练的 LFM-only，其 high-resolution AI MSE 改善的 95% 置信区间排除零；
- state macro-F1 或 boundary F1 至少有一个同步改善；
- seismic 置零或 shuffle 后，该改善明显减弱。

门禁结果只决定是否继续开发完整 HSMM 后验，不触发程序失败。

## 后续边界

MVP 暂不实现 exact forward-backward、HSMM NLL、参数概率分布、backward sampling、32 个 posterior realizations、多 zone、多 seed、真实工区结构化预测或真实井适配。这些仅在阶段 1 通过人工门禁后，从延期的完整 Structured Handoff 中逐项恢复。

## Test Plan

- 阶段 0：
  - 配置路径、外部 checkpoint 初始化及 provenance；
  - 六组实验共享 benchmark、split 和 normalization；
  - 三口排除井不进入 sampler、loss、held-out 或 checkpoint selection；
  - 留出井和排除名单重叠时硬失败；
  - 科学质量与 R0 指标只报告、不终止任务。
- 阶段 1：
  - producer 直接发布 `a/b` 和逐 lateral/object 有效参数；
  - 时间/深度 reader、轴、mask 和 active-zone 语义；
  - NumPy/Torch decoder parity；
  - Oracle high-resolution、projection 和 seismic round-trip；
  - 小序列穷举对照 semi-Markov Viterbi；
  - variable-length zone、padding 和无跨 split 泄漏；
  - encoder 输入只能包含 seismic/LFM；
  - zero/shuffle 敏感性及配对 bootstrap 报告。
- 按项目约定只编写测试，由用户运行。

## Assumptions

- 最新 benchmark 的物理邻道间距约为 25 m；xline 编号步长 4 不表示 100 m。
- 阶段 0 的推荐新增架构是 `trace_lateral_mixer`。
- 三个真实井权重均训练和报告，不预设最终胜者。
- 只有 schema、文件、轴、shape、finite、fingerprint、split 泄漏、Oracle closure 或数学不可计算允许硬失败。
- 新 Handoff 的 Suggested skills 包含：
  - `diagnose`：Oracle、decoder parity 和 HSMM 验证；
  - `improve-codebase-architecture`：producer 合同与共享 decoder 边界；
  - `grill-me`：阶段 1 门禁后决定是否进入完整后验模型。
