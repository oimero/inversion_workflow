# Structured GINN V2 实施规格

## 1. 目标与科学边界

Structured GINN V2 是一个两尺度条件生成器：

```text
seismic + full LFM + lateral context
→ BandlimitedEvidence
→ conditional HSMM
→ K 个横向连续的 high-resolution realizations
→ 代表解 + ensemble summary
```

网络恢复调谐尺度内可约束的 AI 增量、状态占比和界面活动。条件半马尔科夫
先验补全亚调谐尺度的薄层数量、微边界和三参数 latent。

高分辨率输出是给定观测与先验的地质 realization，不表示地震逐层分辨了全部薄层。
正式交付同时包含：

- 带限证据与不确定性；
- 多个条件 realization；
- 一个完整、合法、可复现的代表 realization；
- 明确的观测支持、方向分歧和 LFM 敏感性。

第一版不使用 differentiable forward loss、物理排序、井监督或真实无标签微调。
time/depth forward 只作为 Oracle 和诊断 seam。

### 1.1 冻结的重启依据

旧 boundary observability audit 使用 16 个 parents、1157 个 truth boundaries：

- 调谐尺度中位数约为 49.8 m；
- 最近边界间距中位数为 5 m；
- 59% 的最近间距不超过 5 m；
- 95% 的最近间距不超过 10 m；
- 没有边界独立于一个调谐尺度之外。

因此，generator 微边界、精确 segment count 和单个 `c0/c1/c2` 不是地震网络的
主要监督门禁。旧实验已经证明 seismic 对 decoded profile 和 projected AI 有贡献，
同时也证明旧 boundary/count 输出主要由 duration/state prior 和生成器统计控制。

### 1.2 实施状态

截至 2026-07-30：

- 阶段 0 的代码清理、canonical V2 producer、Oracle、SHA 审计和时深接口已落地；
- 阶段 1 的证据网络、conditional HSMM、ensemble、训练与 section 门禁接口已落地并通过 smoke，正式 corpus 训练尚未执行；
- 阶段 2 的双方向证据融合和两遍流式推理接口已落地，真实工区输入适配与 zero-shot 门禁尚未执行。

“已落地”表示代码合同和小规模验证可运行，不表示相应科学门禁已经通过。

## 2. 固定坐标与数据合同

垂向数据使用 `SampleAxis`：

```text
depth: domain=depth, unit=m, depth_basis=tvdss
time:  domain=time,  unit=s, depth_basis=null
```

同一个 checkpoint 只服务一种 domain。time/depth 复用代码和 interface，不复用权重。

固定约定：

- `inline/xline` 只负责几何寻址；
- 横向物理距离使用 `lateral_m` 或实际 XY；
- 工区 `xline_step=4` 不进入网络、HSMM、forward 或距离计算；
- 垂向 metrics 使用 sample 数、axis unit 或 tuning-scale fraction；
- canonical artifact 只保存 clean seismic；
- dirty waveform 位于 reader 与模型之间，并与 parent identity 原子绑定；
- 缺失字段、轴错位、mask 错位和非有限值直接失败，不提供 fallback。

旧 structured artifact、checkpoint 和 split manifest 不属于新 interface，不提供迁移层。

## 3. 阶段 0：仓库与 canonical corpus 重置

### 3.1 包结构

`src/ginn_v2` 控制为以下 11 个深模块：

```text
__init__.py       # 稳定包 interface
contracts.py      # 公开输入、证据、结果与错误
representation.py # LFM anchor、Torch decoder、结构化表示
evidence.py       # 带限证据网络与 tuning-window targets
semi_markov.py    # conditional HSMM、sampling、ensemble
generator.py      # ConditionalGenerator
learning.py       # 训练、calibration 和正式评价
inference.py      # section/volume 双方向推理
artifacts.py      # checkpoint 与结果 artifact
augmentation.py  # domain-neutral waveform augmentation
runtime.py       # device 与日志
```

包根只暴露 `ConditionalGenerator`、训练/评价/推理/正演诊断入口和少量公开类型。
CLI 只解析参数并调用这些入口，不拥有训练循环、resume journal 或指标聚合实现。

Synthoseis-lite 的 NumPy structured decoder 和 publication Oracle 位于
`cup.synthetic`。producer 不依赖 `ginn_v2`。Torch decoder parity 在 GINN seam 验证。

### 3.2 Canonical corpus V2

一次正式发布的 canonical 数据合同由以下三项组成：

- `synthetic_benchmark.h5`；
- `realization_index.csv`；
- `benchmark_manifest.json`。

这三者分别是数据、索引和发布合同，不是重复 truth。Oracle、QC、日志和图件是
诊断报告，不构成第二套训练数据。

校准报告为每口输入井发布背景/残差、三参数 profile 拟合和对象参数三组图件。
生成报告为每个实际成功的 `scenario_id` 发布一张 projected log-AI、正演地震和
高分辨率 state 的组合图。图件与附加 QC 表属于可选诊断；单井、单场景或整组绘图
失败写入 `skipped_figures.csv` 或 `diagnostic_warnings.json`，不改变 canonical
数据的发布状态。

短 patch corpus：

- accepted parents：2400；
- 每个 parent：25 道、25 m 间隔、600 m 跨度；
- 模型完整上下文：21 道、约 ±250 m；
- none/wedge/pinchout 各 800；
- training/tuning/calibration 为 1680/360/360；
- 每个 family 在三个 split 中为 560/120/120。

最终 section benchmark：

- 48 个完整 parents；
- 每个 family 16 个；
- 每个 family 包含 8 个 IID seed 和 8 个组合 holdout；
- 使用 12 条冻结 blind paths，其中 6 条 inline、6 条 xline；
- 每条 section 约 121 道；
- 不参与训练、调参或 calibration。

横向相关长度使用物理单位：

```text
correlation_length_m = 300 / 900 / 3000
```

组合 holdout 固定为：

- none：900 m 与 high-variability pair；
- wedge：900 m、high-variability pair、right-to-left；
- pinchout：900 m、high-variability pair、right-to-left、065。

单独的 family、direction、correlation 和 variability 值均在训练中出现。

最多规划 4000 个 candidate attempts。preflight 在正式正演前验证每个配额桶具有
足够的 truth-valid candidate；正式生成遇到单个 realization 的 projection、forward
或 benchmark-build 拒绝时记录 warning，并继续消耗同一配额桶的备用 candidate。
备用 candidate 耗尽后仍存在的配额短缺写入 `quota_report.csv`，已生成语料以
`completed_with_warnings` 发布。schema、axis、HDF5、writer/reader 和 Oracle
完整性错误仍然终止发布。

smoke 与正式生成是两个独立命令。smoke 只估算吞吐和体积并报告是否超过约 5 小时
或 3.5 GB，不替正式命令作自动启停决定。

### 3.3 SHA-256 合同

固定规则见 `SHA256_CONTRACT_SLIMMING.md`：

- producer 发布不可变 contract 时计算一次 fingerprint；
- consumer 读取并记录直接上游 fingerprint；
- consumer 不重算上游文件 SHA；
- consumer 不比较已保存 fingerprint 并以不一致拒绝输入；
- schema、status、ID、domain、unit、axis、shape、dtype、mask 和几何关系严格校验；
- 稳定随机流、split identity 和外部下载 checksum 可以使用 SHA。

### 3.4 time/depth 对称

scenario、split、横向采样和 writer/reader 合同位于共享 core。domain adapter 只处理：

- time 的 TWT 与 `forward_time`；
- depth 的 TVDSS、AI--Vp relation 与 `forward_depth`；
- 合法的 domain-specific forward extras。

横向采样间隔不因 domain 硬编码。time/depth seismic 使用同一显式 amplitude
convention。纯合成 time fixture 与 depth fixture 运行相同 writer、reader、batch、
loss、generation 和 diagnostic interface；正式 benchmark 只运行 depth/TVDSS。

## 4. 阶段 1：合成监督条件生成器

阶段 1 最多分三步。

### 4.1 第一步：BandlimitedEvidence

公开证据：

```text
BandlimitedEvidence
├── background_lfm_linear
├── bandlimited_increment_mean / scale
├── state_occupancy[..., 3]
├── interface_activity
├── local_tuning_scale
├── support
├── model/highres axes
└── physical lateral geometry
```

`interface_activity` 是调谐窗口内的阻抗变化活动，不是 generator 微边界概率。

tuning-window target 固定为：

- dominant frequency 来自冻结 wavelet；
- time tuning scale 为 `1 / (2 f_dom)`；
- depth tuning scale 为 `Vp / (4 f_dom)`；
- 平滑核的 FWHM 等于 local tuning scale；
- 同一核作用于 anchor-relative log-AI、state one-hot 和绝对 interface jump measure。

模型为可变长度一维横向网络，使用实际米制距离和显式 mask。25 道 parent 的内侧
5 道具有完整 21 道上下文；边缘道验证部分上下文合同。

固定比较：

- full：seismic + full LFM；
- no-seismic：full LFM；
- single-trace；
- full-LFM-only；
- zone-linear anchor-only；
- matched center、parent 和 neighbor shuffle。

主门禁是 bandlimited increment RMSE/NLL、state occupancy proper score、
interface activity error 及上述 controls 的 parent 配对改善。微边界 F1、精确
segment count 和单系数相关性只作审计。

### 4.2 第二步：Conditional HSMM 与 ensemble

核心 interface：

```python
evidence = generator.observe(observation_tile)
prediction = generator.realize(evidence, generation_policy)
```

HSMM 只消费公开的 `BandlimitedEvidence`，不读取 encoder hidden feature。

固定语义：

- transition/duration base prior 只从 training parents 标定；
- duration 单位为 zone fraction；
- per-trace forward-backward 和 backward sampling 为 exact；
- renewal 与 state change 分离；
- 相邻 segment 可以具有相同 state；
- interface activity 条件化 renewal hazard；
- 三参数 head 只使用 evidence、state、duration、extent 和 LFM anchor；
- 不可识别 segment 以 decoded profile 为主要监督；
- `c0/c1/c2` 是 prior-selected decoder latent。

默认 `K=16`。一次 posterior dynamic program 后执行 K 次 backward sampling。

输出：

- K 个合法 structured realizations；
- high-resolution/model-grid ensemble mean、std 和 occupancy；
- interface density；
- segment count/duration distributions；
- 一个完整、合法、可复现的代表 realization。

代表解是 K 个整套成员中，在有效支持内 projected/bandlimited AI 距离 ensemble mean
最近者；并列时选择条件分数更高者。禁止逐道拼接，禁止把逐点 mean 当成 realization。

主门禁使用 coverage、CRPS/energy score、segment/duration 分布校准和代表解的
bandlimited 表现。精确微边界和单系数指标不决定阶段推进。

### 4.3 第三步：横向耦合与 section 门禁

网络保持一维。二维微结构连续性使用 coordinate-stable random Fourier fields：

- random identity、XY 和 zone identity 唯一决定 latent；
- tile 大小、执行顺序和 GPU batch 不改变输出；
- correlated uniforms 耦合 renewal/state/duration；
- correlated Gaussian latents 耦合三参数；
- birth、death、pinchout 由 evidence 与 topology mask 控制；
- 不宣称求解 exact 2D HSMM。

calibration 后只运行一次 48-parent full-section benchmark，检查：

- 代表解和 ensemble 的横向 variogram/roughness；
- event continuity、birth/death 和 pinchout false bridging；
- direction reversal；
- tile/halo/stitching invariance；
- neighbor shuffle；
- clean/dirty 对称性；
- IID seed 与组合 holdout 的结论一致性。

评价不建立 per-parent journal、checkpoint SHA 或完整数据预扫描。日志按固定
batch/parent 间隔发布；任何合同错误直接终止。

## 5. 阶段 2：真实工区冻结 zero-shot

先运行：

- 6 口可信井剖面；
- 3 口低质量井诊断剖面；
- 至少 2 条 blind sections。

full 与 no-seismic checkpoint 同时运行。inline/xline 分别生成
`BandlimitedEvidence`，融合 calibrated evidence 后只运行一次条件生成器。

section 保存 K 个完整成员。全体积只保存：

- bandlimited evidence mean/scale；
- state occupancy 和 interface activity；
- high-resolution ensemble mean/std；
- representative high-resolution log-AI 和 segment table；
- direction disagreement；
- tuning-scale seismic support；
- LFM variant sensitivity；
- support/stitching mask；
- 可局部重生成的 realization identities。

全体积使用两遍流式生成：第一遍累计 ensemble summary 和成员分数，第二遍按选定
member identity 重生成代表解。结果不依赖 chunk 顺序。

真实井门禁：

- full 相对 no-seismic 在井上对应频带的配对改善为正；
- full 相对 full-LFM 和 zone-linear anchor 有整体增量价值；
- high-resolution ensemble 对井曲线具有合理 coverage；
- 代表解在多数可信井上不相对 LFM 退化；
- 低质量井不进入聚合门禁；
- blind sections 无 seam、方向条带或微结构随机跳变。

剖面门禁通过后才运行全体积。zero-shot 结果出来后再规划井监督、物理训练约束和
真实 adaptation。

## 6. 验证合同

实现至少覆盖：

- canonical V2 quota、split、parent 原子隔离和 V1 明确拒绝；
- producer Oracle、NumPy/Torch decoder 和 projection parity；
- time/depth 同 interface smoke；
- consumer 不因 fingerprint mismatch 拒绝输入；
- tuning-window target 的时深单位一致性；
- `BandlimitedEvidence` 概率、scale、axis 和 support；
- 小序列 HSMM forward-backward/sampling 与 brute force；
- same-state renewal；
- K sampling 只执行一次 forward recursion；
- representative 是 ensemble 中真实成员；
- spatial sampling 的 tile/order/batch determinism；
- inline/xline evidence 先融合、微结构只生成一次；
- section continuity、pinchout 和 false bridging；
- full/no-seismic 真实井配对报告；
- forward diagnostic 不训练、不排序、不替换代表解。
