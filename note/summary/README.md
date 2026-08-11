# 反演研究冻结总结

`note/summary/final_audit/` 保存各轮实验的事实、指标、图件与失败结论。本页不再维护某一
次实验的候选排名，而是总结从连续增量网络、Structured GINN V2、井先验分解、Dynamic
Gain 和稀疏反射系数实验中得到的稳定认识。

## 当前总判断

真实地震能够可靠提供的是带限证据，而不是唯一的高分辨率地下模型。

目前已经有充分证据支持：

1. 神经网络可以从 seismic + LFM 中学习带限 log-AI 增量、状态占比、极性和大尺度界面
   活动；matched seismic shuffle 会明显破坏这些结果，说明地震输入确实被使用。
2. LFM 主要负责低频背景。很多模型相对旧基线的改善，本质上来自更好的 LFM，而不是
   网络独立恢复了绝对低频阻抗。
3. Dynamic Gain 可以提高弱事件的可见性，但不会自动增加过零点、事件数量或地质边界。
4. 正演一致性、重褶积相关性和视觉平滑只能约束等价解集合，不能证明高分辨率结果为真。

目前没有证据支持：

- 从单一叠后深度域地震唯一恢复微边界、精确反射系数或每段 `c0/c1/c2`；
- 由 HSMM、三状态转移、滤波残差、稀疏性或平滑正则选出的高分辨率纹理是真实地下纹理；
- ensemble mean 的平滑性能够代表任一合法 realization 的横向连续性；
- 当前 21 道 lateral mixer 对现有合成语料提供了独立、可重复的增益；
- 仅凭一口高质量井可以学习覆盖全工区的高频残差分布。

## 从失败实验中学到的内容

### 1. 连续增量网络：能修正带限残差，不能凭 loss 创造分辨率

早期 `ginn`、`ginn_depth` 和 ablation 路线证明，L2、TV 和简单正演 loss 能让输出更平滑
或让正演更接近地震，却不能可靠收窄高分辨率解空间。物理 loss 训练过强时还会产生锯齿
等一眼假的纹理。

深度域 ablation 中，nominal 与 mismatch training 基本相同；k=3 横向 mixer 没有增益；
追加 synthetic physics 后，测试 RMSE 约从 0.0706 恶化到 0.0794，相关性约从 0.8009
降至 0.7614。真实工区结果则主要表现为 LFM 周围的小幅振荡和振幅收缩。

### 2. Oracle 分段也没有解除参数欠定

Structured GINN V2 第一步在 truth segmentation 下证明 seismic 可以改善 profile 和 `c0`，
但 `c1` 基本不可恢复，`c2` 很弱。一次代表性审计中，每道约有 68.5 个地震样点，却需要
解释约 118.5 个 segment 参数；很小的地震失配可以与明显错误的 profile 同时存在。

因此，生成器保存了一组唯一参数，不代表这些参数能够被地震识别。短 segment 的系数相关
性更不能作为反演成功的证明。

### 3. HSMM 规范了解的形状，但没有增加可观测信息

HSMM 可以输出合法状态路径、控制 duration 和 segment count，却没有证明微边界来自地震。
实验中 full model 对状态和 AI 有地震贡献，但 ±一个 model sample 的 boundary 指标与
no-seismic 接近，boundary head 的排序能力也接近随机。

三状态先验在真实工区产生高阻—背景—低阻斑马纹；合并同状态续段后，结果又退回带限
尺度。其根因是阈值状态和转移表描述的是 synthetic producer 的构造语义，不是经过真实
工区验证的地质体先验。

### 4. Learned profile、ensemble 和 event track 都没有补上缺失证据

- profile head 在 truth segments 上有效，换成预测 extents 后明显退化；MAP-aware correction
  只有很小改善。
- K-member ensemble 的均值可以很平滑，但 representative 和单个成员仍跳变；高分辨率
  coverage 不足，interface 概率接近常数基线。
- per-trace 候选后匹配无法获得可信横向地质体；event-slot/event-track 模型又出现槽位身份
  不稳定、忽略 evidence 和滚动误差，二维/三维训练规模也超出项目可控范围。

这些失败共同说明：把同一份带限证据送入更复杂的 decoder，不会自动产生新的地下信息。

### 5. 横向连续性必须在单个合法输出上成立

数值 roughness、邻道一致率和 ensemble mean 容易掩盖肉眼明显的逐道跳变。横向门禁必须
直接检查最终交付的单个确定性结果或单个合法 realization，不能用成员平均替代。

现有合成剖面中的邻道本来就高度相似，导致 lateral mixer 与 single-trace 几乎相同。真实
结果看似连续，也可能只是输入地震和 LFM 连续，而不是模型学到了横向地质身份。

### 6. 真实与合成的主要差异包括事件尺度和可见性

井先验实验测得，真实地震事件通常比井上合成事件更宽：真实事件宽度总体中位约 62.5 m，
P10–P90 约 24–115 m；井上合成中位约 30 m，P10–P90 约 17–57 m。真实弱事件可能与
井上合成的强事件对应，说明振幅可见性和有效频带存在明显差异。

这不是简单的“把合成地震做得像真实地震”问题。观测域 augmentation 只能覆盖增益、相位、
频带和噪声等 nuisance，不能证明 synthetic geology 与真实地下具有相同条件分布。

### 7. 井曲线能给尺度认识，尚不能给可迁移高频先验

Gaussian、three-band、forward-consistent 和 body-scale 四轮分解确认：

- 10–75 m 中间带具有显著正演贡献；
- 同号连续变化真实存在，离散三状态交替并不合适；
- 多个不同平滑模型可以得到近似相同的正演响应；
- DoG、Gaussian 过零点和峰谷是滤波结果，不是地质边界。

Continuous Body Decoder 最终只是 5 m 带限证据的分段近似器，没有生成新的可信细节。
目前九口井中只有 NW11 的井震关系较可信，数据不足以标定全工区高频残差分布。

### 8. Dynamic Gain 与稀疏反褶积解决不了同一个问题

NW11 标定的 Dynamic Gain 在完整体上成功运行，但 raw/balanced 事件数量几乎不变。它适合
作为辅助通道，作用是让弱事件更容易被网络看到，而不是恢复绝对地下振幅。

倾角引导稀疏反射系数可以获得 0.983–0.996 的重褶积相关性，但恢复事件与井上真实反射
系数缺乏对应。该原型还使用了近似常速度深度卷积。它保留的核心反例是：高重褶积相关性
并不能识别真实反射系数。

## 重走 `ginn + enhance` 时的边界

这条路线可以作为务实工程方案重新开始，但两个模块的职责必须分开：

```text
seismic + LFM
→ ginn：可由地震验证的带限/中低频 log-AI
→ enhance：在明确监督来源下预测高频残差
→ 组合结果与独立诊断
```

后续实现应遵守以下边界：

1. `ginn` 的主要门禁是相对 LFM/no-seismic 的地震增量价值，不把正演闭合解释成高分辨率
   真值恢复。
2. `enhance` 需要明确的 residual 标签来源、训练/验证井隔离和振幅单位；DoG、反褶积、
   HSMM 输出或其他正则选解不能未经验证直接当伪真值。
3. raw seismic 与 amplitude-balanced seismic 应作为语义独立的输入；后者用于可见性，不
   替代原始绝对振幅通道。
4. 横向模块只有在优于 single-trace、且最终单个输出肉眼和指标都更连续时才保留。
5. 井震、forward 和 spectrum 指标分别报告；任何单项相关性都不能单独证明高频恢复。
6. 当前工区是深度域 TVDSS，垂向尺度使用 `SampleAxis`；xline step=4 只作几何寻址，横向
   距离使用实际米制坐标。

`enhance` 的真实监督仍是当前最大缺口。重新采用旧路线意味着接受这一风险并用有限井验证
约束它，而不是声称多解性已经被解决。

## 冻结目录索引

| 冻结目录 | 核心结论 |
| --- | --- |
| `20260713_D1_ablation_lfm_audit` | LFM 贡献主导，physics 与 lateral mixer 未带来增益 |
| `20260723_ablation_results` | 合成改善不能直接迁移到真实高分辨率结果 |
| `20260726_structured_ginn_v2_stage1_step1` | truth segment 下 `c0` 可学，`c1/c2` 弱 |
| `20260726_structured_ginn_v2_boundary_observability` | Oracle 条件下仍存在严重参数欠定 |
| `20260729_structured_ginn_v2_stage1_step2` | 状态/AI 使用 seismic，微边界仍由 prior 主导 |
| `20260730_structured_ginn_v2_stage1_step3` | boundary target 与可观测尺度不匹配，长 evaluator 无效 |
| `20260803_structured_ginn_v2_observable_hsmm` | 带限 evidence 可学；profile amplitude 是主要瓶颈 |
| `20260804_structured_ginn_v2_profiles_ensemble` | profile 存在 train/inference shift，ensemble member 不连续 |
| `20260806_structured_ginn_v2_stage1_evidence` | 带限 evidence 明确使用 seismic，lateral 增益未建立 |
| `20260806_structured_ginn_v2_stage2_event_tracks` | event slots 忽略 evidence，缺乏稳定地质身份 |
| `20260806_structured_ginn_v2_event_track_generation` | per-trace 后匹配不能形成可信横向 realization |
| `20260809_structured_ginn_v2_real_field_zero_shot` | 真实高分辨率退化为 prior 纹理或带限结果 |
| `20260810_well_prior_texture_and_decoder_failure` | 滤波给出尺度但不给地质体语义，body decoder 失败 |
| `20260811_dynamic_gain_and_sparse_reflectivity` | Dynamic Gain 只增强可见性；高重褶积相关不等于真反射系数 |

`20260711_synthoseis_detail_diagnostic` 与 `20260715_depth_workflow_backup` 是历史数据/工作流
快照，不作为独立科学结论入口。各项具体指标和图件以对应冻结目录中的 README 为准。
