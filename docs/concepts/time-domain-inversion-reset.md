# 时间域反演重置

## 文档地位

本文是第五步之后时间域反演重构的唯一权威入口。

当前稳定生产链截止第五步：

1. `scripts/well_inventory.py`
2. `scripts/well_screen.py`
3. `scripts/well_preprocess.py`
4. `scripts/well_auto_tie.py`
5. `scripts/wavelet_generation.py`

`scripts/well_trajectory.py` 是井轨迹事实与 QC 旁路，也属于保留范围。

## 下一步实施顺序

### 0. 清理遗留代码（已实现）

转入 ginn-v2 分支后，对第六步及以后的代码做清理：

- 详见下面的"清理实施规格"。
- 这么做的目的是为了避免历史遗留代码影响 ginn-v2 阶段 Code Agent 的代码生成。

### 1. 建立 observability 研究闸门（已实现）

把现有“井曲线低通后正演，再与井旁地震比较”的 cutoff 诊断扩展为可观测性分析：

- 计算子波与离散反射系数算子的联合传递响应。
- 分析不同频率处的灵敏度、噪声放大和相位不确定性。
- 在已知真值的合成数据上测量可恢复频带，而不把一次井旁正演相关性直接解释为网络可恢复上限。

通过该闸门只能确定 `synthoseis-lite` 的建议实验范围；训练目标、损失频带和模型输出
分辨率还必须由合成基准上的逆问题可恢复性实验决定。

### 2. 建立 truth-first 的 `synthoseis-lite`（已实现）

先生成地质与高分辨率阻抗真值，再通过统一正演算子生成地震和学习目标：

- 使用真实目标层位或 RGT 框架约束二维地层几何。
- 以层为基本对象，生成横向相关的层厚、阻抗、楔状体、尖灭和倾斜层。
- Semi-Markov 只负责沉积或阻抗状态及其持续长度，不直接生成残差波形。
- 在垂向高分辨率网格上生成真值，经过抗混叠后降采样。
- 使用第五步选定的全局子波和统一 Robinson 正演关系生成合成地震。
- LFM、候选反演目标和残差全部由同一高分辨率真值派生。

高频残差不得作为独立随机曲线生成。它必须是高分辨率真值与指定基准目标之间的差，
并与正演关系、采样率和可观测性结论一致。

`.ref/synthoseis/` 只提供地质生成思路。

### 3. 在统一基准上做模型消融

至少比较以下基线：

- 干净重写的一维逐道基线。
- 二维 patch 反演。
- 加入横向、双边或倾角引导约束的二维模型。
- 直接预测 `log(AI)` 与预测反射系数后积分两种参数化。

旧 GINN 和 deterministic inversion 代码不作为新基线继续修补。需要对照时，
按新数据契约重新实现最小基线。

### 4. 依据证据选择正式架构

只有完成 observability、合成基准和消融后，才确定正式生产架构。当前优先研究：

- 共享空间编码器与多尺度 `log(AI)` 解码器。
- 渐进分辨率训练，而不是永久划分固定的 LFM、GINN 和 enhance 频段。
- 合成监督、物理一致性、失配鲁棒性和真实井微调的分阶段训练。
- 将 waveform correlation 用作物理一致性门槛，而不是唯一排序指标。

### 5. 验收标准

模型选择至少报告：

- 分频段阻抗误差。
- 楔状体最小可分辨厚度。
- 尖灭位置误差。
- 横向连续性和层间边界保持能力。
- 对子波、振幅增益、噪声、相位和时移失配的鲁棒性。
- 按空间簇去偏的留井或留平台真实数据验证。

## 清理实施规格

以下内容在 `ginn-v2` 分支中由 Codex 执行。

清理分为两个可独立审阅的提交：

1. **确定性删除**：只删除旧后半程脚本、`src/ginn/`、`src/ginn_depth/`
   和 `src/enhance/`。
2. **活动树收口**：删除旧 CUP 模块和旧指南，重写配置与契约文档，
   更新包说明、导航和入口。

第一提交只做不影响 MkDocs 导航的无争议代码删除；第二提交处理仍需保持前五步
可运行且文档站可构建的引用收口。

### 删除脚本

删除时间域旧后半程：

- `scripts/well_constraints.py`
- `scripts/lfm_precomputed.py`
- `scripts/dynamic_gain.py`
- `scripts/deterministic_inversion.py`
- `scripts/ginn_train.py`
- `scripts/ginn_inversion.py`

删除全部深度域和旧 enhance 脚本：

- `scripts/dynamic_gain_attr_fitting_depth.py`
- `scripts/dynamic_gain_model_depth.py`
- `scripts/enhance_gallery_depth.py`
- `scripts/enhance_inversion_depth.py`
- `scripts/enhance_qc_depth.py`
- `scripts/enhance_train_depth.py`
- `scripts/facies_control_anchor_depth.py`
- `scripts/frequency_split_diagnosis_depth.py`
- `scripts/ginn_inversion_depth.py`
- `scripts/ginn_train_depth.py`
- `scripts/lfm_facies_control_depth.py`
- `scripts/lfm_precomputed_depth.py`
- `scripts/merge_log_ai_anchor_depth.py`
- `scripts/vertical_well_auto_tie_depth.py`
- `scripts/wavelet_batch_synthetic_depth.py`
- `scripts/well_constraints_depth.py`

### 删除包与 CUP 模块

完整删除：

- `src/ginn/`
- `src/ginn_depth/`
- `src/enhance/`

从 `src/cup/` 删除旧后半程专用模块：

- `src/cup/seismic/facies_control_depth.py`
- `src/cup/seismic/gain.py`
- `src/cup/seismic/lfm_depth.py`
- `src/cup/well/constraints.py`
- `src/cup/well/frequency_bands.py`
- `src/cup/utils/raw_trace.py`
- `src/cup/well/viz.py`

`src/cup/well/frequency_bands.py` 中的 conditioning、分段低通和 cutoff 候选函数
与旧三频带对象及 `src/cup/seismic/lfm_time.py` 强耦合，不迁移到新位置。
未来 observability 研究需要时，依据新输入输出契约重新实现。

当前静态导入扫描确认，五个编号主脚本和井轨迹旁路均不导入上述删除模块：

- `well_inventory.py` 和 `well_trajectory.py` 使用 `cup.seismic.survey`。
- `well_auto_tie.py` 使用 `horizon`、`survey`、`trace_sampling`、`viz` 和 `wavelet`。
- `wavelet_generation.py` 使用 `viz`、`wavelet` 和 `wavelet_consensus`。
- `well_screen.py`、`well_preprocess.py` 只使用保留的 CUP 井与配置模块。

因此当前删除边界不会切断六个保留脚本的静态导入链。真正删除前仍需重复该扫描，
以捕获清理分支开始后出现的新依赖。

### 删除旧指南

- `docs/guide/6-well-constraints.md`
- `docs/guide/7-lfm-precomputed.md`
- `docs/guide/8-ginn-train.md`
- `docs/guide/9-ginn-inversion.md`
- `docs/guide/dynamic-gain.md`
- `docs/guide/deterministic-inversion.md`
- `docs/guide/enhance-synthetic-refactor.md`
- `docs/guide/deviated-well-src-cup-refactor.md`

### 同步收口

- 重写 `src/cup/config/workflow.py` 的旧后半程注册：
  - 从 `_RETIRED_KEYS` 删除所有 `well_constraints.*`、`lfm_precomputed.*`、
    `dynamic_gain.*`、`ginn_inversion.*` 和 `deterministic_inversion.*` 条目。
  - 从 `_SOURCE_RUN_SECTIONS` 删除 `well_constraints`、`lfm_precomputed`、
    `dynamic_gain` 和 `deterministic_inversion`。
  - 从 `_TIME_WORKFLOW_SECTIONS` 删除 `ginn_inversion` 及所有第五步以后 section。
  - 保留通用 retired-key 生成循环，但只让它遍历前五步和井轨迹旁路；删除后检查
    报错信息中不再出现旧 section 的交叉引用。
- 重构 `docs/concepts/csv-contracts.md`：
  - 保留并校正步骤 01 至 05 的 CSV 契约。
  - 将第四步 `well_tie_plan.csv`、`well_tie_metrics.csv` 的消费者收口为第五步
    `wavelet_generation.py`。
  - 将第五步子波产物定义为当前稳定链的终点，不再列出旧第六步、dynamic gain、
    deterministic inversion 或 GINN 消费者。
  - 整段删除步骤 06 至 09、旧旁路以及相关 CSV、NPZ、JSON schema，而不是尝试
    保留 `ginn_target_*`、频带诊断、anchor、enhance supervision 等字段说明。
- 从 `docs/concepts/data-and-coordinate-conventions.md` 删除旧三频带目标分解语义。
- 从 `docs/concepts/script-style.md` 删除旧 GINN、LFM、dynamic gain 和 enhance 所有权说明。
- 清理 `docs/api/cup/seismic.md`、`docs/api/cup/well.md` 和包 docstring 中的失效模块；
  必须同步修改 `src/cup/seismic/__init__.py` 的公开子模块列表，并移除
  `src/cup/seismic/horizon.py` 对已删除 `target_zone` 的说明。
- 将 `mkdocs.yml` 导航收口为：

  ```text
  首页
  快速开始
    时间域反演重置
    核心 CSV 契约
    数据与单位约定
    脚本风格指南
  教程
    01 井资产盘点
    02 LAS 曲线筛选与导出
    03 测井预处理
    旁路 井轨迹 QC
    04 井震自动标定
    05 全局子波生成
    旁路 正演可观测性分析
    旁路 合成基准生成与评估
    常见问题
  API
    cup
    wtie
  ```

  删除 06 至 09、正演增益、确定性反演和整个旧“架构与路线图”导航组。
- 更新根 `README.md` 和 `docs/index.md`，使稳定流程明确终止于第五步。
