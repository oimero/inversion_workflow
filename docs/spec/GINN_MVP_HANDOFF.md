## 阶段 0：连续模型基线

### 前置工作

* 使用最新第五、六步结果重跑真实工区 LFM。
* R0 固定使用比例切片 LFM、当前振幅匹配地震变换和已有固定剖面清单。
* benchmark 不使用自动发现结果；实验配置记录明确目录、manifest 身份和 fingerprint。
* 扩展训练脚本，使其接受可选配置路径。
* 保留从外部不可变 checkpoint 初始化的通用能力，并记录 checkpoint 路径、哈希和来源实验；但阶段 0 的真实井监督实验不使用外部初始化，均从相同随机初始化开始联合训练。
* 真实井联合实验必须复用 mismatch 实验的 synthetic benchmark、parent split、normalization、patch 和训练预算。

### 固定实验矩阵

共训练八个结果，使用相同 seed、父实现划分、训练集 normalization、patch 和训练预算：

1. `trace_dilated_tcn_clean`

   * `trace_dilated_tcn`，hidden 32，depth 5。
   * 只用 base seismic 训练。
   * 在完整 mismatch validation 分布上评估。

2. `trace_dilated_tcn_empirical_only`

   * 同一网络。
   * base/variant 父样本权重各 0.5。
   * variant 仅采样 empirical_mean_gain、empirical_residual_gain、empirical_full_gain。
   * 不包含 simple gain、noise、phase 或 registration views。
   * 用于隔离层序坐标振幅 prior 的贡献。

3. `trace_dilated_tcn_mismatch`

   * 同一网络。
   * base/variant 各 0.5。
   * 使用完整 13 个 amplitude、noise、phase 和 registration views。

4. `trace_dilated_tcn_mismatch_empirical`

   * 同一网络。
   * base/variant 各 0.5。
   * variant 包含 3 个 empirical、2 个 noise、2 个 phase、4 个 registration。
   * 移除 simple gain。
   * 用于检验 simple gain 是否冗余。

5. `trace_dilated_tcn_mismatch_realwell_joint_s010`

6. `trace_dilated_tcn_mismatch_realwell_joint_s030`

7. `trace_dilated_tcn_mismatch_realwell_joint_s050`

   * 三者均从随机初始化开始，不使用预训练 checkpoint。
   * synthetic 与 real-well 从训练开始联合优化。
   * synthetic block 完全复用 mismatch 实验。
   * real-well 每步参与训练，不替代 synthetic。
   * 三组损失份额：

     * s010：synthetic 0.90，real well 0.10；
     * s030：synthetic 0.70，real well 0.30；
     * s050：synthetic 0.50，real well 0.50。
   * 权重总和固定为 1。
   * 使用完整训练预算，不使用短期微调。
   * 排除井不参与训练或选择。
   * 留出井冻结，仅用于诊断。
   * checkpoint 按 synthetic mismatch MSE 选择。
   * 同时保存 best 和 final：

     * best 用于主评估；
     * final 用于观察井适配极限与漂移。
   * 每 epoch 报告：

     * synthetic MSE；
     * 监督井 MSE；
     * 留出井 MSE；
     * 相对 mismatch 的预测漂移。

   * 排除井：L3-NW2A、L6-NW3A、NW7（来自 Step 5 跳过时移扫描名单）。
   * 留出井：2-ANP-2A-RJS（非排除井、非子波来源、可靠候选中有最多评分样本，n=331）。
   * 监督井：L1-NW1、L5-NW5、L9-NW4A、NW8（排除井、留出井和子波来源井 NW11 之外的可靠候选）。
   * real_wells 源使用 `exclude_same_cluster: true, cluster_radius_m: 500.0`。
   * 井角色由 `assign_supervision_roles()` 解析并冻结。

8. `trace_lateral_mixer_mismatch`

   * 引入横向上下文。
   * 与 mismatch 完全相同训练分布。
   * 用于检验邻道信息价值。

### 统一评估

* 使用固定 parent split。
* 输出 clean、逐 view、加权总体指标。
* 报告差值：

  * mismatch − clean；
  * empirical_only − clean；
  * mismatch − empirical_only；
  * mismatch_empirical − mismatch；
  * real-well − mismatch；
  * lateral − mismatch。
* real-well 实验额外报告：

  * 监督井指标；
  * 留出井指标；
  * synthetic 保持程度；
  * 预测漂移；
  * best vs final；
  * R0 剖面对比。
* 不自动选赢家。
* 所有结果仅报告，不触发失败。

### 阶段 0 解释边界

* 真实井监督用于校准映射，不证明空间泛化。
* 留出井仅为弱诊断。
* 高权重实验用于暴露过拟合风险。
* 若井改善但 synthetic 恶化，应考虑结构性改进而非继续加权。

---

## 阶段 1：Structured MVP

### 科研问题

阶段 1 不重复验证连续 Direct 模型中 seismic 与 LFM 的一般通道重要性。阶段 0 已表明，连续 increment 预测主要由 seismic 驱动，LFM 输入贡献较弱。

阶段 1 要回答的专门问题是：

> 在固定的状态转移、持续期、对象参数范围和确定性 decoder 已经提供强结构先验的情况下，正确 seismic 是否仍能为高分辨率对象分段和参数推断提供超出结构化先验的可验证信息？

无 seismic 对照用于排除 Structured 模型仅凭 HSMM 和 calibration prior 生成合理薄层的可能，不解释为重新测量 LFM 对连续反演的贡献。

### MVP 对照

使用 clean base seismic、固定 parent split 和一个训练 seed：

1. **Oracle contract round-trip**

   * 使用真实 segmentation 和真实 `a/b/c0/c1/c2`。
   * 不训练模型。
   * 只验证 latent artifact、decoder、projection 和 forward 合同闭合。
   * 不解释为 seismic 可达到的理论上限。

2. **Direct**

   * 使用阶段 0 冻结的 `trace_dilated_tcn_mismatch`。
   * 只在 model-grid AI 和 seismic 指标上比较。
   * 高分辨率插值仅作参考。

3. **Structured no-seismic control**

   * 与主模型完全相同结构。
   * seismic 输入恒为零。
   * 独立训练。
   * 表示结构化先验 + LFM 的能力上限。

4. **Structured seismic+LFM**

   * 主模型。
   * 使用正确 seismic。
   * 与 no-seismic 对照比较。

### Seismic 证据检验

对主模型执行：

1. **Correct seismic**

   * 使用真实 seismic。

2. **Parent-shuffled seismic**

   * 替换为其他 parent 的 seismic。
   * 保持其他输入不变。

可选保留评估时 seismic 置零作为辅助诊断。

### 主要比较

* Structured seismic+LFM − no-seismic；
* correct − shuffled；
* Structured − Direct；
* Oracle − Structured。

解释：

* 第一项：seismic 增益；
* 第二项：是否使用正确 seismic；
* 第三项：结构化 vs 连续；
* 第四项：latent gap。

### 人工继续门禁

必须同时满足：

1. high-resolution AI MSE 改善显著；
2. state 或 boundary 指标改善；
3. shuffled 后性能下降；
4. 改善体现在结构指标，而非仅 seismic closure。

以下不算通过：

* 仅 seismic closure 改善；
* 无结构指标改善；
* shuffled 无影响；
* 仅生成更高频但边界不准。

门禁仅决定是否继续，不触发失败。
