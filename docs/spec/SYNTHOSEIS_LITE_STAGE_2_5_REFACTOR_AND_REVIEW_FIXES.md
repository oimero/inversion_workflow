# Synthoseis-lite 阶段 2.5 稳定化合同

本文记录 v4 主链在微纹理和 GINN v2 消费之前的稳定化合同。Synthoseis-lite 继续使用
`synthoseis_lite_v4`；生成器按时间域主链和深度域扩展组织，两个域共享样本消费合同，
但保留各自的正演实现。

## 单一有效掩码

v4 manifest 必须包含：

```yaml
mask_contract:
  id: single_valid_mask_v1
  semantics: roi_exact_full_support
  dataset: masks/valid_mask
```

`masks/valid_mask` 是完整目标 ROI。合成生成器不根据某个观测变体的有限性缩小该
区域；对象上下文和 halo 必须保证 ROI 内的目标、canonical background、increment、
输入 LFM、observed seismic 和 model-consistent seismic 全部有限。任何数组在 ROI
内缺少支持都会拒绝当前 attempt。

时间域和深度域的 reader、sample protocol、sample index 与 GINN batch 只消费这一
个 mask。`seismic_observed` 是网络输入，`seismic_model_consistent` 只供 physics
和 closure。mismatch 变体共享父 realization 的 `valid_mask`；变体支持不足时拒绝
整个 attempt，不裁剪共同 ROI，也不单独保留变体掩码。

GINN patch 在 ROI 外只做有限的零填充，`valid_mask` 仍是唯一训练和 physics loss
支持；填充值不参与统计、采样或 loss。

高分辨率正演支持数组可以作为 `truth/forward_valid_mask_highres/*` 等生成 QC 保存，
但它们不进入 reader、训练协议或 loss。每行可记录 `valid_sample_count` 作为报告
统计；它不作为 patch 接纳条件。

缺少 `mask_contract`，或仍写入旧的 observed/physics mask index 字段的 v4 产物，
由 reader 直接拒绝，不做字段回退或自动迁移。

## 深度域正演 backend

深度域配置显式声明：

```yaml
seismic_forward:
  backend: auto        # auto | numpy | torch_cuda
  dtype: float64
```

`auto` 在 CUDA 可用时选择 Torch CUDA，否则选择 NumPy；`numpy` 是显式 CPU 覆盖；
`torch_cuda` 在 CUDA 不可用时失败。GPU 路径复用 `cup.physics.torch_backend.forward_depth`，
输入、输出和采样轴均为 float64。HDF5 写入和 mask/QC 仍在 CPU。manifest 记录请求
backend、解析 backend 和 dtype。

base、phase、wavelet shift 与 combined mismatch 的重新正演都使用同一 backend。
模型网格 closure 不在每个 attempt 中重复调用同一正演；NumPy/Torch 数值一致性由
独立 fixture 测试覆盖。时间域本阶段不增加 GPU backend。

深度静态变体先在带 halo 的完整源道支持上做平移，再将结果支持与父 realization 的
公共 ROI 相交；不能把 ROI mask 当作静态插值的源支持域。若相交后的支持不能完整覆盖
父 ROI，则整个 attempt 拒绝。该规则同样适用于 combined 中包含的静态平移。

## 包边界

- `cup.synthetic.core` 提供合同、随机流、artifact 和公共协议，不导入 time/depth；
- `cup.synthetic.time` 保持时间域主链；
- `cup.synthetic.depth` 提供深度域生成、米制抗混叠和 AI–Vp 正演适配；
- `cup.synthetic.readers` 只读取物化字段并校验 manifest/index 合同；
- `cup.synthetic.reporting` 不参与生成或训练。

微纹理 emitter、paired A/B/C benchmark、GINN physics halo、R0/R1 和旧产物迁移
不属于本阶段。physics 的 halo/central-crop 规则只在 canonical increment 规格中
作为后续 GINN 训练约束记录。

## 验收门禁

本地 ignored tests 应覆盖：合同缺失和旧字段拒绝、ROI 内有限性硬断言、变体共享
支持、time/depth reader 字段树、单一 mask 的 GINN batch、CPU/CUDA float64 parity、
backend 选择错误和 xline 步长 4 的坐标语义。完成合同测试后，先运行新的深度
`field_conditioned --debug-attempt-limit 1` writer → manifest → reader smoke，再在
新输出目录重新生成完整 v4 工区；冻结历史目录不覆盖。
