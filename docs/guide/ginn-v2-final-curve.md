# GINN v2 完整曲线平滑

网络使用地震、初始阻抗模型及几何特征预测 log 阻抗修正量。初始模型加上修正量后，对完整曲线施加一次半高全宽为 25 m 的高斯平滑。该结果用于井监督、地震正演和体数据导出。

初始模型可以使用既有的 150 m 低通比例切片克里金模型；150 m 参数只属于初始模型的制作及结果诊断。网络修正量允许包含长波成分。

## 井监督

原生采样的已滤波井曲线首先在有效连续区间内重采样到地震网格，然后整条曲线进行一次 25 m 高斯平滑，作为监督目标。目标数值由井曲线确定。初始模型的有效掩码用于限制共同建模区间。

井曲线的原生采样平滑副本仅用于固定参考图和诊断；实际监督链路使用重采样后的一次平滑结果。

## 损失与训练

| 损失 | 当前实验权重 |
|---|---:|
| 地震波形形状 | 1.0 |
| 井 log 阻抗 | 1.0 |
| 井曲线垂向导数 | 0.5 |
| 井旁地震波形形状 | 0.25 |

形状项使用相关性及标准化波形误差，其中后者内部权重为 0.25。正演子波、速度场以及网络输入预处理沿用既有设置。预训练只使用地震形状项，微调交替使用地震窗口与五口井监督。

实验保留 1 轮预训练、3 轮微调、固定随机种子与相同训练/验证位置。训练结束后根据已有选模评分选择 checkpoint，逐轮结果均保留。

## 运行与对比

```powershell
python scripts/ginn_v2_final_curve.py --config scripts/ginn_v2_final_curve.yaml --output-dir experiments/ginn_v2/results/final_curve_20260930
$env:PYTHONPATH = "$pwd/src"
python -m ginn_v2.final_curve_report --before experiments/ginn_v2/results/lfm_no_gain_20260929/proportional --after experiments/ginn_v2/results/final_curve_20260930
```

实验驱动读取配置中明确列出的初始模型、速度和地震缓存，并调用当前训练器及推理接口。旧的无增益补偿实验作为对照；本轮同时改变输出平滑位置、井目标和损失组成，结果反映这组联合修改。

比较曲线同时保存原始初始模型、零修正时的平滑初始模型、实际井目标、统一井参考和最终预测。训练与推理在相同井位置核对数值一致。完整配置、checkpoint 和逐轮图件保存在输出目录。

## 全工区预测

当前体预测配置使用选中的第 3 轮权重。运行：

```powershell
python scripts/body_infer.py --config experiments/ginn_v2/ginn_v2.yaml --output-dir experiments/ginn_v2/results/volume_final_curve_20261001
```

两个剖面方向的完整曲线先取平均，再对边界缺失位置传播邻近修正量，最后沿物理深度坐标统一做一次 25 m 平滑。输出包括线性阻抗 SEG-Y、相对初始模型的对数阻抗改变量 SEG-Y、中心剖面图及覆盖情况。目标层段外保留空值；道头和网格继承输入地震。