# Marmousi2 基准

基准通过公开数据适配器调用主工作流的训练、主体平滑、低频投影和预测接口。默认输入是时间域 Kirchhoff 偏移剖面。理想褶积输入用于检查数据管线。

## 数据与权重

公开数据使用仓库根目录下的目录结构：

```text
opendata/raw/marmousi2/
opendata/pretrained/wtie/
opendata/pretrained/rgt/
opendata/prepared/marmousi2_postm_time/
```

Wtie 的模型参数与权重配套读取；RGT 推理使用公开预训练权重和随适配器保存的网络实现。私有工区使用自身的数据目录。

默认伪井沿用 PostM 实验的实际位置，三个训练井与两个留出井分别位于剖面约 18.09%、32.57%、68.71%、77.65% 和 84.12% 处。训练与初模构建只读取训练井曲线。

## 准备与初模

在仓库根目录使用本机反演环境依次执行：

```powershell
python scripts/marmousi2_prepare.py
python scripts/marmousi2_wavelet.py
python scripts/marmousi2_prepare_lfms.py --device cuda
```

第一步读取速度、密度和地震，生成规则时间轴、伪井、固定井评价区间与训练配置。第二步使用 Wtie 提取子波，并调用主工作流的共识子波优化；合成数据的精确时深关系固定保留。第三步生成井趋势初模和沿 RGT 切片的井控低通克里金初模，默认配置指向后者。

准备产物使用 4 毫秒采样。默认主体平滑宽度为 10 毫秒，低通截止频率为 5 Hz。地震支撑依据原始逐道均方根振幅，阈值为剖面中位振幅的四分之一；该支撑用于训练中心、邻道输入和预测中心。

每口井的第六步与第八步波形评价使用同一固定目标区间，默认从 0.65 秒到剖面末端前 0.08 秒。全剖面指标同时报告全部道与有地震支撑的道，使用完整准备时间轴。短波指标分别报告 20 Hz 以上残差的均方根振幅与能量占比。

真值趋势初模可通过以下命令单独准备，作为合成数据的参考条件：

```powershell
python scripts/marmousi2_prepare_lfms.py --variants truth_huber_trend --default-variant truth_huber_trend
```

## 对照配置与运行

生成五种候选、三个种子的配置清单：

```powershell
python scripts/marmousi2_benchmark.py --comparison-plan --config opendata/prepared/marmousi2_postm_time/ginn_v2.yaml --seeds 20261004 20261005 20261006
```

候选包括物理预训练后混合微调、随机初始化后混合微调、物理预训练后遮挡与井微调、降低可见地震物理权重的混合微调，以及仅井微调。

默认每轮使用 256 次无标签更新与 256 次井更新。混合微调将无标签预算分为 128 次遮挡更新与 128 次可见更新；遮挡微调使用 256 次遮挡更新；仅井条件的无标签预算为零。清单记录轮数和更新预算，训练产物记录实际更新次数。

加入以下选项执行训练与数值评价：

```text
--run-training
```

训练产物、预测体和比较表保存到准备目录内的基准运行目录。种子汇总表按候选报告指标均值、最大最小值跨度，以及优于初模的种子数量。井误差约束下的轮次选择与可见地震权重见[主体反演基准对照配置](body-comparison-options.md)。

## 数值评价

保存的预测体可直接评价：

```powershell
python scripts/marmousi2_benchmark.py --prediction-npz path/to/prediction.npz --lfm-npz path/to/lfm.npz --scope final
```

评价报告包括原始井曲线与主体平滑目标的误差、训练井和留出井误差、全剖面与支撑剖面误差、正演相关性、低频漂移、高频与短波能量，以及真值正演和观测的相关性。验证范围使用训练井与验证井；最终范围包含测试井和全剖面真值。架构选择参考验证结果，测试井与剖面真值用于最终报告。

正演相关性同时受到波阻抗、子波和偏移地震与一维褶积模型之间差异的影响。真值正演结果帮助判断物理拟合目标的合理范围。
