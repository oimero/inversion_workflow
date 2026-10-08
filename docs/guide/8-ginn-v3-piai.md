# GINN v3 PIAI 反演

这一流程使用地震道和低频模型作为两个输入。网络在每条道上直接预测阻抗修正和自由子波，最终阻抗定义为低频模型加上网络修正。物理正演只使用固定的采样轴和速度关系；训练时的子波系数由网络学习。

流程同时适用于时间域和深度域。时间域使用统一的时间采样轴，深度域使用统一的垂深采样轴，并把低频模型一次转换成固定速度体。井控曲线只用于有标签损失和质量检查，体预测保留低频模型支持范围之外的缺失值。

## 联合训练机制

网络采用三个残差时序卷积块和两组三层双向循环网络，分别输出对数阻抗修正和子波。
训练从第一次更新同时优化三项损失：井上线性阻抗误差与井真值正演误差、预测阻抗正演误差、
以及井批与未标注批交换平均子波后的正演误差。井批子波用于未标注道时停止其梯度。
地震误差使用一份冻结的全局均值和标准差，阻抗误差使用训练井的线性阻抗标准差。

最终对数阻抗直接等于低频初模加原始网络修正。井标签沿用原生井控曲线的分段重采样值，
网络输出和井标签均不追加高斯平滑、频带投影或低频锚定。子波振幅自由学习。
更新次数固定，默认采用最后一次更新的模型。

这里沿用PIAI的网络与联合训练机制，并采用工作流统一的反射系数和正演采样对齐定义。
时间域使用褶积，深度域按固定速度计算两程时间；子波时间轴始终以秒表示。
直井位于地震道之间时，按实际井位采样地震、低频模型和固定速度。

## 配置

复制并修改配置文件中的输入运行目录、变体名称和可信井名称。学习子波窗口可以按总物理时长或奇数样点数确定；当前 Xihu 配置使用 0.200 秒的学习子波总时长，时间采样间隔为 0.002 秒，对应 101 个奇数样点。

```yaml
network:
  wavelet_duration_s: 0.200
  # wavelet_samples: 101  # 与 wavelet_duration_s 二选一
```

深度域使用参考子波的时间间隔建立固定正演时间轴。参考子波始终保留完整原始时间轴，独立用于正演质量检查；学习子波窗口按配置的物理时长或样点数建立。网络不会以参考子波初始化或归一化学习子波。

配置中的训练项控制更新次数、批大小、学习率和三个同时优化的损失权重。输入目录必须指向已经完成并可读取的低频模型、井控和正演输入运行。

当前 Xihu 工区使用 `experiments/ginn_v3/ginn_v3.yaml`，包含 8 口直井。第六步和第七步的配置文件分别位于 `experiments/ginn_v3/field_inputs_20261007/step6.yaml` 和 `experiments/ginn_v3/field_inputs_20261007/step7.yaml`；主配置引用对应的井控与低频模型运行目录。

## 训练

```powershell
python scripts/piai_train.py `
  --config experiments/ginn_v3/ginn_v3.yaml `
  --output-dir scripts/output/xihu_ginn_v3_piai_run
```

更新次数或计算设备可以通过命令行覆盖：

```powershell
python scripts/piai_train.py `
  --config experiments/ginn_v3/ginn_v3.yaml `
  --updates 1000 `
  --device cuda
```

训练目录包含最后一次更新的检查点、训练历史和输入合同。检查点保存网络配置、采样轴、冻结的归一化量、学习子波的平均值以及参考输入信息。

井上质量检查使用第六步保存的完整评价支撑段，并同时报告参考子波与学习子波的正演相关性。

## 体推理

```powershell
python scripts/piai_infer.py `
  --config experiments/ginn_v3/ginn_v3.yaml `
  --checkpoint scripts/output/xihu_ginn_v3_piai_run `
  --output-dir scripts/output/xihu_ginn_v3_piai_infer
```

推理以批次读取单条道，并把结果写入连续的 `npy` 映射文件。输出包含对数阻抗、有效样本掩码和全体有效道的平均学习子波。无低频支持的道保持为缺失值，不进行邻近填充。加入 `--smoke-tile-size` 可以只处理工区中央的小方块；加入 `--skip-segy-export` 可以只保存 `npy` 结果。

```powershell
python scripts/piai_infer.py `
  --config experiments/ginn_v3/ginn_v3.yaml `
  --checkpoint scripts/output/xihu_ginn_v3_piai_run `
  --smoke-tile-size 16 `
  --skip-segy-export
```

导出的物性体由对数阻抗转换得到，体坐标沿用输入地震的线号和采样轴。训练和推理使用同一份固定速度关系与正演时间轴，学习子波的物理振幅通过地震归一化尺度恢复后才用于物理对比。
