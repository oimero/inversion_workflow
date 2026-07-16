# Synthoseis-lite science v2 时间域与深度域差异报告

## 结论

时间域与深度域采用同一套对象科学、规则轴投影、低频模型退化、随机命名和几何路径
Implementation。领域差异只保留在垂向坐标来源、单位、正演物理和 Artifact 表达中。

Artifact schema 保持 `synthoseis_lite_v4`，科学合同固定为：

- `science_revision = synthoseis_lite_science_v2`
- `projection_contract_version = finite_support_projection_v1`
- `lfm_degradation_contract_version = controlled_lfm_v2`
- `seismic_variant_contract_version = seismic_variants_v2`
- `random_stream_contract_version = synthoseis_random_v2`

这些字段属于配置、校准、生成、fingerprint、manifest、HDF5 根属性、Reader 和随机
namespace 的共同合同。当前 Reader 严格拒绝缺少这些字段的旧 v4 产物。

## 保留的领域差异

| Seam | 时间域 | 深度域 | 差异依据 |
|---|---|---|---|
| 垂向坐标 | TWT，单位秒 | TVDSS，单位米 | 坐标物理不同 |
| 井曲线落轴 | MD 经时深表进入 TWT | MD、KB 转为 TVDSS | 输入坐标转换不同 |
| 层位与 survey | 时间域层位和时间地震体 | 深度层位和深度地震体 | 数据来源不同 |
| Forward Adapter | 反射系数与时间子波卷积 | AI 转 Vp 后执行变速深度正演 | 正演方程不同 |
| forward halo | 高分辨率子波半宽 | 最大 Vp、双程时间和子波跨度 | 物理传播范围不同 |
| canonical cutoff | cycles/s | cycles/m | 轴单位不同 |
| axis static | 默认不生成常量时间静差 | 可生成米制深度静差 | 当前产品选择 |
| Artifact extras | 反射系数和时间轴字段 | Vp、TVDSS 和 depth basis | 消费合同不同 |

## 共享科学合同

### 对象真值

对象序列、背景、三参数对象剖面、横向相关、wedge、pinchout、RGT、catalog 和科学
拒绝统一由科学核心生成。请求中的轴单位只参与校验和元数据，不选择对象算法。

公共 calibration record 使用中性垂向坐标、zone bounds、真值采样间隔、轴单位和可选
深度基准。输入 Adapter 负责 TWT 或 TVDSS 转换，Serializer 负责领域 Artifact 字段名。

### Projection

连续真值和 RGT 使用同一个零相位有限支撑 FIR：

```text
numtaps = 32 * oversampling_factor + 1
cutoff_output_nyquist_fraction = 0.9
kaiser_beta = 8.6
```

完整 FIR 窗口定义 projection support。support 外保留模型中心点数值，只作为 operator
context。categorical 使用 centered factor-plus-one 窗口；public ROI 使用模型采样点的
高分辨率 state，有效 categorical 窗口只作为诊断。

两个 Forward Adapter 按同一公式申请上下文：

```text
required_context = max(
    projection_fir_half_width,
    forward_input_halo + observed_decimation_fir_half_width,
    domain_extra_halo,
)
```

Builder 强制 public ROI 同时属于 projection、observed-forward 和 physics support。

### Controlled LFM v2

顺序固定为 constant bias、axis trend、zonewise bias、lateral smooth bias、amplitude
scale、local missing-control bias、over-smoothing，最后只对 degradation 执行一次
canonical low-pass。

- 垂向趋势使用 public ROI 轴范围归一化到 `[-1, 1]`。
- zone bias 使用模型网格上的 zone ID。
- 横向相关使用实际米制距离，有效长度至少为四倍中位道间距。
- amplitude scale 围绕每道 public-valid 样点均值缩放。
- local kernel 为横向与垂向可分离 compact cosine；两个中心和幅值使用独立随机流。
- 时间配置把 Hz 转成 cycles/s，深度配置把波长转成 cycles/m。
- public mask 外统一为 NaN，QC 记录实现值、组件 RMS、相关长度、kernel 和总体 RMS。

### Seismic variants v2

公共语义包括 white/colored noise、global/tracewise/axis-lateral gain、wavelet phase、
wavelet shift、可选 axis static 和 combined。噪声目标 RMS 为 fraction 乘以去均值后的
observed RMS；零能量 ROI 直接拒绝。gain 始终为正，二维 gain 的横向与垂向随机流独立。

wavelet phase 和 shift 修改 wavelet，并通过领域 Forward Adapter 从高分辨率真值重新
正演。纯 gain/noise 使用 nominal observed。所有 variant 继续引用 nominal
model-consistent dataset，且 public valid mask 必须与 base 完全相同。

### Geometry

section path 的按距离重采样、line/XY roundtrip 和实际 inline/xline step 使用公共
Implementation。xline 步长为 4 时不会按步长 1 推断网格；连续剖面采样仍按实际米制
距离生成。领域 Adapter 只负责打开 survey、构建目标区和采样层位。

## 产品与 Artifact

生成入口只产出 field-conditioned Benchmark。standalone canonical sample suite 和 CLI
选择已删除；canonical background、target increment、controlled LFM 中的 canonical
variant、increment closure 和 GINN v2 训练语义保留。

HDF5 dataset 路径、dtype、sample index 主列、TimeBenchmark、DepthBenchmark 和 sample
view Interface 保持 v4 合同。manifest 保留 `suite=field_conditioned` 和
`canonical_enabled=false`。

## 门禁

- science revision 在配置、calibration、manifest、HDF5、fingerprint、Reader 和随机
  namespace 中一致。
- 同一无量纲 truth 放到秒轴或米轴后，projection 数值与 support 一致。
- LFM 逐组件顺序、独立随机 key、compact-cosine 支撑、相关长度 floor 和 QC 通过。
- wavelet variant 必须经过 Forward Adapter，variant mask 必须等于 base mask。
- xline step 1 和 step 4 均通过 line/XY roundtrip 与距离重采样。
- v4 Artifact 路径、dtype、sample index 和按样本惰性读取保持稳定。
