# SHA-256 契约瘦身规范

## 1. 状态与目的

本文是 SHA-256 跨步骤契约的**现行规范和迁移审计**。仓库已按本文完成 schema
硬切换；旧 run 不兼容且必须重建。后续新增内部运行时文件哈希比较均视为回归。

本文覆盖[工作流总览](../index.md)中的稳定链、深度域一次性路径和研究旁路。
CSV 字段语义仍以[核心 CSV 契约](../concepts/csv-contracts.md)为准，domain、单位、
坐标与采样轴仍以[数据与坐标约定](../concepts/data-and-coordinate-conventions.md)为准。

目标不是消灭 SHA-256，而是把它限制在一个职责上：**生产者发布不可变契约时，
生成一个供实验溯源使用的内容指纹**。SHA-256 不再充当内部步骤之间的运行时准入门。

## 2. 结论

1. 内部步骤消费上游 run 时，不得重新计算文件 SHA-256，也不得因 SHA-256 不一致拒绝输入。
2. 所有成功发布的 run 都不可原地修改。输入、配置或产物变化必须产生新 run。
3. 每个可独立选择和消费的契约只发布一个 `contract_fingerprint_sha256`。
4. 下游只记录直接上游的契约指纹，不复制逐文件哈希，也不展开全部祖先链。
5. schema、状态、显式 ID、domain、depth basis、单位、轴、shape、dtype、mask 和坐标
   关系仍必须严格校验；这些才是工作流正确性的准入条件。
6. 旧 schema 与旧 run 不兼容。迁移不提供双读、自动升级或 fallback，旧产物必须重建。

这里的“只记录、不校验”是有意选择。该仓库用于研究，产物由本地工作流生成，
主要风险是错域、错单位、错轴、错 variant 或错字段语义，而不是对抗恶意文件替换。
在这个前提下，遍历大型地震体、HDF5、每井 NPZ 和全部旁支文件做重复哈希，成本高，
却无法证明科研语义正确。

## 3. 术语和不可变 run

### 3.1 契约实体

“契约”是一个可被下游独立选择的发布单元，而不一定等于一个目录：

- Step 1–6 通常一个成功 run 对应一个契约；
- Step 7 的每个可选 LFM variant 是独立契约；
- 一次 Synthoseis calibration、一个 benchmark、一个模型 run、一次 R0 预测和一次 R1 诊断分别是独立契约；
- QC 图、日志、临时表和可重新生成的报告附件不是契约。

### 3.2 发布规则

生产者必须在临时目录完成写出和语义校验，最后计算契约指纹，再写最终 manifest，
然后原子发布或关闭该 run。只有成功状态的 run 才能拥有 `contract_fingerprint_sha256`。
“成功状态”由各 schema 定义，包括 `success`、`ok`、可消费的
`completed_with_warnings` / `development_limited`，以及 R0 的终态
`needs_forward_diagnostic`；`failed` 和构建中状态绝不能拥有契约指纹。

成功发布后：

- 不得覆盖、追加或手工修补其中的文件；
- 显式 `--output-dir` 也不得指向已有成功 run；
- 修改任何输入或业务配置必须创建新 run；
- 失败或构建中的目录不得被下游当作契约消费。

当前 01–07、研究旁路、Synthoseis、模型和 R0/R1 发布入口均拒绝复用已有输出目录；
成功 run 的不可变性已经作为本次 schema 硬切换的一部分实施。

## 4. 唯一契约指纹

### 4.1 Manifest 接口

新 schema 的发布 manifest 顶层统一包含：

```json
{
  "schema_version": "<producer-contract-schema>",
  "status": "success",
  "contract_fingerprint_schema": "contract_fingerprint_v1",
  "contract_fingerprint_sha256": "<64 lowercase hex>",
  "input_contracts": {
    "<logical-role>": {
      "path": "<repo-relative run or manifest path>",
      "contract_fingerprint_sha256": "<direct upstream fingerprint>"
    }
  }
}
```

`input_contracts` 只列直接输入。例如 R0 记录所选 LFM variant、显式消费的 WellControlSet、
模型 run 和子波契约，但不展开这些输入各自记录的更早祖先。层位、井和地震来源等完整历史
通过逐层 manifest 追溯。

路径用于定位，指纹用于标识。路径、输出根目录和时间戳不参与指纹计算。

### 4.2 指纹载荷

公共实现必须构造下面的逻辑载荷，再对规范 JSON 的 UTF-8 字节计算 SHA-256：

```json
{
  "fingerprint_schema": "contract_fingerprint_v1",
  "contract_schema_version": "<producer-contract-schema>",
  "semantics": {
    "sample_domain": "time|depth|none",
    "sample_unit": "s|m|none",
    "depth_basis": "tvdss|none",
    "other_contract_semantics": "<producer-defined JSON values>"
  },
  "business_config": "<resolved semantic configuration>",
  "input_contracts": {
    "<logical-role>": "<direct upstream fingerprint>"
  },
  "primary_artifacts": {
    "<logical artifact name>": "<transient file-content SHA-256>"
  }
}
```

规范 JSON 固定使用：对象键按字典序排序、`ensure_ascii=false`、紧凑分隔符、UTF-8；
禁止 NaN 和 Infinity。列表顺序必须由契约语义明确，不能依赖文件系统枚举顺序。

`business_config` 只包含会改变科研含义的已解析配置。以下运行环境信息不得进入指纹：

- 输出路径、run 名称和时间戳；
- CPU/GPU 设备选择、线程数和日志级别；
- 图件尺寸、终端显示和纯诊断开关；
- 本机绝对路径。

Git commit、命令行、环境版本和普通来源路径仍可作为 provenance 记录，但它们不是
`contract_fingerprint_sha256` 的替代字段，也不应再各自派生 SHA 字段。

### 4.3 主产物摘要

`primary_artifacts` 中的逐文件摘要只是生产者计算顶层指纹时的临时数据，不写入 manifest。
为兑现“路径不参与身份”，JSON、CSV 和 NPZ 中只用于定位的 `*_path`、`*_dir`、
`*_file`、输出根目录、时间戳、设备和日志 provenance 在内容摘要前规范化移除；数组值、
dtype、shape、结构键和非定位元数据仍参与摘要。二进制主件则按其内容顺序读取。

纳入的文件必须满足“其字节会被下游作为科研输入直接消费”，例如：

- Step 3 的状态表和预处理 LAS；
- Step 4/5 的正式子波、转换表、被下游引用的 LAS 和正式清单；
- Step 6 的 manifest 与全部成功井 NPZ；
- 某个 Step 7 variant 的主 LFM、必要 sidecar 和语义元数据；
- calibration 主 JSON、benchmark HDF5 与 sample index；
- 模型主 checkpoint、normalization 和输入参考统计；
- R0 的正式预测数组。

以下内容不得纳入：

- 承载 `contract_fingerprint_sha256` 的最终发布 manifest 自身，避免自引用；
- PNG/PDF 图件、日志、训练 history 和终端转储；
- 只供人工查看的 QC CSV/JSON；
- 能从主产物和配置确定性重建的派生报告；
- 临时目录、失败样本的调试附件和缓存。

文件名中含有 `manifest` 不等于发布 manifest。`well_control_manifest.csv`、
`variant_manifest.csv`、`sample_index.csv` 等表本身就是下游读取的数据索引，应作为主产物；
承载顶层指纹并最后写出的 `run_summary.json` 或等价发布 JSON 才需要排除。

若一个主产物是目录，生产者以稳定的逻辑成员名枚举其中真正可消费的文件。不得把整个
目录树不加区分地纳入。HDF5、NPZ、checkpoint 等单文件主件只在发布时顺序读取一次。

### 4.4 生产者与消费者职责

生产者：

1. 写出主产物；
2. 执行全部结构和领域语义校验；
3. 对主产物计算一次临时内容摘要；
4. 结合直接上游指纹和业务配置生成唯一契约指纹；
5. 最后写 manifest 并发布不可变 run。

消费者：

1. 要求显式支持的 schema、成功状态和必要逻辑 ID；
2. 检查文件存在并按其真实格式读取；
3. 执行结构和领域语义校验；
4. 将上游 `contract_fingerprint_sha256` 原样记录为直接输入 provenance；
   此处“记录”特指复用 §4.1 的结构，写入消费者自身发布 manifest 的
   `input_contracts.<logical-role>.contract_fingerprint_sha256`，并按 §4.2 纳入消费者自身指纹计算；
   不得另建平行的 provenance 哈希字段；
5. **不得**对上游文件重算 SHA-256，不得沿祖先链回读源文件做哈希核验。

64 位小写十六进制格式属于 schema 形状校验，不代表消费者验证过内容。

## 5. SHA-256 不能替代的校验

删除运行时哈希后，下列校验必须保留或加强：

- CSV：必需列、唯一键、显式状态、合法枚举、1:1 联接和路径字段语义；
- LAS：曲线 mnemonic、单位、正值约束、采样轴及缺失值传播；
- NPZ/HDF5：精确 key/dataset、dtype、shape、axis order、domain、unit 和 mask 一致性；
- 地震与 LFM：`sample_domain`、`depth_basis`、线性/对数 AI 语义和采样轴一致；
- 模型：`model_id`、架构元数据、输入通道、patch 规格、normalization 和 checkpoint 可加载性；
- variant：显式 `variant_id`、baseline/modifier 身份和主产物元数据一致；
- run：schema、成功状态、不可变发布状态以及直接输入契约角色完整。

当前深度域地震体的 xline 步长是 4。即使错误代码把 xline 线号差直接当数组下标，
所有文件 SHA-256 仍可能完全一致。因此必须读取显式 inline/xline 轴，通过
`SurveyLineGeometry` 或等价几何对象完成线号到数组位置的映射，并校验轴单调性、实际步长
和数组 shape。SHA-256 对此没有任何证明力。

## 6. 迁移前使用审计

### 6.1 用途分类

迁移前仓库中的 SHA-256 混合了四类用途；下表保留作为删改依据：

| 类别 | 代表位置 | 目标 |
|------|----------|------|
| 文件运行时准入 | WellControlSet、LFM resolver、Synthoseis reader、checkpoint loader | 删除，改为语义校验 |
| provenance 记录 | 岩石物理、R0/R1、评估 summary、各种输出清单 | 收敛为单一契约指纹 |
| 组合身份 | `forward_model_inputs_sha256`、benchmark/source hash chain | 由统一契约指纹替代 |
| 确定性算法 | 随机流派生、父 realization 划分、batch 序列指纹 | 保留，不属于文件契约 |

检索覆盖了 `scripts/`、`src/cup/`、`src/ginn_v2/` 及现有 `docs/`。主要集中在：

- `cup.utils.io` 当时的文件/数组哈希工具；
- `cup.well.real_field_controls`、`cup.well.anchor`；
- `cup.seismic.lfm.pipeline`、`cup.seismic.lfm.artifacts` 和体导出元数据；
- `cup.synthetic` 的 calibration、generation、core artifacts、reader 和 depth 分支；
- `ginn_v2` 的数据划分、训练、real-delta、真实工区加载与入口脚本；
- 岩石物理、benchmark 评估、R0 和 R1 脚本。

### 6.2 工作流边界矩阵

| 边界 | 迁移前 SHA 行为 | 合理性判断 | 已实施状态 |
|------|---------------|------------|----------|
| 01 → 02 → 03 → 04 → 05 | 主链基本不做运行时 SHA；依赖 CSV、LAS、TDT 和状态语义 | 合理，也证明 SHA 不是步骤契约的必要条件 | 各成功 run 发布一个指纹；消费者只做语义校验 |
| 井轨迹 → 04 | 无 SHA 准入，按井名、坐标和轨迹字段联接 | 合理；风险是坐标语义而非字节替换 | 发布单一轨迹契约指纹，保留空间语义校验 |
| 深度 Step 3 → Step 4 → Step 5 | 当时无正式哈希链 | 合理；应优先校验 depth/TVDSS、时间子波和深度平移语义 | 三个 run 各自发布一个指纹，Step 5 只记录直接输入 |
| Step 4/5 → 正演可观测性 | 当时不以 SHA 作为输入准入，按井、子波场景、空间簇和频率证据联接 | 合理；主要风险是场景联接和窗口语义 | 可观测性 run 发布一个指纹，只记录直接的 Step 4/5 输入 |
| Step 3 → 岩石物理 | 读取 LAS 时计算 `las_sha256`，并给 inventory、关系、图件等记录哈希 | 多数只是分散 provenance；图件哈希没有契约价值 | 岩石物理 run 发布一个指纹，QC/图件全部排除 |
| Step 4/5/可观测性 → 时间域 calibration | calibration 保存多份 `source_hashes`，generate 再逐文件比较 | 重复读取早期产物；路径和哈希链耦合，语义校验更重要 | calibration 只记录直接上游契约指纹；generate 不重算 |
| 岩石物理 + 深度 Step 5 → 深度 calibration | 严格核验 wavelet、AI–Vp 关系、inventory，并为 shifted LAS 等逐文件哈希 | 过度；真正需要的是关系单位、正速度、LAS 曲线和 domain/basis | 直接上游指纹 + calibration 自身指纹；移除逐文件准入 |
| calibration → Synthoseis generate | 使用 `forward_model_inputs_sha256`、`impedance_calibration_sha256` 和来源哈希锁链 | 多套身份字段表达同一事实，容易漂移 | generate 记录 calibration 和其他直接输入契约指纹 |
| benchmark → reader/evaluate/train | manifest 校验全部输出文件；reader 又遍历 HDF5 dataset 重算数组 SHA | 整文件与逐 dataset 双重全量读取，成本最高且与 schema 校验重复 | 一个 benchmark 指纹；删除 `files` 哈希门和 dataset `sha256` 属性 |
| Step 1/4/5 → Step 6 | Step 6 记录 source summary、LAS、transform、inventory、地震等哈希 | 可作溯源，但下游再次回读早期源文件使契约不自洽 | Step 6 主 manifest + 成功井 NPZ 形成一个 WellControlSet 指纹 |
| Step 6 loader | 校验 manifest、每井 NPZ，并递归校验 NPZ metadata 中的早期来源文件 | 典型过度校验；历史源文件移动也会让冻结产物失效 | 只校验 NPZ 结构、轴、单位、mask 和元数据；不哈希源文件 |
| Step 6 → Step 7 | Step 7 先触发完整 Step 6 哈希校验，又比较目标地震哈希 | 重复且不能证明地震域/几何正确 | Step 7 run 只记录直接 WellControlSet 指纹，严格校验几何 |
| Step 7 variant → real-delta/R0 | 校验地震、层位、manifest、LFM、summary、sidecar、body、Step 6 manifest 和 summary 哈希链 | 链条最深；一个来源文件变化会让已发布 variant 无法消费 | 每个可选 variant 发布自己的指纹；消费者只读所选 variant 并做语义校验 |
| benchmark → GINN-v2 train/eval | 模型 manifest 再记录 benchmark 三文件、patch index、normalization、日志和 history 哈希 | 主输入可追溯有价值，但逐文件字段和日志哈希没有准入价值 | 模型 run 记录 benchmark 直接指纹；模型自身发布单一指纹 |
| 模型 run → R0 | checkpoint 做运行时 SHA 校验；report card 有哈希字段但部分路径只检查摘要格式 | 策略不一致；格式正确的摘要并未证明报告内容 | checkpoint 纳入模型指纹；R0 校验 checkpoint 格式/模型语义但不重算 SHA |
| Step 7 + Step 6 + 模型 + Step 5 → R0 | R0 继续校验地震/LFM 链，并记录大量 `source_file_sha256` | 与不可变直接输入契约重复 | 只记录 LFM variant、显式消费的 WellControlSet、模型和子波等直接契约指纹 |
| R0 → R1 | R1 记录 `zero_shot_summary_sha256` 和 wavelet hash，主要用于 summary | 没有形成一致的准入策略 | R1 记录 R0 和子波直接契约指纹，不新增逐文件 SHA |

### 6.3 HDF5 特例

迁移前 Synthoseis 同时存在：

1. benchmark manifest 中的整个 HDF5 文件 SHA；
2. 每个 dataset 的 `sha256` 属性；
3. reader 构造时遍历全部 dataset、读取完整数组并重算哈希；
4. 同一 reader 随后再执行 shape、axis、domain 等语义检查。

现行实现已删除第 2、3 项，并用 benchmark 顶层契约指纹替代第 1 项。HDF5 reader 仍须校验：

- 文件可打开、schema/status 可消费；
- dataset 路径存在；
- dtype、shape、axis path/order、sample domain 和单位正确；
- sample index 引用的 group/dataset 存在；
- 高分辨率轴与模型轴嵌套关系正确；
- target、seismic、prior 与 mask 的 N 点契约一致。

## 7. 明确保留与明确删除

### 7.1 保留

- 统一的 `contract_fingerprint_sha256` 生产者端计算；
- 直接上游指纹的 provenance 记录；
- `hashlib.sha256` 用于确定性随机流、稳定 split 或 batch 序列审计；
- 用户显式执行的离线审计工具；离线审计不得成为正常工作流的隐式步骤；
- 外部下载在导入时与发布者提供 checksum 的一次性比较。

### 7.2 删除或迁移

- `las_sha256`、`source_*_sha256`、`well_npz_sha256` 等逐文件 CSV 扩散字段；
- `target_seismic_sha256`、`lfm_sha256`、`variant_summary_sha256` 等运行时准入链；
- `forward_model_inputs_sha256`、`benchmark_hashes`、`source_hashes` 等并行组合身份；
- checkpoint、report card、normalization、prediction 和 R0/R1 summary 的零散哈希字段；
- 原始 YAML 的文件哈希，以及高分辨率子波、正演滤波器、抗混叠 taps、单个合成数组等
  运行内数组哈希；其业务语义进入 resolved config 或正式主产物，不另建 SHA 字段；
- PNG、QC CSV、日志和训练 history 的哈希；
- benchmark 评估结果中 `benchmark_files`、`output_hashes` 一类重复来源/输出哈希表；
- benchmark `files` 的全输出哈希表；
- HDF5 dataset `sha256` 属性与 reader 的数组全量重算。

字段删除应以各生产者 schema 升级为边界。不要保留空字符串占位，也不要在读取端猜测旧字段。

## 8. 已执行的迁移顺序

1. **不可变 run**：先让 01–05 及所有旁路统一使用新目录发布，拒绝覆盖成功 run。
2. **公共指纹生成器**：实现规范 JSON、业务配置过滤、稳定逻辑名称和一次性文件摘要。
3. **生产者发布**：为每类契约升级 schema，写 `contract_fingerprint_sha256` 和直接
   `input_contracts`；manifest 最后写出。
4. **消费者瘦身**：删除 `sha256_file(...) != expected`、递归来源哈希校验和逐文件必需列；
   保留并补齐语义校验。
5. **HDF5 清理**：停止写 dataset SHA 属性，删除 reader 全量重算和 benchmark 全文件哈希门。
6. **旧字段清理**：删除输出、CSV 契约、guide 和错误信息中的旧 SHA 字段，不提供双读。
7. **重建产物**：旧 schema run 直接拒绝，按依赖顺序重新生成，不自动升级。

## 9. 实施验收

迁移测试与静态审计至少验证：

- 同一语义配置和主产物在不同输出路径/时间戳下得到相同指纹；
- 修改业务配置、直接上游指纹或任一主产物字节都会改变指纹；
- 修改日志、图件、QC 表、设备或输出路径不会改变指纹；
- 消费者加载上游时不调用文件/数组 SHA 工具；
- 删除旧逐文件 SHA 字段后，新 schema 能完整运行；旧 schema 明确失败；
- 损坏 CSV 列、NPZ key、HDF5 shape、单位、domain、depth basis 或 mask 时仍立即失败；
- 使用 xline 步长 4 的体数据测试线号到数组位置的映射，确保不存在 1 步长假设；
- benchmark reader 不再为了哈希遍历并读取全部 HDF5 dataset；
- SHA 驱动的随机流、split 和 batch 序列在迁移前后保持确定性。

完成迁移后，仓库级检索到的 SHA 使用应只落入三组：公共契约指纹生成、确定性算法、
显式离线/外部 checksum 审计。任何新增的内部运行时文件哈希比较都视为违反本规范。
