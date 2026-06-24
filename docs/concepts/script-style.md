# 脚本风格

本文只描述当前稳定的前五步和井轨迹旁路。研究阶段的新脚本不得沿用已经删除的
后半程接口。

## 脚本骨架

时间域脚本统一采用以下结构：

```text
1. 英文模块 docstring
2. from __future__ import annotations
3. 标准库与第三方依赖
4. SCRIPT_DIR / REPO_ROOT / SRC_DIR bootstrap
5. cup 与 wtie 导入
6. DEFAULT_CONFIG
7. parse_args()
8. 输入和输出路径解析
9. main()
10. if __name__ == "__main__": main()
```

## 约束

- CLI 只暴露单次运行需要覆盖的参数。
- 顶层工区事实通过 `cup.config.workflow.TimeWorkflowConfig` 解析。
- 步骤默认配置使用 `cup.utils.config.merge_dict_defaults`。
- 路径使用 `cup.utils.io` 中的解析和 repo-relative 工具。
- 带采样轴、单位或 domain 的井曲线和地震道优先使用 `wtie.processing.grid`
  对象或项目 dataclass，不在脚本中长期传递裸 `np.ndarray`。
- 简单保存逻辑可留在脚本内；可复用的业务计算和绘图进入 `src/cup/`。
- 跨步骤 CSV 的字段语义以[核心 CSV 契约](csv-contracts.md)为准。
- 新研究模块应先确定独立契约，再创建包名和脚本编号。

## 稳定能力地图

| 需求 | 模块 | 入口 |
|------|------|------|
| 共享配置 | `cup.config.workflow` | `TimeWorkflowConfig.from_mapping` |
| 配置默认值合并 | `cup.utils.config` | `merge_dict_defaults` |
| YAML、JSON 与路径 | `cup.utils.io` | `load_yaml_config` / `write_json` / `resolve_relative_path` / `repo_relative_path` |
| 类型转换 | `cup.utils.coerce` | `as_bool` / `optional_float` |
| 掩码连续区间 | `cup.utils.masks` | `true_runs` |
| 空间簇与统计 | `cup.utils.statistics` | `radius_connected_components` / `aggregate_cluster_then_global` |
| Petrel 资产读取 | `cup.petrel.load` | 井头、井分层、checkshot 和解释层位读取函数 |
| 打开地震体 | `cup.seismic.survey` | `open_survey` / `segy_options_from_config` |
| 工区几何 | `cup.seismic.geometry` | `SurveyLineGeometry` 及坐标变换工具 |
| 单层位处理 | `cup.seismic.horizon` | `HorizonSurface` / `build_horizon_surface` |
| 沿轨迹取道 | `cup.seismic.trace_sampling` | `build_nearest_trace_sample_plan` / `assemble_nearest_trace_from_plan` |
| 井震与子波 QC | `cup.seismic.viz` | `plot_well_waveform_qc` |
| 子波处理 | `cup.seismic.wavelet` | 加载、裁剪、归一化和频谱属性函数 |
| 共识子波搜索 | `cup.seismic.wavelet_consensus` | `build_wavelet_pca_basis` / `optimize_consensus_wavelet` |
| 前向可观测性 | `cup.seismic.observability` | 离散算子、扰动灵敏度、场景与空间簇证据聚合 |
| 合成基准校准 | `cup.synthetic.calibration` | 背景拟合、三态识别、对象轮廓拟合、层级化收缩 |
| 合成基准生成 | `cup.synthetic.generation` | 随机地质真值、几何事件、探针矩阵、地震变体 |
| 合成基准评估 | `cup.synthetic.metrics` / `cup.synthetic.dataset` | 回归指标、基准消费接口 |
| 合成基准工具 | `cup.synthetic.forward` / `cup.synthetic.lfm` / `cup.synthetic.probes` | 抗混叠正演、LFM 推导、频率探针构建 |
| 合成基准配置 | `cup.synthetic.config` | `parse_synthoseis_config` / `resolve_sources` |
| 井资产 | `cup.well.assets` | `normalize_well_name` / `build_file_lookup` |
| LAS I/O | `cup.well.las` | 标准 LAS 加载、扫描和导出函数 |
| 曲线识别 | `cup.well.curves` / `cup.well.mnemonics` | 分类、primary 选择和 mnemonic 规则 |
| 测井预处理 | `cup.well.preprocess` | 单位标准化、常值段和异常值处理 |
| 缺口处理 | `cup.well.gaps` | `fill_short_joint_gaps` / `prepare_continuous_tie_logs` |
| 时深表 | `cup.well.td` | Petrel 和工作流 TDT 读取、转换与输出 |
| 井轨迹 | `cup.well.trajectory` | `WellTrajectory` 及 TWT 轨迹采样 |
| 井震标定 | `cup.well.tie` | 路由、搜索空间、结果对象和子波评价 |

## 跨步骤契约

脚本之间传递文件，不传递进程内对象。修改任何 CSV 列、路径含义或坐标语义前，
必须同步更新：

- [核心 CSV 契约](csv-contracts.md)
- [数据与单位约定](data-and-coordinate-conventions.md)

当前正式契约覆盖步骤 01 至 05、井轨迹旁路、`forward_observability.py`
研究闸门，以及 `synthoseis_lite.py` calibrate/generate 与
`evaluate_synthoseis_lite.py` 研究闸门。
