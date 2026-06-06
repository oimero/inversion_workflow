# 脚本风格

本文是对重构后代码组织方式的极简归纳。如何在 `scripts/` 下写一个合规的新脚本、每个公共函数去哪个模块找、`src/` 下每个文件管什么——都可以在这里查到。

---

## 脚本骨架

每个时间域主链脚本统一为以下结构，从上到下：

```text
1. 英文 docstring（消费什么、输出什么、怎么运行）
2. from __future__ import annotations
3. import argparse / logging / sys / pathlib / 三方库
4. Bootstrap block（SCRIPT_DIR → REPO_ROOT → SRC_DIR，把 src 加入 sys.path）
5. from src import ...（库函数和 dataclass）
6. DEFAULT_CONFIG（脚本级默认值字典）
7. parse_args()
8. _resolve_inputs() / _resolve_output_dir()（路径与上游 run 发现）
9. main()（构建上下文 → 调 src 函数 → 写 summary → 打印一行简短日志）
10. if __name__ == "__main__": main()
```

### 具体约束

- **CLI**：`parse_args()` 只解析 `--config`、`--output-dir` 和可选单井过滤参数（如 `--well`）。
- **配置合并**：用 `cup.utils.config.deep_merge_dict(DEFAULT_CONFIG, cfg.get("<section>"))` 合并用户配置到默认值。
- **run 发现**：用 `cup.utils.io.latest_run(output_root, prefix, required_file)` 定位上游 run 目录。
- **路径解析**：用 `cup.utils.io.resolve_relative_path` / `repo_relative_path`；可选的配置路径用 `resolve_optional_path`；从上游元数据里取出的 artifact 路径用 `resolve_artifact_path`。
- **日志**：`import logging; logger = logging.getLogger(__name__)`，主流程用 `logger.info()`，只在脚本最后用 `print()` 输出一行简短汇总。
- **图表保存**：简单 `_save_fig()` 留在脚本内；只有带业务语义的图件绘制逻辑才迁入 `cup.well.viz` 或 `cup.seismic.viz`。
- **ginn_train.py**：可保持 dataclass 驱动的极简风格，但必须补齐英文 docstring 和 bootstrap 分隔。

---

## 常用函数去哪儿找

| 需求 | 模块 | 函数 |
|------|------|------|
| 递归合并两个 dict | `cup.utils.config` | `deep_merge_dict` |
| 校验 source_runs.mode | `cup.utils.config` | `require_latest_mode` |
| 定位最新上游 run | `cup.utils.io` | `latest_run` |
| 构造时间戳输出目录 | `cup.utils.io` | `resolve_timestamped_output_dir` |
| 解析脚本级路径 | `cup.utils.io` | `resolve_relative_path` |
| 可移植 repo-relative string | `cup.utils.io` | `repo_relative_path` |
| 可选路径（支持 none/null） | `cup.utils.io` | `resolve_optional_path` |
| 上游元数据 artifact 路径 | `cup.utils.io` | `resolve_artifact_path` |
| 加载 YAML 配置 | `cup.utils.io` | `load_yaml_config` |
| 写 JSON 产物 | `cup.utils.io` | `write_json` |
| JSON-safe 序列化 | `cup.utils.io` | `to_json_compatible` |
| 文件名安全化 | `cup.utils.io` | `sanitize_filename` |
| 打开地震体 | `cup.seismic.survey` | `open_survey` |
| 构造 SEG-Y 选项 | `cup.seismic.survey` | `segy_options_from_config` |
| 井位 → inline/xline | `cup.seismic.geometry` | `resolve_well_line_position` |
| 目标层构建 | `cup.seismic.target_zone` | `TargetZone` |
| 解释层位读取 | `cup.petrel.load` | `import_interpretation_petrel` |
| 井头读取 | `cup.petrel.load` | `import_well_heads_petrel` |
| 井分层读取 | `cup.petrel.load` | `import_well_tops_petrel` |
| 井名规范化 | `cup.well.assets` | `normalize_well_name` |
| 文件 lookup（stem→path） | `cup.well.assets` | `build_file_lookup` |
| 井轨迹解析 | `cup.well.trajectory` | `WellTrajectory.from_petrel_trace` |
| 标准 LAS 加载 | `cup.well.las` | `load_vp_rho_logset_from_standard_las` |
| LAS 曲线扫描 | `cup.well.las` | `scan_las_curves` |
| 曲线分类与 primary | `cup.well.curves` | `classify_curves_by_rules` / `select_primary_curves` |
| 曲线 mnemonic 规则 | `cup.well.mnemonics` | `CURVE_CATEGORY_MNEMONICS` |
| 预处理清洗 | `cup.well.preprocess` | `standardize_curve_unit` / `replace_constant_runs` / `remove_outliers` 等 |
| 时深表加载 | `cup.well.td` | `load_petrel_time_depth_table` / `load_workflow_time_depth_table_csv` |
| 子波加载 | `cup.well.wavelet` | `load_wavelet_csv` / `crop_wavelet_center_energy_normalize` |
| 子波共识搜索 | `cup.well.wavelet_consensus` | `build_wavelet_pca_basis` / `optimize_consensus_wavelet` |
| 井震标定路由 | `cup.well.tie` | `build_tie_plan` / `WellTiePlan` / `evaluate_wavelet_on_well` 等 |
| 井约束点级事实 | `cup.well.constraints` | `build_vertical_point_facts` / `build_deviated_point_facts` / `lowpass_values_on_twt` 等 |
| dynamic gain 计算 | `cup.well.gain` | `positive_ls_gain` / `fit_gain_relationship` / `build_gain_volume` / `write_gain_npz` |
| LFM 时间域建模 | `cup.seismic.lfm_time` | `build_lfm_time_model_from_points` |
| LFM 低通滤波 | `cup.seismic.lfm_time` | `lowpass_twt_log` |
| 沿轨迹取道 | `cup.seismic.trace_sampling` | `build_nearest_trace_sample_plan` / `assemble_nearest_trace_from_plan` |
| GINN 数据加载 | `ginn.data` | `load_lowfreq_model` / `_validate_time_lfm_contract` 等 |
| GINN anchor 读写 | `ginn.anchor` | `build_log_ai_anchor_bundle` / `validate_log_ai_anchor` / `save_log_ai_anchor_npz` |
| GINN 训练 | `ginn.trainer` | `Trainer` |
| GINN 配置 | `ginn.config` | `GINNConfig` |
| enhance 监督包 | `enhance.supervision` | `WellHighSupervisionBundle` / `save_well_high_supervision_npz` |
| 统计工具 | `cup.utils.statistics` | `pearson_r` / `spearman_rho` / `ols_fit` / `radius_connected_components` / `aggregate_cluster_then_global` |
| 滑动窗口 | `cup.utils.raw_trace` | `centered_moving_rms_axis` / `centered_moving_average` / `centered_moving_sum_axis` |
| 类型转换 | `cup.utils.coerce` | `as_bool` / `optional_float` |

## 脚本之间的契约

脚本之间传的是文件，不是内存对象。下游脚本通过 `latest_run` 定位上游输出目录后，必须检查必需文件是否存在并做 schema 校验。每个步骤产出的 CSV 和 NPZ 字段约定见 `docs/concepts/csv-contracts.md`；TWT、TVDSS、inline/xline 等坐标约定见 `docs/concepts/data-and-coordinate-conventions.md`。

修改任何跨步骤的 CSV 列或 NPZ 键之前，必须先更新对应的 contracts 文档。
