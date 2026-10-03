# 脚本风格

## 骨架

脚本统一采用以下结构：

```text
1. 英文模块 docstring
2. from __future__ import annotations
3. 标准库与第三方依赖
4. SCRIPT_DIR / REPO_ROOT / SRC_DIR bootstrap
5. cup 与 wtie 导入
6. parse_args()
7. 输入和输出路径解析
8. main()
9. if __name__ == "__main__": main()
```

其中：

**1. docstring**：描述脚本职责，含 ``Usage::`` 块给出命令行示例。

**4. bootstrap**：统一使用以下模式，不额外插入 `REPO_ROOT` 到 `sys.path`：

```python
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
```

**7. 路径解析**：输出目录解析统一命名为 `_resolve_output_dir()`；输入路径
统一使用 `resolve_relative_path()`。

## 约束

- CLI 只暴露单次运行需要覆盖的参数。
- 顶层工区事实通过 `cup.config.workflow.WorkflowConfig` 解析。
- 路径使用 `cup.utils.io` 中的解析和 repo-relative 工具。
- 运行目录、产物定位和发布状态由 `cup.config.artifacts` 处理；来源选择由 `cup.config.sources` 组织。
- 步骤默认配置优先使用 `dict.setdefault` 或 `dict.get(key, default)`；
  复杂嵌套默认值可用 `merge_dict_defaults`。
- 带采样轴、单位或 domain 的井曲线和地震道优先使用 `wtie.processing.grid`
  对象或项目 dataclass，不在脚本中长期传递裸 `np.ndarray`。
- 简单保存逻辑可留在脚本内；可复用的业务计算和绘图进入 `src/cup/`。

### 契约版本常量

持久化产物的版本标识由负责该产物契约的模块定义。模块只负责一种产物时使用
`SCHEMA_VERSION`；模块负责多种产物时使用带产物名的常量，例如
`MODEL_RUN_SCHEMA_VERSION`。

写入产物、计算契约指纹和读取校验引用同一个版本常量。生产者与消费者分属不同
模块时，版本常量放在 `src/` 下对应领域的契约模块中，由双方导入。版本标识属于
文件契约，不属于运行配置。
