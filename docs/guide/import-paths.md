# 导入路径

本页用于统一团队导入方式，避免在空初始化子包中迷路。

## 推荐导入

```python
import wtie.modeling as modeling
import wtie.learning as learning
from wtie.processing import grid
from wtie.optimize import tie, autotie
from wtie.utils import viz
```

## 说明

- 顶层入口是首选，便于后续统一维护
- 若某个子包初始化文件未导出对象，可直接从模块路径导入
- 复杂流程建议在 Notebook 或脚本中封装二次入口
