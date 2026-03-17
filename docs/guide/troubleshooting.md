# 常见问题

## Q1: mkdocs serve 失败，提示找不到命令

先安装文档依赖：

```bash
pip install -r requirements-docs.txt
```

## Q2: API 页面为空或内容很少

可能原因：

- 模块或函数缺少 docstring
- 导入路径写错
- 未包含目标模块页面

建议先确认 API 页面里的模块路径是否正确。

## Q3: 本地中文显示异常

建议使用 UTF-8 终端与浏览器，必要时重启 mkdocs serve。
