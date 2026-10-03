# 常见问题

## 启动文档服务时提示找不到命令

在运行文档服务的同一个 Python 环境中安装文档依赖，再启动服务：

```powershell
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
```

使用指定的本机解释器时，将下面的占位符替换为可执行文件路径：

```powershell
& "<python-executable>" -m pip install -r requirements-docs.txt
& "<python-executable>" -m mkdocs serve
```
