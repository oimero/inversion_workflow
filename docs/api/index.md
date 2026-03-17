# API 总览

文档按命名空间组织，优先保证后续可扩展性。

## 顶层公开入口

`wtie` 顶层入口定义在包初始化文件中，首批重点关注以下对象：

- wtie.modeling
- wtie.learning
- wtie.processing.grid
- wtie.optimize.tie
- wtie.optimize.autotie
- wtie.utils.viz

## 子包导航

- `api/wtie/` 下是当前首个命名空间
- 新增包时，建议按 `api/<package_name>/` 平行扩展
- 当前子模块页：
	- `wtie.modeling`
	- `wtie.learning`
	- `wtie.processing`
	- `wtie.optimize`
	- `wtie.utils`
