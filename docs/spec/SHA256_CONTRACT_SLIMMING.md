# SHA-256 Contract Slimming

## 1. 固定职责

SHA-256 fingerprint 用于 provenance，不用于 consumer 侧重复证明文件内容。

producer 在不可变 contract 发布完成时，可以对 primary artifacts 和直接上游
contract identities 计算一次：

```text
contract_fingerprint_sha256
```

consumer：

- 要求上游 contract 已成功发布；
- 校验 schema、status 和科研语义字段；
- 读取并记录直接上游 fingerprint；
- 不重新读取文件计算 SHA-256；
- 不比较两个已保存 fingerprint 并因差异拒绝输入。

## 2. 必须保留的严格校验

以下检查不属于过度防御：

- artifact type 与 schema version；
- publication status；
- required IDs 与主键唯一性；
- sample domain、unit 和 depth basis；
- axis 单调性、间隔、嵌套与支持；
- array shape、dtype 和 finite values；
- mask shape 与 support；
- survey geometry 和米制横向距离；
- time/depth forward 所需的 domain extras。

这些语义不一致时直接失败，不提供 fallback。

## 3. 允许继续使用 SHA 的场景

- producer publication fingerprint；
- 稳定随机流和 augmentation identity；
- parent split 的稳定随机顺序；
- 外部下载文件的独立 checksum；
- 用户明确要求的文件完整性校验。

稳定随机 identity 不是 artifact admission gate。

## 4. 禁止模式

```text
consumer recomputes upstream file SHA
stored fingerprint A != stored fingerprint B → reject
checkpoint file SHA included in evaluation admission plan
split manifest recomputes and verifies its own stored fingerprint
```

直接上游 provenance 可以原样写入新 manifest，但不得被提升为跨运行一致性证明。

