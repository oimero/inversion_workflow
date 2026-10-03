# 工作流总览

```mermaid
flowchart TB
    S1["01 井资产盘点"] --> S2["02 LAS 曲线筛选与导出"] --> S3["03 测井预处理"] --> S4["04 井震自动标定"] --> S5["05 全局共识子波生成"] --> S6["06 真实工区井控数据集"] --> S7["07 真实工区低频模型"]

    WT["旁路 井轨迹 QC"]
    S1 -.-> WT -.-> S4

    RP["旁路 岩石物理分析"]
    S3 -.-> RP
```

## 配置文件

| 步骤 | 配置文件 |
|------|---------|
| 01–07 | `experiments/common/common.yaml` |
| 07 · modifier 实验 | `experiments/real_field_lfm/real_field_lfm.yaml` |
| 旁路 · well_trajectory | `experiments/common/common.yaml` |
| 旁路 · rock_physics_analysis | `experiments/common/common.yaml` |
| 旁路 · synthoseis_lite | `experiments/synthoseis_lite/synthoseis_lite.yaml` |
| GINN V2 body 反演 | `experiments/ginn_v2/ginn_v2.yaml` |

## 深度域工作流

深度域是一次性处理路径，Step 1–3、6 与时间域共享。
Step 4/5 使用独立脚本，详见
[深度域工作流](guide/depth-domain-workflow.md)。
