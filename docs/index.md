# 工作流总览

```mermaid
flowchart TB
    subgraph prod["稳定生产链"]
        S1["01 · well_inventory"] --> S2["02 · well_screen"] --> S3["03 · well_preprocess"] --> S4["04 · well_auto_tie"] --> S5["05 · wavelet_generation"]
        WT["旁路 · well_trajectory"] -.-> S4
    end

    prod --> FO["旁路 · forward_observability"]
    S3 --> RP["旁路 · rock_physics_analysis"]

    S1 -.-> S7["07 · real_field_lfm"]
    S4 -.-> S7

    S6 --> SL["旁路 · synthoseis_lite"]
    FO -.-> SL
    SL --> GV["旁路 · ginn_v2"]

    S5 -.-> R0["08 R0 · real_field_zero_shot"]
    S7 --> R0
    GV --> R0

    R0 --> R1["08 R1 · real_field_forward_diagnostic"]
```

## 配置文件

| 步骤 | 配置文件 |
|------|---------|
| 01–05 + well_trajectory | `experiments/common/common.yaml` |
| 旁路 · forward_observability | `experiments/common/common.yaml` |
| 旁路 · rock_physics_analysis | `experiments/common/common.yaml` |
| 旁路 · synthoseis_lite | `experiments/synthoseis_lite/synthoseis_lite.yaml` |
| 旁路 · ginn_v2 | `experiments/ginn_v2/train.yaml` |
| 07 · real_field_lfm | `experiments/common/common.yaml` |
| 08 R0 · real_field_zero_shot | `experiments/common/common.yaml` |
| 08 R1 · real_field_forward_diagnostic | `experiments/common/common.yaml` |

## 深度域工作流

深度域是一次性处理路径，Step 1–3、6 与时间域共享。
Step 4/5 使用独立脚本，详见
[深度域工作流](guide/depth-domain-workflow.md)。
