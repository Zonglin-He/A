# Qwen 单帧三维指代

论文主方法为 **S5-pass**。冻结 Qwen3-VL 两次读出，以 Minimal-D 为锚生成最多 13 个物理候选，再由只用 Waymo 训练的 Verifier 计算相对几何效用。推理使用一个分数和一个替换阈值。 [论文主方法登记](../QWEN_PAPER_METHOD.json) · [数学方法](../studies/3eed_minimal_verifier_20261128/METHOD_ZH.md) · [配对结果](../studies/3eed_minimal_verifier_20261128/REPORT_ZH.md)

## 推理入口

[3EED 恢复入口](../tools/3eed_recovery/README.md) 的 `predict` 默认运行 `s5_pass`。已有冻结 Qwen 回复时，可从仓库根目录运行下列入口。

```python
from pathlib import Path
import sys

sys.path.insert(0, "tools/3eed_recovery")
from run_qwen_3eed import PAPER_REGISTRATION, read, registered_predictor, source_paths

source_paths(Path("../assets/upstream-3eed"))
predict = registered_predictor(read(PAPER_REGISTRATION))
prediction = predict(frame, replies)
```

`prediction["box"]` 是 LiDAR 坐标系下的 `[x, y, z, length, width, height, yaw]`。`verifier_replaced` 记录是否替换 Minimal-D，`verifier_utility` 记录最高的有效替代候选分数。历史脚本绑定原 [Minimal-D 登记](../QWEN_CURRENT_METHOD.json)，CLI 可用 `--method minimal-d` 回放。

## 当前论文主表

每格为 mIoU / Acc@50。两套均为历史复测。665 条开发查询含 576 条 3EED 和 89 条 KITTI Tracking，Mono 148 条覆盖 35 段录制。

| 方法 | 开发 665 条 | Mono3DRefer 148 条 |
| --- | ---: | ---: |
| Minimal-D | 34.31% / 35.19% | 34.44% / 37.16% |
| Verifier v2 | 37.84% / 39.55% | 39.31% / 42.57% |
| **S5-pass** | **38.10% / 40.15%** | **39.22% / 42.57%** |

S5-pass 相对 v2 的开发 mIoU 和 Acc@50 分别提高 0.26 和 0.60 pp，配对区间均包含零。 [聚合结果](../studies/3eed_minimal_verifier_20261128/results.json)

## 普通 Validation

以下为历史射线加顶缩的结果。

沿用已曝光的 9,310 条 3EED 普通 Validation 描述，每条描述评测一次。汇总按三个目标平台等权。

| 目标平台 | 描述数 | Acc@25 | Acc@50 | mIoU |
|---|---:|---:|---:|---:|
| Waymo | 3,782 | 80.49% | 59.04% | 50.20% |
| Drone | 2,792 | 34.20% | 25.97% | 23.38% |
| Quad | 2,736 | 48.32% | 26.68% | 28.06% |
| **三平台等权** | **9,310** | **54.34%** | **37.23%** | **33.88%** |

历史射线加顶缩相对同批冻结 `C_ray_simple` 的 Acc@50 提高 **2.82 个百分点**，相对三轴收缩的 Acc@50 低 **0.24 个百分点**。两种收缩入口均在 9,310 条上逐框复现封存预测。 [逐平台与录制结果](../studies/3eed_qwen_extent_height_only_20261122/results/metrics.csv) · [配对结果](../studies/3eed_qwen_extent_height_only_20261122/results/paired.csv) · [入口复放](../studies/3eed_qwen_extent_height_only_20261122/results/live_replay.json)

## 论文实验

在与普通 Validation 的录制、RGB 和 LiDAR 均分离的 **1,536 条**普通描述上，历史射线加顶缩的 Acc@25、Acc@50 和 mIoU 分别为 **58.98%、34.18%、33.73%**。冻结 `C_ray_simple` 的 Acc@50 为 31.51%，配对救回 48 条、误改 7 条。两批数据均有项目研发曝光记录。 [双队列实验](../studies/3eed_qwen_height_only_paper_20261123/REPORT_ZH.md) · [实验图](../studies/3eed_qwen_height_only_paper_20261123/results/paper_figure.pdf) · [独立复算](../studies/3eed_qwen_height_only_paper_20261123/results/independent_recount.json)

同框高、同观测的消融中，保留原底面在普通 Validation 上比尽量对称收缩高 **2.12 pp Acc@50**。换用 WildDet3D 的冻结三维输出后，顶缩的配对增量为 **+0.48 pp**。Talk2Car–mini 的 113 条外部复测中，两组 Acc@50 均为 **11.50%**。 [同高消融、三维前端替换与外部复测](../studies/3eed_height_only_transfer_20261123/REPORT_ZH.md) · [补充实验图](../studies/3eed_height_only_transfer_20261123/results/transfer_figure.pdf)

## Talk2Car–mini 调参

同一批 113 条指令上，单目标提示、144 token 和窗口中位深度的外部开发配置达到 **19.47% Acc@50**。相对原提示的同一读出增加 **6.19 个百分点**。完整「只缩顶部」按录制留一选参为 **13.27%**。 [完整调参报告](../studies/3eed_talk2car_parameter_tuning_20261123/REPORT_ZH.md)
