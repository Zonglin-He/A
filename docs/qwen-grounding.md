# Qwen 单帧三维指代

最终论文方法为 **Minimal-D（Bind–Project–Select）**。冻结 Qwen3-VL 两次读出，四尺度绑定物理支持。统一轴区间算子恢复两个姿态的框，以首次二维投影排序，仅在平局时用射线代价。运行结果继续使用 `ray_equivalent_extent_2d_rank` 标识。 [主方法登记](../QWEN_CURRENT_METHOD.json) · [论文稿](../studies/3eed_minimal_pipeline_20261128/MANUSCRIPT_ZH.md) · [等价性与消融](../studies/3eed_minimal_pipeline_20261128/REPORT_ZH.md)

## 推理入口

使用 [3EED 恢复入口](../tools/3eed_recovery/README.md) 准备样本、运行冻结 Qwen 并对封存预测评分。已有缓存回复时，可从仓库根目录装载正式几何入口。

```python
from pathlib import Path
import sys

sys.path.insert(0, "tools/3eed_recovery")
from run_qwen_3eed import REGISTRATION, read, registered_predictor, source_paths

source_paths(Path("../assets/upstream-3eed"))
predict = registered_predictor(read(REGISTRATION))
prediction = predict(frame, replies)
```

将原 manifest 读取的 `frame` 与冻结 Qwen 的 `replies` 交给 `predict`。有效预测的 `prediction["box"]` 为 LiDAR 坐标系下的 `[x, y, z, length, width, height, yaw]`。`extent_corrected` 表示最终选中姿态的框是否沿任一轴收缩。 [锁定入口](../studies/3eed_minimal_pipeline_20261128/paper_entry.py) · [Qwen 回答协议](../studies/3eed_qwen_full_ordinary_validation_20261121/PROTOCOL.json)

## 当前论文主表

每格为 mIoU / Acc@50。665 条开发查询含 576 条 3EED 和 89 条 KITTI Tracking，外部子集来自 Mono3DRefer 的 18 段录制。两套面板均已参与项目评估。

| 方法 | 开发面板 665 条 | Mono3DRefer 118 条 |
|---|---:|---:|
| 历史只缩顶部＋首次二维排序 | 34.296% / 35.038% | 26.872% / 27.966% |
| Minimal-D | **34.312% / 35.188%** | **26.904% / 27.966%** |

Minimal-D 与原三轴入口在 873 条缓存查询上逐项相等。开发面板上的三轴配对增量为 +0.016 pp mIoU，95% CI [−0.052, +0.085]。当前得分来自历史曝光面板。 [原 2×2 结果](../studies/3eed_extent_2d_factorial_20261128/results.json) · [原三轴登记](../studies/3eed_extent_2d_factorial_20261128/REGISTRATION_EXTENT_2D.json) · [冻结后评估](../studies/3eed_minimal_pipeline_20261128/FINAL_EVALUATION_ZH.md)

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
