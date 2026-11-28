# Qwen 单帧三维指代

当前主方法为 `ray_equivalent_height_only_2d_rank`。冻结的 Qwen3-VL-4B-Instruct 提供首次二维框与三维先验。原四尺度点云区域产生 PCA 和 Qwen 两个姿态候选，各自保留原保护点和 Selection 首交证据执行顶缩。选择与首次二维框标定投影重叠更大的候选，失效和并列沿用原射线赢家。 [主方法登记](../QWEN_CURRENT_METHOD.json) · [定型报告](../studies/3eed_2d_pose_finalization_20261128/REPORT_ZH.md)

## 推理入口

在已配置 3EED 数据桥和几何依赖的研究环境中，从仓库根目录导入几何入口。将帧记录 `frame` 与已按固定协议解析的 Qwen 回答 `replies` 传给 `predict`。

```python
import sys

sys.path.insert(0, "studies/3eed_qwen_extent_height_only_20261122")
from live_2d import predict

prediction = predict(frame, replies)
```

有效预测的 `prediction["box"]` 为 LiDAR 坐标系下的 `[x, y, z, length, width, height, yaw]`。`extent_corrected` 记录本条是否收缩了顶面。历史入口 `live.predict` 保留旧射线选框和三轴消融。 [当前入口](../studies/3eed_qwen_extent_height_only_20261122/live_2d.py) · [历史入口](../studies/3eed_qwen_extent_height_only_20261122/live.py) · [Qwen 回答协议](../studies/3eed_qwen_full_ordinary_validation_20261121/PROTOCOL.json) · [复现命令](../studies/3eed_qwen_extent_height_only_20261122/REPORT_ZH.md)

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
