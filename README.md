# Qwen 单帧三维指代

当前主方法是**只缩顶部**（`ray_equivalent_height_only`）。它使用冻结的 Qwen3-VL-4B-Instruct 二维定位和三维框，结合单帧 LiDAR 与相机标定。四尺度点云区域和双姿态射线评分给出初始框。随后只收缩顶面，保留原框内的实测支持点以及用于选框的射线命中状态和首交深度。 [主方法登记](QWEN_CURRENT_METHOD.json) · [推导与消融](studies/3eed_qwen_extent_height_only_20261122/REPORT_ZH.md)

## 推理入口

在已配置 3EED 数据桥和几何依赖的研究环境中，从仓库根目录导入几何入口。将帧记录 `frame` 与已按固定协议解析的 Qwen 回答 `replies` 传给 `predict`。

```python
import sys

sys.path.insert(0, "studies/3eed_qwen_extent_height_only_20261122")
from live import predict

prediction = predict(frame, replies)
```

有效预测的 `prediction["box"]` 为 LiDAR 坐标系下的 `[x, y, z, length, width, height, yaw]`。`extent_corrected` 记录本条是否收缩了顶面。三轴消融使用同一入口并传入 `height_only=False`。 [入口源码](studies/3eed_qwen_extent_height_only_20261122/live.py) · [Qwen 回答协议](studies/3eed_qwen_full_ordinary_validation_20261121/PROTOCOL.json) · [复现命令](studies/3eed_qwen_extent_height_only_20261122/REPORT_ZH.md)

## 普通 Validation

沿用已曝光的 9,310 条 3EED 普通 Validation 描述，每条描述评测一次。汇总按三个目标平台等权。

| 目标平台 | 描述数 | Acc@25 | Acc@50 | mIoU |
|---|---:|---:|---:|---:|
| Waymo | 3,782 | 80.49% | 59.04% | 50.20% |
| Drone | 2,792 | 34.20% | 25.97% | 23.38% |
| Quad | 2,736 | 48.32% | 26.68% | 28.06% |
| **三平台等权** | **9,310** | **54.34%** | **37.23%** | **33.88%** |

主方法相对同批冻结 `C_ray_simple` 的 Acc@50 提高 **2.82 个百分点**，相对三轴收缩的 Acc@50 低 **0.24 个百分点**。两种收缩入口均在 9,310 条上逐框复现封存预测。 [逐平台与录制结果](studies/3eed_qwen_extent_height_only_20261122/results/metrics.csv) · [配对结果](studies/3eed_qwen_extent_height_only_20261122/results/paired.csv) · [入口复放](studies/3eed_qwen_extent_height_only_20261122/results/live_replay.json)

## 论文实验

在与普通 Validation 的录制、RGB 和 LiDAR 均分离的 **1,536 条**普通描述上，主方法的 Acc@25、Acc@50 和 mIoU 分别为 **58.98%、34.18%、33.73%**。冻结 `C_ray_simple` 的 Acc@50 为 31.51%，配对救回 48 条、误改 7 条。两批数据均有项目研发曝光记录。 [双队列实验](studies/3eed_qwen_height_only_paper_20261123/REPORT_ZH.md) · [实验图](studies/3eed_qwen_height_only_paper_20261123/results/paper_figure.pdf) · [独立复算](studies/3eed_qwen_height_only_paper_20261123/results/independent_recount.json)
