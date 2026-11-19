# R180 复现

沿用 [v1 的运行环境](../../v1/RUN.md) 以及已有的 [原始候选与源域特征封存](../3eed_v1_supervised_instance_grounding_20261119/RUN.md)。四个参照槽和冻结的多槽读出来自当前 v1 与 [已有对照实现](../3eed_v1_retrieval_binding_refinement_20261119)。工作区的 v1 推理入口需要应用前一实验的冻结特征捕获补丁。

将本实验的 `own_code` 脚本和 `PROTOCOL.json` 放在项目的 `studies/3eed_v2_subject_retrieval_20261119`，与旧研究目录并列。前一实验的源域特征、普通目标清单、候选和原 v1 输出用协议所列 SHA-256 复用。实验只拟合小型主体检索器，所需的权重和私有数据由本地实验生成或读取。

在项目根目录将 `study` 设置为上述研究目录，使用已有 CUDA Python 环境。

```bash
flock research/GPU_RESOURCE.lock python "$study/source180.py" fit
flock research/GPU_RESOURCE.lock python "$study/source180.py" calibrate
flock research/GPU_RESOURCE.lock python "$study/evaluate180.py" smoke
flock research/GPU_RESOURCE.lock python "$study/evaluate180.py" predict
python "$study/score180.py"
flock research/GPU_RESOURCE.lock python "$study/verify180.py"
```

`fit` 使用源域冻结表示和原框的有向 IoU 标签。 `calibrate` 在同源但按录制隔离的留出集上选 K。 `predict` 复算每条冻结 v1 候选池和原输出，封存所有检索列表和终端决策。 `score180.py` 读取目标标签并在 CPU 按有向 3D IoU 验证。 `verify180.py` 检查实时候选输出和封存结果一致。

## 单帧接口

```bash
python "$study/ground180.py" --source waymo --target drone \
  --points "$lidar" --rgb "$camera_image" --calibration "$camera_json" \
  --text 'the vehicle to the left of the pedestrian' --k 8
```

省略 `--k` 时使用对应源域锁定的 K。通过 `--retrieval native` 运行同样预算和候选排序规则的冻结前端分数对照。返回框采用目标 LiDAR 坐标系；所有选项都在原始 A256、现有 PCA 朝向、固定 v1 参照输入下运行。
