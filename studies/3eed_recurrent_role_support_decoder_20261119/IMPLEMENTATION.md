# 冻结特征上的角色、实例支持与框解码

训练与评价使用 3EED 的现有 CUDA 环境。将本目录放在项目的 `studies/` 下，保留已有注册和源模型。公开包提供新增代码、配置和结果。运行依赖本地许可数据、第三方原实现、历史缓存和三份原 checkpoint。

| 程序 | 输入与输出 |
|---|---|
| `prepare.py` | 根据已注册的源 TRAIN 清单构建自然描述和双端可核验的合成关系，按录制留出。 |
| `evidence.py` | 冻结 3EED 骨干、投影和已有图像编码器。点坐标核对后以标定和可见性绑定 RGB，保留原候选作为初始化。 |
| `decoder.py`、`train.py` | 每个角色四个槽，共享更新三次。读取全部 seed，训练主体与已知参照的支持和连续框。每个源和对照训练固定 12 轮。 |
| `predict.py` | 锁定十二个已训练状态。复用同点集匹配的 R165 先验和 R158＋PCA 对照，在普通 Validation 上生成冻结预测。完成六方向后封存。 |
| `score.py`、`independent_audit.py` | 在封存后读取真值，主评分使用 float64 有向框 IoU，再用传感器坐标下的独立凸体求交复核阈值与配对计数。 |
| `diagnostics.py` | 汇总源域录制留出、支持质量和以前已有的目标身份核验。 |
| `infer.py` | 对给定单帧、标定和描述运行已训练完整解码器。输出目标 LiDAR 坐标中的连续七维框。 |

使用项目既有单卡互斥锁执行提取、训练和预测。下例里的 `python` 指项目的现有 CUDA 环境。训练阶段预计在本机 tmpfs 使用约 11 GB 冻结特征。端点、输入封存和训练状态写在本目录的私有输出目录。

```bash
python check_implementation.py
python prepare.py waymo drone quad
python train.py waymo drone quad
python predict.py
python score.py
python independent_audit.py
python diagnostics.py
```

独立帧入口需要源平台、目标平台、点文件、RGB、相机标定和文本。Waymo 图像方向通过 `--view` 指定。默认加载已训练的对应平台完整解码器。

```bash
python infer.py --source waymo --target drone \
  --points POINT_FILE --rgb IMAGE_FILE --calibration CAMERA_JSON \
  --text "the target to ground"
```

R158＋PCA 的成绩使用与新臂逐条配对的现有冻结端点。固定候选消融的监督与完整臂一致。它输出原候选中心、尺寸和既有 PCA 朝向，两个臂的训练权重逐张量一致。关闭反馈同时移除角色间消息、成对打分以及预测框反馈，保留同一输入文本、全局读取范围、初始化与监督。关闭 RGB 仅去除绑定到点 seed 的图像描述子，保留相机坐标几何。
