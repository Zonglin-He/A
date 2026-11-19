# R179 运行方式

使用 [v1 的运行环境](../../v1/RUN.md) 和前一项 [实例头实验的本地资产](../3eed_v1_supervised_instance_grounding_20261119/RUN.md)。保留前一实验的源域特征封存、修框头权重和同一普通 TRAIN 面板的六方向预测。将公开的 `own_code` 内四个脚本放在与前一实验并列的目录 `studies/3eed_v2_fixed_instance_geometry_20261119`，把 `PROTOCOL.json` 放到同一目录。代码按协议检查源域、目标预测和 v1 资产的 SHA-256。工作区的 `ground.py` 需要沿用前一实验的冻结特征捕获补丁。

在 CUDA 环境中设置 `study` 为上面的研究目录，在项目工作区依序执行。

```bash
flock research/GPU_RESOURCE.lock python "$study/run179.py" fit
python "$study/run179.py" predict
python "$study/run179.py" score
flock research/GPU_RESOURCE.lock python "$study/verify179.py"
```

第一步从缓存提取同一源域胜出候选的两个框，拟合并封存三个源域的相对质量模型及各阈值。第二步使用旧封存几何和同一批点云在 CPU 上决定采用哪个框，写入新预测封存。第三步读取标注并运行两个有向 IoU 实现。第四步抽样核验真实推理入口。拟合权重、点支持特征、内部候选及私有清单留在本地。

## 单帧实验入口

```bash
python "$study/ground179.py" --source waymo --target drone \
  --points "$lidar" --rgb "$camera_image" --calibration "$camera_json" \
  --text 'the vehicle to the left of the pedestrian' \
  --selector physical_pair
```

输出使用目标 LiDAR 坐标，包含七维框、所用几何及支持是否足够。实验入口先复用 v1 给出的候选身份，只有相对质量超过对应源域阈值且支持充分时返回修正后的框。
