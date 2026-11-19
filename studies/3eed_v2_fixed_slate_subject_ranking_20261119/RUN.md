# R182 复现

使用 [v1 运行环境](../../v1/RUN.md)、[R180 冻结检索器](../3eed_v2_subject_retrieval_20261119/RUN.md) 和 [R181 原始短名单及分数封存](../3eed_v2_readout_calibration_20261119/RUN.md)。将本研究的代码放在项目工作区的 `studies/3eed_v2_fixed_slate_subject_ranking_20261119`。需要项目本地的源域清单、输入、标注及模型资产。

从项目根目录执行。令 `study` 为本研究目录。

```bash
flock research/GPU_RESOURCE.lock python "$study/capture182.py" source-smoke
flock research/GPU_RESOURCE.lock python "$study/capture182.py" target-smoke
flock research/GPU_RESOURCE.lock python "$study/capture182.py" source
python "$study/fit182.py"
flock research/GPU_RESOURCE.lock python "$study/capture182.py" target
python "$study/replay182.py" predict
python "$study/replay182.py" score
flock research/GPU_RESOURCE.lock python "$study/verify182.py"
```

冻结特征经 GPU 提取，排序头在 CPU 拟合。训练、权重选择和测试预测各有封存。测试预测之后，评分步骤读取 R181 对同一批原框独立复核的 IoU。公开版本提供实现、精确协议、汇总与匿名计分。模型参数、原始输入、实例标注与特征由本地环境生成。

用已拟合的排序器运行单帧实验入口。

```bash
python "$study/ground182.py" --source waymo --target drone \
  --points "$lidar" --rgb "$image" --calibration "$camera_json" \
  --text 'the vehicle to the left of the pedestrian' --arm rank_joint
```

接口按 R180 的固定规则产生 K=16 并保留 v1 原赢家。结果框使用目标 LiDAR 坐标，平移与尺寸单位为米，偏航单位为弧度。
