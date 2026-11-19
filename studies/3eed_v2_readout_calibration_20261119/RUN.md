# R181 复现

沿用 [v1 运行环境](../../v1/RUN.md)、[R180 本地资产](../3eed_v2_subject_retrieval_20261119/RUN.md) 和 [源域特征封存](../3eed_v1_supervised_instance_grounding_20261119/RUN.md)。将本研究的 `own_code` 和 `PROTOCOL.json` 放在项目工作区的 `studies/3eed_v2_readout_calibration_20261119`。使用 R180 的冻结主体检索器、当时的 K=16 目标候选封存、v1 的原始几何和参数。

从根目录运行，令 `study` 为本研究目录。首轮固定的目标分数完成封存后运行诊断，校准的训练和选参数来自各自源平台。输出校准预测后才进行完整终端计分。

```bash
flock research/GPU_RESOURCE.lock python "$study/capture181.py" smoke
flock research/GPU_RESOURCE.lock python "$study/capture181.py" target
python "$study/score181.py" diagnose
flock research/GPU_RESOURCE.lock python "$study/capture181.py" source
python "$study/calibrate181.py"
python "$study/score181.py" predict
python "$study/score181.py" score
flock research/GPU_RESOURCE.lock python "$study/verify181.py"
```

冻结分数、测试案例私有的详细分解、模型参数、输入与标注保存在本地输出中。公开结果提供各方向指标和错误变化、诊断统计、标定汇总、校验哈希及去关联的逐例指标。

## 单次定位的实验入口

```bash
python "$study/ground181.py" --source waymo --target drone \
  --points "$lidar" --rgb "$camera_image" --calibration "$camera_json" \
  --text 'the vehicle to the left of the pedestrian' --arm calibrated_joint
```

运行时对单个输入按 R180 的 K=16 原规则生成主体短名单，保留 v1 原赢家作为候选，然后运行指定的冻结或校准读出。返回选中的原始七维框和原赢家是否被保留。
