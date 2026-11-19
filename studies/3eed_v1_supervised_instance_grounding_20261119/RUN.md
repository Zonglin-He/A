# 运行复现

沿用 [v1 运行环境](../../v1/RUN.md) 和已登记的三套源平台 3EED 资产。实验代码放在原工作区的 `studies/3eed_v1_supervised_instance_grounding_20261119`，即脚本能从上两级访问 `crossplatform` 和 `research`。将公开的 `own_code/*.py` 复制到该实验目录，工作区的 v1 推理入口需要应用 `own_code/ground_capture.patch` 来提供冻结特征。该补丁为默认 v1 路径保留原输出。

复用前一项候选研究的 `mechanisms.py` 和组件消融的 `score_ablation.py`。它们在本地分别位于 `studies/3eed_v1_retrieval_binding_refinement_20261119` 和 `studies/3eed_v1_component_ablation_20261119`。IoU 数值实现使用工作区的 `research/round6/source_validation_scoring/batch_yaw_iou.py` 和 `tools/obb_iou.py`。 [候选研究依赖](../3eed_v1_retrieval_binding_refinement_20261119/own_code/mechanisms.py) · [评分代码](../../v1/ablation/own_code/score_ablation.py) · [IoU 实现归档](../3eed_counterfactual_binding_consensus_direct_validation_20261116/own_code)

## 拟合和评测

在已配置 CUDA 环境、本地数据、普通目标评测清单及旧版封存预测可用时运行。设置 `study` 为上述实验目录。

```bash
python "$study/scope.py"
python "$study/features.py"
python "$study/train.py"
python "$study/evaluate.py" smoke
python "$study/evaluate.py" predict
python "$study/evaluate.py" score
```

`scope.py` 先按录制隔离源域拟合与选型，并排除目标清单中的物理帧及重复点云。新实例头从相应源平台标注拟合，目标域仅执行冻结推理。目标预测封存写入后，再执行 CPU 评分。可选的盲核验使用 `reference_audit.py prepare` 生成本地图页，录入独立判断后使用 `reference_audit.py score`。

## 单帧入口

```bash
python "$study/ground_instance.py" --source waymo --target drone \
  --points "$lidar" --rgb "$camera_image" --calibration "$camera_json" \
  --text 'the vehicle to the left of the pedestrian' \
  --head-root "$study/weights_PRIVATE" --readout full_pool_joint
```

`--readout` 缺省时使用源域录制选出的读出。命令返回目标 LiDAR 坐标系下的主体和参照七维框。参照不足时 `reference_box` 为 `null`。图像路径用于读取宽高，点云和相机标定提供空间几何。

公开结果包含去关联的标量、分方向汇总和审计校验。精确重放需要按固定协议取得本地清单、源域数据、运行资产和封存文件，并检查各自 SHA-256。

## 保留原框的读出

在同一环境中，可在 Python 使用 `RetainedGeometryGrounder`。初始化和调用参数与 `SourceInstanceGrounder` 相同。主体仍由源域选定策略决定，输出使用该候选的原始几何。

```python
from retained_geometry import RetainedGeometryGrounder

model = RetainedGeometryGrounder("waymo")
prediction = model.predict(
    target="drone", points=lidar_path, rgb=image_path,
    calibration=camera_path, text="the vehicle to the left of the pedestrian",
)
```

已有封存预测时，使用 `geometry_readout_check.py seal` 和 `geometry_readout_check.py score` 复现固定选中候选的 CPU 比较。使用 `verify_retained_geometry.py` 核对六方向的实际推理入口和三份源域留出数据。
