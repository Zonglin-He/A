# 单帧跨平台 3D 指代定位

给定目标平台的一帧点云、同步 RGB、相机标定和一句英文描述，程序返回目标的七维框。Waymo → Drone 使用已有的 Waymo epoch100 和同源的冻结 RGB–LiDAR 融合、语言匹配及竞争头。Drone 在当地独立采集观测。三个源平台各有一套权重，六个方向共用一个入口。[源权重校验清单](own_code/crossplatform/single_frame_sources.json)

## 运行

在本项目根目录运行，替换示例中的三个输入文件。程序的标准输出是一行 JSON。

```bash
crossplatform/env/bin/python crossplatform/ground_single_frame.py \
  --source waymo --target drone \
  --points observation/points.bin \
  --rgb observation/image.jpg \
  --calibration observation/camera.json \
  --text "the white pickup next to the streetlight"
```

在 Python 中可以复用加载好的源模型处理多条观测。每条观测单独解析文字并重新生成候选。

```python
from crossplatform.ground_single_frame import SingleFrameGrounder

model = SingleFrameGrounder("waymo")
box = model.predict(
    target="drone",
    points="observation/points.bin",
    rgb="observation/image.jpg",
    calibration="observation/camera.json",
    text="the white pickup next to the streetlight",
)
```

| 输入或输出 | 约定 |
|---|---|
| 点云 | 原始目标平台坐标，单位米。支持有限的 XYZ 数组（`.npy`，至少三列）和每点四个 float32 的 XYZI 二进制文件。模型使用 XYZ。每帧采样 16,384 点。不足时有放回采样。默认种子 0，可用 `--seed` 指定。 |
| RGB 与标定 | 使用这帧 RGB 对应的 `image_intrinsic`、`image_extrinsic`，以及可选 `camera_distortion`。按既有 3EED 数据约定，Drone、Quad 的外参从原始点云到相机，Waymo 使用数据集的相机到车辆外参及其相机轴变换。缺畸变参数时按零处理。 |
| Waymo 目标相机 | 前视为默认。左前、右前、左侧、右侧图像分别传 `--waymo-view FL`、`FR`、`SL`、`SR`。选择采集这张 RGB 的相机。 |
| 文字 | 使用原模型的英文词表和已有的本地 Stanza 英文解析器。含模型固定后缀的最长输入为 256 个 token。 |
| 输出 | `box` 的顺序为 `[x, y, z, length, width, height, yaw]`。中心和尺寸单位米，yaw 单位弧度，坐标系是输入点云的原始目标平台坐标系。长轴方向以 π 为周期。 |

## 选框与朝向

完整描述参与 3EED 的点云—语言计算。冻结读出将主体词的原生匹配与已训练的局部实例竞争分数结合，生成 256 个候选的概率。R158 先用候选之间的三维框重叠估计重复程度，再选与校正后概率分布期望重叠最大的原始候选。周围出现许多近似框时，同一片区域获得的总票数会受限。[R158 实现](own_code/research/round158/grounding158.py)

PCA 使用生成该批候选的同一次点云采样。框内至少有 8 个不同点，预测长宽比至少为 1.5，且水平点分布具有可用方差时，使用点集的水平长轴。其余情况沿用 3EED 模型坐标系的原轴向。尺寸和中心来自所选候选。输出前还原 Drone 的高度偏移或 Waymo 的视角旋转。[朝向实现](own_code/crossplatform/replication36_surface/common.py)

当前项目环境提供带 CUDA 点算子的 3EED、RoBERTa、本地英文解析器，以及已有的三套源权重。每次实例化只加载指定源的 3EED 和匹配模块，目标端更新次数为 0。已配准的目标平台第二帧可用于单独的研究对照。[评估与负例](REPORT_ZH.md)

迁移到另一台机器时，需要带已编译点云算子的 3EED 工作环境及已有源权重。此处核对的上游版本为 `843a901f`。沿用项目的 `crossplatform/round3/p0/upstream` 运行代码、`crossplatform/upstream/data/roberta_base` 语言资产，以及 `crossplatform/round20` 下的 Stanza 运行库与本机模型路径配置。公开包提供第一方推理代码与源权重校验值。运行包复核使用单套 Waymo 权重及本机依赖，从空目录完成了 Waymo → Drone 出框。[上游 3EED](https://github.com/worldbench/3EED/tree/843a901f5367a6be5cc484b97c6ecad462961dfa)

公开包的 own_code/ 保留工作区相对结构。将其中的代码置于配有所需依赖的 3EED 工作区，源权重按清单保存在本地。命令行已在仅挂载该源权重、官方运行库及文本资产的独立目录复核。[运行包复核](BUNDLE_REPLAY.json)
