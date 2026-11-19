# v1 运行说明

## 环境和本地资产

在具备 CUDA 的 Python 环境中运行。需要已编译点云算子的 3EED、PyTorch、Transformers、NumPy、SciPy、Pillow 及 Stanza 英文依赖解析。发布的代码使用已经准备好的源权重；本仓库提供权重路径及校验值。

| 相对 `--asset-root` 的路径 | 文件或目录 |
|---|---|
| `crossplatform/round3/p0/upstream` | 3EED 原生模型和已编译的 PointNet++ 算子，使用上游提交 `843a901f5367a6be5cc484b97c6ecad462961dfa` 的兼容工作树。 |
| `crossplatform/upstream/data/roberta_base` | 本地 RoBERTa 权重及 tokenizer。 |
| `crossplatform/round20/runtime` | 可选的 Stanza 依赖目录。也可在 Python 环境中安装相同版本 Stanza 1.15.0。 |
| `crossplatform/round20/assets` | Stanza 的 `STANZA_ASSET_RECEIPT.json` 和 `stanza` 资源目录。英语模型为 `stanfordnlp/stanza-en` 修订 `88e0cd3ddaa0c5a4b682895527ac013ce0241c21`。配置中的模型路径可为绝对路径或相对资产目录。 |
| `crossplatform/v1_weights` | 已登记的 `waymo.pth`、`drone.pth`、`quad.pth` 解码器权重。 |
| [源权重清单](crossplatform/single_frame_sources.json)中的路径 | 各源平台的 epoch100 与源域准备的匹配权重文件。运行时从匹配文件加载点云适配器和匹配投影。 |

初始化时核对所选源平台的模型、匹配和解码器权重 SHA-256。使用现有已配置工作区时，`--asset-root` 指向该工作区。 [上游 3EED](https://github.com/worldbench/3EED/tree/843a901f5367a6be5cc484b97c6ecad462961dfa) · [解码器权重清单](crossplatform/CURRENT_METHOD.json)

## 命令行和 Python

从 `v1/` 目录运行 `python crossplatform/ground.py --help` 查看参数。 [命令示例](../README.md#运行)

```python
from crossplatform.ground import Grounder

model = Grounder("waymo", asset_root="/path/to/3eed-workspace")
box = model.predict(
    target="drone",
    points="/path/to/frame/points.bin",
    rgb="/path/to/frame/image.jpg",
    calibration="/path/to/frame/camera.json",
    text="the white vehicle next to the pole",
)
```

- `--source` 和 `--target` 接受 `waymo`、`drone`、`quad`。同一个实例可复用该源平台的权重。六个迁移方向使用同一解码配置。
- 点云使用原始 LiDAR 坐标和米。支持至少三列的 `.npy` 数组，或每点四个 float32 的 XYZI 二进制文件。模型使用 XYZ，采样 16,384 点。默认种子为 0，点数不足时有放回采样。
- `--rgb` 为几何投影提供图像宽高。相机 JSON 提供 `image_intrinsic`、`image_extrinsic` 和可选 `camera_distortion`。外参沿用 3EED 各平台约定。Waymo 相机默认为前视，其他视角传 `--waymo-view FL`、`FR`、`SL`、`SR`。
- 输入一条英文描述，含模型固定后缀的长度最多为 256 个 token。
- 命令行标准输出为一行 JSON。 `box` 是目标 LiDAR 坐标系中的 `[x, y, z, length, width, height, yaw]`，yaw 以弧度计，模 π。Python 接口返回同一七维框。

## 推理过程

冻结的点云与语言前端为 256 个原始候选打分。候选重叠校正用于主体排序。主体与参照各保留四个候选，使用同次点云采样中的框内点估计朝向。第一轮读取局部点支持，交换两个角色的消息并对主体槽应用固定框几何反馈。第二轮重新读取点支持，通过配对分数选出最匹配的主体框。没有参照角色时，由主体分数选择。

六个方向分别核验了点云匹配分数和图像像素替换行为。更换为同宽高的黑图时，输出七维框逐值相同。

隔离发布包的命令行输出已与封存预测逐值核对。 [运行核验](verification/bundle_check.json)
