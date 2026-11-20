# 四尺度物理区域联合读出运行说明

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

默认入口是 `crossplatform/ground_regions.py`。首次处理明确带紧邻或深度次序的描述时，使用 C++17 编译器在用户缓存目录生成树递推核。普通左右关系使用 NumPy 闭式计算，其他描述走地面回退。安装依赖后可以离线运行。

```python
from crossplatform.ground_regions import Grounder

grounder = Grounder("waymo", asset_root="/path/to/3eed-workspace")
box = grounder.predict(
    target="drone",
    points="/path/to/frame/points.bin",
    rgb="/path/to/frame/image.jpg",
    calibration="/path/to/frame/camera.json",
    text="Find the car immediately to the left of the bus in the camera image.",
)
```

`--source`、`--target` 接受 `waymo`、`drone`、`quad`。点云为 .npy XYZ 或原始 float32 XYZI 二进制文件。固定 `--seed` 决定同帧 16,384 点采样，默认为 0。相机内外参及可选畸变参数沿用 3EED 约定。Waymo 可设置 `--waymo-view F|FL|FR|SL|SR`。输出中心和尺寸单位为米，yaw 单位为弧度。

默认调用和显式 `--scales 4` 都使用四层区域，分别对应前景点云邻接图边权的 6.25%、31.25%、68.75%、93.75% 分位。关系句在有可用区域时额外运行一次主体短语候选提名。参照名词使用冻结语义在区域上评分，候选对按句中的相机左右关系联合选择。有紧邻或深度条件时仍使用树递推。输出框按已选主体点簇的可见面补全。回退沿用基础冻结读出与地面修正。[当前配置](crossplatform/CURRENT_METHOD.json) · [基础配置](crossplatform/FROZEN_V1.json)

复现上一版八尺度及其参照短语投票时，使用当时的发布提交。[八尺度历史版本](https://github.com/Zonglin-He/A/tree/4f6292f8eec44cc3b9dc2adc999b33165ea49275)
