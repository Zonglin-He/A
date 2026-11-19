# v1

v1 是单帧跨平台三维指代定位方法。输入目标平台的 LiDAR 点云、同步 RGB、相机标定和英文描述，输出目标在原始 LiDAR 坐标系中的七维框。冻结源模型生成候选，联合解码器通过两轮交互比较四个主体槽和四个参照槽。推理时使用对应源平台的已有权重，目标端参数更新为零。

## 运行

使用已有的 3EED CUDA 环境。将 `--asset-root` 指向本地运行依赖及权重所在的工作区。从仓库根目录执行以下命令。 [依赖和权重位置](v1/RUN.md) · [推理配置](v1/crossplatform/CURRENT_METHOD.json)

```bash
cd v1
python crossplatform/ground.py \
  --asset-root /path/to/3eed-workspace \
  --source waymo --target drone \
  --points /path/to/frame/points.bin \
  --rgb /path/to/frame/image.jpg \
  --calibration /path/to/frame/camera.json \
  --text "the white vehicle next to the pole"
```

输出 JSON 的 `box` 顺序为 `[x, y, z, length, width, height, yaw]`，中心和尺寸单位为米，yaw 单位为弧度。

## 六个跨平台方向

同一批普通 Validation 的 **18,620 条方向记录**使用相同的单源 epoch100、同一帧和同一次点云采样。每格依次是 **Acc@25 / Acc@50 / mIoU**。差值按未四舍五入的原值计算。 [精确数值 CSV](v1/RESULTS.csv) · [评测口径](v1/EVALUATION.json)

| 单源 → 目标 | 记录数 | Frozen（%） | v1（%） | v1 − Frozen（pp） |
|---|---:|---:|---:|---:|
| Waymo → Drone | 2,792 | 19.054 / 3.438 / 8.872 | 19.771 / 9.742 / 11.845 | +0.716 / +6.304 / +2.973 |
| Waymo → Quad | 2,736 | 32.164 / 15.716 / 17.704 | 37.792 / 21.820 / 21.680 | +5.629 / +6.104 / +3.976 |
| Drone → Waymo | 3,782 | 30.063 / 5.843 / 13.705 | 35.008 / 8.065 / 16.405 | +4.944 / +2.221 / +2.700 |
| Drone → Quad | 2,736 | 22.186 / 3.070 / 12.747 | 34.905 / 9.539 / 17.102 | +12.719 / +6.469 / +4.355 |
| Quad → Waymo | 3,782 | 22.845 / 7.509 / 11.474 | 23.533 / 7.985 / 11.936 | +0.687 / +0.476 / +0.463 |
| Quad → Drone | 2,792 | 8.990 / 1.361 / 6.211 | 14.004 / 4.226 / 8.355 | +5.014 / +2.865 / +2.144 |
| **六方向等权平均** | 18,620 | **22.550 / 6.156 / 11.785** | **27.502 / 10.230 / 14.554** | **+4.952 / +4.073 / +2.768** |

Frozen 使用官方发布代码的末层 BBF 和评测注释提供的类别跨度，框沿用原始轴对齐输出。v1 额外读取同步 RGB 和标定，以原句定位，并用框内 LiDAR 点估计朝向。以上数值衡量整条推理流程在已参与研发的 Validation 范围上的差值。 [Frozen 原始结果](https://github.com/Zonglin-He/A/blob/0c68f2c94c0d00cf91305f16df4e0884c97300e0/studies/3eed_official_cross_platform_validation_ablation_20261118/results/metrics.csv) · [v1 原始结果](https://github.com/Zonglin-He/A/blob/b8cc5449d3cbd709a71fdc64c43edf00af28f561/studies/3eed_r178_r174_combined_simplification_20261119/results/endpoints.csv)

[研究历史](studies/) · [封存评测报告](https://github.com/Zonglin-He/A/blob/b8cc5449d3cbd709a71fdc64c43edf00af28f561/studies/3eed_r178_r174_combined_simplification_20261119/REPORT_ZH.md)
