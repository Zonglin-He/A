# v1

v1 是单帧跨平台三维指代定位方法。点云和英文描述生成并评分原始候选，候选重叠校正后保留四个主体槽。联合解码器结合四个参照槽及相机几何，通过两轮交互选择目标。用框内点的 PCA 估计朝向，输出目标 LiDAR 坐标系中的七维框。

推理复用各平台的既有源权重。图像文件只提供宽高，标定用于计算相机投影和可见性。目标端参数更新为零。

## 运行

在已有的 3EED CUDA 环境中运行，将 `--asset-root` 指向本地依赖及权重所在的工作区。 [依赖和权重位置](v1/RUN.md) · [推理配置](v1/crossplatform/CURRENT_METHOD.json)

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

输出 JSON 中的 `box` 顺序为 `[x, y, z, length, width, height, yaw]`。中心和尺寸单位为米，yaw 单位为弧度。

## 六个跨平台方向

在已登记的普通 TRAIN 面板上，1,536 条描述由两个外域源模型分别预测，共 3,072 条方向记录。指标依次为 Acc@25 / Acc@50 / mIoU。Frozen 使用同输入的原生 3EED 文本读出。

| 训练 → 测试 | 方向记录 | Frozen（%） | v1（%） | v1 − Frozen（pp） |
|---|---:|---:|---:|---:|
| Waymo → Drone | 512 | 33.398 / 10.156 / 15.549 | 36.328 / 26.953 / 22.012 | +2.930 / +16.797 / +6.464 |
| Waymo → Quad | 512 | 33.203 / 20.508 / 18.647 | 35.156 / 24.805 / 20.708 | +1.953 / +4.297 / +2.061 |
| Drone → Waymo | 512 | 28.711 / 6.250 / 13.691 | 31.836 / 9.570 / 15.811 | +3.125 / +3.320 / +2.120 |
| Drone → Quad | 512 | 28.711 / 6.641 / 14.131 | 25.977 / 9.180 / 13.295 | -2.734 / +2.539 / -0.836 |
| Quad → Waymo | 512 | 21.680 / 6.641 / 10.909 | 23.438 / 6.445 / 11.884 | +1.758 / -0.195 / +0.976 |
| Quad → Drone | 512 | 15.234 / 7.812 / 9.912 | 21.484 / 9.180 / 12.616 | +6.250 / +1.367 / +2.704 |
| **六方向等权平均** | 3,072 | 26.823 / 9.668 / 13.806 | 29.036 / 14.355 / 16.054 | +2.214 / +4.688 / +2.248 |

相对同面板上的精简前 v1，当前 v1 的三项变化为 **+0.326 / +0.391 / +0.149 pp**。

[精简复核](v1/verification/README.md) · [六方向结果](v1/RESULTS.csv) · [历史组件消融](https://github.com/Zonglin-He/A/blob/dd5b3c94393420a11de6f207daf2a506f973f9cb/v1/ablation/README.md)
