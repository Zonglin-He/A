# 八尺度物理区域联合读出

针对单帧跨平台三维指代定位，方法用一句话中的主体、参照物及相机左右关系，共同选择来自同一帧 LiDAR 的两个物理区域。从选中的主体区域测量框的朝向和可见表面，再用冻结模型的名词尺寸补全遮挡的一侧。缺少可解析的双角色关系或可行区域对时，对精简 v1 的主体框做局部地面修正。三种源模型都在目标平台冻结，RGB 文件仅提供相机画幅。

## 运行

在配置好源权重与 3EED CUDA 算子的环境中运行。

```bash
cd v1
python crossplatform/ground_regions.py \
  --asset-root /path/to/3eed-workspace \
  --source waymo --target drone \
  --points /path/to/frame/points.bin \
  --rgb /path/to/frame/image.jpg \
  --calibration /path/to/frame/camera.json \
  --text "Find the car immediately to the left of the bus in the camera image."
```

默认使用八尺度。传入 `--scales 4` 可运行经逐框复核的四尺度简版，保留原第 1、3、6、8 层。输出为目标 LiDAR 坐标系的 `[x,y,z,length,width,height,yaw]`。[运行环境](v1/RUN.md) · [主方法登记](v1/crossplatform/CURRENT_METHOD.json)

## 六方向普通 Validation

下表为 Acc@25 / Acc@50（%）。9,310 条目标描述各由另外两种源模型评测，共 18,620 条方向记录。1,294 条进入物理分支，17,326 条回退。官方 Frozen 取同一普通清单的历史 BBF 读出，使用官方类别跨度及原朝向。六方向等权，救回和误改按方向记录计数。

| 源 → 目标 | 记录 | Frozen 3EED | 精简 v1 | 八尺度主方法 | 四尺度 | 主方法相对精简 v1 的 Acc@50 救回 / 误改 |
|---|---:|---:|---:|---:|---:|---:|
| Waymo → Drone | 2792 | 19.054 / 3.438 | 19.986 / 10.064 | 19.986 / 10.494 | 19.986 / 10.494 | 23 / 11 |
| Waymo → Quad | 2736 | 32.164 / 15.716 | 38.816 / 22.405 | 37.354 / 21.418 | 37.317 / 21.418 | 12 / 39 |
| Drone → Waymo | 3782 | 30.063 / 5.843 | 34.638 / 8.065 | 35.325 / 11.185 | 35.431 / 11.317 | 135 / 17 |
| Drone → Quad | 2736 | 22.186 / 3.070 | 33.845 / 8.406 | 32.639 / 8.882 | 32.675 / 8.955 | 24 / 11 |
| Quad → Waymo | 3782 | 22.845 / 7.509 | 24.352 / 8.117 | 25.595 / 10.259 | 25.621 / 10.259 | 96 / 15 |
| Quad → Drone | 2792 | 8.990 / 1.361 | 12.607 / 3.904 | 13.145 / 4.477 | 13.145 / 4.513 | 20 / 4 |
| 六方向等权 | 18620 | 22.550 / 6.156 | 27.374 / 10.160 | 27.341 / 11.119 | 27.363 / 11.159 | 310 / 97 |

八尺度的六方向 mIoU 为 14.8022%，四尺度为 14.8188%。该普通验证清单已用于开发。四尺度改变 364 个主体框，Acc@50 救回 15 条、误改 7 条。只去八尺度的水平尺寸补全，Acc@50 降低 0.6766 个百分点。全量包含所有回退与退化方向。[完整消融报告](studies/3eed_eight_scale_component_ablation_20261120/REPORT_ZH.md) · [六方向与成对差值](studies/3eed_eight_scale_component_ablation_20261120/results/metrics.csv)
