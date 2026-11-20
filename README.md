# 四尺度物理区域联合读出

在同一帧 LiDAR 中，用四个邻接阈值构造嵌套的实测区域。对普通左右句，每个区域的主体得分等于**主体证据乘以合法方位的参照证据之和**。最高分区域的点云给出 PCA 朝向和可见面，再按主体名词尺寸补全三维框。紧邻和深度描述使用有序关系递推。缺少可解析的双角色关系或可行区域对时，用冻结的精简 v1 选框并修正框底。三种源模型均冻结，RGB 文件提供相机画幅。[公式与配对消融](studies/3eed_four_scale_direct_subject_20261120/REPORT_ZH.md)

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
  --text "The car to the left of the bus in the camera image."
```

直接运行上述命令即为发布配置。区域构造使用原八层中的第 1、3、6、8 层，只有主体短语额外运行一次候选提名。输出为目标 LiDAR 坐标系的 `[x,y,z,length,width,height,yaw]`。[运行环境](v1/RUN.md) · [主方法登记](v1/crossplatform/CURRENT_METHOD.json)

## 六方向普通 Validation

9,310 条目标描述各由两个外域源模型评测，共 18,620 条方向记录。1,294 条进入区域读出，17,326 条走地面修正回退。Frozen 为同清单的历史官方 BBF，使用类别跨度及原朝向。最后一列是相对上一版四尺度读出的 Acc@50 阈值切换。

| 源 → 目标 | 记录 | 官方 Frozen Acc@25 / Acc@50（%） | 本版 Acc@25（%） | 本版 Acc@50（%） | 本版 mIoU（%） | 救回 / 误改 |
|---|---:|---:|---:|---:|---:|---:|
| Waymo → Drone | 2792 | 19.054 / 3.438 | 19.628 | 10.208 | 11.860 | 1 / 6 |
| Waymo → Quad | 2736 | 32.164 / 15.716 | 37.208 | 21.235 | 21.496 | 0 / 4 |
| Drone → Waymo | 3782 | 30.063 / 5.843 | 35.643 | 11.502 | 17.497 | 5 / 2 |
| Drone → Quad | 2736 | 22.186 / 3.070 | 32.566 | 8.845 | 16.519 | 2 / 4 |
| Quad → Waymo | 3782 | 22.845 / 7.509 | 25.674 | 10.286 | 13.158 | 0 / 4 |
| Quad → Drone | 2792 | 8.990 / 1.361 | 13.073 | 4.513 | 8.063 | 2 / 0 |
| 六方向等权 | 18620 | 22.550 / 6.156 | 27.298 | 11.098 | 14.766 | 10 / 20 |

相对官方 Frozen，本版六方向等权 Acc@25 / Acc@50 提高 4.7481 / 4.9416 个百分点。相对上一版四尺度，分别下降 0.0587 / 0.0589 个百分点。区域分支的 Acc@50 从 17.5110% 变为 16.5295%。普通验证已用于开发。[各方向与分支结果](studies/3eed_four_scale_direct_subject_20261120/results/metrics.csv) · [全部逐例结果](studies/3eed_four_scale_direct_subject_20261120/results/ordinary_rows.csv.gz) · [Frozen 评测口径](studies/3eed_eight_scale_component_ablation_20261120/results/frozen_comparison.csv)
