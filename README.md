# 四尺度物理区域联合读出

针对单帧跨平台三维指代定位，方法用一句话中的主体、参照物及相机左右关系，共同选择来自同一帧 LiDAR 的两个物理区域。主体短语提出候选框，按实测点支持向各区域投票。参照物由区域内的冻结名词证据与空间关系定位。选定主体后，以可见表面为锚补全框的遮挡侧。缺少可解析的双角色关系或可行区域对时，对精简 v1 的主体框做局部地面修正。三种源模型都在目标平台冻结，RGB 文件仅提供相机画幅。

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

直接运行上述命令即为发布配置。区域构造使用原八层中的第 1、3、6、8 层，只有主体短语额外运行一次候选提名。输出为目标 LiDAR 坐标系的 `[x,y,z,length,width,height,yaw]`。[运行环境](v1/RUN.md) · [主方法登记](v1/crossplatform/CURRENT_METHOD.json)

## 六方向普通 Validation

9,310 条目标描述各由两个外域源模型评测，共 18,620 条方向记录。1,294 条进入区域读出，17,326 条走地面修正回退。Frozen 取同一普通清单的历史官方 BBF、类别跨度及原朝向。最后一列统计本版相对上一版八尺度、带参照短语投票的 Acc@50 阈值切换，所有方向均保留。

| 源 → 目标 | 记录 | 官方 Frozen Acc@25 / Acc@50（%） | 本版 Acc@25（%） | 本版 Acc@50（%） | 本版 mIoU（%） | 救回 / 误改 |
|---|---:|---:|---:|---:|---:|---:|
| Waymo → Drone | 2792 | 19.054 / 3.438 | 19.878 | 10.387 | 12.012 | 1 / 4 |
| Waymo → Quad | 2736 | 32.164 / 15.716 | 37.317 | 21.382 | 21.593 | 1 / 2 |
| Drone → Waymo | 3782 | 30.063 / 5.843 | 35.510 | 11.423 | 17.395 | 10 / 1 |
| Drone → Quad | 2736 | 22.186 / 3.070 | 32.639 | 8.918 | 16.595 | 2 / 1 |
| Quad → Waymo | 3782 | 22.845 / 7.509 | 25.833 | 10.391 | 13.245 | 7 / 2 |
| Quad → Drone | 2792 | 8.990 / 1.361 | 12.966 | 4.441 | 8.001 | 2 / 3 |
| 六方向等权 | 18620 | 22.550 / 6.156 | 27.357 | 11.157 | 14.807 | 23 / 13 |

六方向等权，Acc@50 相对八尺度带参照投票提高 0.0378 个百分点（23 条救回、13 条误改）。固定四尺度比较，去掉参照投票的 Acc@50 变化为 −0.0024 个百分点（12 条救回、10 条误改），推理时少一次参照短语候选提名。普通验证已参与开发。三份源模型均冻结，FINAL 保留。[同批配对结果](studies/3eed_eight_scale_component_ablation_20261120/results/four_default_comparisons.csv) · [全部逐例结果](studies/3eed_eight_scale_component_ablation_20261120/results/compression_rows.csv.gz) · [八尺度历史消融](https://github.com/Zonglin-He/A/blob/4f6292f8eec44cc3b9dc2adc999b33165ea49275/studies/3eed_eight_scale_component_ablation_20261120/REPORT_ZH.md)
