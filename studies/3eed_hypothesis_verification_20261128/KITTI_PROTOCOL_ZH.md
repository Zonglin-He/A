# KITTI Tracking 录制独立车辆复测

从 KITTI Tracking 公共发布包的 21 段有标注录制中以固定时间分位点各抽 6 帧。下载只包含选定图像、Velodyne 和标定。选择和查询均先于下载目标标注。固定英文查询为 “The passenger car whose image bounding box center is closest to the horizontal middle of the image. Select a passenger car, excluding vans and trucks.” 没有自然语料时用统一规则限定此诊断范围。

按真实相机坐标把 KITTI rectification、投影平移和外参组合，使用正式项目的前向车载坐标分支。同一 Qwen 初始定位、正式框、回波、相机射线和物理约束用于三个尺寸读出。使用上一轮已固定的 Waymo TRAIN 物理及 Qwen 尺度表和残差系数。每条记录评估 C_ray_simple、只缩顶部、正式姿态＋类别尺度、正式姿态＋正确实例残差和同预测类别打乱残差。缺失支持或车型表时保留正式框。

预测封存之后才下载/读取 KITTI 标注。以框高度至少 25 像素、截断不超过 0.50、遮挡等级不超过 2 的 Car 类为合格对象。将标注的图像水平框中心最接近画面水平中央的一辆作为参考。图像宽度的 1% 内并列的帧记为指代歧义，缺少合格车辆另计。报告固定入选帧、可评分子集、正式物理锚适用子集和按录制重采样的置信区间。全部参考车辆由独立评分计算，推理从未接收 KITTI 目标类别标注、目标框或实例 ID。此为独立录制的模板车辆外域机制检验。
