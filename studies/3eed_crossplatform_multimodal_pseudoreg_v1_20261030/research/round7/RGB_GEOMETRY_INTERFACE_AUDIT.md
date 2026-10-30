# R7 RGB、点云及标定接口审计

本地 RGB 和标定足以构建一个明确约定下的非学习投影接口；现有材料不能证明逐像素精确对齐，也不能把 RGB+文字定位当作已有能力。此次没有运行目标模型、TTA 或目标 GT 评分。

## 实际资源与读取边界

全目录只读库存覆盖 20367 个数据目录：Waymo 5409、Drone 7098、Quad 7860。每个目录均有 `image.jpg` 和对应 LiDAR 文件；所有 JPEG 文件头可识别为 RGB，终止 EOI 标记存在，K/E/pose 字段存在。该检查不等于逐像素完整解码或图像与 LiDAR 时间同步验证，RGB 文件模式也不证明画面实际有可区分颜色。

Waymo 图像分别有 4223 张 1920×1280、1186 张 1920×886；Drone、Quad 都为 1280×800。Waymo K 是 3×4，第四列全零；后两平台是 3×3，因此目前 `K[:3,:3]` 不丢失非零投影列。所有平台 E 为刚体 4×4，旋转正交性及行列式检查通过，所有主点落在对应图像范围内。全部 20367 份 metadata 都没有 `camera_distortion`。

审计器使用 `ijson.parse` 流式解析，只有 `image_intrinsic/image_extrinsic/camera_distortion/pose/image_path/lidar_path_proj/timestamp` 白名单子树被构造成对象。包含 ground_info 的 JSON 容器字节被打开、解析，ground_info/框/类别/positive_map/原 caption 的结构和数值没有被应用程序索引、输出或作为对象构建。目录图像只读取文件头与尾部，未读取点云内容。库存细节绑定 `RGB_CAMERA_HEADER_INVENTORY.json`；逐目录行仅本地私有保存。

## 坐标合同

令列向量 `p_model` 表示直接送入 3EED 的 XYZ，原生预测为同坐标中的 `[cx,cy,cz,sx,sy,sz]`，yaw 固定 0。原模型不接收 RGB；官方 `_get_3eed_pcd` 实际返回 XYZ，之后只做 Waymo 后缀预旋转，或 Drone 的 z+=1.8。

- Waymo：后缀 0..4 分别是 F/FL/FR/SL/SR，对应模型预旋转角 0/−45/+45/−90/+90 度。投影应使用 `M=A @ inverse(E) @ Rz(-angle)`，其中 `A=[[0,-1,0],[0,0,-1],[1,0,0]]`。E 是相机到 vehicle 的外参；不能直接把 active 官方通用 helper 的 `E @ p_model` 当作 Waymo 投影。官方 `transform_to_front_view` 中 translation_vectors 仅定义、未使用，不得补加。vehicle 坐标点也不得再乘 inverse(pose) 当成 global 点。
- Drone：模型点比原始点 z 高 1.8m，故 `M=E @ Translate(z=-1.8)`。
- Quad：当前接口采用 `M=E`。Drone/Quad 的 E 方向沿用释放数据及先前无标签往返/FOV 检查，尚未追溯到 M3ED→3EED 完整生成脚本，因此不夸大为完整原始标定恢复。

将 6D 框投影时，应先在 model 坐标生成八个 yaw0 角点，再逐角点应用 M。不能只变换中心而把 model 轴向 size 当作 camera 轴向 size。伪框及其梯度目标最后仍必须回到同一 model 坐标与米制单位。图像原像素坐标下使用 K；若任何后续 resize/crop，必须同步变换 K 或 uv，保留明确像素约定。

## 尚未补齐的物理标定信息

缺失 D 不能解释为真实无畸变。Waymo 官方 schema 明确原 intrinsic 还含畸变参数，并且 K 对应缩放后的图像；其投影实现还使用相机 pose、时间、速度与 rolling-shutter 信息。3EED 本地只保留上述较少字段。M3ED 原始数据也单列相机 distortion_model/coefficients；这些字段未进入本地 3EED metadata。尚未找到 3EED JPEG 在导出过程中是否已去畸变、相机选择、裁剪/缩放、时间同步补偿的完整生成代码。

因此缺 D 的默认策略应当拒绝，或者由登记计划显式选择 `assume_zero_declared` 并记录这是近似。当前 CPU CameraSpec 的默认拒绝符合这一边界；如果保持默认，本地所有实例都会因缺 D 跳过，而不是静默成功。过去 R4 72 个 Waymo 点集在 A@inv(E) 下完全落入正深度 FOV，说明坐标方向一致；它不是实测像素对应误差或跨模态语义正确率。

## 可检验的非学习反馈及其边界

首版可以固定纯文本选出的 source top1 身份，以开头目标短语中的有限颜色词将文字关联到 RGB；在 source 框附近，用点云投影、全 P 的 z-buffer 可见性和颜色支持构造局部几何候选。颜色阈值、支持量、连接半径、关联歧义门禁均需在真实执行前固定。多个候选同色、反光、阴影、车窗、曝光变化、背景同色、遮挡与点稀疏都可能让颜色证据错误或消失。第一版 parser 不是通用 2D grounding 或完整关系语义推理。

中心不能直接由局部可见点均值替代。保守版本仅在同轴两相反 source 框面均有支持、横向跨度及两侧残差一致时更新该中心坐标；不具备两侧支持的坐标权重为 0。单视图不透明实体通常难以提供完整相反表面，允许多数实例跳过。尺寸维持 source prior、三个尺寸权重全 0，保留六维 API 不等于已经恢复六维完整对象形状。

至少分别保留 source、直接伪框输出、回归参数更新和同信息的等价输出空间步；G 的伪标签位移上限不自动约束实际 SGD 位移，后者还依赖学习率、步数、冻结特征范数/Gram。PC-only 与固定颜色置换反馈可以检验 RGB 颜色信息，但两者门禁的实际接受数也必须分别报告。无 teacher/EMA 并不自动保证伪标签正确；只有后续封存后任务评分才能判断收益。

## 可追溯来源

- [Waymo dataset.proto](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto)：相机外参方向、缩放内参与畸变字段。
- [Waymo Perception coordinate conventions](https://waymo.com/intl/fr-ca/open/data/perception/)：vehicle/sensor 轴定义。
- [Waymo camera model operator](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/camera/ops/camera_model_ops.cc)：时间/pose/velocity/rolling-shutter 输入。
- [M3ED Data Files](https://m3ed.io/data_overview/datafiles/)：原始 imager 标定与 distortion model/coefficients。
- [3EED 官方仓库](https://github.com/worldbench/3EED)：RGB/LiDAR 数据结构及 Waymo/M3ED 来源；本地 frozen official loader/transform 与 R4 corrected camera、R5 `render_cases.py` 的实际代码已只读复核。
