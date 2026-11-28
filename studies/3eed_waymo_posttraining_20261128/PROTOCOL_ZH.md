# 单源 Waymo 后训练与零样本迁移

Waymo Ordinary Train 以录制轮转和固定 SHA 排序选 256 条，每张 RGB 仅取一个表达。Waymo Ordinary Val 独立选 64 条。训练端的文件审计拒绝读 Drone、Quad、外部传感器与旧实验运行目录。所有适配器使用固定停止步数。先验证优化确实改变 LoRA 参数并封存训练回执，再在独立进程选取 Drone、Quad 和 Mono3DRefer 各 64 条评估。目标从未用于梯度、示例库、超参数或 checkpoint 选择。目标只提供当条 RGB、语言、标定和真实 LiDAR。目标真值在推理和几何输出封存之后进入单独评分进程。3EED 和外部队列使用已经过项目评估的公开数据。

## 方向 1 · 后端闭环 GRPO

以同一 Qwen3-VL-4B 指令权重初始化。仅优化语言模型最后八层的 Q/K/V/O LoRA，秩 8、缩放 16/8，视觉塔及基础参数冻结。联合 JSON 包括首次 2D 框、相机系 9 参数 3D 框和模型类别。Waymo GT 用已注册相机标定转换给 SFT。128 步监督热启动同时作为额外 48 步普通 SFT 和 48 组 GRPO 的初值。两臂固定停止步数，训练样本使用同一 Waymo 训练数据池。保留冻结正式入口的两次读出、冻结模型的联合提示、热启动、继续 SFT 和 GRPO 的对照。

每组当前策略采样 4 条，温度 0.75，计算冻结正式几何后端恢复后的真实源域 IoU。总奖励为 `IoU3D + 0.15 IoU2D + 0.05 valid`。3D 与 2D 格式有效性分开记录，组内奖励全相同则跳过优化。组内标准化优势和逐 token 裁剪比率沿用结果监督的 GRPO。每个有效组做两步更新，KL 参考是固定的 SFT 热启动。训练后端与目标推理用同一个已注册几何入口。单模型联合读出与两次独立读出存在提示和计算预算差别。GRPO 和继续 SFT 的优化步数及样本调用分别记录。 [DeepSeekMath 中的 GRPO](https://arxiv.org/html/2402.03300#S4.SS1)

## 方向 2A · 物理条件化几何补全

仅用冻结 Qwen 的首次 2D 框取真实 LiDAR 回波。对 ROI 中 Fit 回波使用深度中位数和稳健范围筛出支持点，向模型提供 LiDAR 坐标中的中位中心和 5%–95% 可见跨度。监督只有源域目标长宽高和轴向 yaw。Waymo 和目标读取完全相同的条件。端到端评估把补全尺寸和 yaw 合入缓存的冻结 3D 读出，保留原 2D、中心和预测类别，再运行同一正式几何后端。无法生成补全值的条目回退到冻结基准并保留在分母。并列未训练同提示对照、冻结原始 3D 尺寸和朝向以及共同有效样本的误差。同一训练 checkpoint 另做单样本反事实，将测量中心和跨度都乘以 1.4，其余输入保持一致。

## 方向 2B · 场景图与反事实监督

用全相机视野的四尺度 LiDAR 提案显示前八个数字标记。提案成员和排序与指代表达、目标 2D 框及 GT 无关。自然指代的训练 ID 来自源域支持点与 GT 的交叠，缺少合格提案时监督 0。另一半训练样本来自 64 对反事实，每对固定 RGB、语言、标记和可选 ID，仅交换两个节点的度量属性，正确答案随深度次序反转。训练、测试均使用同一场景图表示。目标自然选择比较 SoM、真图和错配图，各有冻结与训练读出。天然与反事实都按原计分分母报告。反事实同时答对率衡量度量条件响应，自然指代的同池选择和最终框用于评估实际任务收益。用相同的预先决定的后端补全框，报告提案的离线覆盖及 Oracle。

## 复跑

环境为现有 3EED 冻结 Qwen 工作环境，RTX 5090、Python 3.12.16、PyTorch 2.15.0+cu132、Transformers 4.57.1。LoRA 由 PyTorch 实现。使用锁定的 Qwen 权重、本机 3EED 发布包及既有 Mono3DRefer/KITTI 数据和回执。各阶段在独立进程运行，权重和逐例信息留在本机 `runs/waymo-posttraining-20261128`。公开源码、协议和汇总。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
STUDY=studies/3eed_waymo_posttraining_20261128
python "$STUDY/source_data.py"
python "$STUDY/train.py" warm
python "$STUDY/train.py" source_sft
python "$STUDY/train.py" grpo
python "$STUDY/conditioning.py"
python "$STUDY/train.py" shape
python "$STUDY/train.py" graph
python "$STUDY/transfer.py" lock
python "$STUDY/transfer.py" plan
python "$STUDY/transfer.py" predict
python "$STUDY/transfer.py" geometry
python "$STUDY/transfer.py" score
```
