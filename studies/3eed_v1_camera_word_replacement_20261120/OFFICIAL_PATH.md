# 单目标官方协议的代码核对

**本研究沿用七维真值，与公开 3EED 单目标加载器一致。** 原先的[评测附记](EVALUATION_NOTE.md)记录了评分前登记的敏感性分析。核对整条调用链后，应修正其中对“原始真值的全部姿态”的解释。

1. 在官方提交 `843a901f` 中，加载器读取 Waymo 的七个字段，或 Drone、Quad 的九个字段。它分配形状为 `(MAX_NUM_OBJ, 7)` 的输出，写入 `bbox[:7]` 后返回。取点实例监督也只使用前七个字段。[官方单目标加载器](https://github.com/worldbench/3EED/blob/843a901f5367a6be5cc484b97c6ecad462961dfa/src/joint_det_dataset.py#L485-L501) · [实际返回的真值](https://github.com/worldbench/3EED/blob/843a901f5367a6be5cc484b97c6ecad462961dfa/src/joint_det_dataset.py#L540-L549)
2. 常规单目标入口接收上述返回值，训练时使用其中心与尺寸，评测时使用同一批框。独立 IoU 函数可接收七维或九维真值，官方单目标调用实际收到七维。原模型的预测框有六维，采用轴对齐几何。官方 Top-1 评价使用严格大于 0.25 和 0.50 的 IoU 阈值。[数据入口](https://github.com/worldbench/3EED/blob/843a901f5367a6be5cc484b97c6ecad462961dfa/src/joint_det_dataset.py#L702-L729) · [评测入口](https://github.com/worldbench/3EED/blob/843a901f5367a6be5cc484b97c6ecad462961dfa/src/grounding_evaluator.py#L244-L294)
3. 在本研究使用的源域拟合和录制留出清单中，Drone 的 704 个原始框第九列全为 1，Quad 的 800 个全为 2。这里没有独立的字段语义证明。输出文件里的 `full_pose` 指强制把第八、九列按 pitch、roll 的弧度值传入盒体计算。该列仅记录这一假设的数值后果，模型选择、正式协议比较和论文性能结论使用各自有依据的评分。官方单目标成绩不能从这项附加分析得到。[源域字段核对](results/training_scope.csv)

## 使用标注的边界

每个源的系数仅从该源官方 TRAIN 的原始框与原句估计。用于拟合的是框的前三维中心、相机标定和原句中的位置词。改写的句子只供预测时进行第二次读取，训练和评分都没有把原框指定为改写句的正确答案。简化 v1 的已有训练、每源新增的一个标量、额外的语言条件前向，分别列账。源域的录制留出约束针对本轮标量拟合；这些记录属于官方 TRAIN，既有骨干或解码器在过去的训练中可能已接触这些数据。[训练实现](calibrate.py) · [推理实现](predict.py)

精简 v1 继承的解码器已经经过源域训练。其自然描述使用主体真值框、未核验参照作为潜变量。源帧中满足构造条件的对象另行生成关系表达和双端监督。对比总体方法时，这些派生训练对和训练预算应随原有源域适配器一起披露。[已有解码器监督和各源样本量](https://github.com/Zonglin-He/A/blob/261676dd7ed9cc9b6d87aa9514d5a083b4e5c860/studies/3eed_recurrent_role_support_decoder_20261119/REPORT_ZH.md) · [v1 实现登记](https://github.com/Zonglin-He/A/tree/d0592f91d698a632d5c40783c3ef9518625411aa/v1)

对三份源权重各取第一块中首个可实际比较的源域留出输入，删去临时相机文件中的所有 `ground_info` 等标注字段，仅给相机内外参及可用畸变参数，重算原句与反事实两次前向。三个样本的全部框、读出、原始分数、可比较掩码与最终赢家均逐值复现。原始标注文件仅被读取。[复核回执](results/annotation_access_check.json)

项目使用同一七维真值和同一有向 IoU 比较各个 v1 变体，允许每个输出保留 PCA 的 yaw。与论文排行榜对齐还需核对官方评测清单、官方输入与输出约定、评分实现，以及开发成绩的曝光情况。3,072 条的结果用于六方向内部开发比较。[3EED 跨平台设定](https://arxiv.org/html/2511.01755v1#S4.SS1) · [本轮对照协议](PROTOCOL.json)
