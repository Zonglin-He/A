# 方法相似性与待检验区别

| 工作 | 已有机制 | 对本实验的边界 |
|---|---|---|
| [3EED](https://arxiv.org/html/2511.01755v1) [官方代码](https://github.com/worldbench/3EED) | 跨平台零样本评测和端到端语言条件点云定位。实现提供双向点—词注意力、以采样点生成查询、分层解码 | 继承模型和协议。取出内部权重、继续选择现有 3D 候选本身属于复用。 |
| [BUTD-DETR，ECCV 2022](https://arxiv.org/html/2112.08879v2) [官方代码](https://github.com/nickgkan/butd_detr) | 第 3 节的候选／点／语言三路注意力、非参数查询和检测短语监督。消融比较去除候选输入及检测短语 | 将候选区域和文本对应已有先例。本实验在冻结预测过程中有条件地改变一个词角色的视觉读入，并比较接收方候选的相对响应。 |
| [EDA，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_EDA_Explicit_Text-Decoupling_and_Dense_Alignment_for_3D_Visual_Grounding_CVPR_2023_paper.html) [官方代码](https://github.com/yanmin-wu/EDA) | 显式主体、属性、参照和关系解耦，训练时对齐文本成分和空间查询 | 角色分词、文本成分相似度为已有机制。新增检验问一个具有特定物理支持的参照区域能否改变某个主体的匹配，反方向同样检验。 |
| [CORE-3DVG，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/9b91ee0da3bcd61905fcd89e770168fc-Paper-Conference.pdf) | 第 3.2 节以实体短语分别定位，并学习参照伪标签。第 3.3 节显式为候选对打分，用目标真值构造训练监督 | 重查询和成对目标都已有。首轮机制用直接测量的两个角色读入响应构造冻结边，并检验去掉可吸收到单体分的成分后是否仍有效。 |
| [Cross3DVG](https://arxiv.org/html/2305.13876) | 第 4 节检索候选相关的多视图图像，CLIP 文本条件加权，结合几何特征做跨数据集零样本迁移 | 跨传感器泛化及候选级 RGB 用于定位已有先例。本实验固定单帧和已有 RGB 路径。 |
| [Multi-View Transformer，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Multi-View_Transformer_for_3D_Visual_Grounding_CVPR_2022_paper.html) [官方代码](https://github.com/sega-hsj/MVT-3DVG) | 聚合多个虚拟视角的关系表示 | 本实验保持一帧相机坐标与相同几何关系项。 |
| [R66](https://github.com/Zonglin-He/A/tree/main/studies) | 在区域内先于融合删除视觉特征，匹配移除的种子和 decoder query keys。对完整及类别文本比较同一个区域候选物的概率下降 | 本实验只遮断特定词槽内某实例的原生 value 项，测量另一身份的匹配分。通过两个查询的角色互换取得双方向响应，再以交互矩阵和同响应加法读出比较。 |
| [R169–R172](https://github.com/Zonglin-He/A/blob/89a3dca18ac40a53e03ec9b8984365c63c4448b6/studies/3eed_joint_score_attribution_cross_query_verification_20261118/REPORT_ZH.md) | 同场景双角色重查询、原生参照池、左右关系、同物理实例的排名融合 | 第一项验证只冻结候选和既有分数。普通精确选对、重复框抑制与左右因子均是已有组件。 |

区别是可证伪的机制假说。仅当真实交互优于同一响应矩阵的单体分和错配，并改善正确双端身份与主体定位时，才考虑作为方法候选。

## 两角色置换

[CoT3DRef，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/32f9049217da6e718a426b07242dff73-Paper-Conference.pdf) 以 anchor chain 作逐步预测。[ORD，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_ORD_Object-Relation_Decoupling_for_Generalized_3D_Visual_Grounding_CVPR_2026_paper.pdf) 的消融表 7 比较锚点、相对位置和多视点的可训练融合组件。这些方法均表明显式实体、角色和几何解耦已有充分先例。第二项比较用两份冻结前向回答同一个物理组合在原角色与交换角色上何者受到更好支持；同信息单体与置换边为可证伪对照。源码验证已参考 EDA/BUTD-DETR 的候选与语义分支和 3EED 的点—词 cross encoder，本地运行中的 3EED 层文件与读到的官方副本 SHA 相同。3EED 发表在 NeurIPS 2025 Datasets and Benchmarks。

[DEER-3D 预印本](https://arxiv.org/html/2511.14086v2) 第 3 节通过场景物理编辑生成谓词条件的监督，并迭代训练。身份交换对照属于已有广义反事实思想。本研究使用实际同帧候选的交叉分数，源端只标记可区分的真实物理组合，使用几何近似匹配的错误配对拟合微型读出，目标部署全冻结。不能以这一差别单独宣称新颖性。

## 参照词的局部实测外观

[EG-3DVG，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Park_EG-3DVG_Expression_and_Geometry_Aware_Grounding_Decoder_for_3D_Visual_CVPR_2026_paper.html) 的 [训练损失实现](https://github.com/Gwan9Wook/EG3DVG/blob/main/models/losses.py#L620) 在 query–token 相似度中分别标记主体、属性和关系，也加权其他实体的负 logits。同类的显式语义对照和源监督已有先例。此处单独训练一个小读出，输入为同一物理框中通过标定对应的 RGB/LiDAR，检验框内读入与扣除图像／点云近邻背景。该模块没有通过源录制复验，未产生目标预测。它无法支持联合推理贡献。

## 词面极性干预

[Visual Programming for Zero-shot Open-Vocabulary 3DVG，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Yuan_Visual_Programming_for_Zero-shot_Open-Vocabulary_3D_Visual_Grounding_CVPR_2024_paper.pdf) 用基于锚点的定位与显式空间谓词程序。因此从实体出发计算 LEFT/RIGHT 不能单独当作新意。R167 的同证据左右重放也已试过对左右几何项改符号。本次新比较只改输入语言中的一个方向词并运行四次完整冻结查询。各实体短语不变，所有词面响应再用真实物理支持传回固定候选。参照身份未优于同证据错配时，禁止把神经分数的方向敏感当成参照身份信号。
