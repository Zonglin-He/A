# R50：覆盖入口与跨模态关联的原论文／官方实现审查

审查范围：仅文献、公开作者代码和本地 R37/R38/R50 程序；未读取样本、媒体、标注、特征或预测，未执行模型、训练或评分。本文件不提供实验结果。采用研究迭代 skill 的“竞争解释、关键 tricks、最小区分实验”要求，但不增加已授权实验矩阵。

## 可迁移机制与不能借用的结论

| 工作 | 原工作实际机制 | R50 可借鉴的具体环节 | 不具备的前提／不能宣称的内容 |
|---|---|---|---|
| SeeGround，CVPR 2025 | 以三维物体查找表保存对象 ID、位置与框，渲染带对应标记的图像，语言／视觉判断返回对象 ID，再取回三维框；视角调整服务于查询及参照物。 | 视觉证据必须绑定明确的三维预测载体；保留 mask、投影点与 query 的可追溯对应，避免图像判断后另猜一个三维对象。 | 它有重建场景、三维实例与可渲染视角。当前稀疏室外点云、重叠 query 框和独立二维 masks 不能直接等同。没有证明 R50 的新质量公式或无类别入口一定有效。 |
| OpenMask3D，NeurIPS 2023 | 类别无关三维实例是载体，利用相机内外参、深度与可见性为每个实例选视图，提取多尺度图像特征再聚合。 | 实例存在与是否获得图像证据，不应完全取决于封闭类别分数；先考虑真实可见支持，再关联／聚合，同一实例的证据保持同一索引。 | 其多视图 RGB-D、配准与三维实例 proposal 不是当前所有帧都具备的条件；二维 mask 与源框的点支持比例不自动成为身份正确率。 |
| MM-TTA，CVPR 2022 | 各模态先以 slow/fast 预测一致性改进伪标签，再以模态可靠性作跨模态选择；共同输出是对应点上的同一类别分布。 | 跨模态可靠性必须建立在共同预测对象／支持上；不能用点数多或单次视觉回答自信替代关联验证。 | 其 soft selection 是模态权重归一化，不是保留未分配点质量；其双分支、BN 和 slow/fast 状态也非当前架构。不能将 R50 命名为 MM-TTA 复现。 |
| Reg-TTA3D，ECCV 2024 | 在回归为主要误差的检测设置中，以扰动稳定性过滤 RoI 伪框，用同类高置信尺寸统计修正回归，并以类别平衡 EMA 更新回归层。 | 以后接回几何适应前，须先证明监督对象正确、几何修正有净收益；重复 query 不应冒充独立目标统计。 | 识别汽车类别与识别文字指定的某一辆汽车不同；冻结框的 R50 不复现该回归适应方法，也不需要为此新增教师。 |

上表依据：[SeeGround 官方项目](https://seeground.github.io/)、[SeeGround 原文](https://arxiv.org/pdf/2412.04383)、[OpenMask3D 官方项目](https://openmask3d.github.io/)、[OpenMask3D 原文](https://arxiv.org/html/2306.13631v2)、[MM-TTA 原文 §3.3–3.4](https://openaccess.thecvf.com/content/CVPR2022/papers/Shin_MM-TTA_Multi-Modal_Test-Time_Adaptation_for_3D_Semantic_Segmentation_CVPR_2022_paper.pdf)、[Reg-TTA3D 原文 §3](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05971.pdf)。

## 官方实现中值得保留的细节与差异

**SeeGround 的论文可见性叙述不能直接当作默认代码保证。** 当前固定作者提交为 `928d827b18345454c6083cf49c9a7348cc04f6c1`（已核验 refs/heads/master 对象类型为 commit）。`projection.py` 虽计算深度图，但调用标注函数时传入 `depth_map=None`；所查 `draw_ids` 路径使用框角点投影。该默认路径没有兑现“传入深度图并作遮挡过滤”的完整链条。因此，R50 需要自行明确 nearest-depth 和多 mask 歧义处理，不应以“沿用 SeeGround 官方遮挡过滤”作为免责解释。[作者代码：projection.py](https://github.com/iris0329/SeeGround/blob/928d827b18345454c6083cf49c9a7348cc04f6c1/inference/projection.py#L86)

**OpenMask3D 的可见性依赖真正的深度观测。** 固定作者提交为 `3bc3fc52693b25668d0e91d55a2ea714544a4749`（已核验 main 的 commit）。所查实现使用传感器深度与投影点深度的绝对差阈值，并按每个实例在视图中的可见投影像素数选择 top-k；图像特征归一化后聚合。它不是“只要投到 mask 就计作可靠关联”。稀疏 LiDAR 的最近深度处理只能近似处理当前点集的遮挡，不能宣称恢复了该方法的完整 RGB-D 可见性。[作者特征提取代码](https://github.com/OpenMask3D/openmask3d/blob/3bc3fc52693b25668d0e91d55a2ea714544a4749/openmask3d/mask_features_computation/features_extractor.py)、[作者投影工具代码](https://github.com/OpenMask3D/openmask3d/blob/3bc3fc52693b25668d0e91d55a2ea714544a4749/openmask3d/mask_features_computation/utils.py)

**另外两篇的实现核验边界。** MM-TTA 论文给出的作者项目地址本次无法访问，未取得可核验的完整官方可执行实现；因此仅按论文与补充材料审查，不使用同名第三方仓库替代。Reg-TTA3D 原文说明基于 3DTrans，但基础框架不等于完整方法发行版；本次未确认作者完整 Reg-TTA3D 代码，不能宣称已审计其所有实现 tricks。[MM-TTA 作者项目入口](https://www.nec-labs.com/~mas/MM-TTA/)、[MM-TTA 补充材料](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Shin_MM-TTA_Multi-Modal_Test-Time_CVPR_2022_supplemental.pdf)、[3DTrans 基础实现](https://github.com/PJLab-ADG/3DTrans)

## 与本地 R37/R38 的机制区别

本节只读代码，不引用缓存内容或数据结果。

- `crossplatform/round37/prepare37.py:13` 的 full 字典，是将每个 mask 关联到满足原 `support>=3 / coverage>=0.15 / projectionIoU>=0.1`、且只关联一个 mask 的 query 列表。覆盖“所有 mask 的字典键”，不等于所有 mask 有三维支持，也不等于保留所有未分配质量。
- `crossplatform/round37/common37.py:62` 的 point/SoM 读出，先选择一个 mask，再从其有效 query 组内按源分数选代表；无组时回源第一名。它不是本轮的连续关联证据传播。
- `crossplatform/round38/predict38.py:12` 处理同一批 R37 原始回答的兼容解析与 strict/fullmap 对照，没有改变三维支持矩阵或新增可见性估计。

因此不能说“R37 做过全 mask 字典，所以 R50 的入口×关联已经做过”；也不能以此预判 R50 会有效。新的区分变量是：保留旧视觉证据时，谁获得证据、未分配和稀疏支持如何限制证据传播。

## R50 数学接口审查：未知质量不能变成分数零点偏差

以下是针对当前接口的推导，不是上述论文原公式。

若把 `W_im=C_im/(N_i+k)` 直接用于绝对证据 `W@e`，其行和为 `rho_i<1`。当所有视觉证据共同变化 `e_m→e_m+c` 时，query 获得 `rho_i*c` 的不同偏移；这可以在没有实例区分信息的情况下改变排序。原 full−gray 的 Yes/No 差分也不是经校准概率，零值不能直接充当可靠的未知类分数。

已检查 R50 锁前修订 `crossplatform/round50/pool_weights50.py` 与 `W_POOL_READOUT_CARD_DRAFT.md`：先计算条件关联 `a_im=C_im/S_i`、`S_i=sum_pool C_im`；用 `a@e` 形成旧 q0／原 median 回退锚点与原 clip；最后以 `rho_i=S_i/(N_i+k)` 收缩证据差。P/R 采用相同顺序。`a*rho` 只用于质量守恒审计，不能在锚定和 clip 前代回旧算子。

这是有必要的实现修订：常量证据不得制造候选间差异，rho=0 不得改变源分数，弱支持的每项证据增量应受 rho 限制。它仍然没有证明关联正确：一点入口可能是邻物，完整库存可能漏检，遮挡可使正确目标的 rho 偏低。rho 应称观测支持系数，不称置信概率或正确率。

旧 `legacy_operator` 必须保持原样；新分支的数学修订不能偷偷修进 oldG。用户的覆盖×关联五行负责测试整体净收益，不额外展开校准、阈值或新 VLM 设计。若共享 k 登记两个候选，须明确披露两个新关联配置的全部结果；不能把挑选后的五行描述为仅运行五个唯一配置。

## 对本轮采纳的约束

文献支持的是对象载体、对应支持和可靠性需要明确，不支持任何预先的涨分结论。保留旧主体／属性证据、冻结框与源分，使用同一固定开发集评判两平台等权总体 mIoU；覆盖率、稀疏点数与少数案例只解释结果。扩大入口同时绕开既有类别 ID 工程错位时，应披露该竞争解释，不能将全部增益归因于新关联算法。

本轮直接使用无净收益则停止该分支；有净收益才锁定整体评估，再比较相同信息的共识读出与参数适应。没有增加模型规模、源训练、回归分支或额外 TTA 损失的文献依据。本文件不改变 FINAL384 保护、定时任务关闭状态或任何实验锁。
