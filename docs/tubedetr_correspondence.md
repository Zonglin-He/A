# 跨帧对应传递奖励资格

本轮只检验一个新信号：在固定Vid128、相同原八帧上，从既有单ReferDINO特征建立相邻观测帧的双向互近邻匹配，把另一帧教师框内的匹配点传递到当前帧。先检验能否评价候选，未通过不做学生适应或新全量。

为什么检验：同帧ROI及原生mask都以当前帧教师区域为参照，教师选错或事件外观测不会被自匹配纠正。跨帧传递检验相邻证据能否提供相对独立的位置支持，不假定互近邻等于正确身份。TubeDETR原论文的时间self-attention说明临时状态会传播到其他时间，STCAT关注跨帧一致性；它们不证明此奖励或OPD有效。

实现固定：有效图像坐标中心采样到最长边32的等比例网格，余弦最近邻及反向argmax一致，不做相似度阈值搜索。匹配的源点须落在另一帧固定教师框内；当前帧教师框不参与自身支持图构造。双向相邻来源取均值，固定3×3均值滤波填补离散网格，再用已验证的连续面积积分soft Dice给同32候选评分。框面积保留越界部分。空支持为0、并列top1采用原平均regret，不删除query或空帧。候选框连续梯度对Direct和OPD均开放。

预算128×7个至多1024²相似度矩阵，复用已封存特征/候选，0次新教师、0次学生fit、0参数搜索。完整reward seal后独立CPU读取GT，报告原Geometry/Layout3/新TransportDice的成对排序、Spearman、top1 regret；事前门槛为排序和regret均不劣几何且至少一项改善。另离线测量GT条件下同一query目标内互近邻的对应精度及空支持率，不用GT门控。只有资格通过才冻结另一有限同奖励OPD/Direct公平验证，不自动开全量。

来源与边界：
- [TubeDETR CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_TubeDETR_Spatio-Temporal_Video_Grounding_With_Transformers_CVPR_2022_paper.html)：时间self-attention与时间对齐cross-attention。
- [STCAT原论文](https://arxiv.org/abs/2209.13306)：用全局模板联系不同帧预测；不是本奖励实验依据。
- [Learning Correspondence From the Cycle-Consistency of Time CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Learning_Correspondence_From_the_Cycle-Consistency_of_Time_CVPR_2019_paper.html)：通过时间循环训练对应特征；原文同时指出循环可有捷径、遮挡/姿态变化可破坏循环。
- [Space-Time Correspondence as a Contrastive Random Walk NeurIPS2020](https://proceedings.nips.cc/paper_files/paper/2020/hash/e2ef524fbf3d9fe611d5a8e90fefdc9c-Abstract.html)：路径级对应自监督。此处没有训练新对应特征，不复现该论文，不声称本来的ReferDINO特征已具备对应能力。

主要失败风险是八观测时间间隔大、背景/相似外观造成互近邻错误、教师框背景污染和稀疏支持。资格结果决定是否继续，不依据novelty要求宣称性能。

## 完整资格结果

128条全部封存并独立评分。TransportDice成对排序74.2740%（Geometry74.9952%）、top1 regret0.145525（Geometry0.121997），未通过；差值regret+2.352803pp，父bootstrap95%CI[0.109021,4.713861]。1024观测中62个传递支持为空，平均相邻观测间隔108.80原始帧；GT条件下正确目标内源点的对应精度80.85%只在有GT可计算的对应上成立，不代表错误教师/事件外也可靠。局部对应正确并不足以证明支持图适合矩形定位边界或奖励排序。未做学生适应，不扩大参数或开全量。

下一步先用已封存对应/教师框做独立CPU原因分解：源点是否本来来自目标、传递支持在当前GT内的质量、空支持与长间隔的关系、教师较差/事件漏观测分组；不先引入新奖励。该分析须独立冻结范围，不能根据GT改门控或挑样本。

解释校正：上述80.85%是GT矩形内端点的条件包含率，不是像素级身份对应准确率。完整框范围分解见[reward extent](tubedetr_reward_extent.md)。
