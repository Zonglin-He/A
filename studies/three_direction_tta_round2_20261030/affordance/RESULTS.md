# GREAT round2：固定FPS视图与任务式残差适应

本轮完成 dev16 与预先冻结的 confirm32。**不采纳局部支持结构机制。** 完整更新在确认集的 aIoU 高于开发阶段选定的最强无更新 q75 集成，但 AUC、SIM、MAE 都退化；去掉局部支持项的 HM-only 在 aIoU 上更高。现有证据支持“发生了真实参数更新，并改变了覆盖/概率读出”，不足以支持“局部几何支持修复了功能语义定位”。这不是对全部3D affordance TTA路线的否定。

`TRICKS_AUDIT.md` 和 `PROTOCOL_DRAFT.md` 是首次真实执行前的审计/草案，开头的“尚未运行”描述其保存时状态；本文件及 `STATUS.json` 记录执行后的状态。机制、cohort、执行源码锁和全部原预测保持不变。

## 实验条件与工程复原

使用官方 GREAT commit `8ab28ec4dd7bbfe4d3a70aa9fe9b7ef1295d0cef`，历史40000词文本配置与完整严格加载的恢复权重。当前官方代码写RoBERTa-base，而原历史/权重形状对应RobBERT；这项来源矛盾没有被宣称解决。所有旧参数和buffer保持eval/frozen，新增129参数head逐parent/逐arm零初始化与独立optimizer，固定Adam .01、3步，logit残差上限.5。

新source纠正了旧runner“先float32再归一化”的顺序，采用官方float64-centroid/radius后cast32；dev另保留同seed的legacy32 source桥。这个修正不属于TTA增益。图像为原生RGB resize224和ImageNet normalization，沿用离线CoT；无现场MLLM生成。

K5只改变FPS随机起点，保持原点序，消除任意permutation同时改变邻域截断与decoder固定图像索引的混杂。真实cached native replay/同seed fresh repeat均bitexact；独立固定encoder测试中，只重排点索引造成输出变化，成对恢复图像投影索引恢复输出。这支持具体的decoder索引敏感性，不能据此声称模型完全不理解几何。

所有适应和零head读出在同CUDA设备执行，逐视图与实际native model sigmoid bitexact；没有以CPU sigmoid对自身相等冒充native GPU桥。HM的loss/gradient按官方literal运算顺序通过CPU精确比较。初次代数重排移植未通过工程断言，已保存无效记录，随后在首次真实forward前修正。

## 样本与隔离

先锁confirm32，后锁dev16，旧12全部视为已曝光。以三列XYZ规范排序hash、PCA符号等价、有限近形状比较和图像复用连接做分组；dev/confirm与旧集无已知几何/图像重叠。独立审计额外检查了全部所选同类跨split近形状配对。它是保守的泄漏控制，并非真实mesh/物理ID的证明；PCA退化、不同重采样与近重复图像可能逃逸。

选择器打开原NPY容器，仅将前三列XYZ复制到适应缓存；GT第四列由独立评分进程在预测seal之后读取。适应和选参不含GT数据流。confirm32冻结后不换样本，开发后不改步数、学习率、损失权重或局部规则。官方source训练/选模来源仍不完全明确，因此新确认隔离只针对本次TTA开发。

## 全部对照

下表AUC/aIoU单位为百分比，SIM/MAE为0–1；MAE越小越好。每项指标有效分母均为16或32，无全零/全一GT导致未定义样本。所有无更新集成共享K5真实source forward。更新的零步读出严格等于median-logit集成，而非source-only。

| arm | dev AUC | dev aIoU | dev SIM | dev MAE | confirm AUC | confirm aIoU | confirm SIM | confirm MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 67.9935 | 13.3406 | .271422 | .134534 | 75.2785 | 17.6737 | .373384 | .117450 |
| mean-prob K5 | 68.4828 | 13.5886 | .278367 | .132569 | 75.3240 | 18.0917 | .381466 | .115134 |
| median-logit K5 | 68.5461 | 13.5227 | .275555 | .132226 | 75.2305 | 18.2234 | .380368 | .115034 |
| q75-prob K5 | 68.5449 | 13.9153 | .282432 | .136172 | 75.3651 | 18.4687 | .383694 | .117845 |
| HM-only residual | 68.4313 | 14.2629 | .272627 | .141356 | 75.1688 | 19.4858 | .377697 | .123024 |
| HM + local-support residual | 68.5973 | 14.0300 | .276019 | .135278 | 74.8474 | 19.0839 | .379100 | .119544 |

开发集按预锁规则在3种无更新集成中以aIoU选择q75主比较，独立decision在confirm执行前冻结。完整机制对q75在确认集 aIoU +0.6151pp，21例提高、8例下降、3例相同；对HM-only则 −0.4019pp，7提高、22下降、3相同。不能把source→teacher集成收益归给参数更新，也不能在看到confirm后改选评价指标。

## 确认后诊断：覆盖与概率变化

以下为已曝光确认集上的有限事后描述，不是新的独立验证，不用于门控、选步或修改模型。父样本配对bootstrap10000次、seed42得到full−q75 aIoU区间[+0.3271,+0.9460]pp；小规模分层/近似物理分组限制其推断范围，不能替代新的held-out复验。

median→full点概率Spearman均值.98945。将有效最终logit残差投影到`span(1,z)`，平均54.08%残差能量由平移+温度分量解释；HM-only为66.40%。所有拟合温度斜率为正，未生成或评分新的校准预测。全局平移单独只能解释2.70%/6.50%；所以不是纯偏置，仍有46%/34%未被仿射分量解释的空间残差。高排序保持和IoU/排序指标分歧与校准/覆盖解释相容，不证明全部增益都来自校准。

在固定0.5阈值下，相对q75，full每parent平均TP36.47→37.25，FP103.41→108.91，目标召回.3256→.3614；平均目标/非目标MAE均变差。这不是普遍背景抑制。HM-only的预测正点与FP扩张更大，并取得更高aIoU；局部支持项削弱这一扩张，但没有带来一致的任务质量改善。原生soft-target focal+双侧Dice在p=teacher处本来就存在非零梯度，teacher拟合损失下降不能作为定位正确的证据。

固定选择full−q75 aIoU最大正/负各1例，保存两相同物理视角的private点图，未增加模型调用：

- 正例Kettle/pour：aIoU +3.41245pp；TP45→46，FP65→60。GT大致对应壶嘴区域，多个arm都保留非目标壶把响应。改善更接近已定位后的局部覆盖/概率修正，median aIoU40.52%仍高于full38.36%，full与HM-only几乎相同。
- 负例Kettle/pour：aIoU −.47445pp；TP56→54，FP保持40。full比q75目标平均概率下降、非目标平均概率上升，AUC94.83→92.56；假阳性壶把响应仍在。不能以“区域缩小”概括成有效背景抑制。

两例交互图与点云本来可来自不同实例，不能解释为图像到点云的像素对齐误差。图像、原始GT、坐标和预测只保留本地private，不进入公开包。

## 成本与边界

dev/confirm共240次primary source forward，另32次dev预处理桥/重复forward和7次首parent工程forward；cached decoder诊断另计。确认集source每次同步forward均值.1151秒，K5约.576秒；三步cached full更新均值.0191秒。source峰值CUDA allocated约1.14GB，adapter约69.8MB。这些是局部计时/分进程allocated窗口，排除模型加载、预处理、数据I/O、离线CoT生成，不能当作完整部署延迟或总显存证明。

全部129参数臂均有非零梯度/参数delta，残差不超过.5，零head/冻结source/缓存与跨arm无更新输出检查全部通过。无效工程尝试、原锁及全部封存预测保留。父代理独立rank AUC及其余三项指标复核dev16/confirm32均通过，最大误差2.22e-16；本代理保存报告消息receipt，原审计产物由父代理归档。

本轮停止于**不采纳局部支持结构机制**。HM-only的aIoU提升可作为后续校准/覆盖研究的候选证据，但当前其他指标退化且确认已曝光，不能直接升级为已验证的语义TTA方法。本轮没有进行后确认调参、替换样本、单调校准重评或新增GPU实验。
