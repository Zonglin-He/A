# GREAT round2：关键实现条件审计

状态：CPU准备；尚无round2模型输出、开发分数或确认分数。旧v1/v2、计划、源码、预测、环境和公开包均保留。这里只把已核实现与待检验机制分开，不把候选写成有效方法。

## 来源和基准身份

官方HEAD已在线回读，仍为 `8ab28ec4dd7bbfe4d3a70aa9fe9b7ef1295d0cef`（2025-08-15）。原论文和补充材料已分别下载到 `round2/sources/GREAT_CVPR2025_{paper,supp}.pdf`；补充的A.1/A.2/A.3与实际代码共同作为依据。

- [CVPR论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Shao_GREAT_Geometry-Intention_Collaborative_Inference_for_Open-Vocabulary_3D_Object_Affordance_Grounding_CVPR_2025_paper.pdf)
- [补充材料](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Shao_GREAT_Geometry-Intention_Collaborative_CVPR_2025_supplemental.pdf)
- [官方实现](https://github.com/yawen-shao/GREAT_code/tree/8ab28ec4dd7bbfe4d3a70aa9fe9b7ef1295d0cef)
- 原PIAD任务同样不要求图像/点云来自标定同一实例：[ICCV2023论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Grounding_3D_Object_Affordance_from_2D_Interactions_in_Images_ICCV_2023_paper.pdf)。因此不引入未经定义的pixel–point对应损失。

## 已核关键条件

| 条件 | 论文/代码/本地证据 | round2处理及解释边界 |
|---|---|---|
| Checkpoint与文本编码器 | 当前`model/GREAT.py:197/236`写roberta-base，但原git历史为robbert-v2-dutch-base；下载权重两路embedding均40000词，现roberta-base为50265。官方issue8称占位已改，issue13仍报告40000不匹配。 | 固定历史checkpoint兼容配置和现有严格加载的全部827条state，不另换语言编码器。两embedding相同但pooler不同，保持各自原值。恢复成功不证明论文文本编码器身份矛盾已解决。 |
| 模型恢复可信度 | 当前model-only文件SHA256 `2226bc80bc7c369fd2e55852a6ccd5e9baf4310403f971d044f7c8bff9bd17bf`；来源ZIP成员size/CRC、所取成员SHA、alias映射均保留，strict/finite已过。 | 工程复原与方法有效性分开；原始训练archive字节未全部下载，不能声称与完整原archive逐tensor的更强审计已经完成。 |
| XYZ归一化dtype | 官方dataset保留NPY原dtype，`pc_normalize`先减centroid、除最大半径，eval后才`point.float()`。新48例raw均float64。旧runner先float32再归一化。 | 新source统一官方float64-normalize→float32。dev16输入数值桥最大差3.7252903e-7；GPU阶段需另留legacy32 source预测桥。可能改变FPS/radius边界，任何收益归为预处理修正，不计TTA。旧成对结论不改写，但绝对复现忠实度补记此差异。 |
| 图像输入 | release dataset RGB、PIL resize224×224、ImageNet均值/方差；旧runner与之相符。补充A.1提训练随机crop，但发布loader没有crop。 | 推理保持release native规则，不主动加crop冒充缺失推理trick；训练差异是复现来源不确定项。 |
| 点编码器 | SA层FPS为512→128→64；FPS初始`torch.randint`在CPU生成再送device。ball query以输入索引排序后截断半径内前k个邻居。FP使用最近3点、`1/(d+1e-8)`权重。 | 只seed CUDA不够。适应视图固定原点序，仅改变FPS随机种子。任意perm混合FPS、邻域截断和固定索引图像分支，不能解释为单纯几何信号。 |
| Decoder索引与cache | `decoder.reshape`把49位置映射到2048索引；`Decoder.forward`会替换传入列表的`p3[1]`。 | 每次decoder replay复制层级容器/张量，不能把已融合的p3再次融合。索引诊断固定完整encoder层级，分别重排p0和成对重排图像索引投影；无变化native replay和paired restore作为正控。这仅是cached机制诊断。 |
| 预测尺度 | 最后64→1 Linear后pointwise sigmoid；不是24类softmax，也不对2048点sum-normalize。 | 捕获真实pre-sigmoid logits；teacher在logit域做分位数，再sigmoid。只有SIM评价做sum-normalization。 |
| 官方focal | `utils/loss.py::HM_Loss`：gamma2，正项alpha=.25，负项1-alpha=.75，epsilon1e-6；先B/N维均值再通道求和。 | 已literal移植并CPU验证loss/gradient逐元素bitexact。不能把标准BCE/二类entropy当成同一训练目标。 |
| 官方双侧Dice | 正/负intersection分别除以各自pred+target之和，没有标准2×intersection；`mean(1.5-dice_pos-dice_neg)`。 | 不改成标准Dice。hard perfect target仍有0.5常数offset，不是优化失败；soft target处gradient通常非零，不能用loss下降证明任务正确。CPU同值soft target测试最大梯度约.00863，属数学/工程测试，不是数据任务结果。 |
| 配对及梯度尺度 | 训练n=2图像–同类/同affordance点云在线配对；`train.py:133–148`两项HM相加，日志再除pair数，不是反向前平均。 | 不把日志loss尺度当更新梯度尺度。round2视图损失明确按K batch采用literal HM均值；它是新TTA配置，不声称复现训练步长。 |
| 优化/冻结 | 补充A.3：ImageNet初始化图像主干、冻结文本主干、点云主干从头训练；65epoch/lr1e-4/batch16。release unseen配置50epoch，Adam+cosine；image小lr param groups在代码里被注释。 | 不把image预训练误说成image冻结。TTA全部旧模块eval/frozen，仅新129参数head更新；不更新BN统计或开启冻结RoBERTa dropout。实际发布权重完整训练配置不能只从文件名证明。 |
| CoT输入 | MLLM微调与下游GREAT训练是两件事；补充给InternVL LoRA rank16、7135例、4460iter/lr4e-5/batch4。GREAT输入图像、点云及离线对象/意图文本，human按逗号拆为3条、object一条且截断512 token。 | 使用原图对应离线CoT，不输入GT动作类别做更新。新48例human恰为3条。未重做现场InternVL推理；显存/计时不包含它。CoT错/歧义可能导致稳定错误，几何稳定不是功能正确。 |
| 原生指标 | GT>=.5；aIOU20个0..1阈值均值；AUC排序，SIM sum-normalize，MAE保留soft GT。 | 预测封存后独立CPU评分；固定每metric有效掩码、有效分母、全部正/负尾。不得按GT选择视图、步数、回滚。 |
| 原source选模 | unseen YAML val/test同列表，训练用val aIOU选best；issue14用户也提醒跨分区复用会泄漏。 | 当前固定source的训练/选模来源仍有不确定性，fresh confirm对TTA开发隔离不等于完全没被原source选模看过。不能宣称端到端完全untouched benchmark。 |

## 本轮竞争解释

1. **概率平均收缩**：它可压误激活，也可损失目标覆盖。固定区域面积不能证明定位正确。对照mean-prob、median-logit、q75-prob，区分参数适应和直接保留幅度的集成。
2. **损失/接口不匹配**：旧LN MSE/entropy未保留原focal+双侧Dice的前后景竞争。literal HM-only residual与HM+local-support residual配对检验；原损失用于pseudo target也不自动可靠。
3. **点序混杂**：先用固定点序/FPS-only变化与固定encoder索引控制区分；不把任意perm信号直接用于适应。
4. **稳定错误/知识条件错误**：如果强无更新median/q75仍胜过新head，或保护稳定的错误部位加剧负尾，不继续把teacher拟合包装成方法收益。

## 新样本与当前状态

`cohorts/COHORT_LOCK.json`已先锁confirm32、再取dev16，hash为 `0a59c0e7a542a06c7a20556e5abf0c10618bf8df6cf6076d056a80d3270ca930`。从2013行/1236点路径按XYZ规范hash、PCA符号等价、有限近形状比较及图像复用连通分组，排除旧12全部已知几何/图像组；dev-confirm无parent/component/image/geometry重叠，根已独立复核。它不等于真实mesh/物理实例ID证明，PCA退化、不同表面重采样、近重复场景仍是限制。

NPY容器含GT列，选择器打开/mmap过容器；只有前三列被逻辑复制到geometry缓存。没有把“使用含GT的容器”虚写成从未接触该文件。适应/分组/选参数据流不含第4列，评分器单独读取其GT引用。

目前只有CPU合成工程正控和坐标桥，没有round2真实预测/任务分数。首个代数重排HM移植未通过bitexact梯度断言，已保存`ENGINEERING_ATTEMPT1_INVALID.json`，随后按原操作顺序修正通过；不把这项工程失败当成方法阴性结果。GPU必须由根授owner后，通过统一flock执行。
