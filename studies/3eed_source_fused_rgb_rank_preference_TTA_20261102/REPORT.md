# R36：复用源训练融合模型的有限rank-TTA因子实验

**固定预算实验已完成，计量和完整模型闭环通过。RGB TTA通过事先规定的“超过自身frozen及同信息记忆”门，但RGB适应增量比无RGB小1.393059pp，且绝对精度明显低于原R30。保留门通过的事实，停止这个固定RGB分支；不把门改判失败，也不否定所有RGB TTA。**

本轮只用A208与已暴露DEV64，没有读U/FINAL，没有新源训练、额外教师或EMA。已有U确认不参与本轮选择或拟合。

## 固定模型、信息与更新接口

复用R25由WaymoVAL512预选的residual_rank seed0：RGB epoch1与匹配的noRGB epoch1。没有改选源epoch、家族或种子。noRGB是独立源监督训练的同结构模型，目标输入RGB置零，不是把RGB源权重在测试时简单清零当成匹配训练对照。源模型细节和历史负结果见上轮只读审查；R25的001错误缓存没有参与本轮。

仅更新已有rank MLP的12,929参数，其余source融合、gate、QKV、delta、原projection、点云/文字/RGB编码器与候选框全部冻结。两个mode分别独立初始化Drone/Quad，共4状态，AdamW lr5e-4、wd0、16步、beta1、tau0.1，总64步，没有扫参。

监督为原22条P/N局部偏好与原U96/93整行输出保持。使用实际推理native分数 s=logmass+alpha*rank。数值实现为s0+alpha*(rank_theta(u)-rank_source(u))，保证初始化逐位复现各自frozen分数；它是同一native函数的代数重写，完整路径已核对，没有拿旧noun log-odds替代实际rank读出。保持项相对各自frozen q0定义；原6条legacy B仅作状态诊断，不作为新source模型必须换目标的资格门。

A重复候选按各source-fused native分数固定代表；frozen分组选择与原raw argmax完全一致。A上同信息无更新对照是同观测直接QP；DEV上是跨观测记忆：每source native分数下margin0.1最小组L2投影，全部A queries，U行raw修正0，冻结200维rank输入作key，原GPU max/topk4取首邻k1，转移raw后当前q0锚定。没有额外R34硬门，没有新增负标签。

## 总体DEV结果

| 模型／读出 | Drone mIoU% | Quad mIoU% | 平均mIoU% |
|---|---:|---:|---:|
| RGB frozen | 21.497008 | 23.813667 | 22.655338 |
| RGB 同信息200维记忆 | 21.497008 | 23.813667 | 22.655338 |
| RGB rank TTA | 21.921824 | 23.813667 | 22.867746 |
| noRGB frozen | 20.478494 | 26.068754 | 23.273624 |
| noRGB 同信息200维记忆 | 20.478494 | 26.068754 | 23.273624 |
| noRGB rank TTA | 21.497008 | 28.261174 | 24.879091 |
| 原XYZT source | 20.653307 | 27.883355 | 24.268331 |
| 原XYZT R30 | 22.225026 | 27.956782 | 25.090904 |

RGB TTA相对自身frozen及记忆均+0.212408pp，通过原双比较门；收益只来自Drone（+0.424816pp，4条改善、2条恶化、12条query变化），Quad没有query变化。noRGB TTA相对自身frozen/记忆+1.605467pp：Drone+1.018514、Quad+2.192420，各有1条改善、没有IoU恶化，各3次query变化。

预先指定的配对机制量 ΔTTA_RGB−ΔTTA_noRGB 为 **−1.393059pp**。RGB源模型和无RGB源模型权重不同，该因子比较反映源训练/输入组合的适应差异，不能当作严格的单一RGB输入因果效应。当前结果仍削弱了“学生获得RGB后应更好迁移这些反馈”的具体假设。

RGB TTA绝对平均22.867746%，低于原XYZT frozen24.268331%和R30的25.090904%。noRGB TTA24.879091%也未超过原R30。不能把新source架构的变化与测试时增益混为一谈。

## 监督落实与有限预算

| mode／平台 | 末态偏好方向成立 | legacy B方向／最终正组 | rank参数L2变化 |
|---|---:|---:|---:|
| rgb / drone | 7/10 | 0/0 | 0.824430 |
| rgb / quad | 10/12 | 1/1 | 0.760442 |
| no_rgb / drone | 7/10 | 0/0 | 0.685546 |
| no_rgb / quad | 10/12 | 1/1 | 0.768939 |

参数更新、监督下降和输出变化均真实发生。RGB Drone偏好loss从首步前6.4486降至第16步前3.5657，Quad从1.8541降至0.9503；固定预算未充分落实全部偏好。此结果不证明接口容量在任何预算下都无法拟合，也没有按损失继续加步来挽救本轮。新的source q0和native分数与旧XYZT不同，不能直接用原B换目标含义判它“适应失败”。

## 证据、成本与限制

- A208的002 RGB缓存全部复用；只新增DEV64 RGB提取，耗时约4.24秒。没有提取R/S/T/U/FINAL。
- 272输入×2 source modes frozen表示捕获及准备总计约12.80秒；新rank拟合、QP和全部预测约6.35秒，均通过night_guard。
- 原R25 A208×2mode共416条raw native分数/argmax/框复现通过，最大分数误差7.63e-06。
- 独立CPU重放544个输入×mode行的完整native向量和全部QP/memory；最大误差1.28e-06，CPU/GPU query差异0，按保存的GPU邻居重算而未重新选邻居。
- 在任何本轮任务评分前，预先按输入hash固定A/DEV×平台4条，两个source mode各做source与adapted完整点云/RGB/融合读出，共16次真实前向，约11.14秒；非rank权重不变、框逐位相同、全部选中query重现。
- 全部544条输入×mode任务记录和457条负向比较保留；这些重复mode/控制不是新增独立样本。
- DEV64历史已暴露，Drone/Quad来自少数已有场景；本轮是固定开发机制测试，没有新独立确认或总体分布显著性主张。
- RGB记忆在DEV的任务均值与frozen相同：RGB只有1条query变化但IoU不变，noRGB无query变化。超过这个固定控制不等于战胜所有可能的非参数读出。
- ROI仍来自预测3D框投影，背景/邻物、可见性、原候选池及native yaw限制保留。融合不自动解决Qwen身份错误；所用低维ResNet ROI也未必拥有Qwen全部语义证据。

## 科学裁决与资源决定

**原方法门：通过。RGB主机制假设：在本次source家族、rank接口与预算下受到削弱。资源决定：不扩展该RGB分支，不提取T/U或FINAL，不扫层、步数、种子或源训练。** 原DECISION.json保持不变；RESOURCE_DECISION.json独立记录这一决定。它不意味着所有RGB-TTA无效，也不以几个负例替代总体指标。

论文原型与来源：WildRefer/xMUDA仅提供训练过的多模态接口和区分模态特有信息的启发，本轮不是两者复现。未加入新teacher/EMA。

主要入口：PLAN_LOCK.json、SUMMARY.json、PAIRED_MECHANISM.json、DECISION.json、RESOURCE_DECISION.json、CPU_AUDIT.json、ALL_NEGATIVES.json、SCENE_ROWS_PRIVATE.json、fit272_attempt001/FIT_RECEIPTS.json和closure4_attempt001/CLOSURE_AUDIT.json。私有权重、缓存、完整邻居和数据标识不公开导出。

## 固定代表读出的单独说明

全部A frozen raw argmax与分组代表读出一致。以下是适应后raw query与已登记固定source代表的差异计数，不是旧source未复现，也未替换官方预测或按GT挑读出：

- rgb/drone：{'fused_frozen': 0, 'rank_TTA': 6, 'direct_QP': 0}。
- rgb/quad：{'fused_frozen': 0, 'rank_TTA': 1, 'direct_QP': 0}。
- no_rgb/drone：{'fused_frozen': 0, 'rank_TTA': 5, 'direct_QP': 0}。
- no_rgb/quad：{'fused_frozen': 0, 'rank_TTA': 0, 'direct_QP': 0}。

完整别名与原始索引只保留在 RAW_VS_GROUP_READOUT_AUDIT.json 的私有明细。此项仅解释重复候选读出协议，不增加一组事后任务结果。


## 独立复核补充与计时口径

由未实现R36算法的报告／审计代理另行运行CPU审计，经night_guard守卫，不运行GPU、optimizer或autograd，也未读取U/FINAL。`independent_audit/AUDIT_RESULT.json`为PASS：4状态及原生source q0、原U189和22偏好核对；38组QP独立active-set/KKT重算；544条输入×mode完整rank/memory/query及2720个任务IoU、457条负比较、20个场景汇总复现。rank CPU64最大误差1.281786e-6，保存GPU近邻所选余弦最大差4.982526e-7，CPU最终query差0。旧R25 A208×2模式416条native分数、raw argmax及框复现。实际GPU近邻索引作为推理合同保留，CPU没有重新替换topk；4条完整模型闭环只复核原16前向回执，未由审计再次运行GPU，不能声称全部272条都重做完整backbone闭环。

表中legacy B方向／最终正组的0/0或1/1表示两个计数，各平台原有3条B，并非该平台没有B。独立审计的44条“mode×偏好”落实表与全部12条raw/固定代表差异已保留。没有追加raw读出的事后任务评分。

计时有包含关系：准备／捕获的12.8042秒从DEV RGB提取之前开始，包含4.2364秒提取；不能把这两项相加。拟合／QP／预测6.35秒与闭环11.14秒是其他阶段；审计、数据读取、旧源训练与历史缓存成本另计。这些不是经过部署优化的单观测端到端延迟。原预设gate=true与独立资源stop分别保存，本补充不更改预测、指标、登记规则或裁决。
