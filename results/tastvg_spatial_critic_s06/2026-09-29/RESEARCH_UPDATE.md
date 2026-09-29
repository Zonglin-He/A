# S0.6 → S1：critic有方向信息，首版persistent OPD任务增益近零

按18928796执行了两步：先用已有864条学生候选和Sa2VA缓存做S0.6；确认排序有信息后，按条件授权马上运行S1。旧3–5pp oracle门槛已撤销，没有继续扩候选池，没有新增专家或joint temporal实验。

| S0.6 指标 | Corruption | Clean |
|---|---:|---:|
| 全部候选对准确率 | **72.07%** | 73.15% |
| 四对antithetic方向准确率 | **68.50%** | 70.00% |
| antithetic来源级95%区间 | **[52.50%,83.50%]** | [53.33%,86.67%] |
| critic直接top1的sIoU增益 | **+0.5740 pp** | +0.5695 pp |
| 现有9候选oracle增益 | +0.8119 pp | +0.8101 pp |

这些结果支持“Sa2VA能为学生自己的局部参数方向提供偏好”。奖励只是非空reference mask框与学生候选框的平均IoU；不生成候选、不进入teacher回归loss。96cell中87有专家证据；corruption准确率用72cell/15来源，包含2592普通对/288正负对；另8cell无专家保留native计入16来源task分数。GT ties0，expert ties按半分；排除expert ties后的antithetic准确率为67.67%，CI[51.33%,83.00%]。若严格按三值sign相等、将expert tie记错，则为66.83%，CI[49.83%,82.50%]，后者跨50%；不同tie口径均保留，不能混称同一个准确率。先seal奖励和reward-only分桶边界，再读取旧GT scalar。

Margin不支持简单的单调结论：corruption低/中/高antithetic正确率55.60/82.38/72.61%，覆盖14/14/11来源。没有把分桶变成runtime gate。不同bin的来源组成不一样，也不能由这张表推出因果可靠性机制。

随后实际执行S1：1792维空间query＋final decoder LN参数跨arrival继承；六条独立16-arrival流，原序，流间source reset。专家位置0/4/8/12，24次scheduled中source0全空导致6次no-op，其余18次各做一步SGD，固定eta=.005、tauE=tauS=1。学习率沿用既有Vid-source接口的数值作为首版设置，旧优化器不同，不称最优。每次专家到达都围绕current policy重新生成9条候选，固定原方向/绝对radius1.57658；24组新候选中12组的中心已与source输出不同。非专家不读mask。

Reverse-KL的target只有detached student rollouts；Sa2VA只提供detached scalar奖励形成q。当前到达以更新前central输出评价，更新留给后续样本，避免混入专家直接rerank收益。Frozen与BudgetedRerank对照保留；在主nonexpert子集二者相同。没有teacher坐标回归、H更新、margin gate、clip、replay或临时改温度/步数。

| S1主结果：未来nonexpert，来源宏平均 | Corruption（60cell/12来源） | Clean（12cell/12来源） |
|---|---:|---:|
| ΔsIoU | **+0.000228 pp** | +0.000057 pp |
| ΔsIoU 95% CI | [+0.000091,+0.000389] pp | [−0.000121,+0.000229] pp |
| ΔvIoU | **+0.000089 pp** | −0.000004 pp |
| ΔvIoU 95% CI | [−0.000026,+0.000214] pp | [−0.000138,+0.000107] pp |

这是几乎没有实用意义的变化，不能凭极小正均值或sIoU区间偏正宣称方法成功。每条流最早3个nonexpert在首个有效更新之前；另列之后9来源的诊断，仍属于极小变化。六流复用同16曝光源，bootstrap只是描述性来源不确定性，不是独立stream确认。

机制核验没有发现“没更新”的情况：18/18 query和LN梯度非零，18/18 KL下降；actual SGD步长中位6.6831e−5、范围1.0148e−5至.0011599，中位步长/探测半径约4.24e−5。teacher q概率跨度中位.0025322，student p中位.0204415。参数确实继承，后续预测确实变化；这版概率尺度、SGD步长和稀疏更新组合尚未把critic偏好变成有用任务收益。这里没有单独隔离学习率、温度或compatibility geometry，不能唯一归因某一项，更不能写成OPD普遍不可能。

S0.6为CPU缓存分析，783独立reward几何＋32716公开排序/统计检查通过。S1 GPU进程47.92秒，18 backward、216新current-policy候选，新增expert/backbone capture均0；4个learnt-state完整回插、96状态链/6reset、18 KL/SGD算术重建、192双指标、3个新CPU合同与公开标量复算通过。独立KL回读曾因float64对FP32的2e−7容差过紧而失败（最大2.76e−7）；记录后将审计绝对容差改为1e−6，未改loss/梯度/预测，未重跑GPU。原log和恢复记录保留。

本轮保留“critic qualification通过”的正结果，S1记为“实现与持久更新有效、固定配置实用收益近零”。没有事后增加grid、temporal/joint OPD或生产晋升；CURRENT不变。后续若继续，应针对偏好分布到实际参数位移的转换提出一个明确最小改动，而不是重回candidate oracle门槛。
