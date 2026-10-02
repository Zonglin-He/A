# P0：保留同状态 RKL 幅度，替换参数更新方向

**实际结果（future corrupt nonexpert）**：vidstg：E−A -0.6871 [-2.3992, +0.4388] pp，F−A -0.6537 [-2.3708, +0.4716] pp，开发优先臂 A；hc2：E−A -0.7692 [-2.9528, +0.7882] pp，F−A -0.9420 [-3.5475, +0.8571] pp，开发优先臂 A。区间为95%配对source-bootstrap；方法决定限于此曝光开发面板。

本轮实际执行 A/E/F 三臂：A 原 rank-RKL SGD；E 四组正负 probe 的 rank contrast 加权方向；F 最大绝对 rank contrast 的单轴方向。E/F 在自己的当前状态计算原 RKL 梯度，以 lr×全参数梯度范数确定位移幅度，不复制旧 A 轨迹的范数。原 teacher 温度和几何 KL 仍用于这一个幅度参照与诊断。

VidSTG 与 HC-STVG-v2 各原32个历史开发来源、一query/源、两序、clean+五类5%瞬时部署corruption、25%专家，每臂384到达，总2304。官方同域checkpoint、原Paper48像素和采样、原九个probe、原1792空间参数和两offset不变。Vid K1、HC K8、lr/temperature使用v3封存值。原Sa2VA、UniversalVTG与H缓存全部复用。当前预测在更新前封存；空间更新只服务后续arrival，时间Fast rerank不变。

全部六stream预测共同封存之后才读取已历史曝光的开发GT，完成独立参数算术/状态链/教师奖励/指标核验。这里的置信区间是10000次配对source-bootstrap，seed20261001；不把重复开发曝光结果写成fresh测试。

## vidstg

封存参数：`{"lr": 0.033761698432507946, "rho": 0.05, "teacher_temperature": 0.34902548789596055, "steps": 1, "student_temperature": 1.0, "direction_count": 4}`。本轮开发下一步优先臂 **A**，只按未来corrupt nonexpert源宏平均，未晋升生产。

| 臂 | 全部corrupt ΔvIoU vs Frozen (pp, 95%CI) | 未来nonexpert ΔvIoU vs Frozen | 未来 ΔvIoU vs A |
|---|---:|---:|---:|---:|
| A | +1.2384 [-0.4533, +3.3695] | +0.6652 [-0.4635, +2.3858] | — |
| E | +0.5569 [-0.5176, +1.9324] | -0.0219 [-0.0960, +0.0464] | -0.6871 [-2.3992, +0.4388] |
| F | +0.5876 [-0.4833, +1.9739] | +0.0115 [-0.0569, +0.0869] | -0.6537 [-2.3708, +0.4716] |

| 臂 | 相对A未来gross gain/loss (pp) | 相对A未来arrival损害>5pp | 首步唯一有益teacher首选但更新GT受损 | 无rank方向/eligible步 |
|---|---:|---:|---:|---:|
| A | 0.0000 / 0.0000 | 0 / 240 | 3 / 75 | 11 / 75 |
| E | 0.5185 / 1.2056 | 15 / 240 | 0 / 75 | 10 / 75 |
| F | 0.5415 / 1.1951 | 15 / 240 | 0 / 75 | 10 / 75 |

几何RKL与偏好方向的余弦、四轴投影、实际/参照位移、probe半径比、GT配对方向正反/中性、每一步及整个arrival固定时间空间效用、loss下降但GT受损、正负cases和完整计算量均公开。GT-neutral的正负probe效用并列不计作critic误选。

绝对probe半径 1.5765801926。零rank方向时E/F严格no-op，此时无法满足非零RKL幅度匹配，单独报告；A仍保留旧RKL的熵项行为。E/F轨迹分叉后其RKL参照梯度也不同，不能将此实验称为跨臂每个arrival完全相同的实际位移分布。

## hc2

封存参数：`{"lr": 0.006097133675874025, "rho": 0.05, "teacher_temperature": 1.0, "steps": 8, "student_temperature": 1.0, "direction_count": 4}`。本轮开发下一步优先臂 **A**，只按未来corrupt nonexpert源宏平均，未晋升生产。

| 臂 | 全部corrupt ΔvIoU vs Frozen (pp, 95%CI) | 未来nonexpert ΔvIoU vs Frozen | 未来 ΔvIoU vs A |
|---|---:|---:|---:|---:|
| A | +1.0970 [-1.3329, +4.4006] | +1.3149 [-0.9088, +4.6787] | — |
| E | +0.3323 [-0.9333, +1.7503] | +0.5458 [-0.1342, +1.7262] | -0.7692 [-2.9528, +0.7882] |
| F | +0.1743 [-0.9251, +1.2733] | +0.3729 [-0.0608, +1.1345] | -0.9420 [-3.5475, +0.8571] |

| 臂 | 相对A未来gross gain/loss (pp) | 相对A未来arrival损害>5pp | 首步唯一有益teacher首选但更新GT受损 | 无rank方向/eligible步 |
|---|---:|---:|---:|---:|
| A | 0.0000 / 0.0000 | 0 / 240 | 2 / 75 | 0 / 600 |
| E | 0.6963 / 1.4655 | 13 / 240 | 2 / 75 | 24 / 600 |
| F | 0.7370 / 1.6790 | 14 / 240 | 3 / 75 | 32 / 600 |

几何RKL与偏好方向的余弦、四轴投影、实际/参照位移、probe半径比、GT配对方向正反/中性、每一步及整个arrival固定时间空间效用、loss下降但GT受损、正负cases和完整计算量均公开。GT-neutral的正负probe效用并列不计作critic误选。

绝对probe半径 1.56435026619。零rank方向时E/F严格no-op，此时无法满足非零RKL幅度匹配，单独报告；A仍保留旧RKL的熵项行为。E/F轨迹分叉后其RKL参照梯度也不同，不能将此实验称为跨臂每个arrival完全相同的实际位移分布。

## 为什么减少坏步仍未获得未来收益

Vid首步唯一有益teacher首选却更新GT受损，A为3次、E/F为0次；HC全部步GT受损A为167次，E/F为78/119次。坏步减少也伴随有益变化减弱，不能只用坏例计数决定方法。

| 数据集/臂 | 每step平均fixed-time GT ΔvIoU (pp) | 平均normalized框坐标绝对变化 | 平均参数step L2 |
|---|---:|---:|---:|
| vidstg/A | +0.937705 | 0.02251689 | 0.43004376 |
| vidstg/E | +0.272812 | 0.00167618 | 0.41229004 |
| vidstg/F | +0.196325 | 0.00125331 | 0.40578599 |
| hc2/A | +0.044522 | 0.00108483 | 0.02076741 |
| hc2/E | +0.002071 | 0.00008662 | 0.02238659 |
| hc2/F | +0.003362 | 0.00006094 | 0.02174503 |

同状态的参数位移匹配，不等于框输出变化匹配。新方向平均框坐标变化比A小约13–18倍，原RKL范数仅约6.6–6.7%在四轴投影中；符合四轴子空间函数敏感性较弱的解释，但该posthoc观察没有单独建立因果。它阻止将此负结果解读为已排除所有direction机制，亦不证明事件参考采样已成为唯一瓶颈。框变化是封存normalized cxcywh的逐帧四坐标平均绝对差，无额外forward/GT读取。FUNCTIONAL_STEP_ROWS与独立标量核验可重建全部mean/median。

## 解释边界与成本

E/F的改动同时将更新约束到原四个probe轴；因此胜出可以支持“参数空间偏好方向有用”，不能单独证明所有坏例唯一来自KL的方向翻译。F按专家正负轴rank contrast选轴，不是teacher/student高disagreement选择。E合成方向并没有被专家直接评价；F沿轴的执行长度也未必等于probe长度，均无GT安全保证。

负均值且区间跨零只表明此锁定实现没有建立优于A的证据，不排除其他方向估计/幅度机制；正开发均值也不自动成为论文全量结论。保留原A正控、全部阴性和负例，不追加温度/学习率/预算搜索，不按本轮GT修改线上gate。

本轮零新backbone forward、零新expert inference。每个inner step复用缓存H，经TA-STVG后半段产生九候选、原RKL backward与post诊断；E/F没有省掉RKL的幅度计算。所有实际replay/provider/backward计数和worker wall time见COSTS.json，wall包含模型载入与保存/检查，不称纯GPU kernel时间。原批次的96次full parity forward不计入本轮新执行。

语言模型原论文仅提供方向信息检查的动机，本方法是直接参数偏好执行，不是原生token policy OPD的复现：[KL-free OPD](https://arxiv.org/abs/2609.33791)。[Decomposed OPD](https://proceedings.mlr.press/v306/yoon26f.html)研究VLM视觉/语言梯度分量，不提供本STVG配置的理论保证。

事件条件专家取帧与定位质量critic是尚未执行的后续问题，本轮没有自动加新专家输入、时间参数学习、模型、baseline或全query队列。生产注册保持原值。

