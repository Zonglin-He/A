# HC64：真实临时状态候选OPD

本轮原由用户授权HC128，运行中缩为64父来源：固定选择原HC128输入顺序前64、每父1query，不按新结果选样。方法、参数、每query预算与在途源码不变；独立scope64控制器在第64条完整写出后封存并停止原worker，后64取消。原128配置与锁保留作执行来源，实际结果只登记HC64。唯一注册方法不改，原几何OPD结果保留；正式Vid8405/10303与CG持续暂停。方法、参数及预算在任何新预测前锁定；64来源范围按用户要求于运行中缩减。没有用新结果选参数或样本，不开展温度/学习率扫描。属于反复曝光的开发面板，不能作独立确认。

## 固定状态空间与真实候选

四个新方法均使用同一个32维变量z，形状8×4。固定基底P为缓存Source的全部有效encoded_feature token（encoded_mask=False）中心化协方差的前4个特征向量，CPU float64计算，列符号按绝对最大元素为正确定。P不读取教师或GT，不训练任何网络。物理content状态为：

    eta(z) = (z P^T) * 4 / sqrt(16 + ||z||_F^2).

总content范数小于4，零点导数为固定基底P的正交嵌入，即delta eta=delta z P^T；它不是32维到2048维的单位映射。每query、每方法从z=0和新Adam开始。所有新方法共享这一可达空间，不能据结果泛称超过所有2048维Direct。旧几何OPD使用原完整空间，仅是原方法性能参照。

StateOPD/DirectMC每轮从当前z采32个真实候选，sigma=0.1，16个独立高斯噪声及其反向对；两方法使用同seed41、query、round键的同一噪声流。每个候选必须经冻结decoder生成完整轨迹，不能拼接逐帧随机框。候选前向使用eta(z+sigma*eps)。Source编码、RGB与专家前向均复用旧缓存。

## 同一任务分数与轨迹保护

固定使用原ReferDINO已经选中的跨帧query及八位置mask。mask logits先按官方规则插值至真实resized输入尺寸，再sigmoid；同一身份跨帧绑定，不逐帧换query，不把其他query自动作为负例。空mask只按固定soft mass≤1e−6识别，不按GT或结果筛选。

目标范围分数为mask的x/y最大投影与候选框在两轴范围的soft Dice均值，再对有效观测位置平均。轨迹保护使用完整物理网格，不按GT事件或Strong区间筛点：

    delta_t = (chart(B_t)-chart(E_t)) / (Ew_t, Eh_t, 1, 1)
    d_t = RMS(delta_t)
    h_t = relu(d_t - 0.25)^2
    U = mean_anchor projectionDice - 0.1*(0.5*mean_t h_t + 0.5*max_t h_t).

chart=(cx,cy,logw,logh)，E为固定ERef轨迹。保护限制没有新证据支持的大偏移，不提供未观测GT正确性保证，也可能限制有益大纠正。没有分割/遮挡GT；教师框与mask可能共同错选，不能宣称身份已经解决。

## 更新与共同接纳

统一tau=0.01，Adam lr=0.014、betas=(0.9,0.999)、eps=1e−8。不对候选奖励做额外标准化或截断。

| 方法 | 更新目标 |
|---|---|
| StateOPD | 真实候选U/tau的softmax权重，停止候选和权重梯度，拟合latent高斯均值；不经decoder反传 |
| Direct | 直接优化−U/tau，保留上游原reference梯度截断 |
| DirectMC | 相同候选与tau的−logmeanexp(U_k/tau)，流式求每个候选的路径梯度，不重复forward |
| DirectFullRef | 同Direct，但仅在隔离decoder实例恢复层间reference梯度，前向必须不变 |

每次Adam提案之后，沿该delta按1、1/2、1/4、1/8、1/16依次真实解码，首个U>当前U+1e−8的提案接纳。全部失败则恢复z及Adam全部历史；Point/FullRef停止，避免重复相同提案；OPD/MC可在下一轮刷新噪声。拟合后的均值也必须重新评价，不能用高分候选代替其真实分数。

## 预算与读出

每方法每query最多1321次完整轨迹评估，包括初态、候选、梯度点、提案与所有回溯。OPD/MC最多40轮；Point/FullRef最多660次优化尝试。若需回溯，实际轮数可能减少；预算不足32候选＋至少1次提案验证时停止。无padding调用。构造及原/完整参考图数值兼容性检查单列为setup。

这是同前向预算上限，不能称相同wall成本：MC和Direct还有不同数量的反向，FullRef图也更大。分别报告完整轨迹评估、两个offset实际调用、反向次数、GPU秒及峰值显存。缓存捕获旧成本不能冒充当前完整部署成本。

全部使用旧Strong分布及原MBR区间，空间轨迹来自最终接纳状态，不从GT挑候选、方法或停止步。保存所有正负query。全部选定64条预测、receipt、trace独立封存且原worker退出后才独立CPU读取原开发标签，报告完整vIoU、tIoU、观测与未观测实际节点sIoU、改善比例、>5pp退步、父视频配对区间及实际成本。

原worker入口为scripts/state_opd/run.py，HC64边界控制与CPU评分入口为scripts/state_opd/scope64.py --watch --score；固定配置为scripts/state_opd/config.json；结果独立保存在outputs/state_opd_hc128，公开结果导出至results/state_opd_hc64。旧实验和正式队列均不覆盖、不恢复。
