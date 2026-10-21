# HC64：单次纠正后的学生续走价值

本轮先回答一个问题：在学生实际到达的同一历史上，执行当前教师建议后，学生能否在后面的未观测帧表现更好？不修改奖励、候选、传播、时间分支、策略参数或训练规则，不先实现新蒸馏。

## 输入与固定选择

使用已曝光HC64前64父来源、每父1query，单一seed41；每query读取上轮最终B_OPD八权重并冻结。完整框候选库、Q4反馈、Strong时间区间全部逐位复用。新独立命名空间生成一条当前学生采样路径；不是GT挑选的失败路径，且另核原greedy轨迹逐位等于上轮B_OPD。

教师建议为同一真实学生状态上的Q4有效动作argmax。最多四处：以固定Strong事件节点的首末物理frame_id为上下界四等分，每格选择距中心最近的分歧点，同距取更早。必须是active节点，教师建议与实际采样动作不同，且后面至少存在一个active未观测节点；选择不检查GT。教师与无噪声学生argmax不同标为policy_disagreement；若两者相同但采样不同则标noise_only_disagreement，避免将探索错误直接等同于部署策略错误。

## 三个单动作干预分支

每个点从完全相同的原始历史出发，独立比较Student原采样动作、Teacher建议、Random合法非原动作。Random在合法非原动作中均匀选择，允许恰等于Teacher；仅有两个合法动作时两者必相同，按2/3动作支持分别报告，不能把这类机械相同当作独立证据。

每个分支只强制这一个动作；之后一直由同一当前学生参数与同一后续Gumbel数继续，每个分支依据自己的实际选择更新memory/previous/previous-feature。干预点之间不串联，不挑最好分支作为输出。逐位核共同prefix、相同incoming state、相同参数/噪声、Student分支完整等于原采样路径、Teacher=Random动作时两分支完整相同、inactive仅E且不写memory。

## 先封存，后离线GT

所有64查询的基准路径、全部预选干预和三分支框/动作/路径/receipt完整封存后，独立评分器才读取已锁GT。主指标是强制点之后(t'>t)的active未观测有GT节点sIoU，明确排除强制点本身。另列全部后续未观测、紧随的三个节点、三节点以后的远处、基准IoU=.5两侧的恢复/损害、后续实际动作变化。完整vIoU与强制点即时IoU只作辅助，不能用它们代替后续改善。

每个预选点等权；没有tail GT的点差值记0，仍计入该query的平均；无干预或全无tail GT的query差值记0。然后按query平均汇总全部64父来源，另报有可评分干预query子集。重叠tail节点和多干预不是独立样本；95%区间使用10000次父来源配对bootstrap。保留全部正负和严重损伤。

进入后续纠正轨迹蒸馏的事先资源门槛：all64主tail指标中Teacher−Student及Teacher−Random两比较95%区间下界均严格>0。否则本轮保存negative/inconclusive结果，不自动改变反馈、提高预算或进入新蒸馏。这是有限资源决定，不代表否定所有纠正轨迹或OPD机制。

## 耗时拆分

另在固定前16个旧输入上原样重放B_OPD/FixedKD/Direct。独立wrapper记录exclusive wall：历史构建、Q反馈、loss、backward、Adam/clip、greedy接纳与其余开销；嵌套计时不重复相加。参数、梯度、loss、每步接纳及最终权重/框与旧保存结果逐位核验，不一致即停。没有新科学配置或精度实验；只报告该小面板当前实现成本，不宣称OPD天然比FixedKD快，不把缓存后时延当端到端。

参考动机来自用户提供的[教师单动作干预分析](https://arxiv.org/html/2610.19145v1)和[Semi-OPD](https://arxiv.org/html/2610.11291v1)，两者主要验证语言模型，不作为本STVG实验成功保证。这里不额外加入固定初始学生轨迹蒸馏对照，按当前授权先检验纠正价值。

```bash
CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/correction_value/run.py --run
CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/correction_value/score.py --run-after-seal
```

所有原数据、GT、框路径、特征和临时状态只保留本地；唯一注册主方法、已完成旧结果及正式Vid/CG暂停不变。
