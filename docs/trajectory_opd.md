# HC64：跨帧证据与多模式真实轨迹 OPD

状态：隔离研究候选。不是已验证的主方法。旧几何主方法、全部正负结果和暂停的正式任务保持原样。

当前事实是：恢复完整8×256状态后，旧mask投影StateOPD的完整vIoU为31.1395%，Direct为31.4752%，几何OPD为31.3868%。没有建立OPD优势。此次不再把局部参数收缩解释为唯一问题，也不声称换成全轨迹评价就必然成功。

## 证据先于学生更新固定

只使用缓存的同一ReferDINO选轨query、同八帧mask，以及当前Source全部65物理节点的融合外观网格。用encoded_pos严格恢复网格，排除文本和padding。没有新的RGB、Source编码器、专家或网络训练。

在同一Source特征空间，以L2特征余弦、温度0.1和固定top20关联，从每个节点前后最近的观测mask独立传播。前后传播的soft Dice一致性与返回锚点的cycle Dice共同形成可靠性权重。保留soft mass，不将模糊证据重新归一化成强监督。mask、对应矩阵、置信度在更新前固定。它们是伪证据；循环一致不等于身份正确，教师框与mask也可能共同错选。

该机制参考[DINO官方视频传播实现](https://github.com/facebookresearch/dino/blob/main/eval_video_segmentation.py)与[Space-Time Correspondence](https://proceedings.neurips.cc/paper/2020/hash/e2ef524fbf3d9fe611d5a8e90fefdc9c-Abstract.html)。这些工作的对应特征和监督条件不同，不能把其准确率继承给TA-STVG融合特征。

旧缓存只保留Strong最终区间，没有完整Strong后验。本轮把同一固定区间的节点indicator作为事件先验，不声称获得了独立事件证据；Strong时间输出与原MBR始终不变。

## 全轨迹奖励与未观测帧

对真实decoder轨迹B计算：

    U(B) = 0.5 observed_range(B)
         + 0.5 mean_unobserved [event_t confidence_t range(B_t)]
         + 0.1 mean_adjacent [event_pair confidence_t confidence_prev link(B_t,B_prev)]
         - 0.1 whole_track_guard(B,ERef).

range为同一soft目标mask的x/y投影Dice。link比较候选框内目标支持与前帧支持沿固定视觉对应的传输，依赖整条轨迹，不能逐帧任意更换教师query。分母使用固定节点数而非置信度之和；无证据项为0。whole_track_guard沿用中心/对数尺寸相对ERef、deadzone0.25、mean/max各半的保护。

未观测帧有两条明确的新通路：奖励直接评价该帧的实例支持；8×256临时content节点通过物理时间插值、乘固定证据置信度后注入所有有效物理节点，两个offset仍分别由原冻结decoder执行。锚点保持one-hot，事件外未观测节点不直接注入。零状态必须精确恢复Source/ERef。该表示是共享纠正状态，不是可解释的身份标签，也不保证未知遮挡处正确。

## OPD的改动

策略包含三个独立的高斯content状态模式，均为8×256，初始均值是0及一个固定seed扰动的正负方向。三个均值始终分别经过decoder，不在模式之间取平均。sigma固定0.31，沿用前轮无GT校准所得尺度，不开展新扫描。

每轮从当前三模式策略分层抽32个反向成对真实状态，执行完整decoder并得到U。条件奖励温度按当轮奖励的ESS=16/32确定；平坦反馈保留均匀权重。各模式的更新为：

    w_ck = softmax_k(U(B(z_ck))/tau)
    m'_c = sum_k w_ck z_ck.

这是固定方差下经验奖励加权Gaussian最大似然的闭式解，近似连续奖励倾斜目标的条件KL投影，不经decoder坐标梯度。不是将原子经验分布对连续Gaussian的KL当作有限量直接求解。模式概率按带当前先验的有限候选partition function更新；分层采样量随模式概率改变，同时保留每模式最少三对探索。概率数值下限1e-12仅防止浮点吸收为零。这与reward-weighted regression和分布搜索有关，不声称仅凭该形式获得新的定位信息或普遍优于Direct。

每个新均值必须重新执行完整decoder并通过相同U的回溯接纳；候选高分不能代替均值的实际后果。最终使用U最高的**已适应模式均值**经decoder得到轨迹，包含零状态模式作为初始保护；不输出最佳单个采样候选，不用GT挑输出。DirectMulti和DirectMC具有同样的初始模式与同一读出。单峰局部线性条件下仍可能接近预条件梯度，OPD并没有普遍优越性保证。

## 公平对照与执行锁

一次HC64固定开发面板、每父一条query、seed41。保留几何OPD；新比较TrajectoryOPD、Direct、DirectMulti、DirectMC。Direct从零状态开始；Multi和MC获得相同三个初始模式。所有Direct恢复上游层间reference梯度，获得完全相同的固定证据、状态传播、可微奖励、回溯与整轨迹保护。没有人为detach、硬化框奖励或限制Direct可用的信息。

另保存InitialModes读出：只对相同三个初态均值评分、选U最高者，不进行适应。它不新增模型运行，用于区分随机多初态/读出和后续策略更新的收益。DirectMC优化每模式各自的条件 `-logmeanexp(U/tau)` 路径梯度；它不是整体混合partition function对均值的精确欧氏梯度，不能混称。所有方法共同优化同一U并以相同规则验证真实均值，MC条件权重与OPD相同。

每方法上限1321次完整轨迹评价；OPD/MC配置最多40轮，但三模式每轮至少35次、初态3次，因此预算实际最多37完整轮，回溯可能进一步减少。Direct系列最多660次更新尝试，DirectMulti的三初态也计入预算。lr0.014只用于Direct梯度对照；OPD使用闭式条件均值投影，不使用学习率或Adam。Direct的独立Adam状态逐模式回滚。相同预算上限不等于同实际耗时；单独记录反向、setup和证据构建成本。候选预算计入所有均值验证和回溯。温度的ESS16目标使用模式内中心化后合并的32项；真实条件权重ESS另行记录，不声称均等于16，离散相同奖励还可能使目标不可达。

完整64条先封存，独立CPU才能读取开发标签；不评分未完成前缀。报告完整vIoU、观测与未观测sIoU、全部正负/严重下降、父视频配对区间、真实成本，以及证据置信度、模式保留、最终纠正幅度。若新证据同时帮助Direct而OPD仍不优，不能把收益归给OPD。

入口：`python -B scripts/trajectory_opd/run.py --run`。配置：`scripts/trajectory_opd/config.json`。私有产物：`outputs/trajectory_opd_hc64`。主配置不修改，正式Vid与CG不恢复。
