# GREAT round2机制/协议草案

要检验的是：在固定原点序下，源模型对FPS起点的波动能否提供可用于更新的局部证据；任务式损失和局部支持下界能否在不规定区域面积的前提下，减少单纯平均引起的覆盖收缩。该机制尚未实测。

## 样本与隔离

旧12仅作已曝光开发证据。新cohort已冻结confirm32、dev16；confirm在任何开发评分前固定，不换确认集。按三列XYZ中心/尺度规范化及排序hash分组，并加入PCA/近形状与复用图像连接控制；分组描述及限制见COHORT_LOCK，不能声称已证明真实物理实例独立。类别/affordance仅用于预设分层和报告，不进入更新标签。每个parent/component选一条配对，保留原row与独立GT引用。

## 算子视图与source修正

每例K=5，固定原始点序及图像/CoT。视图seed为`(sample_seed + 104729*v) mod 2**31`，v=0..4。CPU torch RNG驱动原始FPS起点，旧模型eval、全部参数冻结。所有primary arm共享这5次真实forward和同一份64维pre-last-Linear hidden/logits；不把它当成5个独立测试样本。

source采用官方raw float64归一化后cast float32。dev另留一次legacy32归一化的source预测桥，使用同一首视图seed，不进入teacher或适应。这是工程复原差异，不是TTA收益。

固定encoder层级的native replay必须匹配fresh source。仅重排p0、再逆恢复用于索引敏感性诊断；同时重排图像49→2048投影的物理对应作为恢复正控。不把该cached诊断成本或效果混成部署输入不变性。

## 共享残差head与目标

取每视图64维hidden h，取物理坐标kNN16（排除自己）的`neighbor_mean(h)-h`，拼为128维。仅在当前样本K×N范围内逐通道中心化/RMS标准化（std floor1e-6），不使用跨样本状态。新增W(1×128)、b(1)，共129参数，直接零初始化，不消耗随机初始化。

`delta_v = 0.5 * tanh(W*x_v+b)`；旧模型logit z_v变为 z_v+delta_v。零head须与每个source-view概率和零更新median-logit读出bitexact。该上限是logit信赖域，不是目标区域面积。

对student视图v，只用其余4视图构造：

- `median_i = Q0.5(z_peer,i)`；soft teacher t=sigmoid(median)。
- 下界 `b_i=Q0.25(z_peer,i)`。
- 每peer的邻域中位数为`m_peer,i=Q0.5_j(z_peer,j), j∈kNN16(i)`。
- 稳定局部对比 `c_i=max(0,Q0.25_peer(z_peer,i-m_peer,i))`。
- MAD=`Q0.5_peer(abs(z_peer,i-median_i))`。
- `r_i=c_i/(1+c_i)*exp(-MAD)`。

全部分位数使用linear interpolation，FP32构造并detach。r是局部稳定证据权重，不等于预测正确概率；不按0.5或固定top面积选foreground。

损失为literal官方HM（focal alpha.25/gamma2 + 非标准2intersection的双侧Dice）加：

`L_support = sum(r*relu(b-(z+delta))**2)/(sum(r)+1e-6)`，lambda=1。

参数正则为`.001*(mean(W**2)+mean(bias**2))`。Adam lr=.01，固定3步；报告0步与3步，不按GT选步。每样本/每arm重建零head和optimizer，旧BN/dropout/statistics不变；无跨样本缓存写回。

最终读出为`sigmoid(Q0.5_v(z_v+delta_v))`，即所有适应视图的median-logit。不是把source-only读出偷偷拿来对比更贵的集成；零head对应同预算median集成。

## 对照与流程

同K、同输入、同source：首视图source、零更新mean-prob、零更新median-logit、零更新q75-prob；真实更新对照为同129参数的HM-only，以及HM+local-support。HM-only与完整机制只有support term不同。所有零更新强对照完整报告；开发集用aIOU在三种无更新集成中选定primary comparator，并在confirm前写独立decision冻结，confirm仍保留全部对照。

cachebuilder只读XYZ/image/CoT并保存hidden/logits/prob/operator trace；adapter单独进程只读无GT cache；scorer再单独进程在预测seal后读取GT列。预测NPZ沿用source/ensemble/adapted键，另加median/q75；0-step adapted应等于median baseline。每个case按parent/id关联，保留hash、有效分母、正负尾、无效run、梯度/参数/残差边界、源调用数和计时窗口。cached replay不是部署端到端速度证明。

GPU命令必须使用`flock -n <workspace>/research/round2/gpu.lock`且根先授owner。工程阶段任何source/zero-head/seed隔离不变量失败，保留无效运行并修复新命名版本；不得评分或静默丢例。Confirm固定，不因开发结果换样本；没有实际任务增量时不把loss下降、teacher收益或全局概率变低称作TTA成功。
