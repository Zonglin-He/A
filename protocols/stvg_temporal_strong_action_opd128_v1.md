# R26：Strong基底上的观察尺度Action条件OPD（实现草稿，尚未预锁/运行）

## 范围与新的依据

用户授权自主继续时间/联合方向，空间实现/config及E/Live41/42/43坐标全部冻结。不加专家/检查点、不训练新网络/头；同TA-STVG Vid Source SHA `5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83`。只使用已有RC128全部曝光开发父query和原65物理节点/既有八专家证据，不取TC109选择方法。

R24完整节点数条件化：教师期望GTv +.833746pp、CI正，硬−.482295pp、CI跨零，未达原>=2pp门槛，相对强参照expected CI也跨零；该配置保持关闭，不追认通过。R25条件MC无净增量，保持关闭。R14 HiResAuxMix完整v +1.606434pp、CI正，为现有强静态参照。R6实际448 RGB前向后的第二pass Action已缓存128，与224同65物理网格，128曲线均不同，RMS差均.779162；std均1.474406/1.469095。这只证明响应不同，不证明更准确或原视频均含更细像素。旧HiResFull/Fluct已经使用448 Action（HiResFluct +1.556405pp、CI跨零），不能称为第一次使用。

此轮新开发决定：把既有观察尺度变化的task-trained Action证据，用同一个完整节点数条件化规则，在固定强时间基底上比较，并完整执行实际Direct/Fixed/Live。它不同于R24的原Native基底/224证据及纯能力检查，不恢复温度/先验强度/步数菜单。不以hard成绩为执行实际OPD的前置；同时承认R24 expected门槛也未达，不能仅靠撤回hard门槛重新包装旧配置。

## 唯一候选与匹配控制

令B为预先固定、精确复用的HiResAuxMix完整后验。B同时是教师构造基底和学生零临时状态的时间分布；Native原始后验单独保留。只改变时间，空间零/最终坐标都不变。

k=j-i+1（包含节点数，非精确物理时长），E_H(I)=sum原448 Action logits[n],n=i..j，E_N为224同式。唯一新教师T_H(I)=B(k) B(I) exp E_H(I)/sum_{J:k(J)=k} B(J) exp E_H(J)；匹配T_N用E_N。原B正支持不增、不加epsilon，系数1、原raw单位、无median/MAD/新温度；组内log-sum-exp稳定计算。仅保留目标后验的B(k)，不保证拟合后的学生或最终MBR动作长度保持。原完整2080动作MBR不改。

所有实际学生均p_eta(I)∝B(I)exp(u_i+v_j)，130临时bias独立重置，无网络参数加载/更新。沿用已审计`stvg_temporal_conditional_opd128_v1.py`的原定义：DirectJoint为完整目标joint KL；DirectConditional为teacher-forced状态的双边界条件KL；fresh Fixed每步从初始B采样32个区间，Live从当前p_eta采样32个区间。每个采样区间的起/终状态请求构造教师相应两份最多65类条件分布，不是32个bbox IoU奖励或decoder产生的完整视频轨迹。固定教师权重/曲线不妨碍按当前边界状态查询新的条件目标。学生分布与teacher分别冻结/保留梯度，实际样本、目标digest、eta/梯度/optimizer链/最终后验均保存。

T_H、T_N分别完整运行TeacherDirect、DirectJoint、DirectConditional、Fixed41/42/43、Live41/42/43。固定40步、Adam lr .1、M32、原loss .5比例、三seed，配对随机流；不因直接读出不好跳过。NoAction=T=B应精确no-op，实际调用helper验证零update，B及Native静态对照保留。WrongH与ShiftH为控制，不作eligible候选：WrongH复用R18/24的固定下一alias循环供体映射，用448曲线按归一化物理时间插值并匹配受者mean/std（按原退化规则），ShiftH固定roll32；两者同B同节点数条件化、各自只跑Live41/42/43，并记录TeacherDirect。不得用Wrong224冒充Wrong448，不重选供体/错位量。

本轮最多每父22个非平凡fit（两教师各8，加两错对应各3）、另一次NoAction exact资格调用；128父2944fit、最多112640实际update，所有无支持/flat/noop仍保留，实际计数为准。TeacherDirect等静态控制不算学生fit。模型与专家新调用0。旧448观测成本来自R6，部署不能把这份证据说成免费或原224单次开销。

## 核验与评价

先完成实现、锁前审阅/性质检查，再prepare锁全部code/config/I/O/评估依赖SHA及原128 High/Native Action、Strong、源空间、Wrong映射缓存。首2父全矩阵无GTCPU测速，保守maxcase×128+30<900s，否则止，不缩矩阵/改参数。每阶段CPU900s、own非阻塞run.lock、host6GiB/disk30GiB保护，0GPU。失败保存实际已完成partial、完整调用/更新账；不信号其它任务。

性质：零eta exact B、NoAction helper exact无更新；T_H/T_N/WrongH/ShiftH目标归一化/共同支持/B(k)守恒/常数或任意只依赖k项消除。记录H/N教师TV和仅常数差是否消除，不能仅用curve RMS宣称信息不同。DirectJoint有限40步不是最优投影；130bias未必表达节点数归一化项，记录target与final joint KL、两端marginal、节点数TV、实际更新/eta范数，不扫拟合程度。

全部128所有teacher/q/轨迹/实际适应链先封存GLOBAL barrier后才新GT。评分前用独立NumPy重算目标组式/节点边际、gradient/Adam、fresh采样和当前状态链、首步Fixed/Live exact、反馈条件digest；复核B/Native与原结果、所有空间数组exact。官方Dense指标和所有2080候选质量、全部signed/unknown、正负贡献/严重尾/留一及成本。主随机方法取时间/空间同seed41/42/43配对后父内均值；静态/Direct在三个冻结空间上父内平均，保留全crossseed输出，种子不当独立父。额外全9配对只作预登记敏感性，不能按它更好替换主值。

报告总收益LiveH−Native；本轮增量LiveH−B；OPD价值LiveH−DirectJointH/DirectConditionalH/FixedH/TeacherDirectH；观察尺度价值LiveH−LiveN；错对应控制LiveH−LiveWrong/LiveShift。Teacher期望候选质量独立报告，不代替实际轨迹收益，也不能因teacher hard失败否定已运行OPD。

唯一LiveH总完整v>=2pp且CI正，且对B/两个Direct/Fixed/TeacherDirect及LiveN/错对应控制均CI正，才冻结一次父分离确认配置；同时看未观测实际节点固定空间sIoU不变的事实及严重尾，不制造空间增量。若静态teacher有收益而Live无额外价值，只能支持时间证据/静态读出，不叫OPD贡献。未通过不按结果扫温度/步数/基底mix/权重/供体，不挑控制晋升。RC128开发选择不冒充独立确认。

完成/失败同turn研究总档案check snapshot check；GitHub allowlist发布完整代码/配置/结果/负发现，双ref freshCAS与实际HTTPbytes/SHA/blob/tree收据，排除私有媒体/GT/权重/原始缓存/文本和父ID。空间/CURRENT/MAIN/PAPER不自动晋升。代码已实现为锁前草稿；尚无PLAN_LOCK、实际128父fit或新成绩。

锁前工程补充：新helper逐句沿用原Policy/Adam算术，仅插入预算和progress observer；十二个四模式/三seed玩具用例的q/eta及所有trace字段须bitexact原fitter。逐fit attempt、逐Adam已返回步和完整trace步分别计数，异常保存当前fit已完成CPU trace与已完成其它输出。NoAction名义5120跳过步与最大112640非平凡update分开记账，117760只称包含NoAction的名义步预算。measure安装实际HC标签文件拒读audithook。
