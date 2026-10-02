# H-lite：学生时间共识条件化空间偏好

**完整结果（未来 corrupt nonexpert）**：vidstg H-lite−A -0.4891 [-1.1995, +0.0693] pp；hc2 H-lite−A -0.3294 [-2.1077, +0.8787] pp。95%区间为配对source-bootstrap；这是历史曝光开发面板，不是fresh验证。

本轮只比较A与H-lite。A保持原Rank-RKL；H-lite从当前central学生的原生≤8去重时间候选产生等权共识w，停止梯度后同时加权原五均匀Sa2VA有效参考奖励和native系数L1/GIoU几何。原rank teacher、RKL、普通SGD、学习率、温度及K不换，非空flat rewards保留原来的loss语义。

每数据集原32历史开发源、一query/源、双序、clean+五类5%瞬时部署corruption、25%专家；384到达/臂，共1536新到达。官方同域TA-STVG checkpoint、原Paper48帧/像素、H、Sa2VA/UniversalVTG缓存、两offset、九probe和1792空间参数固定。VidK1/HC K8每步刷新central时间候选和空间rollout；每arrival专家只fetch一次，当前输出封存后才更新空间state。Temporal Fast输出仍正常rerank，w不使用其分数或hard选区。

缓存参考上valid weight mass=0时加权reward无定义，按无证据不更新并记录原因，不加平滑/全clip回退或GT gate。H-lite不能产生缺失的事件参考；权重归一化也不能表示绝对证据可信度。候选共识不是已经校准的event posterior，相关候选可能给出集中但错误的支持。

四stream全部预测共同封存后才CPU读取同一批已有曝光GT。主指标是未来corrupt nonexpert源宏dense ΔvIoU，paired10000 source-bootstrap seed20261001。每组先按source/order/condition聚合、再条件/序/源等权；不把重复arrival当独立source。A每个384全流输出、梯度和状态逐值复现e3578af。

## vidstg

封存参数：`{"lr": 0.033761698432507946, "rho": 0.05, "teacher_temperature": 0.34902548789596055, "steps": 1, "student_temperature": 1.0, "direction_count": 4}`。开发优先臂 **A**，依预登记strict future mean规则；没有生产晋升。

| 臂 | 全部corrupt ΔvIoU vs Frozen (pp,95%CI) | 未来nonexpert ΔvIoU vs Frozen | 未来 ΔvIoU vs A |
|---|---:|---:|---:|
| A | +1.2384 [-0.4533, +3.3695] | +0.6652 [-0.4635, +2.3858] | — |
| H | +0.7033 [-0.6920, +2.3398] | +0.1761 [-0.6344, +1.3394] | -0.4891 [-1.1995, +0.0693] |

| 臂 | 未来gross gain/loss vs A (pp) | 局部GT gain/loss steps | 有益selected目标却执行受损（全部/首步） | >5pp未来受损到达 |
|---|---:|---:|---:|---:|
| A | 0.0000/0.0000 | 40/10 | 3/3 | 0/240 |
| H | 0.4476/0.9368 | 34/11 | 8/8 | 13/240 |

| 臂 | corrupt首步非空参考arrival | 首步参考完全没落在GT支持 | 时间共识下参考mass=0（A仅离线诊断） | 实际可更新步数 | 平均局部固定时间ΔvIoU(pp) |
|---|---:|---:|---:|---:|---:|
| A | 75 | 20 | 17 | 75 | +0.937705 |
| H | 75 | 20 | 17 | 58 | +0.724111 |

H-lite−A：区间跨0，方向未确认。H-lite实际 17 个corrupt step因数学零加权证据不更新。
primary未来子集 240 到达、32 来源；两序均为expert的source只从未来子集排除，保留在全组。

## hc2

封存参数：`{"lr": 0.006097133675874025, "rho": 0.05, "teacher_temperature": 1.0, "steps": 8, "student_temperature": 1.0, "direction_count": 4}`。开发优先臂 **A**，依预登记strict future mean规则；没有生产晋升。

| 臂 | 全部corrupt ΔvIoU vs Frozen (pp,95%CI) | 未来nonexpert ΔvIoU vs Frozen | 未来 ΔvIoU vs A |
|---|---:|---:|---:|
| A | +1.0970 [-1.3329, +4.4006] | +1.3149 [-0.9088, +4.6787] | — |
| H | +0.7648 [-0.7172, +2.4737] | +0.9855 [-0.1247, +2.6414] | -0.3294 [-2.1077, +0.8787] |

| 臂 | 未来gross gain/loss vs A (pp) | 局部GT gain/loss steps | 有益selected目标却执行受损（全部/首步） | >5pp未来受损到达 |
|---|---:|---:|---:|---:|
| A | 0.0000/0.0000 | 393/167 | 19/2 | 0/240 |
| H | 0.6659/0.9953 | 440/80 | 14/0 | 5/240 |

| 臂 | corrupt首步非空参考arrival | 首步参考完全没落在GT支持 | 时间共识下参考mass=0（A仅离线诊断） | 实际可更新步数 | 平均局部固定时间ΔvIoU(pp) |
|---|---:|---:|---:|---:|---:|
| A | 75 | 10 | 5 | 600 | +0.044522 |
| H | 75 | 10 | 5 | 560 | +0.071975 |

H-lite−A：区间跨0，方向未确认。H-lite实际 5 个corrupt step因数学零加权证据不更新。
primary未来子集 240 到达、30 来源；两序均为expert的source只从未来子集排除，保留在全组。

## 根审查后的具体判断

两个数据集H-lite主指标均低于A，配对95%CI均跨0；目前没有建立优势，也没有统计确证总体伤害。按预登记规则两集仍保留A作为开发比较基线。

Vid的学生支持在首步平均覆盖63.04%的GT采样帧，归一化支持权重仅47.36%落在GT计分帧；这些是离线覆盖读出，不是正确概率。75个非空参考arrival有17个加权质量为0，其中5个的原参考实际含GT事件帧。H-lite只能抑制已有证据，不能生成新的事件内参考；当前学生支持有时还会排除原本在事件内的参考。H有效更新75→58，有益selected目标却执行受损3→8，未来相对A超过5pp的受损到达13/240。

HC学生首步GT帧覆盖较高（94.11%），共识下加权参考质量为0的5个arrival均没有原GT参考。H的局部受损步167/600→80/560、局部平均固定时间增益+.04452→+.07198pp，但未来nonexpert净收益仍低于A。局部支持改善没有自动转化成跨query迁移收益；不同轨迹的计数不证明共享状态是唯一原因。

本P0既发现Vid事件支持可能漏掉已有有用证据，也发现HC局部更新改善与future结果不一致。它没有证明只要把temporal branch置于上游就能增益，亦没有测试重新在事件内调用Sa2VA的H-full；后者是不同的证据收集干预。

上述新解释只读取本轮封存匿名标量。GT不用于线上权重、门槛、参数或后续自动调度。

## 机制解释与边界

H-lite同时改变reward和geometry支持，是一个匹配的支持干预；它不能单独把变化归因于reward或loss哪一侧，也不验证重新取事件帧。GT support、selected有益却执行有害、full-D下降而事件内D上升等都只作离线诊断，不作为在线选择规则。不同臂走不同on-policy状态轨迹，局部counts分母/有效步数不同，必须与首步及净future指标共同看。

H-lite通过候选共识归一化空间支持，可能集中错误支持或抛弃uniform cache仅有的证据；归一化不表示证据覆盖可信度。全局shared空间state、critic误排、有限步长和几何代理错配仍是竞争解释。不能用这一次开发均值否定全部event-conditioned监督、OPD或global-state路线。

Temporal在框架中既保留当前时间输出修正，又为H-lite提供上游学习支持；本批没有新增temporal参数更新、confidence gate、entropy multiplier、质检网络或专家推理。当前selected tube只是诊断参照，H-lite实际目标仍是Rank-RKL分布，没有悄悄采用G hard target。

A的时间支持诊断仅第一步有已存六层候选；H每inner step观察。first比较具有相同arrival分母，A后续step的参考GT覆盖可复算但没有重新执行decoder补充support。full-D与GT计分支持D均从封存框在CPU重建。候选target也被停止梯度，w不通过边界求导。

clean/expert/all分组、全部匿名arrival和每步记录、正负cases、gross与严重尾部保留。formal worker wall包含checkpoint载入、I/O及核验，不称纯GPU kernel时间；smoke等资格检查时间没有混称为formal inference时间。

首个no-GT smoke的独立reward auditor以FP32计算而原overlap使用FP64，最大差1.0493e−7。原log/source/lock保留，修复只改CPU独立复核精度、没有改模型目标/权重/更新，revision001后重做两集smoke再正式执行。

本P0控制器不自动启动H-full新参考帧、critic calibration/reliability耦合、context-memory、grid、新模型或旧fullquery任务；资源决定与机制证伪分开，当前部署注册未变。

