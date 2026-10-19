# R27：仅开放冻结时间解码器的跨 offset 自注意力

2026-10-20。用户允许持续时间/联合探索，空间实现、配置和E/Live41/42/43坐标冻结。不新增专家、权重、网络、训练头或网络参数训练。仅现有TA-STVG Vid检查点和RC128曝光开发缓存；TC109不选择方法。上一R26完成实际Direct/Fixed/Live，Live−Native+.809812pp、CI跨零，−Strong−.796622pp、CI跨零，Live−Fixed+.799105pp、CI正，但未超过直接拟合；其配置关闭，不调Action强度/温度/步数救援。

## 可证伪机制与历史区别

当前模型在33/32两组交错物理帧分别作TimeDecoder temporal self-attention，最后用原两流envelope概率合并。它们虽覆盖同一段视频，每个流的decoder状态不能读取另一个流的时间状态。本轮只打开此通路，不重编码视频、融合或重建query。R8 joint_motion仅改变VideoSwin编码；R11 full_context同时改变全部编码/融合/query/位置和单流概率，不能隔离本通路。旧阴性保留，跨流有用仍是未验证假设。

保留每流原缓存第二pass query_tgt/query_time、原逐帧motion/text CA memory/pos/mask和六层原参数。每层先从两个流同一层输入状态同时计算SA：流s的Q仍为自身tgt+原time；K/V先连接自身，再连接另一流的tgt+原time/tgt。SA之后各流独立调用原norm1、逐帧CA、norm3、FFN、norm4以及原最终norm/head，形状与原流一致。不是让本流先更新、再被另一流读取。原query_content未被原TimeDecoder实际使用，仍保留不注入。

不新建全65位置编码；原两个offset的局部索引/归一化可能相近、重复或略有不同，这属于本轮保留的已训练输入，也限制跨流时间解释。初始每流query常量，第一层跨流量不代表读取到了逐帧视觉差异；真正逐帧信息主要从CA后进入后续SA，必须报告逐层跨流注意力质量和内容响应。改变读取范围可能仍损坏已训练机制，不能承诺2–3pp。

唯一能力候选CrossOffset，WrongBinding为控制：每流SA读取另一流时，将另一流当前content及其V沿时间固定循环移动floor(T_other/2)，位置码留在原slot。Own Q/K/V不动。不能把另一流K/V/pos共同排列，那会因attention置换不变成为伪控制。错配不是新的RGB证据，第一层常量content时可能无差，保存实际差异。

## 最小矩阵与读出

Native（逐流原生decoder），CrossOffset，WrongBinding，以及现有强静态HiResAuxMix。新两条件只用各流最终原head logits，仍经原probabilities→mass_table两流envelope→同原65/2080动作MBR。不改为65单流start/end乘积，不加mix/温度/新支持/边界修补；不按结果挑控制。Native/Strong完整后验与interval精确复用。模型/空间不更新，本轮只是教师能力检查，不称已完成OPD。

关闭跨流的实现直接走原生两次TimeDecoder调用，以bit-exact恢复Native；不使用65长度block mask浮点近似冒充exact。每父新条件前/后都原生重放并与旧logits逐数组exact。首两父额外关闭通路资格调用。每层记录跨流attention质量；首两父保存两条件所有SA的实际Q/K/V、权重、投影权重及输出，用独立NumPy FP64重算softmax/context，预定缩放容差2e-5；模型toy性质只检查算术与错配有效，不代替实际检查。

## 边界、预算、数据与判定

全部128父、四输出先全局封存后新GT。锁代码/模型/原缓存及评分依赖SHA，原共享非阻塞gpu.lock，GPU>=4GiB启动/运行5GiB reserve、host6GiB、disk30GiB、.8allocator，有限GPU900秒含加载与清理；首两父最大单例耗时×128+加载+60必须<900。无新RGB/Sourceencoder/DINO/ReferDINO调用，无学生或空间更新。记录实际TimeDecoder/SA/CA/head调用与失败partial，禁止信号其他进程。

独立审计全128新两流CDF、原MBR标量效用、512冻结空间数组、旧Native/Strong精确、首两实际SA以及模型状态/flags不变，之后才GT。父视频为统计单位，三个冻结空间seed父内均值；报告完整v/t、起止误差、固定GT支持与未观测实际节点sIoU不变、正负/严重尾/留一、硬输出与完整后验期望候选GT质量。2080候选质量由独立标量计算，不用GT选新输出或参数。

唯一CrossOffset若hard>=2pp且Native/Strong/Wrong对应CI均为正，可冻结一次父分离静态确认，不称OPD；若expected>=2pp且Native/Strong/Wrong对应CI均为正，则可另立一次同信息Direct/freshFixed/Live实际转移（不能仅因hard未涨否定分布）。如只有相对Wrong好或hard与expected均没有足够增量，关闭该精确接口，不扫层/桥接强度/位置编码。任一dev通过都不算用户目标完成，仍需真实OPD相对对照和独立确认。不新增自动化/旧队列，不晋升CURRENT/MAIN/PAPER。完成或失败同turn档案check/snapshot/check及GitHub代码协议全signed/unknown、负结果与远端双ref实际bytes/hash/tree收据；排除媒体、GT、权重、原缓存、文本/父ID。
