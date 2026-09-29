# 最新S0.5：Student-native spatial rollout support

已按8b54f006完成。候选只来自TA-STVG自身空间参数邻域；生成阶段不读取Sa2VA、旧propagation结果或GT，也不使用任何teacher-box回归loss。先封存864条完整候选tube，再读取同16条旧GT做离线oracle评分。

**本配置有正的候选上限，但没有扩大到足以进入S1的程度。**

| Corruption诊断 | 结果 |
|---|---:|
| Frozen native sIoU | 45.3509% |
| 原生六层oracle增益 | +0.6965 pp |
| 新9条参数rollout oracle增益 | **+0.8119 pp** |
| 六层∪新rollout oracle增益 | **+1.2441 pp** |
| 旧S0六层∪teacher轨迹oracle增益 | +1.3886 pp |
| 新并集−旧S0并集 | −0.1444 pp，95% CI [−0.4291,+0.1573] |

新增rollout oracle区间[+.5383,+1.1255]pp，并集区间[+.7766,+1.7489]pp。Clean对应rollout+.8101pp、并集+1.2838pp，与corruption接近。没有达到附件举例的3–5pp，也没有清楚超过现有1.39pp支持量，所以本轮不启动S1，不自动扫radius/K，也不恢复temporal OPD。

实际配置：原16已曝光VidSTG父源/16query×clean及五类既有source-hash seed0 transient5%条件，共96cell；原Vid-source TA checkpoint5ab12c86。1792维空间参数接口由256维second-pass query residual和最后空间decoder block5的norm1/norm3/norm4 weight+bias组成。固定seed20260929生成4个正交方向，正负配对，加native共9候选；所有样本同方向。唯一半径为原1792维参数L2范数的5%，绝对范数1.57658019。这个数与之前H的.004更新不是同一量纲。

每候选都走原生动态suffix，原参数随后恢复；H不变。96cell均9个不同tube，最不相似候选的self-sIoU源平均仍为93.6581%，说明本轮确实改变预测，但只覆盖这个固定局部邻域。接口可用于将来的persistent参数更新，本轮没有学习、跨arrival继承或OPD，不把support screen称作online TTA成功。

未观察帧在整管oracle所选候选上：same12来源+.8535pp（CI[+.5228,+1.2363]），观察帧+1.0096pp；全16来源未观察+.8016pp另列。这里“观察”只使用与S0相同的五个均匀位置，完全无需读专家mask。GT只选完整tube，不逐帧拼接。源内五corruption平均再16源宏平均；10000源bootstrap，历史开发曝光而非untouched确认。

保留成功例：Q09/Q06/Q13的rollout oracle分别+2.3194/+1.7970/+1.5912pp；Q06并集比旧S0多+1.2967pp。也保留缺口：Q12并集比旧S0低1.2955pp、Q07低.7683pp。固定方向臂有明显方向性：arm8均值+.4479pp，但其反向arm7为−.4258pp；这是事后诊断，未据此晋升或更新参数。各固定臂完整成绩保留在ROWS和SUMMARY。

本轮不使用teacher坐标生成候选，满足最新接口限制；但结果没有验证Sa2VA偏好能选中、更没有验证Reverse-KL或未来arrival收益。以后若重开，应针对尚未覆盖的参数邻域或选择机制提出新信息；本轮阴性只限4方向/5%半径/该1792维接口，不能推论所有空间OPD不可能。附件未来的compatibility softmax是有限候选上的替代分布，尚不是已验证的native输出似然。

之前阈值传播v1、soft-moments传播v2在新指令到达前均已完成，依新路线标superseded并完整保留。它们的corruption并集上限分别+2.1408/+2.0765pp，不能抹掉或改成“teacher regression全部失败”；它们也不满足本次student-only候选约束，因此不是当前方法。

新试验GPU进程78.65秒，864下游候选、2次非零参数完整回插，0新expert、0梯度、0GPU失败。3个CPU合同、96原生精确、1440双实现指标、NumPy参数半径/正交/成对核验、2039项公开标量/CI复算通过。首次摘要写入遇到NumPy int64不能JSON序列化，修为Python int后仅CPU重评分，逐行等于已保存ROWS；原失败日志保留，无推理重跑。研究总档案已更新，CURRENT生产方法不变。
