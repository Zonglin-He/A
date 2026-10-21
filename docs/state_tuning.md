# HC64：空间可达性与采样校准

状态：容量对照及完整状态四组真实适应64条均已完成并封存评分。主方法登记不变，旧几何、布局与PCA4/R4结果独立保留。正式Vid与CG保持暂停。

## 容量干预

旧几何最终content没有写入历史预测文件。使用原冻结decoder及完全相同的缓存输入、参数、seed、policy噪声重放40步；64条最终GeometryOPD轨迹逐位一致后才取得theta。禁止从输出框反推状态，禁止用布局state替代几何state。

对同一个theta真实解码四种干预：完整theta、仅径向限制到4*(1-1e-6)、投影到原PCA4、PCA4后再限制半径。半径干预位于旧smooth map的可达开球之内；投影是状态L2最近点，不是该空间的最优定位解。对照固定输入、ERef、Strong/MBR与decoder。容量结果属于诊断，不能作为新的无GT方法成绩。

## 唯一新适应配置

使用完整8×256临时content，identity映射，取消PCA4与全局半径4。保留原生mask x/y投影soft Dice和整轨迹相对ERef的mean/max保护，公式与上一轮完全相同。权重冻结，每query重置，固定同八个观测、同一ReferDINO跨帧query、同Strong区间。

采样尺度与温度采用全64无GT校准：先由原几何theta范数中位数除sqrt(2048)得到尺度参照，考察半倍/一倍/两倍，四舍五入到两有效位，对应sigma0.15/0.31/0.61。每档32个真实初态候选，选择平均候选guard为零比例至少0.8的最大尺度，否则最小尺度。得到sigma0.31；tau取该档各query候选U标准差的中位数并保留两有效位，得到0.00037。校准不打开GT或任何任务评分结果，不搜索vIoU最优sigma/tau，配置在新适应前锁定。

四个适应组：StateOPD、同奖励Direct、同候选DirectMC、只恢复层间reference导数的DirectFullRef。同lr0.014/Adam/seed41、同初态、同8×256可达空间和同目标。每组每query最多1321次完整轨迹前向（含候选、梯度点、提案、回溯），setup另计；OPD/MC最多40轮，Direct/FullRef最多660次。保持原五级回溯、U提高接纳及拒绝回滚Adam；不padding，不把forward allowance写成等耗时。

## 复现与输出

代码和精确配置位于scripts/state_tuning。顺序为capacity.py、独立CPU score.py、calibrate.py、run.py、独立CPU score.py。所有操作只复用已有缓存与冻结decoder，不新增RGB、Source编码器或专家前向。全部64新预测/trace/receipt封存后才读取标签评分。

正式比较报告完整vIoU、tIoU、观测/未观测实际节点sIoU、所有正负样本、超过5pp损失、父来源配对区间及实际计算量。HC64仍是已曝光开发面板，不作独立确认。


## 完成结果与停止后续扫描

| 方法 | 完整vIoU % | 未观测节点sIoU % |
|---|---:|---:|
| ERefStrong |31.0918|50.7055|
| GeometryOPD |31.3868|51.2469|
| StateOPD |31.1395|50.8890|
| Direct |31.4752|51.2019|
| DirectMC |31.2412|50.9188|
| DirectFullRef |31.4135|51.2575|

StateOPD相对ERef完整vIoU+0.0477pp，父视频配对95%区间[0.0091,0.0961]；相对Direct为−0.3357pp，区间[−0.7281,0.0396]。未观测sIoU相对ERef+0.1834pp，[0.0369,0.3970]。相比旧PCA4/R4确有改善，但没有达到实质增量，也没有建立OPD优于Direct。不能因跨零就把数值更高的Direct称为已稳定优胜。

几何状态原范数中位数13.9068，全部64大于4；PCA4保留能量中位数2.3466%。径向R4保留了几何正收益组约23.27%的完整vIoU贡献，PCA4仅约1.07%。这是同一几何state的干预，不是受限空间的最优解。

新OPD采样ESS中位数22.70/32，反馈已不再近乎均匀；最终state范数中位数4.7896，仍低于Direct等组。后续曾按保存状态范数比提出共享lr0.04的有限对照，但用户随后要求改为整轨迹算法设计，该对照在5条完整query边界保全停止，未评分，不登记为完成，不从该前缀选参数。新算法证据与接口单独见trajectory_opd.md。

完整结果、全部匿名正负及成本：results/state_tuning_hc64/hc64.json；容量干预：results/state_capacity_hc64。缓存decoder增量时间不能当成端到端推理时延。
