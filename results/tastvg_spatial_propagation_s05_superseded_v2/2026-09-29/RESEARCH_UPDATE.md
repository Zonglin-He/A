> Superseded by the latest teacher-independent native parameter rollout route. This completed result is preserved as history.

# S0.5 soft-moments v2（已完成，随后依用户改线保留为历史）

最新8b54f006已要求改做student-native parameter rollout。v2在该指令到达前已完成96条件推理、seal、评分和NumPy验证；没有在改线后新增teacher回归实验。旧阈值v1与本版均不再作为当前候选路线，仍保留全部结果。

本版严格按1d6fd366：前景/背景各自prototype＋local max，固定温度1 softmax；cell-center soft moments，width/height=sqrt12variance，角点clip；原reference mask框覆盖、H0计算target一次固定；同原L1+GIoU系数、除T、H_app三步×.004、dynamic suffix。原16曝光VidSTG父源/16query×clean/五类既有5% transient共96cell，同Vid-source TA checkpoint，0新expert/下载/采集。

Corruption：same12未观察oracle增益+1.0152pp，相对S0+.3145pp增加+.7006pp，描述性配对源CI[+.0537,+1.4815]；整管oracle+1.5401pp，并集+2.0765pp（S0并集1.3886pp）。Clean same12未观察+.8186pp，整管+1.4566pp，并集+2.0561pp。所有数字都是候选oracle上限，不是在线方法收益，也不能改写为“direct regression完全无效”。

66cell有前景token bank并传播到6672未观察frame-cells；30cell沿用稀疏证据（9原mask空，21投影后无前景token）。不根据结果补阈值/温度搜索。原参数及text/motion不改，预测seal后仅same16旧GT评分；source内部五corruption平均，再source macro/10000 bootstrap。11CPU合同、960双指标/576独立loss、96area/66概率/6672moment框、2编辑完整回插、1728scalar＋240配对检查通过。GPU进程71.03秒，288梯度，无失败。

该版与v1不仅readout不同，还增加background local affinity并改为/T分母，因此不作为纯readout因果消融。S1/OPD未实施；最新方向为无teacher的native参数候选生成，不能以这里正信号抵触用户最新范围。
