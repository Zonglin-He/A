# 稀疏原始DINO监督原生目标状态：路线比较32

**已完成一轮真实路线比较。直接稀疏框监督产生小幅跨帧收益，HC未观察实际帧有正向开发证据，Vid仍不确定；同接口完整轨迹OPD不如直接监督，Live没有超过Fixed。新组合未超过封存Fine-Center，不支持采用或继续围绕本接口调参。**

这不是又一次Fine-Center温度、步长或读出实验。适应对象从独立每帧中心变为共享原生末层目标条件，监督从ROI相似度变为原始DINO框定位损失；所有实际帧的中心和尺寸都能经过训练好的decoder产生变化。它仍不证明身份错误是主要瓶颈，也未验证身份纠正。

## 先封存基线，保留已做历史

Fine-Center配置HCstep1/Vidstep.5/tau.1及原输出/结果在新运行前hash固定，原baseline不改、不再搜索其参数。CURRENT/MAIN/PAPER部署登记与研究基线分开，不自动晋升。

历史520484a5已经测过E保留initial-QR256的条件endpoint拟合，b13be259已测过真实QR轨迹的稀疏DINO-IoU reward更新，并且近E/无实质增量。两项负结果保留，本轮不能叫首次query适应或首次完整轨迹OPD。

本轮选择的不同接口是原生第六层空间decoder的ca_qcontent_proj共享临时bias256。它只影响末层读取当前帧视觉memory的目标条件；先前五层输出/reference不依赖该变量，因此没有旧初始QR那一条被detach截断的跨层依赖。所有预训练权重/LN冻结、eval；phi=sigma*m，sigma为当前视频原生末层qcontent投影通道RMS，m从0开始，每query/arm丢弃。代数上等价于对既有projection bias加临时delta，没有新增/源域训练网络或定位头。这个入口只适用于当前TA-STVG，不预宣称backbone无关。

E-retaining仍是中心E+(native_phi−Source)、尺寸E*exp(log(native_phi)−log(Source))；m0逐query必须Source/E bitexact。保留E不代表decoder已学会E。新状态可能改变所有实际帧的位置和尺寸，不在17个E局部平移框中挑位置。Temporal始终固定原finalMBR。

## 实际范围与五组

原NH16/NV16开发名单，每父1query，共32父；冻结TA-STVG跨域EMA（HC用Vid源权重、Vid用HC源权重，checkpoint/代码/prefix hash预锁）、同E+finalMBR、同4个计划DINO观测帧和E已选raw DINO references。实际可用参考HC15/16、Vid11/16，缺失/错误参考继续保留；没有新DINO、图像编码、专家、Source或特征提取。FH/FV正式64排除并hash保护。本组已多次曝光，只用于开发路线比较，不是独立确认。

E和Fine-Center精确复用封存输出。新增Direct_anchor、State_fixed、State_live共96episodes/192scheduled steps，2步后均值固定读出。新3组同256维临时bias和Adam（HC .01、Vid .03，beta .9/.999、eps1e−8、每query/arm fresh），并非沿用Fine中心坐标的步长数值。没有本轮GT选参或扫描。

监督为E已接纳原始DINO框，保留错误、无参考和unknown情况；等权1/A不是正确概率。共同loss为`5*mean_anchor(sum4 L1) + 2*mean_anchor(1-GIoU)`，系数来自已有TA-STVG5/2配置。

- Direct真实反传该raw稀疏box loss到临时state。
- Fixed/Live每轮16反对称完整state候选，实际运行全部6层冻结空间decoder得到整段轨迹，再用同一稀疏box loss评价；不是人工撒每帧框，也不是旧ROI反馈。
- teacher工作权重为`w ∝ v exp(-loss)`，unit loss temperature；v为当前Gaussian相对proposal的RN权重。梯度`sum(v-w)*actions`经同Adam更新。它是有限self-normalized/control-variate likelihood拟合，不是精确连续KL最优解或语言GKD复现。
- Fixed两轮围绕初始state0采样，Live第二轮跟随实际更新mean。同噪声，首轮candidate/loss/gradient/mean/output exact。DINO目标不刷新，刷新的是学生完整轨迹及其锚点定位loss；不能称DINO重新读取新视觉证据。

无参考/恒定反馈不调用optimizer，禁止momentum-only漂移。两种更新共用变量/损失/optimizer/次数，但实际梯度方向、位移和计算预算不相同；不是等FLOP算法优越性证明。无GT选步/样本/候选或质量回退。全32×5输出、actual候选、权重与state全封存后才一次GT评分。

## 主结果：完整tube与未观察实际帧

下表为父macro配对差值pp及描述性95%bootstrap区间，seed20261018/10000，未多重性校正。每域主分母16父，原all-known-unobserved actual分母，不以active-only替代；unknown/no-ref/零分样本全部保留。原时间不变，各域允许固定参数不同。

| 对比，pp | HC 完整vIoU | Vid 完整vIoU | HC 未观察actual sIoU | Vid 未观察actual sIoU |
|---|---:|---:|---:|---:|
| Fine-Center − E | +0.249252 [+0.018279, +0.528445] | +0.145613 [+0.015699, +0.335593] | +0.440565 [+0.010807, +0.961029] | +0.292150 [+0.028585, +0.610493] |
| Direct − E | +0.026937 [+0.011041, +0.043954] | +0.023918 [+0.001001, +0.051172] | +0.036461 [+0.010629, +0.069117] | +0.028206 [-0.000397, +0.060624] |
| Fixed轨迹OPD − E | +0.004503 [+0.000582, +0.008744] | +0.006984 [+0.000510, +0.015209] | +0.005415 [-0.000567, +0.012187] | +0.004919 [-0.005400, +0.014967] |
| Live轨迹OPD − E | +0.004402 [+0.000559, +0.008549] | +0.006740 [+0.000514, +0.014986] | +0.005097 [-0.000655, +0.011513] | +0.004525 [-0.005011, +0.013836] |
| Live − Direct | -0.022535 [-0.036555, -0.009397] | -0.017178 [-0.036866, -0.000536] | -0.031364 [-0.058550, -0.009192] | -0.023681 [-0.048498, -0.002939] |
| Live − Fixed | -0.000101 [-0.000284, +0.000059] | -0.000244 [-0.000837, +0.000253] | -0.000318 [-0.000768, +0.000027] | -0.000394 [-0.001418, +0.000493] |
| Live − Fine-Center | -0.244850 [-0.522377, -0.015046] | -0.138873 [-0.321010, -0.014243] | -0.435468 [-0.952362, -0.007691] | -0.287625 [-0.600471, -0.028851] |

Direct−E完整v HC+0.02694/Vid+0.02392pp，两区间正；未观察actuals HC+0.03646pp区间正、Vid+0.02821pp跨零。说明稀疏监督经过这个原生状态，能产生一些未观察实际帧改善的开发信号，幅度很小，尚未双域确认传播收益。

Live−Direct完整v−0.02254/−0.01718pp，未观察actuals−0.03136/−0.02368pp，四个区间均负。Live−Fixed近零且四主区间跨零，没有on-policy刷新增量。Live−Fine-Center四项均低、区间负；普通直接适应也没有超过封存Fine。不能为了OPD名称删除直接监督控制，或把本轮小正值写成新方法超过现有基线。

## 监督被吸收，但不等于足够任务增量

26个有参考query中，新3组最终稀疏伪标签loss均低于初始；Direct损失变化均值HC−0.00679/Vid−0.01372，Live−0.00131/−0.00239。Direct和OPD都真的更新；192slots中156次实际optimizer、36次无参考exact no-op。

active-only最终标准化state范数Direct约.3194/.8930，Live .2476/.7316；输出相对E全轨迹L2 Direct .00384/.01019，Live .00115/.00445。两域各975/1030个实际节点的尺寸确实发生改变，未局限中心移动。范数是过程量，不是GT正确率或跨变量可比的效果保证。

这些观察不支持把差距全部归为‘OPD没运行’或梯度未接通；也未唯一证明采样数、方向噪声、温度或教师是原因。本轮不据此追加扫描、更多轮次或另一个接口小修。

锚点GT sIoU Direct−E HC+0.09805pp（区间正）、Vid+0.06163pp（跨零）；Live+0.01758/+0.01714。Sparse loss下降并未带来强完整轨迹收益。DINO仅4帧监督，未观察帧没有独立dense reward，GT只用于封存后衡量其真实推广。

成功和损伤都保留：Direct完整v HC11改善/4下降/1不变，Vid5/1/10；Live HC11/4/1，Vid5/1/10。Direct最差NH014−0.01418pp、NV014−0.00374pp；最佳NH002+0.10125、NV005+0.15454pp。Live最差NH010−0.00717、NV014−0.00457，最佳NH002+0.02314、NV005+0.05473pp。无<-5pp负尾不意味着稳定或安全保证，很多零完整v仍受原时间支持限制；本轮不重新选时段。

## 工程与独立核验

固定NH001/NV001的零Source/E精确匹配；smooth output VJP与实际FD误差2.05e−6/3.00e−6（约0.4%/1.0%），通过事前阈值。逐actualdecoder调用验证前5层native预测bitexact不受state影响，共2936次；每query预训练完整state hash不变。此保证排除本接口那条旧reference依赖缺失，不保证有限步或DINO伪标签有效。

Root CPU独立核对1664个跨臂采样轨迹及E映射、104次RN/teacher/score梯度、192scheduled state/Adam路径、96最终mean、960指标、96父bootstrap；loss误差<=8.89e−16、OPD梯度4.72e−16、Adam状态7.46e−9、指标0。独立CPU审计没有重跑全32 GPU模型，直接VJP由固定2工程probe与明确site依赖支持。后续固定两条耗时复测还精确复现其Direct step0梯度（误差0），但不冒称全32独立GPU复现。

准备阶段AST发现controller条件表达式空格语法，在模型/GT/预测均未开始时修正；原source/runtime保存preflight001，独立pin修订记录，适应变量/损失/采样/optimizer/readout不变。没有实际模型失败、补样或重选科学配置。所有原结果与负历史保持。

## 成本与裁决

主测量1500次唯一完整轨迹PosDecoder组合调用（3000offset调用），1664是含首轮Fixed/Live重复记录的候选审计数，不混计。52次Direct反向、2次FD-probe反向。原GPU主分配保守计88.5984s（worker85.5957s另加预锁3s开销），包含模型加载/前向/反向/哈希与同步，不是纯kernel时间。

为补前后向成本，输出封存评分后只重放固定NH001/NV001的Direct零态作耗时验证，无fit/选参/新成绩/GT。forward+loss wall .03370/.01897s，backward wall .12389/.00323s；首个反向含冷启动，只有两个样本，不外推稳定速度或当全运行精确分解。CUDA event时间与wall分别公开。这项额外6 trajectory forward/2 backward及两次模型加载GPU计21.3491s。总新增GPU分配上界109.9475s<900s，两个lease释放、ownedunits success inactive。原Source/E/DINO/prefix生成成本与CPU评分/audit/发布另列，不称首次部署延迟。

本轮只做一次路线选择：保留原Fine-Center作为封存研究基线；直接稀疏监督保留为有微小传播信号的控制，不能替换当前基线；**当前末层bias完整轨迹OPD未建立超越直接监督的价值，不扩大或再调该配置**。不能将这个有限结果推广为所有共享state/decoder适应或OPD不可能。没有自动下一轮，也不恢复夜间heartbeat，CURRENT/MAIN/PAPER保持。

实现/协议/实际参数/全匿名signedunknown/成本/负结果/审计公开；媒体、GT坐标、DINO框坐标、原始轨迹/weights/prefix/raw特征/父ID与文本排除。GitHub main+research freshCAS及实际远端字节/哈希/tree核验后才算公开完成。
