# 项目历史与当前状态

当前任务：正式HC1902已评分，Vid8405/10303按用户要求暂停并保全，CG暂停。HC128布局奖励对照及奖励机制研究已完成并发布；HC64首轮真实状态候选已封存评分，未显示有效收益；按新授权完成PCA/R4容量干预与无GT校准，正在完整8×256空间运行四组公平适应。注册几何方法及参数保持不变。

## 清理与此前结果记录

2026-10-20，用户明确将“单 ReferDINO 初始化＋空间 OPD＋Strong 时间分支”及简化参数作为唯一主方法，并授权删除废弃开发产物。该整理替代以前的多方法登记；唯一配置为 `configs/default.json`，唯一运行入口为 `python -m stvg`。

当前 HC 参数为 lr0.014、tau0.14、40步、初始化gamma0.5、输出alpha1；Vid为0.01、0.5、30步、gamma1、alpha1。权重冻结，seed41。保留原256条正式结果、所有正负样本和严重尾损说明。本轮只是依赖迁移、清理与工程回归，没有改变算法、重新选参或运行新实验。

清理前的完整公开证据版本：[7a0a813](https://github.com/Zonglin-He/A/commit/7a0a8131733d124c9262db0678260c46f095ead4)。旧方案、完整参数矩阵及诊断可通过正常 Git 历史恢复，例如：

```bash
git show 7a0a8131733d124c9262db0678260c46f095ead4:results/stvg_error_state/2026-10-20/single_referdino_parameter_search256/rounding/REPORT.md
```

清理前的私有研究总档案已在项目外单独保存；不把私人记录上传 GitHub。其他任务留下的未跟踪文件也在项目外保留，未作为废弃代码删除。

工程回归：两条已有查询从视频到最终输出均与旧封存结果逐位一致；原256条预测的五行指标全部复核，最大绝对误差约1.6e−15。核心policy、共识初始化、时间后验及MBR等23个运算函数的 AST 一致。检查只验证重构兼容性，不产生新科学成绩。结果分别见 results/migration_checks.json、results/score_regression.json 和最终 cleanup_checks.json。

历史结论保留：本方法的现有结果属于已曝光开发面板，Vid的适应均值增量尚不稳定；BV030等负例未删除。工作树只保留当前可运行方案不代表此前负结果被反驳，也不代表已经完成全量泛化验证。

2026-10-20 整理更新：主方法迁入独立 `stvg-opd/`，共享外层必需资产，不改变冻结配置。按用户追加要求并行审计推理标签隔离、开发数据曝光和公开导出；最终审计见 docs/data_integrity.md 与 results/data_integrity.json。Git历史保留，因此本轮公开内容检查限定于新提交树，不能视为删除历史公开记录。

泄漏审计结论：未发现当前推理读取GT；128条离线选参合法，不计为数据泄漏。已修复封存校验/标签读取顺序、解码前schema检查及注释软链接防护，14项CPU检查通过。新正式评估按用户最新指令暂停，本轮不安排新实验。迁移到独立目录后的两条原查询完整回归仍逐位一致。

GitHub同步完成：实现提交 [87294995](https://github.com/Zonglin-He/A/commit/87294995eea16161e3edb88289ee80463c87cb13)，tree `b42ffc2848e8e76e1db2c48189686a1bf7177686`。main与原research分支以fresh expected-SHA、非强制快进更新。远端重新fetch后，248个文件／2,023,178字节全部与本地导出SHA256一致，旧7a0a813提交仍是祖先；详见 docs/publication.json。本条及回执在随后的档案提交中保存，不代表新实验或新方法晋升。

本地Git迁入独立目录后，补核Git忽略规则对资产软链接同样生效（目录专用尾斜杠改为兼容文件/链接的根路径规则）。公开导出始终使用明确allowlist，私有链接未进入任何本轮提交。最终远端核验回执另存本地 outputs/maintenance/publication.json。

2026-10-20 16:45 北京时间执行更新：55项CPU工程检查通过；此前真实HC回归、Vid缓存命中回归均逐位复现旧封存预测。锁定TA专用有限计划后启动，plan SHA256 `d95e1fa35f4e854d4981edf2ffb0ca0a7087988cd1d59ba91ad3a038fa4634af`。当前是正式推理运行中，未完成全量评分，不发布或推断新成绩。每个完整方向全部封存后才独立读GT评分；遇错停止，不跳过样本或自动重试。

正式评估启动代码已同步：提交 [e95459b5](https://github.com/Zonglin-He/A/commit/e95459b55998c3e16ceb637358ecc9b15f6eb119)，远端main和research均以freshCAS非强制快进；重新fetch逐一核对330文件、2,555,220字节与本地SHA256及tree完全一致，旧历史仍是祖先。此次只发布实现、冻结协议和运行状态，不宣称完整评估已完成或已产生正式成绩。回执见 docs/publication.json。

2026-10-20 运行监控授权：用户要求一个Luna/max子代理每30分钟巡检，异常报告根代理并允许自行修复工程故障。已建立本聊天专用heartbeat `ta-stvg-luna`，复用唯一 `ta_luna_watch`（gpt-6-luna、max），正常不通知；根代理核证后按冻结计划恢复，不改科学配置、跳过样本或影响无关服务。TA全部结果同步核验后暂停监控，CG仍待TA全部完成。此前已暂停的研究自动化保持暂停。

2026-10-20 17:27 北京时间划分纠正：直接HTTP Range读取TA官方README所链Gstar666/TASTVG data.tar（revision38e89a5895346d77cba87b0273f9ede95874befc）得到hc-stvg2/annos/test.json1902、train.json9586。raw3482完整包含test1902、另1580不在official train/test；共同成员534条bbox不同，时间/文本/大小相同。原先将raw3482称官方验证划分属核验错误，撤回该命名及基于此总量的正式ETA，旧记录只作历史。已停止自有controller408783/worker408784/teacher408935并核退出，295条partial保留，其中162条属于正确名单；未读GT评分。Luna heartbeat暂停防止误恢复，待完成两数据集审计与划分修复。

2026-10-20 双数据集划分修复完成：HC extra1580来自保留raw val_v2，无法从当前证据确定作者删减规则，不能称为重复或训练混入；534共有查询的官方bbox同步更正。Vid官方原始JSON与本地完全一致，10303query/732video成员、对象、文本、窗口完全匹配；8558缺最后一框、1745区间end多1帧均按官方loader修正，2727412官方框独立对VidOR逐帧一致。旧正式根移至outputs/superseded/formal_raw3482_20261020，295原预测不删，162复用逐文件/request/runtime哈希通过。新清单和离线标签哈希一起锁定；60项CPU测试通过；没有新GT任务评分、方法改动或调参。新增公开离线转换入口与详尽限制见 docs/split_audit.md、results/formal/split_audit.json；历史256开发成绩保留旧口径，不自动晋升。

修复后执行接续：新TA-only计划SHA256 `4e0f79f69b9dba066c3c6734f7608de37f59754ca29f4a8f400d99fdef1969b7`，包含HC1902、Vid10303及独立标签哈希，已实际启动新controller649926；只复用162条完整兼容预测，原3482计划不恢复。半小时Luna/max监控已用新计划及新计数恢复。当前仍为推理运行中，无正式任务评分，不把数据修复说成新性能结果。

标注修复已公开并核验：提交 [03936491](https://github.com/Zonglin-He/A/commit/0393649115784ba9b9f163435bfbdab211e5227f)，main与research均freshCAS非强制快进；远端重新fetch逐一验证335文件、2,579,816字节与本地SHA256及tree完全一致，历史保留。公开内容不含任何原始标注、媒体或缓存。本次是数据/标签转换纠正和正式运行接续，不是新性能成绩。

2026-10-20 22:07北京时间：HC1902条预测全部封存，原推理进程正常退出。原CPU评分在最终JSON写出时因70个numpy.int64计数失败；失败state/log完整保存。采用独立CPU适配器，仅在原evaluation模块的json.dumps中将NumPy标量转换为等值原生数，未改冻结源码、指标计算或预测。原验证器在修复前后复核全部预测、运行证明、清单、标签和封存哈希；9510次官方/独立dense指标交叉检查通过，JSON往返与原评分返回值一致，60项CPU检查通过。原计划哈希保持4e0f79f69b9dba066c3c6734f7608de37f59754ca29f4a8f400d99fdef1969b7。HC五行等查询主表m_vIoU为Source24.9948、Spatial26.1279、Temporal25.7652、WithoutOPD26.6088、Full26.8784（%）；Full−Source+1.8836pp，父视频cluster区间[1.4186,2.3439]；Full−WithoutOPD+0.2696pp，[0.1503,0.3891]。完整划分含开发曝光，负例全部保留。原controller通过原有resume校验，跳过已评分HC，于22:07:46启动Vid10303；新controller1508751、Vid worker1508820，CG暂停。Vid结束若出现同一JSON写出类型错误，沿用独立CPU适配器处理，不能修改推理代码或放宽指纹。正式运行不因测速中断。

HC1902结果已公开并逐文件核验：提交 [481de74f](https://github.com/Zonglin-He/A/commit/481de74fc9aaca83c401795dbc9c4738676f0bed)，main与research均freshCAS非强制快进；远端338文件、8,918,514字节与本地SHA256和tree一致，全部正负匿名逐query结果保留。Vid仍运行，效率独立基准与定性结果待两方向完成后执行；本次不宣称TA全阶段完成。

2026-10-20 23:02北京时间工程加速：按用户顺序在隔离副本实现精确缓存命中时跳过无效权重/特征搬运、448仅时间FP32捕获、单网格区间几何索引缓存。70项CPU检查通过；四条两域已有查询的12次有限GPU回归中，Source/E_Ref/Live_raw/Live/时间区间与原封存预测逐位一致，Native/Strong完整后验、两offset两轮门控、完整六层时间hidden、CPU/CUDA RNG和冻结权重均通过一致性检查。教师只重放精确RGB/文本/位置绑定的缓存，新增专家前向0，不读GT。一次维护保全HC1902和Vid296条，原运行及源码快照完整归档；原验证器在归档源代码下核验旧全部输出，逐条保留原receipt/预测/执行成本与显式兼容性来源，未放宽resume校验。新计划6c5c2d33a2561004cd45f0b799c9816bab27d63f2c9181a4df2ec03f08ecb862，新runtime059775a6d5e2921fc9c94b90e059d9bf5ed3332ca50890cedaece1b0aee16285，仅六个工程源文件改变，方法参数、checkpoint、采样、标签、seed与精度全部不变。HC已发布评分原字节保留。新controller1645280继续Vid，旧pre-speed计划禁止恢复，CG仍暂停。工程回归中的重复Vid输入12.953→8.889秒，长200节点19.775→18.475秒；这是有限回归，不能代替全量正式效率。工程回归最初一次仅测试夹具对native/replay hidden次数计数错误，修正夹具后全验证通过，原失败日志亦保留。见results/engineering_speed.json。

工程加速实现与回归已公开并核验：提交 [33508f50](https://github.com/Zonglin-He/A/commit/33508f50f32dca434fb90e3c0fe268f3f913f303)，main/research以freshCAS非强制快进；343文件、8,967,458字节的远端内容与本地SHA256及tree完全一致，原Git历史保留。已恢复半小时巡检并绑定唯一新计划；正式Vid继续，无额外测速任务。

2026-10-21 HC空间机制离线分析：按用户任务书，在已完整封存的HC1902/227父来源上计算Full−WithoutOPD的恢复、损伤、GT方向、时间重叠、教师质量和传播距离；未读取运行中Vid标签，没有模型/专家前向或调参。未观测实际节点sIoU+0.4979pp，父来源聚类区间[0.3000,0.7092]；Recovery合并帧363/10331，Damage255/24090，两者分母不同；完整vIoU+0.2696pp。教师最低质量四分位未观测增量−0.0950pp（跨零），最高+1.0387pp；仅事后解释，不在线GT筛选。正式worker未保存path/rounds，不能生成真实逐轮cost/ESS曲线；不存在同条件Direct/Fixed正式对照，不能宣称本分析证明OPD优于传统KD。所有预测和标签哈希前后核验、原任务指标重算最大误差1.33e−15、时间分解误差3.28e−16、合成分母/缺帧/几何核对及70项CPU检查通过。结果见 results/formal/spatial_analysis_hc2.md 和JSON，分析代码在scripts/不改变正式plan；Vid继续运行、CG暂停。

HC空间离线分析已公开并逐文件核验：提交 [a8c3aa2d](https://github.com/Zonglin-He/A/commit/a8c3aa2d4e7077c5d419243e0865200898a4246a)，main/research均freshCAS非强制快进；346文件、9,313,578字节与远端Git blob SHA256及tree完全一致，历史保留。公开结果仅聚合数值；逐query/GT/专家缓存不上传。Vid正式推理持续，五行全阶段发布尚待其完成。

2026-10-21 后续执行授权：用户要求先等当前Vid正式任务完成，将旧几何奖励版本两方向五行及空间OPD净增量上传GitHub并核验，再在原HC128/Vid128上比较3×3布局奖励OPD、原几何奖励OPD和同布局Direct。禁止覆盖旧结果；新试验独立配置、目录与指纹；未新增当前GPU任务或修改运行依赖。队列协议见 docs/layout_reward_plan.md；CG继续暂停。

2026-10-21 14:40北京时间用户改序：明确要求全量先暂停保存、优先新奖励。已暂停原heartbeat，在query边界停止并确认自有controller1645280/worker1645347/teacher1647897退出；Vid8405/10303完整预测逐NPZ/receipt哈希核验保存，无孤立临时输出，HC1902评分不变。状态标为paused_by_user，禁止自动恢复；plan、runtime及科学参数不改，暂停证明在outputs/maintenance/user_pause_20261021。新布局开发对照获准提前执行，旧“全量发布后才运行”前置条件由本次用户指令撤销；保留所有旧结果，CG暂停。

布局奖励开发试验已在隔离副本实际启动：outputs/layout_reward_dev/workspace，controller2364458/startticks213045387。首条RC001通过Source/E_Ref/八位置与旧封存逐位核验，LayoutOPD/Direct各完成40次实际更新；奖励自匹配、候选区分、ROI坐标自动梯度/有限差分及越界负分CPU检查通过。固定stride16投影特征、3×3有序布局、tau0.1，lr/steps其余沿用两域配置；GeometryOPD复用精确旧结果，未看新GT成绩选参。8帧Source/专家导出成本单列，完整256条先封存再评分。旧全量暂停状态不变。

2026-10-21 单方向收缩：用户要求新奖励只跑一个方向，保留正在运行的Vid→HC2 HC128，取消尚未开始的Vid128。原推理源码/科学参数/指纹不改；独立scope_hc2范围记录与外部边界控制器已启动，在HC128完整写出后核验所有NPZ/receipt并停止经过身份核验的自有推理/教师，独立封存并CPU评分。未提前读GT，已有输出保留，不把原256规划登记为完成。监控已更新为HC128；正式全量与CG持续暂停。

2026-10-21 HC128布局对照完成：全部128条预测先封存，再CPU评分，Vid128取消且未启动。边界辅助器读取/proc时因其他进程名含空格而解析失败，原推理已在128条安全停止；修复辅助器解析并保留原失败日志，仅完成封存/评分，推理源码及预测不改。官方与独立dense指标640项对照最大误差9.99e−16。ERef+Strong/GeometryOPD/LayoutOPD/LayoutDirect完整vIoU分别28.5877/29.0669/28.8628/28.8058%；未观测实际节点sIoU49.9315/50.9973/50.4255/50.2491%。布局−几何为−0.2041pp vIoU、−0.5718pp未观测sIoU，两项本面板父视频配对区间低于零；布局OPD−同布局Direct仅+0.0571pp vIoU且跨零。相对ERef，几何92改善/33退步/3不变且无vIoU超过5pp下降；布局87/38/3且1条严重下降，Direct88/37/3且无严重下降。固定本次配置不能支持替换几何奖励，唯一注册主方法不变，不再擅自调参。全部正负匿名指标及退步表独立保留于results/layout_reward，正式全量和CG仍暂停。

HC128新奖励结果已公开并逐文件核验：提交 [87f28ca8](https://github.com/Zonglin-He/A/commit/87f28ca892a5d3cf862e259e148e3177959b5276)，main/research均freshCAS非强制快进，远端359文件/9,740,420字节与本地SHA256及tree一致，历史保留。本次未支持替换几何奖励，所有匿名正负结果与同布局Direct均公开，原始缓存和私有ID不上传。布局试验已结束、巡检暂停；原正式Vid8405/10303和CG维持用户暂停，未自动恢复。

2026-10-21 奖励失败只读研究：依用户要求并行核查数据、网络/算法/参数和原论文/官方代码，CPU流式读取HC128原缓存及少量原帧，未启动模型/专家/decoder或调参，未读取暂停formalVid标签。布局奖励确实被优化（OPD121/128上涨、ESS14.75），但37条局部定位下降、24条观测改善而未观测退步；同候选query宏排序rho布局0.5579低于几何0.6059，小目标普遍不足九格不成立。发现奖励读取文本/时间融合前特征，当前policy仍是逐帧框提案，E仅输出校正；原生对象mask/竞争轨迹已计算却未导出。优先研究以原生身份/范围证据评价真实content纠正的跨帧后果，明确Direct-point与Direct-MC公平对照、参数未最优及开发版本边界；128开发选参不是泄漏。新方向仅设计，不登记为主方法。见 docs/reward_study.md 和 results/reward_study/summary.json；私有逐query/媒体留在outputs/layout_reward_dev/research。

用户追加七层核查后：已用CPU缓存补128条严格同初态content梯度/Adam首步、固定初态seed41–45的ROI候选噪声、两offset覆盖、GT仅事后的坐标梯度/有限框步与时间分组；不同路径的全程步长不再被当作同状态比较。原生mask摘要不足以检验完整性，随后启动同一权重/同八位置的128条原生输出诊断导出，原RGB哈希与选轨八框逐条严格一致，无GT输入、不更新任何学生或网络权重。另准备固定H036/H084/H125的6状态缓存decoder导数对照及最多3个初态Jacobian，按GPU锁串行执行；不恢复正式全量、不生成新方法主表或调参。


2026-10-21 奖励研究第二阶段完成：HC128同一ReferDINO/同八RGB原生输出128条先封存，selected八框与query及输入哈希逐条严格复核，无新增观测或学生更新。初态同32候选query宏rho布局0.5635、几何0.6120、mask投影0.6406，投影同时提高Point对GT梯度cos至0.4598，不能据此宣称OPD独有优势；Mean缩框及Coverage全图退化经固定缩放反例验证。固定H036/H084/H125的6相同状态/Adam历史和3初态原图导数已核验原路径逐位一致；H125未观测计分节点55主导实际退步。随后54次physical Replay中央差分与内存全reference梯度诊断确认上游层间detach导致原反传图不等于完整forward导数，两种现有loss共同受影响；所有forward框逐位不变，原源码/正式paused state哈希不变。纠正旧“完整Jacobian”和单纯“有限步非线性”的口径，完整导数下H125敏感性仍存在，原Adam大步非线性亦保留。没有恢复formal、运行新适应、改参数、训练网络或增加专家。70项CPU检查通过。综合证据与限制、强Direct及真实状态候选研究方向见docs/reward_study.md和results/reward_study；公开代码副本仅路径/匿名行序/导入转换，原执行源和封存证明本地保留。

奖励研究已公开并逐文件核验：提交 [6a21bf77](https://github.com/Zonglin-He/A/commit/6a21bf7773570695ea897d97c94f7fb03b015d39)，main/research以freshCAS非强制快进；386文件、10,243,099字节的远端Git内容与本地导出SHA256及tree一致，旧历史保留。公开聚合统计与15份审计计算源码，不含私有数据、描述、GT坐标、mask/特征/状态张量。本次只完成机制研究及有限诊断，不宣称新奖励取得适应成绩；正式暂停和唯一注册方法保持不变。


2026-10-21 HC128真实状态候选实验获用户明确授权并启动：固定同一Source有效特征PCA4基底、8×4 latent状态及总content半径4，真实冻结decoder候选、同ReferDINO原生mask投影、全轨迹相对ERef保护；比较StateOPD、Direct、DirectMC、仅恢复reference梯度的DirectFullRef，旧几何OPD保留参照。seed41、sigma0.1、tau0.01、lr0.014，统一五级回溯与Adam完整回滚，最多1321次真实完整轨迹评估allowance，OPD/MC最多40轮，Point/Full最多660提案；不padding、不按GT选停止步。科学配置于任何预测前锁定，数学CPU检查通过。初始化身份检查曾把E校准model identifier与checkpoint文件SHA误作等值，实际均在加载模型前失败，0预测；已改为核官方资产登记文件SHA并分别保留校准摘要，错误日志与首次prepared lock保留，不属于方法负结果。执行指纹fdf458c0dbca4991c3035134caea683f53c68b5bc0b4f2c3b8d47e95d431b0e9，真实PID2743719于16:05北京时间启动；首条StateOPD40轮1321真实轨迹评估及初态Source/E逐位核验已完成，尚无GT方法评分。新独立目录outputs/state_opd_hc128，原几何/布局/全量结果不覆盖；正式Vid与CG持续暂停。协议见docs/state_opd.md。

真实状态试验首查询的DirectMC触发FP64奖励权重×FP32状态梯度赋值错误；原PID2743719退出后，0条完整query产物的日志/锁/源码快照保存在attempt0_dtype_failure，未读GT。修复仅在optimizer边界将流式聚合梯度转换回状态dtype，纯CPU与同目标autograd混精回归通过，科学配置不变；原前两个arm不冒充完整结果复用。新PID2760006、指纹ea28cc8fd6f27d0402541e9f36b3c3a2801443eb2dc0f94a4c8f2f7b22530fdb于16:08北京时间重启；首查询StateOPD、Direct和DirectMC均已实际完成，后者1321前向/1280反向正常完成。旧全量及CG未恢复。

新指纹首条完整query已独立只读核验：四组固定P/basis SHA逐位相同、初态E及protected U相同，OPD/MC初轮候选逐位一致、后续共享噪声流；所有path的guard/U由独立NumPy公式复算误差0，共同接纳单调、预算和反向计数一致，FullRef前向逐位检查通过。真实成本StateOPD21.45秒、Direct30.11秒、MC38.09秒、FullRef36.62秒，均1321次轨迹评估，setup另列。此处不含GT方法成绩，作业继续。

运行中用户补充紧凑、可解释且跨帧一致的表示目标。已明确当前8×4 PCA content仅压缩更新自由度，不自带几何/身份可解释性，也不证明多目标关联成立；不据此中途更换本轮状态或配置，待完整结果解释。

用户运行中将本轮缩至64父来源，固定取原HC128输入顺序前64、每父1query，不读取新GT或按结果选样。原在途run/core/config/指纹及科学参数预算不改；独立scope64控制器在第64条完整NPZ/trace/receipt写出后只停止经PID/startticks/cwd验证的自有worker，另封存scope_hc64，后半64取消。已有完整前缀预测直接保留。CPU评分器仅将N和范围通用化，保持原指标及逐query基线核验，不拿旧128均值作64基线。正式Vid与CG仍暂停。


2026-10-21 17:45北京时间HC64真实状态候选实验完成：固定前64父来源、每父1query，全部四组预测及trace/receipt先封存，身份核验后仅自有worker按边界退出，无范围外完整输出。CPU评分完成448项官方/独立dense核对，最大误差9.99e−16；Source/ERef/Strong与旧预测逐位核验，所有正负query保留。ERefStrong/GeometryOPD/StateOPD/Direct/DirectMC/DirectFullRef完整vIoU为31.0918/31.3868/31.0942/31.0965/31.0938/31.1023%，未观测实际节点sIoU为50.7055/51.2469/50.7248/50.7883/50.7800/50.7469%。StateOPD−ERef仅+0.0025pp，−Geometry为−0.2925pp，−DirectFullRef为−0.0080pp（区间[-0.0171,-0.0007]）；没有建立OPD的实质优势。新四组对ERef均无完整vIoU超过5pp下降；StateOPD对旧几何仍损失1条超过5pp完整vIoU及3条超过5pp未观测sIoU的纠正。

同32维新状态内公平比较，旧几何2048维只作方法参照；不能单独归因奖励替换。主适应每query实际平均StateOPD18.03秒、Direct16.86秒、DirectMC32.60秒、DirectFullRef22.10秒，setup与反向单列；这是已有缓存下的增量decoder成本，不是端到端时延。初始化阶段及首轮dtype工程失败已保留，未混用产物。全部91个冻结源、正式paused state及主配置哈希不变，未恢复全量Vid或CG、不追加扫描。独立结果见results/state_opd_hc64；主方法不更改。

同轮事后只读trace审计核256个arm-query全部通过，无模型/新候选/新GT评分。StateOPD输出与E同节点IoU均值0.99484、中位数0.99860，平均角点位移仅占归一化图像坐标0.0585%；所有64条protected U上涨，但22条vIoU下降。2486/2557采样轮ESS≥31/32，候选std(U)/tau中位数0.001235，权重多数近均匀。guard仅4/64条路径激活，不能把零增量全归给保护；也不能因ESS高就认定温度为唯一原因，Adam归一化、基底/半径/decoder响应仍耦合。H014旧几何相对E+8.9822pp，新StateOPD对E−0.0146pp，表现为没有保留旧正收益。聚合与匿名案例见results/state_opd_hc64/dynamics.json，portable CPU核验见scripts/state_opd/audit.py。70项CPU检查通过；未根据结果更改配置或追加实验。

HC64真实状态对照已公开并逐文件核验：提交 [4a9b6a0f](https://github.com/Zonglin-He/A/commit/4a9b6a0f33e8e53890a54efe510ebc7063eaa7c7)，main/research均freshCAS非强制快进；398文件/10,836,351字节远端Git内容与本地导出SHA256及tree完全一致，原历史保留。代码/锁定配置/全部64匿名正负结果/成本/portable事后CPU审计公开，不含数据、GT坐标、mask/特征/状态张量、权重或私人描述。新奖励未支持替换几何主方法，未追加调参或恢复formal/CG。旧两个工程失败单列保存，不混入方法结果。最终档案回执见docs/publication.json，本条及回执在随后文档提交保存。


2026-10-21 用户授权继续HC64：保持mask投影和整轨迹保护奖励不变，先隔离32维PCA子空间与半径4，再按无GT候选奖励校准尺度/温度并真实适应，保留几何与公平Direct。旧几何最终content未在旧产物保存，故用原缓存/原40步/原参数/原seed重放，64条最终框均与旧GeometryOPD逐位一致后取得真实状态。完整state范数中位数13.9068（全部>4），PCA4能量保留中位数0.0234655。完整state/仅R4/仅PCA4/PCA4+R4真实解码vIoU为31.3868/31.1539/31.0910/31.0910%，ERef31.0918；容量结果先全64封存后CPU评分，证实这些限制不能保留原几何的有效纠正，但不是受限空间内最优解的oracle。

后续独立outputs/state_tuning_hc64使用完整8×256 identity content、无半径限幅，奖励/保护/FrozenReplay与上一轮AST完全相同。无GT校准在全64初态各评3×32个真实候选：sigma0.15/0.31/0.61；预设选候选guard=0比例均值≥0.8的最大档，得到0.31（0.88916），0.61为0.78516未通过。tau固定为选中档候选std(U)跨query中位数两有效位，得到0.00037；不按视频挑参数、不用GT校准。配置锁定后启动StateOPD/Direct/DirectMC/DirectFullRef真实适应，同lr0.014、seed41、1321轨迹前向allowance、同回溯与接纳，Point最多660/OPD与MC最多40轮。几何结果复用不覆盖，正式Vid与CG继续暂停，尚无本轮适应后GT成绩。


2026-10-21 后续用户要求从局部参数试验转向事件内、身份一致的整轨迹OPD，并明确要求未观测帧获得正向纠正。已在完整query边界停止自有state_step_hc64进程3186831，核PID/startticks/cwd及退出，保留5条完整四arm预测、trace、receipt和原指纹4536b83d；未读GT、未对不完整面板评分，不将其登记为完成。新算法仍隔离于唯一注册主方法，正式Vid8405/10303与CG保持暂停。已有完整state_tuning_hc64结果不撤销：StateOPD完整vIoU31.1395%、Direct31.4752%、Geometry31.3868%，未观测sIoU50.8890/51.2019/51.2469%；扩容/无GT尺度校准没有建立OPD优势。新方向先复用Source全部时间节点视觉特征与同八ReferDINO mask，核查正向证据传播及真实轨迹多模式比较接口，不再续开旧学习率扫描。


2026-10-21 19:24北京时间整轨迹算法修改已隔离实现并启动HC64真实适应：同一ReferDINO query的八mask经已有Source全部65节点融合视觉网格的固定双向特征关联传播，cycle/前后一致性作未校准置信度，Strong原区间仅作事件先验；未观测节点直接纳入范围奖励与相邻轨迹连接奖励。8×256临时content knots按固定物理时间/置信度映射到各物理节点，两个offset仍由冻结原decoder执行。三模式OPD用32反向成对真实状态候选、经验奖励加权Gaussian条件均值投影，各模式不平均，所有新均值实际解码并经整轨迹保护/回溯接纳。最终来自适应模式均值，不用单个高分候选或GT选择。共同初始三模式读出InitialModes单列。Direct、DirectMulti和DirectMC获得完全相同证据与状态接口，均恢复上游层间reference梯度；MC为各条件风险的路径梯度强对照。每方法1321完整轨迹前向allowance，sigma0.31固定，温度按模式内中心化合并ESS16决定，OPD闭式更新不使用lr，Direct lr0.014。没有新增RGB/Source编码器/专家、网络训练或时间适应。19项新CPU机制/梯度测试、70原工程检查及全64缓存网格/物理映射/固定证据构造通过；置信度不归一化放大，不把cycle一致解释为身份正确。运行指纹8ba4676615af13fa4428559c1d9651dc8a0570ee1cbbb994fcf6720f5335ee19，PID3266238。配置在预测前锁定，完整面板封存后独立CPU评分，未读取新GT成绩。前次仅prepared未执行的锁快照保留在maintenance，不混用。唯一注册方法、旧全部结果、正式Vid与CG暂停均不变。

整轨迹候选实现与已完成的容量/完整状态负结果已公开并逐文件核验：提交 [369202c7](https://github.com/Zonglin-He/A/commit/369202c7320b80c59f63157b29aec30383911ed0)，main/research均freshCAS非强制快进，427文件/11,920,975字节与远端blob SHA256和tree一致、历史保留。新HC64仍在运行，首条独立只读审计通过共同证据/W、实际1312/25/74/1321轨迹调用、零状态/FullRef/冻结及模式均值读出；没有新GT方法评分。此提交不宣称新算法已优于Direct，不变更主方法或恢复正式队列。


2026-10-21 用户要求工程减负和真实评价预算压低至256：已在第12条完整query边界保全旧1321队列全部NPZ/trace/evidence/receipt哈希，核自有worker3266238及CPU完成supervisor3298340身份并停止、确认退出，未读取新GT或对部分队列评分。原源码、锁与12条全部正负输出不改，旧完成发布watch已撤销。独立trajectory_fast目录实现相同top20索引权重聚合、OPD32候选batch reward及事件/置信/网格/E固定统计缓存；不增加传播模块、模式、专家或对照。64缓存CPU奖励值/梯度及稀疏support核验通过，实际缓存decoder的GPU数值/接纳回归待完成。统一256由用户事先指定，先按保存路径恢复12条已完成前缀的严格预算状态，必要仅补每组一次冻结decoder读出，单独封存后检验预算收益保留；12条是运行顺序前缀，不能称完整HC64确认。随后同四组/相同64来源使用256预算独立运行，不混用不同引擎输出。正式Vid和CG仍暂停。


2026-10-21 用户再次明确改变计算/适应结构：尚未启动的256完整轨迹队列和输出相关高斯6-forward实验取消，准备代码及CPU数学检查保留，未生成方法成绩。旧1321轨迹队列12条前缀保持保全/未GT评分，不再恢复。稀疏batch工程的两条已有query GPU回归已完成：相同256调用预算下四方法最终状态、框、所有接纳与模式逐位一致；纯奖励差异在浮点舍入内，但DirectMC其中一条耗时增加，不能宣称所有方法通用加速。新请求将适应位置移至输出侧轻量序列匹配策略，确属方法改变：同一ReferDINO八证据和Source缓存形成最多4候选，学生实际选择更新目标记忆，在访问状态枚举qE并更新6个临时匹配权重，非观测帧直接选择证据支持区域。固定HC64、seed41、3轮×2路径，OPD/固定教师路径KD/完整梯度软路径Direct和无更新固定传播；保留旧几何参照，全64先封存后离线评分，不调参、不新专家/Source/decoder/RGB。新协议docs/propagation_opd.md，配置scripts/propagation_opd/config.json；不是等价加速也不变更唯一注册方法，正式Vid与CG持续暂停。


2026-10-21 20:04北京时间轻量传播策略HC64已真实完成并评分：运行指纹56f2951ad16528eae281cf51002ac05f196cae84a7216b1d045e97830e91e16b，全部64来源预测/路径/候选/receipt先封存后GT，448项官方/独立dense核验误差≤9.99e-16。固定3轮×2路径，三适应组各3次实际Adam/反向，只更新query内6权重；Direct完整软记忆/关联/目标反传，固定KD只生成两教师路径再复用。重网络、decoder、专家、RGB调用0，正式paused及注册配置哈希不变。ERefStrong/Geometry/FixedPropagation/OPD/FixedKD/Direct完整vIoU为31.0918/31.3868/31.1687/31.2165/31.2716/31.2826%，未观测sIoU50.7055/51.2469/51.3409/51.5249/51.5584/51.6015%；OPD相对E+0.1248pp（区间跨零），对FixedKD−0.0551、Direct−0.0661pp，未建立OPD优势。OPD 10改善/15退步/39不变，所有新组共同H035严重下降约10.45pp，不能仅归因OPD。全64临时权重均更新，OPD实际改变158未观测节点；但与固定KD最终仅8query/20节点不同，vIoU仅4query不同。事后candidate GT诊断189/1365未观测节点存在>5pp更好候选，分布45query；固定未观测节点GT读出vIoU32.9824仅作oracle诊断，未在线选择。共享固定证据/候选CPU均耗.0984/.0335秒，OPD.0621秒，含4组输出的64运行阶段29.01秒，非端到端部署成本。还核实本版qE可由显式thetaE=(0,.1,1,.1,.1,0)精确表示，不能声称必须蒸馏才能得到教师策略；FixedPropagation是强基线。新结果results/propagation_opd_hc64及ANALYSIS，保存全部正负与失败；不续扫参、不改主方法、不恢复formal/CG。


HC64轻量传播策略与完整负/正结果已公开并逐文件核验：提交[7d778f9d](https://github.com/Zonglin-He/A/commit/7d778f9d1f18273454efd44f310913b3f95e7bb1)，main/research均freshCAS非强制快进，465文件/13,030,375字节与远端Git blob SHA256/tree一致、历史保留。107项新数学/缓存工程检查和70项原工程检查通过；新方法真实更新/完整梯度、所有inactive fallback及实际计数均核验。全部匿名query正负指标、严重损伤和候选oracle边界公开；不含数据、GT坐标、mask/特征/临时状态或私有ID。档案与发布回执在后续文档提交同步；正式队列保持暂停，不继续参数扫描。


2026-10-21 用户提供v0附件并授权按其执行：新独立propagation_v0，原HC64结果不覆盖、注册方法和formal/CG暂停不变。附件要求候选上限/反馈/历史净收益诊断先于适应；本次固定3候选连通区域(.5相对峰值、IoU.7)、独立前后map及cycle加权支持、无agreement置信惩罚，phi0全0原框先验.7，学生六特征排除u，teacher温度.2且不加pi0，memory写入上限.2*c*u，normclip1、lr.1、prior KL.05、Σc损失归一化与真实greedy fixedscore下降回滚停止。保留有限图精确DP、memory/no-memory诊断与固定KD/完整梯度软Direct，最多3轮×2路径。未在看本轮GT后选工程阈值，附件哈希锁配置。阶段1全64先封存再离线GT诊断，预先锁全方案go/no-go门槛；任何一项不通过就按附件停止适应，不因结果调参或自动扩大backbone。全流程仅缓存CPU，无新重网络/专家/RGB。详见docs/propagation_v0.md。


2026-10-21 v0阶段1按附件完成并在适应前终止：32项v0合成CPU测试（含Direct完整有限差分/非空Adam回滚/DP穷举）及70项原工程检查通过。固定HC64全64 bank/初态两学生路径/MemoryGreedy/NoMemoryGreedy/EvidenceDP真实预测先封存后离线GT，指纹38b838b3d74cea3aacccc5b300c59e0ac96b277ecff0f83c44f4b674b06af5d6，384 dense核验误差<1e-15，旧baseline逐位/指标复核通过。候选节点oracle未观测+5.4833pp（248/1365节点>5pp，545 E-only节点仍计分母），GT逐节点读出vIoU34.3111%；反馈macro pair54.9747%、rho.06137，区间分别跨.5/0，仅过最低point筛查不称可靠。MemoryGreedy/NoMemoryGreedy/EvidenceDP完整vIoU28.8461/28.9642/29.3919%，均低于ERef31.0918；Memory相对NoMemory v−.1181pp、未观测s−.3727pp，远节点s−.9762pp。预注册memory两均值不降且至少一项升的门槛失败，gate=false，按附件先诊断通过再适应的顺序未运行真实OPD/FixedKD/Direct更新（0查询/0更新），未看GT后改参数或强行继续。DP62条代理score上涨而38条vIoU下降，15条严重下降，说明fixedscore接纳非GT保证；不把此诊断说成未运行OPD失败。阶段1CPU运行封存18.45秒，queryloop中位.2307/P90.2539秒，仅缓存增量、未冒充端到端时延。独立status/diagnostics/OUTCOME及全匿名正负保存results/propagation_v0_hc64；原正式paused/default及所有旧源哈希不变。


v0实现与完整HC64适应前筛查已公开并逐文件核验：提交[67621fb1](https://github.com/Zonglin-He/A/commit/67621fb17fcf0e4f301483c0201d3619f9a7b470)，main/research均freshCAS非强制快进；481文件/13,602,274字节与远端Git blob SHA256及tree一致、原历史保留。全部匿名正负、反馈排序/候选oracle覆盖/距离分箱/固定分数错配公开；没有把未运行的OPD/KD/Direct登记为完成方法成绩。原始数据/GT坐标/预测mask/特征/私有ID及权重不上传，正式暂停与唯一注册方法不变。发布回执及档案在随后文档提交同步。


2026-10-21 固定v0候选评分审计与held-anchor缓存探针完成：旧64候选、路径、预测、失败gate未改，0新OPD/KD/Direct适应。EvidenceDP实际746个有GT替换中456伤害(61.126%)，平均节点IoU−5.2015pp；这些伤害在实际邻居上下文全部正ΔS，含incoming/outgoing复核误差2.32e−16。Memory/NoMemory损伤比例亦约62%，support为456/486个Memory坏替换最大正反馈项。219有GT观测节点原生框57.6332%、E53.2302%、mask组件45.0919%；77个Memory观测坏替换中38有更好的native但全选坏组件(17native仍在bank、21被NMS删)，39native自身也差；65坏组件中59面积小于GT，不能把问题全归教师或全归传播距离。几何质量不能单独判语义身份。结果results/score_audit/fixed与mechanism，全匿名正负保留。

同轮跨锚点探针固定64父来源×8fold，rawSource基线、先切keep7 logits重选同一query、只7 masks/boxes构建3槽，全部512 proposal/predicted maps封存后才读held native mask，所有反馈读出封存后才独立GT。只比较同bank投影−.1G与单跳Source对应2DDice−.1G，无memory/link/适应或参数搜索，held自身不替换。Cross相对Projection完整vIoU+0.0627pp[-.6749,.8779]，观测s+2.8926pp[.8987,5.1162]，未观测s−.1943pp[-1.2383,.8361]；actual错误替换比例54.90→55.33%，替换减少但未建立可靠接纳。macro pair56.63→60.66%，配对+4.03pp[.27,7.88]，说明总体排序正信号未兑现为未观测净收益。Cross相对Source v−.1372pp，19升/44降/1不变。不得把Source新bank与旧ERef-v0相减归因。

本轮下游held隔离全512检查及24真实fold极值扰动逐位复建通过；cached ReferDINO仍含8-frame tracking/temporal上下文，明确非独立7帧教师。14新合成+70原工程测试通过，1536dense核验最大6.66e−16，所有旧源码/registered config/formal paused hashes保持。新模型/专家/decoder/RGB调用0；proposal41.67秒+feedback11.96秒仅缓存CPU增量。执行决定不进入OPD完整适应，不增采样/记忆、不改主方法或恢复formalVid/CG；公开代码、锁定配置、聚合与全部匿名正负、隔离证明和负结果，见docs/score_audit.md与results/score_audit/REPORT.md。

本轮评分审计及全部正负结果已公开并逐文件核验：提交[b9d94762](https://github.com/Zonglin-He/A/commit/b9d94762dca48c3a37f3207fc364c758efa1828c)，main/research均freshCAS非强制快进；503文件、15,269,092字节与远端Git blob SHA256及tree完全一致，旧历史保留。公开内容不含数据、GT坐标、mask/特征/轨迹、权重或私有ID。无新适应结果、未更改唯一注册方法或恢复formal/CG；发布回执及档案在后续文档提交同步。


2026-10-21 用户授权完整框范围修正后，独立box_propagation HC64已实际完成：ERef/Strong/原v0六权重策略固定，原生8框连续保留且无E-NMS；softmask×两向Source top20/.1互为对应权重逐轴WLS估计scale/delta，ESS≥3、variance>1e−6、scale[.5,2]及图像交集硬验，失败方向fallback，不clip或组件重建。F/B完整框独立，unary=.5 fullbox RBF+.5 positivecos关联，再按cycle加权；原link/guard/memory/3×2/lr.1/tau.2/prior.05/seed41均不改。所有8native含Strong外/空mask均存slot1，inactive所有真实路径/DP仍E且不写memory。新旧bank各真实执行OPD/FixedKD/Direct，共384arm-query、每组每query3次实际Adam/反向；旧v0未适应历史和failedgate不改，Old三组明确本轮新fit，旧Fixed仅复用封存DP。

全64先封存再GT，指纹c0c372a2246b3bde961263b48304d4ddc6773cdaa0a98dcbb046b5c8c2e66e02，封存SHA7119ec9fedc0e68e6081028a915af24cdb7a16f046cff9eafba7aefe77a307ec；768dense核验最大9.99e−16，22新合成和70原工程通过。完整框Fixed/OPD/KD/Direct vIoU31.3985/31.1909/31.0459/31.1997%，未观测s50.9672/50.7916/50.6770/50.8086%；E31.0918/50.7055。Fixed−旧组件Fixed v+2.0066pp[.5748,3.4517]，未观测s+1.8267pp区间跨零；相对E Fixed仍9条>5pp下降，不能说反馈可靠。OPD−旧组件OPD v+.0754pp、未观测−.0024pp均跨零；OPD−Direct v−.0088pp，63相同/1Direct更好，OPD−KD+.1450pp区间跨零。OPD相对E4升2降58不变，无v>5pp损伤；candidate实际eligible oracle未观测+5.4833→6.6157pp，fallback545→397。范围/固定解码损伤缓解，但未建立OPD优势，不改正式主方法/参数或续扫参。56.75秒为六组适应+bank的CPU缓存增量；新增RGB/模型/decoder/专家0，原formalVid/CG仍暂停。详见docs/box_propagation.md、results/box_propagation_hc64全部签名正负。

同轮独立无GT结构审核确认512原生框逐位保留，1645事件内未观测节点1478有替代、1036双方向、167fallback；WLS5691/7181方向通过，1085因尺度拒绝，传播面积/native中位0.8473且精确等sx×sy（无额外组件截断，但不保证尺度无收缩偏差）。OPD实际只替换15个事件未观测节点，Direct16、Fixed966。第一条receipt独立复算valid/invalid WLS、unary全量、所有候选成员、相同随机流与inactive memory通过，未生成新路径/读取GT。结构代码独立于冻结run源，见scripts/audit_box_propagation.py和results/box_propagation_hc64/structure.json。

完整框范围传播实现与本轮匹配适应结果已公开并逐文件核验：提交[6dff0367](https://github.com/Zonglin-He/A/commit/6dff03674fe2e62fa238bf323e738915bd75e899)，main/research以freshCAS非强制快进；518文件、17,268,905字节与远端Git blob SHA256及tree完全一致，历史保留。公开全部匿名正负/严重退步/成本/固定DP/候选oracle边界/结构审计，不上传数据、GT坐标、mask/特征/轨迹张量、私有ID或权重。本轮真实适应已完成但不建立OPD优势，不改注册方法/科学参数或恢复formalVid/CG；发布回执在后续文档提交同步。


2026-10-21 用户授权冻结完整框bank，执行A/B各OPD/FixedKD/Direct受控HC64试验。A为8特征(原6+delta_u+c*delta_u)与3outer×2路径×最多10inner，B只将teacher即时r改成H4精确离散期望；冻结参考softmax(r/.2)无学生prior/phi，rho=active*r、原memory与previous转移，边界truncate不除H。OPD每outer缓存两当前hardhistory，KD固定最初teacher两路径，Direct每inner重算两softhistory且完整参考prob/memory/target梯度，同outer共享噪声。原prior.7、lr.1、teacherτ.2、clip1/KL.05/memory.2不改；每inner固定S非降检查、失败完整Adam回滚停全fit不读GT。37新合成与70原工程检查通过，H1即时/独立穷举H4/full gradient/storage sharing/margin JSON已验证。

本轮已在CPU实际启动，独立outputs/evidence_policy_hc64，指纹a075662b05e0f7bb5393fcfbda7e14d459aa43e82a2dba87889597a050ff5f9c；只复用上轮64完整bank，六条旧基线逐位继承，不改传播/候选/Strong，不增RGB、Source/decoder或专家调用。所有六组真实输出/路径/receipt全64先封存后GT评分。signed logits间距、prior barrier、实际替换/内层loss/回滚/Q偏好变化和严重退化集成在真实对照，不另开诊断工程。原主方法与正式Vid/CG暂停保持，旧结果不覆盖。


2026-10-21 A/B证据策略HC64已实际完成384个arm-query并封存评分，指纹a075662b05e0f7bb5393fcfbda7e14d459aa43e82a2dba87889597a050ff5f9c；seal SHA0af2c4f3d526627706d069a537950edd24e6a1f1968a4ca8fc5aa129d383e54e。六旧基线逐位复用，768dense核验最大1.11e−15，基线指标差0；37新合成与70原工程通过。A OPD/KD/Direct完整vIoU31.0930/31.3064/31.0989%，未观测s50.6948/50.9100/50.7475%；B31.1227/31.2453/31.1048%，未观测50.7581/50.8315/50.8170%。A/B OPD实际替换事件未观测518/529节点（旧15），logit宏间距−1.3115→+.0727/+.0938；表示/拟合不再不动，但净收益未建立。B−A OPD v+.0298pp区间跨零，未观测+.0633pp[.0033,.1584]；B OPD−Direct v+.0179pp[-.1111,.1875]、未观测−.0589pp，−KD v−.1226pp，仍低于固定DP31.3985与旧几何31.3868。B保存状态Q/r有379/4578偏好不同；OPD/KD3768不同历史中132次目标argmax不同，未证明历史纠错优势。

A OPD7条、B OPD/Direct各6条v>5pp下降，全部公开；B未观测GT替换297中144伤害(48.48%)，恢复48/560同时损坏51/805。S非降接纳不保证GT正确；不事后更换接纳。实际均13.3–13.9尝试步、12.5–13.1接受步，由原S回滚停，不声称跑满30。A/B OPD缓存后.146/.149秒，Direct.482/.702秒；全64六组封存158.44秒，未包含旧缓存重网络成本。全部预测/路径在GT前封存，无新增RGB/模型/decoder/专家；111冻结源、主配置及formal paused state哈希不变。结果results/evidence_policy_hc64含完整正负/间距/损失/退化，本轮完成后不追加扫描或恢复formal/CG，唯一主方法不变。

独立只读64保存trace审计通过660配对field/1984 signed行；A/B OPD outer1/2/3分布37/14/13及38/13/13，多数首次outer因S回滚而未刷新，两者各9query完整30步。B margin同保存history代数：legacy6平均+1.2435、新2+.1418、prior1.3332，isE+.9254，64/64 isE权重负、58/64 negative_guard权重负，不能把增加替换全部归因新证据或称因果消融。完整无新GT/路径/Q的机制审计公开mechanism.json。

A/B证据策略、精确H4反馈与全部HC64正负结果已公开并逐文件核验：提交[c16760ec](https://github.com/Zonglin-He/A/commit/c16760ecbf0b4dea9a35e8717e0d70f40633db91)，main/research均freshCAS非强制快进，533文件/35,097,468字节与远端Git blob SHA256/tree一致，原历史保留。公开完整匿名指标/严重损伤/logit与实际优化轨迹聚合/无GT保存状态机制审计及其复现脚本；不上传数据、GT坐标、mask/特征/轨迹张量、私有ID或权重。没有把弱点估计登记为OPD优势，唯一主方法与formalVid/CG暂停不变。发布回执见docs/publication.json，由随后文档提交保存。


2026-10-21 用户授权先做HC64单次纠正价值与耗时拆分，再按结果决定是否进入纠正轨迹蒸馏。独立correction_value使用同64完整bank/最终B_OPD八权重/Q4/Strong，冻结参数；单条独立seed41学生采样路径，固定Strong物理节点区间四分位每格至多一分歧点(最多4)，必须有后续active未观测节点，不看GT。独立Student/Teacher/Random分支只强制当前一次，其后相同学生/噪声各自memory续走；2动作Teacher=Random必然，policy/noise-only分歧单列。主评价严格排除强制点，预选无tailGT点与空query差0保留全64，按query平均与10000父bootstrap；Teacher−Student及Teacher−Random两主tail95CI下界都>0才进入新蒸馏，否则保存负/不确定结果停止，不改变反馈/预算。16项新core/score合成通过，同prefix/相同原动作完整复现/无GT选点/空值与支持/随机性核验通过，独立只读源码审查无阻断。正式Vid/CG与注册主配置不改。

另固定前16缓存B三组的原样成本重放已完成，48/48所有权重/梯度/loss/接纳/Adam哈希/最终框逐位一致，exclusive计时守恒残差0，6合成通过。profile指纹3c83d74c7d7486ff1d30b1b44acddf222f9aa7974a9e3bea5f8258b61c2aafe7；耗时只拆当前实现，不给新精度成绩；首OPD冷进程启动峰单列，不当作接纳开销或方法速度退化。新增重网络/decoder/专家/RGB和GT读取均0。


2026-10-21 HC64单点纠正价值实验已实际完成并评分：指纹eba054b4ba27b85a02c522a421f52575ce30655b01000f1e3a2d463cf7c1ec3f，seal SHA1e2641f329fd1689326cd27208f6336e1e9c9f34b25bc4213dd8d87b2e0bd379；62/64query预选237点，2无点仍保留，203点/60query主tail有GT，所有预选无GT点差0按query汇总。先全64封存后GT，839dense核验最大1.11e−15，CurrentGreedy逐位/指标等旧B_OPD，115源/default/formal paused哈希不变。22新合成与70原工程通过，无新参数更新/模型/decoder/专家/RGB。

排除强制点的active未观测tail：Teacher−Student+.0078826pp[0,.0236479]，1升0降63同；Random−Student+.0716693pp[0,.2071252]，2升0降62同；Teacher−Random−.0637866pp[−.1913599,0]。教师只改变3/2814次后续active未观测动作，全在随后3节点内，第4及以后0/2276动作/GT定位变化。237点中93为2动作、144为3动作，167Teacher=Random完整相同；101真策略分歧的Teacher尾部收益0，136仅采样分歧有极小正信号。无tail跨.5恢复/损坏或query严重退化，绝大多数没变，不能称有效保护。辅助完整v+.0539pp区间跨零且包含强制点，不能当后续收益。资源gate=inconclusive/false，两主CI下界没有均>0，按用户顺序不进入新纠正轨迹蒸馏、不改反馈/预算/记忆。运行封存21.66秒仅缓存阶段，完整正负/所有分层/成本见results/correction_value_hc64。

独立保存状态审计711分支逐位prefix/incoming/phi/noise通过，无新路径/Q/GT评分。237/237强制后memory变化，2814后续节点memory/logp全变化；far memory差中位.00586、relative-logit扰动.000390、采样winner margin1.05597，previous-feature/link差far均0，实际far动作0变化。无噪声argmax仍17/2814、far5/2276变化，不能泛称所有读出不敏感。已评分174强制点Teacher即时IoU均+2.5067pp(92升72降10同)，92个局部改善点后续主tail动作均不变，说明当前状态影响未兑现为后续纠正，而非memory未传。公开state_flow.json与复现脚本，原推理/评分源哈希不变。

单点纠正价值、保存状态因果链审计与独立耗时拆分已公开逐文件核验：提交[9478c009](https://github.com/Zonglin-He/A/commit/9478c00996a870f2a1dc50a269558151fc212dce)，main/research均freshCAS非强制快进，551文件/38,778,922字节与远端Git blob SHA256/tree一致，历史保留。全部匿名正负/零query/噪声及动作支持分层/失败资源门槛、缓存成本与冷启动限制公开，不含数据、GT坐标、私人ID、mask/特征/状态张量或权重。未在gate失败后启动新蒸馏或调参，正式Vid与CG仍暂停。发布回执见docs/publication.json，随后文档提交保存。

2026-10-21 用户授权直接针对纠正传播弱和教师错选改算法并做真实适应。独立committed_policy保留封存完整框bank/Strong/8教师/Source缓存，修改外部跨锚点一致性加权与持久证据来源承诺两处；不能作单因素因果归因。56有向跨锚点WLS完整框/ROI支持，排除自身，固定分母14；质量≥.1且至少2外部互证、正支持增量才写入来源状态，后续可靠冲突解除。来源不是已验证身份，相关教师一致跟错仍无法识别。四正有界权重/无可学E偏置，3outer×2路径×10inner、所有组30backwards，outer实际固定utility下降完整Adam回滚但继续下轮。OPD当前硬历史、FixedKD初始教师历史/每轮刷新目标、Direct完整软转移与Q4全部可用梯度，固定初态/精确DP/旧几何等强对照。按新请求全64真实三组适应后封存再GT，无诊断门槛，不覆盖旧结果或恢复formalVid/CG；唯一注册方法不变。协议docs/committed_policy.md，配置scripts/committed_policy/config.json。

本轮28项新CPU合成测试和70项原工程检查通过；Direct全梯度有限差分、H4独立穷举、DP穷举、长期承诺/冲突reset与非空Adam回滚后继续均验证。正式HC64真实三组CPU缓存适应已启动，指纹e3bc150097eef5251428acdebba07d291f35f7f6b69e181ce5273b66b39f38c6；全部116非测试运行依赖及每条旧bank/八证据receipt/seal身份已锁定。全部64×3真实适应后再评分，不按首批结果筛选。低质量规则仅抑制承诺写入和route偏置，不删除合法候选，不能宣称教师错误已解决。

2026-10-21 持久证据来源承诺HC64已真实完成192个arm-query、各30反向且全64先封存后GT；指纹e3bc150097eef5251428acdebba07d291f35f7f6b69e181ce5273b66b39f38c6，seal SHA ebf539aa993e565a9a4710b8233884b4db4ae5ed2287d521eb9a9dd7022a9d39。28新合成/70原工程通过，576dense最大1.11e−15，旧4baseline逐位/指标差0，116冻结源与default/formal pause不变。CommitInitial等E31.0918/50.7055，OPD31.0126/50.5983、KD/Direct31.0615/50.6437（完整v/未观测s%），OPD−Direct v−.0489pp[−.1467,0]、0升1降63同；相对E0升2降62同，H043−3.1306pp、H063−1.9353pp，无v>5pp下降。本版DP31.5294/51.2172，较E v+.4376pp区间跨零且6条>5pp下降，不凭点均值推广。

三组最终均6段非E来源持续≥4节点、最长12，长route正bias并实际选非E的未观测节点23/2query；未做route消融，不作GT或单因素因果证明。真实final reset0次，不能称已解教师错误。OPD替换29未观测节点，其中9有GT：2改善6下降1同，4原≥.5跌破.5；Direct/KD27节点其中7有GT：2改善4下降1同。512锚316通过跨锚点门、2800合法active替代槽1489可commit，质量/见证仅抑制620承诺不删候选；最终teacher25个可评分未观测建议4有害（OPD状态），一致错选仍未解决。三组接受outer189/187/189，拒绝完整Adam回滚仍执行下一轮，非只诊断。缓存适应均.1543/.1538/1.9681秒，全64封存175.59秒，原缓存成本不计入、非端到端。全正负/严重/更新与结构公开results/committed_policy_hc64；不追加GT选参、不改主方法或恢复formalVid/CG。

本轮保存态读出核验（无新路径/Q/GT）另确认OPD teacher/student72节点分歧，非E目标65；非E相对E平均logit−.740625=先验−1.124557+已学特征.302753+承诺.081178，重建误差8.88e−16。四权重[0,4]局部上界仍无法越E仅2/65，不能泛称硬界锁死；共享四权重/当前目标拟合和读出仍有限制。structure.json保留全部匿名计数和真实长状态/行动核验，无route消融不作因果或语义正确性保证。本轮没有按这些事后诊断追加fit或调参。

持久来源承诺实现与完整HC64真实适应负/正结果已公开逐文件核验：提交[9714ae4c](https://github.com/Zonglin-He/A/commit/9714ae4cea5fa86f3fba2d7d7691a83bfa7b735f)，main/research均freshCAS非强制快进，566文件/55,035,885字节与远端Git blob SHA256/tree完全一致，历史保留。完整匿名正负/严重/actual30步更新及固定DP、缓存成本、保存态结构和读出限制全部公开，不含数据、GT坐标、mask/特征/状态轨迹、权重或私有ID。未宣称解决一致错选或OPD优势，不改注册方法、正式Vid/CG暂停；发布回执见docs/publication.json，由随后文档提交保存。

2026-10-21 用户新附件明确覆盖旧唯一OPD约定：空间主线切换原BoxFixed（31.3985版本）固定完整框传播/原score/精确DP，Strong保留已知参考；不选committed新DP、不继续空间OPD。旧Git579757a与私有outputs/maintenance/boxfixed_switch_20261021/legacy保存完整旧模块/116原源哈希及原default，旧receipt/lock/seal不修改。新增stvg.evidence/spatial/native/tube_replay和独立BoxFixed入口，默认无OPD/optimizer/可训练状态，raw视频文本同一八帧专家一次输出框mask可独立运行，缓存可选。全64迁移CPU回归2240张量逐位，原DP动作/框/score/Strong区间保持，见results/boxfixed_migration.json。full_timeline是明确新扩展，geometry独立于Strongevent，非等价整理。

本轮固定HC64 A旧BoxFixedStrong/Bnative224/Ctube/D同面积r/time/预算固定空间置换，alpha.5/context1.5不扫描；B/C/D不运行448、不生成Strong专用六层融合。必要224捕获末次真实时间decoder motion/text/query/pos/padding/twooffset输入，最多每offset额外重放1次，最终新区间只解一次BoxFixed，全部封存后GT。另固定同空间轨迹评价纯时间作用，成本拆共享/分支及cachehit/miss与历史A，不预称加速/等精度。协议docs/boxfixed_temporal.md。正式Vid/CG持续暂停，其他backbone新接口未验证、不自动启动。

本轮迁移验收中发现并修复原零残差加法的布局作用：移除可训练delta后保留固定零加法，以维持原expandedquery物化，实际Source/E与旧缓存逐位恢复。时间缓存仅clone数值会丢stride0/gap/offset，新增actual prehook布局元数据及最终设备as_strided恢复，alpha0真实GPU的hidden/logits/物理后验/MBR已逐位通过；不放宽阈值。后续raw BoxFixed进一步查出CPU4线程与原单线程FP32对应点积产生约1e−8框差，正以原单线程作用域保留算术，再验收。工程失败与快照私有保留，全64正式时间试验尚未启动，无GT成绩，不作为方法负结果。

本轮工程验收已通过：90项当前工程检查、64空间迁移2240张量逐位、真实GPU Source/E/8观察位置/原BoxFixed+Strong框区间与alpha0 hidden/logits/posterior/MBR逐位。固定零加法/actual tensor stride/offset恢复/CPU单线程作用域分别解决布局与内核差异，失败尝试私有保留，未放宽容差。已有query独立raw视频文本、无feature/evidence缓存注入运行tube通过，单专家实际1前向、4480、额外时间decoder2、无更新，wall12.407秒（Source冻结权重已CPU驻留，非冷应用总启动）。同query单独448分支3.883秒与C重放/两DP .0869秒为有限分项实测，不当64组端到端对照；完整smoke见results/boxfixed_native_smoke.json。新研究指纹99b5f8f873b2accb32c0fa46a9e91a9f4e1190d77265023b33c0e4fb25315c9c，现按锁定配置启动完整64，不扫参，不恢复formalVid/CG。

2026-10-21 BoxFixed主线迁移及固定HC64时间A/B/C/D已实际完成，指纹99b5f8f873b2accb32c0fa46a9e91a9f4e1190d77265023b33c0e4fb25315c9c，seal SHA8ece18c5e62561a099ca1dc93eba0341977cb70dbdd11cf15598d4dae6bb6206；全64先封存后GT，512dense核验最大5.55e−16，旧A指标误差0。原BoxFixedStrong/native224/tube/置换完整v31.3985/29.0568/28.8196/29.1586%，t58.6043/54.8230/54.4344/55.1254%，BoundaryMAE2.5113/2.6821/2.7075/2.6754秒。C−B v−.2372pp[−.4401,−.0710]、t−.3886pp[−.6844,−.1370]，C−D v−.3390pp亦负；纯时间固定A框C−B−.2378pp，平均损失主要为时间区间。C−B4升9降51同，无v>5pp损伤但有1条t>5pp退化；C−A19条v>5pp损伤全部公开。不能声称空间位置反馈有效或等精度替代Strong，默认BoxFixed+Strong保持，tube为失败的固定实验配置，不追扫参数。

本轮真实64均新RGB/224、八专家缓存全hit，Source/E/位置逐位旧缓存；B额外time0，C/D每query2（每offset1），无448/新专家/参数更新。共享capture均4.0364秒，C额外decoder/head .0430、两DP+重放/搬运finish .0810秒；单路径分项估计C5.1991秒排除outputIO，实际三分支trial5.3211秒/条，封存全wall356.64秒，不能相混。有限同query4483.8835与C .0869秒附加分支差3.7966秒不当64端到端倍率；原始专家miss smoke12.407秒单列且Source已CPU驻留。全部结果/匿名正负/区间损伤/纯时间与成本在results/tube_temporal_hc64；旧源码116哈希、旧回执、formal paused状态保持。当前正式空间无optimizer/可训练state，不依赖scripts或历史outputs；TA接口已验，其他backbone未验，不恢复formalVid/CG。

全64独立保存态结构审计通过：429源码/快照、正式暂停状态/default不变，C/D辅助框动作score/r/时间支持全同；4160节点M合法直方图逐位、4149实际置换，1536层每head/query attention视觉mass最大误差2.98e−7，text/padding0。真实64均无padding格，不能扩写成非空padding实测；合成非空padding验收另存。预算nativeoffset128、原time/position各512，C/D各额外128time，448/新专家/反向0。匿名structure.json及独立只读脚本公开，不读取GT或生成新路径。

BoxFixed独立主线与固定HC64时间验证已公开并逐文件核验：实现提交[1af13d7c](https://github.com/Zonglin-He/A/commit/1af13d7cb4d94b74881477b8de233496ed83afbf)，main/research均freshCAS非强制快进；590文件/56,263,682字节与远端Git blob SHA256/tree完全一致，保留旧历史与全部旧路径。公开本轮负结果/768匿名签名比较/损伤/纯时间/真实成本/迁移与注意力审计，不含数据、标签坐标、视频正文、原始轨迹/mask/特征/状态、权重、私有ID或凭据。默认BoxFixed+Strong；不恢复formalVid/CG，不追加参数搜索。发布回执docs/publication.json由后续文档提交保存。

2026-10-21 用户明确整体收益以未适应Source为基础，Strong仅历史精度/成本参考，不作为新空间方案必须超越的门槛。本轮授权空间锁定原BoxFixed，仅用已封存G/T/P各六层hidden经同Source原冻结时间头读出，先逐位核验末层logits/物理后验/MBR和旧BoxFixed动作/框/score，再平均各层归一化物理区间后验；固定比较qG、.5qG+.5qT、.5qG+.5qP，不均logits、不调整attention/context/r、不得新RGB/encoder/decoder/专家/训练。Source基线为原生未适应Source框＋native224末层区间；相同A空间轨迹另看纯时间贡献。全64先封存再GT，独立outputs/tube_fusion_hc64，不覆盖旧结果；若真实反馈仍无优势停止该方向，不扫融合权重，不恢复formalVid/CG。四项纯合成核验已通过，实际头读出核验随后执行。

2026-10-21全局/轨迹六层后验等权缓存融合HC64已实际完成并评分：指纹fa9f761b40c1b4b66272ccb1d221567ca8aa4d0faf71e5c9292e8a882c6cc947，seal SHA76997dd14e54c80ba2c0435a6089f9018e485c96432300413948177631d3ba80。Source_Native224明确为未适应原生Source框/末层时间，不含Strong或ERef；Source/Global6/GlobalTube6/GlobalPermuted6 v26.8938/29.6952/29.7235/29.6977%、t54.8230/55.9613/55.8081/55.9221%、BoundaryMAE2.6821/2.6034/2.6163/2.5990秒。融合−Source v+2.8297pp[.9075,4.8356]，41升22降1同且8条>5pp损伤全部公开，整体收益成立于此开发面板；t+.9851pp区间跨零。融合−Global v+.0283pp[−.1076,.1781]、t−.1532pp，融合−置换v+.0258pp亦跨零、t−.1140pp。固定A空间v差−.0137/−.0161pp，不能把最终选框微小收益当时间贡献。Strong31.3985仅参考，不是门槛；结束该反馈方向的依据是无反馈/置换无优势，不扫融合权重。

全64新输出先封存后CPU标签，640dense最大6.66e−16、Strong指标差0。新增RGB/encoder/decoder/ReferDINO/反向/更新0；只原model_ema时间头66306冻结参数，384实际head批次末层逐位恢复，192后验/MBR与192旧空间动作/框/score恢复。同Source checkpoint SHA与原捕获runtime资产一致，1152各层规范化/均后验及64MBR独立保存态审计通过，14个融合区间不同于Global、18不同于置换。执行及末尾核验阶段6.515秒不含prepare，当前三读出head/后验/DP及恢复检查均.04497秒/query、inputIO.00988/outputIO.00332；不是部署速度，部署仍需两个offset时间重放及共享224/专家/传播。本轮结果results/tube_fusion_hc64，协议docs/tube_fusion.md，生产配置及所有旧seal/formalVid8405暂停/CG暂停保持。

本轮94项当前工程测试通过（新增4项），档案check通过。公开896条匿名配对signed行，完整Source总体对照、无反馈/置换消融及Strong参考分开；不将CI跨零写为等精度。保存态审计不生成新模型路径或GT分数。后续仅归档与GitHub发布核验，停止实验。

本轮固定缓存六层后验融合代码/配置与全部正负已公开逐文件核验：实现提交[c181e2c4](https://github.com/Zonglin-He/A/commit/c181e2c475ff953ee42270e3fafce16debb02a8b)，main/research均freshCAS非强制快进；602文件/57,504,130字节与远端Git blob SHA256/tree完全一致，保留所有历史。公开896匿名signed、Source整体收益与无反馈/置换消融、8严重损伤、纯时间和缓存/部署成本边界；不上传数据、标签坐标、原始query/视频、权重或原始缓存。正式Vid/CG和生产配置保持，按锁定结论停止反馈方向，无融合权重扫描。发布回执docs/publication.json由后续文档提交保存。

2026-10-21 用户授权最后两轮时间任务书（5c47b041）：固定原BoxFixed/Source/224两offset六层/八专家/原MBR，不动readout观察规则；第一轮HC64缓存signed误差/有效质量/13候选oracle诊断，第二轮只改完整区间均匀混合vs旧hull做2×2 MAP/MBR，随后两方向各32预锁parent复核。最新补充明确需全面时间瓶颈排序、自主依据证据改进，最终同时轻量与涨点；不能把纯缓存速度当部署收益。无新448/条件decoder/专家数量/训练或OPD，不扫权重。

确认provenance已只读核：两域历史full均曝光，可证从未曝光parent均0，不包装成独立确认；按当前官方manifest排除开发HC128/Vid128父后分别有105/604可用父，按(parentID,queryID)固定序每父首query取32，两方向各32与当前开发父零交集，称冻结父隔离开发复核。名单outputs/interval_pool/confirmation_selection.json SHA c59a0e54b3f9a6bc9f6b7e601054aa31d6482bc9afd1b95efc0e5b39737dcb94在任何新GT评分前锁定。选中HC32/Vid30虽有正式Source/E最终框与boxes-only教师，但全64均无Source spatial/六层hidden/native mask，故第二轮按授权每条一次共享224＋一次同专家八帧含mask捕获，三臂共享，绝不补448或恢复formal。9项新区间算子纯合成测试通过，MBR原实现不改，特别验证最优决策可为q0候选。第一轮只运行冻结小头与概率/MBR读出，全部seal后独立GT。


2026-10-22 最后两轮时间任务已实际完成。第一轮HC64原小头/13个预封存读出及hull/mix后验先封存后GT，旧Source/Global6恢复，1024dense最大6.66e−16。起点signed−.9683秒/终点−.0364秒，过覆盖3.0144/欠2.0825秒；短事件偏过覆盖、长事件偏欠覆盖。hull有效质量m.5/.7相对原view平均+19.8471/+9.5888pp，均CI>0，长度恒等误差1.95e−14秒，因此“外包为主要损伤”未获支持。13固定读出oracle t+7.6795pp仅GT离线诊断，不作为部署成绩；完整448归因未测，不补448。

第二轮HC64固定2×2全部先封存后GT。Hull_MBR/Hull_MAP/Mix_MBR/Mix_MAP完整v29.6952/27.4833/26.4743/13.2052%，t55.9613/50.9166/49.2917/23.6288%；新主臂较旧v−3.2209pp[-4.5709,−1.8970]、t−6.6697pp[-8.8123,−4.4825]，13升48降3同、18条v>5pp下降，固定空间亦降；原MBR对各自MAP均改善，未改原接口或q0候选，不把分布失败归为MBR错误。

随后事前锁定HC32/Vid32父/视频隔离开发复核全部实际新capture，唯一指纹45d0f98444c0d64a8f38d5b4255f29e1ee774f1c1a60175691975a759f2e7ceb；64条均预测seal后独立CPU GT。HC Source/Hull/Mix完整v23.4655/27.8907/27.3192%、t50.4639/51.6010/49.2260%；Vid15.6772/23.9576/22.9325%、t34.0110/35.6554/33.8749%。Mix−Hull HC v−.5715pp/t−2.3750pp、Vid−1.0251/−1.7805pp，各配对CI跨零但双域点估均未过预锁无退化判据；严重v下降5/6条，固定空间亦负。Hull−Source完整v HC+4.4253pp[-3.3213,12.3936]、Vid+8.2804pp[3.5570,13.5748]，作为开发信号，不宣称独立泛化或OPD/空间时间反馈收益。按事前规则淘汰唯一mix、保留Global6+旧hull+原MBR研究读出，停止无第三权重/温度/长度方案；生产默认Strong未自动切换。

两域新64条各一次共享224、同一八帧专家各一次，448/条件decoder/反向/更新0；62已有历史Source及8位置逐位恢复，768物理view独立scatter差1.39e−17。105工程测试通过，本轮干预+双域1024dense最大1.44e−15，正式状态/default/431冻结源哈希不变。HC/Vid共享capture均3.5968/3.8904秒、RGB .9046/.3411秒，三臂query wall4.5314/4.4724秒；首条含模型延迟加载，process/import/setup/哈希在外，不称单方法完整冷端到端。Vid后验构造hull179.7521ms/mix4.9987ms，仅构造更快但精度下降，未登记为有效加速方法。全部signed/严重/阈值/固定空间/实际成本在results/interval_pool，全面时间瓶颈按严重性、结构事实/待验根因及原论文来源在docs/temporal_audit.md/temporal_sources.md。私有快照/缓存/输入与旧结果保留；正式Vid8405/10303及CG持续暂停。后续仅归档与GitHub完整发布核验。


最后两轮区间研究、双域冻结复核、全面时间审计及全部正负结果已公开逐文件核验：实现提交[6d021fe9](https://github.com/Zonglin-He/A/commit/6d021fe9eea56abe6e2645eb8b8a2cf7ce370334)，main/research均freshCAS非强制快进；633文件/60,681,542字节与远端Git blob SHA256及tree完全一致，原历史保留、无删除。新mixture按双向失败判据淘汰，保留旧Global6研究读出，生产Strong默认不变；不将Vid构造179.75→5.00ms当端到端或精度贡献。全部匿名signed/严重损伤/首条及余31成本/概率映射与暴露边界公开，不含数据、GT坐标、原视频query、mask/特征/状态张量、权重或凭据。正式Vid/CG暂停保持，停止本轮实验；发布回执docs/publication.json由后续文档提交保存。


2026-10-22 用户进一步明确授权：上述任务完成后在两个方向按既往单因素/机制解释/粗细邻域方法有限调参，根全程盯调参，锁定后运行新全量队列，可用一个gpt-6-luna/max巡检，异常由根核证修复续跑。此授权覆盖旧本轮“停止调参”约定，但不恢复旧formal8405队列、不混用旧方法预测或改标签。新outputs/boxfixed_tuning_v1使用HC64/Vid64各一query/父，HC复用已验证完整缓存，Vid64一次共享224/八专家捕获已全封存，后续候选均0新增重网络。固定Source/8观察/原完整框候选/初始化gamma/原hull与精确MBR，只调最终时间温度、观测/未观测权重、几何保护、邻帧连接。每方向最多18配置，根逐阶段回读完整v/t/未观测s、严重>5pp损伤与后验熵/长度/实际动作变化；先单因素粗搜，最多4邻域细化、4参数撤回归因及1交互，禁止全笛卡尔/选seed。plan SHA537aac2b6523406788cbce804ee8b3364c29743bdaac9e87dfb266df1cf3744e在首个新GT前锁定。64封存后CPU评分；参数锁定后32×2一次冻结开发复核，若未过事前判据回退原配方，不据复核继续调；所有数据历史曝光如实标记。全量用新独立目录与新不可变profile，不覆盖旧源码/结果或放宽resume指纹。


本轮调参实际完成HC14/Vid17配置，所有64预测seal后CPU评分，根逐轮选取与解释保存在results/boxfixed_tuning_v1/decisions.jsonl。开发最佳HC(temperature1.5,obs.75,guard.2,link.25)完整v/t31.1883/57.7346%，较原+1.4931/+1.7732pp、未观测s+1.1807pp；Vid(3,1,.2,.25)23.8144/45.7255%，较原+5.1881/+7.0192pp、未观测s+.8945pp。Vid obs1使未观测直接unary为0，仍有传播候选与link，不能把收益称为更强未观测监督。逐参数撤回和邻值结果保留，不无限扫边界。

随后两域所选配置在一次冻结32父开发复核均失败：HC v−.2159pp/t−.9061pp/未观测s+1.2465pp，Vid−2.1487/−3.0871/−.9330pp，Vid严重Source损伤7vs基线3，按事先判据两域完整退回原(1,.5,.1,.25)，不据复核再选参。Vid开发/复核GT事件时长中位6.6400/2.5192秒、<5秒23/64vs25/32；高温展宽适配长事件开发的解释与此相符，但非受控因果证明。原生224/八专家/候选/gamma和精度评价不变；113工程测试通过。缓存数值回归两域各miss/hit，共4已有query前向、0新专家，Source/E/8位置/全部六层logits/31几何张量逐位；启用512MiB精确RGB和4GiB实际tensor visual LRU，禁止文本/fusion/decoder跨query复用。

用户授权的新全量plan已锁定2ebd134dfd055c8e17e01859b706d56d187b15b8907a8129c49669539d84ea6e，目录outputs/boxfixed_full_v1，HC1902→Vid10303，原formal暂停不改。configs/tuned_boxfixed_v1.json明确两域fallback；Final与Untuned为alias，不重复DP/后验/评分，不冒充独立两次实验。192完整capture严格内容绑定导入，cache原成本不计当前；全部新真实流含Source配对，完整seal后才CPU固定labels/parent-cluster CI，不看full结果调参。首个后台启动PID606545未存活、空log且无state/worker/预测，核/proc及pgrep无自有进程后保留空日志，以Popen新session启动同一plan PID607450，未改冻结代码或参数，原因未证实不编造。单Luna max旧ta_luna_watch复用，只读巡检；ta-stvg-luna原heartbeat按新plan更新为半小时ACTIVE，正常静默，实质故障根核证修复、不重复启动、不放宽hash/混用指纹。所有其他旧队列/CG仍暂停；全量结束后汇总完整signed/严重损伤/效率并发布核验。


双向有限调参、31配置全部匿名signed、双向冻结复核及回退、缓存逐位回归与新全量控制代码已发布并逐文件核验：[e4b317ae](https://github.com/Zonglin-He/A/commit/e4b317aea8895d884b1f97469d58f2e0bde14c13)。main/research均freshCAS非强制快进，688文件/67,265,274字节与远端Git blob/SHA256/tree一致，历史无删改。开发最佳不迁移，最终两域均原参数；不宣称调参收益或全量完成。当前新HC1902→Vid10303运行中，由唯一Luna max半小时巡检；旧formal8405/CG仍暂停。docs/publication.json保存实现核验回执，后续文档提交只保存该证明；全量结果等待完整封存后另行发布。


2026-10-22 03:10北京时间半小时巡检确认新全量HC1902已全部seal并CPU评分完成，原controller607450自动切至Vid worker776574/startticks217423834，plan/frozen434源/profile/default/旧formal哈希均不变。HC seal SHA d0eaf834a3af27b8fda22693d03b741d54387fc5e981e82684df6bdec4d41ee3；1902预测与各自receipt及3公开文件逐文件hash复核，3804官方/dense最大1.33e-15，Final/Untuned1902别名相同。HC Source/Final完整v24.9948/26.7597%，v@.3 37.8023/40.4311%、v@.5 12.8812/15.0894%；配对v+1.7649pp，227父query加权cluster95%CI[1.2804,2.2751]，1099升678降125同、266条降>5pp；未观测actual s+1.7881pp[1.0863,2.4902]，t−.0666pp[-.3408,.2169]，不能宣称时间净收益或OPD贡献。当前方法为完整框+Global6，不是旧OPD。完整划分含开发曝光按原声明。HC1839新capture、63完整复用、零visual/RGB LRU命中（精确网格唯一），新capture均wall4.2775s，所有1902均4.1366s是当前配对工作成本、非Source/Final独立端到端。Vid19:10UTC264/10303，RGB153hit/108miss、visual612hit/432miss均58.62%，近期3.5159s/条，剩余估计9.804h为早期输入分布估计；不打断测速、不改配置或据全量再选参。HC报告及全部signed已生成results/boxfixed_full_v1/hc2，等待双域完整结束统一GitHub发布，未宣称已远端上传本次HC结果。旧formal/CG保持暂停。

同turn只读自然正式回执分项：Vid首285条含3完整capture复用；其余164条视觉全命中、118条未命中，命中/未命中query wall均2.5474/4.5373秒，Source224均1.0432/2.5654秒，RGB均.0125/.3145秒；整体285均wall3.3457秒，当前专家wall均.5869秒、Source迁入/出GPU均.1342/.2701秒、posterior.1254秒、DP.00144秒。这是不同query组的早期观测，非同输入受控加速倍数，不称端到端独立单方法性能；不额外开模型或打断任务。私有明细outputs/boxfixed_full_v1/monitor/vidstg_early_timing.json保留，完整域后再统一成本统计。

2026-10-22 用户明确要求先离线验证原骨干/完整框空间轨迹是否已有事件和边界信息，自主探索注意力而非先验否定。根未委派新研究agent、不改运行中的正式队列，仅封存缓存CPU审计：HC64＋父隔离HC32＋Vid32共128query/10017节点，八项主问题先锁定，所有无标签曲线先seal再独立GT；历史曝光如实保留。原生真实TimeDecoder输入用同model_ema仅CPU子图重建256次，1536层hidden核验最大3.33786e-6低于事前1e-4；新RGB/encoder/专家/GPU/更新均0，无标签提取22.3407秒是诊断计算而非部署速度。

未观测事件内外真实−同面积置换位置attention AUC HC64/HC32/Vid32=.6531/.6258/.4684，HC主问题BH q=.0004/.0064但Vid未复现；原空间r/u接近随机，不能当事件概率。目标ROI注意力在起止普遍增大未获支持，HC64起点附近面积校正注意力比事件内部低.01664，95%CI[−.02778,−.00588]。三组hidden变化的边界AUC=.6404/.6640/.6288，BH q=.0004/.0008/.0120；已有时间后验正控更强，事件AUC=.8402/.8256/.7161。明确看到主结果后再做缓存位置/裁剪/原头敏感性，另先seal后GT；排除裁剪边缘35/20/14query仍有hidden关联，但位置/锚点/面积残差仅.5314/.5357/.4911，再扣原boundary后验为.5069/.5270/.4882，CI均跨.5，未建立独立新增时间信息；不把残差阴性当空间时间联合不可能，也不据原相关直接改方法。

独立rank-sum重算245项与主AUC完全一致，原128input/receipt/posterior/signal hash复核、八观察和attention概率关系通过，HC父交集0。434正式源/plan/default/profile不变，Vid实际worker身份及进度4763→4846前进；旧formal/CG不恢复。新代码scripts/temporal_signal_audit、结果results/temporal_signal_audit_v1、协议docs/temporal_signal_audit.md，完整正负及次级层/起终点/事后控制公开，不含query/GT坐标/原始attention/特征/框/权重。没有新时间预测、vIoU成绩、调参或方法采用；下一方法必须证明超出现有时间后验的纠错价值。

缓存时间信号审计代码、128匿名正负关联、全部次级层/起终点/事后敏感性与独立核验已公开逐文件验证：[1160b1bb](https://github.com/Zonglin-He/A/commit/1160b1bbb4ec52f40614ddda95385ad63a572c57)。main/research均freshCAS非强制快进，702文件/73,944,036字节与远端Git blob SHA256/tree一致，历史无删除。空间/时间原配置及正式434源不变，无新预测/训练/专家/GPU。当前HC full已封存但其公开文件暂留本地，Vid完整域尚在运行；本次离线审计提交明确排除独立的待统一全量报告，不宣称双域完成或新方法涨点。发布回执docs/publication.json由后续文档提交保存。
