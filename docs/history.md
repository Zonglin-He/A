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
