# 项目历史与当前状态

当前任务：TA-STVG在HC1902条/227父上已完成封存与五行评分；Vid10303条/732父于22:07北京时间启动，完成296条后通过一次维护切换到数值一致的工程加速版本。评分JSON标量写出故障已独立修复，冻结源码、计划、参数、标签及预测均不变。CG仍暂停。

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
