# 项目历史与当前状态

当前任务：修正官方标注后的TA-STVG双向评估已重新启动，HC1902条/227父、Vid10303条/732父；162条同输入同模型预测核验复用。旧raw3482任务及295条partial保留归档，CG仍暂停。方法参数、输入采样和专家不变。

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
