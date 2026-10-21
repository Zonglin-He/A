# BoxFixed主线与单次时间重放：冻结实验卡

授权为用户2026-10-21附件（67f00835-f809-47f0-bce9-26884eed8300）：空间锁定原BoxFixed31.3985版本，替换原OPD主线；先工程迁移，再HC64固定一次时间验证。不是继续空间奖励/OPD研究。数据为原HC128前64父来源、每来源一query、已曝光开发数据，所有正/负/不变样本保留。

## 工程与历史

正式空间代码在stvg.evidence/spatial/_spatial_math，主入口python -m stvg。核心不import scripts.*，不读历史lock/gate/outputs，无优化器/可训练状态/Q4/持久承诺。支持原视频文本/Source/同一ReferDINO独立推理；专家同一次返回所选query八框和native软mask。原ERef校准/强度、Source实际FP16前缀与FP32两pass/offset顺序保留。默认Strong是已知参考；native224/tube_replay显式模式，后者实验性，不据开发均值自动成为最终时间方法。

旧代码/配置已在Git579757a及私有outputs/maintenance/boxfixed_switch_20261021/legacy逐源哈希保存，旧116源、所有receipt/lock/seal及资产不改。历史验证从旧快照运行，绝不放宽哈希。configs/legacy_opd.json保留旧数值。正式Vid/CG持续暂停；CG/TubeDETR新空间和重放接口尚未验证，公开入口不能静默改用其旧OPD。

HC64 CPU迁移恢复全部几何张量、原区间event/active/valid/score表、DP动作/框/分数/Strong区间，逐位结果见[迁移证明](../results/boxfixed_migration.json)。这是工程回归，不是新精度。

## 时间方法与固定对照

几何不接收时间mask。full_timeline辅助视图沿用原固定评分项和权重，在全时间轴选择轨迹：obs8、unobsT−8、全时间边T−1，原confidence乘积下权，guard全T。与原interval解码区分，是明确扩展；不读Strong event/active/区间内外身份，不新增专家位置。

保存原生224实际末次TimeDecoder输入，含motion/textmemory、query/content/time、pos/padding和两offset物理帧号；不以空间appearance ROI代替motion。上下文范围为辅助完整框宽高1.5倍，仅构造连续网格占据M，不改输出框；r来自原对应可靠性，有效seed/对应缺失时0，不是事件概率。alpha=.5，w=1−alpha*r*(1−M)，逐head/query保持同帧合法视觉attention总mass再分配，文本attention不直接改动。alpha0直接原路径并需逐位恢复。每offset只重放一次必要时间decoder/head，两offset实际记2次；原生多次调用亦独立计数。不运行448、不生成Strong专用六层平均后验、不训练头/权重/动作监督。

| 设置 | 空间/时间 |
|---|---|
| A_Strong | 原封存BoxFixed＋Strong参考 |
| B_Native224 | 相同几何，原生224最后层后验＋物理MBR，然后最终interval DP |
| C_TubeReplay | 全轴辅助DP→一次缓存时间重放→MBR→一次最终interval DP |
| D_Permuted | 与C完全相同预算，M在每帧合法视觉token固定半循环位移；保持M多重集/面积、r、时间支持/padding |

C/D均独立读相同辅助轨迹与r；不根据GT选置换。一个query共享同一次224捕获/同八专家证据，分支各自计重放。最终区间后只解一次空间DP，不迭代。研究复用既有expert native maskpack时，严格核字节/Sourcecache/selectedquery/八框/位置/帧号，实际专家调用0，不把旧专家成本当本轮新计算；另对已有query验证原始输入独立运行与真实专家cache miss。

## 评价与成本

全64所有A/B/C/D预测/interval/receipt/capture先封存再独立读取GT。主要C−B和C−D，报告tIoU、起止边界MAE秒、完整vIoU、区间长度秒/变化、全部匿名signed、严重损伤和10000父来源配对区间。相同A空间轨迹分别用各时间输出再评分，分离纯时间变化与最终DP选择的影响。跨零不是证明等精度。

成本分开原生224、专家当前/原始、证据、辅助DP、额外decoder/head、时间后处理、最终DP、搬运/IO和wall。B/C/D共享捕获成本与各自实际分支成本组成单路径估计，多分支trialwall另报，不冒充每组独立wall。A复用旧预测没有本轮同条件完整平均时延，旧448成本只能明确标为历史；已有query的真实Strong/raw smoke可以给有限同query比较。缓存命中和未命中分开，无448调用审计，不能只报decoder毫秒或预先承诺倍率。

仅锁一次配置，无参数扫描，不根据GT改变alpha/上下文/置换/样本/接纳规则。完成后报告主线切换、空间反馈净收益/损伤、实际计算与证据边界，不再自动启动新OPD分支或全量任务。

## 已完成的固定试验

64/64封存后CPU评分完成，见[结果判定](../results/tube_temporal_hc64/ANALYSIS.md)。A/B/C/D完整vIoU31.3985/29.0568/28.8196/29.1586%，tIoU58.6043/54.8230/54.4344/55.1254%。C−B v−.2372pp、t−.3886pp，区间均在零下；C也低于置换D，不提升默认。C全finish均.0810秒；64条专家全hit、224全miss，不能把共享分项估计写成独立端到端加速。完整损伤/纯时间/成本与哈希证明均保留，不追加搜索。
