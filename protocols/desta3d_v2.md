# DESTA-3D v2：独立证据路径与源训练修复

登记日期：2026-09-27。依据用户最新直接指令；旧 v1 E5 已完成并保留，未启动的 v1 residual-scale 扫描和 E6 后续被本版取代。v1 阴性排序不作为本版架构结论。

## P0：语义与真实接口

冻结官方 PTD-4B（权重 SHA cc78a8bc1d3d341f70b6b235af160d1c85635046fb78b82fc2f9759e717e5a88）。保留官方 visual merger 输出的真实 THW grid，2560 维。共同 learned projection + 轻 THW stem 后，分别通过 caption-conditioned FiLM 与 spatial/event reader。每个位置仅在 channel 维做 LayerNorm。

caption context 来自同一冻结模型的 stock prompt prefill，在原 prompt 中按 tokenizer 字符 offset 精确映射真实 caption token，保留整个 token 序列。排除 instruction-only、生成 response 和 GT；边界合并的引号/标点 token 明示记录。两个 learned attention pooling 各自生成 query context；不预设它们自然学成 noun/action 分解，该语义需后验检验。

默认 hidden=128，P0 不开 dF/dt/pyramid。三架构使用同一 stem、同一语义接口；early_factorized 是局部算子因子化控制，不冒充 matched-stage pool-early E2。容量实际计数随登记保存（约2.5M），不是宣称参数完全相等；后续 E2 同层、同投影、同 probe 容量另登记。

Spatial reader 输出 M[B,T,H,W]；event reader 的 feature 用 logmeanexp-XY 池化后经独立 MLP 输出 a-logit[B,T]。a 直接对应 frame event presence，非 max occupancy。joint 可用 relu(sigmoid(a)-TopKPool(sigmoid(M)))²，首轮 P0 不训练它、不以 joint 为成功前提。

两路独立投影和非零 sigmoid(-6) gate（projection std=.001）：V_E=V+alpha_E W_E F_E，V_S=V+alpha_S W_S F_S，禁止平均。真实推理先官方 temporal_localization 在 V_E 上产生 reference/time，停止于时间段，不探测框；再新 KV cache 在 V_S 上产生自己的 reference，仅强制继承上一 pass 的 I*，用官方 parallel box probe 解码整段。现有通用空间入口仍计算一次随后被固定 I* 替换的 temporal probe，其成本计入。caption capture 额外一个 stock prefill，不能把总成本写成仅两次前向。

只对4个既有 metadata 选定 source-validation query/4 parents 做工程验收，不评分/读其GT；真实 first-step task gradient 另用按key排序的首个 eligible source-train query。无优化步、无TTA：

- zero gate 两路回到相同 stock PTD 输出；无效格式保留。
- event pass 不产生空间 box probe；固定区间必须一致。
- 一路 gate 干预不能改变另一路注入，event输出不能被 spatial gate 改变。
- 源 teacher-forced PTD CE 分路反向，第一步 reader/pooling/FiLM/out/gate 有限非零，另一分支专属参数梯度为零，冻结4B无梯度；shared stem允许两路梯度。
- 只mask CE labels，不改变 official PTD attention/position/context metadata。

输入、代码、权重、token mask、预测、梯度和实际耗时在 `artifacts/desta3d_v2/p0/<run_id>` 锁定/保存。失败后保留旧锁/输出，修正另run；不得热改 active pins。

## 源拟合与优化对照（待P0通过后锁定运行）

冻结 PTD，使用真实 source GT。先 evidence pretraining（M occupancy 有效标注支持，event frame presence），再 integration：event pass 的 reference/time CE + spatial pass 的 box CE，加 evidence 辅助项。source训练的 spatial pass 使用源 GT reference/time 的官方 teacher forcing；目标自由推理使用 event pass 的预测 I*，两者不同需明确。

比较 Frozen、early_factorized、shared3d-v2、dual3d-v2。新 recipe 使用 reader LR3e-5/head-out1e-4，5% warmup+cosine、accumulation4、clip1、有限5–8epoch及source validation选态；确切 A/B epoch和loss权重须在首轮前固定。保留一个旧recipe对照，避免把多项recipe共同变化归因单一trick。nonzero gate是P0共同语义条件；不把zero gate当作首步梯度修复。不是无边界网格。

先完成 source 媒体谱系与 parent/hash 去重，再决定混合HC+Vid清单。本地HC跨版本存在同名影片重叠，不能把HC2-train名义当独立source；若先沿原Vid95父源/618query、source-val31父源/198query做sanity，HC结论只可表述为cross-dataset transfer，不能称balanced mixed source。

用户明确要求：dual3d-v2 source-val 父源宏 vIoU 至少不低于匹配 Frozen，才进入目标64源TTA。CI、空间/时间、原好保持、>5pp损害及worst tail均保留；desired dual>shared>early 是待测假设。不得用目标GT选状态/筛源，也不把source supervised正值称目标TTA成功。

## 之后的无GT TTA

仅在源门通过后，先8父源工程与有限配对对照，再另锁64历史开发源。对比 conv/readers update 与 FiLM+全部channel-LN affine+gates校准，后者P0宽128实际66,818参数、stem/readers卷积冻结。query projection不是默认TTA参数。alignment在query conditioning后，候选权重.01，加入source-anchor/trust-region；确切loss及view需先登记。不能复用v1 feature moments冒充v2统计。

P1物理时间差分及dilation1/2/4代码可CPU检验，但单独开关和机制对照后再运行。禁止恢复 explicit temporal offset/dual timeline；VideoSwin/VGGT/OPD/32B/8B训练均非本轮。

## 资源及档案

累计GPU cap=null，v1结账15185.645719446977秒；v2新增装载、失败、重放和正常运行继续累计。P0有限1800秒复盘，8GiB磁盘保护；此工程阶段上限不等于研究路线停止。串行GPU、单Luna max只读监测，健康不重复排队。CURRENT与旧暂停队列保持。阶段结束独立回读、更新研究总档案并check/snapshot/check。

## 执行修订与实际状态（2026-09-27）

P0 v3已完成并独立回读：四源stock/zero/initial两pass均合法、zero输出精确等价、BF16实际非零，单source train的两路纯PTD任务梯度均可达并相互隔离。首次v2 scalar hash失败4.606188秒、v3成功23.711176秒全部累计，source-fit前合计15213.963083秒；没有把612条CPU响应检查算作612次梯度。

source-fit配置在 `artifacts/desta3d_v2/source_fit/CONFIG.json` 与 `LOCK.json` 实际锁定、fit001已启动。三个repaired架构用A1轮+B最多5轮；单个dual_current使用同v2结构/非零gate/初始权重/分路task mean目标，旧optimizer recipe跑6轮integration。两组同query与pixel曝光，不声称优化步数或PTD CE次数相同；repaired按accum4、current每query一步。此对照检验组合recipe（evidence预训、分组LR、warmup/cosine、accum），不能归因某一个trick；gated residual是四臂共同P0条件，不是本recipe对照的变量。

新recipe reader LR3e-5、head/out/gate1e-4，AdamW wd0/clip1，5%warmup+cosine，A→B新optimizer，B阶段不重置。A损失为L_ref+L_frame_event；B为两支各自均值PTD CE的和，加.1 L_ref与.1 L_frame_event。joint=0、identity loss未开、P1关闭。B阶段source-val父源宏vIoU选best，至少两轮B后仅在所有臂连续两B轮未改进时共同停止，否则max6总轮数；完整历史保留。

固定Vid95/618与31/198；target64 metadata父源和video SHA交集0，HC不混入。612条eligible response的官方time端点与event_active首末采样端点全部一致；另6条保留辅助训练。source validation先四臂792输出封存，再读本轮源标签。已有Frozen源验证缓存锁定同198个query，source指标定义沿旧源scorer，不能与目标corruption口径混算。

源拟合每个完整全臂累计窗口原子保存模型/optimizer/RNG/cursor，checkpoint携带最后窗口的history commit，可恢复幂等写入history_windows；JSONL仅辅助流。验证预测原子写入，恢复检查key/source/frame_ids/video SHA/adapter。阶段3600秒是保存与复盘间隔，持续预算cap=null。首16 query已CPU核验三个新臂各4步、current16步，参数确实变化/有限，A两gate固定−6；尚无源效用结果。
