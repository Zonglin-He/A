# 最新GitHub方法迁移TubeDETR：固定128开发后完整双方向

2026-10-22用户明确授权读取GitHub最新方法、修改TubeDETR、按此前“先分析、有限顺序调参、锁定后全量”的方式重跑。最新main是5bd213f7bdb4241d4d759ab4fb8b20f8f7117b57，其维护方法为完整原生框BoxFixed＋Global6＋原物理MBR，不是旧OPD。旧暂停数值/空间队列不恢复，新执行独立于全部旧输出；不触碰其他TA-STVG checkout/进程。新代码在tubedetr分支的tools/boxfixed_tube，不把广泛研究树覆盖回干净main。

## 移植与资格

上游_spatial_math/evidence/spatial逐文件保留，仅import命名空间变为本地相对路径；记录upstream_port.json。完整教师框经双向轴仿射拟合传播；不阈值mask成分量框、不裁剪完整框、不加入NMS，不使用本地失败v0传播头。原ERef共识校准、每域初始gamma和八处观察规则保留。

TubeDETR一次native224真实前向读取六层原时间头输出，六个完整网格端点后验等权平均，再走原物理MBR。TubeDETR本身输出完整物理网格，不伪造TA的两个offset，不增加448分支。空间证据使用其真实decoder img_memory的256维视觉前缀，按真实pos/valid mask核对网格、排除文本槽；不替换成旧1024维ResNet特征。接口与原两域Source/Native输出逐位检查，原八位置及新取的原生ReferDINO框也逐位核对旧封存证据。

每query仅一个ReferDINO、一次八帧原生调用，选中query的原始mask logits按上游bilinear到教师图像尺寸→sigmoid→area到学生网格处理。学生/教师权重全冻结、无优化器/反向/临时学习状态。专家在独立resident进程中读取同一权重；只有新代码/权重/实际RGB/文本内容绑定的精确缓存可复用。旧概率mask不逆logit冒充原生logits。GPU遵守共享gpu.lock；任何旧任务不kill、不重启。

## 数据、曝光与有限调参

Vid沿用已登记seed41输入哈希128query/128父视频清单c7df9b65…ec4；HC沿用已登记去数字切片前缀的128query/128父来源4cf05614…1664，不重新抽样。完整总体仍Vid10303/732、HC1902/227；HC注释SHA61f7a2a74d39c13dd380540d4ac819edb7abcde1826d44229e539d128bf3260c。已曝光开发和全量重叠公开，不称独立测试。

新方法没有OPD，因此不复活旧lr/tau/steps。保持特征对应温度.1/topK20、WLS可靠性/完整范围规则、Global6温度1及原物理MBR固定。按原调参思路只试以下四个数值因素，两域独立：

1. gamma=0/.0125/.025/.05/.1/.25/.5/.75/1。先隔离共识初始化强度。新方法gamma0仍有BoxFixed传播，是完整算法的零初始化修正控制，不再等同旧Temporal-only。
2. 可靠性准入=.05/.1/.2/.4；检查有效传播覆盖与完整128负例。
3. 几何保护权重=.025/.05/.1/.2/.4；检查替代ERef的风险和尾损。
4. 连接权重=0/.125/.25/.5；零为公开无连接控制，检查跨帧项实际贡献。
5. 所选其他参数下只复查相邻gamma最多2个。

默认HCgamma.5/Vidgamma1、准入.1/保护.1/连接.25一直在候选集合。每阶段在固定128上选择Full等query m_vIoU最大值，严格并列保留默认配置；同时公开dense/观察/未观察空间指标、相对Temporal增量、>5pp损伤数、准入与改动数量。每阶段decisions写事前理由、全部候选、实际差值/空间/尾损和下一阶段理由，禁止每query选参/GT门控。最多24不同profile/域；每query仅捕获一次GPU证据供所有CPU参数profile复用。gamma缓存改动与完整几何重建逐位回归，上游默认DP输出/分数逐位回归。

每个profile全部128预测seal后独立CPU读标签。两域都生成selected_profile锁及预测绑定后，才依次执行Vid10303和HC1902，已选128预测内容绑定精确复用，其余真实执行新方法。全量结果不回流选参。任何工程失败明确记录并停止，不用旧fingerprint续跑；只能按显式相同plan、输入、payload哈希继续当前未完成结果。

## 结果与执行

Source=原生空间/Native时间；Spatial=BoxFixed/Native；Temporal=Source/Global6；Full=BoxFixed/Global6。没有OPD，不沿用WithoutOPD名字。原消融完整保留。完整seal后CPU统计query均值、父来源均值、父聚类区间、空间补充与负例；首3条还核对上游官方指标函数，绝不评分部分总体作完整主表。缓存成本与新端到端成本分开。

入口：`.conda/tubedetr/bin/python -B -u tools/tubedetr_boxfixed_development.py`。计划和接口检查在results/tubedetr_boxfixed_development；当前状态/每query进度/失败/证据在outputs/tubedetr_boxfixed_development；运行身份在runtime/tubedetr_active_jobs.json的boxfixed_controller/worker。新全量路径位于outputs/tubedetr_boxfixed_development/{vidstg,hc2}/full，旧outputs/tubedetr不改。

这是用户明确授权的新一轮完整实验，取代此前停止扩大v0的限制；不把本次新算法当旧OPD改进，也不声明当前128调参或全量已经完成。

## 工程隔离恢复v2

v1计划校验在安装推理guard之前打开标签文件计算哈希，未解析坐标或使用标注，但不符合严格不访问标签文件的约束。已保存旧代码、计划、128证据及部分profile至outputs/tubedetr_boxfixed_pre_gt_hash_guard_fix，停止仅本流程的进程。新checked仅检查输入manifest和代码；GPU/profile在校验前装guard，标签hash与解析移到完整seal后的CPUscore。新增单测显式用不可访问的标签路径证明推理校验不打开它，86检查通过。v2全新预测输出重做128，不绕过指纹或续用旧预测；仅教师实现/权重/实际RGB/文本相同、context及payload哈希完全匹配的原生教师缓存复用。数值方法、网格与选择目标不变，不用v1的profile成绩选参。详细回执见engineering_recovery.json。

## 双方向完整终态（2026-10-22）

两域128各21配置均完成锁定，Vid10303/732、HC1902/227均已完整预测封存和独立CPU评分，运行控制器正常终态。详见[完整汇总](../results/tubedetr_boxfixed_development/terminal_report.md)、[逐项结果和父聚类区间](../results/tubedetr_boxfixed_development/terminal_summary.json)、[调用计数与成本口径](../results/tubedetr_boxfixed_development/execution_audit.json)。当前v2 plan/code与原seal保持不可变；fullmetrics覆盖/改动两项仅128复用子集，分别用metric_scope_audit澄清，定位指标不受影响。完整结果不回流调参，旧暂停队列不恢复。最终发布与远端字节核验后结束半小时监控。
