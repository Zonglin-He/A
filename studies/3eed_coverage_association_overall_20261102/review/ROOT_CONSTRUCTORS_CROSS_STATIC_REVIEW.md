# R50 两个root构造器独立交叉静态审查

仅阅读构造器、当前已审worker/auditor/scorer源码及既有GT前锁／合同；未调用构造器函数、未读取future stage数据、未运行GPU、审计、评分或修改任何运行代码／既有锁。

## 结论

Full postseal构造器 `7b3bf9fff0502210dd861217b0e3ffaf73d09104c324fd9e1ff6bcceb9ebff0d` 与现有四种消费者的字段匹配。

Episodic原构造器 `f887c7b289bedd7f37a2112dc369a5cad6ddf1d6dd7f6a467f3ebae0583b2ae2` 有一处必要provenance补充：CODE列表第54–63行遗漏实际IoU实现 `tools/obb_iou.py`，虽然包含导入它的 `independent_core_v2.py`。该问题已报root与作者，作者仅按root裁定新增这一项pin，旧稿保存在 `preexecution_episodic_builder_revision001`。修订构造器 `98a25349496d83c729fa32ba69af1f7b3f19fc5f0bbbcfb119682e71306efb4a` 已静态复核；不涉及worker、方法、阈值、预算或既有锁变化。

除上述已修项，没有发现其他会因字段缺失、固定路径或授权名称错配而阻塞当前预定流程的问题。该结论不是未来实际产物的审计PASS或执行授权。

## Full postseal四模式

|模式|核对项目|结果|
|---|---|---|
|prediction|生成 `FULL_PREDICTION_PLAN_LOCK` 与 `FULL_PREDICTION_AUTHORIZATION`；plan包含manifest/runroot/output/evidence_case_index、三臂、program/input pins和sealed_roots；grant.state与worker相同|匹配。聚合state和sealed_chunks/CASE_INDEX字段也与full_evidence_batch50最终输出一致。|
|cost|stage配置包含evidence_plan_file、worklist_root及seal、chunk_seals、recipe和assets；expected pool9459/44093与wholecase224/857；grant使用audit_plan_sha256|匹配独立cost auditor。两阶段各24个既定chunk，此处是冻结当前方案的约束，不是通用任意chunk-size构造器。|
|arithmetic|生成producer_prediction_plan及program/input pins；grant使用independent_plan_sha256和prediction_seal_sha256|匹配独立arithmetic auditor。producer output是predicted5912_attempt001；原R8 n_samples和新预测n字段未混用。|
|scoring|先验证arithmetic/cost各自PASS、程序及计划摘要；绑定prediction、两种独立结果和原DEV评分；GT索引/receipt在此模式才pin；grant使用scoring_plan_sha256、prediction_seal和independent_prediction_audit_sha256|匹配full scorer。cost独立结果由构造器前置检查；worker的直接gate以prediction arithmetic结果为主，不能绕过构造器另造无cost的授权。|

补充核对：full arithmetic auditor当前只输出 `ANONYMOUS_ROWS.json` 与 `AUDIT_RESULT.json`，**没有独立SEALED**；postseal scoring构造器没有错误地索取它。cost auditor输出SEALED，构造器确实读取其seal。未来full saved-scalar审计另有SEALED，不与上述两者混用。

构造器的四个mode分别需要rootconstruction授权；它只写计划与grant，不调用worker。完整输入的actualsha和case成员最终仍由worker/auditor核验；构造器对index结构的检查不是替代数值审计。

## Episodic七阶段与全零边界

|阶段|消费者关键字段／产物|静态匹配|
|---|---|---|
|engineering_capture|mode=engineering2、n2、rows_path、engineering_selection_salt、readout/boxes、B_sampling_salt、source bound|hash2选样与capture相同；4个source forward上限。|
|engineering_fit|capture_root+seal、arm_names、steps1/3、lr/wd/tau、n2与初始bridge bound|匹配fit。engineering_bridge索取的group_A_G_literal_query_and_box_exact确为当前fit记录字段。|
|dev_capture|仅engineering source/G/consensus桥接完成后，n336与双采样、672forward|匹配；工程不读任务GT，不以其任务好坏决定样本。|
|dev_fit|capture seal、原六臂、逐样本源MLP重置、最多1008 optimizer steps|匹配；实际steps仍由接受反馈数3E记录，拒绝不消失。|
|dev_closure|n_per_platform1、steps1/3、selection_salt、bound2e-5、max8、capture/fit seals|匹配；plan的n336是输入队列，closure artifact的n是实际选中episode数0–2，构造器未错误要求closure.n=336。|
|independent_audit|固定 `EPISODIC_INDEPENDENT_AUDIT_PLAN_LOCK` 与 `ROOT_INDEPENDENT_EPISODIC_AUDIT_AUTHORIZATION`；capture/fit/closure roots/plans、runtime versions、numeric bounds、no-GT flag|匹配独立auditor固定入口和输出目录；命令不追加该脚本不接受的CLI参数。|
|single_score|prediction_root、independent_audit_result、arms6/task_values2016、固定step选择规则；grant的prediction_seal_sha256和independent_audit_sha256|匹配scorer。新增obb_iou pin使评估依赖链完整；P5 index还由PREPARATION_RECEIPT内摘要在消费者中验证。|

全零closure的实际worker只写 `SELECTED_BEFORE_FORWARD_PRIVATE.json` 和 `SEALED.json`，state为 `R50_EPISODIC_CLOSURE_NOT_AVAILABLE_NO_NONZERO_ACCEPTED_STATES`、n=0、grounder_forwards=0；不产生 `RESULT_PRIVATE.json`，也不包含plan_sha256。构造器独立审计阶段仅要求closure seal存在，独立auditor在无候选分支不索取RESULT；closure plan仍作为显式input pin绑定。该缺省形式没有造成字段错误，也没有被当作真实adapted closure。

非零参数分支硬要求 `installed_closure_requirement_satisfied=true`。全零分支硬要求参数元素不同数0、全部source字节相同、1/3原生输出与最终共识等于0步，以及 `installed_closure_state=NOT_APPLICABLE_VERIFIED_NO_PARAMETER_UPDATE`、requirement_satisfied=false；计划开关预设true才允许纯诊断score。scorer还把TTA_NOT_ACTUALLY_TESTED和强制STOP记录下来，不将它当适应PASS。

独立审计SEALED包含AUDIT_RESULT与匿名数值／更新行；构造器通过member读取AUDIT_RESULT符合该schema。正常closure选样至多2条、每条4个完整source forward的证据也由独立auditor核查，不能用旧历史head closure替代。

## 授权顺序和剩余执行边界

Episodic构造器先验证full score seal、独立saved-scalar seal/PASS和GT前两参考门，确认原NumPy及独立fmean都通过才读取old336选工程样本。默认只写AUTH_DRAFT；真正执行grant需要root显式允许当次stage。grant-existing-plan还核原plan摘要和全部pins，不覆盖计划。生成的command只是receipt字符串，不被构造器执行。

所有执行仍依赖真实stage seal、root grant和manual资源guard。静态兼容不提前证明source桥接、非零更新、installed closure或任务总体收益。未来若发生技术失败，保留原attempt并停止对应阶段，不能基于任务GT改field、阈值、输出代表或评分规则。
