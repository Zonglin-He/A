"""Document completed case diagnosis; no experiment or method selection."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round84'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=read(H/'CASE_DIAGNOSIS_SUMMARY.json');v=read(H/'VISUAL_GROUP_COUNTS.json');c=read(H/'ORACLE_PROXY_SUPERVISION_SUMMARY.json');native=read(H/'independent_native_audit_attempt001/AUDIT_RESULT.json');a=read(H/'independent_scalar_visual_audit_attempt001/AUDIT_RESULT.json');counts=read(R/'crossplatform/round84/capture33_attempt001/COUNTS.json');render=read(R/'crossplatform/round84/atlas_PRIVATE/COUNTS.json')
report='''# R84：先看原图，再定位双向融合与恢复监督的断点

本轮完成33条事后案例的原图核验和132页本地图册，没有新增算法、参数更新、损失、训练或任务GT评分。结论不是“统一由语言读出失败”，而是至少有三条不同路径：**源域准备后的表示/融合路径改变原有正确排序；点恢复的锚点保护留下目标候选内部的直接监督缺口；部分IoU变化只是同物体重复框切换。** 仍有遮挡与相邻实例无法独立确认的案例，保留UNKNOWN。

## 范围与证据边界

完整纳入R82冻结双向→正确绑定回溯结果的全部31条IoU变化（23改善、8恶化），额外纳入2条明确要求的“原步更新与回溯相反”案例。它们来自已暴露DEV336、VAL1280、TRAIN1024；这是按已知结果选出的机制诊断集合，**不是新的总体性能评估，也不据此重新筛评价分母**。没有读取FINAL。

沿用同一Waymo-only epoch100、R76已选源域模块、R80/R82保存状态、原始A256框和既有IoU表。重新执行的仅是已注册案例的冻结后缀与辅助头，用于捕获内部读取和空间误差；没有优化器。每条都保留source、frozen、raw、BT、raw_wrong、BT_wrong与oracle，保留所有正负及不变路径。Oracle只用于封存后的诊断，不进入构造监督、验收或更新。

本机每例四页：完整原句/完整RGB/局部裁剪/全局和局部BEV/排名及框重叠；原生query→seed读取；query→seed→局部RGB的一跳注意力路径；点/RGB/深度恢复误差空间变化。全部33条已目视核验完整RGB和裁剪，两条重点案例进一步核验支持图与误差图。其余误差图已经生成并完成数值审计，**不声称全部完成语义空间归因**。公开版本只含匿名标量与自有代码，图像、原句、坐标、原始标识、query索引、特征、注意力和权重留在本机。

## 1. 相反案例：同一个回溯准则不能辨别两种任务方向

下表IoU单位为原始0–1。CASE001/002是两个预先指定重点案例，不是本轮挑出的新方法赢家。

|案例|Source|冻结双向|大步raw|回溯BT|A256最高IoU|
|---|---:|---:|---:|---:|---:|
|CASE001|0.398376|0.379814|0|0.379814|0.420827|
|CASE002|0.737252|0|0.519559|0|0.737252|

CASE002的原图显示：source和raw的两个框覆盖同一个远处行人；冻结双向与BT选到近处植被区域，BEV深度也明显不同。这是真实区域选错，不是同一人体的细微框变动。source中好框已经存在并获第一名；源域准备后该框排名下降，raw重新提升另一条覆盖同一人的候选，BT又回到植被。错配绑定raw也选择同一行人框，因此**不能把纠错归因于正确跨模态绑定**。raw的点恢复误差在拟合遮挡与第三遮挡中均增加，而BT降低这些误差却没有恢复目标选择。

CASE001原图显示：source/frozen/BT框覆盖目标车辆，raw移动到右侧非目标区域。车辆三个近似候选高度重叠；raw错误区域与其在三维上分离。冻结时目标与错误区域的候选概率分别约0.03147与0.03068；raw后变为0.03073与0.03125，BT后为0.03110与0.03096。只是很小的候选相对分数变化就越过决策边界。错配BT仍选择零IoU候选，但与错配raw的具体query并非相同，不能写成完全同一输出路径。

CASE001的raw和BT均能降低整体恢复误差，CASE002的raw却提高点恢复误差。这解释了当前恢复准则为什么接受/缩小这些步，**不证明它能在测试时区分任务纠错与误改**。不能根据这两条事后GT结果制定部署阈值。

## 2. 2×2换头核验：当前证据支持表示/融合组，而非匹配副本单独致错

将原始/准备后特征组与原始/源域训练后匹配投影交叉，真实重算归一化、文字匹配与选框。原始后缀输出与原缓存隐藏表示误差为0。结果：

|干预|33条中Top1改变数|
|---|---:|
|原始特征上只换匹配头|0|
|准备后特征上只换匹配头|0|
|使用原始匹配头，换为准备后特征组|25|

CASE002两个匹配头配原始特征均选0.737252框，配准备后特征均选零IoU框；CASE001两个头配原始特征均0.398376，配准备后特征均0.379814。选中33条中source→准备后有16条IoU降低、8条提高，另9条不变。这个比例**不能当总体发生率**。

这里的“特征组”包含点/图像Adapter、双向交互、下游cross-encoder/decoder及文字表示变化、保留的late RGB残差。交叉组合可能偏离训练分布；未单独隔离CNN、各方向、可见性与late残差。因此可以下调“匹配头副本单独导致这些案例失败”的解释，**不能直接宣布RGB错绑或某个融合子层已经被因果定位**。

## 3. 找到具体实现条件：目标候选内的seed经常全部被保护

原生互补遮挡在生成点恢复目标后，将全部query锚点seed从点遮挡中移除，以保护固定候选锚点。用缓存最高IoU候选框作为事后几何代理，检查同一确定性遮挡：

|33条中的监督覆盖诊断|数量|
|---|---:|
|拟合遮挡没有框内点恢复监督seed|27|
|拟合及第三遮挡都没有框内点恢复监督seed|26|
|有可见框内seed，但全部也是受保护query锚点|18|
|拟合RGB投影代理区域无监督格子|0|
|投影代理区域无稀疏深度监督格子|0|

CASE001框内8个seed全部可见且全部受保护；CASE002框内9个同样全部受保护。两条拟合和第三遮挡中，**框内点恢复直接监督都是0**。CASE001仍有4个拟合RGB格子、11个深度格子；CASE002有8个拟合RGB格子、10个深度格子。

这是源于原生遮挡代码的可复核覆盖缺口，而非从注意力图猜测。独立CPU调用原生mask函数核验99个遮挡，确认保护与零监督。它说明整体点恢复下降并不意味着这两条目标候选内部的点特征获得直接恢复监督。**它不意味着目标完全没有梯度**：RGB/深度、框外上下文与共享参数仍有作用。

候选框不等于实例mask，投影矩形会混入邻物；“框外”也不等于背景。该统计由GT选出的oracle代理产生，只能诊断当前任务区域覆盖，不能成为部署时基于GT的采样器。取消锚点保护是否有用尚未测试；保护本身原本用于固定候选身份，不能未经控制直接删除。

## 4. 31条变化的身份初审：不是所有变化都在纠正主体

|IoU方向|同一选中物体/区域的框变化|明确不同区域→目标|UNKNOWN|合计|
|---|---:|---:|---:|---:|
|改善|9|5|9|23|
|恶化|4|0|4|8|

“同一选中物体/区域”包含仍停留错误参照/邻物的框变化，不等同正确定位。UNKNOWN包括遮挡、相邻车投影重叠、目标可见性不足及同一区域但目标身份不确定。8条恶化中有4条确认是同物体/区域的框切换，另4条无法认证实例身份；**不能说全部是实例选错，也不能说UNKNOWN没有实例误改**。

两条用户指出的小幅退化已查图：其中一条框对三维AABB IoU约0.83，覆盖同一车辆；另一条也覆盖同一车辆但框范围不同。后者原始source为0.241171、冻结双向为0.248436，不能把冻结分数误写为source。另一个显著改善的人体案例，从较短/偏移框换到更完整框，并没有换到另一个人。也有可确认的车辆/行人实例纠错，全部保留在标量表和本地图册。

33条中有26条A256最高IoU小于0.5，2条小于0.25。这只解释这些选中案例的固定框上限，不能解释CASE002那种已经有0.737框却掉到0的错误。

## 5. 空间误差与读取：能说什么，不能说什么

P/RGB/D误差图使用各自原损失系数0.1/0.1/0.01和原权重；拟合两遮挡平均、第三遮挡单列。同一信号的raw/BT和拟合/第三面板采用统一色标。五种保存状态都在相同真实对应下计算误差，错配训练状态不混成“错配测量”图；实际错配训练loss仍沿历史记录保留。第三遮挡和拟合遮挡可能覆盖同一恢复目标，不是独立新观测。

CASE002的点恢复变化主要发生在oracle框代理以外，框内因锚点保护没有该项直接监督。CASE001也是如此。完整图可见其他位置同时有误差增减，**尚不能把每个框外变化命名为路面/背景并作语义统计**。396行局部/全局误差变化都公开匿名标量，语义归属不清楚的保持未知。

原生query→seed及一跳RGB读取在两个重点案例中都较分散，不是沿目标边界的实例mask。不同候选的图看起来相近，不能据此认定某个像素导致选错；同理读取有响应不等于RGB提供了区分目标的新证据。当前不支持从图直接宣布“主体/参照角色混淆”是主因。

## 核验、成本与裁决

原生独立回放核验全部33条、5个准备后状态和原始桥接，1485个加权图检查；保存概率最大误差0、局部误差图最大误差0、原始隐藏桥误差0。独立审计核验132个JPEG可读、31+2样本关联、全部缓存IoU及正负方向、坐标回接最大误差小于1e-6、原生遮挡99次。没有新增GT几何评分或参数更新。

捕获198次冻结grounding后缀，用时8.09秒；独立原生核验6.46秒；CPU图册生成57.26秒。它们依赖已存前缀，**不是完整模型端到端速度**；不将人工看图和文件整理时间计入模型时间。

本轮不采纳新方法，不增加TTA损失，不改源域模型或已有R82/R83结论，自动化保持关闭。后续有依据优先检验的两项是：**保持候选身份前提下的恢复目标覆盖，及源域准备后整个融合/表示路径为何破坏既有排序**。两项需要在固定开发分母上作最小区分对照；本轮没有执行这些修改。角色化读出仍是候选解释，不因本次分析升级为主要原因。两个相反案例尚没有经验证的无GT选择规则。
'''
(H/'REPORT.md').write_text(report)
dump(H/'DECISION.json',{'state':'R84_COMPLETED_POSTHOC_VISUAL_CASE_DIAGNOSIS_NO_NEW_METHOD','adopt_new_TTA':False,'case_count':33,'all_existing_changed_cases_retained':31,'unknowns_retained':True,'matching_head_alone_not_supported_in_selected_cases':True,'representation_group_implicated_not_individual_RGB_causality':True,'anchor_protection_direct_point_supervision_gap_verified':True,'oracle_proxy_is_offline_only':True,'new_label_free_raw_vs_BT_selector_established':False,'new_source_training':False,'new_parameter_updates':0,'new_task_GT_scores':0,'FINAL_access':False,'new_algorithm_experiments_queued':False,'automations_remain_off':True})
dump(H/'PUBLIC_FIXED_CONFIGURATION.json',{'source_model':'Waymo_only_epoch100','source_checkpoint_sha256':'0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd','prepared_state_sha256':'84f3455b52c1ca9ded0dca394af17b9ea9949cafb720eb24728c505e79ed9f3e','A256_fixed':True,'posthoc_selected_diagnosis_only':True,'n':33,'changed_cases':31,'supplements':2,'new_parameter_updates':0,'FINAL_access':False,'attention_is_not_instance_mask':True,'oracle_proxy_never_used_for_TTA':True,'fit_views':2,'third_mask_not_independent_observation':True,'auxiliary_coefficients':{'point':.1,'RGB':.1,'depth':.01},'public_contains_no_media_caption_coordinates_or_original_identifiers':True})
dump(H/'NORMALIZED_COSTS.json',{'counts':{'cases':33,'changed_cases':31,'supplements':2,'JPEG_pages_local_only':132,'capture_grounding_suffixes':counts['grounding_suffixes'],'weighted_map_checks':native['weighted_map_checks'],'native_mask_checks':99,'new_GT_scores':0,'new_parameter_updates':0,'new_vision_expert_calls':0},'timing_seconds':{'capture':counts['seconds'],'independent_native_audit':native['seconds'],'CPU_atlas_render':render['seconds']},'peak_bytes':{'capture_GPU_allocated':counts['peak_GPU_bytes']},'flags':{'cached_prefix_not_end_to_end':True,'original_3EED_unchanged':True,'FINAL_access':False,'third_mask_is_not_independent_new_observation':True}})
dump(H/'NORMALIZED_HISTORY.json',{'events':[]})
dump(H/'PUBLIC_PROVENANCE.json',{'design_sha256':sha(H/'experiment/DESIGN_LOCK.json'),'case_registration_sha256':sha(H/'experiment/CASES33_PRIVATE.json'),'capture_seal_sha256':sha(R/'crossplatform/round84/capture33_attempt001/SEALED.json'),'atlas_seal_sha256':sha(R/'crossplatform/round84/atlas_PRIVATE/SEALED.json'),'manual_review_sha256':sha(H/'MANUAL_VISUAL_REVIEW_PRIVATE.json'),'native_audit_sha256':sha(H/'independent_native_audit_attempt001/AUDIT_RESULT.json'),'supervision_audit_sha256':sha(H/'independent_supervision_support_audit_attempt001/AUDIT_RESULT.json'),'scalar_visual_audit_sha256':sha(H/'independent_scalar_visual_audit_attempt001/AUDIT_RESULT.json'),'raw_private_material_not_exported':True})
print({'report':'research/round84/REPORT.md','new_methods':0})
