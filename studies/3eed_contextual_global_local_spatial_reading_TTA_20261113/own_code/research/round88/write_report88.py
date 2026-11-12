"""All registered outcomes, structure/update separation and bounded decision."""
from core88 import *
ARMS=ARMS88+['full_frozen3'];su=read(H88/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];vv=su['balanced_mIoU_percent'];d=read(H88/'DECISION.json');co=read(B88/'prediction336_attempt001/COUNTS.json');au=read(H88/'independent_combined_native_audit_attempt001/AUDIT_RESULT.json');tasks=read(H88/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');cov=read(H88/'EVIDENCE_PREPARATION_ROWS.json');updates=read(H88/'ALL_UPDATE_ROWS.json');held=read(H88/'HELDOUT_OBSERVATION_ROWS.json')
text='''# R88：上下文角色引导的全局—局部空间读取与TTA

本轮保留R76已准备双向RGB–LiDAR模型、Waymo-only epoch100原始3EED和固定A256框。没有新视觉专家、源域训练或FINAL访问。使用历史暴露DEV336（Drone/Quad各168、五个历史场景），不是新独立测试。预先固定一步主方法，三步只作强度对照，不按事后最高分替换主方法。

## 依据与实现条件

'''+(H88/'LITERATURE.md').read_text()+'''

## 与R87不同的实际计算

1. 复用完整句编码，不重新把主体短语送入较弱source作全局教师。沿用R87修复的实体清单，从完整句offset提取实体及依存属性token；结构化外观字段仍属于主体。原句所有token保留在原encoder/decoder路径中。继承的解析仍有未知与漏检，不代表真实身份金标。
2. 用冻结R76最后cross-encoder的原生cross_lv注意力提出每角色最多三个不同支持区域。区域取自原A256与实际1024seed支持，不新增物体检测或实例mask。保留各假设的点/像素支持、均值与方差，统计位置不被合并成一个虚构物体中心。缺少支持26条同样留在总体分母。
3. 在cross_encoder.layers[2].cross_layer.cross_lv内部，原有Q/K/V对每假设做加性软空间偏置读取，保留50%原全局输出与50%局部混合。局部最低支持0.05，不把其他区域硬删除。使用原训练投影，无新随机语义头。不同属性token使用完整句上下文，但仍依赖原模型能否理解该属性。每个局部假设分别计算读取向量用于对应监督，实际文本残差按假设质量混合；没有新增经过训练的候选条件文本解码器。
4. 明确图像参考系的左右关系在实体假设对之间执行一次对称软证据交换，改变下一次局部读取的混合质量；没有训练新的关系网络。可见点分位范围用于不确定关系约束，不当作完整物体边界。未知参考系/不可测关系不作强判断。这个计算可能仍绑定邻物，不称为正确监督。
5. decoder[5].cross_v实际接收候选局部支持、全角色上下文与主体多假设的软注意力偏置，最低保留0.25全局读取。修改后的文本表示与视觉读取都进入真实最终匹配，仍选择原A256。不是导出attention热图后再拟合旧候选分布。no_local撤回两处局部读取，零更新逐值恢复原冻结R76。
6. 更新前固定区域、关系、所有像素/点对应及两组遮挡。局部物理恢复沿用已训练R76头；对应目标是冻结完整输入的同区域语言条件读取向量，学生在两种遮挡下恢复它。它提供对应稳定性，不提供目标身份真值。部分候选比较只使用可执行关系与可见支持，覆盖9条输入/13对；参照假设仍可错，不升级为唯一正标签或完整256维教师。
7. 每输入从同一R76状态恢复，只更新原102498个Adapter/双向交互投影与门控。原PointNet++、RoBERTa、CNN、decoder、matching及辅助头冻结。目标为既有局部物理恢复＋0.1区域对应＋0.01局部偏好＋原输出/表示保持；SGD0.1、梯度范数上限1，固定最多九档Armijo回溯，无GT步长选择。完整分布仅用于保持原行为，不作为新增绝对伪标签。

## 公平对照

full_direct：新结构零更新；full_frozen3：同一新结构下完整输入＋本次TTA的两个遮挡输入，固定等权直接平均；full_tta1：相同信息真实参数更新。full_tta3只作次要强度对照。三读出控制与TTA使用相同的三个输入版本，TTA额外承担反向和回溯中的重复计算，二者不是相同运行成本。no_relation只撤回关系对局部读取的反馈；no_local撤回局部空间读取；wrong_space置换点与读取支持的对应；role_swap置换实体角色的局部支持。两个错配臂均完整报告，不选最差者作为唯一对照。错配主要针对读取路径，真实物理恢复与保留的少量关系偏好不被同时打乱，因此不等同于完全错误监督。

physical_only1撤回新增对应/偏好，保留相同新结构与保持项；no_rank1、no_correspondence1分别检验新增监督。历史ABC、R59和R87结果作为开发参照，跨轮遮挡/机制预算不同，不能替代本轮同信息对照。

## 全DEV336结果

|方法|Drone mIoU %|Quad mIoU %|平台平均 %|平均Acc@25 %|平均Acc@50 %|
|---|---:|---:|---:|---:|---:|
'''
for arm in ARMS:text+=f"|{arm}|{su['drone'][arm]['mIoU_percent']:.6f}|{su['quad'][arm]['mIoU_percent']:.6f}|{vv[arm]:.6f}|{np.mean([su[p][arm]['Acc25_percent'] for p in ['drone','quad']]):.6f}|{np.mean([su[p][arm]['Acc50_percent'] for p in ['drone','quad']]):.6f}|\n"
text+='\n固定主臂增量：'+', '.join(k+f' {v:+.6f} pp' for k,v in d['primary_gains_pp'].items())+'。\n'
text+=f"\n结构本身 full_direct−冻结R76：{vv['full_direct']-vv['frozen_bi']:+.6f} pp；更新 full_tta1−full_direct：{vv['full_tta1']-vv['full_direct']:+.6f} pp；更新相对同观测三读出：{vv['full_tta1']-vv['full_frozen3']:+.6f} pp。三者不能混作同一种收益。\n"
text+='\n## 组件作用\n\n以下为full减对应消融，正值才表示该组件在该组合有净帮助；不可相加成独立贡献。\n\n|移除/替换组件或对照|读出|净作用 pp|full更好/更坏条数|\n|---|---|---:|---|\n'
for r in read(H88/'COMPONENT_EFFECT_ROWS.json'):text+=f"|{r['component_removed']}|{r['readout']}|{r['component_net_pp']:+.6f}|{r['full_better_n']}/{r['full_worse_n']}|\n"
text+='\n## 全体正负变化与固定旧案例\n\n|参照|主方法IoU改善|恶化|不变|query改变|\n|---|---:|---:|---:|---:|\n'
for ref in ['frozen_bi','full_direct','full_frozen3','R59']:
 delta=np.array([t['IoU']['full_tta1']-t['IoU'][ref] for t in tasks]);text+=f"|{ref}|{int((delta>1e-10).sum())}|{int((delta< -1e-10).sum())}|{int((abs(delta)<=1e-10).sum())}|{sum(t['queries']['full_tta1']!=t['queries'][ref] for t in tasks)}|\n"
text+='\n|全局选取案例|平台|原因|冻结双向|新结构直接|同观测三平均|TTA一步|\n|---|---|---|---:|---:|---:|---:|\n'
for c in read(H88/'GOOD_BAD_CASE_ROWS.json'):text+='|'+c['case']+'|'+c['platform']+'|'+c['reasons']+'|'+'|'.join(f"{c[a+'_IoU']:.6f}" for a in ['frozen_bi','full_direct','full_frozen3','full_tta1'])+'|\n'
text+='\n|R87固定案例|冻结双向|新结构直接|TTA一步|\n|---|---:|---:|---:|\n'
for c in read(H88/'FIXED_R87_CASE_ROWS.json'):text+='|'+c['case']+'|'+'|'.join(f"{c[a+'_IoU']:.6f}" for a in ['frozen_bi','full_direct','full_tta1'])+'|\n'
if (H88/'CASE_REVIEW.md').exists():text+='\n'+(H88/'CASE_REVIEW.md').read_text()+'\n'
delta=np.array([t['IoU']['full_tta1']-t['IoU']['frozen_bi'] for t in tasks]);text+=f"\n最大单正例对均值贡献{delta.max()*100/336:+.6f} pp，总增量{delta.mean()*100:+.6f} pp，其余335条仍以336为分母合计{(delta.sum()-delta.max())*100/336:+.6f} pp。这只是尾部贡献诊断，不从总体分母剔除该案例。\n"
pref=read(H88/'PARTIAL_PREFERENCE_AND_SUPPORT_ROWS.json');text+=f"\n## 支持、监督覆盖与第三遮挡\n\n全体角色{sum(c['roles'] for c in cov)}个、路由token{sum(c['routed_tokens'] for c in cov)}个、多空间假设{sum(c['hypotheses'] for c in cov)}个、可执行关系{sum(c['executable_relations'] for c in cov)}条；未知支持{sum(c['unknown_support'] for c in cov)}条输入。部分关系偏好{sum(c['partial_pairs'] for c in cov)}对、覆盖{sum(c['partial_pairs']>0 for c in cov)}条输入。\n"
text+=f"\n封存后按候选IoU诊断部分比较：更高{sum(c['pairs_higher_IoU'] for c in pref)}、更低{sum(c['pairs_lower_IoU'] for c in pref)}、相等{sum(c['pairs_equal_IoU'] for c in pref)}。IoU排序不是实例身份金标；不能据此按GT筛选训练偏好。\n"
hh=[x for x in held if x['third_mask_representation_error_before'] is not None];text+=f"\n第三组遮挡未参与更新、验收或选参；有可比较读取的{len(hh)}条，同区域表示误差均值从{np.mean([x['third_mask_representation_error_before'] for x in hh]):.8f}到{np.mean([x['third_mask_representation_error_after'] for x in hh]):.8f}。该误差衡量当前区域表示稳定性，不能替代grounding。\n"
cv=read(H88/'GLOBAL_SUPPORT_COVERAGE_ROWS.json');text+=f"\n空间支持资格诊断（非身份纯度）：全部角色的全局注意力落入任一A256支持的质量中位数{np.median([x['global_attention_mass_in_any_A256'] for x in cv]):.6f}，落入三个假设的质量中位数{np.median([x['global_attention_mass_in_three_hypotheses'] for x in cv]):.6f}。本版的显式假设仍由A256支持提出；候选之外的参照物只保留在全局路径中，没有新建候选外实例发现能力。\n"
text+='\n## 场景与证据边界\n\n'+'''全部样本和失败/未知/零更新保留；场景表单独公开。DEV此前已多轮曝光，五个场景没有增加，不将帧看作独立场景或给出逐帧独立置信区间。本轮不使用GT跨度/实例标签来生成路由、偏好、更新或步长验收。新任务GT几何评分为0，只在预测封存、独立原生/损失审计后关联既有A256 IoU。病例选择和第三遮挡诊断不能反过来修改本轮方法。

局部空间偏置并非源域训练过的完整EG-3DVG能力。继承的实体解析、A256/seed支持、注意力和稀疏可见点都可能绑定错误对象；没有新增观测分辨率或RGB语义专家。一次固定倍率配置的阴性结果只能约束当前迁移实现，不能否定经过源域训练的全局—局部结构。
'''
text+=f"\n## 审计与成本\n\n21臂原生产耗时{co['seconds']:.3f}s，原生suffix调用{co['main_calls']}，目标计算{co['objective_calls']}，实际接受{co['accepted_updates']}次更新，峰值已分配GPU显存{co['peak_GPU_bytes']} bytes。额外相同三视图零更新控制1008次suffix调用。独立检查{au['saved_endpoint_replays']}个端点、{au['actual_loss_scalar_checks']}个实际损失标量，读出误差上限{au['max_probability_or_score_error']:.3g}，路由重建误差{au['max_route_error']:.3g}，损失误差{au['max_actual_loss_error']:.3g}。这些是冻结前缀重放成本，不是端到端系统延迟；图册原媒体/描述/坐标仅本地PRIVATE。\n"
text+='\n## 采用判断\n\n'+('主方法超过本轮同信息直接、三读出和冻结R76，但仍只是在已曝光DEV上的开发候选，不宣称稳健独立验证或全部机制成立。' if d['passes_matched_increment_checks'] else '主方法没有同时超过本轮同信息直接、三读出和冻结R76，不采用这套完整TTA配方。分别保留有总体净作用的读取组件为开发线索，不以辅助误差下降代替任务收益。')+' 不追加倍率/损失/步数搜索，不自动开始源域训练，FINAL继续封存，无后续实验排队。\n'
(H88/'REPORT.md').write_text(text);prep=read(B88/'evidence336_attempt001/COUNTS.json');zc=read(B88/'frozen_routed3_control336_attempt001/COUNTS.json');dump(H88/'PUBLIC_FIXED_CONFIGURATION.json',{'design':read(H88/'experiment/DESIGN_LOCK.json'),'same_budget_control':read(H88/'experiment/ZERO_CONTROL_ADDENDUM_LOCK.json'),'historical_status_provenance':read(H88/'experiment/METADATA_STATUS_ADDENDUM_LOCK.json')});dump(H88/'NORMALIZED_COSTS.json',{'counts':{'main':336,'arms':22,'TTA_parameters':co['TTA_parameters'],'accepted_updates':co['accepted_updates'],'grounding_suffix_calls':co['main_calls'],'zero_control_suffix_calls':1008,'independent_endpoint_replays':au['saved_endpoint_replays'],'new_GT_scores':0},'timing_seconds':{'preparation':prep['seconds'],'producer':co['seconds'],'zero_control':zc['seconds'],'independent_audit':au['seconds']+au['extra_independent_control_seconds']},'peak_bytes':{'producer_GPU_allocated':co['peak_GPU_bytes']},'flags':{'cached_prefix_not_end_to_end':True,'FINAL_access':False,'new_external_vision_models':False}});dump(H88/'NORMALIZED_HISTORY.json',{'events':[{'stage':'native_route_engineering','attempt':1,'status':'completed','before_GT':True,'prior_attempt_preserved':True},{'stage':'Armijo_derivative','attempt':1,'status':'completed','repair_type':'raw_gradient_dot_clipped_direction','before_GT':True,'prior_attempt_preserved':True},{'stage':'same_budget_control','attempt':1,'status':'completed','before_GT':True,'producer_predictions_changed':False},{'stage':'historical_status_provenance','attempt':1,'status':'completed','before_GT':True,'producer_predictions_changed':False},{'stage':'complete_prototype','attempt':1,'status':'completed','producer_predictions_changed':False}]});print(d)
