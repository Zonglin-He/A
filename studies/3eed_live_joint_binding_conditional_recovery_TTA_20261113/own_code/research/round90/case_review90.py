"""Record visual observations after sealed evaluation, without changing any method."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[2];H=R/'research/round90';read=lambda p:json.loads(p.read_text());cases=read(H/'experiment/GOOD_BAD_CASES_PRIVATE.json')+read(H/'experiment/FIXED_R89_CASES_PRIVATE.json');tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');sup={r['row']:r for r in read(H/'SUPPORT_ROLE_ROWS.json')}
notes={
 'CASE001':('wrong_region_to_target_region_static_gain','RGB中从远处无关区域移到左侧车辆范围；本条roles=0，来自query后备路径。TTA没有进一步改变IoU。','high_visual_region_only'),
 'CASE002':('likely_same_person_duplicate_boxes','完整场景和放大图中两框都围绕同一个近景行人；TTA恢复R76原query，使IoU .4461→.3280。不能称作换了目标身份。','medium_no_independent_instance_annotation'),
 'CASE003':('static_wrong_person_region','R76框在近景运动者周围，联合零步/TTA移到图像最右后方另一行人区域。错误先发生在零步；只训头/继续匹配分支保留R76范围。','high_visual_region_only'),
 'CASE004':('static_neighbor_vehicle_region','原R76围绕浅色车辆，联合零步/TTA转到相邻深色SUV范围。注意力/支持图只显示代理读取，不能证明错误由某个参照槽单独导致。','high_visual_region_only'),
 'CASE005':('same_vehicle_box_switch','两框都围绕左缘同一车辆；TTA选择恢复R76原query，.4139→.5231，直接表示融合也恢复。不是新实例身份纠错。','medium_no_independent_instance_annotation'),
 'CASE006':('likely_same_vehicle_box_switch','远处车范围接近，投影/裁剪可见两候选在同一车辆附近；TTA精确恢复R76 query，.1879→.2035。远距离遮挡使精确3D身份仍仅作视觉代理。','medium_no_independent_instance_annotation'),
 'CASE007':('no_adaptation_change_mean_harms','零步与TTA仍选同一远处行人候选；直接三表示融合切到附近近似框，IoU .5816→.3079。不能将保持原结果归因于本次更新纠错。','medium_no_independent_instance_annotation'),
 'CASE008':('no_adaptation_change_mean_harms','概览与逐例记录：零步/TTA .6073、融合 .4728。画面与CASE007近似，主体较小；未单独放大复核该页，物体身份维持UNKNOWN，不将IoU差直接标作身份错。','UNKNOWN_identity_overview_only'),
 'CASE009':('missed_foreground_target_mean_recovers','放大图中零步/TTA框在后方行人区域，直接融合框转到近景主体范围，IoU0→.5738。存在可利用候选，但恢复更新没有实现这次排序改变。','high_visual_region_only'),
 'CASE010':('missed_other_person_mean_recovers','零步/TTA保持右侧人/树附近，直接融合转到左侧人附近，IoU0→.4668。远处外观细节不足以单靠图验证完整文字身份；结合封存IoU说明融合有收益，不将关系解析成功当根因证据。','medium_identity_not_independently_certified'),
 'CASE011':('stable_high_score','原始、冻结、TTA和直接融合均保持同一高IoU候选 .8127；它是稳定正例，不是适应增益证据。','high_visual_region_only'),
 'CASE012':('subject_reference_type_binding_unresolved','文字主体是骑车人，最终框持续落在右侧车辆范围；离线oracle在左侧人车范围，原A256有可恢复框。多个分支仍错，不能只归于TTA步长。','high_visual_region_only'),
 'R89_CASE002':('prior_case_no_new_gain','固定回访全部本轮主要读出 .2180；旧轮的静态改善没有保留。','UNKNOWN_not_reinspected_full_page'),
 'R89_CASE003':('prior_static_harm_absent_now','固定回访本轮 .3436；旧轮静态误改不在当前结果中，不能据此认定新绑定成立。','UNKNOWN_not_reinspected_full_page'),
 'R89_CASE005':('prior_adaptation_gain_now_only_direct','完整页看到当前零步/TTA仍落在目标后方别车，直接融合回到近景车范围 .4455；旧轮唯一适应正例没有在完整联合方法重现。','high_visual_region_only'),
 'R89_CASE008':('duplicate_of_CASE009','同一输入与CASE009重复回访，不算新增独立证据。','same_observation_as_new_case'),
 'R89_CASE011':('duplicate_of_CASE012','同一输入与CASE012重复回访，不算新增独立证据。','same_observation_as_new_case')}
lines=['# R90 PRIVATE — postseal case review','', '17图页＝12个本轮固定选择页＋5个R89固定回访；其中2个回访与本轮页重叠，共15个唯一观测。先查看12页概览，再放大11个本轮页和R89_CASE005；CASE008只有概览，R89_CASE002/003只复核封存数值，身份保留UNKNOWN。非独立人工标注，不将视觉判断当训练/门控标签，不据个例修改方法。完整RGB、原句、投影框、全局注意力、实际更新后候选支持和BEV都在图页；注意力及支持均非实例mask。','']
public=[]
for c in cases:
 t=c['task'];typ,note,confidence=notes[c['case']];v=sup[t['row']];lines+=['## '+c['case'], '', c['row']['caption'], '',note,'', 'Review: '+confidence+'; role_count='+str(v['roles']), '', '| state | IoU | selected query PRIVATE |','|---|---:|---:|']
 for a in ['source','frozen_bi','candidate_joint_frozen','candidate_joint_tta1','candidate_joint_tta3','candidate_joint_feature3','candidate_joint_physical1','candidate_joint_shuffle1']:lines.append(f'| {a} | {t["IoU"][a]:.9f} | {t["queries"][a]} |')
 lines+=['', '[完整私有图页](WORKSPACE_ROOT/crossplatform/round90/atlas_PRIVATE/'+c['case']+'.jpg)', '']
 public.append({'case':c['case'],'platform':t['platform'],'postseal_visual_interpretation':typ,'visual_confidence':confidence,'spatial_role_count':v['roles'],'same_R76_and_TTA_query':t['queries']['frozen_bi']==t['queries']['candidate_joint_tta1'],'used_for_method_or_update_selection':False,'identity_is_independent_GT':False})
(H/'CASE_REVIEW.md').write_text('\n'.join(lines));(H/'CASE_REVIEW_ANONYMOUS_ROWS.json').write_text(json.dumps(public,indent=2)+'\n')
with (H/'REPORT.md').open('a') as f:f.write('''
## 原图复核后的具体归因

17页图册包含15个唯一观测，两个旧页回访与新页重合，不能当独立样本累加。12新页概览后，放大11新页和1旧页；其余页的身份保留UNKNOWN，详情在本地CASE_REVIEW及匿名审查表。

两条一步TTA改善CASE005/006均精确恢复R76原query；放大后主要是同一车辆附近的近似框切换，直接表示融合也恢复，不能证明新实例绑定。唯一一步恶化CASE002也回到R76原query，两框主要包围同一行人范围。CASE003/004则在联合零步前向已经从主体区域转向另一行人/邻车区域；随后TTA没有救回。CASE009/010由同信息直接融合选到更有重叠的候选，TTA仍未实现相应选择。CASE011是所有状态一致的稳定正例；CASE012仍将主体与参照类别混淆。CASE001无空间角色，只支持query后备路径的静态改进。固定R89_CASE005的旧适应正例在本轮只由直接融合保留。

保存分数进一步区分“模块未执行”与“模块没有任务价值”：撤去参照的零步分布最大绝对差均值为0.000230892，改变1个query但0个IoU；旧式池化和错位支持零步分别改变0个query。完整一步与纯物理/绕过条件表示相比，概率分布均确有数值差，但336条Top1 query完全相同。这说明新增路径在计算，尚未推动决定性的目标排序；不能再用参数非零更新作为其有效性证明。
''')
print({'pages':len(cases),'unique_observations':len({c['row']['id'] for c in cases}),'anonymous_review_rows':len(public)})
