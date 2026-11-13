"""Postseal whole-cohort interpretation and reviewed-case record; no method changes."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round89';read=lambda p:json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');routing={r['row']:r for r in read(H/'LEARNED_ROUTING_ROWS.json')};s=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336']['balanced_mIoU_percent'];pairs=[('candidate_frozen','frozen_bi'),('candidate_frozen','continue_frozen'),('candidate_frozen','query_frozen'),('candidate_tta1','candidate_frozen'),('candidate_tta1','continue_tta1'),('candidate_tta1','candidate_physical1')];rows=[]
for known in [False,True]:
 tt=[t for t in tasks if (routing[t['row']]['roles']>0)==known]
 for a,b in pairs:
  v=np.array([t['IoU'][a]-t['IoU'][b] for t in tt]);rows.append({'spatial_roles_available':known,'n':len(tt),'arm':a,'reference':b,'subset_delta_pp':float(v.mean()*100),'contribution_to_all336_pp':float(v.sum()*100/336),'better_n':int((v>1e-10).sum()),'worse_n':int((v< -1e-10).sum())})
dump(H/'KNOWN_ROLE_PARTITION_ROWS.json',rows);d=read(H/'DECISION.json');d.update(full_TTA_recipe_adopted=False,source_prepared_binding_robustly_established=False,same_information_development_increment_positive=True,primary_vs_matching_continuation_TTA_pp=s['candidate_tta1']-s['continue_tta1'],primary_vs_physical_only_pp=s['candidate_tta1']-s['candidate_physical1'],full_structure_preparation_vs_query_only_pp=s['candidate_frozen']-s['query_frozen'],primary_vs_own_frozen_better_n=1,primary_vs_own_frozen_worse_n=2,primary_vs_own_frozen_unchanged_n=333,decision_reason='tiny_positive_increment_but_source_prepared_structure_and_added_correspondence_not_supported_against_stronger_controls',no_follow_on_queued=True);dump(H/'DECISION.json',d)
review='''# R89 private case review

All11 distinct selected pages were visually inspected with full caption, full scene, diagnostic crop, source/R76/new frozen/TTA/mean/wrong-space boxes, global/local support and ranking table. Three fixed R88 pages duplicate CASE007/002/008. No case was used to change a method or select a threshold. IoU is not itself an instance identity label. Raw captions/coordinates/indices/media remain in the private atlas only.

- CASE001: new frozen and TTA put the box onto the clipped foreground target area rather than high background. However roles=0: this is the trained query-only fallback, not evidence that spatial-language binding succeeded. Frozen0→0.3185; TTA unchanged; mean loses it.
- CASE002: all R76/new/TTA hypotheses lie around the same rear vehicle; the main difference is overlapping candidate boxes, not a demonstrated target-identity correction. New frozen0.4108 vsR760.2180; TTA returns0.2180. Physical-only and no-correspondence preserve0.4108. Added correspondence causes the main harm.
- CASE003: R76 box covers the described dark-vehicle area. New head selects the adjacent light vehicle at left; TTA cannot repair. Wrong-space and query-only/continued-head controls retain the better area. This is a concrete static-interface contamination case, not a missing A256 box.
- CASE004: R76 selects the distant target-vehicle area, while new frozen/TTA select an elevated background box. Large context and tiny object limit fine appearance verification, but box displacement is visible. Wrong-space/query-only recover the original region. Do not infer calibration corruption solely from this projection.
- CASE005: TTA changes from a background/other-instance box to the foreground target vehicle (0→0.4455). Frozen same-information mean reaches the same box. Pure source already0.4675. Thus a real recovery exists, but it is not an exclusive parameter-update success or evidence of new information beyond source/multi-read.
- CASE006: both predictions remain away from the best target box, with overlapping candidate boxes switching0.0043→0. This is a tiny task degradation, not a large successful/failed identity event.
- CASE007 /R88_CASE002: R76/new frozen/TTA all preserve the target area0.4530, while mean moves to distant background. The R88 reading-induced harm is not reproduced in this new interface; no added TTA gain on this input.
- CASE008 /R88_CASE007: source/new/TTA select a background person; same3read mean selects the visibly foreground described person0.5738. Subject/reference parsing distinguishes three people, yet spatial reading remains background-focused. This is failure to use available same-observation information.
- CASE009: direct mean improves0.4432→0.6179 on overlapping boxes around the same person. It supports better within-instance box selection, not necessarily fixing the referred identity. TTA remains0.4432.
- CASE010: all methods keep a high-quality person box0.8127. Successful unchanged example retained.
- CASE011: graph has a person as subject and vehicle/tree as references; predicted box instead covers the vehicle, while a person candidate with0.4793 exists. This is visible subject/reference or class binding failure despite correct coarse textual roles, not proof the parser alone caused it.

Images support these bounded region/role observations. They do not authenticate every 3D instance or every color/relationship, and attention/support overlays are not instance masks.
''';(H/'CASE_REVIEW.md').write_text(review)
text=(H/'REPORT.md').read_text();head=f'''## 最终综合裁决（同预算强对照与原图复核）

主要一步更新的同模型增量为+0.073904pp，但只来自1条改善、2条恶化、333条不变；唯一改善的全体正贡献为+0.132602pp，超过全部净增量。新冻结结构22.565283低于R7622.611596，也低于仅query残差22.623837。相同源域预算继续训练既有匹配头＋同样TTA达到22.828464，比完整新方法高0.189277pp。只做物理恢复为22.696590，比完整方法高0.057404pp；去掉对应项与其任务输出相同。去掉参照物后22.641382也未下降。

因此必须同时保留两项事实：**参数更新相对自身零步和同信息平均有微弱开发正增量；完整候选条件化绑定方案没有战胜更简单的同预算对照，不能据此采纳整套TTA或宣称可靠实体绑定已建立。** 数值增量检查通过不等于完整机制或稳健方法通过。

第三种未拟合遮挡的局部表示误差均值从0.00267514647变为0.00271294790，未显示平均恢复改善。加入对应项不应以“优化成功”作为正证据。

### 原图复核的代表性结论

已逐页查看11个不同样本及3个重复固定跟踪页。CASE005确实从其他区域回到目标车辆，但直接平均也成功，且纯3EED原先已有更高IoU；CASE002主要是同一车辆的重叠框切换，新增对应项撤回零步收益；CASE003/004显示新接口先把原有目标区域改为邻物或背景；CASE008直接平均选到前景目标，TTA仍选择背景人物；CASE011文字图已区分主体和参照类型，最终却仍选择参照车辆。CASE001静态收益来自无角色时的query后备路径，不能计作空间绑定成功。

这些案例解释总体差异，不用于修改规则。不能把所有坏例归为同一种语言错误，也不能把个别成功归给全部新组件。

### 整体按空间角色可用性分层

仅按无标签图中的角色是否存在分组，事后列性能；不筛预测、不重选参数。

| 有空间角色 | n | 方法 | 对照 | 子集Δpp | 对全336贡献pp | 改善/恶化 |
|---|---:|---|---|---:|---:|---:|
'''
for r in rows:head+=f'| {r["spatial_roles_available"]} | {r["n"]} | {r["arm"]} | {r["reference"]} | {r["subset_delta_pp"]:+.6f} | {r["contribution_to_all336_pp"]:+.6f} | {r["better_n"]}/{r["worse_n"]} |\n'
head+='\n结论限定为冻结R76特征＋轻量接口、512条源训练及当前固定适应规则。更充分的源域准备、不同局部支持或任务对齐仍是竞争解释；本轮没有检验它们，也不继续按已暴露目标结果加步数/倍率/规则。当前不采纳完整方法，保留新架构、三个源域准备对照及全部正负例作为证据。\n\n---\n\n';(H/'REPORT.md').write_text(head+text);print(d)
