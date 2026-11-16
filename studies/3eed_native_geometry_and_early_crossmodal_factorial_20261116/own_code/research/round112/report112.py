"""Full component causal factorial, fixed primary and all negative task results."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round112';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;agg=rd(O/'AGGREGATE_ROWS.json');aa={r['arm']:r for r in agg if r['group']=='all_six_equal'};v=lambda k:aa[k]['mIoU_percent'];p='noPA_correct_fusedABC';controls=['original_source','original_source_matchedcoverage','prepared_frozen','real_frozen_fusedABC','native_real_fusedABC','direct_three_geometric','noPA_wrong_fusedABC','noEarlyRGB_correct_fusedABC','neither_correct_fusedABC'];rows=[r for f in O.glob('*_ALL_ROWS_PRIVATE.json') for r in rd(f)];conc=[];sub=[];cases=[];factor=[];latent=[];activity=[]
 for c in controls:
  d=np.array([r['arms'][p]['IoU']-r['arms'][c]['IoU'] for r in rows]);conc.append({'control':c,'n':len(d),'gain_pp':float(d.mean()*100),'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum()),'largest_positive_single_input_pp':float(max(0,d.max())*100/768),'largest_negative_single_input_pp':float(min(0,d.min())*100/768)})
 def arm(variant,binding,view):return ('prepared_frozen' if view=='A' else 'real_frozen_fusedABC') if variant=='full' and binding=='correct' else variant+'_'+binding+'_'+view
 for binding in ['correct','wrong']:
  for view in ['A','fusedABC']:
   full,nop,nor,neither=[v(arm(x,binding,view)) for x in ['full','noPA','noEarlyRGB','neither']];factor.append({'binding':binding,'view':view,'full':full,'noPA':nop,'noEarlyRGB':nor,'neither':neither,'remove_PA_with_RGB_active_pp':nop-full,'remove_PA_with_RGB_off_pp':neither-nor,'remove_earlyRGB_with_PA_active_pp':nor-full,'remove_earlyRGB_with_PA_identity_pp':neither-nop,'interaction_pp':full-nop-nor+neither,'full_system_readout_late_and_competition_retained':True})
 for label in [x['arm'] for x in agg if x['group']=='all_six_equal' and x['arm'].startswith(('noPA_','noEarlyRGB_','neither_','full_wrong_'))]:
  rr=[r['status']['component_edits'][label] for r in rows if not r['status']['protected_alias_fallback']];latent.append({'arm':label,'n':len(rr),'hidden_RMS_from_native_mean':float(np.mean([x['hidden_RMS_from_native'] for x in rr])),'text_RMS_from_native_mean':float(np.mean([x['text_RMS_from_native'] for x in rr])),'not_grounding_correctness_proxy':True});activity.append({'arm':label,'n':len(rr),'mean_changed_values':float(np.mean([x['changed_values'] for x in rr])),'mean_parameter_L2':float(np.mean([x['parameter_L2'] for x in rr])),'static_constant_edit_not_TTA':True})
 for label in ['all','nonalias','actual_C_available','no_actual_C_nonalias','protected_alias']:
  rr=[r for r in rows if label=='all' or (label=='nonalias' and not r['status']['protected_alias_fallback']) or (label=='actual_C_available' and r['status']['families'].get('real',{}).get('C_available',False)) or (label=='no_actual_C_nonalias' and not r['status']['protected_alias_fallback'] and not r['status']['families'].get('real',{}).get('C_available',False)) or (label=='protected_alias' and r['status']['protected_alias_fallback'])]
  if rr:
   for k in [p]+controls:sub.append({'subset':label,'n':len(rr),'arm':k,'mIoU_percent':float(np.mean([r['arms'][k]['IoU'] for r in rr])*100)})
 for direction in rd(R/'crossplatform/round112/MATRIX_SEALED.json')['directions']:
  rr=[r for r in rows if r['source']+'_to_'+r['target']==direction]
  for control in ['real_frozen_fusedABC','noPA_wrong_fusedABC']:
   for kind,sign in [('good',1),('failure',-1)]:
    part=[r for r in rr if sign*(r['arms'][p]['IoU']-r['arms'][control]['IoU'])>1e-12]
    if part:
     r=max(part,key=lambda r:abs(r['arms'][p]['IoU']-r['arms'][control]['IoU']));cases.append({'source':r['source'],'target':r['target'],'row_PRIVATE':r['id_PRIVATE'],'kind':kind,'control':control,'delta_IoU':r['arms'][p]['IoU']-r['arms'][control]['IoU'],'postseal_only':True})
 gate=all(v(p)>v(c) for c in controls);decision={'primary':p,'mIoU_percent':v(p),'gains_pp':{c:v(p)-v(c) for c in controls},'passes_static_repair_gate':gate,'TTA_run':False,'robust_TTA_adoption':False,'decision':'static_repair_candidate_requires_locked_expansion_not_TTA_claim' if gate else 'do_not_adopt_fixed_noPA_primary_as_method','n':768,'primary_not_switched':True,'FINAL_access':False}
 for name,x in [('DECISION',decision),('CONCENTRATION_SUMMARY',conc),('SUBSET_RESULTS',sub),('FACTORIAL_EFFECTS',factor),('LATENT_COMPONENT_DIAGNOSTIC',latent),('STATIC_EDIT_ACTIVITY',activity),('GOOD_FAILURE_CASES_PRIVATE',cases)]:dump(H/(name+'.json'),x)
 lines=['# R112 — 点云Adapter与早期跨模态交换的受控撤回','','R111已确认原始与prepared匹配投影在三个source逐值位等，且native同证据路径比prepared好。因此本轮只改变新增特征路径，不再改匹配投影、BN或拟合新伪标签。全部768历史暴露开发输入，89读出：75旧读出/评分复用，14新增。44保护别名保留冻结回退。没有TTA、优化、新loss、source准备或专家，原A256/六层位置固定，FINAL未动。','','## 实际干预与保留部分','','2×2固定干预：point_adapter.up.weight+bias归零使残差成为恒等，gate_P归零阻止早期RGB→LiDAR；二者单独和组合对照。RGB编码器、image Adapter、反向point_to_image、lateRGB和竞争接口保留。正确与一次固定visible-only错配绑定均执行，A及同一实测点云fusedABC都比较。route成员和实体假设来自原fullR91当前A并在所有干预前固定。','','12实际工程输入验证完整R91重放、实参改写与功能式旁路概率/隐藏表示/late/r位等、独立CPU读出、源状态恢复。二者都撤回时，匹配投影之前的h和u与R111native A/fused在全部非别名输入上位等。这只证明干预位置准确，不证明最终grounding正确。最终保留late/r，故不要求完整最终分数等于裸source。','','## 因子作用（任务百分点）','','```json',json.dumps(factor,indent=2),'```','','该表是全系统固定读出的因果干预效应；其中包含后续文字交互、RGB反向路径和late/r的传播，不是某个通道的独立语义贡献。交互项非零时不可把单模块效果简单相加。','','## 全体/平台结果','','| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |','|---|---|---:|---:|---:|']
 for x in agg:lines.append(f"| {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 全部配对结果','','| Source | Target | 方法 | 对照 | Δ pp | 改善/恶化/不变 |','|---|---|---|---|---:|---:|']
 for x in rd(O/'ALL_PAIRED_ROWS.json'):lines.append(f"| {x['source']} | {x['target']} | {x['method']} | {x['control']} | {x['gain_pp']:+.6f} | {x['better']}/{x['worse']}/{x['same']} |")
 lines+=['','## 固定主项裁决','','```json',json.dumps(decision,indent=2),'```','','保持总体分母、两域/三域结果、所有场景聚类与正负尾部。固定撤回不是测试时学习；即便分数上涨，也不能包装为参数TTA贡献。错配有时不改变预测是合法结果，不要求每条反转。候选框覆盖/实体身份仍需独立分清；隐藏偏移变小不是有效性的替代。未过门槛不通过扩大倍率或额外门槛挽救，也不把事后最优对照改为本轮主项。'];(H/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
