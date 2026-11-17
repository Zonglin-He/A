"""Post-seal mechanism, radius and all 18 pre-existing good/bad case audits."""
from pathlib import Path
import json,sys,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round130';B=R/'crossplatform/round130';Q=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text());sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
PAIRS=[('proper_half','wrong_half'),('partial','wrong_base'),('proper_double','wrong_double')]
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def compare(rr,k,c):
 d=np.array([r['arms'][k]['IoU']-r['arms'][c]['IoU'] for r in rr]);return {'n':len(rr),'gain_pp':float(d.mean()*100) if len(d) else 0.,'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum()),'query_changes':sum(r['arms'][k]['query_PRIVATE']!=r['arms'][c]['query_PRIVATE'] for r in rr),'largest_positive_contribution_pp':float(d.max()*100/len(d)) if len(d) else 0.,'largest_negative_contribution_pp':float(d.min()*100/len(d)) if len(d) else 0.}
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==2304;rows=[r for f in Q.glob('*_ALL_ROWS_PRIVATE.json') for r in rd(f)];process=rd(H/'PROCESS_ROWS_PRIVATE.json');out=[];pstats=[];quality=[];changes=[];cases=[];reg={(e['cohort'],e['source'],e['target'],e['row']['id']):e for e in rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json')}
 for co in ['DEV','EXP']:
  rr=[r for r in rows if r['cohort']==co]
  for label,ss in [('all',rr),('partial_active',[r for r in rr if r['partial_eligible']]),('effective_shuffle',[r for r in rr if r['effective_shuffle']]),('identity_controls',[r for r in rr if r['partial_eligible'] and not r['effective_shuffle']])]:
   for k,w in PAIRS:
    for v in ['F','C']:
     for c in ['zero_'+v,'direct_partial_'+v,w+'_'+v]:out.append({'cohort':co,'subset':label,'method':k+'_'+v,'control':c,**compare(ss,k+'_'+v,c)})
  for setting in ['proper_half','proper_double','wrong_half','wrong_base','wrong_double']:
   pp=[p for p in process if p['cohort']==co and p['setting']==setting];pstats.append({'cohort':co,'setting':setting,'active':len(pp),'nonzero':sum(p['parameter_L2']>0 for p in pp),'accepted_steps':sum(p['accepted_steps'] for p in pp),'Armijo_rejected_trials':sum(p['Armijo_rejected_trials'] for p in pp),'mean_parameter_L2':float(np.mean([p['parameter_L2'] for p in pp])),'mean_task_before':float(np.mean([p['task_before'] for p in pp])),'mean_task_after':float(np.mean([p['task_after'] for p in pp])),'mean_keep_after':float(np.mean([p['feature_keep_after'] for p in pp]))})
  qual=[]
  for r in rr:
   if not r['partial_eligible']:continue
   sid,name=r['id_PRIVATE'],r['source']+'_to_'+r['target'];ctrl=np.load(B/'control_PRIVATE'/co/name/(sid+'.npz'));scope=np.load(R/'crossplatform/round129/scope_PRIVATE'/co/name/(sid+'.npz'));a=ctrl['assignment'];S,C,Cw=ctrl['S'],ctrl['C'],ctrl['wrong_C'];p=scope['pF'];gp=np.array([p[a==j].mean() for j in range(len(S))]);gp=gp/gp[S].sum();qual.append({'q0_in_C':r['control_metadata']['q0_in_C'],'q0_in_wrong_C':r['control_metadata']['q0_in_wrong_C'],'q0_in_S':r['control_metadata']['q0_in_S'],'conditional_mass_C':float(gp[C].sum()),'conditional_mass_wrong':float(gp[Cw].sum()),'query_mass_C':float(p[C[a]].sum()),'query_mass_wrong':float(p[Cw[a]].sum()),'effective':r['effective_shuffle']})
  quality.append({'cohort':co,'active':len(qual),'effective':sum(q['effective'] for q in qual),'q0_in_S':sum(q['q0_in_S'] for q in qual),'q0_in_C':sum(q['q0_in_C'] for q in qual),'q0_in_wrong_C':sum(q['q0_in_wrong_C'] for q in qual),'mean_conditional_mass_C':float(np.mean([q['conditional_mass_C'] for q in qual])),'mean_conditional_mass_wrong':float(np.mean([q['conditional_mass_wrong'] for q in qual])),'mean_query_mass_C':float(np.mean([q['query_mass_C'] for q in qual])),'mean_query_mass_wrong':float(np.mean([q['query_mass_wrong'] for q in qual]))})
 oldrows={(r['cohort'],r['source'],r['target'],r['id_PRIVATE']):r for f in (R/'research/round129/scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json') for r in rd(f)}
 for r in rows:
  co,src,tgt,sid=r['cohort'],r['source'],r['target'],r['id_PRIVATE'];old=oldrows[(co,src,tgt,sid)];base=old['arms']['zero_F']['IoU'];old_delta=old['arms']['partial_F']['IoU']-base
  for k in [a+'_'+v for a,b in PAIRS for v in ['F','C']]+[b+'_'+v for a,b in PAIRS for v in ['F','C']]:
   c='zero_'+k.rsplit('_',1)[1];delta=r['arms'][k]['IoU']-r['arms'][c]['IoU']
   if abs(delta)>1e-12:changes.append({'cohort':co,'source':src,'target':tgt,'id_PRIVATE':sid,'scene_PRIVATE':r['scene_PRIVATE'],'method':k,'control':c,'baseline_IoU':r['arms'][c]['IoU'],'adapted_IoU':r['arms'][k]['IoU'],'delta_IoU':delta,'effective_shuffle':r['effective_shuffle'],'identity_audit':'IoU change is not semantic instance identity'})
  if abs(old_delta)<=1e-12:continue
  assert not r['protected_alias_fallback'];e=reg[(co,src,tgt,sid)];pred=np.load(B/'formal'/co/(src+'_to_'+tgt)/'samples_PRIVATE'/(sid+'.npz'));probs=dict(zip(pred['arm_names'],pred['scores']));boxes=np.c_[pred['source_boxes'],np.zeros(256)];boxes[:,3:6]=np.maximum(boxes[:,3:6],1e-6);gt=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json')[e['row']['anno_index']]['source_GT7_float32'];ys=many_iou(boxes,np.array(gt));ctrl=np.load(B/'control_PRIVATE'/co/(src+'_to_'+tgt)/(sid+'.npz'));a=ctrl['assignment'];S,C=ctrl['S'][a],ctrl['C'][a];better=ys>base+1e-8;good=ys>.25
  kind='source_already_IoU_oracle' if not better.any() else 'all_better_candidates_outside_S' if not (better&S).any() else 'S_has_better_but_C_excludes_all_better' if not (better&C).any() else 'C_has_better_candidate_but_internal_order_unconstrained'
  info={'cohort':co,'source':src,'target':tgt,'id_PRIVATE':sid,'scene_PRIVATE':r['scene_PRIVATE'],'physical_frame_PRIVATE':hashlib.sha256((tgt+'|'+e['row']['scene']+'|'+e['row']['pointpath']).encode()).hexdigest(),'R129_effect':'improved' if old_delta>0 else 'worsened','R129_delta_IoU':old_delta,'source_IoU':base,'oracle_IoU':float(ys.max()),'diagnostic_location':kind,'S_has_Acc25':bool((S&good).any()),'C_has_Acc25':bool((C&good).any()),'q0_in_S':bool(S[r['arms']['zero_F']['query_PRIVATE']]),'q0_in_C':bool(C[r['arms']['zero_F']['query_PRIVATE']]),'selected_partial_in_C':bool(C[r['arms']['partial_F']['query_PRIVATE']]),'geometry_is_not_instance_identity':True}
  ids=np.flatnonzero(C)
  if len(ids):
   bestc=int(ids[np.argmax(ys[ids])]);info['best_C_IoU']=float(ys[bestc]);info['best_C_rank_zero']=int((probs['zero_F']>probs['zero_F'][bestc]).sum()+1);info['best_C_rank_partial']=int((probs['partial_F']>probs['partial_F'][bestc]).sum()+1)
  for k in ['partial_F','proper_half_F','proper_double_F','wrong_base_F','direct_partial_F']:info[k+'__IoU']=r['arms'][k]['IoU']
  cases.append(info)
 assert len(cases)==18 and sum(c['R129_effect']=='improved' for c in cases)==11 and sum(c['R129_effect']=='worsened' for c in cases)==7
 dump(H/'SUBSET_COMPARISONS.json',out);dump(H/'PROCESS_SUMMARY.json',pstats);dump(H/'CONTROL_QUALITY.json',quality);dump(H/'CHANGED_CASES_PRIVATE.json',changes);dump(H/'R129_ALL18_CASES_PRIVATE.json',cases);dump(H/'CASE_SUMMARY.json',{'n':len(cases),'improved':11,'worsened':7,'physical_frames':len({c['physical_frame_PRIVATE'] for c in cases}),'scene_clusters':len({(c['target'],c['scene_PRIVATE']) for c in cases}),'location_counts':{k:sum(c['diagnostic_location']==k for c in cases) for k in sorted({c['diagnostic_location'] for c in cases})},'GT_used_only_after_seal_for_diagnosis':True,'same_frame_or_scene_not_independent':True});print({'case_n':len(cases),'control_quality':quality})
if __name__=='__main__':main()
