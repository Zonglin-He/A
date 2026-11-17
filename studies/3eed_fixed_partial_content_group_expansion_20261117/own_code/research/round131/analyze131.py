"""Post-seal explanatory analysis; no method or output selection."""
from pathlib import Path
import json,collections,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round131';O=H/'scores_attempt001'
rd=lambda p:json.loads(Path(p).read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==1798
 rows=[r for f in sorted(O.glob('*_ALL_ROWS_PRIVATE.json')) for r in rd(f)];proc=rd(H/'PROCESS_ROWS_PRIVATE.json');methods=rd(O/'ALL_METHOD_ROWS.json');tails=[];summaries=[];subset=[]
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if (r['source'],r['target'])==(src,tgt)];den=sum(r['sampling_weight'] for r in rr);pr=[p for p in proc if (p['source'],p['target'])==(src,tgt)];diag=[r['diagnostics'] for r in rr];active=[r for r in rr if r['partial_eligible']];recoverable=[d for d in diag if d['recoverable_zero_error_Acc25']]
  summaries.append({'source':src,'target':tgt,'n':len(rr),'content_groups':len({r['content_group_PRIVATE'] for r in rr}),'frame_paths':len({r['frame_PRIVATE'] for r in rr}),'recordings':len({r['scene_PRIVATE'] for r in rr}),'B_available':sum(r['B_available'] for r in rr),'scope_eligible':sum(r['scope_eligible'] for r in rr),'active':len(active),'effective_shuffle':sum(r['effective_shuffle'] for r in rr),'identity_control':sum(r['partial_eligible'] and not r['effective_shuffle'] for r in rr),'source_oracle_Acc25':sum(d['oracle_Acc25'] for d in diag),'recoverable_zero_errors':len(recoverable),'recoverable_S_has_good':sum(d['S_has_Acc25'] for d in recoverable),'recoverable_C_has_good':sum(d['C_has_Acc25'] for d in recoverable),'active_q0_in_S':sum(r['diagnostics']['q0_in_S'] for r in active),'active_q0_in_S_but_excluded_C':sum(r['diagnostics']['q0_in_S'] and not r['diagnostics']['q0_in_C'] for r in active),'within_C_eligible_recoverable':sum('within_C_teacher_already_prefers_better_Acc25_candidate' in d for d in diag),'within_C_eligible_same_q0_group':sum('within_C_teacher_already_prefers_better_Acc25_candidate' in d and d['best_C_same_geometry_group_as_q0'] for d in diag),'within_C_teacher_prefers_better':sum(d.get('within_C_teacher_already_prefers_better_Acc25_candidate',False) for d in diag),'model_seconds_mean':float(np.mean([r['seconds'] for r in rr])),'partial_accepted_steps':sum(p['accepted_steps'] for p in pr if p['setting']=='partial'),'partial_norm_mean_active':float(np.mean([p['parameter_L2'] for p in pr if p['setting']=='partial' and p['active']])) if active else 0.})
  for name,predicate in [('all',lambda r:True),('active_partial',lambda r:r['partial_eligible']),('effective_shuffle',lambda r:r['effective_shuffle']),('no_B',lambda r:not r['B_available']),('recoverable_zero_error',lambda r:r['diagnostics']['recoverable_zero_error_Acc25'])]:
   ss=[r for r in rr if predicate(r)]
   if not ss:continue
   for control in ['zero_F','direct_partial_F','wrong_base_F']:
    d=np.array([r['arms']['partial_F']['IoU']-r['arms'][control]['IoU'] for r in ss]);w=np.array([r['sampling_weight'] for r in ss]);subset.append({'source':src,'target':tgt,'subset':name,'n':len(ss),'control':control,'raw_gain_pp':float(d.mean()*100),'weighted_gain_pp':float(np.average(d,weights=w)*100),'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum())})
  for r in rr:
   a=r['arms'];d=a['partial_F']['IoU']-a['zero_F']['IoU']
   if abs(d)<=1e-12:continue
   z=r['diagnostics'];category='better_candidate_inside_C' if z['C_has_better_IoU_than_zero'] else ('better_candidate_in_S_excluded_C' if z['S_has_better_IoU_than_zero'] else ('zero_already_A256_IoU_oracle' if a['zero_F']['IoU']>=z['oracle_IoU']-1e-8 else 'better_candidate_outside_S'))
   tails.append({k:r[k] for k in ['source','target','id_PRIVATE','frame_PRIVATE','content_group_PRIVATE','scene_PRIVATE','partial_eligible','effective_shuffle']}|{'change':'better' if d>0 else 'worse','category':category,'zero_IoU':a['zero_F']['IoU'],'direct_IoU':a['direct_partial_F']['IoU'],'partial_IoU':a['partial_F']['IoU'],'wrong_IoU':a['wrong_base_F']['IoU'],'gain_IoU':d,'weighted_six_direction_contribution_pp':d*r['sampling_weight']/den*100/6,**z})
 controls=[]
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if (r['source'],r['target'])==(src,tgt) and r['partial_eligible']];aa=[]
  for r in rr:
   f=np.load(R/'crossplatform/round131/formal'/(src+'_to_'+tgt)/'features_PRIVATE'/(r['id_PRIVATE']+'.npz'));ass=f['assignment'];S=f['S'][ass];C=f['C'][ass];W=f['wrong_C'][ass];p=r['arms'] # only opens sealed feature artifact, no labels for construction
   with np.load(R/'crossplatform/round131/formal'/(src+'_to_'+tgt)/'samples_PRIVATE'/(r['id_PRIVATE']+'.npz')) as a:prob=dict(zip(a['arm_names'],a['scores']))['zero_F']
   aa.append((prob[C].sum()/prob[S].sum(),prob[W].sum()/prob[S].sum(),C[p['zero_F']['query_PRIVATE']],W[p['zero_F']['query_PRIVATE']]))
  controls.append({'source':src,'target':tgt,'active_n':len(rr),'correct_initial_retained_mass_mean':float(np.mean([a[0] for a in aa])) if aa else None,'wrong_initial_retained_mass_mean':float(np.mean([a[1] for a in aa])) if aa else None,'correct_q0_retained':sum(bool(a[2]) for a in aa),'wrong_q0_retained':sum(bool(a[3]) for a in aa)})
 dump(H/'PROCESS_SUMMARY.json',summaries);dump(H/'SUBSET_COMPARISONS.json',subset);dump(H/'CONTROL_QUALITY.json',controls);dump(H/'CHANGED_CASES_PRIVATE.json',tails)
 out={'direction_inputs':len(rows),'content_groups':len({r['content_group_PRIVATE'] for r in rows}),'frame_paths':len({r['frame_PRIVATE'] for r in rows}),'recording_clusters':len({(r['target'],r['scene_PRIVATE']) for r in rows}),'changed_inputs':len(tails),'improved':sum(r['change']=='better' for r in tails),'worsened':sum(r['change']=='worse' for r in tails),'changed_content_groups':len({r['content_group_PRIVATE'] for r in tails}),'changed_recordings':len({(r['target'],r['scene_PRIVATE']) for r in tails}),'categories':dict(collections.Counter(r['category'] for r in tails)),'largest_positive_contribution_pp':max([0.]+[r['weighted_six_direction_contribution_pp'] for r in tails]),'largest_negative_contribution_pp':min([0.]+[r['weighted_six_direction_contribution_pp'] for r in tails]),'post_GT_analysis_only':True,'IoU_does_not_certify_instance_identity':True,'new_scene_generalization_established':False,'single_Drone_recording_no_population_scene_CI':True};dump(H/'CASE_SUMMARY.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
