"""A208 postseal task/coverage tables; no model changes or hidden feedback filter."""
import sys,collections,numpy as np
from common37 import *
from local_engine37 import check_amendment
def main():
 check_amendment();check_plan();out=BASE/'inference208_attempt002';verify(out);audit=read(EXP/'CPU_AUDIT.json');assert audit['state']=='PASSED' and audit['prediction_seal_sha256']==sha(out/'SEALED.json');sys.path.insert(0,str(ROOT/'research/round27'));from evaluate27 import ious
 records=[];negative=[];summary={};coverage={};scene=[]
 for r in rows():
  z=load_npz(out/'predictions_PRIVATE'/(r['id']+'.npz'));rr=read(out/'predictions_PRIVATE'/(r['id']+'.json'));pool=arrays(r)['source_boxes'];uq,inv=np.unique(z['queries'],return_inverse=True);v=ious(r,pool[uq])[inv];rec={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'queries':dict(zip(ARMS,map(int,z['queries']))),'IoU':dict(zip(ARMS,map(float,v))),'readouts':rr['readouts']};records.append(rec)
  for arm in ARMS:
   for ref in ['source','R29_direct','legacy_G','point_fullmap','som_fullmap']:
    if arm!=ref and rec['IoU'][arm]<rec['IoU'][ref]-1e-10:negative.append({'alias':r['alias'],'platform':r['platform'],'arm':arm,'reference':ref,'delta_pp':100*(rec['IoU'][arm]-rec['IoU'][ref])})
 for pl in ['drone','quad']:
  rr=[r for r in records if r['platform']==pl];summary[pl]={};coverage[pl]={}
  for arm in ARMS:
   v=np.array([r['IoU'][arm] for r in rr]);d=v-np.array([r['IoU']['source'] for r in rr]);summary[pl][arm]={'n':len(rr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'delta_vs_source_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][arm]!=r['queries']['source'] for r in rr)}
  for arm in ['point_fullmap','point_oldsubset','som_fullmap','som_oldsubset']:
   st=collections.Counter(r['readouts'][arm]['status'] for r in rr);coverage[pl][arm]={'n':len(rr),'mapped':sum(r['readouts'][arm]['mapped'] for r in rr),'sourcefallback_unknown':sum(not r['readouts'][arm]['mapped'] for r in rr),'mapped_can_keep_q0':sum(r['readouts'][arm]['mapped'] and not r['readouts'][arm]['query_changed'] for r in rr),'status_counts':dict(st)}
  for j,s in enumerate(sorted({r['scene_PRIVATE'] for r in rr})):
   ss=[r for r in rr if r['scene_PRIVATE']==s];scene.append({'platform':pl,'scene_alias':pl+'_scene'+str(j),'n':len(ss),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in ss])*100) for a in ARMS}})
 summary['balanced_mIoU_percent']={a:float(np.mean([summary[pl][a]['mIoU_percent'] for pl in ['drone','quad']])) for a in ARMS};contrasts=[];rng=np.random.default_rng(37);clusters={pl:[[r for r in records if r['platform']==pl and r['scene_PRIVATE']==s] for s in sorted({r['scene_PRIVATE'] for r in records if r['platform']==pl})] for pl in ['drone','quad']}
 for arm,ref,label in [('point_fullmap','source','native_point_net'),('point_fullmap','som_fullmap','interface_at_full_mapping'),('point_oldsubset','som_oldsubset','interface_at_old_mapping'),('point_fullmap','point_oldsubset','point_mapping_coverage'),('som_fullmap','som_oldsubset','SoM_mapping_coverage'),('point_fullmap','legacy_G','legacy_precision_reference')]:
  by={pl:float(np.mean([r['IoU'][arm]-r['IoU'][ref] for r in records if r['platform']==pl])*100) for pl in ['drone','quad']};dist=[]
  for _ in range(2000):
   d=[]
   for pl,cc in clusters.items():
    rs=[r for j in rng.integers(0,len(cc),len(cc)) for r in cc[j]];d.append(np.mean([r['IoU'][arm]-r['IoU'][ref] for r in rs]))
   dist.append(100*np.mean(d))
  contrasts.append({'contrast':label,'arm':arm,'reference':ref,'platform_delta_pp':by,'balanced_delta_pp':float(np.mean(list(by.values()))),'descriptive_scene_bootstrap95_pp':np.quantile(dist,[.025,.975]).tolist(),'scene_clusters':{p:len(c) for p,c in clusters.items()},'few_exposed_scenes_not_independent_confirmatory_CI':True})
 cost=[]
 for interface in ['point','som']:
  cc=[read(out/'calls_PRIVATE'/(r['id']+'_'+interface+'.json'))['result'] for r in rows()];cost.append({'interface':interface,'calls':len(cc),'image_count_per_call':1 if interface=='point' else 2,'input_tokens_total':sum(c['input_tokens'] for c in cc),'image_visual_tokens_total':sum(sum(c['image_visual_tokens']) for c in cc),'generated_tokens_total':sum(c['generated_tokens'] for c in cc),'generate_seconds_total':sum(c['seconds_generate'] for c in cc),'parse_failure':sum(c['parsed']['status']=='parse_failure' for c in cc),'explicit_null':sum(c['parsed']['status']=='explicit_null' for c in cc),'no_TTA_steps':True})
 b=summary['balanced_mIoU_percent'];decision={'state':'COMPLETED_G_ONLY_DEVELOPMENT','native_point_balanced_gain_vs_source_pp':b['point_fullmap']-b['source'],'native_point_balanced_gain_vs_allmask_ID_pp':b['point_fullmap']-b['som_fullmap'],'task_point_estimate_positive':b['point_fullmap']>b['source'],'identity_review':'separately blinded assistant provisional review, not read by inference or used to filter; root/influence_audit to join after annotation seal','no_parameter_TTA_or_new_boxes':True,'no_U_T_FINAL_inference_or_scoring':True,'not_independent_confirmation':True,'automatic_next_method_or_prompt_change':False}
 dump(EXP/'TASK_SUMMARY.json',summary);dump(EXP/'ALL_ROWS_PRIVATE.json',records);dump(EXP/'ALL_NEGATIVES.json',negative);dump(EXP/'COVERAGE_UNKNOWN.json',coverage);dump(EXP/'SCENE_ROWS.json',scene);dump(EXP/'FACTOR_CONTRASTS.json',contrasts);dump(EXP/'COSTS.json',cost);dump(EXP/'DECISION.json',decision);print('R37_DECISION',decision);print('R37_BALANCED',b);print('R37_COSTS',cost)
if __name__=='__main__':main()
