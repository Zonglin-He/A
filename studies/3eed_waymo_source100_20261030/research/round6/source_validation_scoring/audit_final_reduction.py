"""Final-only independent scalar reduction from sealed IoUvectors/native retrieval.
Requires all10sourceVALscores + actual training100complete. No GT/model/GPU access.
The main scorer already crosschecks everyselectedTop1 by independent3Dgeometry;
this auditor independently recomputes finaltopk/means instead of rerunning a model.
"""
from pathlib import Path
import json,hashlib,math
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[2];RUN=ROOT/'crossplatform/round6/waymo_source100';MON=R/'monitoring';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def main():
 all10=read(MON/'ALL_TEN_MONITORING_COMPLETE.json');assert [x['epoch'] for x in all10['epochs']]==list(range(10,101,10));training=read(RUN/'TRAINING_COMPLETE.json');assert training['primary_epoch']==100 and training['completed_optimizer_steps']==15300 and training['no_joint_checkpoint_initialization'] and training['no_Drone_or_Quad_data_access'];primary=all10['epochs'][-1];assert primary['checkpoint_sha256']==training['primary_checkpoint_sha256'];pred=Path(primary['prediction_dir']);score=Path(primary['score_dir']);predseal=read(pred/'sealed.json');ss=read(score/'SEALED.json');assert sha(score/'SEALED.json')==primary['score_seal_sha256'];assert sha(score/'RESULTS_PRIVATE.json')==ss['files']['RESULTS_PRIVATE.json'];j=read(score/'RESULTS_PRIVATE.json');assert j['epoch']==100 and j['n']==3782;assert len(j['rows'])==3782;rows={x['anno_index']:x for x in j['rows']};assert set(rows)==set(range(3782));totals={};worst=0.
 for i in range(3782):
  fp=pred/f'{i:06d}.npz';ip=score/f'{i:06d}_IOU_PRIVATE.npz';assert sha(fp)==predseal['files'][fp.name] and sha(ip)==ss['files'][ip.name]
  with np.load(fp,allow_pickle=False) as z,np.load(ip,allow_pickle=False) as values:
   for h,head in enumerate(['last_','proposal_']):
    v=values[head]
    for mode in ['official_bbf','official_bbs','text_bbf','text_bbs']:
     arm=('official_full_' if mode.startswith('official') else 'text_full_')+head+mode.split('_')[-1];top=z[mode+'_top10'][h];a=rows[i]['arms'][arm];assert int(top[0])==a['query'];u=float(v[int(top[0])]);worst=max(worst,abs(u-a['iou']));assert u==a['iou'];group=totals.setdefault(arm,{'selected':[],'acc':{str(k):{25:0,50:0} for k in [1,5,10,256]},'oracle':[]});group['selected'].append(u);group['oracle'].append(float(v.max()))
     for k in [1,5,10,256]:
      mx=max(float(v[int(q)]) for q in top[:k]) if k<256 else max(float(x) for x in v)
      for t in [25,50]:
       good=mx>t/100;group['acc'][str(k)][t]+=good;assert good==a['topk'][str(k)]['acc'+str(t)]
 summary={x['arm']:x for x in j['summaries']};result=[]
 for arm,x in totals.items():
  n=len(x['selected']);avg=math.fsum(x['selected'])/n;pool=math.fsum(x['oracle'])/n;reference=summary[arm];difference=abs(avg-reference['mean_iou']);worst=max(worst,difference);assert difference<=1e-14 and abs(pool-reference['mean_candidate_oracle_iou_GT_assisted'])<=1e-14
  for k in [1,5,10,256]:
   for t in [25,50]:assert x['acc'][str(k)][t]==reference['topk'][str(k)]['acc'+str(t)]
  result.append({'arm':arm,'n':n,'independently_reduced_mean_iou':avg,'independently_reduced_topk':x['acc'],'mean_abs_difference_from_main':difference})
 payload={'state':'final_epoch100_source_result_independently_reduced','primary_epoch':100,'n':3782,'completed_optimizer_steps':15300,'all10_monitor_epochs_verified':True,'all10_score_seals_sha256':[x['score_seal_sha256'] for x in all10['epochs']],'training_complete_sha256':sha(RUN/'TRAINING_COMPLETE.json'),'primary_checkpoint_sha256':training['primary_checkpoint_sha256'],'primary_prediction_seal_sha256':sha(pred/'sealed.json'),'primary_score_seal_sha256':sha(score/'SEALED.json'),'summary':result,'max_reduction_delta':worst,'independent_selected3D_verification_from_frozen_scorer':j['independent_selected_audit'],'no_additional_geometry_model_GPU_GT_reads':True,'target_domains_read':False,'no_score_based_primary_selection':True,'auditor_code_sha256':sha(Path(__file__))};(MON/'FINAL_INDEPENDENT_REDUCTION.json').write_text(json.dumps(payload,indent=2));print(json.dumps({'state':payload['state'],'primary_checkpoint_sha256':payload['primary_checkpoint_sha256'],'max_reduction_delta':worst},indent=2))
if __name__=='__main__':main()
