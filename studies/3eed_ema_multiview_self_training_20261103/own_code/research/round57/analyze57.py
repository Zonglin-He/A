"""Postseal analysis of every registered arm; no selection of a best hyperparameter."""
import time
from common57 import *
H=ROOT/'research/round57';P=ROOT/'crossplatform/round57/adapted336_attempt001';score=H/'score_dev336_attempt001';receipt=read(H/'saved_scalar_audit_attempt001/AUDIT_RESULT.json');assert receipt['state'].startswith('PASS_');tasks=read(score/'ALL_TASK_ROWS_PRIVATE.json');summary=read(score/'TASK_SUMMARY.json')['ALL336'];scenes=read(score/'ALL_SCENE_SUMMARIES_PRIVATE.json');overall=[s for s in scenes if s['scope']=='ALL336'];scenesafe=[];fits=[]
for pl in ['drone','quad']:
 ss=sorted([s for s in overall if s['platform']==pl],key=lambda s:s['scene_PRIVATE'])
 for idx,s in enumerate(ss):
  for arm in ARMS:scenesafe.append({'platform':pl,'scene_cluster':idx+1,'n':s['n'],'arm':arm,'mIoU_percent':s['mIoU_percent'][arm],'gain_vs_source_pp':s['mIoU_percent'][arm]-s['mIoU_percent']['source'],'gain_vs_frozen_ABC_pp':s['mIoU_percent'][arm]-s['mIoU_percent']['frozen_ABC']})
for i,t in enumerate(tasks):
 rec=read(P/'cases_PRIVATE'/(t['id_PRIVATE']+'.json'))
 for kind in ['fixed','ema']:
  for k in [1,3,16]:
   tr=rec['episodic'][kind]['trace'][k-1];fits.append({'row':f'DEV336_{i+1:05d}','platform':t['platform'],'protocol':'episodic','teacher':kind,'teacher_views':'AB','step':k,'KL_before':tr['KL_before'],'KL_after_same_detached_target':tr['KL_after_same_detached_target'],'gradient_norm':tr['gradient_norm'],'teacher_step_L2':tr['teacher_step_L2']})
 for key,r in rec['online'].items():
  kind,aug=key.split('_');fits.append({'row':f'DEV336_{i+1:05d}','platform':t['platform'],'protocol':'online','teacher':kind,'teacher_views':aug,'step':r['optimizer_step'],'KL_before':r['KL_before'],'KL_after_same_detached_target':r['KL_after_same_detached_target'],'gradient_norm':r['gradient_norm'],'teacher_step_L2':r['teacher_step_L2']})
metrics=[]
for arm in ARMS:
 metrics.append({'arm':arm,'Drone_mIoU_percent':summary['drone'][arm]['mIoU_percent'],'Quad_mIoU_percent':summary['quad'][arm]['mIoU_percent'],'balanced_mIoU_percent':summary['balanced_mIoU_percent'][arm],'balanced_Acc25_percent':float(np.mean([summary[p][arm]['Acc25_percent'] for p in ['drone','quad']])),'balanced_Acc50_percent':float(np.mean([summary[p][arm]['Acc50_percent'] for p in ['drone','quad']])),'gain_vs_source_pp':summary['balanced_mIoU_percent'][arm]-summary['balanced_mIoU_percent']['source'],'gain_vs_frozen_ABC_pp':summary['balanced_mIoU_percent'][arm]-summary['balanced_mIoU_percent']['frozen_ABC'],'improve_vs_source':sum(summary[p][arm]['improved_vs_source'] for p in ['drone','quad']),'worsen_vs_source':sum(summary[p][arm]['worsened_vs_source'] for p in ['drone','quad']),'query_changes_vs_source':sum(summary[p][arm]['query_changes_vs_source'] for p in ['drone','quad']),'scene_improve_vs_source':sum(s['mIoU_percent'][arm]>s['mIoU_percent']['source']+1e-10 for s in overall),'scene_worsen_vs_source':sum(s['mIoU_percent'][arm]<s['mIoU_percent']['source']-1e-10 for s in overall)})
factorial=[]
for aug in ['A','AB']:
 for who in ['student','teacher']:
  for timing in ['pre','post']:
   a=f'online_ema_{aug}_{who}_{timing}';b=f'online_fixed_{aug}_student_{timing}';factorial.append({'contrast':'EMA_'+who+'_minus_fixed_student','teacher_views':aug,'timing':timing,'delta_balanced_pp':summary['balanced_mIoU_percent'][a]-summary['balanced_mIoU_percent'][b]})
for kind in ['fixed','ema']:
 for who in (['student'] if kind=='fixed' else ['student','teacher']):
  for timing in ['pre','post']:
   a=f'online_{kind}_AB_{who}_{timing}';b=f'online_{kind}_A_{who}_{timing}';factorial.append({'contrast':'AB_minus_A_'+kind+'_'+who,'teacher_views':'AB_minus_A','timing':timing,'delta_balanced_pp':summary['balanced_mIoU_percent'][a]-summary['balanced_mIoU_percent'][b]})
cov=read(H/'prediction_audit_attempt001/ALL_ASSOCIATION_AND_GRADIENT_SCALARS_PUBLIC_SAFE.json');assoc={}
for pl in ['drone','quad']:
 rr=[r for r in cov if r['platform']==pl];assoc[pl]={v:{'median_unknown':float(np.median([r[v+'_unknown'] for r in rr])),'min_unknown':min(r[v+'_unknown'] for r in rr),'max_unknown':max(r[v+'_unknown'] for r in rr)} for v in ['B','C']}
analysis={'n':336,'arm_metrics':metrics,'online_factorial_contrasts':factorial,'association':assoc,'episodic_nonzero_first_gradient_cases':sum(r['epi_ema_first_gradient']>0 for r in cov),'identical_input_zero_gradient_controls':read(P/'IDENTICAL_INPUT_ZERO_GRADIENT_CONTROLS.json'),'no_new_hyperparameter_selection':True,'single_stream_order_only':True,'historically_exposed_development_not_independent_confirmation':True}
dump(H/'ANALYSIS_SUMMARY.json',analysis);dump(H/'ALL_FIT_DIAGNOSTICS_PUBLIC_SAFE.json',fits);dump(H/'ALL_SCENE_METRICS_PUBLIC_SAFE.json',scenesafe)
print(json.dumps({'metrics':metrics,'factorial':factorial,'association':assoc}),flush=True)
