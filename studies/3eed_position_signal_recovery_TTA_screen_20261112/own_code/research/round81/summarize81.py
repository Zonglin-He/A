"""Saved-only audit of all cells, negatives, recording effects and accepted steps."""
from core81 import *
import itertools
method=[];scenes=[];counts={};tasksall=[];endpoint=[];checks=0;trials=[]
for part,n,stage in [('DEV336',336,'dev'),('VAL1280',1280,'extension'),('TRAIN1024',1024,'extension')]:
 F=H81/('score_'+part.lower()+'_attempt001')
 if not F.exists():continue
 seal0=read(F/'SEALED.json')
 for f,h in seal0['files'].items():assert sha(F/f)==h
 tt=read(F/'ALL_TASK_ROWS_PRIVATE.json');su=read(F/'TASK_SUMMARY.json')['ALL'+str(n)];ng=read(F/'ALL_ORDERED_NEGATIVE_COMPARISONS.json');arms=read(F/'SCORING_RECEIPT.json')['arm_names'];assert len(tt)==n;tasksall+=tt;want=[(r['row'],a,b,100*(r['IoU'][a]-r['IoU'][b])) for r in tt for a,b in itertools.permutations(arms,2) if r['IoU'][a]-r['IoU'][b]<-1e-10];assert want==[(r['row'],r['arm'],r['reference'],r['delta_pp']) for r in ng];counts[part]={'n':n,'arms':len(arms),'task_values':n*len(arms),'ordered_negatives':len(ng)}
 for pl in ['drone','quad']:
  rr=[r for r in tt if r['platform']==pl]
  for arm in arms:
   v=np.array([r['IoU'][arm] for r in rr]);d=su[pl][arm];assert abs(v.mean()*100-d['mIoU_percent'])<1e-12 and np.mean(v>.25)*100==d['Acc25_percent'] and np.mean(v>.5)*100==d['Acc50_percent'];method.append({'cohort':part,'platform':pl,'arm':arm,**d});checks+=3
  for j,sc in enumerate(sorted({r['scene_PRIVATE'] for r in rr})):
   subset=[r for r in rr if r['scene_PRIVATE']==sc]
   for arm in arms:scenes.append({'cohort':part,'platform':pl,'scene_alias':('D' if pl=='drone' else 'Q')+'_S'+str(j+1).zfill(2),'arm':arm,'n':len(subset),'mIoU_percent':float(np.mean([r['IoU'][arm] for r in subset])*100)})
for stage in ['dev','extension']:
 f=H81/(stage.upper()+'_ENDPOINT_SCALAR_ROWS.json')
 if not f.exists():continue
 rows=read(f);coh={} if stage=='dev' else read(ROOT/'research/round69/input/COHORT_BY_ID_PRIVATE.json');manifest=[] if stage=='dev' else allrows80()[336:]
 for r in rows:
  part='DEV336' if stage=='dev' else coh[manifest[int(r['row'][1:])-1]['id']];endpoint.append({'cohort':part,**r})
 manifest0=target_rows() if stage=='dev' else manifest
 base=B81/('dev336_attempt001' if stage=='dev' else 'extension2304_attempt001')
 for i,r in enumerate(manifest0):
  part='DEV336' if stage=='dev' else coh[r['id']]
  for arm,rec in read(base/'records_PRIVATE'/(r['id']+'.json')).items():
   for j,trial in enumerate(rec['attempts']):trials.append({'cohort':part,'row':('D' if stage=='dev' else 'E')+str(i+1).zfill(5),'platform':r['platform'],'arm':arm,'attempt':j+1,**trial})
if (H81/'SIGNAL_CONTROL_ENDPOINT_SCALAR_ROWS.json').exists():
 coh=read(ROOT/'research/round69/input/COHORT_BY_ID_PRIVATE.json');manifest=allrows80()[336:]
 for r in read(H81/'SIGNAL_CONTROL_ENDPOINT_SCALAR_ROWS.json'):endpoint.append({'cohort':coh[manifest[int(r['row'][1:])-1]['id']],**r})
 trials+=read(H81/'SIGNAL_CONTROL_ATTEMPT_ROWS.json')
stats=[]
for part,pl,arm in sorted({(r['cohort'],r['platform'],r['arm']) for r in endpoint}):
 rr=[r for r in endpoint if (r['cohort'],r['platform'],r['arm'])==(part,pl,arm)];increases=sum(r['accepted'] and r['objective_after']>r['objective_before']+1e-10 for r in rr);assert increases==0;stats.append({'cohort':part,'platform':pl,'arm':arm,'scope_parameters':rr[0]['scope_parameters'],'n':len(rr),'accepted_n':sum(r['accepted'] for r in rr),'zero_parameter_step_n':sum(r['parameter_step_L2']==0 for r in rr),'query_changes':sum(r['query_changed'] for r in rr),'mean_alpha':float(np.mean([r['alpha'] for r in rr])),'median_parameter_L2':float(np.median([r['parameter_step_L2'] for r in rr])),'mean_heldout_mask_auxiliary_delta':float(np.mean([r['heldout_true_auxiliary_after']-r['heldout_true_auxiliary_before'] for r in rr])),'heldout_mask_auxiliary_improved_n':sum(r['heldout_true_auxiliary_after']<r['heldout_true_auxiliary_before'] for r in rr),'accepted_objective_increases':increases})
for n,x in [('ALL_METHOD_ROWS',method),('ALL_SCENE_ROWS',scenes),('ALL_ENDPOINT_SCALAR_ROWS',endpoint),('UPDATE_SUMMARY_ROWS',stats),('ALL_LINESEARCH_ATTEMPT_ROWS',trials)]:dump(H81/(n+'.json'),x)
O=H81/'independent_scalar_audit_attempt001';assert not O.exists();O.mkdir();extra=read(H81/'independent_signal_control_scalar_audit_attempt001/AUDIT_RESULT.json') if (H81/'SIGNAL_CONTROL_SUMMARY.json').exists() else None;out={'state':'PASS_R81_ALL_SAVED_TASK_SCALARS_NEGATIVES_RECORDINGS_AND_ARMIJO_STEPS','n':sum(v['n'] for v in counts.values()),'task_values':sum(v['task_values'] for v in counts.values())+(2304 if extra else 0),'base_table_values':sum(v['task_values'] for v in counts.values()),'supplemental_new_values':2304 if extra else 0,'supplemental_copied_references_not_added_to_evidence_count':6912 if extra else 0,'ordered_negatives':sum(v['ordered_negatives'] for v in counts.values())+(extra['new_control_ordered_negatives'] if extra else 0),'checks':checks+(extra['checks'] if extra else 0),'cohorts':counts,'accepted_objective_increases':0,'no_new_GT_geometry_or_models':True};dump(O/'AUDIT_RESULT.json',out);print(out)
