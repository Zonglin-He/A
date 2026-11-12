"""Independent scalar recomputation, loss/task associations and complete negative tails."""
from core82 import *
import itertools,collections
P=B82/'prediction2640_attempt001';pin(read(H82/'experiment/SCORE_CODE_LOCK.json'));end=[];methods=[];scenes=[];diagnostics=[];traces=[];updates=[];tails=[];effects=[];alltasks=[];negative_n=checks=0
for part,n in [('DEV336',336),('VAL1280',1280),('TRAIN1024',1024)]:
 F=H82/('score_'+part.lower()+'_attempt001');ss=read(F/'SEALED.json')
 for f,h in ss['files'].items():assert sha(F/f)==h
 tasks=read(F/'ALL_TASK_ROWS_PRIVATE.json');summary=read(F/'TASK_SUMMARY.json')['ALL'+str(n)];neg=read(F/'ALL_ORDERED_NEGATIVE_COMPARISONS.json');assert [(r['row'],r['arm'],r['reference'],r['delta_pp']) for r in neg]==[(r['row'],a,b,100*(r['IoU'][a]-r['IoU'][b])) for r in tasks for a,b in itertools.permutations(ARMS82,2) if r['IoU'][a]-r['IoU'][b]<-1e-10];negative_n+=len(neg);alltasks+=tasks
 counts=collections.Counter(r['platform'] for r in tasks);flat=[]
 for pl in ['drone','quad']:
  rr=[r for r in tasks if r['platform']==pl]
  for arm in ARMS82:
   v=np.array([r['IoU'][arm] for r in rr]);s=summary[pl][arm];assert abs(v.mean()*100-s['mIoU_percent'])<1e-12 and (v>.25).mean()*100==s['Acc25_percent'] and (v>.5).mean()*100==s['Acc50_percent'];methods.append({'cohort':part,'platform':pl,'arm':arm,**s});checks+=3
  for j,scene in enumerate(sorted({r['scene_PRIVATE'] for r in rr})):
   rs=[r for r in rr if r['scene_PRIVATE']==scene]
   for arm in ARMS82:scenes.append({'cohort':part,'platform':pl,'scene_alias':('D' if pl=='drone' else 'Q')+'_S'+str(j+1).zfill(2),'n':len(rs),'arm':arm,'mIoU_percent':float(np.mean([r['IoU'][arm] for r in rs])*100)})
 for r in tasks:
  rec=read(P/'records_PRIVATE'/(r['id_PRIVATE']+'.json'))
  for wrong in [False,True]:
   a='bt_wrong' if wrong else 'bt_correct';raw='R80_raw_wrong' if wrong else 'R80_raw_correct';c=rec[a];old=c['raw_record'];e={k:v for k,v in c.items() if k not in ['attempts','raw_record']};e.update(row=r['row'],cohort=part,platform=r['platform'],arm=a);end.append(e)
   for t in c['attempts']:traces.append({'row':r['row'],'cohort':part,'platform':r['platform'],'arm':a,**t})
   for stage,arm in [('raw',raw),('backtracked',a)]:
    before=old.get('auxiliary_before',c['objective_before']) if stage=='raw' else c['objective_before'];after=old['auxiliary_after'] if stage=='raw' else c['auxiliary_after'];J0=old.get('objective_before',c['objective_before']) if stage=='raw' else c['objective_before'];J1=old['objective_after'] if stage=='raw' else c['objective_after'];held0=c['heldout_true_auxiliary_before'];held1=c['raw_heldout_true_auxiliary_after'] if stage=='raw' else c['heldout_true_auxiliary_after'];d=r['IoU'][arm]-r['IoU']['frozen_bi'];flat.append({'row':r['row'],'cohort':part,'platform':r['platform'],'stage':stage,'binding':'wrong' if wrong else 'correct','arm':arm,'aux_delta':after-before,'objective_delta':J1-J0,'heldout_true_delta':held1-held0,'IoU_delta_pp':100*d,'balanced_contribution_pp':50*d/counts[r['platform']],'improved':d>1e-10,'worsened':d< -1e-10,'unchanged':abs(d)<=1e-10,'parameter_step_L2':old['parameter_step_L2'] if stage=='raw' else c['parameter_step_L2']})
   effects.append({'row':r['row'],'cohort':part,'platform':r['platform'],'binding':'wrong' if wrong else 'correct','raw_gain_vs_frozen_pp':100*(r['IoU'][raw]-r['IoU']['frozen_bi']),'bt_gain_vs_frozen_pp':100*(r['IoU'][a]-r['IoU']['frozen_bi']),'bt_minus_raw_pp':100*(r['IoU'][a]-r['IoU'][raw]),'raw_harm_reduced':r['IoU'][raw]<r['IoU']['frozen_bi']-1e-10 and r['IoU'][a]>r['IoU'][raw]+1e-10,'new_harm_vs_frozen':r['IoU'][raw]>=r['IoU']['frozen_bi']-1e-10 and r['IoU'][a]<r['IoU']['frozen_bi']-1e-10})
 for binding in ['correct','wrong']:
  a='bt_'+binding;rr=[v for v in end if v['cohort']==part and v['arm']==a];hist=collections.Counter(v['alpha'] for v in rr)
  for alpha,num in sorted(hist.items()):updates.append({'cohort':part,'arm':a,'alpha':alpha,'beta':alpha/.5,'n':num})
  updates.append({'cohort':part,'arm':a,'n':n,'accepted':sum(v['accepted'] for v in rr),'fallback':sum(not v['accepted'] for v in rr),'reused_R81':sum(v['reused_completed_R81_correct_search'] for v in rr),'new_tail_accepted':sum(0<v['alpha']<.0078125 for v in rr),'step_L2_mean':float(np.mean([v['parameter_step_L2'] for v in rr])),'step_L2_min':min(v['parameter_step_L2'] for v in rr),'step_L2_max':max(v['parameter_step_L2'] for v in rr),'mean_objective_delta':float(np.mean([v['objective_after']-v['objective_before'] for v in rr])),'mean_heldout_true_delta':float(np.mean([v['heldout_true_auxiliary_after']-v['heldout_true_auxiliary_before'] for v in rr]))})
 for stage in ['raw','backtracked']:
  for binding in ['correct','wrong']:
   rr=[v for v in flat if v['stage']==stage and v['binding']==binding]
   for dimension in ['aux_delta','objective_delta','heldout_true_delta']:
    for descent in [True,False]:
     group=[v for v in rr if (v[dimension]<-1e-10)==descent];diagnostics.append({'cohort':part,'stage':stage,'binding':binding,'measure':dimension,'descent':descent,'n':len(group),'improved':sum(v['improved'] for v in group),'worsened':sum(v['worsened'] for v in group),'unchanged':sum(v['unchanged'] for v in group),'balanced_total_contribution_pp':sum(v['balanced_contribution_pp'] for v in group),'mean_aux_delta':float(np.mean([v['aux_delta'] for v in group])) if group else None,'mean_heldout_true_delta':float(np.mean([v['heldout_true_delta'] for v in group])) if group else None})
 for arm in ['bt_correct','bt_wrong']:
  for ref in ['frozen_bi','frozen_direct3','R80_raw_correct','R80_raw_wrong','bt_wrong','R59']:
   if arm==ref:continue
   ds=np.array([50*(r['IoU'][arm]-r['IoU'][ref])/counts[r['platform']] for r in tasks]);delta=np.array([r['IoU'][arm]-r['IoU'][ref] for r in tasks]);tails.append({'cohort':part,'arm':arm,'reference':ref,'gain_pp':float(ds.sum()),'improved':int((delta>1e-10).sum()),'worsened':int((delta< -1e-10).sum()),'query_changes':sum(r['queries'][arm]!=r['queries'][ref] for r in tasks),'largest_positive_contribution_pp':float(ds.max()),'excluding_largest_positive_original_denominator_pp':float(ds.sum()-max(0,ds.max()))})
 diagnostics+=[]
 for v in flat:v.pop('arm')
 dump(H82/(part+'_LOSS_TASK_ROWS.json'),flat)
dump(H82/'ALL_METHOD_ROWS.json',methods);dump(H82/'ALL_SCENE_ROWS.json',scenes);dump(H82/'ALL_ENDPOINT_SCALAR_ROWS.json',end);dump(H82/'ALL_LINESEARCH_ATTEMPT_ROWS.json',traces);dump(H82/'LOSS_TASK_CONTINGENCY_ROWS.json',diagnostics);dump(H82/'UPDATE_SUMMARY_ROWS.json',updates);dump(H82/'TASK_EFFECT_AND_TAIL_ROWS.json',tails);dump(H82/'RAW_TO_BACKTRACKING_CASE_EFFECT_ROWS.json',effects)
violations=sum(v['accepted'] and v['objective_after']>v['objective_before']+1e-10 for v in end);assert violations==0;O=H82/'independent_scalar_audit_attempt001';O.mkdir();result={'state':'PASS_R82_ALL_SCALARS_NEGATIVES_SCENES_ASSOCIATIONS_AND_ACCEPTANCE_SUMMARIES','n':len(alltasks),'task_values':len(alltasks)*len(ARMS82),'ordered_negatives':negative_n,'checks':checks,'updated_endpoints':len(end),'accepted_objective_increases':violations,'no_new_GT_models_media_or_geometry':True};dump(O/'AUDIT_RESULT.json',result);print(result)
