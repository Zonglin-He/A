"""Fixed-arm ablations and postseal symmetric good/bad case selection."""
from core87 import *
P=B87/'prediction336_attempt001';tasks=read(H87/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');summary=read(H87/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];rows=read(H87/'experiment/DEV336_PRIVATE.json');method=[];comp=[];update=[];coverage=[];selected=[];reasons={}
for arm in ARMS87:
 method.append({'platform':'balanced','arm':arm,'mIoU_percent':summary['balanced_mIoU_percent'][arm]})
 for pl in ['drone','quad']:method.append({'platform':pl,'arm':arm,**summary[pl][arm]})
for mode in MODES87:
 for suffix in ['direct','tta1']:
  arm=mode+'_'+suffix;full='full_'+suffix;d=np.array([r['IoU'][full]-r['IoU'][arm] for r in tasks]);comp.append({'component_removed':mode,'readout':suffix,'full_mIoU_percent':summary['balanced_mIoU_percent'][full],'ablated_mIoU_percent':summary['balanced_mIoU_percent'][arm],'component_net_pp':float(d.mean()*100),'full_better_n':int((d>1e-10).sum()),'full_worse_n':int((d< -1e-10).sum())})
for r,t in zip(rows,tasks):
 rec=read(P/'records_PRIVATE'/(r['id']+'.json'))
 for mode,stats in rec['targets'].items():coverage.append({'row':t['row'],'platform':r['platform'],'mode':mode,**stats})
 for mode,tr in rec['variants'].items():
  for st in tr:update.append({'row':t['row'],'platform':r['platform'],'mode':mode,**{k:v for k,v in st.items() if k!='trials'}})
def take(indices,reason):
 for i in indices:
  if i not in selected:selected.append(i)
  reasons.setdefault(i,[]).append(reason)
for ref in ['frozen_bi','full_direct']:
 d=np.array([r['IoU']['full_tta1']-r['IoU'][ref] for r in tasks]);take([i for i in np.argsort(-d,kind='stable') if d[i]>1e-10][:2],'gain_vs_'+ref);take([i for i in np.argsort(d,kind='stable') if d[i]<-1e-10][:2],'harm_vs_'+ref)
v=np.array([r['IoU']['full_tta1'] for r in tasks]);take(np.argsort(-v,kind='stable')[:2].tolist(),'absolute_high_IoU');ys={r['id_PRIVATE']:r['candidate_IoUs_PRIVATE'] for r in read(ROOT/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')};take([i for i in np.argsort(v,kind='stable') if max(ys[tasks[i]['id_PRIVATE']])>.25 and v[i]<=.25][:2],'recoverable_low_IoU');assert len(selected)<=12
cases=[];pub=[]
for k,i in enumerate(selected):
 alias='CASE'+str(k+1).zfill(3);t=tasks[i];cases.append({'case':alias,'row':rows[i],'task':t,'reasons':reasons[i]});o={'case':alias,'platform':t['platform'],'selection_reasons':';'.join(reasons[i]),'new_identity_not_automatically_certified':True}
 for arm in ARMS87:o[arm+'_IoU']=t['IoU'][arm]
 pub.append(o)
sr=read(H87/'score_dev336_attempt001/ALL_SCENE_SUMMARIES_PRIVATE.json');ss=[]
for pl in ['drone','quad']:
 vals=sorted({r['scene_PRIVATE'] for r in sr if r['platform']==pl})
 for s in sr:
  if s['platform']!=pl:continue
  for arm in ARMS87:ss.append({'platform':pl,'scene_alias':pl[0].upper()+'_S'+str(vals.index(s['scene_PRIVATE'])+1),'n':s['n'],'arm':arm,'mIoU_percent':s['mIoU_percent'][arm]})
for k,v in [('ALL_METHOD_ROWS',method),('COMPONENT_EFFECT_ROWS',comp),('ALL_UPDATE_ROWS',update),('EVIDENCE_COVERAGE_ROWS',coverage),('GOOD_BAD_CASE_ROWS',pub),('ALL_SCENE_ROWS',ss)]:dump(H87/(k+'.json'),v)
dump(H87/'experiment/GOOD_BAD_CASES_PRIVATE.json',cases);print({'cases':len(cases),'all_component_rows':len(comp),'method_rows':len(method)})

# Cached geometric oracle used only for postseal error decomposition.
err=[]
for pl in ['drone','quad']:
 rr=[r for r in tasks if r['platform']==pl]
 for arm in ['source','frozen_bi','full_direct','full_tta1','wrong_tta1']:
  for threshold in [.25,.5]:
   covered=np.array([max(ys[r['id_PRIVATE']])>threshold for r in rr]);ok=np.array([r['IoU'][arm]>threshold for r in rr]);baseok=np.array([r['IoU']['frozen_bi']>threshold for r in rr]);err.append({'platform':pl,'arm':arm,'threshold':threshold,'n':len(rr),'oracle_candidate_covered':int(covered.sum()),'task_above_threshold':int(ok.sum()),'recoverable_ranking_failures':int((covered&~ok).sum()),'candidate_coverage_failures':int((~covered&~ok).sum()),'frozen_success_now_failure':int((baseok&~ok).sum()),'frozen_failure_now_success':int((~baseok&ok).sum()),'offline_GT_diagnostic_only':True})
dump(H87/'ERROR_DECOMPOSITION_ROWS.json',err)
