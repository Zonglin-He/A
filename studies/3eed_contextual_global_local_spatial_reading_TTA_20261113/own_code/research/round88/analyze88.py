"""All method/component rows and symmetric cases, strictly after sealed scoring."""
from core88 import *
ARMS=ARMS88+['full_frozen3'];P=B88/'prediction336_with_control_attempt001';tasks=read(H88/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');su=read(H88/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];rows=read(H88/'experiment/DEV336_PRIVATE.json');method=[];components=[];updates=[];pairs=[];case_rows=[];cases=[];selected=[];reasons={};ys={r['id_PRIVATE']:r['candidate_IoUs_PRIVATE'] for r in read(ROOT/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')}
for arm in ARMS:
 method.append({'platform':'balanced','arm':arm,'mIoU_percent':su['balanced_mIoU_percent'][arm]})
 for pl in ['drone','quad']:method.append({'platform':pl,'arm':arm,**su[pl][arm]})
for mode in ['no_relation','no_local','wrong_space','role_swap']:
 for suffix in ['direct','tta1']:
  a='full_'+suffix;b=mode+'_'+suffix;d=np.array([t['IoU'][a]-t['IoU'][b] for t in tasks]);components.append({'component_removed':mode,'readout':suffix,'component_net_pp':float(d.mean()*100),'full_better_n':int((d>1e-10).sum()),'full_worse_n':int((d< -1e-10).sum())})
for arm in ['physical_only1','no_rank1','no_correspondence1','full_direct','full_frozen3','frozen_bi']:
 d=np.array([t['IoU']['full_tta1']-t['IoU'][arm] for t in tasks]);components.append({'component_removed':arm,'readout':'tta1','component_net_pp':float(d.mean()*100),'full_better_n':int((d>1e-10).sum()),'full_worse_n':int((d< -1e-10).sum())})
for r,t in zip(rows,tasks):
 rec=read(P/'records_PRIVATE'/(r['id']+'.json'));ec=load(B88/'evidence336_attempt001/samples_PRIVATE'/(r['id']+'.npz'));given=ec['rank_pairs'];assert len(given) or np.array_equal(load(P/'samples_PRIVATE'/(r['id']+'.npz'))['scores'][ARMS.index('full_tta1')],load(P/'samples_PRIVATE'/(r['id']+'.npz'))['scores'][ARMS.index('no_rank1')]);pairs.append({'row':t['row'],'platform':r['platform'],'partial_pairs':len(given),'pairs_higher_IoU':sum(ys[r['id']][i]>ys[r['id']][j]+1e-10 for i,j in given),'pairs_lower_IoU':sum(ys[r['id']][i]<ys[r['id']][j]-1e-10 for i,j in given),'pairs_equal_IoU':sum(abs(ys[r['id']][i]-ys[r['id']][j])<=1e-10 for i,j in given),'target_hypotheses':int((ec['hypotheses_PRIVATE'][0]>=0).sum()) if len(ec['hypotheses_PRIVATE']) else 0,'target_hypothesis_oracle_IoU':max([ys[r['id']][i] for i in ec['hypotheses_PRIVATE'][0] if i>=0],default=0.) if len(ec['hypotheses_PRIVATE']) else 0.,'source_oracle_IoU':max(ys[r['id']]),'postseal_descriptive_not_identity_gold':True})
 for variant,tr in rec.items():
  for st in tr:updates.append({'row':t['row'],'platform':r['platform'],'mode':variant,**{k:v for k,v in st.items() if k not in ['trials','parts_before','parts_after']},**{'before_'+k:v for k,v in st['parts_before'].items()},**{'after_'+k:v for k,v in st['parts_after'].items()}})
def take(ix,reason):
 for i in ix:
  if i not in selected:selected.append(i)
  reasons.setdefault(i,[]).append(reason)
for ref in ['frozen_bi','full_direct','full_frozen3']:
 d=np.array([r['IoU']['full_tta1']-r['IoU'][ref] for r in tasks]);take([i for i in np.argsort(-d,kind='stable') if d[i]>1e-10][:2],'gain_vs_'+ref);take([i for i in np.argsort(d,kind='stable') if d[i]<-1e-10][:2],'harm_vs_'+ref)
v=np.array([t['IoU']['full_tta1'] for t in tasks]);take(np.argsort(-v,kind='stable')[:1],'absolute_high');take([i for i in np.argsort(v,kind='stable') if max(ys[tasks[i]['id_PRIVATE']])>.25 and v[i]<=.25][:1],'recoverable_failure')
for k,i in enumerate(selected):
 alias='CASE'+str(k+1).zfill(3);t=tasks[i];cases.append({'case':alias,'row':rows[i],'task':t,'reasons':reasons[i]});case_rows.append({'case':alias,'platform':t['platform'],'reasons':';'.join(reasons[i]),**{a+'_IoU':t['IoU'][a] for a in ARMS}})
fixed=[]
for c in read(H87/'experiment/GOOD_BAD_CASES_PRIVATE.json'):
 i=next(i for i,r in enumerate(rows) if r['id']==c['row']['id']);t=tasks[i];fixed.append({'case':'R87_'+c['case'],'platform':t['platform'],**{a+'_IoU':t['IoU'][a] for a in ARMS}})
scene=[];sr=read(H88/'score_dev336_attempt001/ALL_SCENE_SUMMARIES_PRIVATE.json')
for pl in ['drone','quad']:
 vals=sorted({r['scene_PRIVATE'] for r in sr if r['platform']==pl})
 for s in [s for s in sr if s['platform']==pl]:
  for arm in ARMS:scene.append({'platform':pl,'scene_alias':pl[0].upper()+'_S'+str(vals.index(s['scene_PRIVATE'])+1),'n':s['n'],'arm':arm,'mIoU_percent':s['mIoU_percent'][arm]})
for k,v in [('ALL_METHOD_ROWS',method),('COMPONENT_EFFECT_ROWS',components),('ALL_UPDATE_ROWS',updates),('PARTIAL_PREFERENCE_AND_SUPPORT_ROWS',pairs),('GOOD_BAD_CASE_ROWS',case_rows),('FIXED_R87_CASE_ROWS',fixed),('ALL_SCENE_ROWS',scene)]:dump(H88/(k+'.json'),v)
dump(H88/'experiment/GOOD_BAD_CASES_PRIVATE.json',cases);gains={a:su['balanced_mIoU_percent']['full_tta1']-su['balanced_mIoU_percent'][a] for a in ['frozen_bi','full_direct','full_frozen3','R59','wrong_space_tta1','role_swap_tta1']};dump(H88/'DECISION.json',{'state':'R88_COMPLETED_CONTEXTUAL_SPATIAL_READING_DEVELOPMENT_EVALUATED','primary_arm':'full_tta1','primary_gains_pp':gains,'passes_matched_increment_checks':all(gains[a]>0 for a in ['frozen_bi','full_direct','full_frozen3']),'robust_adoption':False,'independent_confirmation':False,'FINAL_access':False,'source_training':False,'new_external_vision_models':False,'no_follow_on_queued':True});print({'gains_pp':gains,'cases':len(cases)})
