"""Independent secondary saved predictions/IoU/status/scene/negative audit."""
from pathlib import Path
import json,numpy as np,itertools
R=Path(__file__).resolve().parents[2];H=R/'research/round94';O=H/'score_solver336_attempt001';P=R/'crossplatform/round94/solver_repair336_attempt001';rd=lambda p:json.loads(p.read_text());T=rd(O/'ALL_TASK_ROWS_PRIVATE.json');su=rd(O/'TASK_SUMMARY.json')['ALL336'];arms=rd(O/'SCORING_RECEIPT.json')['arm_names'];new=rd(P/'SEALED.json')['arm_names'];Y={x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in rd(R/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')};old={x['id_PRIVATE']:x for x in rd(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json')}
for t in T:
 z=np.load(P/'samples_PRIVATE'/(t['id_PRIVATE']+'.npz'));q=dict(zip(new,map(int,z['scores'].argmax(1))))
 for k in arms:assert t['queries'][k]==(q[k] if k in new else old[t['id_PRIVATE']]['queries'][k]) and t['IoU'][k]==Y[t['id_PRIVATE']][t['queries'][k]]
 for k in new:
  rec=rd(P/'records_PRIVATE'/(t['id_PRIVATE']+'.json'))[k.rsplit('_',1)[0]];assert t['statuses'][k]['fallback']==(k.endswith('_bt') and not rec['accepted'])
for k in arms:
 vals=[]
 for p in ['drone','quad']:
  tt=[x for x in T if x['platform']==p];v=np.array([x['IoU'][k] for x in tt]);s=su[p][k];assert len(v)==168 and s['mIoU_percent']==float(v.mean()*100) and s['Acc25_percent']==float((v>.25).mean()*100) and s['Acc50_percent']==float((v>.5).mean()*100);vals.append(s['mIoU_percent'])
  for st in ['unknown','fallback','no_parameter_update']:assert s[st+'_n']==sum(x['statuses'][k][st] for x in tt)
 assert su['balanced_mIoU_percent'][k]==float(np.mean(vals))
for sc in rd(O/'ALL_SCENE_SUMMARIES_PRIVATE.json'):
 tt=[x for x in T if x['platform']==sc['platform'] and x['scene_PRIVATE']==sc['scene_PRIVATE']];assert len(tt)==sc['n']
 for k in arms:assert sc['mIoU_percent'][k]==float(np.mean([x['IoU'][k] for x in tt])*100)
neg=rd(O/'ALL_ORDERED_NEGATIVE_COMPARISONS.json');expected=[(t['row'],a,b,100*(t['IoU'][a]-t['IoU'][b])) for t in T for a,b in itertools.permutations(arms,2) if t['IoU'][a]-t['IoU'][b]<-1e-10];assert [(x['row'],x['arm'],x['reference'],x['delta_pp']) for x in neg]==expected
A=H/'independent_solver_scalar_audit_attempt001';A.mkdir();v={'state':'PASS_R94_SECONDARY_ALL_SCALAR_ROWS_SCENES_STATUSES_NEGATIVES','n':336,'task_values':336*len(arms),'ordered_negatives':len(neg),'new_GT_scores':0,'FINAL_access':False};(A/'AUDIT_RESULT.json').write_text(json.dumps(v,indent=2)+'\n');print(v)
