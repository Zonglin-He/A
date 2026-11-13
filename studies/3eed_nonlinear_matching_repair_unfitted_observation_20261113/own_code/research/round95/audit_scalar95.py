"""Independent saved argmax/IoU/platform/scene/status/all-negative audit."""
from pathlib import Path
import json,numpy as np,itertools,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round95';P=R/'crossplatform/round95/prediction336_attempt001';read=lambda p:json.loads(p.read_text());O=H/'score_dev336_attempt001';tasks=read(O/'ALL_TASK_ROWS_PRIVATE.json');arms=read(P/'SEALED.json')['arm_names'];su=read(O/'TASK_SUMMARY.json')['ALL336'];ys={r['id_PRIVATE']:r['candidate_IoUs_PRIVATE'] for r in read(R/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')};statuses={r['id']:r['arm_statuses'] for r in read(P/'READOUT_STATUS_PRIVATE.json')};assert len(tasks)==336
for r in tasks:
 a=np.load(P/'samples_PRIVATE'/(r['id_PRIVATE']+'.npz'),allow_pickle=False)
 for i,k in enumerate(arms):assert r['queries'][k]==int(a['scores'][i].argmax()) and r['IoU'][k]==ys[r['id_PRIVATE']][r['queries'][k]] and r['statuses'][k]==statuses[r['id_PRIVATE']][k]
for k in arms:
 av=[]
 for pl in ['drone','quad']:
  rr=[r for r in tasks if r['platform']==pl];v=np.array([r['IoU'][k] for r in rr]);s=su[pl][k];assert len(v)==168 and s['n']==168 and s['mIoU_percent']==float(v.mean()*100) and s['Acc25_percent']==float((v>.25).mean()*100) and s['Acc50_percent']==float((v>.5).mean()*100);av.append(s['mIoU_percent'])
  for st in ['unknown','fallback','no_parameter_update']:assert s[st+'_n']==sum(r['statuses'][k][st] for r in rr)
 assert su['balanced_mIoU_percent'][k]==float(np.mean(av))
for sc in read(O/'ALL_SCENE_SUMMARIES_PRIVATE.json'):
 rr=[r for r in tasks if r['platform']==sc['platform'] and r['scene_PRIVATE']==sc['scene_PRIVATE']];assert len(rr)==sc['n']
 for k in arms:assert sc['mIoU_percent'][k]==float(np.mean([r['IoU'][k] for r in rr])*100)
neg=read(O/'ALL_ORDERED_NEGATIVE_COMPARISONS.json');ex=[(r['row'],a,b,100*(r['IoU'][a]-r['IoU'][b])) for r in tasks for a,b in itertools.permutations(arms,2) if r['IoU'][a]-r['IoU'][b]<-1e-10];assert [(r['row'],r['arm'],r['reference'],r['delta_pp']) for r in neg]==ex
A=H/'independent_scalar_audit_attempt001';A.mkdir();out={'state':'PASS_R95_SCALARS_ALL_ROWS_STATUSES_SCENES_THRESHOLDS_AND_NEGATIVES','n':336,'task_values':336*len(arms),'ordered_negatives':len(neg),'new_GT_scores':0,'FINAL_access':False};(A/'AUDIT_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
