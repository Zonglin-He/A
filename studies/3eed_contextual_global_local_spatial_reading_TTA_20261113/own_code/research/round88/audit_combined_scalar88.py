from pathlib import Path
import json,numpy as np,itertools
R=Path(__file__).resolve().parents[2];H=R/'research/round88';P=R/'crossplatform/round88/prediction336_with_control_attempt001'
def read(p):return json.loads(p.read_text())
tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');su=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];sc=read(H/'score_dev336_attempt001/ALL_SCENE_SUMMARIES_PRIVATE.json');neg=read(H/'score_dev336_attempt001/ALL_ORDERED_NEGATIVE_COMPARISONS.json');ys={r['id_PRIVATE']:r['candidate_IoUs_PRIVATE'] for r in read(R/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')};arms=read(P/'SEALED.json')['arm_names'];assert len(tasks)==336
for r in tasks:
 a=np.load(P/'samples_PRIVATE'/(r['id_PRIVATE']+'.npz'),allow_pickle=False)
 for k,arm in enumerate(arms):assert r['queries'][arm]==int(a['scores'][k].argmax()) and r['IoU'][arm]==ys[r['id_PRIVATE']][r['queries'][arm]]
for arm in arms:
 values=[]
 for pl in ['drone','quad']:
  vv=np.array([r['IoU'][arm] for r in tasks if r['platform']==pl]);s=su[pl][arm];assert len(vv)==s['n']==168 and s['mIoU_percent']==float(vv.mean()*100) and s['Acc25_percent']==float((vv>.25).mean()*100) and s['Acc50_percent']==float((vv>.5).mean()*100);values.append(s['mIoU_percent'])
 assert su['balanced_mIoU_percent'][arm]==float(np.mean(values))
for s in sc:
 rr=[r for r in tasks if r['platform']==s['platform'] and r['scene_PRIVATE']==s['scene_PRIVATE']];assert len(rr)==s['n']
 for arm in arms:assert s['mIoU_percent'][arm]==float(np.mean([r['IoU'][arm] for r in rr])*100)
expected=sum(r['IoU'][a]-r['IoU'][b]<-1e-10 for r in tasks for a,b in itertools.permutations(arms,2));assert len(neg)==expected;O=H/'independent_scalar_audit_attempt001';O.mkdir();v={'state':'PASS_R88_ALL336_CACHED_IOU_JOIN_FIXED_DENOMINATORS_SCENE_THRESHOLDS_ALL_NEGATIVES','n':336,'main_n':336,'task_values':len(arms)*336,'ordered_negatives':len(neg),'new_GT_scores':0,'FINAL_access':False};(O/'AUDIT_RESULT.json').write_text(json.dumps(v,indent=2)+'\n');print(v)
