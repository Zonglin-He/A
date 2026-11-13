"""Independent array-index, platform/scene denominator, thresholds and all-negative audit."""
from pathlib import Path
import json,numpy as np,itertools,argparse
R=Path(__file__).resolve().parents[2];H=R/'research/round91';ap=argparse.ArgumentParser();ap.add_argument('--source',action='store_true');a=ap.parse_args();read=lambda p:json.loads(p.read_text());S=H/('score_source128_attempt001' if a.source else 'score_dev336_attempt001');P=R/'crossplatform/round91'/('source128_attempt001' if a.source else 'prediction336_attempt001');tasks=read(S/'ALL_TASK_ROWS_PRIVATE.json');su=read(S/'TASK_SUMMARY.json');arms=read(P/'SEALED.json')['arm_names'];neg=read(S/'ALL_ORDERED_NEGATIVE_COMPARISONS.json');assert len(tasks)==(128 if a.source else 336)
if not a.source:ys={r['id_PRIVATE']:r['candidate_IoUs_PRIVATE'] for r in read(R/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')}
for r in tasks:
 z=np.load(P/'samples_PRIVATE'/(r['id_PRIVATE']+'.npz'),allow_pickle=False);y=np.load(R/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id_PRIVATE']+'.npz'),allow_pickle=False)['ious'] if a.source else ys[r['id_PRIVATE']]
 for k,arm in enumerate(arms):assert r['queries'][arm]==int(z['scores'][k].argmax()) and r['IoU'][arm]==y[r['queries'][arm]]
for arm in arms:
 vv=[]
 for pl in (['waymo'] if a.source else ['drone','quad']):
  values=np.array([r['IoU'][arm] for r in tasks if r['platform']==pl]);x=su['ALL128'][arm] if a.source else su['ALL336'][pl][arm];assert len(values)==x['n']==(128 if a.source else 168);assert x['mIoU_percent']==float(values.mean()*100) and x['Acc25_percent']==float((values>.25).mean()*100) and x['Acc50_percent']==float((values>.5).mean()*100);vv.append(x['mIoU_percent'])
 if not a.source:assert su['ALL336']['balanced_mIoU_percent'][arm]==float(np.mean(vv))
for sc in read(S/'ALL_SCENE_SUMMARIES_PRIVATE.json'):
 rr=[r for r in tasks if r['platform']==sc['platform'] and r['scene_PRIVATE']==sc['scene_PRIVATE']];assert len(rr)==sc['n']
 for arm in arms:assert sc['mIoU_percent'][arm]==float(np.mean([r['IoU'][arm] for r in rr])*100)
expected=sum(r['IoU'][x]-r['IoU'][y]<-1e-10 for r in tasks for x,y in itertools.permutations(arms,2));assert len(neg)==expected;O=H/('independent_source_scalar_audit_attempt001' if a.source else 'independent_scalar_audit_attempt001');O.mkdir();out={'state':'PASS_R91_SAVED_IOU_JOIN_ALL_DENOMINATORS_THRESHOLDS_SCENES_NEGATIVES','n':len(tasks),'main_n':len(tasks),'task_values':len(arms)*len(tasks),'ordered_negatives':len(neg),'new_GT_scores':0,'FINAL_access':False};(O/'AUDIT_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
