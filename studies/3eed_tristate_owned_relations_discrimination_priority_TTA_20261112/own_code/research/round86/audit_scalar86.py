"""Independent cached-IoU and full-denominator scalar verification."""
from pathlib import Path
import json,hashlib,itertools,collections
import numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round86';P=R/'crossplatform/round86/prediction366_attempt001'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
seal=read(P/'SEALED.json');rows=read(H/'experiment/ALL366_PRIVATE.json');natives=read(H/'independent_native_audit_attempt001/AUDIT_RESULT.json');assert natives['prediction_seal_sha256']==sha(P/'SEALED.json');ys={}
for d in ['research/round55/score_dev336_attempt001','research/round69/score_all2304_attempt001']:ys.update({x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in read(R/d/'ALL_CANDIDATE_IOUS_PRIVATE.json')})
arms=seal['arm_names'];seen=set();counts={};totalneg=0
for part,n in [('dev336',336),('supp30',30)]:
 root=H/('score_'+part+'_attempt001');rr=read(root/'ALL_TASK_ROWS_PRIVATE.json');summary=read(root/'TASK_SUMMARY.json')['ALL'+str(n)];neg=read(root/'ALL_ORDERED_NEGATIVE_COMPARISONS.json');scene=read(root/'ALL_SCENE_SUMMARIES_PRIVATE.json');assert len(rr)==n
 for r in rr:
  sid=r['id_PRIVATE'];assert sid not in seen;seen.add(sid);z=np.load(P/'samples_PRIVATE'/(sid+'.npz'),allow_pickle=False);assert z['arm_names'].tolist()==arms
  for k,a in enumerate(arms):assert r['queries'][a]==int(z['scores'][k].argmax()) and r['IoU'][a]==ys[sid][r['queries'][a]]
 expected=sum(r['IoU'][a]-r['IoU'][b]<-1e-10 for r in rr for a,b in itertools.permutations(arms,2));assert len(neg)==expected;totalneg+=expected
 for a in arms:
  means=[]
  for pl in ['drone','quad']:
   vv=np.array([r['IoU'][a] for r in rr if r['platform']==pl]);s=summary[pl][a];assert s['n']==len(vv) and s['mIoU_percent']==float(vv.mean()*100) and s['Acc25_percent']==float((vv>.25).mean()*100) and s['Acc50_percent']==float((vv>.5).mean()*100);means.append(s['mIoU_percent'])
  assert float(np.mean(means))==summary['balanced_mIoU_percent'][a]
 for sc in scene:
  subset=[r for r in rr if r['platform']==sc['platform'] and r['scene_PRIVATE']==sc['scene_PRIVATE']];assert len(subset)==sc['n']
  for a in arms:assert sc['mIoU_percent'][a]==float(np.mean([r['IoU'][a] for r in subset])*100)
 counts[part]={'n':n,'task_values':n*len(arms),'ordered_negatives':expected}
assert seen=={r['id'] for r in rows} and counts['dev336']['n']==336
out=H/'independent_scalar_audit_attempt001';out.mkdir();result={'state':'PASS_R86_ALL366_FIXED_DENOMINATORS_CACHED_IOU_JOINS_PLATFORM_SCENE_AND_NEGATIVE_ROWS','n':366,'main_n':336,'supplement_n':30,'counts':counts,'task_values':366*len(arms),'ordered_negatives':totalneg,'new_GT_scores':0,'FINAL_access':False,'all_failures_unknowns_and_no_updates_retained':True};(out/'AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
