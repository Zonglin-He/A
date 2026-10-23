"""Post-seal reliability/provenance diagnostics of EXISTING full predictions.

No prediction or setting selection; GT is permitted only in this offline audit.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,hashlib,csv
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools')]
import numpy as np
from full import RUN,PROVIDER,read,sha,atomic,identity,check
from full_score import cluster
from stvg.metrics import box_iou
from stvg.metrics_base import xyxy
D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1';p=read(D/'offline_plan.json');assert sha(Path(__file__))==p['code_sha256'];check();atomic(D/'offline_worker.json',identity());started=time.perf_counter();results={};inputs=[]
known={ds:set()for ds in ['hc2','vidstg']}
# Traceable prior development exposure ONLY; this is NOT a complete history.
f=PROVIDER/'data/evaluation/development.jsonl';inputs.append(dict(path=str(f),sha256=sha(f)))
for r in map(json.loads,f.read_text().splitlines()):known[r['dataset']].add(str(r['metadata']['source']))
f=RUN/'promotion_inputs.json';inputs.append(dict(path=str(f),sha256=sha(f)))
for r in read(f):known[r['dataset']].add(str(r['row']['metadata']['source']))
for ds in ['hc2','vidstg']:
 seal=read(RUN/ds/'sealed.json');assert seal['GT_read']is False and seal['queries']==(1902 if ds=='hc2'else 10303);j=read(RUN/'plan.json')['jobs'][ds];lp=PROVIDER/j['labels'];assert sha(lp)==j['labels_sha256'];truth={x['id']:x for x in map(json.loads,lp.read_text().splitlines())if x['dataset']==ds};records={r['id']:r for r in map(json.loads,(PROVIDER/j['manifest']).read_text().splitlines())};old=read(RUN/ds/'metrics.json')['private_rows'];rows=[];quality=[];dev=[]
 for i,x in enumerate(old,1):
  key=x['id'];r=records[key];g=truth[key];boxes={int(k):v for k,v in g['boxes'].items()};e=seal['files'][key];assert sha(RUN/ds/e['path'])==e['sha256']
  with np.load(RUN/ds/e['path'])as z:
   predicted=xyxy(z['native_boxes'],r['metadata']['width'],r['metadata']['height']);predicted=np.maximum(predicted,0)if ds=='hc2'else predicted;ids=z['frame_ids'];pos=z['positions8'];scores=[float(box_iou(predicted[a],boxes[int(ids[int(t)])]))for a,t in enumerate(pos)if int(ids[int(t)])in boxes];q=float(np.mean(scores))if scores else None
  group='unscorable_no_GT_anchor'if q is None else ['IoU_0_25','IoU_25_50','IoU_50_75','IoU_75_100'][min(int(q*4),3)];was=str(r['metadata']['source'])in known[ds];rows.append(dict(parent=x['parent'],values={a:x['values'][a]for a in ['Source','SharedInit','Interpolation','OursS','Full']},quality_group=group,known_prior_development=was));quality.append(dict(id=key,mean_anchor_iou=q,scorable_anchors=len(scores),group=group));dev.append(dict(id=key,parent=x['parent'],known_prior_development=was))
  if i%100==0:atomic(D/'offline_progress.json',dict(dataset=ds,completed=i,total=len(old),labels='offline_existing_full_only',no_new_prediction=True))
 groups={}
 for field in ['quality_group','known_prior_development']:
  groups[field]={}
  for value in sorted({r[field]for r in rows},key=str):
   rr=[r for r in rows if r[field]==value];groups[field][str(value)]=dict(queries=len(rr),parents=len({x['parent']for x in rr}),means={a:{f:float(np.mean([x['values'][a][f]for x in rr]))for f in ['v','t']}for a in ['Source','SharedInit','Interpolation','OursS','Full']},paired=cluster(rr,'Full','Source'))
 report=dict(dataset=ds,queries=len(rows),quality_groups=groups['quality_group'],known_exposure_groups=groups['known_prior_development'],registry_known_parents=len(known[ds]),registry_completeness='INCOMPLETE: union of only two explicitly saved registries; unlogged!=unseen. Full official outcomes already examined. Do not label complement independent unseen confirmation.',quality_scope='Offline average expert anchor IoU only where actual GT exists; unannotated/out-of-event anchors not assigned fake zero quality. No online gating/selection.',existing_distance_curve=read(P.parent/'simplepath_full_v2'/ds/'report.json')['distance_curve'],interpolation_reused_full_12205=True,interpolation_definition='Same raw8 boxes, center+logscale interpolation betweenanchors, E outsidecoverage, fixed Source nativeMBR time; not a new teacher call.',fingerprint=p['fingerprint']);atomic(P/'offline'/ds/'report.json',report);atomic(D/'offline'/ds/'private.json',dict(rows=rows,anchor_quality=quality,exposure=dev));results[ds]=report
 with(P/'offline'/ds/'anchor_quality_signed.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','mean_anchor_iou','scorable_anchors','group','known_prior_development','Full_minus_Source_t_pp','Full_minus_Source_v_pp'])
  for i,(q,r)in enumerate(zip(quality,rows),1):w.writerow([f'Q{i:05d}',q['mean_anchor_iou'],q['scorable_anchors'],q['group'],r['known_prior_development'],100*(r['values']['Full']['t']-r['values']['Source']['t']),100*(r['values']['Full']['v']-r['values']['Source']['v'])])
atomic(D/'offline_receipt.json',dict(status='completed_offline_existing_full',queries=12205,seconds=time.perf_counter()-started,registry_inputs=inputs,new_model=0,new_expert=0,new_prediction=0,production_unchanged=True,public_files={str(f.relative_to(R)):sha(f)for f in(P/'offline').rglob('*')if f.is_file()}));print('Offline reliability/exposure12205 done',flush=True)
