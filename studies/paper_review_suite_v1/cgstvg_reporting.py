"""Postseal CG reporting from saved metrics; no video/model/annotation read."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path.cwd();D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1'
sys.path.insert(0,str(R/'tools'))
from full_score import cluster
ARMS=['SourceNative','Source','SharedCorrection','Space','Full']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(prefix,stage):
 seal_path=D/(prefix+'_'+stage+'_sealed.json');score_path=D/(prefix+'_'+stage+'_score_receipt.json')
 assert seal_path.exists()and score_path.exists(),'Whole-stage sealed/scored metrics required before reporting'
 seal=read(seal_path);score=read(score_path);assert score['status']=='scored'and score['seal_sha256']==sha(seal_path)
 assert seal['counts']['queries']==(128 if stage=='screen'else 256)and not seal['GT_read']
 result={};checks=0
 for ds,files in seal['files'].items():
  pub=P/prefix/ds/stage;metrics=read(D/prefix/stage/ds/'metrics.json');rows=metrics['private_rows'];report=read(pub/'report.json')
  assert len(files)==len(rows)==(64 if stage=='screen'else 128)
  for arm in ARMS:
   for f in ['t','v','s']:assert abs(np.mean([x['values'][arm][f]for x in rows])-report['means'][arm][f])<2e-15
  costs=[];segments={}
  for e in files.values():
   for k,h in [('path','sha256'),('capture_path','capture_sha256'),('receipt_path','receipt_sha256')]:assert sha(D/e[k])==e[h];checks+=1
   costs.append(e);w=e['worker'];key=str(w['pid'])+'/'+str(w['startticks']);b=segments.setdefault(key,dict(queries=0,Source_forwards=0,Source_seconds=0.,pipeline_seconds=0.,recorded_expert_new_calls=0,expert_cache_hits=0))
   engineering=e['details'].get('engineering_verification_Source_seconds')
   extra=int(engineering is not None)
   b['queries']+=1;b['Source_forwards']+=e['source_forwards']-extra;b['Source_seconds']+=e['source_seconds']-(engineering or 0.);b['pipeline_seconds']+=e['pipeline_seconds'];b['recorded_expert_new_calls']+=e['expert_new_forwards'];b['expert_cache_hits']+=int(e['expert_cache_hit'])
   if extra:
    w7=read(D/'cgstvg_v7_screen_predict_child.json');k7=str(w7['pid'])+'/'+str(w7['startticks'])
    v=segments.setdefault(k7+'/Source_only_engineering_verification',dict(queries=0,Source_forwards=0,Source_seconds=0.,pipeline_seconds=None,recorded_expert_new_calls=0,expert_cache_hits=0))
    v['Source_forwards']+=1;v['Source_seconds']+=engineering
  assert sum(x['Source_forwards']for x in segments.values())==sum(x['source_forwards']for x in costs)
  assert abs(sum(x['Source_seconds']for x in segments.values())-sum(x['source_seconds']for x in costs))<1e-8
  comparisons={}
  pairs=[('Source','SourceNative'),('SharedCorrection','Source'),('Space','SharedCorrection'),('Full','Space'),('Full','Source')]
  for a,b in pairs:
   comparisons[a+'-'+b]=dict(paired=cluster(rows,a,b,fields=[('t',100),('v',100),('s',100)]),severe_t=sum(x['values'][a]['t']-x['values'][b]['t']<-.05 for x in rows),severe_v=sum(x['values'][a]['v']-x['values'][b]['v']<-.05 for x in rows),severe_union_once=sum(x['values'][a]['t']-x['values'][b]['t']<-.05 or x['values'][a]['v']-x['values'][b]['v']<-.05 for x in rows))
  summary=dict(dataset=ds,stage=stage,queries=len(rows),comparisons=comparisons,worker_segments=segments,cost_mean=dict(Source_seconds=float(np.mean([x['source_seconds']for x in costs])),pipeline_seconds=float(np.mean([x['pipeline_seconds']for x in costs]))),time_groups=report['time_groups'],expert_original_cost_means={k:float(np.mean([x['expert_cost'][k]for x in costs]))for k in ['expert_seconds','preprocess_seconds','forward_seconds','postprocess_seconds']},feedback_seconds_mean=float(np.mean([x['details']['feedback_seconds']for x in costs])),cost_scope='Per-query costs exclude video decoding/subject parsing/some initial loading. Expert original cost is retained on cache hits, not zero deployment cost. Source bridge repeats and original segment costs retained; failed/duplicate engineering attempts separately disclosed in recovery evidence. Nested component costs must not be summed.',CI='Pointwise parent-cluster bootstrap, no familywise correction. Fullsets historically exposed.',GT_scoring_added=0,source_seal_sha256=sha(seal_path),source_score_sha256=sha(score_path))
  (pub/'tails_timegroups_cost.json').write_text(json.dumps(summary,indent=2)+'\n')
  with(pub/'all_signed_severe_timegroups.tsv').open('w')as out:
   w=csv.writer(out,delimiter='\t',lineterminator='\n');w.writerow(['query','duration_group','comparison','delta_t_pp','delta_v_pp','delta_s_pp','severe_t','severe_v','severe_union_once'])
   for i,x in enumerate(rows,1):
    for a,b in pairs:
     z={f:100*(x['values'][a][f]-x['values'][b][f])for f in ['t','v','s']};w.writerow([f'Q{i:03d}',x['duration_group'],a+'-'+b,z['t'],z['v'],z['s'],z['t']< -5,z['v']< -5,z['t']< -5 or z['v']< -5])
  result[ds]=summary
 verification=dict(status='verified_postseal_report_from_already_scored_metrics',stage=stage,prefix=prefix,queries=sum(x['queries']for x in result.values()),hash_checks=checks,GT_scoring_added=0,model_or_expert_calls=0,seal_sha256=sha(seal_path),score_sha256=sha(score_path))
 (P/prefix/(stage+'_reporting_verification.json')).write_text(json.dumps(verification,indent=2)+'\n');print(json.dumps(verification))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prefix',default='cgstvg_v9');ap.add_argument('--stage',choices=['screen','confirm'],required=True);a=ap.parse_args();run(a.prefix,a.stage)
