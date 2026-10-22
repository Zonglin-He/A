"""Both complete domains seal before any new independent CPU evaluation."""
import os,sys,json,csv
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.singleview_budget.recheck import OUT,PUBLIC,validate,read,write,sha
from scripts.interval_pool.score import score_boxes,summarize,paired,signed

def aggregate(rr,arms):
 out={}
 for a in arms:
  vals=np.array([r['costs'][a]['actual_query_wall_seconds']for r in rr]);high=np.array([r['costs'][a]['capture_cost'].get('extra_temporal_branch_seconds',0.)for r in rr]);out[a]=dict(actual_end_to_end_seconds=dict(mean=float(vals.mean()),median=float(np.median(vals)),p90=float(np.quantile(vals,.9)),excluding_first_mean=float(vals[1:].mean()),all_queries=vals.tolist()),extra_temporal_branch_seconds=dict(mean=float(high.mean()),excluding_first_mean=float(high[1:].mean())),relative_end_to_end_reduction_vs_Strong=float(1-vals.mean()/np.mean([r['costs']['Strong']['actual_query_wall_seconds']for r in rr])))
 return out

def score():
 torch.set_num_threads(1);p=validate();seal=read(OUT/'sealed.json');assert seal['queries']==64 and not seal['GT_read'];assert not(PUBLIC/'summary.json').exists()
 for r in seal['files']:assert sha(OUT/r['path'])==r['sha256']
 for ds,items in p['items'].items():
  lab=p['labels'][ds];assert sha(ROOT/lab['path'])==lab['sha256'];labels={g['id']:g for g in map(json.loads,(ROOT/lab['path']).read_text().splitlines())};receipts=[r for r in seal['files']if r['dataset']==ds];rows=[];checks=0;error=0.
  for item,r in zip(items,receipts):
   assert r['id']==item['query_id'];z=np.load(OUT/r['path']);gt=labels[r['id']];truth={int(k):v for k,v in gt['boxes'].items()};values={};fixed={}
   for arm in ['Source',*p['arms']]:
    span=z[arm+'_interval'].tolist()
    for dest,boxes in [(values,z[arm]),(fixed,z['FixedSpatial'])]:
     dest[arm],e=score_boxes(boxes,item['record'],truth,span,gt['interval'],z['positions8'].tolist());error=max(error,e);checks+=1
   rows.append(dict(query=r['query'],parent=gt['parent'],values=values,fixed_spatial=fixed))
  assert len(rows)==len({r['parent']for r in rows})==32;names=['Source',*p['arms']];pairs=[(a,b)for a in p['arms']if a not in ['Strong','Global6']for b in ['Strong','Global6','Source']]+[('Strong','Global6'),('Global6','Source')];cost=aggregate([r for r in receipts if r['costs']],p['arms'])
  result=dict(status='completed',dataset=ds,queries=32,parents=32,exposure='Existing exposed DEVELOPMENT RECHECK; not independent generalization/equivalence test. Offset0 andoffset1 both fixed before recheck, never per-query selected.',means=summarize(rows,names,'values'),paired=paired(rows,pairs,'values'),fixed_spatial_means=summarize(rows,names,'fixed_spatial'),fixed_spatial_paired=paired(rows,pairs,'fixed_spatial'),costs=cost,cost_scope=p['timing'],signed_changes=signed(rows,pairs,'C'if ds=='hc2'else'V'),queries_anonymous=[{k:v for k,v in r.items()if k!='parent'}for r in rows],proof=dict(all64_times_arms_sealed_before_GT=True,seal_sha256=sha(OUT/'sealed.json'),labels_sha256=lab['sha256'],plan_fingerprint=p['fingerprint'],official_dense_checks=checks,official_dense_max_error=error,actual_teacher_forwards=4*len(p['arms']),actual_timing_queries=4,observations_per_forward=8,actual_complete_pipeline_bitwise_matches_cached_precision=True,old_Global6_and_Source_bitwise_recovered=True),production_changed=False)
  dest=PUBLIC/ds;dest.mkdir(parents=True,exist_ok=True);write(dest/'report.json',result)
  with(dest/'signed_queries.tsv').open('w')as f:
   ss=result['signed_changes'];w=csv.DictWriter(f,fieldnames=list(ss[0]),delimiter='\t');w.writeheader();w.writerows(ss)
  with(dest/'cost_queries.tsv').open('w')as f:
   rs=[dict(query=r['query'],method=a,order_index=r['order'].index(a),actual_end_to_end_seconds=r['costs'][a]['actual_query_wall_seconds'],extra_temporal_branch_seconds=r['costs'][a]['capture_cost'].get('extra_temporal_branch_seconds',0.),RGB_seconds=r['costs'][a]['RGB_seconds'],teacher_seconds=r['costs'][a]['capture_cost']['expert_current_wall_seconds'])for r in receipts if r['costs']for a in p['arms']];w=csv.DictWriter(f,fieldnames=list(rs[0]),delimiter='\t');w.writeheader();w.writerows(rs)
  print(json.dumps(dict(dataset=ds,means=result['means'],costs=cost),indent=2))
 write(PUBLIC/'summary.json',dict(status='completed',datasets=['hc2','vidstg'],arms=p['arms'],actual_accuracy_query_arm_results=64*len(p['arms']),actual_complete_timing_runs=8*len(p['arms']),production_changed=False,screen_gate=read(SCREEN/'decision.json')['gate']));write(PUBLIC/'protocol.json',{k:v for k,v in p.items()if k not in ['items','inputs','configs','labels','sources','assets']});state=read(OUT/'state.json');state['status']='scored_awaiting_publication';write(OUT/'state.json',state)
if __name__=='__main__':
 from scripts.singleview_budget.run import OUT as SCREEN
 score()
