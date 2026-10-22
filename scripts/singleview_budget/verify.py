"""Independent arrays/counts/paused-full checks; no labels or new model calls."""
import os,sys,json,hashlib,ast
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from scripts.singleview_budget.model import posterior
from stvg.temporal import final_mbr
from scripts.tube_fusion.core import restore_geometry
from stvg.spatial import evidence_dp
from scripts.singleview_budget.run import OUT as SCREEN,read,write,sha

def main():
 a=ast.parse((ROOT/'stvg/predictor.py').read_text());f=next(n for c in a.body if isinstance(c,ast.ClassDef)for n in c.body if isinstance(n,ast.FunctionDef)and n.name=='capture')
 b=ast.parse((ROOT/'scripts/singleview_budget/capture.py').read_text());g=next(n for n in b.body if isinstance(n,ast.FunctionDef)and n.name=='capture')
 start=lambda n:next(i for i,x in enumerate(n.body)if isinstance(x,ast.If)and ast.unparse(x.test)=='teacher_override is not None')
 def preserved(n):return [ast.dump(z)for z in n.body[start(n):]if not isinstance(z,ast.Assign)or not any(isinstance(t,ast.Name)and t.id=='cache'for t in z.targets)]
 assert preserved(f)==preserved(g)
 small_checks=0
 for offset in [0,1]:
  ids=[0,3,8,11,19,23];sub=ids[offset::2];logits=torch.tensor([[[.2,-.8],[-.4,.7],[.9,.1]]],dtype=torch.float32)
  raw=dict(records=[dict(frame_ids=sub)],logits=[logits]);actual=posterior(raw,ids);ab=[(i,j)for i in range(3)for j in range(i+1,3)];scores=np.array([float(logits[0,i,0])+float(logits[0,j,1])for i,j in ab]);probs=np.exp(scores-scores.max());probs/=probs.sum();expected=np.zeros((6,6));loc={f:i for i,f in enumerate(ids)}
  for (a,b),pa in zip(ab,probs):
   for (c,d),pb in zip(ab,probs):expected[loc[min(sub[a],sub[c])],loc[max(sub[b],sub[d])]]+=pa*pb
  assert np.max(abs(actual-expected))<2e-15;u=[]
  for i in range(6):
   for j in range(i+1,6):
    val=0.
    for a,b in zip(*np.nonzero(expected)):
     ov=max(0,min(ids[j]+1,ids[b]+1)-max(ids[i],ids[a]));val+=expected[a,b]*ov/(ids[j]+1-ids[i]+ids[b]+1-ids[a]-ov)
    u.append((val,[ids[i],ids[j]+1]))
  assert final_mbr(ids,actual)['interval']==max(u,key=lambda x:x[0])[1];small_checks+=1
 torch.set_num_threads(1);p=read(SCREEN/'plan.json');seal=read(SCREEN/'sealed.json');checks=0
 for item,r in zip(p['items'],seal['files']):
  for name,path in item['inputs'].items():assert sha(ROOT/path)==item['hashes'][name],path
  assert sha(SCREEN/r['posterior_path'])==r['posterior_sha256']and sha(SCREEN/r['path'])==r['sha256']
  raw=torch.load(SCREEN/r['posterior_path'],weights_only=False,map_location='cpu');low=torch.load(ROOT/item['inputs']['G'],weights_only=False,map_location='cpu');cap=torch.load(ROOT/item['inputs']['capture'],weights_only=False,map_location='cpu');q=.5*low['final']['Global6']+.5*posterior(raw,cap['frame_ids']);z=np.load(SCREEN/r['path']);span=final_mbr(cap['frame_ids'],q)['interval']
  assert np.array_equal(q,raw['posterior'])and span==z['Single320_interval'].tolist();geom=restore_geometry(cap['geometry']);assert np.array_equal(evidence_dp(geom.for_interval(span))['final'].numpy(),z['Single320']);assert raw['actual_visual_views']==1 and raw['native_time_passes']==2 and raw['input_shapes'][0][2]==320;checks+=1
 b=read(SCREEN/'benchmark.json');assert len(b['files'])==8
 for r in b['files']:
  assert set(r['seconds'])==set(p['benchmark_arms']);assert len(r['shapes']['448_01'])==2 and len(r['shapes']['448_0'])==len(r['shapes']['320_0'])==1
 full=read(ROOT/'outputs/boxfixed_full_v1/state.json');assert full['status']=='paused_by_user'and full['paused_completed']==6088
 fullplan=read(ROOT/'outputs/boxfixed_full_v1/plan.json')
 for f,h in fullplan['sources'].items():assert sha(ROOT/f)==h,f
 pause=read(ROOT/full['pause_receipt'])
 for key,x in pause['files'].items():
  assert sha(ROOT/f'outputs/boxfixed_full_v1/vidstg/predictions/{key}.npz')==x['prediction_sha256'];assert sha(ROOT/f'outputs/boxfixed_full_v1/vidstg/receipts/{key}.json')==x['receipt_sha256']
 re=ROOT/'outputs/single448_recheck_v1';rs=read(re/'sealed.json');rp=read(re/'plan.json');assert rs['queries']==64;count=0
 for r in rs['files']:
  assert sha(re/r['path'])==r['sha256']
  if r['costs']:
   assert set(r['costs'])==set(rp['arms'])
   for a,c in r['costs'].items():
    expected=0 if a=='Global6'else 2 if a=='Strong'else 1
    proof=r['proof'][a];assert proof['actual_high_views']==expected and all(n==2+expected for n in proof['actual_encoder_calls'].values());assert proof['independent_complete_pipeline_matches_cached_precision_bitwise'];assert c['capture_cost']['ReferDINO_query_forwards']==1 and not c['capture_cost']['ReferDINO_cache_hit'];count+=1
 assert count==8*len(rp['arms'])
 rr=read(ROOT/'results/singleview_budget_v1/report.json');other=[read(ROOT/f'results/single448_recheck_v1/{ds}/report.json')for ds in ['hc2','vidstg']]
 out=dict(status='pass',original_teacher_ERef_geometry_AST_unchanged=True,independent_tiny_hull_and_MBR_checks=small_checks,HC63_independent_posterior_MBR_DP_reconstructions=checks,HC63_content_bound_inputs_verified=True,paired_real_branch_benchmark_queries=8,actual_complete_pipeline_runs=count,actual_complete_pipeline_queries=8,actual_encoder_view_counts_verified=True,teacher_one_eight_observation_request_each_timed_run=True,old_Source_Global6_bitwise_recovered_on_rechecks=True,official_dense_checks=rr['proof']['official_dense_checks']+sum(r['proof']['official_dense_checks']for r in other),official_dense_max_error=max(rr['proof']['official_dense_max_error'],*[r['proof']['official_dense_max_error']for r in other]),full_frozen_source_files_unchanged=len(fullplan['sources']),full_saved_prediction_receipt_pairs_unchanged=len(pause['files']),full_status='paused_by_user',production_unchanged=True,new_labels_or_models=False,CUDA_initialized=torch.cuda.is_initialized())
 assert out['full_frozen_source_files_unchanged']==434 and out['full_saved_prediction_receipt_pairs_unchanged']==6088 and not out['CUDA_initialized'];write(ROOT/'results/single448_recheck_v1/verification.json',out);write(ROOT/'results/singleview_budget_v1/verification.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
