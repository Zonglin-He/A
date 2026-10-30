"""Real adapter update stage, in a separate process; consumes no GT references."""
import argparse,pathlib,json,hashlib,time,os,traceback,sys
sys.dont_write_bytecode=True
import numpy as np,torch
from algorithm import evaluate_cache,CONSTANTS
ROOT=pathlib.Path(__file__).resolve().parent
def digest(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--cohort',choices=['dev','confirm'],required=True);p.add_argument('--device',default='cuda');p.add_argument('--gpu-owner');a=p.parse_args()
 if a.device.startswith('cuda'):assert a.gpu_owner and os.environ.get('ROUND2_GPU_OWNER')==a.gpu_owner,'Unified flock and root GPU owner required'
 if a.cohort=='confirm':assert (ROOT/'DEV_DECISION_FROZEN.json').exists(),'Freeze development decision before confirmation execution'
 execution_lock=ROOT/'EXECUTION_LOCK.json';execution=json.loads(execution_lock.read_text())
 for name,h in execution['source_sha256'].items():assert digest(ROOT/name)==h,('Locked source changed',name)
 cache_status=json.loads((a.cache/'CACHE_STATUS.json').read_text());assert cache_status['state']=='sealed' and cache_status['execution_lock_sha256']==digest(execution_lock);seal=json.loads((a.cache/'CACHE_SEALED.json').read_text())
 for n,h in seal.items():assert digest(a.cache/n)==h
 rows=json.loads((ROOT/f'cohorts/{a.cohort}_inference.json').read_text())
 if cache_status.get('engineering_subset_first_locked_parent'):rows=rows[:1]
 a.output.mkdir(parents=True,exist_ok=False);protocol={'state':'running','engineering_subset_first_locked_parent':bool(cache_status.get('engineering_subset_first_locked_parent')),'cohort':a.cohort,'count':len(rows),'steps':[0,3],'objectives':['hm_only','hm_support'],'readout':'median-logit of adapted K views; zero-step adapted equals median baseline, not source-only','constants':CONSTANTS,'cache_seal_sha256':digest(a.cache/'CACHE_SEALED.json'),'execution_lock_sha256':digest(execution_lock),'GT_data_read':False,'adaptation_device':a.device,'timing_scope':'cache replay for adapter optimization; source forwards are separately measured, not a deploy end-to-end timing claim'};(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
 try:
  for i,row in enumerate(rows):
   d=np.load(a.cache/(row['id']+'.npz'),allow_pickle=False)
   for variant in ['hm_only','hm_support']:
    for k in [0,3]:
     if a.device.startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
     start=time.perf_counter();output,diag=evaluate_cache(d['view_hidden'],d['view_logits'],d['neighbors'],variant,k,a.device)
     if a.device.startswith('cuda'):torch.cuda.synchronize()
     assert np.array_equal(output['native_views'],d['view_probs']),'Replayed sigmoid differs from actual cached native probabilities in at least one FPS view'
     diag['all_five_native_cached_view_probabilities_bitexact']=True
     assert np.array_equal(output['source'],d['view_probs'][0])
     if k==0:assert np.array_equal(output['adapted'],output['median'])
     diag.update(seconds=time.perf_counter()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None,id=row['id'],parent_id=row['parent_id'])
     if 'legacy32_source' in d:output['legacy32_source']=d['legacy32_source']
     np.savez_compressed(a.output/f'{i:04d}_{variant}_k{k}.npz',**output);(a.output/f'{i:04d}_{variant}_k{k}.json').write_text(json.dumps(diag,indent=2))
   print(i,row['id'],flush=True)
  final={p.name:digest(p) for p in sorted(a.output.glob('*.npz'))};(a.output/'SEALED.json').write_text(json.dumps(final,indent=2));protocol['state']='sealed';(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
 except Exception as e:
  protocol.update(state='invalid',error=repr(e),traceback=traceback.format_exc());(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2));raise
if __name__=='__main__':main()
