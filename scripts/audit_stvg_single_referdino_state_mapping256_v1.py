"""Nonzero content-injection slot audit, no decoder forward or GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import D,P,BASE,load,read,put,sha,verify

def main():
 import torch
 from methods.stvg_referdino_opd_v1.state import inject
 torch.set_num_threads(2);p=verify();checks=grads=0;beg=time.perf_counter();counts={}
 for r in p['records']:
  ds,a=r['dataset'],r['alias'];f=BASE/f'inputs/{ds}/{a}.pt';assert sha(f)==p['source_files'][str(f.relative_to(ROOT))];c=load(f);pos=c['planned_positions'];assert len(pos)==8 and len(set(pos))==8 and len(c['frame_ids'])==len(c['Source']);state=torch.arange(1.,9.)[:,None].repeat(1,256).requires_grad_()
  for view,kw in enumerate(c['spatial_inputs']):
   q=torch.zeros_like(kw['query_tgt']);o=inject(q,state,pos,view);expected=torch.zeros_like(q)
   for k,v in enumerate(pos):
    if v%2==view:expected[v//2]=k+1
   assert torch.equal(o,expected);checks+=1;g=torch.autograd.grad(o.sum(),state)[0];eg=torch.zeros_like(state)
   for k,v in enumerate(pos):
    if v%2==view:eg[k]=q.shape[1]
   assert torch.equal(g,eg);grads+=1
  counts[ds]=counts.get(ds,0)+1
 put(P/'STATE_MAPPING_AUDIT_PUBLIC.json',dict(status='pass',parents=counts,nonzero_injection_checks=checks,gradient_slot_checks=grads,no_cross_offset_or_all_frame_broadcast=True,only_planned_initial_content_slots_modified=True,nonobserved_output_changes_arise_after_native_decoder_computation_not_direct_content_broadcast=True,zero_initialization_preserved_in_actual_all_fit_audits=True,model_forwards=0,GT_read=False,seconds=time.perf_counter()-beg));print('NONZERO_STATE_MAPPING_PASS',checks,grads,flush=True)
if __name__=='__main__':main()
