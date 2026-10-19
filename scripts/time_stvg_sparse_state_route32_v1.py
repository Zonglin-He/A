"""Post-result fixed pilot forward/backward timing; no fit/GT/new output choice."""
import os,sys,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_sparse_state_route32_v1 import D,P,CLOSED,FEAT,SRC,verify,read,put,load,sha

def main():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_opd_frozen_capture_cpu007 import transfer
 from vg_tta.stvg_sparse_state_route32_v1 import Decoder,box_loss
 p=verify(full=True);plan=read(D/'COST_TIMING_LOCK.json');assert sha(Path(__file__))==plan['timing_code_sha256'];lease=gpu();begin=time.perf_counter();remaining=p['GPU_cap_seconds']-read(D/'GPU_COST_RECEIPT.json')['GPU_allocation_seconds'];rows=[];model=None;dec=None
 def check():
  verify();assert time.perf_counter()-begin+3<min(remaining,plan['GPU_cap_seconds']);assert torch.cuda.mem_get_info()[0]>4*2**30
 try:
  for alias in plan['aliases']:
   rec=next(r for r in p['records']if r['alias']==alias);check()
   if model is not None:del model;gc.collect();torch.cuda.empty_cache()
   model=model_for('vidstg'if rec['dataset']=='hc2'else'hcstvg2');mh=state_hash(model.state_dict());original=load(D/'predictions'/(alias+'.pt'));replay=load(CLOSED/'replay_inputs'/(alias+'.pt'));inputs=transfer(replay['spatial_inputs'],'cuda');dec=Decoder(model.ground_decoder.decoder,inputs,original['Source'],original['E'],check);assert dec.scale==original['sigma'];m=torch.zeros(256,device='cuda',requires_grad=True)
   events=[torch.cuda.Event(enable_timing=True)for _ in range(4)];torch.cuda.synchronize();at=time.perf_counter();events[0].record();out=dec.raw(m,True);loss=box_loss(out['boxes'],original['positions'],original['targets']);events[1].record();torch.cuda.synchronize();forward_wall=time.perf_counter()-at
   at=time.perf_counter();events[2].record();g=torch.autograd.grad(loss,m)[0];events[3].record();torch.cuda.synchronize();backward_wall=time.perf_counter()-at;actual=g.detach().cpu().double().numpy();saved=original['fits']['Direct_anchor']['rounds'][0]['gradient'];err=float(abs(actual-saved).max());assert err<1e-7
   assert state_hash(model.state_dict())==mh;rows.append(dict(alias=alias,dataset=rec['dataset'],forward_plus_loss_wall_seconds=forward_wall,backward_wall_seconds=backward_wall,forward_plus_loss_CUDA_event_seconds=events[0].elapsed_time(events[1])/1000,backward_CUDA_event_seconds=events[2].elapsed_time(events[3])/1000,original_Direct_step0_gradient_max_error=err,actual_Decoder_trajectory_calls=dec.calls,PosDecoder_calls=dec.calls*2,backward_calls=1,no_update=True,no_GT=True));dec.close();dec=None;del inputs,replay,original,out,loss,g,m;gc.collect();torch.cuda.empty_cache()
  verify(full=True);check();put(P/'FORWARD_BACKWARD_TIMING_PUBLIC.json',dict(status='complete',fixed_aliases=plan['aliases'],rows=rows,post_result_engineering_measurement_only=True,no_model_parameter_or_state_update=True,no_GT_or_scoring=True,no_new_method_predictions=True,cost_not_whole_run_exact_decomposition=True));print('fixed2 forward/backward timing complete',rows,flush=True)
 finally:
  if dec is not None:dec.close()
  put(D/'TIMING_GPU_COST_RECEIPT.json',dict(status='released',GPU_allocation_seconds=time.perf_counter()-begin+3,pilot_count_completed=len(rows),no_fitting_or_GT=True));lease.close()
if __name__=='__main__':main()
