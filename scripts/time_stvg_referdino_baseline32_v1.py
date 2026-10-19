"""Correct pilot denominator: exclude DINO audit hashing from expert latency."""
import sys,time,gc,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_referdino_native32_v1 import D,P,PRE,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load

def main():
 import torch,numpy as np
 from scripts.run_final_simplification_v1 import lease
 from scripts.run_decota_paper_main_v1 import read_row,frames_for
 from scripts.decota_paper_common_v1 import guard
 from methods.decota_final_simplified_v1.observations import SpatialExpert,observations
 from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
 from methods.decota_final_simplified_v1.tensors import state_hash
 p=verify(True);sys.addaudithook(guard);torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;gate=read(P/'COST_GATE_PUBLIC.json');lock=lease();begin=time.perf_counter();rows=[];cache=collections.OrderedDict()
 try:
  expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);mh=state_hash(expert.model.state_dict())
  for j,a in enumerate(p['configuration']['pilots']):
   verify();r=next(r for r in p['records']if r['alias']==a);row=read_row(r['dataset'],r['query_ordinal']);frames,ids,_=frames_for(r['dataset'],row,cache);prior=torch.load(ROOT/r['prior_input'],map_location='cpu',weights_only=False)
   if j==0:
    torch.cuda.synchronize();t=time.perf_counter();ex=observations(expert,row['parses'],frames,ids,prior['indices'],audit=False);torch.cuda.synchronize();warm=time.perf_counter()-t
   torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();ex=observations(expert,row['parses'],frames,ids,prior['indices'],audit=False);torch.cuda.synchronize();elapsed=time.perf_counter()-t;assert ex['new_DINO']==4;old=load(PRE/'source'/(a+'.pt'))['expert'];maxerr=0.
   for key,v in ex['observations'].items():
    pr=v['probe'];op=old['observations'][key]['probe'];assert pr['candidate_ids']==op['candidate_ids']and pr['accepted']==op['accepted'];maxerr=max(maxerr,float(np.max(abs(np.asarray(pr['boxes'])-np.asarray(op['boxes'])))),float(np.max(abs(np.asarray(pr['target_scores'])-np.asarray(op['target_scores'])))))
   assert maxerr<1e-6;ref=next(v['Refer_expert_seconds']for v in gate['rows']if v['alias']==a);rows.append(dict(alias=a,dataset=r['dataset'],T_expert_seconds_without_audit_hashing=elapsed,T_forward_seconds=sum(x['receipt']['forward_seconds']for x in ex['observations'].values()),Refer_same_sealed_expert_seconds=ref,ratio=ref/elapsed,T_peak_allocated_bytes=torch.cuda.max_memory_allocated(),max_numerical_error_vs_original=maxerr));cache.clear();del frames,ex;gc.collect()
  assert state_hash(expert.model.state_dict())==mh;ratio=sum(r['Refer_same_sealed_expert_seconds']for r in rows)/sum(r['T_expert_seconds_without_audit_hashing']for r in rows);assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['GPU_allocation_seconds']+time.perf_counter()-begin+3<900;verify(True)
  put(P/'TIMING_CORRECTION_PUBLIC.json',dict(status='pass'if ratio<=1.5 else'failed_cost_gate',ratio=ratio,limit=1.5,rows=rows,T_warmup_seconds=warm,extra_T_forwards=20,new_Refer_queries=0,new_Source_OPD_training=0,GPU_allocation_seconds=time.perf_counter()-begin+3,GT_read=False,weights_unchanged=True,reason='Original T audit=True included diagnostic input hashing in model prehook and expert timing; rerun exactly same four T queries audit=False, preserve original gate and all predictions. Ref timers already excluded outside RGB receipt hashing.',quality_not_used_for_correction=True,raw_RGB_decode_and_loading_excluded_both=True));print('CORRECTED_GATE',ratio,flush=True)
 finally:lock.close()
if __name__=='__main__':main()
