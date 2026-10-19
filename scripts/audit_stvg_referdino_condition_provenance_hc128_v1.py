"""Additional input/reference provenance checks; no new inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_referdino_condition_hc128_v1 import D,P,verify

def main():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import unpack_expert
 p=verify(True);assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==128;start=time.perf_counter();checks=dict(original_input_hash=0,original_four_RGB=0,untouched_query_content=0,native_initial_references=0,projected_reference=0);initial_mismatch=0;maxes=[];source_ms=[];all_selected=0
 for r in p['records']:
  assert sha(ROOT/r['prior_input'])==r['prior_sha256'];checks['original_input_hash']+=1;orig=torch.load(ROOT/r['prior_input'],map_location='cpu',weights_only=False);expert=unpack_expert(orig['expert']);old=load(D/'teacher0'/(r['alias']+'.pt'));new=load(D/'teacher1'/(r['alias']+'.pt'));s=load(D/'student1'/(r['alias']+'.pt'));inp=load(D/'inputs'/(r['alias']+'.pt'));native=old['native'];
  for key,v in expert['observations'].items():
   assert old['rgb_sha256'][r['positions8'].index(key[1])]==v['receipt']['rgb_sha256'];checks['original_four_RGB']+=1
  for out,ref in [(old['conditional'],inp['E'][r['positions8']]),(new['conditional'],s['fit']['final'][r['positions8']].numpy())]:
   tr=out['trace'];assert tr['native_tgt_hash']==native['trace']['native_tgt_hash'];checks['untouched_query_content']+=1;assert torch.equal(tr['initial'],native['trace']['initial']);checks['native_initial_references']+=1;b=torch.tensor(ref,dtype=torch.float32).clamp(1e-5,1-1e-5);assert torch.equal(b,tr['injected_reference']);checks['projected_reference']+=1
  ds=(s['fit']['final'].numpy()[:,:2]-inp['E'][:,:2])/np.maximum(inp['E'][:,2:],1e-12);v=np.linalg.norm(ds,axis=1);maxes.append(float(v.max()));source_ms.append(inp['source_seconds']);all_selected+=sum(new['conditional']['trace']['selected_is_intervened'])
 motion=dict(max_actual_center_shift_Ewh_quantiles=dict(zip(['min','q25','median','q75','q95','max'],np.quantile(maxes,[0,.25,.5,.75,.95,1]).tolist())),queries_max_center_shift_gt_001=sum(x>.01 for x in maxes),queries_max_center_shift_gt_005=sum(x>.05 for x in maxes),queries_max_center_shift_gt_01=sum(x>.1 for x in maxes))
 put(P/'INPUT_PROVENANCE_AUDIT.json',dict(status='pass',checks=checks,S1_motion=motion,source_forward_seconds_sum=float(sum(source_ms)),CPU_seconds=time.perf_counter()-start,confirmation32_not_accessed=True,reference_and_query_content_audit_no_GT=True,no_GPU=True));assert not torch.cuda.is_initialized();print(checks,motion,flush=True)
if __name__=='__main__':main()
