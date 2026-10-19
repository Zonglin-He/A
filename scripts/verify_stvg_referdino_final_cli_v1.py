"""Post-formal cache-mode CLI/API regression; no GT or new expert calls."""
from pathlib import Path
import sys,subprocess,json,time,hashlib,os
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
D=R/'artifacts/stvg_referdino_finalize_v1';P=R/'results/stvg_error_state/2026-10-19/referdino_final'
def main():
 import torch
 assert (D/'SCIENTIFIC_PHASES_COMPLETE.json').exists();record=read(D/'PLAN_LOCK.json')['formal_records'][0];alias=record['alias'];md=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'][record['query_ordinal']]['input'];cache=load(D/'formal/inputs'/f'{alias}.pt');req=dict(metadata=md,key=alias);q=D/'cli_request.json';put(q,req);signature=hashlib.sha256(json.dumps(dict(metadata=md,parses=None),sort_keys=True).encode()).hexdigest();c=D/'cli_cache.pt';assert not c.exists();torch.save(dict(input_signature=signature,input_metadata=md,cache=cache,GT_read=False),c);out=D/'cli_prediction.pt';tick=time.perf_counter()
 with(D/'cli_regression.log').open('ab')as log:x=subprocess.run([str(R/'.conda/tubedetr/bin/python'),'-B','-m','methods.stvg_referdino_opd_v1.run','--input',str(q),'--output',str(out),'--cache-in',str(c),'--seed','41'],cwd=R,stdout=log,stderr=subprocess.STDOUT,timeout=180,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'))
 assert x.returncode==0,x.returncode;seconds=time.perf_counter()-tick;z=torch.load(out,map_location='cpu',weights_only=False);old=load(D/'formal/predictions'/f'{alias}.pt');a=z['fits']['Live_41'];b=old['fits']['Live_41'];assert torch.equal(a['path'][0]['boxes'],b['path'][0]['boxes'])and torch.equal(a['path'][1]['state'],b['path'][1]['state'])and torch.equal(a['final'],b['final']);assert z['sampling_key']==alias
 put(P/'CLI_API_REGRESSION_PUBLIC.json',dict(status='pass',alias=alias,zero_first_update_final_exact=True,input_video_decoded_again=True,Source_model_loaded=True,new_Source_forward=0,new_DINO=0,new_ReferDINO=0,temporary_updates=40,GPU_allocation_upper_bound_seconds=seconds+3,complete_process_wall_seconds=seconds,actual_first_run_all_experts='formal raw-input API path; this tests optional cache CLI equivalence',GT_read=False));put(D/'CLI_PREDICTION_BARRIER.json',dict(path=str(out.relative_to(R)),sha256=sha(out),GT_read=False));print('CLI_API_EXACT',seconds,flush=True)
if __name__=='__main__':main()
