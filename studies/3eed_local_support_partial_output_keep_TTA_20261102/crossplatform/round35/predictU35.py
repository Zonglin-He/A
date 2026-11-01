"""Deferred three-head U inference. Requires root primary lock, no labels."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,time,numpy as np
from u35_helpers import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--lock',default=str(DEFAULT_LOCK));a=p.parse_args();check_u_lock(a.lock);verify(ROOT/'crossplatform/round33/sourceU128_attempt001');out=U_OUT;out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir()
 import torch
 torch.set_num_threads(4);torch.manual_seed(0);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 rr=U_rows();assert len(rr)==128 and all(r['stage']=='U' for r in rr);start=time.perf_counter();receipts=[]
 for pl in ['drone','quad']:
  B=[r for r in rr if r['platform']==pl];pp={};zz={};ps,zs=GPU_probabilities_and_logodds(B,source_head());pp['source']=np.stack([U_feature(r)['scores'] for r in B]);zz['source']=zs;sourceerr=float(abs(ps-pp['source']).max());assert sourceerr<2e-5
  for name in CONFIGS:pp[name],zz[name]=GPU_probabilities_and_logodds(B,state_for(pl,name))
  for j,r in enumerate(B):
   z=U_feature(r);prob=np.stack([pp[n][j] for n in ARMS]);ell=np.stack([zz[n][j] for n in ARMS]);q=prob.argmax(-1);assert int(q[0])==int(z['scores'].argmax());np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(ARMS),probability_scores=prob,exact_token_logodds=ell,queries=q,boxes=z['boxes'][q],source_q0=np.array(int(q[0])),source_probability_projection_replay=ps[j])
  receipts.append({'platform':pl,'n':len(B),'heads':CONFIGS,'source_initial_probability_replay_maxerror':sourceerr,'optimizer_steps':0,'all_complete_scores_saved':True});print('PREDICT_U35',pl,len(B),'seconds',round(time.perf_counter()-start,1),flush=True)
 dump(out/'INFERENCE_RECEIPTS.json',receipts);seal(out,'R35_all_three_locked_heads_U128_complete_predictions_before_Ulabels',n=128,heads=6,optimizer_steps=0,new_G_calls=0,root_primary_lock_sha256=sha(a.lock),R35_plan_sha256=sha(EXP/'PLAN_LOCK.json'),R35_training_seal_sha256=sha(BASE/'partial_keep_attempt001/SEALED.json'),confirmation_interface_sha256=sha(EXP/'CONFIRMATION_INTERFACE_LOCK.json'),seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated());print('PREDICT_U35_SEALED')
if __name__=='__main__':main()
