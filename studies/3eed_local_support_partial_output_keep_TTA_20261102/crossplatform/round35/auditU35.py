"""Deferred CPU64 replay of full U score vectors, no task annotation access."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import argparse,numpy as np
from u35_helpers import *
from audit35 import replay

def main():
 p=argparse.ArgumentParser();p.add_argument('--lock',default=str(DEFAULT_LOCK));a=p.parse_args();check_u_lock(a.lock);verify(U_OUT);rr=U_rows();records=[];total=0;maxp=0.;maxz=0.;diff=[]
 for pl in ['drone','quad']:
  states={'source':source_head(),**{n:state_for(pl,n) for n in CONFIGS}}
  for r in [r for r in rr if r['platform']==pl]:
   z=U_feature(r);v=load_npz(U_OUT/'predictions_PRIVATE'/(r['id']+'.npz'));assert v['arm_names'].tolist()==ARMS and v['probability_scores'].shape==v['exact_token_logodds'].shape==(4,256);assert np.array_equal(v['probability_scores'][0],z['scores']);assert np.array_equal(v['queries'],v['probability_scores'].argmax(-1));assert np.array_equal(v['boxes'],z['boxes'][v['queries']])
   for j,n in enumerate(ARMS):
    p0,ell=replay(z,states[n]);pe=float(abs(p0-v['probability_scores'][j]).max());ze=float(abs(ell-v['exact_token_logodds'][j]).max());maxp=max(maxp,pe);maxz=max(maxz,ze);assert pe<2e-5 and ze<5e-4,(r['alias'],n,pe,ze);same=int(p0.argmax())==int(v['queries'][j]);records.append({'alias':r['alias'],'platform':pl,'configuration':n,'probability_maxerror':pe,'exact_logodds_maxerror':ze,'CPU64_argmax_same':same});total+=1
    if not same:diff.append({'alias':r['alias'],'configuration':n,'official_GPU_query':int(v['queries'][j]),'CPU64_query':int(p0.argmax()),'official_GPU_kept_for_task_scoring':True})
 receipt={'state':'PASSED','n':len(rr),'complete_probability_and_exactlogodds_vectors_checked':total,'probability_maxerror':maxp,'exact_logodds_maxerror':maxz,'CPU64_argmax_differences':diff,'no_GPU_GT_training':True,'official_GPU_vectors_not_replaced':True,'prediction_seal_sha256':sha(U_OUT/'SEALED.json'),'root_primary_lock_sha256':sha(a.lock)};dump(EXP/'U128_CPU_REPLAY_ROWS.json',records);dump(EXP/'U128_CPU_AUDIT.json',receipt);print(receipt)
if __name__=='__main__':main()
