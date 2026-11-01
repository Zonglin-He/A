"""Independent CPU rank replay and native QP/memory arithmetic; no task GT."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import numpy as np
from scipy.special import erf
from common36 import *
def rank(u,state):
 x=u.astype(float)@state['0.weight'].astype(float).T+state['0.bias'];x=.5*x*(1+erf(x/np.sqrt(2)));return (x@state['2.weight'].astype(float).T+state['2.bias']).squeeze(-1)
def main():
 check_plan();out=BASE/'fit272_attempt001';verify(out);records=[];maxerr=0.;ties=[]
 for mode in MODES:
  source={k.removeprefix('rank.'):v for k,v in load_npz(source_path(mode)).items() if k.startswith('rank.')}
  for pl in ['drone','quad']:
   state=load_npz(out/'states_PRIVATE'/(pl+'_'+mode+'.npz'));lib=load_npz(out/'libraries_PRIVATE'/(pl+'_'+mode+'.npz'));A=[r for r in rows() if r['platform']==pl];ix={r['id']:j for j,r in enumerate(A)}
   for r in [r for r in rows()+rows('DEV64') if r['platform']==pl]:
    c=captured(r,mode);p=load_npz(out/'predictions_PRIVATE'/(r['id']+'_'+mode+'.npz'));s0=c['native_scores'].astype(float);q0=int(s0.argmax());st=s0+c['gate']*(rank(c['rankinput'],state)-p['source_rank0_used']);err=float(abs(st-p['native_scores'][1]).max());maxerr=max(maxerr,err);assert err<2e-5,(r['alias'],mode,err)
    sourceerr=float(abs(rank(c['rankinput'],source)-p['source_rank0_used']).max());assert sourceerr<2e-5
    if r['stage']=='Q':
     o=p['GPU_obs'];po=p['GPU_pos'];w=p['GPU_weights'];assert o.shape==po.shape==w.shape==(256,4);assert all(len(set(row))==4 for row in o);rr=(lib['raw'][o[:,:1],po[:,:1]]*w[:,:1]).sum(-1)/np.maximum(w[:,:1].sum(-1),1e-12);memory=s0+rr-rr[q0];assert np.allclose(rr,p['retrieved_raw'],atol=1e-12,rtol=0);Q=norm(c['rankinput']).astype(float);actual=np.einsum('qc,qkc->qk',Q,lib['keys'][o,po].astype(float));assert abs(actual-p['GPU_similarity']).max()<3e-6
    else:memory=s0+lib['raw'][ix[r['id']]]
    assert np.array_equal(p['native_scores'][0],s0);assert abs(memory-p['native_scores'][2]).max()<1e-12;assert p['source_q0']==q0;assert np.array_equal(p['queries'],np.array([grouped_readout(r,v,s0) for v in p['native_scores']]));assert np.array_equal(p['boxes'],feature(r)['boxes'][p['queries']]);cpuq=grouped_readout(r,st,s0);official=int(p['queries'][1]);
    if cpuq!=official:ties.append({'alias':r['alias'],'mode':mode,'CPU64_query':cpuq,'GPU_query':official,'official_GPU_not_replaced':True})
    records.append({'alias':r['alias'],'mode':mode,'platform':pl,'native_rank_replay_maxerror':err,'source_rank_replay_maxerror':sourceerr,'memory_arithmetic_exact':True})
 receipt={'state':'PASSED','sample_mode_rows':len(records),'native_vector_replay_maxerror':maxerr,'CPU64_query_differences':ties,'GPU_NN_indices_preserved_not_reselected':True,'all_original_feedback_kept':True,'no_GT_GPU_training':True,'prediction_seal_sha256':sha(out/'SEALED.json'),'plan_sha256':sha(EXP/'PLAN_LOCK.json')};dump(EXP/'CPU_AUDIT.json',receipt);dump(EXP/'CPU_REPLAY_ROWS.json',records);print(receipt)
if __name__=='__main__':main()
