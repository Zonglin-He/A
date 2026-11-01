"""Independent CPU arithmetic replay using saved GPU NN indices, never new NN."""
import argparse,numpy as np
from common34 import *

def independent_vectors(p0,pT,lib,record,key):
 q0=int(p0.argmax());p0=p0.astype(float);pT=pT.astype(float);z0=np.log(p0+1e-6)-np.log(1+1e-6-p0);zT=np.log(pT+1e-6)-np.log(1+1e-6-pT);d0=z0-z0[q0];dT=zT-zT[q0];delta=dT-d0;obs=record[key+'_obs'][:,:1];pos=record[key+'_pos'][:,:1];w=record[key+'_weights'][:,:1];gate=np.max(lib['endpoint'][obs,pos],axis=1).astype(float);raw=np.sum(lib['raw'][obs,pos]*w,axis=1)/np.maximum(np.sum(w,axis=1),1e-12);md=raw-raw[q0];local=d0+gate*delta;local[gate==0]=d0[gate==0];local[gate==1]=dT[gate==1];a=float(gate.mean());b=float(np.linalg.norm(gate*delta)/max(np.linalg.norm(delta),1e-12));return {'scoped_R30':local,'samegate_QP':d0+gate*md,'global_mean':d0+a*delta,'global_norm':d0+b*delta},gate,raw,md

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['diagnostic','U128'],required=True);a=p.parse_args();check_plan();out=output(a.split);verify(out);rr=rows(a.split);libs={(pl,k):load_npz(out/'libraries_PRIVATE'/(pl+'_'+k+'.npz')) for pl in ['drone','quad'] for k in KEYS};maxerr=0.;coserr=0.;count=0
 for r in rr:
  rec=load_npz(out/'predictions_PRIVATE'/(r['id']+'.npz'));z=feature(r);assert rec['arm_names'].tolist()==ARMS;assert np.array_equal(rec['source_probability'],z['scores']);assert np.array_equal(rec['boxes'],z['boxes'][rec['queries']]);p0=rec['source_probability'];pT=rec['R30_probability'];q0=int(p0.argmax());d0=relative_logodds(p0,q0);dT=relative_logodds(pT,q0);v={'source':d0,'R30_ungated':dT}
  for k in KEYS:
   lib=libs[r['platform'],k];n=rec[k+'_obs'];p=rec[k+'_pos'];sim=rec[k+'_similarity'];assert n.shape==p.shape==sim.shape==(256,4);assert np.all(np.diff(sim,axis=1)<=0);assert all(len(set(x))==4 for x in n);assert n.min()>=0 and n.max()<len(lib['keys']) and p.min()>=0 and p.max()<256
   # Validate only saved candidates against source keys in double precision.
   Q=make_key(z,k).astype(float);K=lib['keys'][n,p].astype(float);expected=np.einsum('qc,qkc->qk',Q,K);ce=float(np.max(abs(expected-sim)));coserr=max(coserr,ce);assert ce<3e-6,(r['alias'],k,ce)
   vv,g,raw,md=independent_vectors(p0,pT,lib,rec,k);assert np.array_equal(g,rec[k+'_gate']);assert np.allclose(raw,rec[k+'_raw_retrieved'],atol=1e-12,rtol=0);assert np.allclose(md,rec[k+'_relative_memory'],atol=1e-12,rtol=0);prefix='h_' if k=='h288' else 'noun_';v.update({prefix+s:x for s,x in vv.items()})
  scores=np.stack([v[n] for n in ARMS]);er=float(np.max(abs(scores-rec['complete_scores'])));maxerr=max(maxerr,er);assert er<1e-12,(r['alias'],er);qs=np.array([group_readout(r,x) for x in scores]);assert np.array_equal(qs,rec['queries']);assert qs[0]==int(p0.argmax());assert qs[1]==group_readout(r,pT);count+=len(qs)
 receipt={'state':'PASSED','split':a.split,'observations':len(rr),'complete_score_vectors':count,'score_max_abs_error':maxerr,'saved_GPU_selected_cosine_vs_CPU64_maxerror':coserr,'NN_selection_not_recomputed_or_replaced':True,'all_source_and_R30_endpoints_preserved':True,'no_GT_GPU_or_training':True,'prediction_seal_sha256':sha(out/'SEALED.json'),**plan_receipt()};dump(EXP/('CPU_AUDIT_'+a.split+'.json'),receipt);print(receipt)
if __name__=='__main__':main()
