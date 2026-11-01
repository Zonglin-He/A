"""Postseal A-only decomposition of preservation drift, no GT or new fitting."""
import numpy as np
from common35 import *
from audit35 import replay,independent_masks,CONFIGS

def main():
 out=BASE/'partial_keep_attempt001';verify(out);assert read(EXP/'CPU_AUDIT.json')['state']=='PASSED';fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');s=load_npz(ROOT/'crossplatform/round24/capture336_attempt001/projection_source_PRIVATE.npz');s={k.removeprefix('image.'):v for k,v in s.items() if k.startswith('image.')};result=[]
 for pl in ['drone','quad']:
  A=[r for r in rows() if r['platform']==pl];ff=[f for f in fb if f['platform']==pl];old,partial=independent_masks(A,ff);new=partial&~old;free=~partial;zs=[feature(r) for r in A];q0=np.array([int(z['scores'].argmax()) for z in zs]);z0=np.stack([replay(z,s)[1] for z in zs]);d0=z0-z0[np.arange(len(A)),q0][:,None]
  for name in CONFIGS:
   c=load_npz(EXP/'CPU_REPLAY_PRIVATE'/(pl+'_'+name+'_A208.npz'));z=c['exact_logodds_CPU64'];d=z-z[np.arange(len(A)),q0][:,None];sq=(d-d0)**2;p=load_npz(out/'predictions_PRIVATE'/(pl+'_'+name+'_A208.npz'));rec={'platform':pl,'configuration':name,'original_U_MSE':float(sq[old].mean()),'new_feedback_unjudged_MSE':float(sq[new].mean()),'accepted_endpoint_MSE':float(sq[free].mean()),'all_A_coordinate_MSE':float(sq.mean()),'old_U_coordinate_count':int(old.sum()),'new_feedback_unjudged_coordinate_count':int(new.sum()),'freed_endpoint_count':int(free.sum()),'nonfeedback_query_changes':int((p['queries'][old.any(-1)]!=q0[old.any(-1)]).sum()),'feedback_query_changes':int((p['queries'][~old.any(-1)]!=q0[~old.any(-1)]).sum()),'diagnostic_only_no_label_filtering':True};result.append(rec)
 dump(EXP/'A_PRESERVATION_DRIFT_DECOMPOSITION.json',result);print(result)
if __name__=='__main__':main()
