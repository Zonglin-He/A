"""Small pure-NumPy checks; no real observations, labels, models, or GPU."""
import json
import numpy as np
from common34 import ARMS,KEYS,make_key,relative_logodds,all_scores,group_readout
from coordinate_projection32 import synthetic_checks as qp_checks

def main():
 rng=np.random.default_rng(34);N=256;p0=rng.uniform(.01,.9,N).astype(np.float32);pT=rng.uniform(.01,.9,N).astype(np.float32);p0[3]=p0[7]=1.;pT[17]=pT[20]=1.;q0=3;raw=rng.normal(0,1,(4,N));obs=np.zeros((N,4),int);pos=np.stack([np.arange(N)]*4,1);weights=np.ones((N,4),float);neighbors={k:{'obs':obs,'pos':pos,'weights':weights} for k in KEYS};passed=[]
 def libs(flag,shift=0.):return {k:{'raw':raw+shift,'endpoint':np.full((4,N),flag)} for k in KEYS}
 for flag in [0,1]:
  s,proc=all_scores(p0,pT,libs(flag),neighbors);assert list(s)==ARMS
  for k in ['h','noun']:assert np.array_equal(s[k+'_scoped_R30'],relative_logodds(pT if flag else p0,q0))
  assert group_readout({'stage':'T'},s['source'])==3 and group_readout({'stage':'T'},s['R30_ungated'])==17;passed.append('g'+str(flag)+'_exact_endpoints_and_probability_ties')
 L=libs(0);L['h288']['endpoint'][0,::2]=1;L['noun64']['endpoint'][0,::2]=1;s,p=all_scores(p0,pT,L,neighbors)
 Lshift={k:{'raw':v['raw']+17,'endpoint':v['endpoint']} for k,v in L.items()};ss,_=all_scores(p0,pT,Lshift,neighbors)
 for k in ['h','noun']:assert np.allclose(s[k+'_samegate_QP'],ss[k+'_samegate_QP'],atol=1e-13,rtol=0)
 passed.append('retrieved_correction_anchor_common_offset_invariant_with_nonuniform_gate')
 for k in ['h','noun']:
  assert np.array_equal(s[k+'_scoped_R30'][1::2],s['source'][1::2]);assert np.array_equal(s[k+'_scoped_R30'][::2],s['R30_ungated'][::2]);assert np.allclose(s[k+'_global_mean'],s['source']+.5*(s['R30_ungated']-s['source']))
 passed.append('mixed_gate_endpoint_selection_and_global_mean_control')
 z={'h':rng.normal(size=(256,288)).astype(np.float32),'query':rng.normal(size=(256,64)).astype(np.float32),'tokens':rng.normal(size=(12,64)).astype(np.float32),'token_mask':np.array([True,True]+[False]*10)}
 H=make_key(z,'h288');assert np.allclose(np.linalg.norm(H,axis=-1),1.,atol=1e-6);zz=dict(z,h=z['h']*4);assert np.array_equal(H,make_key(zz,'h288'));passed.append('h_key_L2_and_positive_scale_invariance')
 q=qp_checks();result={'state':'PASSED','pure_numpy_checks':passed,'coordinate_projection_checks':q,'no_real_data_GT_GPU_model_training':True};print(json.dumps(result,indent=2));return result
if __name__=='__main__':main()
