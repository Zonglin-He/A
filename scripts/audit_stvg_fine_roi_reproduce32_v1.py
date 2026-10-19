"""Independent NumPy categorical/gradient/readout checks of the actual cached rerun."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_fine_roi_reproduce32_v1 import D,P,BASE,BP,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def main():
 import numpy as np
 p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32;mx=0.;rounds=0
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 def arr(x):return x.numpy()
 def close(a,b):
  nonlocal mx
  e=float(np.max(np.abs(a-b)))if a.size else 0.;mx=max(mx,e);assert e<3e-12,e
 def soft(x):
  x=x-x.max(-1,keepdims=True);y=np.exp(x);return y/y.sum(-1,keepdims=True)
 for r in p['records']:
  x=load(D/'predictions'/(r['alias']+'.pt'));E=arr(x['outputs']['E_MBR']);allowed=arr(x['allowed'])
  for name,fit in x['fits'].items():
   eta=np.zeros((len(E),2))
   for j,t in enumerate(fit['rounds']):
    before=arr(t['eta_before']);d=arr(t['standardized_candidates']);fb=t['feedback'];valid=arr(fb['valid']).copy();safe=valid.copy();safe[~safe.any(1)]=True;close(before,eta);lp=-.5*np.square(d-before[:,None]).sum(-1);lp[~safe]=-np.inf;sp=soft(lp);close(sp,arr(t['student_prob']));q=arr(fb['prob']);g=((sp-q)[...,None]*d).sum(1);g[~(allowed&arr(fb['informative']))]=0;close(g,arr(t['gradient']));eta=before-.5*g;close(eta,arr(t['eta_after']));b=E.copy();b[:,:2]+=E[:,2:]*.25*eta;close(b,arr(fit['path'][j+1]['boxes']));rounds+=1
   close(b,arr(fit['final']));assert np.array_equal(b[~allowed],E[~allowed]);assert np.array_equal(b[:,2:],E[:,2:])
  assert np.array_equal(arr(x['fits']['Fine_fixed']['path'][1]['boxes']),arr(x['fits']['Fine_live']['path'][1]['boxes']))
  for name,b in x['outputs'].items():assert np.array_equal(arr(b),load(BASE/'predictions'/(r['alias']+'.pt'))['outputs'][name])
 assert rounds==192;assert sha(P/'RESULTS_PUBLIC.json')==sha(BP/'RESULTS_PUBLIC.json');verify()
 put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,actual_rounds=rounds,actual_fits=96,output_arrays_exact=288,original_score_file_exact=True,maximum_Numpy_probability_gradient_state_readout_error=mx,new_GT_scoring=0,new_GPU_model_DINO_training=0,scope='rerun numerical checks and exact inherited metrics; original independent ROI scoring audit remains separate'))
 print('ROOT AUDIT PASS',mx,flush=True)
if __name__=='__main__':main()
