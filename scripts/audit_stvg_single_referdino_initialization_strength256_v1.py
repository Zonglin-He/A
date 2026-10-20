"""Independent Source+gamma*consensus reconstruction before strength scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import D,P,BASE,read,put,sha,load,verify

def main():
 p=verify();bar=read(D/'initialization_strength/PREDICTION_BARRIER.json');err=0.;n=0
 for rec in p['records']:
  ds,a=rec['dataset'],rec['alias'];old=load(BASE/f'predictions/{ds}/{a}.pt');S=np.asarray(old['Source'],dtype=np.float64);delta=np.array(old['initialization']['delta']);f=D/f'initialization_strength/predictions/{ds}/{a}.npz';assert sha(f)==bar['files'][str(f.relative_to(ROOT))]
  with np.load(f)as z:
   for gamma in [0.,.25,.5,.75,1.,1.25,1.5]:
    out=np.concatenate((S[:,:2]+gamma*delta[:2],np.exp(np.log(S[:,2:])+gamma*delta[2:])),axis=-1);e=float(abs(out-z['G'+str(int(gamma*100)).zfill(3)]).max());err=max(err,e);assert e<1e-10,(a,gamma,e);n+=1
 put(P/'INITIALIZATION_STRENGTH_PRE_GT_AUDIT_PUBLIC.json',dict(status='pass',checks=n,max_abs=err,GT_read=False,all256_sealed=True,Source_explicit_float64=True,initial_inline_audit_dtype_issue_corrected_before_GT=True));print('INIT_GAMMA_PRE_GT_AUDIT',n,err,flush=True)
if __name__=='__main__':main()
