"""Independent alpha readout reconstruction, with no GT access."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import ROOT,D,P,BASE,read,put,sha,load,verify

def main(ds,phase):
 p=verify();bar=read(D/ds/phase/'PREDICTION_BARRIER.json');assert read(P/ds/phase/'PRE_GT_AUDIT_PUBLIC.json')['status']=='pass';checks=0;err=0.;beg=time.perf_counter()
 for rec in [r for r in p['records']if r['dataset']==ds]:
  a=rec['alias'];old=load(BASE/f'predictions/{ds}/{a}.pt');E=np.asarray(old['E_Ref']);cache={};f=D/ds/phase/'predictions'/f'{a}.npz';assert sha(f)==bar['files'][str(f.relative_to(ROOT))]
  with np.load(f)as z:
   assert np.array_equal(z['E_Strong'],E)and np.array_equal(z['Strong_interval'],old['intervals']['Strong'])
   for arm,c in bar['definitions'].items():
    key=c['fit_phase'],c['config_id']
    if key not in cache:
     fb=read(D/ds/f'{key[0].upper()}_FIT_BARRIER.json');entry=fb['files'][key[1]+'/'+a];assert sha(ROOT/entry['path'])==entry['sha256'];cache[key]=load(ROOT/entry['path'])['fit']
    fit=cache[key];b=fit['path'][c['steps']]['boxes'].numpy();alpha=c['alpha'];out=np.empty_like(E);out[:,:2]=(1-alpha)*E[:,:2]+alpha*b[:,:2];out[:,2:]=np.exp((1-alpha)*np.log(E[:,2:])+alpha*np.log(b[:,2:]));error=float(np.max(abs(out-z[arm])));err=max(err,error);assert error<1e-10,(a,arm,error);checks+=1
    if alpha==1:assert np.array_equal(z[arm],b)
 put(P/ds/phase/'READOUT_AUDIT_PUBLIC.json',dict(status='pass',all128=True,readout_checks=checks,max_abs=err,alpha_center_and_log_size_independent=True,Strong_unchanged=True,GT_read=False,seconds=time.perf_counter()-beg));print('READOUT_AUDIT_PASS',ds,phase,checks,err,flush=True)
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
