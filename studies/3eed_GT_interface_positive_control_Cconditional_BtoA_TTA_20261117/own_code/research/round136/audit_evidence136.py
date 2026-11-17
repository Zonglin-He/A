"""Independent GT-free B-target and direct-C mass/ratio audit."""
from common136 import *
import numpy as np,sys
sys.path.insert(0,str(R136/'research/round129'));import support129 as U
def main():
 check();seal=rd(B136/'EVIDENCE_SEALED.json');n=0;eligible=0;errmax=0.
 for path,h in seal['files'].items():
  assert sha(R136/path)==h;a=np.load(R136/path);name=Path(path).parent.name;sid=Path(path).stem;S,C=a['S'],a['C'];ass=a['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];old=np.load(R136/'crossplatform/round135/formal'/name/'samples_PRIVATE'/(sid+'.npz'));p0=dict(zip(old['arm_names'],old['scores']))['intact_live_A'];n+=1
  if not bool(a['eligible']):assert np.array_equal(a['direct'],p0) and np.array_equal(a['wrong_direct'],p0);continue
  eligible+=1;B=np.load(R136/'crossplatform/round135/formal'/name/'evidence_PRIVATE'/(sid+'.npz'))['pB'];logs=np.array([np.log(max(float(B[g].max()),1e-300)) for g in groups])[C];target=np.exp(logs-logs.max());target/=target.sum();assert np.allclose(target,a['target'],atol=1e-14);assert np.array_equal(a['wrong_target'],a['target'][a['permutation']]);assert np.array_equal(np.sort(a['permutation']),np.arange(C.sum()));ids=np.flatnonzero(C);assert all(len(groups[ids[i]])==len(groups[ids[a['permutation'][i]]]) for i in range(len(ids)))
  partial=U.direct(p0,groups,S,C,None,'partial');qc=C[ass]
  for kind,q in [('direct',target),('wrong_direct',a['wrong_target'])]:
   p=a[kind];error=max(float(abs(p[~qc]-partial[~qc]).max(initial=0)),abs(float(p[qc].sum()-partial[qc].sum())));mx=np.array([p[g].max() for g in groups])[C];mx/=mx.sum();error=max(error,float(abs(mx-q).max()));errmax=max(errmax,error);assert error<1e-12
   for g in groups:
    live=partial[g]>0
    if live.any():ratio=p[g][live]/partial[g][live];assert np.ptp(ratio)<max(1.,abs(ratio).max())*1e-10
 assert n==1798 and eligible==764;dump(H136/'EVIDENCE_AUDIT.json',{'status':'PASS','n':n,'eligible':eligible,'max_direct_mass_or_target_ratio_error':errmax,'qB_frozen_before_GT_diagnostic':True,'target_entropy_and_multiplicity_preserved_in_wrong_control':True,'GT_read':False,'FINAL_access':False});print(rd(H136/'EVIDENCE_AUDIT.json'),flush=True)
if __name__=='__main__':main()
