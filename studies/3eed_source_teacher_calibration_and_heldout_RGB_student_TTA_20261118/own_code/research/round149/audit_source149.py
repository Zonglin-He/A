"""Independent float64 NumPy matching-head check; no labels."""
from common149 import *
import numpy as np
from scipy.special import logsumexp

def numpy_base(a,W):
 x=a['h'].astype(np.float64)
 for j in [0,2]:x=np.maximum(x@W['match.'+str(j)+'.weight'].astype(float).T+W['match.'+str(j)+'.bias'],0)
 q=x@W['match.4.weight'].astype(float).T+W['match.4.bias']+a['late'];q/=np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12);v=q@a['u'].T/.07;m=a['mask'].astype(bool);return logsumexp(v[:,m],axis=-1)-logsumexp(v[:,~m],axis=-1)
def main():
 check();root=B/'source_head';seal=rd(root/'SEALED.json');n=0;mx=0.;mp=0.
 for f,h in seal['files'].items():assert sha(root/f)==h
 for src in ['waymo','drone','quad']:
  W=dict(np.load(R/'crossplatform/round135'/(src+'_SOURCE_MODEL_PRIVATE.npz')))
  for p in sorted((root/src).glob('*.npz')):
   a=np.load(R/'crossplatform/round145/source'/src/p.name);out=np.load(p);b=numpy_base({k:a['B_'+k] for k in ['h','u','late','mask']},W);err=float(abs(b-out['base']).max());mx=max(mx,err);assert err<3e-4
   for alpha,pr in zip(out['alphas'],out['p']):
    z=alpha*np.log(np.maximum(a['pB'],1e-300))+(1-alpha)*b;q=np.exp(z-logsumexp(z));err=float(abs(q-pr).max());mp=max(mp,err);assert err<5e-5
    if alpha==1:assert np.array_equal(pr,a['pB'])
   n+=1
 assert n==313;dump(H/'SOURCE_HEAD_AUDIT.json',dict(status='PASS',n=n,max_numpy_base_error=mx,max_numpy_probability_error=mp,GT_read=False,original_native_source_reader_audit='R145/SOURCE_AUDIT.json'));print('SOURCE_HEAD_AUDIT',n,mx,mp,flush=True)
if __name__=='__main__':main()
