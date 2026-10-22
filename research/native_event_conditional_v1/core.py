"""Use native clause location evidence conditional on discrete grid-span length."""
import numpy as np

def duration_conditional(native,teacher):
 q=np.asarray(native,dtype=np.float64);t=np.asarray(teacher,dtype=np.float64)
 assert q.shape==t.shape and q.ndim==2 and q.shape[0]==q.shape[1]
 legal=np.triu(np.ones(q.shape,bool),1)
 for x in [q,t]:assert (x>=0).all()and not x[~legal].any()and abs(x.sum()-1)<1e-10
 if np.array_equal(q,t):return q.copy(),0.
 n=len(q);out=np.zeros_like(q);maxerr=0.
 for length in range(1,n):
  i=np.arange(n-length);j=i+length;prior=q[i,j];mass=float(prior.sum());support=prior>0
  if mass==0:continue
  evidence=np.where(support,t[i,j],0);den=float(evidence.sum())
  # Prespecified impossible-teacher support fallback, not confidence abstention.
  # Primary teacher includes half native, so this cannot trigger there.
  out[i,j]=mass*evidence/den if den>0 else prior
  maxerr=max(maxerr,abs(float(out[i,j].sum())-mass))
 assert abs(out.sum()-1)<1e-12 and not out[q==0].any()
 assert maxerr<1e-14;return out,maxerr
