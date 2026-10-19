"""Fixed full/static semantic ablation; never invent posterior support."""
import numpy as np

def normalize_log(logw,mask):
 out=np.zeros(mask.shape,dtype=np.float64);x=logw[mask];x=np.exp(x-x.max());out[mask]=x/x.sum();return out

def posteriors(p,static):
 p=np.asarray(p,dtype=np.float64);mask=p>0;assert np.isfinite(p).all() and abs(p.sum()-1)<1e-10
 lp=np.zeros_like(p);lp[mask]=np.log(p[mask]);square=normalize_log(2*lp,mask)
 if static is None:return dict(PriorSquare=square,Contrast=p.copy())
 s=np.asarray(static,dtype=np.float64);assert s.shape==p.shape and np.array_equal(s>0,mask) and np.isfinite(s).all() and abs(s.sum()-1)<1e-10
 if np.array_equal(p,s):return dict(PriorSquare=square,Contrast=p.copy())
 ls=np.zeros_like(s);ls[mask]=np.log(s[mask]);c=normalize_log(2*lp-ls,mask)
 assert np.array_equal(c>0,mask) and abs(c.sum()-1)<1e-12
 return dict(PriorSquare=square,Contrast=c)

def properties():
 p=np.array([[0.,.1,.2],[0.,0.,.7],[0.,0.,0.]])
 assert np.array_equal(posteriors(p,p)['Contrast'],p)
 assert np.array_equal(posteriors(p,None)['Contrast'],p)
 s=(p>0)/3;r=posteriors(p,s);assert np.allclose(r['Contrast'],r['PriorSquare'],atol=1e-15)
 return dict(status='pass',identical_noop=True,missing_noop=True,uniform_static_equals_sharpening=True,zero_support_preserved=True)
