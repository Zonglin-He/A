"""Fixed native-belief plausibility restriction; same MBR action universe."""
import numpy as np

def restrict(q,h):
 q=np.asarray(q,np.float64);h=np.asarray(h,bool);assert q.shape==h.shape and h.any() and np.isfinite(q).all() and (q>=0).all()
 if np.array_equal(q>0,h):return q.copy()
 out=np.where(h,q,0.);assert out.sum()>0;out/=out.sum();assert np.array_equal(out>0,h)and abs(out.sum()-1)<1e-12
 return out

def heads(outputs):
 p=outputs['Native']['q'];h=(p>0)&(p>=.1*p.max());qs={name:restrict(outputs[source]['q'],h)for name,source in [('NativeHead','Native'),('SquareHead','PriorSquare'),('ContrastHead','Contrast')]}
 return h,qs

def properties():
 p=np.array([[0.,.01,.1],[0.,0.,.89],[0.,0.,0.]]);h=(p>0)&(p>=.1*p.max());assert np.array_equal(h,[[False,False,True],[False,False,True],[False,False,False]])
 q=restrict(p,h);assert np.array_equal(q>0,h)and np.allclose(q[h],p[h]/p[h].sum());assert np.array_equal(restrict(p,p>0),p)
 u=(p>0)/3;assert np.array_equal(restrict(u,u>0),u)
 tie=np.array([.0625,.3125,.625]);assert tie[0]==.1*tie.max() and (tie>=.1*tie.max())[0]
 return dict(status='pass',one_native_H_for_all_arms=True,excluded_belief_mass_zero=True,no_cut_exact_identity=True,tie_included=True,original_MBR_actions_unrestricted=True)
