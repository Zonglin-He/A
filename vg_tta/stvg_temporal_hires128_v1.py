"""Qualified same-code input resolution and physical boundary interventions."""
from contextlib import contextmanager
import inspect,textwrap
import numpy as np

@contextmanager
def resolution(height):
 from methods.decota_final_simplified_v1 import backbone
 original=backbone.make_batch;source=textwrap.dedent(inspect.getsource(original));assert source.count('height = 224')==1
 changed=source.replace('height = 224',f'height = {int(height)}');assert changed.replace(f'height = {int(height)}','height = 224')==source
 ns=dict(vars(backbone));exec(compile(changed,__file__,'exec'),ns);backbone.make_batch=ns['make_batch']
 try:yield
 finally:backbone.make_batch=original

def hybrid(p,h):
 p=np.asarray(p,dtype=np.float64);h=np.asarray(h,dtype=np.float64);assert np.array_equal(p>0,h>0)
 pm=p.sum(1);hm=h.sum(1);pc=np.divide(p,pm[:,None],out=np.zeros_like(p),where=pm[:,None]>0);hc=np.divide(h,hm[:,None],out=np.zeros_like(h),where=hm[:,None]>0)
 out=dict(RootStart=hm[:,None]*pc,RootEnd=pm[:,None]*hc,HiResMix=.5*p+.5*h)
 for x in out.values():assert abs(x.sum()-1)<1e-12 and np.array_equal(x>0,p>0)
 return out

def student_start(p,h,ids,interval):
 """Teacher end distribution at the initial student's actual start state."""
 i=list(ids).index(interval[0]);mass=float(h[i].sum())
 if mass<=0:return p.copy(),dict(available=False,reason='zero teacher mass at original decision start')
 q=np.zeros_like(h);q[i]=h[i]/mass
 return q,dict(available=True,start_index=i,student_start=interval[0])

def properties():
 p=np.array([[0.,.1,.2],[0.,0.,.7],[0.,0.,0.]]);out=hybrid(p,p)
 assert all(np.allclose(v,p,atol=1e-15)for v in out.values());h=np.array([[0.,.2,.1],[0.,0.,.7],[0.,0.,0.]])
 out=hybrid(p,h);assert np.allclose(out['RootEnd'].sum(1),p.sum(1));assert np.allclose(out['RootStart'].sum(1),h.sum(1))
 return dict(status='pass',same_posterior_identity=True,marginals_exact=True,zero_support_preserved=True)
