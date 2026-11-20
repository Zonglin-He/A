"""Enumerate every cut in small forests independently of the production ancestor formula."""
import itertools
import numpy as np
from belief_scene import forest_factors,build
from spatial_simulator import outcomes

def worlds(parent,t):
    def forest(roots):
        result=[(1.,frozenset())]
        for i in roots:
            options=[(t[i],frozenset([i]))]+[((1-t[i])*w,a) for w,a in forest(np.flatnonzero(parent==i))]
            result=[(w*v,a|b) for w,a in result for v,b in options if w*v>0]
        return result
    return forest(np.flatnonzero(parent<0))

def verify():
    rng=np.random.default_rng(306)
    e=0.;trials=0
    for n in range(1,9):
        for rep in range(18):
            parent=np.array([rng.choice([-1,*range(i+1,n)]) for i in range(n)])
            t=rng.uniform(size=n)
            if rep%3==0:t[rng.integers(n)]=rep%2
            indices=np.flatnonzero(rng.integers(0,2,size=n))
            mu,valid,joint,cut,best=forest_factors(parent,t,indices)
            ws=worlds(parent,t)
            np.testing.assert_allclose(sum(w for w,_ in ws),1.)
            peak=max(w for w,_ in ws)
            assert any(abs(w-peak)<1.e-14 and all((idx in active)==v for idx,v in zip(indices,cut)) for w,active in ws)
            for a,i in enumerate(indices):
                np.testing.assert_allclose(mu[a],sum(w for w,active in ws if i in active))
                for b,j in enumerate(indices):
                    cover=[w for w,active in ws if i!=j and {i,j}<=active]
                    target=sum(cover)
                    e=max(e,abs(joint[a,b]-target))
                    np.testing.assert_allclose(joint[a,b],target,atol=1.e-14)
                    np.testing.assert_allclose(np.exp(best[a,b])*peak,max(cover,default=0.),atol=1.e-14)
            trials+=1
    # Separate 2 objects, a shared ancestor, and a competing root.
    parent=np.array([2,2,-1,-1]);stop=np.array([.8,.9,.7,.4])
    _,_,M,_,_=forest_factors(parent,stop,np.arange(4))
    np.testing.assert_allclose(M[0,1],.3*.8*.9)
    assert M[0,2]==0 and M[1,2]==0
    # Independent SE(2) implementation checks the observer rotation sign and metric translation.
    a=rng.uniform(-8,8,(12,7));b=rng.uniform(-8,8,(13,7))
    for deg in (-90.,-14.,0.,30.,90.,145.):
        ang=np.deg2rad(deg);x=np.array([np.cos(ang),np.sin(ang),0.]);f=np.array([-x[1],x[0],0.])
        rot=np.array([x,[0,0,-1],f]); cam=np.eye(4);cam[:3,:3]=rot; cam[:3,3]=[2.,3.,5.]
        delta=a[:,None,:3]-b[None,:,:3]
        labels=outcomes(a,b,cam)
        ref=[delta@x>0,delta@f>0,delta@((x-f)/np.sqrt(2))>0,
             np.linalg.norm((delta+2*x)[...,:2],axis=2)>np.linalg.norm(delta[...,:2],axis=2)]
        for v,w in zip(labels,ref):np.testing.assert_equal(v,w)
    return dict(forests=trials,max_joint_mass_error=e,rigid_matrix_cases=6,oracle_geometry_implementations=2)
if __name__=='__main__':print(verify())
