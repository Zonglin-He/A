"""Directly enumerate the independent sampling law and the constrained face solution."""
import numpy as np
from delivery import distribute,counts,update,weighted_median,fit_faces
from role_cut import structure

def check():
    rng=np.random.default_rng(1463)
    maxerr=0.;checks=0
    for trial in range(36):
        xyz=rng.integers(-4,5,size=(22,3)).astype(float)
        parent=np.array([3,3,-1,-1]);p=rng.uniform(.1,.95,4)
        values=np.array([3,4,7,7])
        a=rng.integers(-1,3,size=len(xyz))
        chain=np.vstack([a,a,np.where((a==0)|(a==1),3,a)])
        data=dict(xyz=xyz,assign=chain)
        measure,_=structure(parent,p,values)
        box=np.concatenate([rng.integers(-5,6,size=(9,3,3)),rng.integers(1,8,size=(9,3,3))],axis=-1)
        if trial%3==0:box[:]=box[:,:1]
        g,supp=distribute(data,box,measure)
        brute=np.zeros_like(g)
        for k in range(9):
            for c in range(3):
                b=box[k,c];inside=np.flatnonzero((abs(xyz-b[:3])<=(b[3:]/2)).all(1));assert len(inside)==supp[k,c]
                for i in inside:
                    for v in set(chain[:,i]) - {-1}:brute[k,c,v]+=measure[v]/len(inside)
        q=rng.dirichlet(np.ones(3),size=9);weights=q*rng.random((9,1))
        m,e=counts(g,q,weights)
        bm=np.zeros_like(m);be=np.zeros_like(m)
        for k in range(9):
            for c in range(3):
                bm[:,c]+=weights[k,c]*brute[k,c]
                for d in range(3):be[:,c]+=weights[k,c]*q[k,d]*brute[k,d]
        np.testing.assert_allclose(g,brute,atol=1.e-14)
        np.testing.assert_allclose(m,bm,atol=1.e-14);np.testing.assert_allclose(e,be,atol=1.e-14)
        maxerr=max(maxerr,float(np.max(abs(g-brute))),float(np.max(abs(be-e))))
        if trial%3==0:
            base=rng.dirichlet(np.ones(3),size=4)
            np.testing.assert_allclose(update(base,m,e),base,atol=1.e-14)
        sample=rng.normal(size=(12,6))
        w=rng.random(12);f=weighted_median(sample,w)
        for d in range(6):
            cost=lambda v:np.sum(w*abs(v-sample[:,d]))
            assert cost(f[d])<=min(cost(x) for x in sample[:,d])+1.e-12
        # Floor/ceiling projection is the constrained L1 minimizer in each coordinate.
        anchor=np.array([0.,0.,0.,1.,2.,2.,.41])
        geom=np.abs(rng.normal(size=(9,3,6)))+.1
        dat=dict(boxes=np.tile(anchor,(4,1)),completed=np.tile(anchor,(4,3,1)))
        for neutral in (True,False):
            f=fit_faces(dat,(g,geom,q,weights),0,1,neutral)
            assert np.all(f[3:6]>=anchor[3:6]-1e-12) and np.isclose(f[6],anchor[6])
        checks+=1
    return checks,maxerr

if __name__=='__main__':
    print(check())

