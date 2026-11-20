"""Small exhaustive world enumeration checks the compiled inference against its definition."""
from itertools import product
import numpy as np
from role_cut import shared,separate,structure,layout,build_library

def exhaustive(parents,stop,p,words,views,side,depth,nearest,valid,mode):
    n=len(stop)
    def expand(i):
        emitted=[({i:c},stop[i]*p[i,c]) for c in range(p.shape[1])]
        children=np.flatnonzero(parents==i)
        return emitted+[(d,(1-stop[i])*w) for d,w in combine(children)]
    def combine(nodes):
        states=[({},1.)]
        for node in nodes:
            states=[(d|a,w*b) for d,w in states for a,b in expand(int(node))]
        return states
    v,r,ok,gap=layout(views,side,mode)
    a=np.zeros((n,n));b=np.zeros(n)
    total=0.
    for world,w in combine(np.flatnonzero(parents<0)):
        total+=w
        ss=[i for i,c in world.items() if c==words[0] and ok[i]]
        rr=[i for i,c in world.items() if c==words[1] and ok[i]]
        for j in rr:
            if any(depth*(r[k,3]-r[j,3])<0 for k in rr):continue
            b[j]+=w
            feasible=[i for i in ss if valid[i,j] and gap[i,j]>0]
            for i in feasible:
                if nearest and any(gap[k,j]<gap[i,j] for k in feasible):continue
                a[i,j]+=w
    np.testing.assert_allclose(total,1,atol=1e-11)
    return a,b

def run():
    build_library()
    par=np.array([3,3,-1,-1,-1]);stop=np.array([.37,.85,.54,.23,.15])
    p=np.array([[.6,.4],[.31,.69],[.02,.98],[.72,.28],[.77,.23]])
    views=np.array([[.2,.21,.23,7,9],[.2,.21,.22,6,8],[.01,.03,.07,4,3],[.18,.20,.3,6,9],[.7,.75,.8,7,8]])
    marg,K=structure(par,stop,np.array([2,2,2,6,2]));valid=K==0
    count=0;maxerr=0.
    for ss,rr,side,depth,near,kind in product(range(2),range(2),[-1,1],[-1,0,1],[False,True],
                                              ['physical','wrong_reference_binding','image_center']):
        a=exhaustive(par,stop,p,(ss,rr),views,side,depth,near,valid,kind)
        b=shared(par,valid,stop,p,(ss,rr),views,side,depth,near,kind)
        for x,y in zip(a,b):
            maxerr=max(maxerr,float(np.max(abs(x-y))))
            np.testing.assert_allclose(x,y,atol=1e-12,rtol=0)
        if kind=='physical':
            a=exhaustive(np.full_like(par,-1),marg,p,(ss,rr),views,side,depth,near,valid,kind)
            b=separate(valid,marg,p,(ss,rr),views,side,depth,near)
            for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=1e-12,rtol=0)
        count+=1
    print(count,'shared tree conditions verified, maximum error',maxerr)
    return count,maxerr
if __name__=='__main__':run()
