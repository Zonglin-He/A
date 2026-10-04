"""Mathematical contracts for the two new interventions, no GT/model."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from vg_tta.decota_optimizer_posterior_r1_v1 import posterior,Energy,authority
from scipy.special import logsumexp

def run():
    checks=0
    ids=np.array([0,1,2,4,7]);ij=np.array(np.triu_indices(5,1));lp=np.linspace(-2,2,len(ij[0]));lp-=logsumexp(lp);cost=np.linspace(1,-1,len(lp))
    p=posterior(lp,cost,ij,ids);assert p['centre_marginal_max_error']<1e-12;checks+=1
    z=posterior(lp,cost,ij,ids,beta=0);assert np.allclose(z['logfull'],lp) and np.allclose(z['logextent'],lp);checks+=2
    p2=posterior(lp,cost+19,ij,ids);assert np.allclose(p2['logfull'],p['logfull']) and np.allclose(p2['logextent'],p['logextent']);checks+=2
    b=torch.tensor([[.5,.5,.4,.4]],requires_grad=True)
    ex={'observations':{(0,0):{'probe':{'boxes':[[.5,.5,.4,.4],[.7,.7,.2,.2]],'target_scores':[.5,.5],'accepted':False},'receipt':{'context_active':False}}}}
    f=Energy(ex,b,'all');assert abs(authority(f))<1e-7;checks+=1
    ff=Energy(ex,b,'admit');assert ff.empty and authority(ff)==0;assert torch.autograd.grad(ff(b),b)[0].abs().sum()==0;checks+=2
    ex['observations'][(0,0)]['probe']['accepted']=True
    ft=Energy(ex,b,'top1');assert ft.frames[0][1].shape==(1,4) and authority(ft)==1;checks+=1
    g=torch.tensor([.03,-.4]);a=torch.zeros(2,requires_grad=True);o=torch.optim.SGD([a],lr=.1);a.grad=g;o.step()
    aa=torch.zeros(2,requires_grad=True);oo=torch.optim.SGD([aa],lr=.1);aa.grad=.01*g;oo.step();assert torch.allclose(aa,.01*a);checks+=1
    # Constant gradient scale cancellation is approximate and eps-dependent.
    q=torch.zeros(2,requires_grad=True);qo=torch.optim.Adam([q],lr=.03);q.grad=g;qo.step()
    qq=torch.zeros(2,requires_grad=True);qqo=torch.optim.Adam([qq],lr=.03);qq.grad=.1*g;qqo.step();assert torch.allclose(q,qq,atol=1e-6,rtol=0);checks+=1
    print('CONTRACTS_PASS',checks);return checks

if __name__=='__main__':run()
