"""Meaningful objective/gradient contracts, without models or labels."""
import sys,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from vg_tta.decota_identity_commitment_v1 import IdentityContrastive
from scripts.score_decota_actuation_scope_v1 import path_numpy,iou
from scipy.special import logsumexp

def ex(frames):
    return dict(observations={(0,j):dict(probe=dict(boxes=bb,target_scores=ss,accepted=True),receipt=dict(context_active=False)) for j,bb,ss in frames})
def run():
    b=torch.tensor([[.3,.4,.25,.3],[.5,.5,.3,.2]],dtype=torch.double,requires_grad=True)
    e=ex([(0,[[.3,.4,.2,.3],[.7,.5,.2,.3]],[.7,.6]),(1,[[.5,.5,.3,.2],[.7,.5,.3,.2]],[.5,.9])])
    f=IdentityContrastive(e,b,[10,20]);ff,paths,lp,_=path_numpy(e,b.detach().numpy(),[10,20])
    assert f.map_index==int(lp.argmax()) and f.paths==paths
    def independent(x):
        c=np.array([np.mean([iou(x[pos],candidates[z[t]],True) for t,(pos,candidates,_) in enumerate(ff)]) for z in paths])
        return logsumexp(c)-c[f.map_index]
    assert abs(float(f(b))-independent(b.detach().numpy()))<1e-10
    g=torch.autograd.grad(f(b),b)[0];x=b.detach().numpy().copy()
    # Non-overlap branches away from equality are used for the finite difference.
    x+=np.array([[.013,.017,.019,.011],[.017,.013,.011,.019]])
    bb=torch.tensor(x,requires_grad=True);gg=torch.autograd.grad(f(bb),bb)[0].numpy();fd=np.zeros_like(x)
    for ix in np.ndindex(x.shape):
        xp=x.copy();xm=x.copy();xp[ix]+=1e-6;xm[ix]-=1e-6;fd[ix]=(independent(xp)-independent(xm))/2e-6
    assert np.max(abs(fd-gg))<1e-6
    # Complete Cartesian contrastive partition equals framewise cross-entropy.
    from methods.decota_final_simplified_v1.objectives import generalized_iou
    per=[]
    for t,(pos,ev,_) in enumerate(f.frames):
        c=generalized_iou(b[pos],ev)/len(f.frames);per.append(torch.logsumexp(c,0)-c[f.paths[f.map_index][t]])
    assert torch.allclose(f(b),sum(per),atol=1e-12,rtol=0)
    m=f.map_index;f(b*1.3);assert f.map_index==m
    single=IdentityContrastive(ex([(0,[[.3,.4,.2,.3]],[.8])]),b,[10,20]);assert single.no_competitor
    assert float(single(b))==0 and torch.count_nonzero(torch.autograd.grad(single(b),b)[0])==0
    empty=IdentityContrastive(ex([]),b,[10,20]);assert empty.empty and float(empty(b))==0
    dup=IdentityContrastive(ex([(0,[[.3,.4,.2,.3],[.3,.4,.2,.3]],[.8,.8])]),b,[10,20]);assert len(dup.paths)==2 and dup.map_index==0
    print('PASS independent E-step,loss,gradient,Cartesian factorization,fixed MAP,singleton,empty,duplicate tie',flush=True)
if __name__=='__main__':run()
