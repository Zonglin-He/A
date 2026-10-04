"""Path factorisation, permutation and differentiability contracts."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from vg_tta.decota_track_critic_r3_v1 import TrackEnergy,FrameSum

def run():
    b=torch.tensor([[.3,.4,.2,.3],[.4,.4,.2,.3]],requires_grad=True)
    def obs():return dict(probe=dict(boxes=[[.3,.4,.2,.3],[.7,.6,.2,.3]],target_scores=[.7,.6],accepted=True),receipt=dict(context_active=False))
    ex=dict(observations={(0,0):obs(),(1,1):obs()})
    f=FrameSum(ex,b);t=TrackEnergy(ex,b,[0,3],native_prior=False,continuity=False)
    assert len(t.paths)==4 and torch.allclose(f(b),t(b),atol=2e-7);checks=2
    assert torch.allclose(torch.autograd.grad(f(b),b)[0],torch.autograd.grad(t(b),b)[0],atol=3e-6);checks+=1
    q=TrackEnergy(ex,b,[0,3]);assert abs(float(q.logpath.exp().sum())-1)<1e-6 and 0<=q.path_authority<=1;checks+=2
    perm=dict(observations={(1,1):obs(),(0,0):obs()});qp=TrackEnergy(perm,b,[0,3]);assert torch.equal(q.logpath,qp.logpath) and torch.equal(q(b),qp(b));checks+=2
    e=TrackEnergy(dict(observations={}),b,[0,3]);assert e.empty and e.path_authority==0 and torch.autograd.grad(e(b),b)[0].abs().sum()==0;checks+=2
    print('TRACK_CONTRACTS_PASS',checks)

if __name__=='__main__':run()
