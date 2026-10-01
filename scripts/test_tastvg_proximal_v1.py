"""Meaningful objective/projector invariants, CPU only."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from vg_tta.tastvg_proximal_method_v1 import target_loss,project_arrival,norm
def run():
 torch.manual_seed(7)
 c=torch.tensor([[.4,.4,.2,.2],[.5,.5,.3,.3]],requires_grad=True)
 q=torch.stack([c.detach(),c.detach()+.01,c.detach()-.01])
 loss,p,t,d,lt,m=target_loss(c,q,torch.ones(3),[5.,2.],.35,1.,'proximal',.02)
 g,=torch.autograd.grad(loss,c);assert loss.item()==0. and torch.count_nonzero(g)==0
 loss,p,t,d,lt,m=target_loss(c,q,torch.tensor([.2,.2001,.1]),[5.,2.],.35,1.,'proximal',.02)
 assert not lt.requires_grad and torch.allclose(t,(1-m['strength'])*p.detach()+m['strength']*m['q_rank'])
 gp,=torch.autograd.grad(loss,c);assert torch.isfinite(gp).all()
 class Actor:
  def __init__(self):self.named=[('a',torch.nn.Parameter(torch.zeros(4))),('b',torch.nn.Parameter(torch.zeros(3)))]
 a=Actor();center={n:p.detach().clone() for n,p in a.named}
 with torch.no_grad():a.named[0][1].add_(.001)
 old=a.named[0][1].detach().clone();z=project_arrival(a,center,.1);assert z['projection_factor']==1 and torch.equal(old,a.named[0][1])
 for k in range(8):
  with torch.no_grad():
   for n,p in a.named:p.add_(.2)
  z=project_arrival(a,center,.1);assert z['actual_arrival_norm']<=.100002 and z['projection_factor']<1
 assert abs(norm({n:p.detach()-center[n] for n,p in a.named})-.1)<1e-6
 print('PASS: exact flat no-op, detached arithmetic target, finite gradient, small update unchanged, shared eight-step budget; CPU only')
if __name__=='__main__':run()
