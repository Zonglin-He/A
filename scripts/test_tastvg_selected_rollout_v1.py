"""CPU analytic VJP, full-space direction, argmax/no-op and normalized actuation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from vg_tta.tastvg_selected_rollout_v1 import apply_selected,functional_movement,target_loss
from vg_tta.tastvg_spatial_online_opd_s1_v1 import geometry
def run():
 # Full Jacobian actuation can use directions outside a one-dimensional probe.
 W=torch.tensor([[1.,0.,2.],[0.,1.,1.]],dtype=torch.float64)
 p=torch.nn.Parameter(torch.zeros(3,dtype=torch.float64));target=torch.tensor([1.,2.],dtype=torch.float64)
 loss=((W@p-target)**2).sum()/2;gg=torch.autograd.grad(loss,p)[0]
 torch.testing.assert_close(gg,-W.T@target,atol=0,rtol=0)
 ga=torch.tensor([.3,-.4,.2],dtype=torch.float64);before=p.detach().clone()
 m=apply_selected([('p',p)],[ga],[gg],.02,2,False)
 expected=-.02*ga.norm()*gg/(gg.norm()+1e-12)
 torch.testing.assert_close(p-before,expected,atol=1e-16,rtol=0)
 assert p[2]!=0 and abs(m['actual_step_norm']-.02*float(ga.norm()))<1e-12
 assert functional_movement((W@before)[None],(W@p)[None],target[None])['output_space_cosine']>0
 for selected,flat,gvec,reason in [(0,False,gg,'central_selected'),(2,True,gg,'flat_rewards'),(2,False,torch.zeros_like(gg),'zero_selected_gradient')]:
  q=torch.nn.Parameter(torch.ones(3,dtype=torch.float64));old=q.detach().clone();a=apply_selected([('q',q)],[ga],[gvec],.1,selected,flat)
  assert torch.equal(q,old) and a['no_op_reason']==reason
 q=torch.nn.Parameter(torch.ones(3));old=q.detach().clone();a=apply_selected([('q',q)],[torch.zeros(3)],[torch.ones(3)],.1,1,False)
 assert torch.equal(q,old) and a['no_op_reason']=='zero_rkl_magnitude'
 assert int(np.argmax([.2,.8,.8]))==1 and int(np.argmax([.8,.8,.2]))==0
 # Target detachment is enforced by the actual geometric objective.
 central=torch.tensor([[.4,.4,.2,.2]],requires_grad=True);cand=torch.tensor([[[.6,.4,.2,.2]]],requires_grad=True)
 gd=torch.autograd.grad(geometry(central,cand,5.,2.).sum(),[central,cand],allow_unused=True)
 assert gd[0] is not None and gd[1] is None
 assert not torch.cuda.is_initialized()
 print('PASS: analytic full VJP, outside-probe actuation, same-state magnitude, central/flat/zero no-op, argmax ties, detached geometric target, functional direction')
if __name__=='__main__':run()
