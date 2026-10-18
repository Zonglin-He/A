"""Mechanism contracts independent of dataset metrics or learned accuracy."""
import torch
from vg_tta.stvg_joint_posterior_v1 import roi,JointPrior,observation_loglik

def run():
 torch.manual_seed(71);torch.set_num_threads(2)
 # An affine feature field has an analytically known ROI derivative.
 yy,xx=torch.meshgrid(torch.linspace(0,1,8),torch.linspace(0,1,8),indexing='ij');tt=torch.linspace(0,1,32)
 maps=(xx[None]+2*yy[None]+3*tt[:,None,None])[None,None]
 box=torch.tensor([[[.4,.6,.2,.3]]],requires_grad=True);time=torch.tensor([[.3]],requires_grad=True)
 value=roi(maps,box,time);g,h=torch.autograd.grad(value.sum(),(box,time));assert abs(float(value)-2.5)<1e-5;assert torch.allclose(g,torch.tensor([[[1.,2.,0.,0.]]]),atol=2e-5)and abs(float(h)-3)<1e-5
 # Compare the marginalized observation gradient to central finite differences.
 x=torch.zeros(1,33,4);x[:,1:]=torch.linspace(-.8,.8,32)[None,:,None];x.requires_grad_();obs=dict(time=torch.tensor([.3,.45]),boxes=torch.tensor([[.8,.7,.3,.4],[.2,.4,.1,.2]]));c=dict(unit=torch.tensor([.01]));p=torch.tensor([.6,.7])
 L=observation_loglik(x,c,obs,p,.02);grad=torch.autograd.grad(L.sum(),x)[0]
 for i,j in [(0,0),(0,1),(12,0),(20,2)]:
  plus=x.detach().clone();minus=x.detach().clone();plus[0,i,j]+=.001;minus[0,i,j]-=.001
  fd=(observation_loglik(plus,c,obs,p,.02)-observation_loglik(minus,c,obs,p,.02))/.002
  assert abs(float(fd)-float(grad[0,i,j]))<.005,(i,j,fd,grad[0,i,j])
 empty=dict(time=torch.empty(0),boxes=torch.empty(0,4));assert observation_loglik(x,c,empty,torch.empty(0),.02).item()==0
 # Architectural separation really excludes both directions of state coupling.
 c.update(maps=torch.randn(1,512,32,8,8),query=torch.randn(1,7,256),qmask=torch.zeros(1,7,dtype=torch.bool),context=torch.randn(1,32,512),native=x.detach())
 records=[]
 for sep in [False,True]:
  m=JointPrior(sep);z=x.detach().clone().requires_grad_();out=m(z,torch.tensor([.3]),c)
  a=torch.autograd.grad(out[:,0,0].sum(),z,retain_graph=True)[0][:,1:].norm();b=torch.autograd.grad(out[:,1:,0].sum(),z)[0][:,0,:2].norm()
  assert (a==0 and b==0)if sep else(a>0 and b>0)
  records.append(dict(separated=sep,parameters=sum(v.numel()for v in m.parameters()),time_from_space=float(a),space_from_time=float(b)))
 return dict(status='pass',affine_ROI_value_and_derivative=True,observation_finite_difference_checks=4,empty_observation_exact_zero=True,coupling=records)
if __name__=='__main__':
 import json;print(json.dumps(run(),indent=2))
