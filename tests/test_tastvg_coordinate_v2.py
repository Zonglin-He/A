import numpy as np
import torch
from scripts.prepare_tastvg_coordinate_tuning_v2 import coarse,refine
from scripts.continue_tastvg_coordinate_v2 import choose
from vg_tta.tastvg_optuna_method_v1 import reverse_kl

def test_boundary_refinement_and_default_order():
 v=coarse(1e-6,1,[.005,.05,.5],.005)
 assert v[0]==.005 and len(v)==24
 assert len(refine(v,min(v)))==len(refine(v,max(v)))==6
 r=refine(v,.005);assert len(r)==12 and all(min(v)<x<max(v) for x in r) and not set(r)&set(v)

def test_failures_not_zero_and_ties_keep_earlier():
 a=[dict(state='FAIL',objective=None,number=0),dict(state='COMPLETE',objective=-.2,number=1),dict(state='COMPLETE',objective=-.2,number=2)]
 assert choose(a)['number']==1

def test_cold_teacher_underflow_is_not_nonfinite_loss():
 torch.manual_seed(22);c=(torch.rand(6,4)*.1+.3).requires_grad_();cs=torch.stack([c.detach()+.002*i for i in range(9)]);r=torch.arange(9,dtype=torch.float64)
 loss,p,q,dist=reverse_kl(c,cs,r,[5,3],.01)
 assert torch.isfinite(loss) and (q==0).any() and torch.isfinite(torch.autograd.grad(loss,c)[0]).all()
 d=dist.detach().double().numpy();rank=np.arange(8,-1,-1,dtype=float);lp=-d-np.logaddexp.reduce(-d);lq=-rank/.01;lq-=np.logaddexp.reduce(lq)
 np.testing.assert_allclose(np.sum(np.exp(lp)*(lp-lq)),float(loss.detach()),atol=2e-5,rtol=2e-5)
