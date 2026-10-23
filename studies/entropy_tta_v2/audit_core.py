"""Independent CPU checks for fixed entropy loss and original-layout restoration."""
from pathlib import Path
import importlib.util,math,json
import numpy as np
import torch
p=Path(__file__).with_name('run.py');spec=importlib.util.spec_from_file_location('experiment',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
x=[torch.zeros(1,4,2,requires_grad=True),torch.zeros(1,5,2,requires_grad=True)];v=m.entropy(x);assert abs(float(v)-.5*(math.log(4)+math.log(5)))<1e-6;v.backward();e=max(float(z.grad.abs().max())for z in x);assert e<1e-7
trace=dict(uniform_entropy=float(v),uniform_gradient_max=e)
g=torch.Generator().manual_seed(20261024);x=[torch.randn(1,7,2,generator=g,requires_grad=True),torch.randn(1,6,2,generator=g,requires_grad=True)];v=m.entropy(x);ref=np.mean([float(-(torch.softmax(z.detach().double(),dim=1)*torch.log_softmax(z.detach().double(),dim=1)).sum(1).mean())for z in x]);assert abs(float(v)-ref)<3e-7;trace['reference_error']=abs(float(v)-ref);v.backward();assert all(torch.isfinite(z.grad).all()and float(z.grad.norm())>0 for z in x)
x=[torch.tensor([[[30.,30.],[-30.,-30.]]])]*2;assert float(m.entropy(x))<1e-20
a=torch.arange(12,dtype=torch.float32).reshape(3,4).T;b=torch.arange(4,dtype=torch.float32)[None,:].expand(3,4);values={'a':a.clone(),'b':b.clone()};layouts={k:dict(shape=list(z.shape),stride=list(z.stride()),storage_offset=z.storage_offset())for k,z in [('a',a),('b',b)]};restored=m.restore_inputs({'inputs':values,'input_layouts':layouts},'cpu');assert all(torch.equal(restored[k],z)and restored[k].stride()==z.stride()for k,z in [('a',a),('b',b)])
print(json.dumps(dict(status='passed',entropy_and_gradient=True,original_strides_and_expansion=True,trace=trace)))
