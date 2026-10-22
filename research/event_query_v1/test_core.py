import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,torch
from core import torch_posterior,score_feedback,target,restore_inputs
from stvg.global6 import posterior
torch.set_num_threads(1)
ids=list(range(17));records=[dict(frame_ids=ids[o::2])for o in range(2)]
torch.manual_seed(20261022);z=[torch.randn(6,1,len(r['frame_ids']),2,requires_grad=True)for r in records]
a=torch_posterior(records,z,ids);b=posterior(records,[v.detach()for v in z],ids)
assert np.max(abs(a.detach().numpy()-b))<2e-14
weights=torch.randn_like(a);loss=(a*weights).sum();grad=torch.autograd.grad(loss,z)[0];h=1e-3
with torch.no_grad():
    zz=[v.clone()for v in z];zz[0][2,0,3,1]+=h;up=(torch_posterior(records,zz,ids)*weights).sum()
    zz[0][2,0,3,1]-=2*h;down=(torch_posterior(records,zz,ids)*weights).sum()
assert abs(float((up-down)/(2*h)-grad[2,0,3,1]))<2e-5
positions=[0,2,4,6,8,10,12,16]
p,raw,pred,meta=score_feedback(np.zeros(8),ids,positions);assert meta['unknown']and not raw.any()and not pred.any()
assert np.max(abs(target(b,p,raw)-b))<1e-14
p,raw,pred,meta=score_feedback(np.array([0,0,2,2,2,0,0,0.]),ids,positions);assert pred.max()>0
# Additive/positive-scale normalization invariance, no calibration fitted to GT.
_,r2,p2,_=score_feedback(np.array([0,0,2,2,2,0,0,0.])*3+8,ids,positions);assert np.max(abs(raw-r2))<1e-14 and np.max(abs(pred-p2))<1e-14
x=torch.randn(1,1,256).expand(7,1,256);off=dict(inputs=dict(q=x.clone()),input_layouts=dict(q=dict(shape=list(x.shape),stride=list(x.stride()),storage_offset=0)))
y=restore_inputs(off,'cpu')['q'];assert torch.equal(x,y)and x.stride()==y.stride()
print('PASS: exact hull CDF versus original mass table; differentiable finite difference; unknown no-op; evidence normalization; expanded query layout')
