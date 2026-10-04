"""Meaningful CPU contracts for no-op, multi-box geometry and soft bias math."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from vg_tta.tastvg_privileged_attention_p0_v1 import proposal_support, evidence_field


def run():
    checks=0
    empty=proposal_support([],[]);f=evidence_field(empty,3,4)
    assert not f['active'] and torch.equal(f['field'],torch.ones(12));checks+=1
    b=[[.2,.4,.2,.4],[.8,.6,.2,.4],[.5,.5,0.,.2],[.4,.4,.2,.2],[float('nan'),.5,.4,.4]]
    s=[.7,.5,.9,.2,.8];p=proposal_support(b,s)
    assert p['indices'].tolist()==[0,1] and len(p['boxes'])==2;checks+=1
    weights=np.exp(np.array([.7,.5])-.7);weights/=weights.sum()
    np.testing.assert_allclose(p['weights'],weights,rtol=1e-6);checks+=1
    pad=torch.tensor([[False,False,False,True],[False,False,False,True],[True,True,True,True]])
    f=evidence_field(p,3,4,pad)
    yy,xx=np.meshgrid((np.arange(3)+.5)/2,(np.arange(4)+.5)/3,indexing='ij')
    z=np.zeros((3,4))
    for box,w in zip(p['boxes'].numpy(),weights):
        z+=w*np.exp(-.5*(((xx-box[0])/(box[2]/2))**2+((yy-box[1])/(box[3]/2))**2))
    z[pad]=1
    np.testing.assert_allclose(f['field'].reshape(3,4),z,rtol=1e-5,atol=1e-7);checks+=1
    assert f['field'].min()>=0 and f['field'].max()<=1 and f['active'];checks+=1
    logits=torch.tensor([[1.,2.,-1.,.2],[.1,-1.,2.,1.]])
    m=torch.tensor([[.1,.7,.2,1.],[.5,.1,.9,1.]])
    actual=torch.softmax(logits+torch.log(m+1e-6),-1)
    expected=torch.softmax(logits,-1)*(m+1e-6);expected/=expected.sum(-1,keepdim=True)
    assert torch.allclose(actual,expected,atol=1e-7);checks+=1
    # Duplicates retain their empirical mass; they are not silently NMS/top-K'd.
    p=proposal_support([[.5,.5,.5,.5]]*3,[.4,.4,.4]);assert len(p['indices'])==3;checks+=1
    # Four requested positions are in the native interval, never GT or outside.
    from methods.decota_final_simplified_v1.observations import uniform_positions
    for interval in [(0,9),(3,4),(5,5)]:
        positions=uniform_positions(list(range(10)),interval,4)
        assert len(set(positions))==len(positions) and all(interval[0]<=x<=interval[1] for x in positions);checks+=1
    return dict(status='pass',CPU_contracts=checks,CUDA_initialized=torch.cuda.is_initialized())


if __name__=='__main__':print(run())
