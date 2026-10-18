"""Small source-free mathematical contracts, no training or dataset access."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    import torch
    from vg_tta.stvg_joint_posterior_repair_v1 import PreconditionedJointPrior,rollout
    from vg_tta.stvg_joint_posterior_v1 import observation_loglik
    torch.set_num_threads(4);torch.manual_seed(193)
    mean=torch.zeros(33,4);scale=torch.ones(33,4)*.7
    net=PreconditionedJointPrior(mean,scale).double().eval().requires_grad_(False)
    c=dict(maps=torch.randn(1,512,32,8,8,dtype=torch.float64),query=torch.randn(1,5,256,dtype=torch.float64),qmask=torch.zeros(1,5,dtype=torch.bool),context=torch.randn(1,32,512,dtype=torch.float64),unit=torch.tensor([.002],dtype=torch.float64),native=torch.randn(1,33,4,dtype=torch.float64)*.1)
    c['native'][:,0,2:]=0;x=c['native'].clone()
    with torch.no_grad():out=net(x,torch.tensor([1e-8],dtype=torch.float64),c)
    assert float((out-x).abs().max())<1e-6
    sig=torch.tensor([.3],dtype=torch.float64);skip,cout,cin=net.coefficients(sig,1)
    assert torch.allclose(skip,torch.full_like(skip,.49/.58))
    assert torch.allclose(cout,torch.full_like(cout,.3*.7/(.58**.5)))
    assert torch.allclose(cin,torch.full_like(cin,1/(.58**.5)))
    mask=torch.ones_like(x);mask[:,0,2:]=0
    test=net.weighted_loss(x+.05,x,sig,mask)
    assert abs(float(test)-(.05**2/float(cout[0,0,0]**2)))<1e-8
    # Full chain-rule derivative through clean model, not detached x0 guidance.
    obs=dict(time=torch.tensor([.6,.7],dtype=torch.float64),boxes=torch.tensor([[.55,.45,.52,.48],[.5,.5,.5,.5]],dtype=torch.float64));prior=torch.tensor([.5,.8],dtype=torch.float64)
    xx=x.clone().requires_grad_(True);clean=net(xx,sig,c);ll=observation_loglik(clean,c,obs,prior,.01).sum();g,=torch.autograd.grad(ll,xx)
    assert g.norm()>0
    fd=[]
    for index in [(0,0,0),(0,0,1),(0,20,0),(0,25,2)]:
        h=1e-5;a=x.clone();b=x.clone();a[index]+=h;b[index]-=h
        with torch.no_grad():f=float((observation_loglik(net(a,sig,c),c,obs,prior,.01)-observation_loglik(net(b,sig,c),c,obs,prior,.01))/(2*h))
        expected=float(g[index]);assert abs(f-expected)<2e-5+abs(expected)*.002,(index,f,expected)
        fd.append(dict(coordinate=list(index),autograd=expected,finite_difference=f))
    # Empty guidance must reproduce unconditional path bit-for-bit.
    empty=dict(time=torch.empty(0),boxes=torch.empty(0,4))
    with torch.no_grad():a,_=rollout(net,c,27,levels=3);b,_=rollout(net,c,27,obs=empty,prior=torch.empty(0),variance=.01,levels=3)
    assert torch.equal(a,b)
    out=ROOT/'artifacts/stvg_joint_posterior_repair_v1/CPU_CONTRACT_COMPLETION.json';out.parent.mkdir(exist_ok=True,parents=True)
    assert not out.exists();out.write_text(json.dumps(dict(status='pass',EDM_coefficients_and_matched_loss=True,low_noise_identity_limit=True,clean_estimate_chain_rule_FD=fd,empty_observation_exact=True,new_training_steps=0,GPU_initialized=False),indent=2)+'\n')
    assert not torch.cuda.is_initialized();print('CPU_CONTRACT pass4FD identity coefficients loss empty')

if __name__=='__main__':main()
