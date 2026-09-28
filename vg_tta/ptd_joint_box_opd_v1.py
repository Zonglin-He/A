"""Exact marginal and unbiased sampled marginal/joint score-function updates."""
import copy,hashlib,itertools,time
import torch
from vg_tta.ptd_spatial_adapter_ab_v1 import Adapter
SEEDS=[20260924,20260925,20260926]
ARMS=['D1','J1','J2']

def sampling_seed(key,seed,step):return int(hashlib.sha256(f'joint-opd|{key}|{seed}|{step}'.encode()).hexdigest()[:12],16)
def sampled(lp,key,seed,step,m=4):
    gen=torch.Generator().manual_seed(sampling_seed(key,seed,step))
    return torch.multinomial(lp.detach().exp().reshape(-1,lp.shape[-1]),m,replacement=True,generator=gen).reshape(lp.shape[0],lp.shape[1],m).permute(0,2,1)
def gather(lp,acts):return lp[:,None].expand(-1,acts.shape[1],-1,-1).gather(-1,acts[...,None]).squeeze(-1)
def surrogate(sample_lp,teacher_segments,joint):
    costs=sample_lp.detach()-teacher_segments.detach()
    returns=costs.flip(-1).cumsum(-1).flip(-1) if joint else costs
    assert returns.shape[1]==4
    baseline=(returns.sum(1,keepdim=True)-returns)/3
    advantage=(returns-baseline).detach()
    return (sample_lp*advantage).mean(),dict(costs=costs,returns=returns,loo_baseline=baseline,advantage=advantage)

class Episode:
    def __init__(self,z,key,arm,seed):
        self.cpu_start=time.process_time()
        self.z=z;self.key=key;self.arm=arm;self.seed=seed
        self.model=Adapter(z['h'].shape[-1],seed=seed)
        self.opt=torch.optim.AdamW(self.model.parameters(),lr=.002,weight_decay=0.)
        self.history=[];self.updates=[];self.initial={k:v.clone() for k,v in self.model.state_dict().items()}
        assert not self.opt.state and torch.equal(self.model(z['h']),torch.zeros_like(z['logits']))
    def policy(self):return (self.z['logits']+self.model(self.z['h'])).log_softmax(-1)
    def snapshot(self,step):
        lp=self.policy()
        item=dict(step=step,state={k:v.detach().clone() for k,v in self.model.state_dict().items()},optimizer=copy.deepcopy(self.opt.state_dict()),logp=lp.detach().clone(),tokens=lp.argmax(-1).detach())
        self.history.append(item);return lp,item
    def update(self,step,qp=None,actions=None,teacher_segments=None):
        lp,item=self.snapshot(step)
        if self.arm=='D1':
            loss=(lp.exp()*(lp-qp.float())).sum(-1).mean();details=dict(exact_KL=float(loss.detach()))
        else:
            acts=sampled(lp,self.key,self.seed,step) if actions is None else actions
            assert torch.equal(acts,sampled(lp,self.key,self.seed,step))
            selected=gather(lp,acts)
            targets=gather(qp.float(),acts) if self.arm=='J1' else teacher_segments.float()
            loss,details=surrogate(selected,targets,self.arm=='J2')
            item.update(actions=acts,sampled_logp=selected.detach(),teacher_segments=targets,**details)
        self.opt.zero_grad(set_to_none=True);loss.backward()
        params=list(self.model.parameters());grad=torch.cat([p.grad.flatten() for p in params]);before=torch.cat([p.detach().flatten().clone() for p in params])
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.));self.opt.step();delta=torch.cat([p.detach().flatten() for p in params])-before
        assert torch.isfinite(delta).all()
        self.updates.append(dict(step=step+1,gradient=grad.detach().clone(),preclip_norm=norm,displacement=delta.clone(),displacement_norm=float(delta.norm()),surrogate_loss=float(loss.detach())))
    def finish(self):
        self.snapshot(3);end=copy.deepcopy(self.model);end.load_state_dict(self.history[-1]['state']);assert torch.equal(end(self.z['h']),self.model(self.z['h']))
        self.model.load_state_dict(self.initial);assert torch.equal(self.model(self.z['h']),torch.zeros_like(self.z['logits']))
        return dict(key=self.key,arm=self.arm,seed=self.seed,history=self.history,updates=self.updates,optimizer_initial_empty=True,reload_exact=True,reset_exact=True,
                    positions=self.z['positions'],native_time=self.z['interval'],GT_used=False,parameter_count=sum(p.numel() for p in self.model.parameters()),CPU_process_seconds=time.process_time()-self.cpu_start)

def synthetic_audit():
    # Exhaustive 3x3x3 common event space with correlated teacher.
    torch.manual_seed(82);logits=torch.randn(3,3,dtype=torch.float64,requires_grad=True)
    actions=torch.tensor(list(itertools.product(range(3),repeat=3)))
    raw=torch.randn(3,3,3,dtype=torch.float64);q=raw.flatten().softmax(0).reshape(3,3,3)
    lp=logits.log_softmax(-1);selected=lp[torch.arange(3)[None],actions];p=selected.sum(-1).exp()
    qt=q[actions[:,0],actions[:,1],actions[:,2]]
    loss=(p*(selected.sum(-1)-qt.log())).sum()/3;exact,=torch.autograd.grad(loss,logits,retain_graph=True)
    t0=q.sum((1,2))[actions[:,0]].log();t1=(q.sum(2)/q.sum((1,2))[:,None])[actions[:,0],actions[:,1]].log()
    t2=(q/q.sum(2)[:,:,None])[actions[:,0],actions[:,1],actions[:,2]].log();t=torch.stack([t0,t1,t2],-1)
    costs=selected.detach()-t;returns=costs.flip(-1).cumsum(-1).flip(-1)
    estimate=(p.detach()*(selected*returns).sum(-1)).sum()/3
    got,=torch.autograd.grad(estimate,logits,retain_graph=True)
    # Expected leave-one-out baseline over independent other trajectories is zero.
    baseline=(p.detach()[:,None]*returns).sum(0)
    centered=(p.detach()*(selected*(returns-baseline)).sum(-1)).sum()/3
    bg,=torch.autograd.grad(centered,logits,retain_graph=True)
    # Counterexample: omitting downstream conditional terms produces a wrong gradient.
    wrong=(p.detach()*(selected*costs).sum(-1)).sum()/3
    wg,=torch.autograd.grad(wrong,logits,retain_graph=True)
    # Adding any logZ constant cannot change the expected gradient.
    shifted=(p.detach()*(selected*(returns+17.)).sum(-1)).sum()/3
    zg,=torch.autograd.grad(shifted,logits)
    error=float((exact-got).abs().max());be=float((exact-bg).abs().max());ze=float((exact-zg).abs().max());we=float((exact-wg).abs().max())
    assert max(error,be,ze)<1e-12 and we>1e-4
    # Verify actual M4 code with a large fixed Monte Carlo sample, no fitted model.
    gen=torch.Generator().manual_seed(89);idx=torch.multinomial(p.detach(),200000*4,replacement=True,generator=gen).reshape(-1,4)
    sel=actions[idx];slp=lp.detach()[torch.arange(3)[None,None],sel];seg=t[idx]
    c=slp-seg;ret=c.flip(-1).cumsum(-1).flip(-1);adv=ret-(ret.sum(1,keepdim=True)-ret)/3
    score=torch.nn.functional.one_hot(sel,3).double()-lp.detach().exp()[None,None]
    groups=(adv[...,None]*score).mean(1)/3
    mean=groups.mean(0);se=groups.std(0)/len(groups)**.5
    assert torch.all((mean-exact.detach()).abs()<6*se+1e-5)
    return dict(events=27,dimensions=3,classes=3,exact_return_to_go_gradient_max_error=error,independent_baseline_max_error=be,
        unknown_partition_gradient_max_error=ze,incorrect_no_downstream_gradient_error=we,M4_monte_carlo_groups=len(groups),
        M4_monte_carlo_max_error=float((mean-exact.detach()).abs().max()),passed=True,no_GPU=True,no_GT=True)
