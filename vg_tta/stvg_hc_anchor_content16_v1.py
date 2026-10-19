"""Independent observed-frame content states in a frozen spatial decoder.

The policy samples native logit box coordinates at observed frames, NOT content
states or actual decoder trajectories. Every policy mean is the actual current
decoder output. Its deterministic full trajectory is decoded after each update.
"""
import copy
import hashlib
import time
import numpy as np
import torch
from methods.decota_final_simplified_v1.tensors import detached
from methods.decota_final_simplified_v1.objectives import generalized_iou
from vg_tta.stvg_dino_query_closedloop32_v1 import corrected
from vg_tta.decota_spatial_opd_tunable_v1 import mean_coordinates, likelihood_loss


def inject(q, state, positions, view):
    """One vector at physical p maps only to offset p%2, local p//2."""
    slots = [i for i, p in enumerate(positions) if p % 2 == view]
    if not slots:
        return q
    local = torch.tensor([positions[i]//2 for i in slots], device=q.device)
    ids = torch.tensor(slots, device=q.device)
    return q + torch.zeros_like(q).index_add(0, local, state.index_select(0, ids)[:, None])


def local_cost(boxes, targets):
    return 5*(boxes-targets).abs().sum(-1) + 2*(1-generalized_iou(boxes, targets))


class Replay:
    def __init__(self, model, cache, source, E, positions, check):
        from vg_tta.stvg_opd_frozen_capture_cpu007 import transfer
        self.decoder = copy.deepcopy(model.ground_decoder.decoder).eval().requires_grad_(False)
        self.inputs = transfer(cache['spatial_inputs'], 'cuda')
        self.source = source.cuda()
        self.E = torch.as_tensor(E, dtype=torch.float64, device='cuda')
        self.positions = list(positions)
        self.state = torch.zeros(len(positions), 256, device='cuda', requires_grad=True)
        self.check = check
        self.calls = self.backwards = 0
        self.seconds = 0.
        with torch.no_grad():
            z = self.values()
        assert torch.equal(z['native'].cpu(), source)
        assert torch.equal(z['boxes'], self.E)

    def values(self):
        self.check(); start=time.perf_counter(); raw=[]
        with torch.autocast('cuda', enabled=False):
            for view, kw0 in enumerate(self.inputs):
                kw=dict(kw0)
                kw['query_tgt']=inject(kw0['query_tgt'], self.state, self.positions, view)
                raw.append(self.decoder(**kw).flatten(1,2)[-1])
            native=torch.stack([raw[p%2][p//2] for p in range(len(self.source))]).float()
            boxes=corrected(native, self.source, self.E)
        torch.cuda.synchronize(); self.calls+=1; self.seconds+=time.perf_counter()-start
        assert torch.isfinite(boxes).all() and (boxes[:,2:]>0).all()
        return dict(native=native, boxes=boxes)

    def mapped_samples(self, samples):
        b=samples.sigmoid().double();src=self.source[self.positions].double()[:,None];e=self.E[self.positions][:,None]
        return torch.cat((e[...,:2]+b[...,:2]-src[...,:2],
                          e[...,2:]*torch.exp(torch.log(b[...,2:])-torch.log(src[...,2:]))),-1)


@torch.no_grad()
def sample_feedback(rp, mu, target, alias, k, cfg):
    seed=int(hashlib.sha256(f'anchor-content16-v1|{alias}|{k}'.encode()).hexdigest()[:15],16)
    g=torch.Generator(device='cpu').manual_seed(seed)
    eps=torch.randn(len(mu),16,4,generator=g,dtype=torch.float32).to(mu)
    eps=torch.cat((eps,-eps),1)
    actions=(mu[:,None]+cfg['sigma']*eps).detach()
    boxes=rp.mapped_samples(actions)
    cost=local_cost(boxes, target[:,None])
    weights=torch.softmax(-cost/cfg['tau'],-1).detach()
    assert torch.isfinite(cost).all() and torch.isfinite(weights).all()
    return dict(proposal_mean=mu.detach().clone(), actions=actions, boxes=boxes,
                cost=cost, weights=weights, seed=seed,
                informative=bool((cost!=cost[:,:1]).any()))


@torch.enable_grad()
def fit(rp, anchors, alias, arm, cfg):
    assert arm in ['Direct','Fixed','Live']
    with torch.no_grad(): rp.state.zero_()
    opt=torch.optim.Adam([rp.state], lr=cfg['lr'], betas=(.9,.999), eps=1e-8, weight_decay=0.)
    target=torch.tensor([a['box'] for a in anchors],device='cuda',dtype=torch.float64).reshape(-1,4)
    assert [a['position'] for a in anchors]==rp.positions
    path=[];rounds=[];fixed=None;start=time.perf_counter();back=0
    try:
        for k in range(cfg['steps']+1):
            rp.check();v=rp.values()
            direct=local_cost(v['boxes'][rp.positions],target).mean() if anchors else v['boxes'].sum()*0
            path.append(dict(step=k, state=rp.state.detach().cpu().clone(),
                native=v['native'].detach().cpu(), boxes=v['boxes'].detach().cpu(), anchor_loss=float(direct.detach())))
            if k==0: assert torch.equal(v['boxes'],rp.E)
            if k==cfg['steps'] or not anchors: break
            rr=None;gm=None;error=None;double_error=None
            if arm=='Direct':
                loss=direct;informative=True
            else:
                mu=mean_coordinates(v['native'][rp.positions])
                rr=fixed if arm=='Fixed' and fixed is not None else sample_feedback(rp,mu.detach(),target,alias,k,cfg)
                if arm=='Fixed' and fixed is None: fixed=rr
                loss=likelihood_loss(mu,rr['proposal_mean'],rr['actions'],rr['weights'],cfg['sigma'])
                gm=torch.autograd.grad(loss,mu,retain_graph=True)[0]
                expected=((mu.detach()-rr['proposal_mean'])/cfg['sigma']**2-
                    ((rr['weights']-1/32)[:,:,None]*(rr['actions']-mu.detach()[:,None])).sum(1)/cfg['sigma']**2)/len(mu)
                error=float((gm-expected).abs().max())
                tolerance=2e-5+8*torch.finfo(mu.dtype).eps*max(1.,float(gm.abs().max()))
                assert error<tolerance
                if error>=2e-5:
                    md=mu.detach().double().requires_grad_(True)
                    ld=likelihood_loss(md,rr['proposal_mean'].double(),rr['actions'].double(),rr['weights'].double(),cfg['sigma'])
                    gd=torch.autograd.grad(ld,md)[0]
                    double_error=float((gd-gm.double()).abs().max());assert double_error<tolerance
                informative=rr['informative']
            opt.zero_grad(set_to_none=True);before=rp.state.detach().clone();gradient=None
            if informative:
                gradient=torch.autograd.grad(loss,rp.state)[0]
                assert torch.isfinite(gradient).all()
                rp.state.grad=gradient.detach();opt.step();back+=1;rp.backwards+=1
            assert torch.isfinite(rp.state).all()
            rounds.append(dict(step=k,updated=informative,loss=float(loss.detach()),
                rollout=detached(rr,'cpu'),policy_mean_gradient=detached(gm,'cpu'),
                analytic_error=error,double_error=double_error,
                content_gradient=detached(gradient,'cpu'),content_step=(rp.state.detach()-before).cpu(),
                Adam=detached(opt.state_dict(),'cpu')))
        assert all(not p.requires_grad and p.grad is None for p in rp.decoder.parameters())
        return dict(arm=arm,path=path,rounds=rounds,final=path[-1]['boxes'],
            positions=rp.positions,parameters=rp.state.numel(),steps=back,seconds=time.perf_counter()-start,
            sampled_policy='observed native logit boxes, not decoder latent states',
            weights_frozen=True,selected_step=path[-1]['step'])
    finally:
        with torch.no_grad(): rp.state.zero_()


def properties():
    q=torch.zeros(4,1,256,dtype=torch.float64)
    state=torch.zeros(3,256,dtype=torch.float64,requires_grad=True);pos=[0,3,6]
    with torch.no_grad(): state[:,0]=torch.tensor([1.,2.,3.])
    a=inject(q,state,pos,0);b=inject(q,state,pos,1)
    assert a[:,0,0].tolist()==[1.,0.,0.,3.] and b[:,0,0].tolist()==[0.,2.,0.,0.]
    loss=a.sum()+b.sum();g=torch.autograd.grad(loss,state)[0]
    assert torch.equal(g,torch.ones_like(g))
    mu=torch.zeros(2,4,dtype=torch.float64,requires_grad=True)
    eps=torch.linspace(-.7,.7,2*16*4,dtype=torch.float64).reshape(2,16,4)
    samples=torch.cat((eps,-eps),1)*.025;weights=torch.full((2,32),1/32,dtype=torch.float64)
    l=likelihood_loss(mu,mu.detach(),samples,weights,.025)
    assert l==0 and torch.equal(torch.autograd.grad(l,mu)[0],torch.zeros_like(mu))
    # Localization cost is informative even for non-overlapping small boxes.
    bs=torch.tensor([[[.1,.1,.05,.05],[.2,.1,.05,.05]]],dtype=torch.float64)
    target=torch.tensor([[[.9,.9,.05,.05]]],dtype=torch.float64)
    costs=local_cost(bs,target);assert costs[0,0]!=costs[0,1]
    return dict(status='pass',physical_offset_scatter=True,independent_state_gradient=True,
        equal_teacher_student_zero_gradient=True,nonoverlap_feedback_not_flat=True,
        no_GPU=True,model_weights_untouched=True)
