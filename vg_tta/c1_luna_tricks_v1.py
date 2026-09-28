"""Experimental optimizer transfers; original box objective and no label access."""
import hashlib
import math

import torch

from methods.decota_final_simplified_v1.objectives import SpatialLoss, generalized_iou
from methods.decota_final_simplified_v1.tensors import detached
from vg_tta.c1_enabling_tricks_v1 import TrickReplay, QUERY

FIT_ARMS = ['Scale06', 'QueryThird', 'Lookahead', 'Cautious', 'CautionNorm', 'PCGrad', 'PCNorm']
ARMS = FIT_ARMS + ['TailMean', 'Last']


def flat(values):
    return torch.cat([x.reshape(-1) for x in values])


@torch.no_grad()
def put(params, vector):
    offset = 0
    for p in params:
        n = p.numel(); p.copy_(vector[offset:offset+n].reshape_as(p)); offset += n


def pcgrad(matrix, key, step):
    seed = int(hashlib.sha256(f'pcgrad-v1|{key}|{step}'.encode()).hexdigest()[:12], 16)
    gen = torch.Generator().manual_seed(seed)
    out = matrix.clone(); orders = []; conflicts = 0
    for i in range(len(matrix)):
        order = [j for j in torch.randperm(len(matrix), generator=gen).tolist() if j != i]; orders.append(order)
        for j in order:
            dot = out[i] @ matrix[j]
            if dot < 0:
                out[i] -= dot / matrix[j].square().sum().clamp_min(1e-30) * matrix[j]
                conflicts += 1
    # Each individual loss already includes planned denominator 4.
    return out.sum(0), orders, conflicts


@torch.enable_grad()
def fit(base, initial, anchors, ids, key, arm):
    assert arm in FIT_ARMS
    base.restore(initial); replay = TrickReplay(base, ids, key, {})
    origin = replay.state(); params = [p for _, p in replay.named]
    lossfn = SpatialLoss(anchors, base.zero['boxes'])
    groups = [{'params':[p], 'lr':.03 if arm=='Scale06' else .05/3 if arm=='QueryThird' and n==QUERY else .05} for n,p in replay.named]
    optimizer = torch.optim.Adam(groups, betas=(.9,.999), eps=1e-8, weight_decay=0.)
    shadow = [p.detach().clone().requires_grad_(True) for p in params] if arm.startswith('PC') else None
    pcopt = torch.optim.Adam(shadow, lr=.05, betas=(.9,.999), eps=1e-8, weight_decay=0.) if shadow else None
    slow = flat(params).detach().clone()
    best, selected, chosen = math.inf, 0, detached(origin)
    path, gradient_calls = [], 0
    try:
        for step in range(1 if lossfn.empty else 11):
            optimizer.zero_grad(set_to_none=True)
            value = replay.values(); loss = lossfn(value['boxes']); lv = float(loss.detach())
            assert math.isfinite(lv)
            state = replay.state()
            if lv < best: best, selected, chosen = lv, step, state
            entry = dict(step=step, loss=lv, state=detached(state,'cpu'), boxes=detached(value['boxes'],'cpu'))
            path.append(entry)
            if lossfn.empty or step==10: break
            grads = torch.autograd.grad(loss, params, retain_graph=shadow is not None)
            gradient_calls += 1
            grad = flat(grads).detach(); mechanism = {}
            if shadow:
                pred=value['boxes'][lossfn.positions]
                terms=(5*(pred-lossfn.targets).abs().sum(-1)+2*(1-generalized_iou(pred,lossfn.targets)))/4
                matrix=[]
                for j in range(len(terms)):
                    gg=torch.autograd.grad(terms[j],params,retain_graph=j+1<len(terms))
                    matrix.append(flat(gg).detach());gradient_calls+=1
                matrix=torch.stack(matrix)
                assert torch.allclose(matrix.sum(0),grad,atol=3e-6,rtol=2e-4)
                projected, orders, conflicts=pcgrad(matrix,key,step)
                put(shadow,flat(params).detach()); offset=0
                for p in shadow:
                    n=p.numel();p.grad=projected[offset:offset+n].reshape_as(p).clone();offset+=n
                pcopt.step()
                pcraw=flat(shadow).detach()-flat(params).detach()
                mechanism.update(anchor_gradients=matrix.cpu(),projected_gradient=projected.cpu(),orders=orders,conflicts=conflicts,pcraw=pcraw.cpu(),pc_adam_steps=[int(pcopt.state[p]['step']) for p in shadow])
            for p,g in zip(params,grads): p.grad=g.detach()
            before=flat(params).detach().clone(); optimizer.step()
            raw=flat(params).detach()-before; direction=raw.clone()
            if arm in ('Cautious','CautionNorm'):
                masks=[]
                for p,g in zip(params,grads):
                    m=(optimizer.state[p]['exp_avg']*g>0).to(p.dtype)
                    masks.append(m/m.mean().clamp_min(1e-3))
                mask=flat(masks); cautious=raw*mask
                direction=cautious if arm=='Cautious' else raw*(cautious.norm()/raw.norm().clamp_min(1e-30))
                mechanism.update(mask=mask.cpu(),surviving=float((mask>0).float().mean()))
            if shadow:
                direction=pcraw if arm=='PCGrad' else raw*(pcraw.norm()/raw.norm().clamp_min(1e-30))
            if arm=='Lookahead' and (step+1)%5==0:
                slow=slow+.5*(before+raw-slow);direction=slow-before
            put(params,before+direction)
            after=flat(params).detach()
            entry['update']=dict(gradient=grad.cpu(),raw=raw.cpu(),direction=direction.cpu(),applied=(after-before).cpu(),
                                 adam_steps={n:int(optimizer.state[p]['step']) for n,p in replay.named},**mechanism)
            assert all(torch.isfinite(p).all() for p in params)
            del value,loss,grads
        return dict(arm=arm,initial=detached(origin,'cpu'),state=detached(chosen,'cpu'),selected_step=selected,
                    final=path[selected]['boxes'],path=path,skipped=lossfn.empty,gradient_calls=gradient_calls,
                    parameter_count=1792,GT_used=False)
    finally:
        replay.restore(origin)


@torch.no_grad()
def tail_mean(base, initial, ids, key, baseline):
    replay=TrickReplay(base,ids,key,{})
    states=[v['state'] for v in baseline['path'][-5:]]
    mean={n:torch.stack([s[n] for s in states]).mean(0) for n in initial}
    replay.restore(mean)
    try:
        boxes=detached(replay.values()['boxes'],'cpu')
        return dict(arm='TailMean',state=mean,final=boxes,averaged_steps=[v['step'] for v in baseline['path'][-5:]],
                    skipped=baseline['skipped'],gradient_calls=0,GT_used=False)
    finally: replay.restore(initial)
