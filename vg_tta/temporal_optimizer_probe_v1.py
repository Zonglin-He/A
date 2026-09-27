"""Isolated fifth-step interventions; no label or sample-selection inputs."""
import copy
import math
import torch

from vg_tta.fullspan_tta import fullspan_prior_loss
from vg_tta.tastvg_baseline_expansion import replay_temporal_head_native

BRANCHES = {
    'base5': dict(scale=1., reset_first=False, fp32_gradient=False),
    'last_lr0': dict(scale=0., reset_first=False, fp32_gradient=False),
    'last_lr0125': dict(scale=.125, reset_first=False, fp32_gradient=False),
    'last_lr025': dict(scale=.25, reset_first=False, fp32_gradient=False),
    'last_lr05': dict(scale=.5, reset_first=False, fp32_gradient=False),
    'last_reset_first': dict(scale=1., reset_first=True, fp32_gradient=False),
    'last_fp32_gradient': dict(scale=1., reset_first=False, fp32_gradient=True),
}


def cpu_state(head):
    return {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}


def logits(head, inputs, fp32=False):
    device = next(head.parameters()).device
    if not fp32:
        return [replay_temporal_head_native(head, h, device) for h in inputs]
    old = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device_type=device.type, enabled=False):
            return [head(h.detach().to(device=device, dtype=torch.float32))[-1] for h in inputs]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def task(head, inputs, fp32=False):
    values = logits(head, inputs, fp32)
    # Match the production gamma=0 differentiable anchor as well as task order.
    return torch.stack([fullspan_prior_loss(z) for z in values]).mean() + next(head.parameters()).float().sum()*0.


def snapshot(head, inputs):
    with torch.no_grad():
        zs = logits(head, inputs)
        losses = [float(fullspan_prior_loss(z)) for z in zs]
        value = float(torch.stack([fullspan_prior_loss(z) for z in zs]).mean())
        fp = logits(head, inputs, True)
        fp_loss = float(torch.stack([fullspan_prior_loss(z) for z in fp]).mean())
    assert math.isfinite(value) and math.isfinite(fp_loss)
    return dict(logits=[z.detach().cpu() for z in zs], loss=value,
                offset_losses=losses, fp32_logits=[z.detach().cpu() for z in fp],
                fp32_loss=fp_loss, head_state=cpu_state(head))


def update(head, inputs, optimizer, fp32_gradient=False):
    before = cpu_state(head)
    optimizer.zero_grad(set_to_none=True)
    loss = task(head, inputs, fp32_gradient)
    if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite pre-update loss')
    loss.backward()
    gradients = {}
    for name, parameter in head.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError('missing/nonfinite gradient: '+name)
        gradients[name] = parameter.grad.detach().cpu().clone()
    optimizer.step()
    after = snapshot(head, inputs)
    dot = g2 = d2 = 0.
    for name, g in gradients.items():
        delta = (after['head_state'][name].double()-before[name].double())
        dot += float((g.double()*delta).sum())
        g2 += float(g.double().square().sum())
        d2 += float(delta.square().sum())
    after['process'] = dict(loss_before=float(loss.detach()), gradient_norm=math.sqrt(g2),
        update_norm=math.sqrt(d2), gradient_dot_update=dot,
        gradient_update_cosine=dot/math.sqrt(g2*d2) if g2*d2 else None,
        gradients=gradients, fp32_gradient=fp32_gradient)
    return after


def choose_backtrack(branches, before_loss, tolerance=1e-12):
    for name in ('base5','last_lr05','last_lr025','last_lr0125','last_lr0'):
        loss = branches[name]['loss']
        if not math.isfinite(loss): raise ValueError('nonfinite candidate loss')
        if loss <= before_loss+tolerance: return name
    raise AssertionError('zero last update must preserve pre-update loss')


def probe(source_head, inputs, lr):
    if not math.isfinite(lr) or lr < 0: raise ValueError('invalid lr')
    if len(inputs) != 2: raise ValueError('exactly two offset inputs required')
    source_state = cpu_state(source_head)
    head = copy.deepcopy(source_head).eval().requires_grad_(True)
    inputs = [h.detach() for h in inputs]
    optimizer = torch.optim.AdamW(head.parameters(),lr=lr,eps=1e-4,weight_decay=0.)
    path = [snapshot(head, inputs)]
    for _ in range(4): path.append(update(head, inputs, optimizer))
    state4 = copy.deepcopy(head.state_dict())
    optimizer4 = copy.deepcopy(optimizer.state_dict())
    branches = {}
    for name, spec in BRANCHES.items():
        head.load_state_dict(state4)
        optimizer.load_state_dict(copy.deepcopy(optimizer4))
        for group in optimizer.param_groups: group['lr'] = lr*spec['scale']
        if spec['reset_first']:
            for state in optimizer.state.values(): state['exp_avg'].zero_()
        branches[name] = update(head, inputs, optimizer, spec['fp32_gradient'])
        branches[name]['configuration'] = dict(spec)
    path.append(branches['base5'])
    assert all(torch.equal(v,branches['last_lr0']['head_state'][k]) for k,v in path[4]['head_state'].items())
    assert branches['last_lr0']['loss'] == path[4]['loss']
    best = min(range(6),key=lambda i:(path[i]['loss'],i))
    head.load_state_dict(path[best]['head_state'])
    restored = snapshot(head, inputs)
    assert all(torch.equal(a,b) for a,b in zip(restored['logits'],path[best]['logits']))
    assert all(torch.equal(source_state[k],v) for k,v in cpu_state(source_head).items())
    # Model state snapshots are CPU; optimizer state is explicitly transferred.
    opt_cpu = copy.deepcopy(optimizer4)
    for state in opt_cpu['state'].values():
        for k,v in state.items():
            if torch.is_tensor(v): state[k]=v.detach().cpu()
    return dict(path=path,branches=branches,selected_best_step=best,
        best_restore_exact=True,selected_backtrack=choose_backtrack(branches,path[4]['loss']),
        source_unchanged=True,optimizer4=opt_cpu,GT_used=False)
