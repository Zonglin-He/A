"""Bounded experimental interfaces. No label access or production registration."""
import hashlib
import math
import torch
from methods.decota_final_simplified_v1.objectives import SpatialLoss, generalized_iou
from methods.decota_final_simplified_v1.tensors import ParameterState, detached

QUERY = 'spatial.query_residual'
MODE = 'spatial.temporal_query_mode'
ARMS = {
    'C1': {}, 'Aux01': {'aux': .1}, 'Aux025': {'aux': .25}, 'Aux05': {'aux': .5},
    'Prog2': {'warmup': 2}, 'Prog4': {'warmup': 4},
    'Early8': {'early': 8}, 'Early6': {'early': 6},
    'LowFreq': {'mode': True}, 'Shuffled': {'mode': True, 'shuffle': True},
    'Relative': {'relative': True}, 'SmoothAbs': {'smooth': True},
}


def time_basis(ids, key, shuffle=False):
    t = torch.tensor(ids, dtype=torch.float64)
    tau = (t-t[0])/(t[-1]-t[0])
    phi = torch.cos(torch.pi*tau)
    phi = phi-phi.mean()
    phi = phi/phi.square().mean().sqrt().clamp_min(1e-12)
    if shuffle:
        seed = int(hashlib.sha256(('c1-enabling-basis|'+key).encode()).hexdigest()[:12], 16)
        phi = phi[torch.randperm(len(phi), generator=torch.Generator().manual_seed(seed))]
    return phi.float()


class TrickReplay(ParameterState):
    def __init__(self, base, ids, key, spec):
        self.base, self.ids, self.spec = base, ids, spec
        self.named = list(base.named)
        self.mode = torch.zeros_like(base.delta, requires_grad=True) if spec.get('mode') else None
        if self.mode is not None:
            self.named.append((MODE, self.mode))
        self.phi = time_basis(ids, key, spec.get('shuffle', False)).to(base.delta)
        self.initial = self.state()

    def values(self):
        layers = []
        with torch.autocast('cuda', enabled=False):
            for offset, inputs in enumerate(self.base.spatial_inputs):
                kwargs = dict(inputs)
                q = kwargs['query_tgt']
                q = q+self.base.delta.to(q.dtype)[None, None, :]
                if self.mode is not None:
                    q = q+self.phi[offset::2, None, None]*self.mode[None, None, :]
                kwargs['query_tgt'] = q
                layers.append(self.base.decoder.decoder(**kwargs).flatten(1, 2))
        allboxes = torch.stack([layers[i % 2][:, i//2] for i in range(len(self.ids))], dim=1).float()
        return {**self.base.zero, 'boxes': allboxes[-1], 'layers': allboxes}


class GeometryLoss:
    def __init__(self, standard, width, height, relative):
        self.standard, self.relative = standard, relative
        self.scale = standard.targets[:, 2:].clamp_min(standard.targets.new_tensor([1/width, 1/height]))

    def __call__(self, boxes):
        q = self.standard
        if q.empty:
            return boxes.sum()*0
        pred = boxes[q.positions]
        if self.relative:
            residual = torch.cat(((pred[:, :2]-q.targets[:, :2])/self.scale,
                                  torch.log((pred[:, 2:]+1e-6)/(q.targets[:, 2:]+1e-6))), -1)
        else:
            residual = pred-q.targets
        loc = torch.nn.functional.smooth_l1_loss(residual, torch.zeros_like(residual), reduction='none', beta=1.).sum(-1)
        return (5*loc+2*(1-generalized_iou(pred, q.targets))).sum()/4


def ln_active(spec, update):
    return update > spec.get('warmup', 0) and update <= spec.get('early', 10)


@torch.enable_grad()
def fit(base, initial, anchors, ids, key, spec, width, height, lr=.05):
    base.restore(initial)
    replay = TrickReplay(base, ids, key, spec)
    origin = replay.state()
    replay.initial = detached(origin)
    lossfn = SpatialLoss(anchors, base.zero['boxes'])
    geom = GeometryLoss(lossfn, width, height, spec.get('relative', False)) if spec.get('relative') or spec.get('smooth') else None
    opt = torch.optim.Adam([v for _, v in replay.named], lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    path, best, selected, backwards, projections = [], math.inf, 0, 0, 0
    chosen = detached(origin)
    startflags = [(p, p.requires_grad) for _, p in replay.named]
    grad_audit = None
    try:
        for step in range(1 if lossfn.empty else 11):
            active = ln_active(spec, step+1)
            for name, param in replay.named:
                param.requires_grad_(name in (QUERY, MODE) or active)
            opt.zero_grad(set_to_none=True)
            value = replay.values()
            final_loss = lossfn(value['boxes'])
            aux_loss = lossfn(value['layers'][2])
            train_loss = (geom(value['boxes']) if geom else final_loss)+spec.get('aux', 0.)*aux_loss
            assert torch.isfinite(train_loss), 'Nonfinite objective'
            lv = float(final_loss.detach())
            if lv < best:
                best, selected, chosen = lv, step, replay.state()
            with torch.no_grad():
                ll = [float(lossfn(b)) for b in value['layers']]
            entry = dict(step=step, loss=lv, train_loss=float(train_loss.detach()), layer_losses=ll,
                         boxes=detached(value['boxes'], 'cpu'), layers=detached(value['layers'], 'cpu'),
                         state=detached(replay.state(), 'cpu'))
            path.append(entry)
            if lossfn.empty or step == 10:
                break
            if step == 0 and spec.get('aux'):
                gg = torch.autograd.grad(aux_loss, [p for _, p in replay.named], retain_graph=True, allow_unused=True)
                g6 = torch.autograd.grad(final_loss, base.delta, retain_graph=True)[0]
                assert all(g is None or torch.count_nonzero(g) == 0 for (n, _), g in zip(replay.named, gg) if n != QUERY)
                grad_audit = dict(aux_query_norm=float(gg[0].norm()), final_query_norm=float(g6.norm()),
                                  cosine=float(torch.nn.functional.cosine_similarity(gg[0][None], g6[None])) if gg[0].norm()*g6.norm()>0 else None,
                                  aux_last_LN_direct_zero=True)
            previous = replay.state()
            train_loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for _, p in replay.named)
            opt.step()
            backwards += 1
            if replay.mode is not None:
                with torch.no_grad():
                    bound = .5*base.delta.norm()
                    norm = replay.mode.norm()
                    if norm > bound:
                        replay.mode.mul_(bound/norm.clamp_min(1e-30)); projections += 1
            now = replay.state()
            if not active:
                assert all(torch.equal(now[n], previous[n]) for n in now if n not in (QUERY, MODE))
            entry['update'] = dict(ln_active=active,
                query_norm=float((now[QUERY]-previous[QUERY]).norm()),
                ln_norm=float(torch.cat([(now[n]-previous[n]).flatten() for n in now if n not in (QUERY, MODE)]).norm()),
                mode_norm=float(now[MODE].norm()) if MODE in now else 0.,
                query_total_norm=float(now[QUERY].norm()),
                adam_steps={n:int(opt.state[p].get('step', 0)) for n,p in replay.named})
            del value, train_loss, final_loss, aux_loss
        return dict(initial=detached(origin,'cpu'),state=detached(chosen,'cpu'),selected_step=selected,
                    final=path[selected]['boxes'],path=path,backwards=backwards,skipped=lossfn.empty,
                    parameter_count=sum(p.numel() for _,p in replay.named),grad_audit=grad_audit,
                    basis=detached(replay.phi,'cpu') if replay.mode is not None else None,
                    projections=projections,spec=spec,GT_used=False)
    finally:
        for param, flag in startflags:
            param.requires_grad_(flag)
        replay.restore(origin)


@torch.no_grad()
def reinsert(model, batch, ids, records, state, basis, expected):
    from methods.decota_final_simplified_v1.backbone import inserted_state, offset_batch
    base = {k:v for k,v in state.items() if k != MODE}
    mode = state.get(MODE)
    boxes, logits, count = [], [], [0]
    with inserted_state(model, base):
        def add_mode(module, args, kwargs):
            count[0] += 1
            if mode is not None and count[0] % 2 == 0:
                offset = count[0]//2-1
                q = kwargs['query_tgt']
                kwargs = dict(kwargs)
                kwargs['query_tgt'] = q+basis[offset::2,None,None].to(q)*mode[None,None,:].to(q)
            return args, kwargs
        hook = model.ground_decoder.decoder.register_forward_pre_hook(add_mode, with_kwargs=True)
        try:
            for offset in (0,1):
                view = offset_batch(batch, offset)
                with torch.autocast('cuda',dtype=torch.float16):
                    out = model(view['videos'],view['texts'],view['targets'],iteration_rate=-1)
                boxes.append(out['pred_boxes']);logits.append(out['pred_sted'])
        finally:
            hook.remove()
    merged = torch.stack([boxes[i%2][i//2] for i in range(len(ids))]).float().cpu()
    assert torch.equal(merged, expected), ('Full-model reinsertion differs', float((merged-expected).abs().max()))
    return logits
