import copy,time
import torch
from .geometry import corrected,local_cost
from .tensors import transfer

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


