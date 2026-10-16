"""R42: isolated pre-LN replay, with fixed teachers and both native decoders.

No labels, source identifiers, evaluation code, or production registry writes.
Only the frozen prefix before encoder.norm is cached. Downstream hard gates
are recomputed in value, with their original nondifferentiable semantics.
"""
import copy
import math
import time
import torch
from vg_tta.temporal_capacity_v1 import detached
from vg_tta.fullspan_tta import fullspan_prior_loss
from vg_tta.metrics import generalized_box_iou_aligned_cxcywh
from vg_tta.time_space_repair_v1 import distribution_loss, legal_logp


def capture_shared(model, batch):
    from scripts.run_time_space_repair_v1 import capture
    prefix, grounds, outputs = [], [], []
    hooks = [
        model.ground_encoder.encoder.norm.register_forward_pre_hook(
            lambda m, a: prefix.append(detached(a[0]))),
        model.ground_decoder.register_forward_pre_hook(
            lambda m, a, kw: grounds.append(detached(kw)), with_kwargs=True),
        model.register_forward_hook(lambda m, a, o: outputs.append(detached(o))),
    ]
    try:
        base, inputs, records, spatial_caches, action = capture(model, batch)
    finally:
        for h in hooks:
            h.remove()
    assert len(prefix) == len(outputs) == 2 and len(grounds) == 4
    views = [dict(prefix=prefix[j], info=grounds[2*j]['encoded_info'],
                  vis_pos=grounds[2*j]['vis_pos'], output=outputs[j]) for j in range(2)]
    return base, inputs, records, spatial_caches, action, views


def spatial_data(boxes, anchors):
    if not anchors:
        return boxes.sum()*0.
    pos = torch.tensor([a['position'] for a in anchors], device=boxes.device)
    target = torch.tensor([a['box'] for a in anchors], device=boxes.device, dtype=boxes.dtype)
    pred = boxes[pos]
    return (5*(pred-target).abs().sum(-1) +
            2*(1-generalized_box_iou_aligned_cxcywh(pred, target))).mean()


def norm_of(tensors):
    return math.sqrt(sum(float(x.detach().double().square().sum()) for x in tensors if x is not None))


class SharedReplay:
    def __init__(self, model, views, n, scope='shared'):
        self.model = model
        self.views = views
        self.n = n
        self.scope = scope
        self.norm = copy.deepcopy(model.ground_encoder.encoder.norm).eval().requires_grad_(False)
        self.decoder = copy.deepcopy(model.ground_decoder).eval().requires_grad_(False)
        self.head = copy.deepcopy(model.temp_embed).float().eval().requires_grad_(False)
        self.delta = torch.zeros(256, device=next(self.norm.parameters()).device)
        self.named = []
        self.groups = {}
        if 'shared' in scope:
            self._add('shared', list(self.norm.named_parameters()))
        if 'head' in scope:
            self._add('head', list(self.head.named_parameters()))
        if scope in ('spatial', 'spatial_ln', 'shared_spatial'):
            if scope != 'spatial_ln':
                self._add('spatial', [('query_residual', self.delta)])
            last = len(self.decoder.decoder.layers)-1
            pp = []
            for name in ('norm1', 'norm3', 'norm4'):
                pp += [(f'layers.{last}.{name}.{k}', v) for k, v in
                       getattr(self.decoder.decoder.layers[last], name).named_parameters()]
            self._add('spatial', pp)
        assert self.named, scope
        self.initial = self.state()

    def _add(self, group, pp):
        for name, param in pp:
            param.requires_grad_(True)
            name = group+'.'+name
            self.named.append((name, param))
            self.groups[name] = group

    def state(self):
        return {n: p.detach().clone() for n, p in self.named}

    def restore(self, state):
        with torch.no_grad():
            for n, p in self.named:
                p.copy_(state[n].to(p.device))
                p.grad = None

    def values(self, tensor_shift=0.):
        boxes, zs, actions, gates = [], [], [], []
        with torch.autocast('cuda', dtype=torch.float16):
            for v in self.views:
                H = self.norm(v['prefix'])
                if tensor_shift:
                    direction = torch.sin(torch.arange(H.shape[-1], device=H.device).float())
                    H = H + (float(tensor_shift)*direction).to(H.dtype)
                info = dict(v['info'])
                info.update(encoded_feature=H, frames_cls=H.mean(0), videos_cls=H.mean(0).mean(0))
                h, w = info['fea_map_size']; count = h*w; nf = H.shape[1]
                fm = H[-count:].permute(1,2,0).reshape(nf,256,h,w).detach()
                fa = H[:count].permute(1,2,0).reshape(nf,256,h,w).detach()
                ft = H[count:-count].mean(1).unsqueeze(0).detach()
                # Native TTS/ASA use detached representations. Parameters stay frozen.
                with torch.no_grad():
                    lm = self.model.t_temporal_clas(fm, ft)
                    la = self.model.s_temporal_clas(fa, ft)
                    prob = (lm.sigmoid()+la.sigmoid())/2
                    fallback = self._indices(prob > 0)
                    first = self._indices(prob > self.model.theta) or fallback
                    itq, isq = self._queries(H, fm, fa, ft, first, count)
                    _, hidden = self.decoder(encoded_info=info, vis_pos=v['vis_pos'], itq=itq, isq=isq)
                    ap = self.model.action_embed(hidden)[-1].squeeze().sigmoid()
                    second = self._indices(ap > .5) or fallback
                itq, isq = self._queries(H, fm, fa, ft, second, count)
                # Match old spatial residual placement exactly: add after expand,
                # not before itq/isq arithmetic, and only in the second decoder.
                def add_query(module, args, kwargs):
                    kw = dict(kwargs)
                    q = kw['query_tgt']
                    kw['query_tgt'] = q + self.delta.to(q.dtype)[None,None,:]
                    return args, kw
                hook = self.decoder.decoder.register_forward_pre_hook(add_query, with_kwargs=True)
                try:
                    pos, hidden = self.decoder(encoded_info=info, vis_pos=v['vis_pos'], itq=itq, isq=isq)
                finally:
                    hook.remove()
                boxes.append(pos.flatten(1,2)[-1])
                zs.append(self.head(hidden)[-1])
                actions.append(self.model.action_embed(hidden)[-1])
                gates.append(dict(first=first, second=second))
        b = torch.stack([boxes[i%2][i//2] for i in range(self.n)]).float()
        return dict(boxes=b, logits=zs, actions=actions, gates=gates)

    @staticmethod
    def _indices(mask):
        ids = torch.nonzero(mask.squeeze()).squeeze().tolist()
        return [ids] if isinstance(ids, int) else ids

    def _queries(self, H, fm, fa, ft, chosen, count):
        with torch.no_grad():
            _, am = self.model.t_spatial_clas(fm[chosen], ft[:,:1])
            _, aa = self.model.s_spatial_clas(fa[chosen], ft[:,:1])
        itq = (H[-count:].permute(1,0,2)[chosen]*am.unsqueeze(2)).mean((0,1))
        isq = (H[:count].permute(1,0,2)[chosen]*aa.unsqueeze(2)).mean((0,1))
        return itq, isq

    def loss(self, value, kind, anchors, reference, signals, eta):
        if kind == 'spatial':
            data = spatial_data(value['boxes'], anchors)
            reg = sum((p-self.initial[n]).square().sum() for n,p in self.named)*1e-4
            return data+reg
        if kind == 'coverage':
            return torch.stack([fullspan_prior_loss(z) for z in value['logits']]).mean()
        if kind == 'boundary':
            return distribution_loss(value['logits'], reference, signals, eta)[0]
        raise ValueError(kind)


def fit_shared(interface, kind, anchors, signals, reference, eta, lrs,
               steps, records, ids, base):
    """One reset episode. Selection uses only its fixed unlabelled objective.

    Spatial and boundary retain F25 backtracking; coverage retains native
    AdamW steps without line search. Every budget includes its step-zero state.
    Both complete outputs, including hard gate values, are kept for every step.
    """
    from vg_tta.time_space_repair_v1 import paired_readout
    it=interface;it.restore(it.initial);start=time.perf_counter()
    groups=sorted(set(it.groups.values()))
    pg=[dict(params=[p for n,p in it.named if it.groups[n]==g],lr=lrs[g]) for g in groups]
    cls=torch.optim.Adam if kind=='spatial' else torch.optim.AdamW
    opt=cls(pg,eps=1e-8 if kind=='spatial' else 1e-4,weight_decay=0.)
    path=[];best=math.inf;best_step=0;saved=it.state();failure=None
    empty=kind=='spatial' and not anchors
    def loss(value):return it.loss(value,kind,anchors,reference,signals,eta)
    def moved(old):
        return {g:norm_of([p.detach()-old[n] for n,p in it.named if it.groups[n]==g]) for g in groups}
    original=dict(boxes=base['raw_boxes'].float().cpu(),
                  logits=[v['output']['pred_sted'].cpu() for v in it.views])
    for step in range((0 if empty else steps)+1):
        opt.zero_grad(set_to_none=True);v=it.values()
        if not all(bool(torch.isfinite(t).all()) for t in [v['boxes']]+v['logits']):
            failure='nonfinite_outputs';break
        objective=loss(v);lv=float(objective.detach())
        if not math.isfinite(lv):failure='nonfinite_loss';break
        if step==0:
            assert torch.equal(v['boxes'].detach().cpu(),original['boxes'])
            assert all(torch.equal(z.detach().cpu(),q) for z,q in zip(v['logits'],original['logits']))
        if lv<best:best=lv;best_step=step;saved=it.state()
        st=dict(step=step,loss=lv,boxes=v['boxes'].detach().cpu(),
                logits=[z.detach().cpu() for z in v['logits']],
                actions=[a.detach().cpu() for a in v['actions']],gates=copy.deepcopy(v['gates']),
                **paired_readout(v['logits'],records,ids,base),
                state={n:p.detach().cpu().clone() for n,p in it.named},group_delta=moved(it.initial))
        path.append(st)
        if step==steps or empty:break
        objective.backward()
        grads={n:None if p.grad is None else float(p.grad.norm()) for n,p in it.named}
        st['gradient_norms']=grads
        if any(g is None or not math.isfinite(g) for g in grads.values()):
            failure='missing_or_nonfinite_gradient';break
        st['group_gradient_norms']={g:norm_of([p.grad for n,p in it.named if it.groups[n]==g]) for g in groups}
        if step==0 and 'shared' in groups:
            st['shared_gradient']=torch.cat([p.grad.detach().reshape(-1).float().cpu() for n,p in it.named if it.groups[n]=='shared'])
        before=it.state();ob=copy.deepcopy(opt.state_dict());opt.step();after=it.state()
        if not all(bool(torch.isfinite(p).all()) for n,p in it.named):
            failure='nonfinite_parameters';break
        trials=[];accepted=kind=='coverage'
        if kind!='coverage':
            tol=1e-7 if kind=='spatial' else 1e-9
            for alpha in (1.,.5,.25,.125):
                it.restore(after if alpha==1. else {n:before[n]+alpha*(after[n]-before[n]) for n in before})
                with torch.no_grad():candidate=it.values();newloss=loss(candidate)
                finite=bool(torch.isfinite(newloss))
                trials.append(dict(alpha=alpha,loss=float(newloss) if finite else None))
                if finite and float(newloss)<lv-tol:accepted=True;break
            if not accepted:it.restore(before);opt.load_state_dict(ob)
        st.update(accepted=accepted,trials=trials,group_update=moved(before))
        del objective,v
    if failure:saved=it.initial;best_step=0
    assert path,'invalid initial state'
    it.restore(saved)
    with torch.no_grad():v=it.values();final_loss=float(loss(v))
    chosen=path[best_step]
    assert torch.equal(v['boxes'].cpu(),chosen['boxes'])
    assert all(torch.equal(a.cpu(),b) for a,b in zip(v['logits'],chosen['logits']))
    assert v['gates']==chosen['gates']
    assert all(torch.equal(p.detach().cpu(),chosen['state'][n]) for n,p in it.named)
    result=dict(kind=kind,scope=it.scope,steps=steps,lrs=lrs,eta=eta if kind=='boundary' else None,
                anchors=anchors if kind=='spatial' else [],parameters={n:p.numel() for n,p in it.named},
                parameter_count=sum(p.numel() for n,p in it.named),path=path,best_step=best_step,
                best_restore_exact=True,failure=failure,skipped='no_anchors' if empty else None,
                final_loss=final_loss,group_delta=moved(it.initial),GT_online=False,
                evidence_frozen=True,teacher_frozen=True,anchor_gamma=1e-4 if kind=='spatial' else 0.,
                line_search=kind!='coverage',seconds=time.perf_counter()-start,
                **paired_readout(v['logits'],records,ids,base))
    it.restore(it.initial)
    assert all(torch.equal(p,it.initial[n]) for n,p in it.named)
    return result
