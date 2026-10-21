"""Frozen-cache, realizable content-state OPD and matched Direct controls.

No file/GT/RGB/teacher reads. The runner supplies sealed cache and native masks.
Only the per-query temporary content state is optimized; all network weights remain frozen.
"""
from __future__ import annotations
import copy, gc, hashlib, inspect, math, sys, textwrap, time, types
import torch
import torch.nn.functional as F
from stvg.capture import transfer
from stvg.state import inject

LOCKED = dict(seed=41, sigma=.1, samples=32, tau=.01, lr=.04,
              opd_rounds=40, direct_steps=660, forward_budget=1321)
BACKTRACK = (1., .5, .25, .125, .0625)
ARMS = ('StateOPD', 'Direct', 'DirectMC', 'DirectFullRef')


def _cpu(x):
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x, dict): return {k: _cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_cpu(v) for v in x]
    return x


def settings(cfg):
    q = dict(LOCKED)
    q.update(sigma=float(cfg['sigma']),tau=float(cfg['tau']),basis_dimensions=int(cfg.get('basis_dimensions',256)),radius=cfg.get('radius'))
    assert q['sigma']>0 and q['tau']>0 and q['basis_dimensions'] in (4,256)
    for k, v in LOCKED.items():
        if k in cfg and k not in ('sigma','tau'):
            if cfg[k] != v: raise ValueError(f'Locked setting {k}: {cfg[k]} != {v}')
    constants = dict(guard_weight=.1, guard_deadzone=.25, guard_mean_weight=.5,
                     guard_max_weight=.5, mask_mass_epsilon=1e-6, accept_epsilon=1e-8,
                     adam_eps=1e-8)
    for k, v in constants.items():
        if k in cfg and cfg[k] != v: raise ValueError(f'Locked constant {k}')
    if 'backtracking' in cfg: assert tuple(cfg['backtracking']) == BACKTRACK
    if 'betas' in cfg: assert tuple(cfg['betas']) == (.9,.999)
    return q


def content_basis(cache, dimensions=4):
    """Top4 of the centered FP64 CPU covariance of all unmasked encoded-feature tokens."""
    if dimensions==256:
        P=torch.eye(256,dtype=torch.float64)
        return P,dict(rank=256,uses_only_unmasked_tokens=False,identity_full_content=True,basis_sha256=hashlib.sha256(P.numpy().tobytes()).hexdigest())
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        xs = []; total_tokens = 0
        for item in cache['spatial_inputs']:
            features = item['encoded_feature'].detach().cpu().double()
            mask = item['encoded_mask'].detach().cpu().bool()
            assert features.ndim == 3 and features.shape[-1] == 256
            assert mask.shape == (features.shape[1], features.shape[0])
            total_tokens += features.shape[0] * features.shape[1]
            xs.append(features[~mask.T])
        X = torch.cat(xs, 0)
        mean = X.mean(0); X = X - mean
        cov = X.T @ X / max(len(X) - 1, 1)
        values, vectors = torch.linalg.eigh(cov)
        P = vectors[:, -4:].flip(1).contiguous()
        idx = P.abs().argmax(0)
        sign = torch.where(P[idx, torch.arange(4)] < 0, -1., 1.)
        P = (P * sign).contiguous()
        assert torch.allclose(P.T @ P, torch.eye(4, dtype=torch.float64), atol=1e-10, rtol=0)
        return P, dict(tokens=len(X), total_tokens=total_tokens, masked_tokens=total_tokens-len(X),
                       uses_only_unmasked_tokens=True, dtype='float64 CPU', rank=4,
                       eigenvalues=values[-4:].flip(0).tolist(),
                       sign_rule='largest absolute entry in each column is positive',
                       basis_sha256=hashlib.sha256(P.numpy().tobytes()).hexdigest())
    finally:
        torch.set_num_threads(before)


def content_state(z, P, radius=None):
    """Smooth bound ||eta||_2 < 4; derivative at z=0 is the orthonormal map."""
    if radius is None:return z @ P.T
    return (z @ P.T) * (radius / torch.sqrt(z.new_tensor(radius**2) + z.square().sum()))


def candidate_noise(key, round_index, q, dtype=torch.float32):
    h = f'state-opd-v1|{q["seed"]}|{key}|{round_index}'
    seed = int(hashlib.sha256(h.encode()).hexdigest()[:15], 16)
    g = torch.Generator(device='cpu').manual_seed(seed)
    eps = torch.randn(16, 8, q.get('basis_dimensions',4), generator=g, dtype=dtype)
    return torch.cat((eps, -eps), 0), seed


class ProjectionEvidence:
    """Same native selected track, official mask resize, soft x/y projection Dice."""
    def __init__(self, pack, cache, device):
        raw = pack.get('raw_out', pack.get('out'))
        teacher = pack.get('teacher', pack.get('z', pack.get('native_result')))
        if raw is None or teacher is None: raise ValueError('Native mask pack is incomplete')
        idx = int(teacher.get('selected_query', pack.get('selected_query')))
        assert idx == int(cache['teacher']['selected_query'])
        assert torch.equal(torch.as_tensor(teacher['boxes']), torch.as_tensor(cache['teacher']['boxes']))
        shape = teacher.get('input_shape', pack.get('input_shape'))
        assert shape and len(shape) == 4 and shape[0] == 8
        assert pack['nested_mask_all_false'] is True
        assert pack['nested_mask_shape'] == [1, 8, *shape[-2:]]
        assert pack['nested_tensor_shape'] == [1, *shape]
        assert list(pack['positions8']) == list(cache['planned_positions'])
        assert list(pack['frame_ids']) == list(cache['frame_ids'])
        masks = raw['pred_masks']
        if masks.ndim == 5: masks = masks[0]
        assert masks.ndim == 4 and masks.shape[0] == 8
        with torch.no_grad():
            logits = masks[:, idx, None].detach().float().to(device)
            p = F.interpolate(logits, size=tuple(shape[-2:]), mode='bilinear',
                              align_corners=False).squeeze(1).sigmoid().double()
            self.h, self.w = p.shape[-2:]
            self.mass = p.sum((1, 2))
            self.active = self.mass > 1e-6
            px, py = p.amax(1), p.amax(2)
            self.mass_x, self.mass_y = px.sum(-1), py.sum(-1)
            self.X = F.pad(px.cumsum(-1), (1, 0))[:, None, None]
            self.Y = F.pad(py.cumsum(-1), (1, 0))[:, None, None]
        self.selected_query = idx
        self.positions = list(cache['planned_positions'])
        self.valid_count = int(self.active.sum())

    @staticmethod
    def _integral(field, interval):
        # field has [8,1,1,L+1]; normalized endpoints integrate fractional pixels.
        n = interval.shape[1]
        grid = torch.stack((2 * interval - 1, torch.zeros_like(interval)), -1)
        return F.grid_sample(field, grid.reshape(8, n * 2, 1, 2),
                             mode='bilinear', padding_mode='border',
                             align_corners=True).reshape(8, n, 2)

    def anchor_scores(self, b):
        assert b.shape[0] == 8 and b.shape[-1] == 4
        corners = torch.cat((b[..., :2] - b[..., 2:] / 2,
                             b[..., :2] + b[..., 2:] / 2), -1).clamp(0, 1)
        x0, y0, x1, y1 = corners.unbind(-1)
        ix = self._integral(self.X, torch.stack((x0, x1), -1))
        iy = self._integral(self.Y, torch.stack((y0, y1), -1))
        sx = 2 * (ix[..., 1] - ix[..., 0]) / (((x1 - x0) * self.w) + self.mass_x[:, None]).clamp_min(1e-15)
        sy = 2 * (iy[..., 1] - iy[..., 0]) / (((y1 - y0) * self.h) + self.mass_y[:, None]).clamp_min(1e-15)
        return .5 * (sx + sy)

    def score(self, boxes):
        assert boxes.ndim == 2
        a = self.anchor_scores(boxes[self.positions, None]).squeeze(1)
        return a[self.active].mean(), a


def protected_utility(boxes, E, evidence):
    """Common evidence and whole-trajectory protection for every arm."""
    mask, anchors = evidence.score(boxes)
    dz = torch.cat(((boxes[:, :2] - E[:, :2]) / E[:, 2:],
                    torch.log(boxes[:, 2:]) - torch.log(E[:, 2:])), -1)
    # vector_norm has the well-defined zero subgradient; sqrt(mean(square)) does not.
    d = torch.linalg.vector_norm(dz, dim=-1) / 2.
    excess2 = torch.relu(d - .25).square()
    guard_mean, guard_max = excess2.mean(), excess2.max()
    guard = .5 * guard_mean + .5 * guard_max
    U = mask - .1 * guard
    return dict(U=U, mask=mask, guard=guard, guard_mean=guard_mean,
                guard_max=guard_max, d_mean=d.mean(), d_max=d.max(),
                anchor_scores=anchors)


class FrozenContentReplay:
    """Identical original offset/inject/corrected forward; no fake K-batch."""
    def __init__(self, model, cache, full_reference=False):
        self.decoder = copy.deepcopy(model.ground_decoder.decoder).eval().requires_grad_(False)
        self.device = next(self.decoder.parameters()).device
        assert self.device.type == 'cuda'
        self.inputs = transfer(cache['spatial_inputs'], self.device)
        self.Source = torch.as_tensor(cache['Source'], device=self.device)
        self.E = torch.as_tensor(cache['E'], device=self.device, dtype=torch.float64)
        self.positions = list(cache['planned_positions'])
        assert len(self.positions) == 8
        self.setup_evals = self.evals = 0
        self.initial_checks = dict(full_reference=bool(full_reference))
        zero = torch.zeros(8, 256, device=self.device)
        with torch.no_grad():
            old, native = self.boxes(zero, setup=True, return_native=True)
        assert torch.equal(native.cpu(), torch.as_tensor(cache['Source']))
        assert torch.equal(old, self.E)
        self.initial_checks.update(original_native_Source_bit_exact=True, original_E_bit_exact=True)
        if full_reference:
            original = self.decoder.forward
            body = textwrap.dedent(inspect.getsource(original))
            token = 'pred_boxes = new_pred_boxes.detach()'
            assert body.count(token) == 1
            changed = body.replace(token, 'pred_boxes = new_pred_boxes')
            namespace = dict(vars(sys.modules[original.__module__]))
            exec(compile(changed, '<state-opd-full-reference-instance>', 'exec'), namespace)
            self.decoder.forward = types.MethodType(namespace['forward'], self.decoder)
            with torch.no_grad(): new = self.boxes(zero, setup=True)
            assert torch.equal(old, new)
            self.initial_checks.update(full_reference_forward_bit_exact=True,
                                      original_forward_sha256=hashlib.sha256(body.encode()).hexdigest(),
                                      full_reference_forward_sha256=hashlib.sha256(changed.encode()).hexdigest())

    def boxes(self, eta, setup=False, return_native=False):
        assert eta.shape == (8, 256)
        raw = []
        with torch.autocast('cuda', enabled=False):
            for view, original in enumerate(self.inputs):
                kw = dict(original)
                kw['query_tgt'] = inject(original['query_tgt'], eta, self.positions, view)
                raw.append(self.decoder(**kw).flatten(1, 2)[-1])
            native = torch.stack([raw[p % 2][p // 2] for p in range(len(self.Source))]).float()
            # Same operation order as stvg.geometry.corrected; validation is centralized.
            b = native.double(); src = self.Source.to(b); e = self.E
            out = torch.cat((e[:, :2] + (b[:, :2] - src[:, :2]),
                             e[:, 2:] * torch.exp(torch.log(b[:, 2:]) - torch.log(src[:, 2:]))), -1)
        if setup: self.setup_evals += 1
        else: self.evals += 1
        return (out, native) if return_native else out


def _brief(u):
    out = {k: float(u[k].detach()) for k in ['U', 'mask', 'guard', 'guard_mean', 'guard_max', 'd_mean', 'd_max']}
    if not all(math.isfinite(v) for v in out.values()): raise FloatingPointError('Nonfinite protected utility')
    return out


def fit(model, cache, maskpack, key, arm, cfg):
    """Execute exactly one locked arm; runner owns locking, sealing and GT isolation."""
    if arm not in ARMS: raise ValueError(arm)
    q = settings(cfg); device = next(model.ground_decoder.decoder.parameters()).device
    if device.type != 'cuda': raise ValueError('Runner must put only the frozen decoder on CUDA')
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device); begin = time.perf_counter()
    evidence = ProjectionEvidence(maskpack, cache, device)
    if evidence.valid_count == 0:
        # Still perform the common initial trajectory check; no adaptation or padding.
        replay = FrozenContentReplay(model, cache, full_reference=arm == 'DirectFullRef')
        with torch.no_grad():
            final = replay.boxes(torch.zeros(8,256,device=device)).detach()
        assert torch.equal(final, replay.E)
        torch.cuda.synchronize()
        result = dict(arm=arm, final=_cpu(final), path=[], rounds=[],
                    stop_reason='no_valid_mask_no_update',
                    initial_checks=dict(**replay.initial_checks,valid_mask_anchors=0),
                    costs=dict(seconds=time.perf_counter()-begin, full_trajectory_evals=1,
                               native_offset_decoder_calls=2, setup_full_trajectory_evals=replay.setup_evals,
                               setup_trajectory_evals=replay.setup_evals,
                               setup_native_offset_decoder_calls=2*replay.setup_evals,
                               decoder_backwards=0,state_only_backwards=0,reward_evals=0,
                               updates=0,accepted_updates=0,accepted=0,rejected=0,
                               stop_reason='no_valid_mask_no_update',allowance=1321,
                               peak_GPU_bytes=torch.cuda.max_memory_allocated(device)))
        del replay,evidence;gc.collect();torch.cuda.empty_cache();return result
    Pcpu, basis_info = content_basis(cache,q['basis_dimensions']); P = Pcpu.to(device=device, dtype=torch.float32)
    replay = FrozenContentReplay(model, cache, full_reference=arm == 'DirectFullRef')
    def state(z,P):return content_state(z,P,q['radius'])
    z = torch.zeros(8, q['basis_dimensions'], device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=q['lr'], betas=(.9, .999), eps=1e-8, weight_decay=0.)
    with torch.no_grad():
        current_boxes = replay.boxes(state(z, P)).detach()
        assert torch.equal(current_boxes, replay.E)
        current_u = protected_utility(current_boxes, replay.E, evidence)
    current = _brief(current_u)
    path = [dict(step=0, z=_cpu(z), eta=_cpu(state(z, P)), boxes=_cpu(current_boxes), **current)]
    rounds=[]; decoder_backwards=state_backwards=accepted=rejected=0
    limit = q['opd_rounds'] if arm in ('StateOPD','DirectMC') else q['direct_steps']
    stop_reason='maximum_rounds'
    try:
        for r in range(limit):
            needed = q['samples']+1 if arm in ('StateOPD','DirectMC') else 2
            if replay.evals + needed > q['forward_budget']:
                stop_reason='forward_budget_before_complete_round';break
            t = time.perf_counter(); before_z=z.detach().clone(); before_opt=copy.deepcopy(opt.state_dict())
            rr=dict(round=r,before=dict(current),accepted=False,valid_mask_anchors=evidence.valid_count)
            opt.zero_grad(set_to_none=True)
            if arm in ('StateOPD','DirectMC'):
                noise, seed = candidate_noise(key,r,q); noise=noise.to(z)
                actions=(before_z[None]+q['sigma']*noise).detach()
                utilities=[]; gradients=[]
                for action in actions:
                    if arm=='StateOPD':
                        with torch.no_grad():
                            b=replay.boxes(state(action,P));u=protected_utility(b,replay.E,evidence)
                    else:
                        candidate=z+(action-before_z).detach()
                        b=replay.boxes(state(candidate,P));u=protected_utility(b,replay.E,evidence)
                        gu=torch.autograd.grad(u['U'],z)[0];decoder_backwards+=1
                        gradients.append(gu.detach())
                    utilities.append({k:u[k].detach()for k in ['U','mask','guard','d_max']})
                    del b,u
                U=torch.stack([x['U']for x in utilities]);weights=(U/q['tau']).softmax(0).detach()
                if not bool(torch.isfinite(U).all()): raise FloatingPointError('Nonfinite candidate reward')
                if arm=='StateOPD':
                    nll=((z[None]-actions).square().sum((1,2))/(2*q['sigma']**2))
                    loss=(weights*nll).sum()  # q already contains exp(U/tau); no extra /tau
                    grad=torch.autograd.grad(loss,z)[0];state_backwards+=1
                else:
                    # Exact pathwise gradient of -logmeanexp(U/tau), retaining one decoder graph at a time.
                    grad=-(weights[:,None,None]*torch.stack(gradients)).sum(0)/q['tau']
                    loss=-(torch.logsumexp(U/q['tau'],0)-math.log(q['samples']))
                rr.update(rng_seed=seed,candidate_z=_cpu(actions),candidate_U=_cpu(U),
                          candidate_mask=_cpu(torch.stack([x['mask']for x in utilities])),
                          candidate_guard=_cpu(torch.stack([x['guard']for x in utilities])),
                          weights=_cpu(weights),ESS=float(weights.square().sum().reciprocal()))
                del utilities,actions,noise,U,weights
            else:
                b=replay.boxes(state(z,P));u=protected_utility(b,replay.E,evidence)
                loss=-u['U']/q['tau'];grad=torch.autograd.grad(loss,z)[0];decoder_backwards+=1
                rr['gradient_evaluation']=_brief(u);del b,u
            if not bool(torch.isfinite(grad).all()): raise FloatingPointError('Nonfinite z gradient')
            # Streaming MC weights are FP64; the differentiable state is FP32.
            # Match autograd's parameter-gradient dtype at the optimizer boundary.
            z.grad=grad.detach().to(z);opt.step();delta=z.detach()-before_z
            rr.update(loss=float(loss.detach()),gradient_norm=float(grad.norm()),Adam_delta_norm=float(delta.norm()),backtracking=[])
            found=False;exhausted=False
            for scale in BACKTRACK:
                if replay.evals>=q['forward_budget']:
                    exhausted=True;break
                with torch.no_grad():
                    z.copy_(before_z+scale*delta)
                    proposal_boxes=replay.boxes(state(z,P)).detach()
                    proposal_u=protected_utility(proposal_boxes,replay.E,evidence)
                proposal=_brief(proposal_u)
                rr['backtracking'].append(dict(scale=scale,**proposal))
                if proposal['U']>current['U']+1e-8:
                    current_boxes=proposal_boxes;current=proposal;found=True
                    rr.update(accepted=True,accepted_scale=scale);accepted+=1;break
            if not found:
                with torch.no_grad():z.copy_(before_z)
                opt.load_state_dict(before_opt);rejected+=1
            rr.update(after=dict(current),full_trajectory_evals=replay.evals,
                      seconds=time.perf_counter()-t)
            rounds.append(rr)
            path.append(dict(step=r+1,z=_cpu(z),eta=_cpu(state(z,P)),boxes=_cpu(current_boxes),**current))
            del loss,grad,delta,before_z,before_opt
            if exhausted:
                stop_reason='forward_budget_during_backtracking';break
            if not found and arm in ('Direct','DirectFullRef'):
                stop_reason='converged_or_rejected';break
        assert replay.evals<=q['forward_budget']
        assert all(not p.requires_grad and p.grad is None for p in replay.decoder.parameters())
        torch.cuda.synchronize()
        return dict(arm=arm,final=_cpu(current_boxes),final_z=_cpu(z),final_eta=_cpu(state(z,P)),
                    basis=Pcpu,basis_info=basis_info,path=path,rounds=rounds,stop_reason=stop_reason,
                    initial_checks=dict(**replay.initial_checks,valid_mask_anchors=evidence.valid_count,
                                        active_mask_anchors=_cpu(evidence.active),selected_native_query=evidence.selected_query,
                                        config=q,line_search_factors=list(BACKTRACK),basis_info=basis_info,
                                        initial_z_zero=True,network_weights_frozen=True,
                                        state_space=dict(dimensions=[8,q['basis_dimensions']],radius=q['radius'],map='identity' if q['radius'] is None else 'smooth_radial'),
                                        projection_mean='mean over fixed active mask anchors',
                                        constants=dict(content_radius=q['radius'],basis_dimensions=q['basis_dimensions'],guard_weight=.1,
                                                       guard_deadzone=.25,guard_mean_weight=.5,guard_max_weight=.5,
                                                       mask_mass_epsilon=1e-6,accept_epsilon=1e-8,
                                                       betas=[.9,.999],adam_eps=1e-8)),
                    costs=dict(seconds=time.perf_counter()-begin,full_trajectory_evals=replay.evals,
                               native_offset_decoder_calls=2*replay.evals,
                               setup_full_trajectory_evals=replay.setup_evals,
                               setup_trajectory_evals=replay.setup_evals,
                               setup_native_offset_decoder_calls=2*replay.setup_evals,
                               decoder_backwards=decoder_backwards,state_only_backwards=state_backwards,
                               accepted=accepted,rejected=rejected,updates=len(rounds),accepted_updates=accepted,
                               reward_evals=replay.evals,stop_reason=stop_reason,
                               peak_GPU_bytes=torch.cuda.max_memory_allocated(device),
                               allowance=q['forward_budget']))
    finally:
        opt.zero_grad(set_to_none=True)
        del replay,evidence,opt
        gc.collect();torch.cuda.empty_cache()
