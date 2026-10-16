"""Frozen final-actionness VJPs at the actual fused encoder/decoder interface."""
import torch


def maps_from_gradient(feature, gradient):
    """[C,H,W]; double reductions, no fake evidence for a zero CAM."""
    f, g = feature.detach().double(), gradient.detach().double()
    return dict(gradcam=(g.mean((1, 2))[:, None, None] * f).sum(0).relu(),
                grad_times_activation=(f * g).abs().sum(0))


def temporal_shift_indices(n):
    out = list(range(n))
    for offset in (0, 1):
        ids = list(range(offset, n, 2))
        for i, position in enumerate(ids):
            out[position] = ids[(i - len(ids) // 2) % len(ids)]
    return out


def select_donors(rows):
    """Only input caption, parsed subject, source and key. Never GT labels."""
    ordered = sorted(rows, key=lambda x: x['key'])
    out = {}
    for r in ordered:
        noun = r['subject'].strip().lower()
        other = [x for x in ordered if x['source'] != r['source']
                 and x['caption'] != r['caption']]
        matched = [x for x in other if noun and x['subject'].strip().lower() == noun]
        pool = [x for x in matched if x['cohort'] == r['cohort']] or matched
        eligible = bool(pool)
        pool = pool or [x for x in other if x['cohort'] == r['cohort']] or other
        after = [x for x in pool if x['key'] > r['key']]
        d = (after or pool)[0]
        out[r['key']] = dict(donor_key=d['key'], source=d['source'], caption=d['caption'],
                            subject=d['subject'], matched_query_noun=eligible, noun=noun)
    return out


def capture(replay):
    replay.delta.requires_grad_(False)
    replay.decoder.requires_grad_(False)
    assert not any(p.requires_grad or p.grad is not None for p in replay.model.parameters())
    maps, actions, norms, audits, boxes, logits = [], [], [], [], [], []
    for j, view in enumerate(replay.views):
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            H0 = replay.norm(view['prefix'])
        H = H0.detach().clone().requires_grad_(True)
        h, w = view['info']['fea_map_size']
        k, n = h * w, H.shape[1]
        info = dict(view['info'])
        info.update(encoded_feature=H, frames_cls=H.mean(0), videos_cls=H.mean(0).mean(0))
        fm = H[-k:].permute(1, 2, 0).reshape(n, 256, h, w).detach()
        fa = H[:k].permute(1, 2, 0).reshape(n, 256, h, w).detach()
        ft = H[k:-k].mean(1).unsqueeze(0).detach()
        chosen = replay.zero['gates'][j]['second']
        with torch.autocast('cuda', enabled=False):
            itq, isq = replay._queries(H, fm, fa, ft, chosen, k)
            # Preserve reference_values' exact zero-residual query construction.
            def native_zero_query(module, args, kwargs):
                kwargs = dict(kwargs)
                q = kwargs['query_tgt']
                kwargs['query_tgt'] = q + replay.delta.to(q.dtype)[None, None, :]
                return args, kwargs
            hook = replay.decoder.decoder.register_forward_pre_hook(native_zero_query, with_kwargs=True)
            try:
                pos, hidden = replay.decoder(encoded_info=info, vis_pos=view['vis_pos'], itq=itq, isq=isq)
            finally:
                hook.remove()
            raw = replay.model.action_embed(hidden)[-1].reshape(-1)
            native_logits = replay.head(hidden)[-1]
        b = pos.flatten(1, 2)[-1].detach()
        assert torch.equal(raw.detach(), replay.zero['actions'][j].reshape(-1)), 'actionness replay mismatch'
        assert torch.equal(native_logits.detach(), replay.zero['logits'][j]), 'temporal replay mismatch'
        assert torch.equal(b, replay.zero['boxes'][j::2]), ('box replay mismatch',float((b-replay.zero['boxes'][j::2]).abs().max()))
        probability = raw.double().sigmoid()
        local, nr, check = [], [], []
        for i in range(n):
            g = torch.autograd.grad(probability[i], H, retain_graph=True)[0]
            assert torch.isfinite(g).all()
            if i == 0:
                gl = torch.autograd.grad(raw[i], H, retain_graph=True)[0]
                scale = probability[i].detach() * (1 - probability[i].detach())
                expected = gl.double() * scale
                error = (g.double() - expected).norm() / expected.norm().clamp_min(1e-30)
                assert float(error) < 2e-5, ('sigmoid chain', float(error))
                check.append(dict(position=i, relative_error=float(error), sigmoid_derivative=float(scale)))
                del gl, expected
            f = H[-k:, i].transpose(0, 1).reshape(256, h, w)
            gm = g[-k:, i].transpose(0, 1).reshape(256, h, w)
            mm = maps_from_gradient(f, gm)
            local.append({name: value.cpu() for name, value in mm.items()})
            nn = dict(appearance=float(g[:k].double().norm()), motion=float(g[-k:].double().norm()),
                      text=float(g[k:-k].double().norm()), own_motion=float(g[-k:, i].double().norm()),
                      own_total=float(g[:, i].double().norm()), total=float(g.double().norm()),
                      sigmoid_saturated=bool(probability[i] == 0 or probability[i] == 1))
            nn['other_frame_total'] = max(0., nn['total'] ** 2 - nn['own_total'] ** 2) ** .5
            nr.append(nn)
            del g, f, gm, mm
        maps.append({name: torch.stack([x[name] for x in local]) for name in local[0]})
        actions.append(dict(raw=raw.detach().cpu(), probability=probability.detach().cpu()))
        norms.append(nr)
        boxes.append(b.cpu()); logits.append(native_logits.detach().cpu())
        audits.append(dict(offset=j, frames=n, h=h, w=w, tokens=H.shape[0], sigmoid_chain=check,
                           exact_native_actions=True, exact_native_logits=True, exact_native_boxes=True))
        del H, H0, info, fm, fa, ft, itq, isq, pos, hidden, raw, probability, native_logits
    def merge(x):
        return torch.stack([x[i % 2][i // 2] for i in range(replay.n)])
    assert not any(p.grad is not None for p in replay.model.parameters())
    return dict(maps={name: merge([v[name] for v in maps]) for name in maps[0]},
                raw_actionness=merge([a['raw'] for a in actions]),
                probabilities=merge([a['probability'] for a in actions]),
                gradient_norms=[norms[i % 2][i // 2] for i in range(replay.n)],
                boxes=merge(boxes), temporal_logits=logits, gates=replay.zero['gates'],
                offset_audits=audits, shift_indices=temporal_shift_indices(replay.n),
                GT_access=False, parameter_updates=0, experts=0,
                vjp_count=sum(a['frames'] + 1 for a in audits),
                target='sigmoid_once(final_pred_actioness_logits)',
                feature='post_multimodal_encoder_norm_motion_tokens',
                gradient='per_output_per_frame_diagonal_not_sum', frozen_native_routing=True)
