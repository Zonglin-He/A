"""TA-EV cached two-stage native suffix. No labels, optimizers or pixel changes.

Evidence gradients intentionally differentiate the actual frozen TTS/ASA
functions w.r.t. H. Official training detaches their inputs; this diagnostic
removes that gradient stop only, not any forward operation. Hard selections
and iterative decoder reference detaches retain official semantics.
"""
import torch


def indices(mask):
    return torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1).tolist()


def visual_mask(H, count):
    mask = torch.ones_like(H)
    mask[count:-count] = 0
    return mask


def suffix(model, view, H, *, evidence_grad=True):
    info = dict(view['info'])
    info.update(encoded_feature=H, frames_cls=H.mean(0), videos_cls=H.mean(0).mean(0))
    height, width = info['fea_map_size']; count = height * width
    assert H.ndim == 3 and H.shape[-1] == 256 and H.shape[0] > 2 * count
    a = H[:count].permute(1, 2, 0).reshape(-1, 256, height, width)
    m = H[-count:].permute(1, 2, 0).reshape_as(a)
    text = H[count:-count].mean(1).unsqueeze(0)
    if not evidence_grad:
        a, m, text = a.detach(), m.detach(), text.detach()
    tm = model.t_temporal_clas(m, text)
    ta = model.s_temporal_clas(a, text)
    prob = (tm.sigmoid() + ta.sigmoid()) / 2
    fallback = indices(prob > 0)
    first = indices(prob > model.theta) or fallback
    assert first, 'Native TTS produced empty fallback'

    def queries(chosen, stage):
        # The original first and second pass call orders are preserved.
        if stage == 1:
            _, am = model.t_spatial_clas(m[chosen], text[:, :1])
            _, aa = model.s_spatial_clas(a[chosen], text[:, :1])
        else:
            _, aa = model.s_spatial_clas(a[chosen], text[:, :1])
            _, am = model.t_spatial_clas(m[chosen], text[:, :1])
        qt = (H[-count:].permute(1, 0, 2)[chosen] * am.unsqueeze(2)).mean((0, 1))
        qs = (H[:count].permute(1, 0, 2)[chosen] * aa.unsqueeze(2)).mean((0, 1))
        return qt, qs, aa, am

    qt1, qs1, aa1, am1 = queries(first, 1)
    _, hidden1 = model.ground_decoder(encoded_info=info, vis_pos=view['vis_pos'], itq=qt1, isq=qs1)
    action = model.action_embed(hidden1)[-1]
    second = indices(action.squeeze().sigmoid() > .5) or fallback
    qt2, qs2, aa2, am2 = queries(second, 2)
    # Existing replay materializes the expanded spatial query with + zero on
    # the SECOND pass. Preserve its strides/kernel arithmetic, not just algebra.
    def native_zero_query(module, args, kwargs):
        kwargs=dict(kwargs)
        q=kwargs['query_tgt']
        kwargs['query_tgt']=q+torch.zeros(256,device=q.device,dtype=q.dtype)[None,None,:]
        return args,kwargs
    hook=model.ground_decoder.decoder.register_forward_pre_hook(native_zero_query,with_kwargs=True)
    try:
        boxes, hidden2 = model.ground_decoder(encoded_info=info, vis_pos=view['vis_pos'], itq=qt2, isq=qs2)
    finally:
        hook.remove()
    logits = model.temp_embed(hidden2)[-1]
    out = dict(TTS_app=ta, TTS_motion=tm, selected_stage1=first,
               ASA1_app=aa1, ASA1_motion=am1, Qs1=qs1, Qt1=qt1,
               actionness=action, selected_stage2=second,
               ASA2_app=aa2, ASA2_motion=am2, Qs2=qs2, Qt2=qt2,
               pred_sted=logits, pred_start=logits[..., 0].softmax(1),
               pred_end=logits[..., 1].softmax(1), pred_boxes=boxes.flatten(1, 2)[-1])
    assert all(torch.isfinite(x).all() for x in out.values() if torch.is_tensor(x))
    return out


def combined(model, views, fields, records, ids, *, evidence_grad=True):
    from methods.decota_final_simplified_v1.objectives import prediction
    results = [suffix(model, v, h, evidence_grad=evidence_grad) for v, h in zip(views, fields)]
    boxes = torch.stack([results[i % 2]['pred_boxes'][i // 2] for i in range(len(ids))])
    pred = prediction([r['pred_sted'] for r in results], boxes, records, ids)
    return results, boxes, pred


@torch.no_grad()
def capture(model, frames, x):
    from methods.decota_final_simplified_v1.backbone import make_batch, query_subject, capture as old_capture
    from methods.decota_final_simplified_v1.backbone import inserted_state, offset_batch
    from methods.decota_final_simplified_v1.tensors import floating32, detached
    q = {**x['input'], 'height': frames.shape[1], 'width': frames.shape[2]}
    batch = make_batch(frames, x['frame_ids'], q, model)
    with query_subject(model, batch, x['subject']):
        views, records = old_capture(model, batch)
        views = floating32(views)
        fields = [model.ground_encoder.encoder.norm(v['prefix']) for v in views]
        results, boxes, pred = combined(model, views, fields, records, x['frame_ids'])
        # Both official full forwards under the established FP32 suffix contract.
        with inserted_state(model, {}):
            for j in (0, 1):
                b = offset_batch(batch, j)
                with torch.autocast('cuda', dtype=torch.float16):
                    full = model(b['videos'], b['texts'], b['targets'], iteration_rate=-1)
                for name in ('pred_boxes', 'pred_sted'):
                    assert torch.equal(full[name], results[j][name]), ('full_suffix_identity', j, name, float((full[name]-results[j][name]).abs().max()))
                assert torch.equal(full['logits_f_a'], results[j]['TTS_app'])
                assert torch.equal(full['logits_f_m'], results[j]['TTS_motion'])
    assert torch.equal(boxes.cpu(), x['native_boxes']), 'Historical native boxes mismatch'
    assert all(torch.equal(r['pred_sted'].cpu(), z) for r, z in zip(results, x['native_logits']))
    compact = []
    for v, H in zip(views, fields):
        count = v['info']['fea_map_size'][0] * v['info']['fea_map_size'][1]
        compact.append(dict(info={k:val for k,val in v['info'].items() if k not in ('encoded_feature','frames_cls','videos_cls')},
                            vis_pos=v['vis_pos'], H=H, H_app=H[:count], H_text=H[count:-count], H_motion=H[-count:]))
    return detached(dict(views=compact, evidence=results, prediction=pred, records=records,
                         exact_full_replay=True, exact_historical_native=True,
                         precision='existing FP16 prefix / FP32 complete suffix'), 'cpu')
