"""Read-only TA signal capture and explicit, untrained diagnostic readouts."""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_convert, roi_align


def continuity(features):
    f = np.asarray(features, dtype=np.float64)
    if f.ndim != 2 or len(f) < 2 or not np.isfinite(f).all():
        raise ValueError('Need finite T x D features, T >= 2')
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    pair = np.clip((f[:-1]*f[1:]).sum(1), -1., 1.)
    value = np.r_[pair[0], (pair[:-1]+pair[1:])/2, pair[-1]]
    return (value+1)/2


def incoming_attention(weights):
    a = np.asarray(weights, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Expected nonnegative square attention')
    incoming = a.mean(0)
    return incoming / max(float(incoming.max()), 1e-30)


def envelope(views, full_ids, *, uniform=False):
    full = np.asarray(full_ids)
    starts, ends = [], []
    for v in views:
        ids = np.asarray(v['frame_ids']); joint = np.asarray(v['joint'], dtype=float)
        if joint.shape != (len(ids),len(ids)) or not np.isclose(joint.sum(), 1.):
            raise ValueError('Joint/grid mismatch')
        if uniform:
            joint = np.triu(np.ones_like(joint), 1); joint /= joint.sum()
        ps, pe = joint.sum(1), joint.sum(0)
        starts.append([ps[ids > t].sum() for t in full])
        ends.append([pe[ids < t].sum() for t in full])
    return np.clip(1-np.prod(starts,axis=0)-np.prod(ends,axis=0), 0, 1)


def endpoint_score(views, full_ids, indices):
    start, end = np.asarray(full_ids)[list(indices)]
    values, invalid = [], 0
    for v in views:
        ids = np.asarray(v['frame_ids']); j = np.asarray(v['joint'])
        a, b = int(abs(ids-start).argmin()), int(abs(ids-end).argmin())
        invalid += int(a >= b)
        values.append(float(np.log(max(float(j[a,b]),1e-30))))
    return float(np.mean(values)), invalid


def interleave(arrays, n):
    arrays = [np.asarray(a) for a in arrays]
    out = np.empty((n,)+arrays[0].shape[1:], dtype=arrays[0].dtype)
    for offset, a in enumerate(arrays):
        if len(a) != len(range(offset,n,2)):
            raise ValueError('Offset mismatch')
        out[offset::2] = a
    return out


@torch.no_grad()
def capture(model, batch):
    """Exact official forward; hooks observe, never replace any tensor."""
    from scripts import run_tastvg_span_tta as ta
    from vg_tta.decota_tastvg_episode_v1 import native_view_indices
    post = ta.official_imports()['postprocessor']()
    records, features = [], []
    n = len(batch['targets'][0]['frame_ids'])
    for offset in (0,1):
        view = ta._make_temporal_view_batch(batch,offset=offset,flip=False)
        cache, calls = {}, {'attention':0}
        def attention_hook(module,args,out):
            calls['attention'] += 1
            cache['attention'] = out[1].detach().float().cpu()[0]
        def input_hook(module,args):
            cache['hidden'] = args[0].detach()[-1,0].float().cpu()
        def visual_hook(module,args,out):
            cache['visual'] = out.detach()
        handles = [model.ground_decoder.time_decoder.layers[-1].self_attn.register_forward_hook(attention_hook),
                   model.temp_embed.register_forward_pre_hook(input_hook),
                   model.input_proj.register_forward_hook(visual_hook)]
        try:
            with torch.autocast('cuda',dtype=torch.float16):
                out = model(view['videos'],view['texts'],view['targets'],iteration_rate=-1)
        finally:
            for h in handles: h.remove()
        assert calls['attention'] == 2, 'Expected original two decoder passes'
        r = ta._postprocess_temporal_view(out,view,post,offset=offset,flip=False)
        rec = {k:r[k].detach().cpu() if torch.is_tensor(r[k]) else r[k]
               for k in ('offset','flip','frame_ids','raw_boxes','boxes_abs','temporal_logits','pred_sted')}
        z = out['pred_sted'][0]
        ij = native_view_indices(z[None]); ids = list(rec['frame_ids'])
        assert list(rec['pred_sted']) == [ids[ij[0]],ids[ij[1]]+1]
        score = (z[:,0].log_softmax(0)[:,None]+z[:,1].log_softmax(0)[None]).double()
        legal = torch.triu(torch.ones_like(score,dtype=torch.bool),1)
        joint = torch.zeros_like(score); joint[legal] = score[legal].softmax(0)
        entropy = -(z.double().softmax(0)*z.double().log_softmax(0)).sum(0)/np.log(len(z))
        visual = cache.pop('visual'); boxes = rec['raw_boxes'].to(visual.device).float()
        xy = box_convert(boxes,'cxcywh','xyxy').clamp(0,1)
        xy *= xy.new_tensor([visual.shape[-1],visual.shape[-2],visual.shape[-1],visual.shape[-2]])
        rois = torch.cat([torch.arange(len(xy),device=xy.device)[:,None],xy],1)
        roi = roi_align(visual.float(),rois,(3,3),sampling_ratio=2,aligned=True).mean((-1,-2))
        feat = dict(frame_ids=ids,joint=joint.cpu().numpy(),entropy=entropy.cpu().numpy(),
            hidden=cache['hidden'].numpy(),roi=F.normalize(roi,dim=-1).cpu().numpy(),
            attention=cache['attention'].numpy(),attention_calls=calls['attention'],
            actionness_logits=out['pred_actioness'].detach().float().cpu().reshape(-1).numpy(),
            tts_appearance=out['logits_f_a'].detach().float().cpu().sigmoid().reshape(-1).numpy(),
            tts_motion=out['logits_f_m'].detach().float().cpu().sigmoid().reshape(-1).numpy())
        assert all(len(feat[k]) == len(ids) for k in ('hidden','roi','actionness_logits','tts_appearance','tts_motion'))
        records.append(rec); features.append(feat)
    base = ta._merge_postprocessed_views(records,batch['targets'][0]['frame_ids'])
    action_logits = interleave([f['actionness_logits'] for f in features],n)
    signals = dict(native_posterior=envelope(features,batch['targets'][0]['frame_ids']),
        uniform_native_prior=envelope(features,batch['targets'][0]['frame_ids'],uniform=True),
        actionness=1/(1+np.exp(-np.clip(action_logits,-80,80))),
        attention=incoming_attention_merge(features,n),
        identity_continuity=continuity(interleave([f['roi'] for f in features],n)),
        state_continuity=continuity(interleave([f['hidden'] for f in features],n)),
        tts_appearance=interleave([f['tts_appearance'] for f in features],n),
        tts_motion=interleave([f['tts_motion'] for f in features],n),constant_half=np.full(n,.5))
    return dict(base=base,views=features,signals=signals,actionness_logits=action_logits,
                uncertainty=float(np.mean([f['entropy'] for f in features])),GT_used=False)


def incoming_attention_merge(views,n):
    return interleave([incoming_attention(v['attention']) for v in views],n)
