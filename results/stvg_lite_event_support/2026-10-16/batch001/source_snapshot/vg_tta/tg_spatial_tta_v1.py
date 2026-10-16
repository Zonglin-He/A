"""Sparse external-box TTA of the final native TA-STVG spatial query.

This module has no dataset-label reader. Final temporal predictions are owned
by the caller and are never decoded or modified here. The expert is frozen.
"""
from __future__ import annotations

import time
import numpy as np
import torch
from torchvision.ops import box_convert, nms
from vg_tta.metrics import generalized_box_iou_aligned_cxcywh


CONFIG = dict(keyframes=8, phrase_threshold=.35, distinct_margin=.05,
              nms_iou=.5, temporal_floor=.1, lr=.05, steps=3,
              anchor=1e-4, l1=5., giou=2., optimizer_eps=1e-8)


def keyframes(ids, extent, native, evidence, k=8):
    """Real-time strata, with priority to newly covered positions in mixed bins."""
    ids = np.asarray(ids); r = np.asarray(evidence)
    a, b = map(int, extent); na, nb = map(int, native)
    assert 0 <= a <= b < len(ids) and len(r) == len(ids)
    k = min(k, b-a+1)
    edges = np.linspace(ids[a], ids[b]+1, k+1)
    selected = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pool = [i for i in range(a,b+1) if lo <= ids[i] < hi]
        if not pool:
            continue
        added = [i for i in pool if not na <= i <= nb]
        selected.append(max(added or pool, key=lambda i:(r[i], -i)))
    # Nonuniform grids may leave empty bins. Fill the remaining budget by
    # farthest real-time distance, with relevance and index as tie breakers.
    while len(selected) < k:
        rest = [i for i in range(a,b+1) if i not in selected]
        selected.append(max(rest, key=lambda i:(min(abs(ids[i]-ids[j]) for j in selected),r[i],-i)))
    return sorted(selected)


VISUAL_NOUNS = set('shirt clothes clothing dress skirt pants trouser trousers jeans hat cap vest suit jacket coat robe sleeve sleeves backpack glasses hair beard moustache mustache shoe shoes collar shorts sweater uniform watch left right middle'.split())
COLORS = set('red blue green white black yellow purple brown gray grey orange pink beige striped plaid blond blonde'.split())
STOP_WORDS = set('a an the in with on of and his her their'.split())


def parse_visual_words(words, leading_question=False):
    """Conservative dependency extraction; no first-noun fallback.

    Input is a single Stanza sentence. Restrict clothing modifiers to direct
    attachments of the referent, excluding other actors and action objects.
    """
    by_id = {w.id:w for w in words}
    root = next((w for w in words if w.head == 0), None)
    nominal = None; rule = None
    if leading_question:
        for w in words:
            if w.lemma.lower() in {'what','which'} and w.deprel == 'det':
                nominal = by_id.get(w.head); rule = 'explicit_wh_noun'; break
        if nominal is None and any(w.lemma.lower() in {'who','whom'} for w in words):
            return dict(phrase='person', entity='person', rule='generic_who', distinctive=False)
        if nominal is None and any(w.lemma.lower() in {'what','which'} for w in words):
            return dict(phrase='', entity='', rule='unknown_queried_entity', distinctive=False)
    if nominal is None:
        candidates = [w for w in words if w.deprel.startswith('nsubj') and w.upos in {'NOUN','PROPN'} and root and w.head == root.id]
        if len(candidates) != 1:
            return dict(phrase='', entity='', rule='ambiguous_or_no_root_nominal_subject', distinctive=False)
        nominal = candidates[0]; rule = 'root_nominal_subject'
    if nominal.upos not in {'NOUN','PROPN'}:
        return dict(phrase='', entity='', rule='non_nominal_referent', distinctive=False)
    keep = {nominal.id}; attr = False
    for w in words:
        if w.head == nominal.id and w.deprel in {'amod','compound'}:
            keep.add(w.id); attr = True
        if w.head == nominal.id and w.deprel.startswith('nmod') and w.lemma.lower() in VISUAL_NOUNS | COLORS:
            keep.add(w.id); attr = True
            keep.update(v.id for v in words if v.head == w.id and v.deprel in {'case','amod','compound','conj','cc'})
        # "a child wearing blue pants" is a visual modifier, not the main event.
        if w.head == nominal.id and w.deprel.startswith('acl') and w.lemma.lower() == 'wear':
            for obj in words:
                if obj.head == w.id and obj.deprel == 'obj' and obj.lemma.lower() in VISUAL_NOUNS:
                    keep.update([w.id,obj.id]); attr = True
                    keep.update(v.id for v in words if v.head == obj.id and v.deprel in {'amod','compound','conj'})
    # Preserve nested visual compounds such as "white-collar", not just collar.
    changed = True
    while changed:
        before = len(keep)
        keep.update(w.id for w in words if w.head in keep and w.deprel in {'amod','compound','cc'})
        changed = len(keep) != before
    selected = sorted([w for w in words if w.id in keep], key=lambda w:w.id)
    phrase = ' '.join(w.lemma.lower() if w.id == nominal.id else w.text.lower() for w in selected)
    return dict(phrase=phrase, entity=nominal.lemma.lower(), rule=rule, distinctive=attr)


def visual_query(parser, caption):
    sentences = parser.nlp(caption).sentences
    if len(sentences) != 1:
        return dict(phrase='',entity='',rule='multiple_sentences',distinctive=False)
    words = sentences[0].words
    question = bool(words and words[0].lemma.lower() in {'what','which','who','whom'})
    return parse_visual_words(words, question)


def choose_candidate(boxes, scores, cfg=CONFIG):
    """Phrase score + duplicate NMS + distinct-instance margin; no student IoU."""
    boxes = boxes.float().cpu(); scores = scores.float().cpu()
    assert boxes.shape == (len(scores),4) and torch.isfinite(boxes).all() and torch.isfinite(scores).all()
    xy = box_convert(boxes, 'cxcywh','xyxy').clamp(0,1)
    keep = nms(xy, scores, cfg['nms_iou'])
    top = keep[:10]; accepted = False; reason = 'no_candidates'; margin = 0.
    if len(keep):
        best = int(keep[0]); runner = float(scores[keep[1]]) if len(keep)>1 else 0.
        margin = float(scores[best])-runner
        if float(scores[best]) < cfg['phrase_threshold']: reason = 'low_phrase_score'
        elif margin < cfg['distinct_margin']: reason = 'ambiguous_distinct_instances'
        elif bool((xy[best,2:] <= xy[best,:2]).any()): reason = 'invalid_box'
        else: accepted = True; reason = 'accepted'
    return dict(accepted=accepted, reason=reason, margin=margin,
                box=box_convert(xy[int(keep[0])], 'xyxy','cxcywh').tolist() if len(keep) else None,
                score=float(scores[keep[0]]) if len(keep) else 0.,
                top_boxes=boxes[top].tolist(), top_scores=scores[top].tolist())


class SpatialExpert:
    def __init__(self, snapshot):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(snapshot, local_files_only=True,
                            disable_custom_kernels=True).cuda().eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, rgb, phrase, entity, cfg=CONFIG):
        text = phrase.lower().strip() + '.'
        inputs = self.processor(images=rgb, text=text, return_tensors='pt').to('cuda')
        tokens = self.processor.tokenizer(text, return_offsets_mapping=True)
        start = text.index(entity); stop = start+len(entity)
        entity_tokens = [i for i,(a,b) in enumerate(tokens['offset_mapping']) if b>a and a<stop and b>start]
        phrase_tokens = [i for i,(a,b) in enumerate(tokens['offset_mapping']) if b>a and text[a:b].strip().isalpha() and text[a:b] not in STOP_WORDS]
        assert entity_tokens and phrase_tokens and inputs['input_ids'][0].tolist() == tokens['input_ids']
        torch.cuda.synchronize(); begin = time.perf_counter()
        # Explicit FP32 expert; use the official processor's 800/1333 image size.
        out = self.model(**inputs)
        torch.cuda.synchronize(); seconds = time.perf_counter()-begin
        probs = out.logits[0].float().sigmoid()
        # A clothing word alone cannot outrank the required referent noun.
        scores = torch.minimum(probs[:,entity_tokens].mean(-1), probs[:,phrase_tokens].mean(-1))
        result = choose_candidate(out.pred_boxes[0], scores, cfg)
        result.update(seconds=seconds, text=text, entity_tokens=entity_tokens, phrase_tokens=phrase_tokens,
                      model_input_shape=list(inputs['pixel_values'].shape), all_boxes=out.pred_boxes[0].cpu(),
                      all_phrase_scores=scores.cpu())
        return result


def detached_tree(x, device=None):
    if torch.is_tensor(x): return x.detach().clone().to(device=device or x.device)
    if isinstance(x,dict): return {k:detached_tree(v,device) for k,v in x.items()}
    if isinstance(x,list): return [detached_tree(v,device) for v in x]
    if isinstance(x,tuple): return tuple(detached_tree(v,device) for v in x)
    return x


def capture_spatial(model, batch):
    from vg_tta.decota_tastvg_episode_v1 import forward
    captures = []
    handle = model.ground_decoder.decoder.register_forward_pre_hook(
        lambda _m,args,kw: captures.append(detached_tree(kw)), with_kwargs=True)
    try: native, _, records = forward(model,batch)
    finally: handle.remove()
    assert len(captures) == 4, 'Expected initial+refinement pass for each of 2 offsets'
    caches = [captures[1],captures[3]]
    for cache,record in zip(caches,records):
        assert cache['query_tgt'].shape == (len(record['frame_ids']),1,256)
    return native,caches,records


def replay_spatial(model,caches,delta,n):
    views = []
    with torch.autocast('cuda',dtype=torch.float16):
        for cache in caches:
            kw = dict(cache)
            kw['query_tgt'] = kw['query_tgt'] + delta.to(kw['query_tgt'].dtype)[None,None,:]
            views.append(model.ground_decoder.decoder(**kw)[-1,0])
    return torch.stack([views[i%2][i//2] for i in range(n)]).float()


def fit_spatial(model,caches,base,accepted,evidence,*,time_weight=True,cfg=CONFIG,lr=None,steps=None):
    """Fresh residual and optimizer per invocation, never updates model weights."""
    n = len(base); delta = torch.zeros(256,device='cuda',requires_grad=True)
    lr = cfg['lr'] if lr is None else lr; steps = cfg['steps'] if steps is None else steps
    torch.cuda.synchronize(); begin = time.perf_counter(); history = []
    if not accepted:
        return base.detach().cpu().clone(),dict(skipped=True,delta_norm=0.,steps=0,seconds=0.,history=[])
    positions = torch.tensor([x['position'] for x in accepted],device='cuda')
    target = torch.tensor([x['box'] for x in accepted],device='cuda')
    weights = torch.tensor([x['score'] for x in accepted],device='cuda')
    if time_weight:
        r = torch.tensor(evidence,device='cuda').clamp(0,1)[positions]
        weights = weights*(cfg['temporal_floor']+(1-cfg['temporal_floor'])*r)
    weights = weights.detach(); opt = torch.optim.Adam([delta],lr=lr,eps=cfg['optimizer_eps'])
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = replay_spatial(model,caches,delta,n)[positions]
        box_loss = cfg['l1']*(pred-target).abs().sum(-1)+cfg['giou']*(1-generalized_box_iou_aligned_cxcywh(pred,target))
        loss = (weights*box_loss).sum()/weights.sum()+cfg['anchor']*delta.square().sum()
        loss.backward(); assert torch.isfinite(delta.grad).all()
        history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(delta.grad.norm())))
        opt.step()
    with torch.no_grad():
        boxes = replay_spatial(model,caches,delta,n)
        pred = boxes[positions]
        final_box_loss = cfg['l1']*(pred-target).abs().sum(-1)+cfg['giou']*(1-generalized_box_iou_aligned_cxcywh(pred,target))
        final_loss = float((weights*final_box_loss).sum()/weights.sum()+cfg['anchor']*delta.square().sum())
    torch.cuda.synchronize()
    return boxes.cpu(),dict(skipped=False,steps=steps,lr=lr,delta_norm=float(delta.detach().norm()),
                           delta=delta.detach().cpu(),seconds=time.perf_counter()-begin,history=history,
                           weights=weights.cpu().tolist(),final_loss=final_loss,
                           max_box_change=float((boxes.cpu()-base.cpu()).abs().max()))
