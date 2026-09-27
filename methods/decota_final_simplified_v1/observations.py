"""Frozen DINO-T S2 observations, extracted unchanged from the executed method.

Only the four-original-frame path is included. Historical interpolation,
tracking, weak-view, association and oracle routes are intentionally absent.
See SOURCE_MAP.json for exact source symbols and hashes.
"""
import hashlib
import time
import numpy as np
import torch
from torchvision.ops import box_convert, nms

CONFIG = dict(phrase_threshold=.35, distinct_margin=.05, nms_iou=.5)


def _grid(frame_ids, extent, k):
    ids = np.asarray(frame_ids, dtype=np.float64)
    if ids.ndim != 1 or not len(ids) or not np.isfinite(ids).all() or not (np.diff(ids) > 0).all():
        raise ValueError('Expected finite strictly increasing original frame positions')
    if len(extent) != 2 or any(isinstance(x, bool) or int(x) != x for x in extent):
        raise ValueError('Invalid extent')
    a, b = map(int, extent)
    if not 0 <= a <= b < len(ids):
        raise ValueError('Extent outside grid')
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError('Budget must be a nonnegative integer')
    return ids, a, b, min(k, b-a+1)


def uniform_positions(frame_ids, extent, k=4):
    """Same physical-frame bin midpoints as the previously tested uniform rule.

The interval is [first frame_id, last frame_id + 1), in original frame units.
Empty bins are filled by the farthest currently uncovered available position.
"""
    ids, a, b, k = _grid(frame_ids, extent, k)
    if not k:
        return []
    edges = np.linspace(ids[a], ids[b]+1, k+1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pool = [i for i in range(a, b+1) if lo <= ids[i] < hi]
        if pool:
            out.append(min(pool, key=lambda i: (abs(ids[i]-(lo+hi)/2), i)))
    while len(out) < k:
        rest = [i for i in range(a, b+1) if i not in out]
        out.append(max(rest, key=lambda i: (min(abs(ids[i]-ids[j]) for j in out), -i)))
    return sorted(out)


class QuerySubjectParser:
    """Query-only Stanza dependency parser, explicitly not official CoreNLP."""
    def __init__(self, model_dir):
        import stanza
        self.version = stanza.__version__
        self.nlp = stanza.Pipeline("en", dir=str(model_dir), package=None,
            processors={"tokenize": "ewt", "mwt": "ewt", "pos": "ewt_nocharlm",
                        "lemma": "ewt_nocharlm", "depparse": "ewt_nocharlm"},
            use_gpu=False, download_method=None, verbose=False)

    def __call__(self, query):
        words = [w for s in self.nlp(query).sentences for w in s.words]
        chosen, rule = None, None
        # A wh-determiner identifies the queried noun even when it is an object.
        for w in words:
            if w.lemma.lower() in {"which", "what"} and w.deprel == "det":
                chosen = next((v for v in words if v.id == w.head), None)
                if chosen:
                    rule = "wh_determiner_head"
                    break
        if chosen is None:
            for w in words:
                if w.lemma.lower() in {"who", "whom"}:
                    return {"subject": "person", "rule": "explicit_who", "parser_version": self.version}
            if any(w.lemma.lower() == "what" for w in words):
                return {"subject": "", "rule": "unknown_what_no_category_in_query", "parser_version": self.version}
            chosen = next((w for w in words if w.deprel.startswith("nsubj") and w.upos in {"NOUN", "PROPN"}), None)
            rule = "dependency_nominal_subject"
        if chosen is None:
            chosen = next((w for w in words if w.upos in {"NOUN", "PROPN"}), None)
            rule = "first_noun_fallback"
        return {"subject": chosen.lemma.lower() if chosen else "", "rule": rule if chosen else "empty_prefix_fallback",
                "parser_version": self.version}


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


def parse_context(parser, query):
    q=query.strip().lower()
    doc=parser.nlp(q)
    audit=[dict(id=w.id,text=w.text,lemma=w.lemma,upos=w.upos,head=w.head,
                deprel=w.deprel,start=w.start_char,end=w.end_char)
           for s in doc.sentences for w in s.words]
    fallback=lambda reason: dict(eligible=False,reason=reason,context=None,span=None,tokens=audit)
    if len(doc.sentences)!=1:return fallback('multiple_sentences')
    words=doc.sentences[0].words
    if not words:return fallback('empty_query')
    by={w.id:w for w in words};root=next((w for w in words if w.head==0),None)
    if root is None:return fallback('no_root')
    first=words[0];wh=first.lemma.lower() in {'who','whom','which','what'}
    target=None;rule=None
    if wh:
        if first.lemma.lower() in {'what','which'} and first.deprel=='det':
            target=by.get(first.head);rule='explicit_wh_nominal'
            if target is None or target.upos not in {'NOUN','PROPN'}:return fallback('unresolved_wh_category')
        elif first.lemma.lower() in {'who','whom'}:
            subjects=[w for w in words if w.head==root.id and w.deprel.startswith('nsubj')]
            if first not in subjects or len(subjects)!=1:return fallback('unresolved_wh_role_keep_S1')
            q='person who'+q[first.end_char:];rule='wh_subject_to_person_who'
            context=q.rstrip('.?!').rstrip()+'.'
            return dict(eligible=True,reason=rule,context=context,span=[0,6],entity='person',tokens=audit)
        else:return fallback('unresolved_wh_category')
    else:
        subjects=[w for w in words if w.head==root.id and w.deprel.startswith('nsubj') and w.upos in {'NOUN','PROPN'}]
        if len(subjects)==1:target=subjects[0];rule='unique_root_nominal_subject'
        elif root.upos in {'NOUN','PROPN'} and first.lemma.lower() in {'a','an','the'} and first.deprel=='det' and first.head==root.id:
            if any(w.deprel=='cc' and w.head in {v.id for v in words if v.deprel=='conj' and v.head==root.id} for w in words):
                return fallback('coordinated_nominal_fragment')
            target=root;rule='initial_determined_nominal_root'
        else:return fallback('unresolved_declarative')
    # Ambiguous explicitly coordinated subjects are not assigned to just one noun.
    if any(w.deprel=='conj' and w.head==target.id and any(v.deprel=='cc' and v.head==w.id for v in words) for w in words):
        return fallback('coordinated_target')
    context=q.rstrip('.?!').rstrip()+'.';span=[target.start_char,target.end_char]
    assert context[span[0]:span[1]]==target.text.lower()
    return dict(eligible=True,reason=rule,context=context,span=span,entity=context[slice(*span)],tokens=audit)


def pool_gate(probe):
    s=np.asarray(probe['target_scores'],float)
    if not np.isfinite(s).all() or len(s)!=len(probe['boxes']):raise ValueError('Invalid scores')
    order=np.argsort(-s,kind='stable');best=int(order[0]) if len(s) else None
    margin=float(s[best]-(s[order[1]] if len(s)>1 else 0)) if len(s) else 0.
    reason='no_candidates' if not len(s) else 'low_target_score' if s[best]<.35 else 'ambiguous_distinct_instances' if margin<.05 else 'accepted'
    return dict(**{k:v for k,v in probe.items() if k not in {'accepted','reason','margin','unary'}},
        unary=probe['target_scores'],accepted=reason=='accepted',reason=reason,margin=margin,admission_winner=best)


def target_tokens(text, offsets, span):
    """Explicit character occurrence, not a bag of all equal noun tokens."""
    a,b=span
    if not 0<=a<b<=len(text):raise ValueError('Invalid character span')
    selected=[i for i,(lo,hi) in enumerate(offsets) if hi>lo and lo<b and hi>a]
    coverage={j for i in selected for j in range(max(a,offsets[i][0]),min(b,offsets[i][1]))}
    if not selected or any(j not in coverage for j in range(a,b) if not text[j].isspace()):
        raise ValueError('Target span truncated or uncovered')
    return selected


class ContextView:
    """Same expert forward and OLD NMS score, but explicit target occurrence."""
    def __init__(self,expert,spec):self.model=expert.model;self.processor=expert.processor;self.spec=spec
    def __call__(self,rgb,phrase,entity):
        import torch
        text=self.spec['context'];inputs=self.processor(images=rgb,text=text,return_tensors='pt').to('cuda')
        tokens=self.processor.tokenizer(text,return_offsets_mapping=True)
        ent=target_tokens(text,tokens['offset_mapping'],self.spec['span'])
        whole=[i for i,(a,b) in enumerate(tokens['offset_mapping']) if b>a and text[a:b].strip().isalpha() and text[a:b] not in STOP_WORDS]
        assert ent and whole and tokens['input_ids']==inputs['input_ids'][0].tolist()
        torch.cuda.synchronize();tick=time.perf_counter();out=self.model(**inputs);torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        assert max(ent+whole)<out.logits.shape[-1],'Token span truncation'
        p=out.logits[0].float().sigmoid();target=p[:,ent].mean(-1);mixed=torch.minimum(target,p[:,whole].mean(-1))
        z=choose_candidate(out.pred_boxes[0],mixed)
        z.update(seconds=elapsed,text=text,entity_tokens=ent,phrase_tokens=whole,model_input_shape=list(inputs['pixel_values'].shape),
            all_boxes=out.pred_boxes[0].cpu(),all_phrase_scores=mixed.cpu(),raw_token_logits=out.logits[0].float().cpu(),
            target_all_scores=target.cpu(),target_span=self.spec['span'],offsets=tokens['offset_mapping'],input_ids=tokens['input_ids'])
        return z


def candidate_stages(detection):
    xy=box_convert(detection['all_boxes'].float().cpu(),'cxcywh','xyxy').clamp(0,1)
    old=detection['all_phrase_scores'].float().cpu()
    keep=nms(xy,old,.5)
    # Match F21: truncate first, then remove invalid geometry.
    old3=keep[:3];old3=old3[(xy[old3,2:]>xy[old3,:2]).all(1)]
    return dict(boxes=box_convert(xy,'xyxy','cxcywh'),raw=list(range(len(xy))),
                nms=keep.tolist(),old_top3=old3.tolist())


def probe_from_detection(detection,position,frame_id):
    stages=candidate_stages(detection);idx=stages['old_top3']
    return pool_gate(dict(boxes=stages['boxes'][idx],target_scores=detection['target_all_scores'][idx],
        candidate_ids=idx,position=position,frame_id=frame_id))


def tensor_hash(x):
    x = x.detach().contiguous().cpu()
    return hashlib.sha256(str((str(x.dtype), tuple(x.shape))).encode() + x.numpy().tobytes()).hexdigest()


def observations(expert,parses,frames,ids,interval,*,audit=False):
    context,s1=parses['context'],parses['old']
    eligible=context['eligible'] and len(expert.processor.tokenizer(context['context'])['input_ids'])<=256
    positions=uniform_positions(ids,interval,4);obs={};anchors=[]
    for pos in positions:
        if not eligible and not s1['phrase']:continue
        rgb=frames[pos];inp={}
        def hook(m,args,kw):
            for k in ('pixel_values','pixel_mask','input_ids','attention_mask'):
                if k in kw:inp[k]=dict(sha256=tensor_hash(kw[k]),shape=list(kw[k].shape),dtype=str(kw[k].dtype))
        h=expert.model.register_forward_pre_hook(hook,with_kwargs=True) if audit else None
        torch.cuda.synchronize();tick=time.perf_counter()
        try:
            with torch.no_grad():
                d=ContextView(expert,context)(rgb,context['context'],context['entity']) if eligible else expert(rgb,s1['phrase'],s1['entity'])
        finally:
            if h is not None:h.remove()
        torch.cuda.synchronize()
        if eligible:z=probe_from_detection(d,pos,ids[pos])
        else:z=dict(position=pos,frame_id=ids[pos],accepted=d['accepted'],reason=d['reason'],margin=d['margin'],
            boxes=torch.tensor([d['box']]) if d['box'] else torch.empty(0,4),target_scores=[d['score']] if d['box'] else [],candidate_ids=[0] if d['box'] else [])
        receipt=dict(position=pos,frame_id=ids[pos],view='original',scale=1.,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest() if audit else None,
            inputs=inp,seconds=time.perf_counter()-tick,forward_seconds=d['seconds'],text=d['text'],
            target_span=context['span'] if eligible else None,context_active=eligible)
        obs[('original',pos)]=dict(probe=z,receipt=receipt)
        if z['accepted']:
            j=int(np.argmax(z['target_scores']))
            anchors.append(dict(position=pos,frame_id=ids[pos],box=torch.as_tensor(z['boxes'][j]).tolist(),score=float(z['target_scores'][j]),weight=1.))
    return dict(observations=obs,positions4=positions,anchors={'single4':anchors},new_DINO=len(obs),actual_observation_positions=sorted(p for _,p in obs))
