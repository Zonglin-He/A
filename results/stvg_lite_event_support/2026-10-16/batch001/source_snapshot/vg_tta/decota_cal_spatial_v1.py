"""Query-reference sparse association candidate; no GT or online learning."""
import numpy as np
import torch
from vg_tta.tg_spatial_tta_v1 import VISUAL_NOUNS, COLORS, STOP_WORDS

STATE = {'seated','sitting','standing','holding','sit','stand','hold','kneeling','lying'}
PERSIST = {'remain','keep','stay'}


def parse_query(parser, caption):
    sentences = parser.nlp(caption).sentences
    if len(sentences) != 1:
        return dict(entity='', phrase='', stable_phrase='', reason='multiple_sentences', tokens=[])
    words = sentences[0].words; by = {w.id:w for w in words}
    tokens = [dict(id=w.id,text=w.text,lemma=w.lemma,upos=w.upos,head=w.head,deprel=w.deprel,
                   start=w.start_char,end=w.end_char) for w in words]
    root = next((w for w in words if w.head == 0), None)
    wh = bool(words and words[0].lemma.lower() in {'who','whom','what','which'})
    nominal = None; reason = None
    if wh:
        for w in words:
            if w.lemma.lower() in {'what','which'} and w.deprel == 'det':
                cand = by.get(w.head)
                if cand and cand.upos in {'NOUN','PROPN'}: nominal=cand; reason='explicit_wh_category'; break
        if nominal is None:
            return dict(entity='',phrase='',stable_phrase='',reason='unresolved_question_category',tokens=tokens)
    else:
        candidates = [w for w in words if root and w.head==root.id and w.deprel.startswith('nsubj') and w.upos in {'NOUN','PROPN'}]
        if len(candidates)==1: nominal=candidates[0]; reason='root_subject'
        # Restricted fallback for declaratives beginning with an explicit noun
        # phrase. In the dog failure Stanza labels dog=root and walks=NOUN.
        if nominal is None and words and words[0].text.lower() in {'a','an','the'}:
            prefix = [w for w in words[:4] if w.upos in {'NOUN','PROPN'}]
            if prefix: nominal=prefix[0]; reason='declarative_explicit_entity_fallback'
    if nominal is None:
        return dict(entity='',phrase='',stable_phrase='',reason='unresolved_referent',tokens=tokens)
    keep={nominal.id}; refs=set(); persistent=any(w.lemma.lower() in PERSIST for w in words)
    for w in words:
        if w.head==nominal.id and w.deprel in {'amod','compound'}:
            keep.add(w.id)
            if w.text.lower() in STATE or w.lemma.lower() in STATE: refs.add(w.id)
        if w.head==nominal.id and w.deprel.startswith('nmod') and w.lemma.lower() in VISUAL_NOUNS|COLORS:
            keep.add(w.id)
        if w.head==nominal.id and w.deprel.startswith('acl') and w.lemma.lower() in {'wear','hold','sit','stand'}:
            keep.add(w.id)
            if w.lemma.lower()!='wear': refs.add(w.id)
            for child in words:
                if child.head==w.id and child.deprel in {'obj','obl'}:
                    keep.add(child.id)
                    if w.lemma.lower()!='wear': refs.add(child.id)
    changed=True
    while changed:
        before=len(keep)
        for w in words:
            if w.head in keep and w.deprel in {'case','amod','compound','cc','conj'}:
                # Never recursively attach another person/object via conjunction.
                if w.deprel=='conj' and w.lemma.lower() not in VISUAL_NOUNS|COLORS: continue
                keep.add(w.id)
                if w.head in refs: refs.add(w.id)
        changed=len(keep)!=before
    stable=keep if persistent else keep-refs
    def text(ids): return ' '.join(w.text.lower() for w in words if w.id in ids)
    entity=nominal.text.lower(); phrase=text(keep)
    # Character ranges in generated prompt retain exact original token origins.
    cursor=0; spans=[]
    for w in words:
        if w.id not in keep: continue
        spans.append(dict(original=[w.start_char,w.end_char],prompt=[cursor,cursor+len(w.text)],
                          token_id=w.id,stable=w.id in stable,entity=w.id==nominal.id,
                          reference_state=w.id in refs))
        cursor+=len(w.text)+1
    return dict(entity=entity,phrase=phrase,stable_phrase=text(stable),reason=reason,
        reference_state=text(refs),event_text=' '.join(w.text for w in words if w.id not in keep),
        persistent_state_marker=persistent,tokens=tokens,prompt_spans=spans)


@torch.no_grad()
def observe(expert, rgb, parsed):
    from torchvision.ops import nms,box_convert
    from vg_tta.anchor_appearance_probe_v1 import visual_roi_features
    text=parsed['phrase']+'.'
    inp=expert.processor(images=rgb,text=text,return_tensors='pt').to('cuda')
    tok=expert.processor.tokenizer(text,return_offsets_mapping=True)
    def token_ids(kind):
        spans=[s['prompt'] for s in parsed['prompt_spans'] if kind=='reference' or s[kind]]
        return [i for i,(a,b) in enumerate(tok['offset_mapping']) if b>a and text[a:b].strip().isalpha()
                and text[a:b] not in STOP_WORDS and any(a<d and b>c for c,d in spans)]
    ent,stable,reference=[token_ids(k) for k in ('entity','stable','reference')]
    assert ent and stable and reference and tok['input_ids']==inp['input_ids'][0].tolist()
    captured=[]
    hook=expert.model.model.backbone.conv_encoder.register_forward_hook(lambda m,a,o:captured.append(o))
    try: out=expert.model(**inp)
    finally: hook.remove()
    p=out.logits[0].float().sigmoid()
    entity=p[:,ent].mean(-1)
    su=torch.minimum(entity,p[:,stable].mean(-1))
    ru=torch.minimum(entity,p[:,reference].mean(-1))
    rankscore=torch.maximum(su,ru)
    xy=box_convert(out.pred_boxes[0].float(),'cxcywh','xyxy').clamp(0,1)
    valid=(xy[:,2:]>xy[:,:2]).all(1)
    rankscore=rankscore.masked_fill(~valid,-1)
    keep=nms(xy,rankscore,.5)[:3]
    keep=keep[valid[keep]]
    boxes=box_convert(xy[keep],'xyxy','cxcywh')
    roi,shapes=visual_roi_features(captured[0],boxes)
    return dict(boxes=boxes.cpu(),unary=su[keep].cpu(),reference=ru[keep].cpu(),roi=roi,
        indices=keep.cpu().tolist(),text=text,entity_tokens=ent,stable_tokens=stable,
        reference_tokens=reference,roi_shapes=shapes,model_input_shape=list(inp['pixel_values'].shape),
        GT_online=False)


def reference_node(probes):
    for i,p in enumerate(probes):
        r=np.asarray(p['reference'],float)
        if not len(r): continue
        order=np.argsort(-r,kind='stable'); j=int(order[0])
        margin=r[j]-(r[order[1]] if len(r)>1 else 0)
        if r[j]>=.35 and margin>=.05:
            return (i,j),dict(reason='first_reliable_query_reference',score=float(r[j]),margin=float(margin))
    return None,dict(reason='no_unambiguous_query_reference')


def associate(probes, weight, require_reference=True, shuffle=False):
    """Exact DAG, skip nodes represent unknown; reference mandatory if enabled."""
    ref,audit=reference_node(probes)
    if require_reference and ref is None: return [],audit
    nodes=[]
    for i,p in enumerate(probes):
        f=np.asarray(p['roi'],float); f=f/np.maximum(np.linalg.norm(f,axis=-1,keepdims=True),1e-12)
        if shuffle and len(f)>1: f=np.roll(f,1,axis=0)
        for j,u in enumerate(np.asarray(p['unary'],float)):
            if u<=0: continue
            if require_reference and i==ref[0] and j!=ref[1]: continue
            nodes.append((i,j,float(np.log(u/.35)),f[j]))
    # State remembers whether path has visited the immutable query reference.
    dp={};paths={}
    for k,(i,j,u,f) in enumerate(nodes):
        seen=int((i,j)==ref) if require_reference else 1
        dp[k,seen]=u;paths[k,seen]=[(i,j)]
        for l,(ii,jj,uu,ff) in enumerate(nodes[:k]):
            if ii>=i: continue
            edge=weight*np.log(max((1+float(np.clip(f@ff,-1,1)))/2,1e-12))
            for old in (0,1):
                if (l,old) not in dp: continue
                state=max(old,seen);value=dp[l,old]+edge+u
                if value>dp.get((k,state),-np.inf):
                    dp[k,state]=value;paths[k,state]=paths[l,old]+[(i,j)]
    eligible=[k for k in dp if k[1]==1]
    if not eligible: return [],{**audit,'reason':'no_eligible_path'}
    best=max(eligible,key=lambda k:dp[k])
    if not require_reference and dp[best]<=0: return [],dict(reason='unreferenced_empty',objective=0.)
    path=paths[best]
    assert not require_reference or ref in path
    return path,{**audit,'reference_node':list(ref) if ref else None,'objective':float(dp[best]),
                 'path':[list(v) for v in path],'skipped_observations':len(probes)-len(path),
                 'weight':weight,'reference_constrained':require_reference,'shuffled':shuffle}


def reconstruct_path(base, ids, probes, path):
    from methods.decota_refine_uniform_v1.api import reconstruct
    pseudo=[dict(position=probes[i]['position'],frame_id=ids[probes[i]['position']],
                 box=np.asarray(probes[i]['boxes'][j]).tolist(),score=float(probes[i]['unary'][j]),margin=0.) for i,j in path]
    boxes,rec=reconstruct(torch.as_tensor(base),pseudo,ids,'absolute')
    origin=['native']*len(ids)
    if len(pseudo)>=2:
        for t in range(pseudo[0]['position'],pseudo[-1]['position']+1):origin[t]='interpolation'
    # Existing reconstruction uses single accepted anchor at its observed frame.
    for p in pseudo:origin[p['position']]='observation'
    changed=(boxes-torch.as_tensor(base)).abs().amax(1)>1e-7
    return boxes,dict(pseudo=pseudo,reconstruction=rec,provenance=origin,spatial_modified=changed.tolist())
