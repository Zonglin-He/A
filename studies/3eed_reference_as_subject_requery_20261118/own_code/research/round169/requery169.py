"""Run each textual grounding request on the same frozen source and physical sample."""
from collections import Counter
from functools import lru_cache
import fcntl
import gc
import hashlib
import time

import numpy as np
from scipy.special import logsumexp

from scope169 import HERE, PANEL, PANEL_N, PLATFORMS, QUERY_ARMS, ROOT, atomic_json, lock, read, roster, sha, status
from cache169 import archived_case
from panel169 import lookup
from common160 import array_sha, npz_atomic
from labels166 import parser_record
from predict165 import cached, points_for
from replay168 import oriented
from retrieval167 import ranked_distinct, support_matrix, support_overlap, transfer
from scope167 import BANK
from select168 import solve
from crossplatform.ground_single_frame import orient_box


def tag(n,label):
    return f'{n:03d}_{label}'


def graph_for(spec,doc):
    roles = ('T','R0')[:len(spec['spans'])]
    entities = [dict(role=k,span=span) for k,span in zip(roles,spec['spans'],strict=True)]
    graph = dict(entities=entities,relations=[])
    if len(entities)==2:
        start = spec['predicate_pos']
        direction = 'LEFT' if spec['sign']==1 else 'RIGHT'
        hits = [(i,w) for i,s in enumerate(doc['sentences']) for w in s['words'] if
                w.get('start_char')==start and w['text'].casefold()==direction.casefold()]
        assert len(hits)==1
        i,w = hits[0]
        graph['relations'].append(dict(subject='T',reference='R0',predicate=direction,known_frame=True,
                                       sentence=i,governor_word=w['id']))
    return graph


def native_forward(model, row, points, indices, parsed, spec=None):
    from grounding158 import select_grounding_box
    from interface66 import capture
    from model51 import SENTINEL
    from model76 import tensor_data, tensor_prefix
    from prepare76 import geometry
    from core89 import route89, tensor_route89
    from parse97 import dependency_document
    from routing88 import token_roles88

    text = row['caption'] if spec is None else spec['text']
    assert len(model.base.tokenizer(text+SENTINEL)['input_ids'])<=256
    t0=time.perf_counter()
    z, receipt = capture(model.base,points,indices,text)
    if spec is None:
        graph, doc = parsed['graph'], parsed['doc']
        modified = row
    else:
        doc = dependency_document(model.parser(text))
        graph = graph_for(spec,doc)
        modified = dict(row,caption=text,id=hashlib.sha256(text.encode()).hexdigest())
    spatial,_ = geometry(modified,z)
    prefix, pixels = tensor_prefix(z), tensor_data(spatial,modified)
    roles, offsets = token_roles88(model.base,modified,z,graph,doc)
    matcher = model.matcher
    assert matcher.route is None and not matcher.use_interface
    layer = model.base.cross_encoder.layers[-1].cross_layer.cross_lv
    cache = []
    hook = layer.register_forward_hook(lambda _m,_a,o:cache.append(o[1][0].cpu().numpy()))
    try:
        initial,_,_ = matcher.main(model.base,prefix,pixels,'bidirectional')
    finally:
        hook.remove()
    attention, = cache
    route,_ = route89(modified,dict(z,frozen_probability=initial.cpu().numpy()),spatial,graph,roles,attention,offsets,doc)
    matcher.route = tensor_route89(route)
    matcher.use_interface = True
    try:
        probability,_,_ = matcher.main(model.base,prefix,pixels,'bidirectional')
    finally:
        matcher.route = None
        matcher.use_interface = False
        matcher.local = {}
    p = probability.cpu().numpy()
    winner,utility = select_grounding_box(z['boxes'],p)
    affinity = ((roles@attention)/np.maximum(1,roles.sum(1))[:,None]).astype(np.float32)
    return dict(z=z,receipt=receipt,roles=roles,affinity=affinity,probability=p,utility=utility,
                subject_winner=int(winner),forward_seconds=time.perf_counter()-t0)


def old_role_evidence(view, role, extra):
    z, roles = view['z'], view['roles']
    native = z['boxes']
    geometry = np.concatenate((np.c_[native,np.zeros(len(native))],extra))
    keep = np.sort(np.unique(z['xyz'][0],axis=0,return_index=True)[1])
    masks = support_matrix(geometry,z['xyz'][0,keep],len(native))
    hits, jaccard = support_overlap(masks)
    count = masks.sum(1)
    allowed = (count>=2)&(geometry[:,3:6]>.12).all(1)
    scores = np.full(len(geometry),-np.inf)
    if roles[role].any() and allowed.any():
        logits = z['query'].astype(float)@z['tokens'].astype(float).T/.07
        v = logsumexp(logits[:,roles[role]],axis=1)-np.log(roles[role].sum())
        displaced = transfer(v-v.max(),hits,jaccard,len(native))
        mass = (masks@view['affinity'][role,keep])/np.sqrt(np.maximum(count,1))
        density = np.log(np.maximum(mass/max(float(mass[allowed].max()),1.e-12),.025))
        scores = (displaced+density)/2
    return scores, allowed, jaccard


def native_donors(view, reference_token_role=None, original_regions=None):
    native = view['z']['boxes']
    count = len(native)
    geom = np.c_[native,np.zeros(count)]
    if reference_token_role is not None:
        extra = np.empty((0,7)) if original_regions is None else original_regions
        score, allowed, jaccard = old_role_evidence(view,reference_token_role,extra)
        shared_input = score
        score,allowed,jaccard = score[:count],allowed[:count],jaccard[:count,:count]
        allowed = allowed & np.isfinite(score)
    else:
        seed = np.unique(view['z']['xyz'][0],axis=0)
        masks = support_matrix(geom,seed,count)
        _,jaccard = support_overlap(masks)
        allowed = (masks.sum(1)>=2)&(native[:,3:]>.12).all(1)
        allowed[view['subject_winner']] = True
        score = np.log(np.maximum(view['utility']/view['utility'].max(),1.e-9))
        shared_input = None
    chosen = ranked_distinct(score,allowed,geom,jaccard)
    assert len(chosen)<=BANK
    if reference_token_role is None:
        assert len(chosen) and int(chosen[0])==view['subject_winner']
    return chosen,score[chosen],shared_input


def associate(donor_masks, common_masks, unary):
    """Match in the shared sample, then transfer one best score to each fixed candidate."""
    n = len(common_masks)
    result = np.full(n,-np.inf)
    match = np.full(len(donor_masks),-1,dtype=np.int16)
    witness = np.zeros(len(donor_masks))
    if not len(donor_masks) or not n:
        return result,match,witness
    hits = donor_masks.astype(np.float32)@common_masks.astype(np.float32).T
    union = donor_masks.sum(1)[:,None]+common_masks.sum(1)[None,:]-hits
    sim = np.divide(hits,union,out=np.zeros_like(union),where=union>0)
    scalar = unary-unary.max()
    for i,row in enumerate(sim):
        j = int(row.argmax())
        if hits[i,j]<3 or row[j]<.25 or (row==row[j]).sum()!=1:
            continue
        match[i] = j
        witness[i] = row[j]
        result[j] = max(result[j],scalar[i]+np.log(row[j]))
    if np.isfinite(result).any():
        result -= result.max()
    return result,match,witness


def pad(data,size,fill):
    dst = np.full((size,*data.shape[1:]),fill,dtype=data.dtype)
    dst[:len(data)] = data
    return dst


def run_case(model, entry):
    row = roster()[entry['target']][entry['ordinal']]
    k, previous, base, earlier = lookup(entry)
    old = archived_case(previous,k)
    points, indices = points_for(row)
    assert array_sha(points)==base['points_sha'][k].tobytes().hex()
    assert array_sha(indices)==base['indices_sha'][k].tobytes().hex()
    parsed = parser_record(row['id'])
    data = native_forward(model,row,points,indices,parsed)
    orig = data['z']
    np.testing.assert_array_equal(orig['boxes'],base['raw'][k])
    np.testing.assert_array_equal(orig['xyz'][0],earlier['seeds'][k])
    lo,hi = base['offsets'][k:k+2]
    np.testing.assert_array_equal(orig['query'],base['q'][k])
    np.testing.assert_array_equal(orig['tokens'],base['tokens'][lo:hi])
    _, _, saved_p = cached(entry['source'],row,orig,data['receipt'],False)
    np.testing.assert_allclose(data['probability'],saved_p,rtol=0,atol=1.e-10)
    assert data['subject_winner']==old['winners'][1,0]
    seed_support = np.unique(points,axis=0)
    native = orig['boxes']
    extra = old['novel'][np.isfinite(old['novel']).all(1)]
    common = np.concatenate((np.c_[native,np.zeros(len(native))],extra))
    role = entry['role']
    positions = old['bank'][role]>=0
    bank_ids = old['bank'][role,positions]
    common_masks = support_matrix(common,seed_support,len(native))[bank_ids]
    common_boxes = oriented(row,old,native,bank_ids,lambda _:points) if len(bank_ids) else np.zeros((0,7))
    out = []
    replay_diff = 0.
    for arm in 'ABCD':
        if arm=='A':
            view = data
        elif entry['status']=='supported':
            view = native_forward(model,row,points,indices,None,entry[arm])
        else:
            view = None
        tmap = time.perf_counter()
        score = old['score'].copy()
        if view is not None:
            z = view['z']
            np.testing.assert_array_equal(z['xyz'],orig['xyz'])
            np.testing.assert_array_equal(z['point_indices'],indices)
            ids = z['query_points_sample_inds']
            selected, values, shared = native_donors(view, role if arm=='A' else (1 if arm=='B' else None),
                                                       extra if arm=='A' else None)
            natural = (np.stack([orient_box(points,z['boxes'][j]) for j in selected]) if len(selected) else np.zeros((0,7)))
            if arm=='A':
                bank_score = score[role,positions].copy()
                if len(bank_score):
                    computed = shared[bank_ids]
                    replay_diff = float(np.abs(computed-computed.max()-bank_score).max())
                    assert replay_diff<1.e-7
                links = np.full(len(selected),-1,dtype=np.int16)
                jaccard = np.zeros(len(selected))
            else:
                from geometry162 import in_box
                # Native query boxes were axis-aligned before the shared PCA readout.
                donor_masks = np.stack([in_box(seed_support,np.r_[z['boxes'][j],0.]) for j in selected]) if len(selected) else np.zeros((0,len(seed_support)),bool)
                bank_score,links,jaccard = associate(donor_masks,common_masks,values)
            subject = orient_box(points,z['boxes'][view['subject_winner']])
            forward_seconds = view['forward_seconds']
            anchor_slots = int((ids==orig['query_points_sample_inds']).sum())
            anchor_shared = len(np.intersect1d(ids,orig['query_points_sample_inds']))
            token_count = len(z['tokens'])
        else:
            natural, links, jaccard = np.zeros((0,7)),np.zeros(0,np.int16),np.zeros(0)
            bank_score = np.full(len(bank_ids),-np.inf)
            subject = np.full(7,np.nan)
            forward_seconds,anchor_slots,anchor_shared,token_count = 0.,-1,-1,0
        if arm!='A':
            score[role,positions] = bank_score
            score[role,old['bank'][role]==-1] = -2. if np.isfinite(bank_score).any() else 0.
        choices = np.stack([solve(old['bank'],score,old['left'],old['overlap'],{role:entry['sign']},
                                   overlap_on=False,independent=independent)[0][[0,role]] for independent in (True,False)])
        boxes = oriented(row,old,native,choices,lambda _:points)
        assert choices[0,0]==old['winners'][1,0]
        np.testing.assert_array_equal(boxes[0,0],old['boxes'][1,0])
        elapsed = time.perf_counter()-tmap
        out.append(dict(natural=pad(natural,BANK,np.nan),native_n=len(natural),links=pad(links,BANK,-1),
                        overlaps=pad(jaccard,BANK,0.),bank_score=pad(bank_score,BANK,-np.inf),
                        chosen=choices,boxes=boxes,native_subject=subject,seconds=forward_seconds,
                        readout_seconds=elapsed,query_slots_equal=anchor_slots,query_seeds_shared=anchor_shared,
                        token_count=token_count,executed=int(view is not None)))
    return dict(**{key:np.stack([e[key] for e in out]) for key in out[0]},
                common=pad(common_boxes,BANK,np.nan),common_n=len(bank_ids),seeds=earlier['seeds'][k],
                replay_diff=replay_diff)


def predict():
    import torch
    from evidence162 import EvidenceGrounder
    from model51 import fingerprint

    design=lock('panel_forward',('panel169.py','cache169.py','requery169.py'))
    assert (HERE/'CACHE_EVALUATION_AUDIT.json').exists()
    panel=PANEL/'INPUT_PRIVATE.json'
    scope=read(HERE/'PANEL_INPUT_SEAL.json')
    assert sha(panel)==scope['roster'] and not scope['target_identity_read']
    records=read(panel)
    dest=PANEL/'predictions'
    dest.mkdir(exist_ok=True)
    results={}
    counts=Counter()
    status('requery', panel_directions=len(records))
    with (ROOT/'research/GPU_RESOURCE.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for source in PLATFORMS:
            model = EvidenceGrounder(source)
            torch.set_num_threads(1)
            original=fingerprint(model.base),fingerprint(model.matcher)
            with torch.inference_mode():
                for i,entry in enumerate(records):
                    if entry['source']!=source:
                        continue
                    data=run_case(model,entry)
                    filename=tag(i+1,'requery')+'.npz'
                    path=dest/filename
                    npz_atomic(path,arms=np.asarray(QUERY_ARMS),**data)
                    results[filename]=sha(path)
                    counts['rows']+=1
                    counts['full_model_forwards']+=int(data['executed'].sum())
                    print('R169 frozen query panel',source,i+1,'ready',int(data['executed'].sum()),flush=True)
            assert original==(fingerprint(model.base),fingerprint(model.matcher))
            del model
            gc.collect()
            torch.cuda.empty_cache()
    assert len(results)==PANEL_N==counts['rows']
    atomic_json(HERE/'QUERY_PREDICTION_SEAL.json',dict(n=PANEL_N,files=results,counts=dict(counts),
                    design_lock=design, panel_sha256=sha(panel), target_identity_read=False,
                    unchanged_model_fingerprints=True,FINAL_access=False))


if __name__=='__main__':
    predict()
