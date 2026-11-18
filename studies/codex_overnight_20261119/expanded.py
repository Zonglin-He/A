"""Run the fixed role permutation comparison on additional ordinary R159 frames."""
from collections import Counter,defaultdict
import argparse
import fcntl
import gc
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from binding_collect import HERE, PLATFORMS, ROOT, SOURCE_SETTINGS, read, sha, atomic_json, measure
from edge_predict import INPUT, PRIOR_INPUT, R171

DATA = HERE/'private/expanded'
PROTOCOL = HERE/'expanded_protocol.json'
ARMS = tuple(read(PROTOCOL)['arms'])
SELECTION = (0,2,3,4,5,6)


def tag(text):
    return hashlib.sha256(text.encode()).hexdigest()


def select():
    from calibrate168 import ordinary_content, edges_of
    from panel169 import rewrite
    from retrieval167 import direct_edges
    from scope170 import R159, REC_INPUT
    from select170 import HORIZONTAL, frame, indices, parser, previously_reviewed

    DATA.mkdir(parents=True,mode=0o700,exist_ok=True)
    out = DATA/'ROSTER.json'
    assert not out.exists()
    receipt = read(R159/'RECORDING_INPUT_SEAL.json')
    assert receipt['input_sha256']==sha(REC_INPUT) and not receipt['FINAL_access'] and not receipt['target_labels_used_for_selection']
    r170_lock = read(ROOT/'research/round170/INPUT_SEAL.json')
    r171_lock = read(R171/'INPUT_SEAL.json')
    assert r170_lock['registry']==sha(PRIOR_INPUT) and r171_lock['registry']==sha(INPUT)
    used = [x['row'] for p in (PRIOR_INPUT,INPUT) for x in read(p)]
    previous = {frame(r).parent.parent for r in used}
    points, rgb, originals, old_recordings = ordinary_content()
    known_frames, known_points, known_rgb = previously_reviewed()
    seen_f = set(known_frames)|set(originals)|{frame(r) for r in used}
    seen_p = set(known_points)|set(points)|{r['point_sha256'] for r in used}
    seen_i = set(known_rgb)|set(rgb)|{r['image_sha256'] for r in used}
    identify = indices()
    parse = parser()
    stats = defaultdict(Counter)
    selected = []
    aliases = {}
    indexed = enumerate(read(REC_INPUT))
    chosen = sorted(indexed,key=lambda it:tag(f"frozen-role-expansion-20261119|{it[1]['id']}"))
    for i,r in chosen:
        p = r['platform']
        stats[p]['registered']+=1
        physical = frame(r)
        rec = physical.parent.parent
        ph,ih = r['point_sha256'],r['image_sha256']
        if rec in old_recordings:
            stats[p]['prior_selector_recording']+=1
            continue
        if physical in seen_f or ph in seen_p or ih in seen_i:
            stats[p]['physical_content_overlap']+=1
            continue
        if r['id'] not in identify:
            stats[p]['unregistered_score_join']+=1
            continue
        loc,index = identify[r['id']]
        assert loc==physical
        row = dict(r,seed=0,caption=r['caption'].strip(),object_index=index)
        label = len(selected)+1
        status = 'no_horizontal'
        roles = graph = None
        role = sign = 0
        if HORIZONTAL.search(row['caption']):
            graph = parse(row['caption'])
            roles = direct_edges(edges_of(graph['graph']))
            status = 'no_eligible_edge'
        if roles:
            role,sign = min(roles,key=lambda ids:tag(f"frozen-role-expansion-20261119|{r['id']}|{ids}"))
            q = rewrite(row['caption'],graph,roles,role,sign)
            status = q['status']
        else:
            q = dict(status=status)
        aliases.setdefault(rec,len(aliases)+1)
        selected.append(dict(case=label,target=p,row=row,baseline_index=i,parsed=graph,role=role,sign=sign,
                             rewrite=q,recording=aliases[rec],mechanism_recording_reuse=rec in previous))
        stats[p][status]+=1
        seen_f.add(physical)
        seen_p.add(ph)
        seen_i.add(ih)
    assert len(selected)==len({x['row']['point_sha256'] for x in selected})
    atomic_json(out,selected)
    atomic_json(DATA/'ROSTER_SEAL.json',dict(inputs=sha(out),design=sha(PROTOCOL),selector=sha(Path(__file__)),
            n=len(selected),stats={p:dict(v) for p,v in stats.items()},
            counts={p:sum(r['target']==p for r in selected) for p in PLATFORMS},
            historical_recordings=len(aliases),new_recordings=sum(rec not in previous for rec in aliases),
            two_directions_per_physical_frame=True,selected_with_target_gt=False,protected_data_access=False))
    print(read(DATA/'ROSTER_SEAL.json'),flush=True)


def collect():
    import torch
    from binding_predict import assignments
    from common160 import npz_atomic
    from evidence162 import EvidenceGrounder
    from model51 import fingerprint
    from predict170 import old_baseline
    from predict171 import predict_case, read_npz

    d = read(DATA/'ROSTER_SEAL.json')
    roster = DATA/'ROSTER.json'
    assert d['design']==sha(PROTOCOL) and d['inputs']==sha(roster) and not d['protected_data_access']
    data = read(roster)
    pins = dict(design=sha(PROTOCOL),roster=sha(roster),code={p:sha(HERE/p) for p in (
        'expanded.py','binding_collect.py','binding.py','binding_predict.py','edges.py')},
        old=sha(R171/'PREDICTION_LOCK.json'),source_settings=sha(SOURCE_SETTINGS))
    path = DATA/'predictions'
    path.mkdir(exist_ok=True,mode=0o700)
    lock = path/'LOCK.json'
    if lock.exists():
        assert read(lock)==pins
    else:
        atomic_json(lock,pins)
    seal = path/'SEAL.json'
    assert not seal.exists()
    settings = read(SOURCE_SETTINGS)['sources']
    files = {}
    salience = Counter()
    start = perf_counter()
    with (ROOT/'research/GPU_RESOURCE.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(1)
        for source in PLATFORMS:
            net = None
            initial = None
            winners = old_baseline(source)
            total = 0
            for entry in data:
                if source==entry['target']:
                    continue
                idx = entry['case']
                name = f'{idx:04d}_{source}.npz'
                dest = path/name
                if dest.exists():
                    saved = read_npz(dest)
                    assert saved['lock']==sha(lock)
                    files[name] = sha(dest)
                    continue
                seen = entry['rewrite']['status']=='supported'
                known = row_support = False
                elapsed = 0.
                if seen:
                    if net is None:
                        net = EvidenceGrounder(source)
                        initial = fingerprint(net.base),fingerprint(net.matcher)
                    before = perf_counter()
                    with torch.inference_mode():
                        old = predict_case(net,entry,settings[source])
                        res = measure(net,entry,settings[source],old)
                        chosen = assignments(old,res,np.ones(2),name)
                    boxes = chosen['boxes'][np.array(SELECTION)]
                    seeds = old['seeds']
                    known = bool(len(res['subject_common']) and len(res['reference_common']))
                    row_support = bool((boxes[4,0]!=boxes[3,0]).any())
                    elapsed = perf_counter()-before
                else:
                    _,fallback = winners[entry['baseline_index']]
                    boxes = np.broadcast_to(np.stack((fallback,np.full(7,np.nan))), (len(ARMS),2,7))
                    seeds = np.empty((0,3),np.float32)
                npz_atomic(dest,boxes=boxes,seeds=seeds,arms=np.asarray(ARMS),lock=sha(lock),
                           eligible=known,rewrite=seen,pair_changed_subject=row_support,seconds=elapsed)
                files[name] = sha(dest)
                total += 1
                salience[(source,'rewritten')] += int(seen)
                if total%80==0:
                    print(source,total,'rewrites',salience[(source,'rewritten')],flush=True)
            if net is not None:
                assert initial==(fingerprint(net.base),fingerprint(net.matcher))
                del net
                gc.collect()
                torch.cuda.empty_cache()
    assert len(files)==2*len(data)
    atomic_json(seal,dict(files=files,n=len(files),lock=sha(lock),seconds=perf_counter()-start,
                          target_label_read=False,protected_data_access=False))
    print('Sealed independent ordinary frame predictions',len(files),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('select','collect'))
    globals()[parser.parse_args().phase]()
