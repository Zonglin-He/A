"""Seal four left/right decisions and check the deployed rule against explicit pair sums."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / '3eed_four_scale_tree_readout_20261120'
TABLE = HERE.parent / '3eed_four_scale_subject_readout_20261120' / 'results/ordinary_rows.csv.gz'
sys.path.insert(0, str(HERE.parent / '3eed_eight_scale_component_ablation_20261120'))
from common import ROOT, catalog, digest, read, row_id, save
from crossplatform.physical_regions import regions, selection, semantic, tree
from predict_delivery import to_model
from research.round158.reader158 import overlaps

CFG = read(HERE / 'PROTOCOL.json')
ARMS = CFG['arms']
PRIVATE = HERE / 'private'
OUT = HERE / 'results'


def pins():
    files = [Path(__file__), HERE/'PROTOCOL.json', ROOT/'crossplatform/ground_regions.py',
             Path(selection.__file__).with_name('role_cut.cpp'), ROOT/'crossplatform/single_frame_sources.json',
             *(Path(m.__file__) for m in (regions, selection, semantic, tree))]
    return {p.name: digest(p) for p in files}


def dense_choice(joint, overlap, weight, smooth):
    mass = joint.sum(axis=1)
    if not np.any(mass > 0):
        return -1, -1
    score = weight * mass
    if smooth:
        score = overlap @ score
    i = int(np.argmax(np.where(mass > 0, score, -np.inf)))
    # Only the diagnostic endpoint uses conditional overlap smoothing in every arm.
    reference = overlap @ ((overlap[i] * weight) @ joint)
    j = int(np.argmax(np.where(joint[i] > 0, reference, -np.inf)))
    return i, j


def arms_for(data, props, words, side, depth, immediate, points):
    parent, stop, p, view = (data[k] for k in ('parent','stop','p','view'))
    n = len(parent)
    if not n:
        return [(-1, -1)] * 4, 0.
    occupancy, overlap = selection.structure(parent, stop, data['size'])
    # Derive disjointness from point-set overlap independently of the interval sweep.
    valid = overlap == 0
    weight = selection.overlap_prior(data, props, 't', points)
    error = 0.
    if depth or immediate:
        joint, _ = selection.shared(parent,valid,stop,p,words,view,side,depth,immediate,'physical')
        choices = [dense_choice(joint, overlap, weight, True)] * 4
    else:
        subject, reference = p[:,words[0]], p[:,words[1]]
        left, right = view[:,0], view[:,2]
        ordered = right[:,None] < left if side < 0 else left[:,None] > right
        product = (occupancy*subject)[:,None] * (occupancy*reference) * valid * ordered
        shared = selection.closed_joint(parent,stop,occupancy,p,words,view,valid,side)
        h = tree.SideEvidence(parent,stop,view,shared_cut=False).marginal(subject,reference,side)
        explicit = product.sum(1)
        np.testing.assert_array_equal(h > 0, explicit > 0)
        np.testing.assert_allclose(h, explicit, rtol=1e-11, atol=1e-14)
        error = float(np.max(np.abs(h-explicit)))
        choices = [dense_choice(joint, overlap, weight, smooth)
                   for joint, smooth in ((shared, True),(product, True),(shared, False),(product, False))]
    actual = selection.select(data,props,words,side,depth,immediate,points)
    emitted = selection.select(data,props,words,side,depth,immediate,points,read_reference=False)
    assert actual == choices[-1] and emitted == (actual[0], -1), (choices,actual,emitted)
    return choices, error


def predict_frame(task):
    frame, rows = task
    assert digest(frame['lex'])==frame['lh'] and digest(frame['word'])==frame['wh']
    with np.load(frame['lex']) as lex, np.load(frame['word']) as word:
        seg = regions.build(lex['sample'],lex['logits'],lex['xyz'],frame['f'],native_logits=True)
        data = regions.build_inventory(seg,word)
    points = regions.Returns(data['xyz'])
    output = []
    for row, old_indices, old_subject, old_ref in rows:
        assert digest(row['role'])==row['rh'] and digest(row['prior'])==row['ph']
        with np.load(row['role']) as role, np.load(row['prior']) as prior:
            words = tuple(row['words'])
            side, depth, immediate = int(role['side']), int(role['depth']), bool(role['immediate'])
            props = {k:role[k] for k in role.files if k.startswith('t_')}
            # Restore the unnormalized box weights used at the public entry point.
            native = props['t_box'][:,:6].copy()
            native[(native[:,3:]<=float(np.float32(1e-6))).any(1),3:] = 0.
            raw = np.maximum(props['t_p'],1e-300)/np.square(overlaps(native)).sum(1)
            np.testing.assert_allclose(raw/raw.sum(), props['t_m'], rtol=1e-10, atol=2e-13)
            props['t_m'] = raw
            choices, err = arms_for(data,props,words,side,depth,immediate,points)
            assert choices[0] == tuple(old_indices), (row_id(row),choices[0],old_indices)
            backup = to_model(row,prior)
            def endpoint(index, category, fallback):
                if index < 0:
                    return backup[fallback][0]
                return regions.complete(data['boxes'][index],data['dimensions'][index,category])
            subjects = np.stack([endpoint(i,words[0],'anchored') for i,_ in choices])
            refs = np.stack([endpoint(j,words[1],'ref_anchored') for _,j in choices])
            np.testing.assert_array_equal(subjects[0],old_subject)
            np.testing.assert_array_equal(refs[0],old_ref)
            output.append((row_id(row),subjects,refs,choices,int(bool(depth or immediate)),err))
    return output


def run(workers):
    code = pins()
    seal = PREVIOUS/'private/SEAL.json'
    baseline = PREVIOUS/'private/decisions.npz'
    assert digest(seal)==CFG['baseline_seal_sha256']
    assert digest(baseline)==CFG['baseline_boxes_sha256']==read(seal)['decisions']
    with np.load(baseline) as z:
        old = {k: (d[:2],s,r) for k,d,s,r in zip(z['keys'],z['details'],z['subjects'],z['refs'])}
    roster = catalog()
    groups = defaultdict(list)
    for row in roster['rows']:
        if row['scope'] in ('ordinary','noun'):
            groups[row['frame']].append((row,*old.pop(row_id(row))))
    assert not old and len(groups)==1358
    tasks = [(roster['frames'][k],v) for k,v in sorted(groups.items())]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n,batch in enumerate(pool.map(predict_frame,tasks,chunksize=2),1):
            rows.extend(batch)
            if n%200 == 0:
                print(f'{n}/{len(tasks)} source-frame inventories',flush=True)
    keys,subjects,refs,choices,ordered,errors = zip(*sorted(rows))
    assert len(keys)==len(set(keys))==1422 and sum(ordered)==128 and pins()==code
    PRIVATE.mkdir(exist_ok=True,mode=0o700)
    path = PRIVATE/'decisions.npz'
    temporary = PRIVATE/'decisions.tmp.npz'
    np.savez_compressed(temporary,keys=keys,subjects=subjects,refs=refs,choices=choices,ordered=ordered)
    temporary.replace(path)
    save(PRIVATE/'SEAL.json',dict(arms=ARMS,previous=digest(seal),decisions=digest(path),code=code,
         frames=len(tasks),rows=len(keys),ordered=sum(ordered),max_marginal_error=max(errors),
         deployed_subject_and_diagnostic_ref_match_dense=True,labels_used=False,FINAL_access=False))
    print('Sealed',len(keys),'rows for all four arms',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=6)
    run(parser.parse_args().workers)
