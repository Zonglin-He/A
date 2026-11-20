"""Seal subject-only replay and the separate independent-inventory ablation before scoring."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / '3eed_eight_scale_component_ablation_20261120'
sys.path.insert(0, str(BASE))
from common import catalog, digest, prior, read, row_id, save
from check_four_default import compact_seal
from crossplatform.physical_regions import regions, selection
from predict_delivery import to_model
from role_cut import separate

CFG = read(HERE/'PROTOCOL.json')
PRIVATE = HERE/'private'
OUT = HERE/'results'


def pins():
    return {p.name:digest(p) for p in (HERE/'PROTOCOL.json', Path(__file__), Path(regions.__file__),
                                       Path(selection.__file__), prior.PREVIOUS/'role_cut.py',
                                       prior.ROOT/'crossplatform/ground_regions.py')}


def replay_frame(item):
    frame, rows = item
    for path, check in [('lex','lh'), ('word','wh')]:
        assert digest(frame[path]) == frame[check]
    with np.load(frame['lex']) as lex, np.load(frame['word']) as word:
        seg = regions.build(lex['sample'],lex['logits'],lex['xyz'],frame['f'])
        assert len(seg['cuts']) == 4
        data = regions.build_inventory(seg,word)
    n = len(data['parent'])
    points = regions.Returns(data['xyz'])
    occupancy, K = selection.structure(data['parent'],data['stop'],data['size'])
    valid = K == 0
    results = []
    for row, expected_digest in rows:
        key = row_id(row)
        path = BASE/'private/compact'/(key+'.npz')
        assert digest(path)==expected_digest and digest(row['role'])==row['rh'] and digest(row['prior'])==row['ph']
        with np.load(row['prior']) as old, np.load(row['role']) as role, np.load(path) as expected:
            backup = to_model(row,old)
            words = tuple(row['words'])
            side,depth,nearest = int(role['side']),int(role['depth']),bool(role['immediate'])
            proposals = {k:role[k] for k in role.files if k.startswith('t_')}
            subject, reference = selection.select(data,proposals,words,side,depth,nearest,points)
            emitted, skipped = selection.select(data,proposals,words,side,depth,nearest,points,read_reference=False)
            assert emitted == subject and skipped == -1
            approx_subject = approx_ref = -1
            if n:
                joint,_ = separate(valid,occupancy,data['p'],words,data['view'],side,depth,nearest)
                weight = selection.overlap_prior(data,proposals,'t',points)
                approx_subject,approx_ref,_ = selection.medoid(joint,K,weight,1.,valid)
            def box(i,word,fallback):
                return backup[fallback][0] if i<0 else regions.complete(data['boxes'][i],data['dimensions'][i,word])
            sb = np.stack([box(i,words[0],'anchored') for i in (emitted,approx_subject)])
            rb = np.stack([box(i,words[1],'ref_anchored') for i in (reference,approx_ref)])
            np.testing.assert_array_equal(sb[0],expected['subject'][1])
            np.testing.assert_array_equal(rb[0],expected['reference'][1])
            results.append((key,sb,rb,[n,data['p'].shape[1],int(depth!=0 or nearest),
                                      subject,reference,approx_subject,approx_ref]))
    return results


def run():
    files = compact_seal()['files']
    assert digest(BASE/'private/COMPACT_SEAL.json') == CFG['compact_prediction_seal']
    cats = catalog()
    groups = defaultdict(list)
    for row in cats['rows']:
        if row['scope'] in ('noun','ordinary'):
            groups[row['frame']].append((row,files[row_id(row)]))
    tasks = [(cats['frames'][key],rows) for key,rows in sorted(groups.items())]
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = sorted(r for batch in pool.map(replay_frame,tasks,chunksize=2) for r in batch)
    keys,subjects,refs,details = zip(*results)
    assert set(keys)==set(files) and len(keys)==sum(CFG['scopes'][k] for k in ('ordinary_relation','controlled_noun_rows'))
    PRIVATE.mkdir(exist_ok=True,mode=0o700)
    temp = PRIVATE/'decisions.tmp.npz'
    np.savez_compressed(temp,keys=keys,subjects=subjects,refs=refs,details=details)
    pred = PRIVATE/'decisions.npz'
    temp.replace(pred)
    save(PRIVATE/'SEAL.json',dict(code=pins(),decisions=digest(pred),compact=CFG['compact_prediction_seal'],
                                 count=len(keys),frame_sources=len(groups),labels_used=False,FINAL_access=False))
    print('Sealed',len(keys),'paired rows across',len(groups),'source-frame keys',flush=True)


if __name__ == '__main__':
    run()
