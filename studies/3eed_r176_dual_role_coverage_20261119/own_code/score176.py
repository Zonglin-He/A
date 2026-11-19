"""Score the four-slot replacement on the sealed ordinary Validation records."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from scope176 import (N, ORDER, ORIGINAL, PRED_ARMS, PRIVATE, R165_DATA, array_sha, json_atomic,
                      locked, npz_atomic, ordinary_rows, points_for, read, sha, state)
from diagnose176 import (DUAL_CUTS, SUBJECT_CUTS, candidate_ious, dual_diagnosis)
from retrieve176 import OUTPUT, verify
from score import labels
from batch_yaw_iou import many_iou


def diagnosis():
    locked()
    p = PRIVATE/'DIAGNOSIS_SEAL.json'
    seal = read(p)
    assert seal['inputs'] == sha(PRIVATE/'INPUT_LOCK.json') and seal['n'] == N
    assert not seal['FINAL_access']
    assert all(sha(PRIVATE/name) == digest for name,digest in seal['files'].items())
    return sha(p)


def score_direction(pair):
    source, target = pair
    route = f'{source}_to_{target}'
    root = OUTPUT/route
    source_dir = R165_DATA/route
    old_dir = ORIGINAL/route
    seal = read(root/'SEAL.json')
    raw_seal = read(source_dir/'SEAL.json')
    old_seal = read(old_dir/'SEAL.json')
    rows = ordinary_rows(target)
    prior = {x['index']:x for x in read(PRIVATE/f'dual_target_{route}.json')}
    references = read(PRIVATE/'REVIEW_JOIN_PRIVATE.json')['records'][target]
    with np.load(PRIVATE/f'subject_{route}.npz', allow_pickle=False) as z:
        existing = {k:z[k] for k in ('stage','pair_iou','ordinal','relation')}
    result, reference = [], []
    for name, digest in sorted(seal['files'].items()):
        assert sha(root/name)==digest
        assert sha(source_dir/name)==raw_seal['files'][name] and sha(old_dir/name)==old_seal['files'][name]
        with np.load(root/name,allow_pickle=False) as f, np.load(old_dir/name,allow_pickle=False) as o:
            ordinals, slots, anchors = f['ordinals'], f['pca'], f['anchors']
            choices, selected = f['choice'], f['boxes']
            old_query = o['anchor_ids'][np.arange(len(ordinals)),0,o['choice'][:,PRED_ARMS.index('pair_only'),0]]
            assert str(f['original_block'])==old_seal['files'][name]
        with np.load(source_dir/name,allow_pickle=False) as raw:
            data = {k:raw[k] for k in ('raw','points_sha','indices_sha')}
            np.testing.assert_array_equal(raw['ordinals'],ordinals)
        for j,i in enumerate(ordinals):
            idx = int(i)
            row = rows[idx]
            gt, _ = labels(row)
            ious = many_iou(slots[j,0],gt)
            choice = int(choices[j,0])
            np.testing.assert_array_equal(selected[j],slots[j,np.arange(2),choices[j]])
            old_pass = existing['stage'][idx,:,0]
            can = old_pass != 0
            got = ious.max() > SUBJECT_CUTS
            found = ious[choice] > SUBJECT_CUTS
            assert np.all(~got | can) and np.all(~found | got)
            stage = np.where(~can,0,np.where(~got,1,2+found))
            result.append(dict(ordinal=idx, iou=ious[choice], slot_ious=ious,
                               stage=stage, chosen_changed=bool(anchors[j,0,choice]!=old_query[j])))
            if idx not in prior:
                continue
            q = prior[idx]
            item = {k:q[k] for k in ('source','target','index','status','relation','recording','frame','origin')}
            if q['status']=='verified':
                truth = np.stack((gt,np.asarray(references[str(idx)]['ref_box'])))
                def load_points():
                    points, ids = points_for(row)
                    assert array_sha(points)==data['points_sha'][j].tobytes()
                    assert array_sha(ids)==data['indices_sha'][j].tobytes()
                    return points
                x, _, _ = candidate_ious(data['raw'][j], truth, anchors[j], slots[j],
                                          load_points, float(DUAL_CUTS.min()))
                dual = dual_diagnosis(x,anchors[j],choices[j,None])
                np.testing.assert_array_equal(np.asarray(dual['good_roles'])[:,0],
                                              np.asarray(q['good_roles'])[:,0])
                np.testing.assert_allclose(dual['role_slot_ious'][0],ious,atol=2e-12,rtol=0)
                item.update(dual)
            reference.append(item)
    assert len(result)==seal['n']==len(rows) and len(reference)==len(prior)
    np.testing.assert_array_equal([r['ordinal'] for r in result],existing['ordinal'])
    dst = PRIVATE/f'new_subject_{route}.npz'
    npz_atomic(dst,**{k:np.array([r[k] for r in result]) for k in result[0]},
               prediction=sha(root/'SEAL.json'))
    dual = PRIVATE/f'new_dual_{route}.json'
    json_atomic(dual,reference)
    print('R176_RETRIEVAL_SCORED',route,len(result),len(reference),flush=True)
    return {dst.name:sha(dst),dual.name:sha(dual)}


def main():
    diag = diagnosis()
    pred = verify()
    state('post_seal_scoring', 'Score the frozen retriever using labels only after all predictions are sealed')
    outputs = {}
    with ProcessPoolExecutor(max_workers=6,mp_context=get_context('spawn')) as pool:
        for value in pool.map(score_direction,ORDER):
            outputs.update(value)
    json_atomic(PRIVATE/'RETRIEVAL_SCORE_SEAL.json', dict(n=pred['n'],diagnosis=diag,
                predictions=sha(OUTPUT/'PREDICTION_SEAL.json'),files=outputs,FINAL_access=False))


if __name__ == '__main__':
    main()
