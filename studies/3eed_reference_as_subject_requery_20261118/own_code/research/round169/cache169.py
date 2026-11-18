"""Enable the exact R168 semantic correction before the old rejection switch."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scope169 import CACHE_ARMS, HERE, N, ORDER, OUT, R168, WORK, atomic_json, lock, read, roster, sha, status
from audit167 import row_edges
from common160 import array_sha, npz_atomic
from scope166 import BLOCK
from labels166 import MISSING_TRUTH, RIGHT, key_for, label_pair, mask_owners, occupied_owners, physical_labels, truth_status
from predict165 import points_for
from replay168 import oriented
from retrieval167 import support_matrix
from score165 import anonymizer, many_iou, sensor_values, table, write_table
from score167 import group_records
from score168 import archived, new_independent_truth, reference_order, verify_seals
from select168 import direct_signs, scramble, solve


def scores_for(old, new, row):
    original, bank = old['score'], old['bank']
    change = np.zeros_like(original)
    # A role-wise offset does not survive the final max-centering in scramble().
    physical = bank >= 0
    change[physical] = new[physical] - original[physical]
    shuffled, _ = scramble(original, change, bank, row['id'])
    return original, new, shuffled


def reloaded_points(row, base, k):
    points, ids = points_for(row)
    assert base['points_sha'][k].tobytes().hex() == array_sha(points)
    assert base['indices_sha'][k].tobytes().hex() == array_sha(ids)
    return points


def archived_case(previous, k):
    fields = ('bank', 'score', 'left', 'overlap', 'winners', 'flip_winners', 'boxes', 'flip_boxes', 'novel')
    return {x: previous[x][k] for x in fields}


def infer(source, target):
    tag = f'{source}_to_{target}'
    dest = WORK / 'cache' / tag
    dest.mkdir(parents=True, exist_ok=True)
    proof = verify_seals()
    saved = {}
    counts = Counter()
    for name, previous, native, base, earlier in cached_blocks(proof, source, target):
        boxes, ids, ranks = [], [], []
        for k, ordinal in enumerate(previous['ordinals']):
            row = roster()[target][int(ordinal)]
            old = archived_case(native, k)
            _, direct = row_edges(earlier, k)
            old_u, new_u, bad_u = scores_for(old, previous['new_score'][k], row)
            common = old['bank'], old['left'], old['overlap']
            sign = direct_signs(direct)
            winner = np.stack([solve(common[0], u, *common[1:], sign, overlap_on=False, independent=(j == 0))[0]
                               for j, u in enumerate((old_u, old_u, new_u, bad_u))])
            np.testing.assert_array_equal(winner[:2], previous['chosen'][k, [0, 2]])
            geometry = oriented(row, old, base['raw'][k], winner, lambda _: reloaded_points(row, base, k))
            np.testing.assert_array_equal(geometry[:2], previous['boxes'][k, [0, 2]])
            rank = np.stack([solve(common[0], u, *common[1:], sign, overlap_on=False, independent=True)[0]
                             for u in (old_u, new_u, bad_u)])
            counts.update(rows=1, relation=int(bool(direct)), new_switch=int(winner[2,0]!=winner[1,0]),
                          control_switch=int(winner[3,0]!=winner[1,0]))
            ids.append(winner)
            boxes.append(geometry)
            ranks.append(rank)
        p = dest / name
        npz_atomic(p, ordinals=previous['ordinals'], chosen=np.stack(ids), boxes=np.stack(boxes),
                   reference_only=np.stack(ranks), arms=np.asarray(CACHE_ARMS))
        saved[name] = sha(p)
    assert counts['rows'] == proof['directions'][tag]['n']
    result = dict(files=saved, counts=dict(counts), parent=proof['directions'][tag]['seal'])
    atomic_json(dest / 'SEAL.json', result)
    print('R169 cache prediction', tag, dict(counts), flush=True)
    return dict(seal=sha(dest / 'SEAL.json'), **counts)


def cached_blocks(proof, source, target):
    for d, a, b, c in archived(proof, source, target):
        yield f"{int(d['ordinals'][0]) // BLOCK:04d}.npz", d, a, b, c


def infer_all():
    locked = lock('cache', ('cache169.py',))
    status('cache_replay')
    with ProcessPoolExecutor(max_workers=6, mp_context=get_context('spawn')) as p:
        calls = {(s,t):p.submit(infer,s,t) for s,t in ORDER}
        ledger = {f'{s}_to_{t}': result.result() for (s,t),result in calls.items()}
    assert sum(v['rows'] for v in ledger.values()) == N
    atomic_json(HERE / 'CACHE_PREDICTION_SEAL.json', dict(n=N, directions=ledger, design_lock=locked,
                parent_seal=sha(R168/'PREDICTION_SEAL.json'), target_label_read=False, FINAL_access=False))


def known_reference(row, earlier, k, ordinal, target, edge, role, sign, names, gt):
    e, _ = row_edges(earlier, k)
    cam = SimpleNamespace(edges=e, role_spans=earlier['spans'][k], width=int(earlier['width'][k]),
            camera_intrinsic=earlier['camera_intrinsic'][k], model_to_camera=earlier['model_to_camera'][k],
            distortion=earlier['distortion'][k])
    kind, ref = label_pair(row, cam, 0, role, sign, names, gt)
    valid = set(new_independent_truth().get((target, int(ordinal), edge), ()))
    if kind == 'visual_verified':
        valid.add(ref)
    return kind, ref, next(iter(valid)) if len(valid) == 1 else -1


def observation(row, source, target):
    _, frame, index = key_for(row)
    folder = Path(frame)
    alias = anonymizer()
    return dict(source=source, target=target, expression=alias('obs',(frame,index)),
                frame=alias('frame',folder), recording=alias('rec',folder.parent.parent.as_posix()))


def score(source, target):
    tag = f'{source}_to_{target}'
    dest = WORK / 'cache' / tag
    seal = read(HERE / 'CACHE_PREDICTION_SEAL.json')
    assert seal['n'] == N and not seal['target_label_read'] and not seal['FINAL_access']
    assert sha(dest / 'SEAL.json') == seal['directions'][tag]['seal']
    parent = verify_seals()
    prior = {r['expression']:r for r in table(R168/'results/all_rows.csv') if (r['source'],r['target'])==(source,target)}
    results, edges = [], []
    maxdiff = 0.
    for name, previous, native, base, earlier in cached_blocks(parent, source, target):
        path = dest / name
        assert sha(path) == read(dest / 'SEAL.json')['files'][name]
        with np.load(path, allow_pickle=False) as data:
            d = dict(data)
        assert tuple(d['arms']) == CACHE_ARMS
        np.testing.assert_array_equal(d['ordinals'], base['ordinals'])
        for k, ordinal in enumerate(d['ordinals']):
            row = roster()[target][int(ordinal)]
            identity = observation(row, source, target)
            last = prior[identity['expression']]
            record = dict(**identity, camera_horizontal=int(last['camera_horizontal']), active_factor=int(last['active_factor']))
            names, gt, seeds, own = physical_labels(row, earlier['seeds'][k])
            output = d['boxes'][k]
            subject = occupied_owners(output[:,0], seeds, own, len(names))
            iou = many_iou(output[:,0], gt[0])
            witness = sensor_values(output[:,0], gt[0], target, int(row['suffix'] or 0))
            maxdiff = max(maxdiff, float(np.abs(witness-iou).max()))
            assert maxdiff < 1.e-7 and all(np.array_equal(iou > t, witness > t) for t in (.25,.5))
            assert iou[0] == float(last['independent_iou']) and iou[1] == float(last['left_only_iou'])
            for j, arm in enumerate(CACHE_ARMS):
                record.update({arm+'_iou':float(iou[j]), arm+'_subject':truth_status(subject[j],0),
                               arm+'_changed':int(d['chosen'][k,j,0]!=d['chosen'][k,0,0])})
            results.append(record)
            if not record['camera_horizontal']:
                continue
            old = archived_case(native,k)
            old_u, new_u, bad_u = scores_for(old, previous['new_score'][k], row)
            extra = old['novel'][np.isfinite(old['novel']).all(1)]
            raw = base['raw'][k]
            geometry = np.concatenate((np.c_[raw, np.zeros(len(raw))], extra))
            declared = mask_owners(support_matrix(geometry,seeds,len(raw)), own, len(names))
            joint = occupied_owners(output,seeds,own,len(names))
            solo = oriented(row, old, raw, d['reference_only'][k], lambda _: reloaded_points(row,base,k))
            ref_solo = occupied_owners(solo,seeds,own,len(names))
            for e, (role,sign) in enumerate(row_edges(earlier,k)[1], 1):
                kind, ref, verified = known_reference(row,earlier,k,ordinal,target,e,role,sign,names,gt)
                datum = dict(**identity, edge=e, prior_truth=kind, visual=int(verified>=0))
                allowed = old['bank'][role] >= 0
                pool = old['bank'][role,allowed]
                datum['pool_count'] = len(pool)
                for scope, actual in (('prior',ref),('visual',verified)):
                    for j, (arm,u) in enumerate(zip(('old','new','permuted'),(old_u,new_u,bad_u), strict=True)):
                        key = scope+'_'+arm
                        rank,_ = reference_order(u[role,allowed], pool, declared, actual)
                        datum[key+'_rank'] = rank
                        datum[key+'_independent_ref'] = truth_status(ref_solo[j,role],actual) if actual>=0 else MISSING_TRUTH
                        datum[key+'_joint_ref'] = truth_status(joint[j+1,role],actual) if actual>=0 else MISSING_TRUTH
                        datum[key+'_joint_both'] = int(actual>=0 and joint[j+1,0]==0 and joint[j+1,role]==actual)
                        datum[key+'_joint_own_subject'] = int(joint[j+1,role]==0)
                edges.append(datum)
    temp = WORK / 'cache_scores' / tag
    temp.mkdir(parents=True, exist_ok=True)
    write_table(temp/'rows.csv',results)
    write_table(temp/'edges.csv',edges)
    print('R169 cache evaluation',tag,len(results),len(edges),flush=True)
    return dict(n=len(results), links=len(edges), max_sensor_error=maxdiff,
                files={p.name:sha(p) for p in (temp/'rows.csv',temp/'edges.csv')})


def metrics(rows):
    gathered, comparisons = [], []
    for scope, sample in (('all', rows), ('camera_horizontal',[x for x in rows if x['camera_horizontal']=='1'])):
        for group, pooled in group_records(sample).items():
            values = {}
            for arm in CACHE_ARMS:
                arm_groups = []
                for part in pooled:
                    v = np.asarray([float(x[arm+'_iou']) for x in part])
                    arm_groups.append(np.c_[v,v>.25,v>.50,[x[arm+'_subject']=='1' for x in part]])
                values[arm] = arm_groups
                v = 100*np.mean([g.mean(0) for g in arm_groups],axis=0)
                gathered.append(dict(scope=scope,group=group,arm=arm,n=sum(map(len,pooled)),mIoU=v[0],Acc25=v[1],Acc50=v[2],
                       subject_identity=v[3],subject_switches=sum(int(x[arm+'_changed']) for part in pooled for x in part)))
            for arm, control in (('old_left','independent'),('new_left','independent'),('new_left','old_left'),
                                 ('permuted_left','old_left'),('new_left','permuted_left')):
                diffs = [a-b for a,b in zip(values[arm],values[control],strict=True)]
                mean = 100*np.mean([x.mean(0) for x in diffs],axis=0)
                fused = np.concatenate(diffs)
                point = dict(scope=scope,group=group,arm=arm,control=control,n=len(fused))
                for i,key in enumerate(('mIoU','Acc25','Acc50','subject_identity')):
                    point.update({key+'_pp':mean[i], key+'_rescue':int((fused[:,i]>1.e-10).sum()),
                                  key+'_harm':int((fused[:,i]<-1.e-10).sum())})
                comparisons.append(point)
    return gathered, comparisons


def ranks(edges):
    phases = dict(visual=[x for x in edges if x['visual']=='1'], prior=[x for x in edges if
                  x['prior_truth'] in ('annotation_proxy','visual_verified')])
    result = []
    for scope, sample in phases.items():
        for arm in ('old','new','permuted'):
            key = scope+'_'+arm
            gold = [int(x[key+'_rank']) for x in sample if x[key+'_rank']!='']
            result.append(dict(scope=scope, arm=arm, n=len(sample), rankable=len(gold),
                        mrr=float(np.mean(1/np.asarray(gold))) if gold else '',
                        **{k:sum(x[key+'_'+k]==str(RIGHT) for x in sample) for k in
                           ('independent_ref','joint_ref','joint_both','joint_own_subject')}))
    return result


def score_all():
    locked = lock('cache', ('cache169.py',))
    assert read(HERE/'CACHE_PREDICTION_SEAL.json')['design_lock'] == locked
    status('cache_scoring')
    with ProcessPoolExecutor(max_workers=6, mp_context=get_context('spawn')) as p:
        calls = {(s,t):p.submit(score,s,t) for s,t in ORDER}
        ledger = {f'{s}_to_{t}': result.result() for (s,t),result in calls.items()}
    final = defaultdict(list)
    for tag,result in ledger.items():
        for name,digest in result['files'].items():
            path = WORK/'cache_scores'/tag/name
            assert sha(path)==digest
            final[name].extend(table(path))
    rows,edges = final['rows.csv'],final['edges.csv']
    assert len(rows)==N and len(edges)==11378 and sum(x['visual']=='1' for x in edges)==54
    stats, pairs = metrics(rows)
    OUT.mkdir(exist_ok=True)
    files = {}
    for name, contents in (('cache_rows',rows),('cache_edges',edges),('cache_metrics',stats),
                           ('cache_paired',pairs),('cache_reference',ranks(edges))):
        p = OUT/(name+'.csv')
        write_table(p,contents)
        files[p.name] = sha(p)
    atomic_json(HERE/'CACHE_EVALUATION_AUDIT.json',dict(n=N, evaluated_edges=len(edges), direction=ledger,
                 exact_old_replay=True, old_gate_used=False, old_overlap_penalty_used=False,
                 prediction_seal=sha(HERE/'CACHE_PREDICTION_SEAL.json'), files=files, FINAL_access=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('predict','score'))
    {'predict':infer_all,'score':score_all}[p.parse_args().phase]()
