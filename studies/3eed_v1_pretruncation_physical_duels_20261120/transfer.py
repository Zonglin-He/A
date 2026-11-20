"""Seal decisions from current-input features first, then evaluate all fixed geometries."""
import csv
from time import perf_counter
from common import ARMS, HERE, PREV, RESULTS, WORK, anon, archive_npz, atomic, digest, legacy, np, read, setup, torch
from fit_duels import all_scores, loaded, versions
from fit import write_csv
from score import annotated, ci_recording, compressed_csv, metrics, score_boxes, transitions
from scope183 import ordinary


def predict():
    lock = read(WORK/'DECODER_LOCK.json')
    assert lock['head_lock'] == digest(WORK/'HEADS_LOCK.json')
    assert read(WORK/'HEADS_LOCK.json')['heads'] == digest(WORK/'heads.npz')
    features = WORK/'target_features'
    receipt = read(features/'SEALED.json')
    assert receipt['lock'] == digest(features/'LOCK.json') and not receipt['target_labels_read']
    old = PREV/'run_PRIVATE/target'
    oldpins = read(old/'SEALED.json')
    heads = torch.from_numpy(loaded(WORK/'heads.npz')['heads'])
    picks, frames, targets, ordinals = [], [], [], []
    for t, files in receipt['files'].items():
        for name,item in sorted(files.items()):
            path = features/t/name
            assert digest(path) == item['sha256']
            previous = old/t/name
            assert digest(previous) == oldpins['files'][t][name]['sha256'] == item['previous']
            features_block, history = loaded(path),loaded(previous)
            seq = features_block['ordinals']
            np.testing.assert_array_equal(seq,history['ordinals'])
            all_j = versions(all_scores(features_block['h'],features_block['j'],heads[0],heads),lock['scales'])
            for k, arm in enumerate(('v1','old_mean','old_worst')):
                np.testing.assert_allclose(all_j[arm],history['scores'][:,k],atol=3e-6,rtol=0)
                np.testing.assert_array_equal(all_j[arm].argmax(-1),history['choices'][:,k])
            picks.append(np.stack([s.argmax(-1) for s in all_j.values()],axis=1).astype(np.uint8))
            frames.append(history['four'])
            targets.extend([t]*len(seq))
            ordinals.extend(seq)
    counts = {t:targets.count(t) for t in set(targets)}
    assert counts == dict(drone=2792,quad=2736)
    out = WORK/'predictions.npz'
    archive_npz(out, arms=np.array(list(all_j)),targets=np.array(targets), ordinals=np.array(ordinals),
                picks=np.concatenate(picks),four=np.concatenate(frames))
    atomic(WORK/'PREDICTIONS_SEAL.json',dict(file=digest(out),source=digest(WORK/'DECODER_LOCK.json'),
        features=digest(features/'SEALED.json'),code=digest(HERE/'transfer.py'),
        rows=counts,target_annotations_read=False,final_access=False))


def study_controls(a):
    controls = ['v1']
    if a in ('assignment','worst','broken_pair'):
        controls.append('independent')
        if a == 'assignment':
            controls.extend(['worst','broken_pair'])
    if a in ('assignment_matched','worst_matched','broken_pair_matched'):
        controls.append('independent_matched')
        if a == 'assignment_matched':
            controls.extend(['worst_matched','broken_pair_matched'])
    if a == 'old_worst_matched':
        controls.append('old_mean_matched')
    return controls


def evaluate_sealed(work, results, controls_for, intervals_for):
    seal = read(work/'PREDICTIONS_SEAL.json')
    assert seal['file'] == digest(work/'predictions.npz') and not seal['target_annotations_read']
    arr = loaded(work/'predictions.npz')
    roster = ordinary()
    arms = arr['arms'].tolist()
    data = {t:[] for t in ('drone','quad')}
    old = legacy.ORDINARY/'audited_iou_PRIVATE'
    oldpins = read(old/'SEALED.json')['files']
    old_preds = {}
    for t in data:
        path = old/f'waymo_to_{t}.npz'
        assert digest(path) == oldpins[path.name]['sha256']
        old_preds[t] = loaded(path)
    errors = [0.,0.]
    for t,idx,picks,four in zip(arr['targets'],arr['ordinals'],arr['picks'],arr['four'],strict=True):
        row = roster[t][idx]
        assert len(data[t]) == idx
        ious,error = score_boxes(four,annotated(row))
        vals = ious[picks]
        parity = max(abs(vals[0]-old_preds[t]['ious'][idx,0]),
                     abs(ious.max()-old_preds[t]['four_best'][idx]))
        errors = [max(errors[0],error),max(errors[1],parity)]
        assert parity < 1e-10 and error < 1e-10
        data[t].append(dict(values=vals,picks=picks,rec=row['recording'],
                            sort=anon(t,row['seed'],row['object_index']),all_ious=ious))
    scopes = {f'waymo_to_{t}':items for t,items in data.items()}
    rec_ids = {}
    for t,items in data.items():
        for n,rec in enumerate(sorted({r['rec'] for r in items},key=lambda v:anon('rec',t,v)),1):
            name=f'{t}_rec_{n}'
            scopes[name] = [r for r in items if r['rec']==rec]
            rec_ids[t,rec] = name
    scopes['two_equal'] = data['drone']+data['quad']
    directions, paired, individual = [],[],[]
    all_stats = {}
    for name,items in scopes.items():
        v,p = [np.stack([r[k] for r in items]) for k in ('values','picks')]
        stats = {}
        for i,a in enumerate(arms):
            stat = (np.mean([all_stats['waymo_to_'+t][a] for t in data],axis=0)
                    if name=='two_equal' else metrics(v[:,i]))
            stats[a] = stat
            directions.append(dict(scope=name,arm=a,n=len(v),Acc25=stat[0],Acc50=stat[1],mIoU=stat[2]))
        all_stats[name] = stats
        for i,a in enumerate(arms):
            if i == 0:
                continue
            for b in controls_for(a):
                j = arms.index(b)
                delta = stats[a]-stats[b]
                paired.append(dict(scope=name,arm=a,baseline=b,n=len(v),
                    switches=int((p[:,i]!=p[:,j]).sum()),Acc25_pp=delta[0],Acc50_pp=delta[1],
                    mIoU_pp=delta[2],**transitions(v[:,i],v[:,j])))
    for i,(t,r) in enumerate(sorted(((t,r) for t,rows in data.items() for r in rows),key=lambda x:x[1]['sort'])):
        record = dict(row=f'T{i+1:04d}',target=t,recording_group=rec_ids[t,r['rec']])
        for j,a in enumerate(arms):
            record[a+'_iou'] = r['values'][j]
            record[a+'_switch'] = int(r['picks'][j]!=r['picks'][0])
            # Compare contender identity across arms without releasing private query indices.
            record[a+'_choice'] = int(list(dict.fromkeys(r['picks'].tolist())).index(r['picks'][j]))
        individual.append(record)
    write_csv(results/'directions.csv',directions)
    write_csv(results/'paired_changes.csv',paired)
    compressed_csv(results/'all_target_rows.csv.gz',individual)
    changes=[r for r in individual if any(r[a+'_switch'] for a in arms)]
    write_csv(results/'changed_rows.csv',changes)
    intervals = {a:ci_recording(data,arms.index(a),0) for a in intervals_for}
    atomic(results/'score_audit.json',dict(target_rows=len(individual),all_switch_rows=len(changes),
        max_iou_implementation_error=errors[0],max_baseline_parity_error=errors[1],
        cluster_intervals_delta_Acc50_vs_v1=intervals,seal=digest(work/'PREDICTIONS_SEAL.json'),
        exposed_ordinary=True,FINAL_access=False))


if __name__=='__main__':
    setup()
    start = perf_counter()
    predict()
    print('Predictions sealed before scoring.',flush=True)
    evaluate_sealed(WORK,RESULTS,study_controls,('assignment','independent','worst'))
    print('Scored in',round(perf_counter()-start,1),'seconds',flush=True)
