"""Open original labels after prediction seal, audit Bayes identity and score both GT conventions."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, softmax
from scipy.spatial.transform import Rotation

from calibrate import HERE, OUT, WORK, ROOT, csv_file, digest, read, save
from factor import ARMS
from predict import inputs

sys.path.insert(0,str(HERE.parent/'3eed_v1_component_ablation_20261119'))
from score_ablation import normalize, sensor_truth, score_boxes, rotation
from obb_iou import oriented_iou


def original_gt(row):
    items = read(row['metapath'])['ground_info']
    if 'object_index' in row:
        item = items[row['object_index']]
        assert normalize(item['caption']) == normalize(row['caption'])
        raw = item['bbox_3d']
    else:
        matches = [i['bbox_3d'] for i in items if normalize(i.get('caption','')) == normalize(row['caption'])]
        if not matches or any(i != matches[0] for i in matches):
            raise ValueError('Ambiguous original task annotation')
        raw = matches[0]
    return np.asarray(raw,dtype=np.float64)


def exact_scores(boxes, raw):
    yaw, pitch, roll = raw[6], *(raw[7:9] if len(raw)==9 else (0.,0.))
    r = Rotation.from_euler('ZYX',[yaw,pitch,roll]).as_matrix()
    gt_radius = np.abs(r) @ raw[3:6]/2
    result = []
    for b in boxes:
        mat = rotation(b[6])
        if np.any(np.abs(raw[:3]-b[:3]) > gt_radius + np.abs(mat)@b[3:6]/2 + 1e-10):
            result.append(0.)
        else:
            result.append(oriented_iou(raw[:3],raw[3:6],r,b[:3],b[3:6],mat))
    return np.array(result)


def check_factors(z, slopes):
    for n in range(len(z['choices'])):
        known = z['valid'][n]
        q = softmax(z['base'][n])
        p = np.exp(z['logits'][n])
        np.testing.assert_allclose(p.sum(1),1.,rtol=0,atol=1e-12)
        np.testing.assert_allclose(p[:,~known],np.broadcast_to(q[~known],(len(ARMS),(~known).sum())),rtol=1e-12)
        np.testing.assert_array_equal(z['choices'][n],p.argmax(1))
        if known.sum() < 2:
            np.testing.assert_allclose(p,np.broadcast_to(q,p.shape),rtol=1e-12)
            continue
        old = q[known] / q[known].sum()
        flipped = softmax(z['other'][n,known])
        implicit = old / (old + flipped)
        sensor = expit(z['sign'][n]*slopes*z['coordinate'][n,known])
        ratio = old * sensor / implicit
        ratio *= q[known].sum() / ratio.sum()
        np.testing.assert_allclose(ratio,p[ARMS.index('replace'),known],rtol=1e-12,atol=1e-14)
        a = (old+flipped)/2
        np.testing.assert_allclose(a*implicit/np.dot(a,implicit),old,rtol=1e-12)


def measure(mat):
    return np.stack((100*(mat>.25).mean(0),100*(mat>.50).mean(0),100*mat.mean(0)),axis=-1)


def paired(mat, picks, current, before):
    a,b = mat[:,current],mat[:,before]
    delta = a-b
    return dict(switches=int((picks[:,current]!=picks[:,before]).sum()),
                better=int((delta>1e-10).sum()),worse=int((delta< -1e-10).sum()),
                rescue25=int(((a>.25)&(b<=.25)).sum()),harm25=int(((a<=.25)&(b>.25)).sum()),
                rescue50=int(((a>.5)&(b<=.5)).sum()),harm50=int(((a<=.5)&(b>.5)).sum()))


def record_intervals(groups, mats, selected, before):
    targets = defaultdict(dict)
    for (s,t), data in groups.items():
        # Both sources of a shared target receive the same resampled recordings.
        rec = data['recording']
        diffs = (mats[s,t][:,selected]>.5).astype(float)-(mats[s,t][:,before]>.5)
        labels = sorted(set(rec))
        targets[t][s] = (labels,np.array([np.sum(diffs[rec==k]) for k in labels]),
                          np.array([np.sum(rec==k) for k in labels]))
    rng = np.random.default_rng(68271)
    scores = []
    for sources in targets.values():
        first = next(iter(sources.values()))[0]
        draw = rng.integers(0,len(first),size=(2000,len(first)))
        for labels,delta,counts in sources.values():
            assert labels==first
            scores.append(delta[draw].sum(1)/counts[draw].sum(1))
    return list(map(float,np.percentile(100*np.mean(scores,axis=0),[2.5,97.5])))


def run(stage):
    root = WORK/stage
    seal = read(root/'SEALED.json')
    assert not seal['task_labels_opened'] and not seal['FINAL_access']
    assert seal['lock'] == digest(root/'LOCK.json')
    cfg = read(root/'LOCK.json')
    assert cfg['arms'] == list(ARMS) and all(cfg['code'][k]==digest(HERE/k) for k in cfg['code'])
    slopes = read(OUT/'calibration.json')['slopes']
    groups = {}
    detailed = []
    coverage = []
    audit = dict(sha256=digest(root/'SEALED.json'),evaluator=digest(Path(__file__)),
                 full_pose_protocol=digest(HERE/'EVALUATION_NOTE.md'),n=0,nonzero_tilt=0,
                 independent_upright_max_error=0.,max_prior_replay_error=0.,FINAL=False)
    for (s,t), rows in inputs(stage).items():
        direction = f'{s}_to_{t}'
        gt7,gt_full,choices = [],[],[]
        recordings=[]
        counters=defaultdict(int)
        for relpath,entry in seal['files'].items():
            if not relpath.startswith(direction+'/'):
                continue
            path=root/relpath
            assert digest(path)==entry['sha256']
            start=int(path.stem)
            with np.load(path,allow_pickle=False) as data:
                z={k:data[k] for k in data.files}
            check_factors(z,slopes[s])
            audit['max_prior_replay_error'] = max(audit['max_prior_replay_error'],float(z['prior_error'].max()))
            for j,picks in enumerate(z['choices']):
                row = rows[start+j]
                rec = row.get('recording') or Path(row['metapath']).parent.parent.name
                recordings.append(rec)
                gt = sensor_truth(t,row['metapath'],row['caption'],row['suffix'])
                raw = original_gt(row)
                assert len(raw)==(7 if t=='waymo' else 9)
                ious,error=score_boxes(z['boxes'][j],gt)
                physical=exact_scores(z['boxes'][j],raw)
                tilt = bool(len(raw)==9 and np.any(np.abs(raw[7:])>1e-8))
                audit['nonzero_tilt']+=int(tilt)
                audit['independent_upright_max_error']=max(audit['independent_upright_max_error'],error)
                gt7.append(ious[picks])
                gt_full.append(physical[picks])
                choices.append(picks)
                valid = z['valid'][j]
                sign = z['sign'][j]
                counters['word']+=int(sign!=0)
                counters['comparable']+=int(valid.sum()>=2)
                counters['four_opportunity50']+=int(ious[picks[0]]<=.5<ious.max())
                if valid.sum()>=2:
                    coords = z['coordinate'][j,valid]
                    odds = z['base'][j,valid] - z['other'][j,valid]
                    for a in range(len(coords)):
                        for b in range(a):
                            if coords[a]!=coords[b] and odds[a]!=odds[b]:
                                counters['word_pairs']+=1
                                counters['word_agrees_with_camera']+=int(sign*(coords[a]-coords[b])*(odds[a]-odds[b])>0)
                row_key=hashlib.sha256(f'{stage}|{direction}|{start+j}'.encode()).hexdigest()[:16]
                key=hashlib.sha256(f'{t}|{rec}'.encode()).hexdigest()[:12]
                public=dict(row=row_key,direction=direction,recording=key,parsed=int(bool(sign)),comparable=int(valid.sum()),
                            has_good50=int(ious.max()>.50),original_gt_tilt=int(tilt))
                for k,pick in enumerate(picks):
                    public.update({f'{ARMS[k]}_switched':int(pick!=picks[0]),
                                   f'{ARMS[k]}_iou':float(ious[pick]),f'{ARMS[k]}_full_pose_iou':float(physical[pick])})
                detailed.append(public)
        assert len(rows)==len(gt7)
        groups[s,t]=dict(historical=np.asarray(gt7),full_pose=np.asarray(gt_full),picks=np.asarray(choices),
                         recording=np.asarray(recordings))
        coverage.append(dict(direction=direction,n=len(rows),recordings=len(set(recordings)),
                             **{k:counters[k] for k in ('word','comparable','four_opportunity50','word_pairs','word_agrees_with_camera')}))
        audit['n']+=len(rows)
    assert audit['n']==seal['n']
    assert sum(c['comparable'] for c in coverage)==seal['comparable']
    pairs=[(a,'v1') for a in ARMS[1:]] + [('replace',a) for a in ('add','balanced_only','reverse_camera','wrong_slot','camera_only')]
    results, changes, intervals = [],[],[]
    macro = {}
    for name in ('historical','full_pose'):
        mats={k:g[name] for k,g in groups.items()}
        metrics={f'{s}_to_{t}':measure(x) for (s,t),x in mats.items()}
        metrics['macro']=np.mean(list(metrics.values()),axis=0)
        macro[name]=metrics['macro']
        for direction, matrix in metrics.items():
            if direction=='macro':
                mat=np.concatenate(list(mats.values()))
                picked=np.concatenate([x['picks'] for x in groups.values()])
            else:
                key=tuple(direction.split('_to_'))
                mat,picked=mats[key],groups[key]['picks']
            for k,arm in enumerate(ARMS):
                acc25,acc50,iou=matrix[k]
                results.append(dict(scorer=name,direction=direction,arm=arm,n=len(mat),Acc25=acc25,Acc50=acc50,mIoU=iou))
            for a,b in pairs:
                i,j=ARMS.index(a),ARMS.index(b)
                delta=matrix[i]-matrix[j]
                changes.append(dict(scorer=name,direction=direction,arm=a,baseline=b,Acc25_pp=delta[0],
                                    Acc50_pp=delta[1],mIoU_pp=delta[2],**paired(mat,picked,i,j)))
                if direction=='macro' and a=='replace':
                    lo,hi=record_intervals(groups,mats,i,j)
                    intervals.append(dict(scorer=name,arm=a,baseline=b,lo=lo,hi=hi))
    csv_file(OUT/f'{stage}_metrics.csv',results)
    csv_file(OUT/f'{stage}_pairs.csv',changes)
    csv_file(OUT/f'{stage}_coverage.csv',coverage)
    csv_file(OUT/f'{stage}_recording_intervals.csv',intervals)
    with gzip.open(OUT/f'{stage}_rows.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=detailed[0].keys())
        writer.writeheader()
        writer.writerows(detailed)
    save(OUT/f'{stage}_audit.json',audit)
    if stage=='pilot':
        selected=next(r for r in changes if r['direction']=='macro' and r['scorer']=='historical'
                      and r['arm']=='replace' and r['baseline']=='v1')
        main=macro['historical'][:,1]
        expand=bool(all(main[2]>main[ARMS.index(k)] for k in ('v1','add','wrong_slot'))
                    and selected['rescue50']>selected['harm50'])
        save(OUT/'pilot_decision.json',dict(expand=expand,sealed=digest(root/'SEALED.json'),primary=selected))
    print(stage,[(ARMS[k],m.tolist()) for k,m in enumerate(macro['historical'])],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('source','pilot','ordinary'))
    run(parser.parse_args().stage)
