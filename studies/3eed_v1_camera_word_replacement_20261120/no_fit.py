"""Isolate the value of source GT calibration on the same sealed scores and boxes."""
import argparse
import csv
import gzip
import hashlib

import numpy as np
from scipy.special import softmax

from calibrate import HERE, OUT, WORK, csv_file, digest, read, save
from factor import ARMS

NAMES = ('v1', 'add_no_fit', 'replace_no_fit', 'add_source_fit', 'replace_source_fit')
COMPARISONS = ((1,0), (2,0), (2,1), (3,1), (4,2))


def select(factual, flipped, horizontal, use, sign):
    """horizontal is normalized to [-1,1]. Unknown slots keep their original mass."""
    initial = int(factual.argmax())
    if np.count_nonzero(use) < 2 or not sign:
        return initial, initial
    sensor = (1. + sign * horizontal[use]) / 2.
    assert np.all((sensor >= 0.) & (sensor <= 1.))
    if sensor.sum() == 0:
        return initial, initial
    total = softmax(factual)
    local = softmax(factual[use])
    balanced = (local + softmax(flipped[use])) / 2.
    winners = []
    for prior in (local, balanced):
        result = total.copy()
        posterior = prior * sensor
        result[use] = posterior * (total[use].sum()/posterior.sum())
        winners.append(int(result.argmax()))
    return tuple(winners)


def seal(stage):
    root = WORK/stage
    out = WORK/(stage+'_no_fit')
    out.mkdir(exist_ok=True)
    original = read(root/'SEALED.json')
    protocol = HERE/'NO_FIT_PROTOCOL.json'
    assert read(protocol)['arms'] == list(NAMES)
    votes = {}
    for name, receipt in original['files'].items():
        path = root/name
        assert digest(path) == receipt['sha256']
        with np.load(path,allow_pickle=False) as batch:
            frozen = batch['choices'][:,ARMS.index('v1')]
            supervised = batch['choices'][:,[ARMS.index('add'),ARMS.index('replace')]]
            no_fit = [select(*row) for row in zip(batch['base'],batch['other'],batch['coordinate'],
                                                  batch['valid'],batch['sign'])]
            assert np.array_equal(frozen, batch['base'].argmax(1))
        target = out/(name+'.npy')
        target.parent.mkdir(exist_ok=True)
        matrix = np.column_stack((frozen,no_fit,supervised)).astype(np.int8)
        np.save(target,matrix,allow_pickle=False)
        votes[name] = digest(target)
    stamp = dict(stage=stage, names=NAMES, n=original['n'], source=digest(root/'SEALED.json'),
                 protocol=digest(protocol), code=digest(__file__), choices=votes)
    save(out/'SEALED.json',stamp)


def score(stage):
    from assess import exact_scores, measure, original_gt, paired, score_boxes, sensor_truth
    from predict import inputs
    root = WORK/stage
    out = WORK/(stage+'_no_fit')
    receipt = read(out/'SEALED.json')
    assert receipt['source']==digest(root/'SEALED.json') and receipt['code']==digest(__file__)
    assert receipt['protocol']==digest(HERE/'NO_FIT_PROTOCOL.json')
    groups = {}
    details = []
    for (src,tgt), rows in inputs(stage).items():
        direction = f'{src}_to_{tgt}'
        by_metric = {'historical':[], 'full_pose':[]}
        selected = []
        for name,file_hash in receipt['choices'].items():
            if name.split('/')[0] != direction:
                continue
            path = out/(name+'.npy')
            assert digest(path)==file_hash
            picks = np.load(path,allow_pickle=False)
            offset = int(name.split('/')[-1].split('.')[0])
            with np.load(root/name,allow_pickle=False) as z:
                boxes = z['boxes']
            for j, choices in enumerate(picks):
                row = rows[offset+j]
                truth = sensor_truth(tgt,row['metapath'],row['caption'],row['suffix'])
                upright, _ = score_boxes(boxes[j],truth)
                full = exact_scores(boxes[j],original_gt(row))
                selected.append(choices)
                public = dict(row=hashlib.sha256(f'{stage}|{direction}|{offset+j}'.encode()).hexdigest()[:16],
                              direction=direction)
                for metric, values in (('historical',upright),('full_pose',full)):
                    by_metric[metric].append(values[choices])
                    for a, v in zip(NAMES,values[choices]):
                        public[f'{a}_{metric}_iou'] = float(v)
                details.append(public)
        assert len(selected)==len(rows)
        groups[direction] = dict(picks=np.asarray(selected),**{k:np.asarray(v) for k,v in by_metric.items()})
    assert len(details)==receipt['n']
    metrics, changes = [],[]
    for metric in ('historical','full_pose'):
        scores={k:measure(g[metric]) for k,g in groups.items()}
        scores['macro']=np.mean(list(scores.values()),axis=0)
        for direction, table in scores.items():
            group = groups.get(direction)
            values = group[metric] if group else np.concatenate([g[metric] for g in groups.values()])
            picks = group['picks'] if group else np.concatenate([g['picks'] for g in groups.values()])
            for a,(acc25,acc50,miou) in zip(NAMES,table):
                metrics.append(dict(scorer=metric,direction=direction,arm=a,n=len(values),
                                    Acc25=acc25,Acc50=acc50,mIoU=miou))
            for a,b in COMPARISONS:
                delta=table[a]-table[b]
                changes.append(dict(scorer=metric,direction=direction,arm=NAMES[a],baseline=NAMES[b],
                                    Acc25_pp=delta[0],Acc50_pp=delta[1],mIoU_pp=delta[2],
                                    **paired(values,picks,a,b)))
    csv_file(OUT/f'{stage}_no_fit_metrics.csv',metrics)
    csv_file(OUT/f'{stage}_no_fit_pairs.csv',changes)
    with gzip.open(OUT/f'{stage}_no_fit_rows.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=details[0])
        writer.writeheader()
        writer.writerows(details)
    save(OUT/f'{stage}_no_fit_audit.json',dict(n=len(details),seal=digest(out/'SEALED.json'),
         parent=receipt['source'],protocol=receipt['protocol'],code=receipt['code']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=('seal','score'))
    parser.add_argument('stage',choices=('pilot','ordinary'))
    args=parser.parse_args()
    (seal if args.operation=='seal' else score)(args.stage)
