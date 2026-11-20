"""Audit availability of supervision and independently recount all published choices."""
import argparse
import csv
import gzip
import sys
from pathlib import Path

import numpy as np

from calibrate import HERE, OUT, PLATFORMS, SOURCE, SUP, WORK, csv_file, digest, read, save, source_rows


def inspect_scope():
    sys.path.insert(0,str(SUP))
    from scope import DATA, evaluation_rows
    manifest = read(SOURCE/'SEAL.json')
    pilot = evaluation_rows()
    counts = []
    for platform in PLATFORMS:
        official = DATA/'splits'/f'{platform}_train.txt'
        assert digest(official)==manifest['split_hashes'][platform]
        allowed = set(official.read_text().splitlines())
        fit = source_rows(platform,'fit')
        hold = source_rows(platform,'calibration')
        test = [r for r in pilot if r['platform']==platform]
        def recordings(rows):
            return {Path(r['metapath']).parent.parent.name for r in rows}
        def clouds(rows):
            return {r['point_sha256'] for r in rows}
        for rows in (fit, hold, test):
            assert all(r['platform']==platform for r in rows)
            assert recordings(rows)<=allowed
        assert not recordings(fit)&recordings(hold)
        assert not clouds(fit)&clouds(hold) and not (clouds(fit)|clouds(hold))&clouds(test)
        original_boxes = np.asarray([read(r['metapath'])['ground_info'][r['object_index']]['bbox_3d']
                                     for r in fit+hold],dtype=float)
        extra = original_boxes[:,7:]
        assert original_boxes.shape[1] == (7 if platform=='waymo' else 9)
        counts.append(dict(source=platform,official_train_sha256=digest(official),
                           fit_n=len(fit),holdout_n=len(hold),dev_target_n=len(test),
                           source_box_fields=original_boxes.shape[1],
                           field8_min=float(extra[:,0].min()) if extra.size else None,
                           field8_max=float(extra[:,0].max()) if extra.size else None,
                           field9_distinct_values=','.join(map(str,np.unique(extra[:,1]))) if extra.size else None,
                           fit_recordings=len(recordings(fit)),heldout_recordings=len(recordings(hold)),
                           dev_target_recordings=len(recordings(test)),
                           fit_holdout_shared_recordings=len(recordings(fit)&recordings(hold)),
                           fit_dev_shared_recordings=len(recordings(fit)&recordings(test)),
                           fit_dev_shared_point_hashes=len(clouds(fit)&clouds(test)),
                           holdout_dev_shared_point_hashes=len(clouds(hold)&clouds(test))))
    csv_file(OUT/'training_scope.csv',counts)
    print([(r['source'],r['fit_n'],r['fit_dev_shared_point_hashes']) for r in counts])


def read_csv(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as stream:
        return list(csv.DictReader(stream))


def recount(stage):
    from factor import ARMS
    original = read(WORK/stage/'SEALED.json')
    rows = read_csv(OUT/f'{stage}_rows.csv.gz')
    assert len(rows)==len({r['row'] for r in rows})==original['n']
    paths = [(rows, list(ARMS), '')]
    if stage!='source':
        from no_fit import NAMES
        sup = read_csv(OUT/f'{stage}_no_fit_rows.csv.gz')
        assert len(sup)==len(rows)
        ids = {r['row']:r for r in sup}
        assert len(ids)==len(rows)
        for r in rows:
            s=ids[r['row']]
            assert r['direction']==s['direction']
            for m,n in (('v1','v1'),('add','add_source_fit'),('replace','replace_source_fit')):
                assert r[m+'_iou']==s[n+'_historical_iou']
                assert r[m+'_full_pose_iou']==s[n+'_full_pose_iou']
        paths.append((sup,list(NAMES),'_no_fit'))
    errors=[]
    for detail, arms, suffix in paths:
        directions=sorted({r['direction'] for r in detail})
        computed={}
        counts={}
        for scorer in ('historical','full_pose'):
            key_suffix = ('_'+scorer if suffix else ('_full_pose' if scorer=='full_pose' else ''))+'_iou'
            for arm in arms:
                for d in directions:
                    data=[float(r[arm+key_suffix]) for r in detail if r['direction']==d]
                    assert all(0.<=i<=1.+1.e-8 for i in data)
                    counts[d]=len(data)
                    computed[scorer,d,arm]=np.array([sum(i>.25 for i in data),sum(i>.5 for i in data),sum(data)])*100/len(data)
                computed[scorer,'macro',arm]=np.mean([computed[scorer,d,arm] for d in directions],axis=0)
        for r in read_csv(OUT/f'{stage}{suffix}_metrics.csv'):
            assert int(r['n'])==counts.get(r['direction'],len(detail))
            x = np.array([float(r[k]) for k in ('Acc25','Acc50','mIoU')])
            errors.append(float(abs(x-computed[r['scorer'],r['direction'],r['arm']]).max()))
        for r in read_csv(OUT/f'{stage}{suffix}_pairs.csv'):
            sub = detail if r['direction']=='macro' else [p for p in detail if p['direction']==r['direction']]
            key_suffix= ('_'+r['scorer'] if suffix else ('_full_pose' if r['scorer']=='full_pose' else ''))+'_iou'
            old = np.array([float(p[r['baseline']+key_suffix]) for p in sub])
            new = np.array([float(p[r['arm']+key_suffix]) for p in sub])
            for t,cut in ((25,.25),(50,.50)):
                assert int(r[f'rescue{t}'])==np.count_nonzero((new>cut)&(old<=cut))
                assert int(r[f'harm{t}'])==np.count_nonzero((new<=cut)&(old>cut))
    assert max(errors)<1.e-10
    save(OUT/f'{stage}_recount.json',dict(n=len(rows),unique_anonymous_rows=len(rows),
         all_condition_metrics_and_threshold_pairs_recomputed=True,
         max_difference_percentage_points=max(errors),elapsed_prediction_seconds=original['elapsed'],
         primary_prediction_seal=digest(WORK/stage/'SEALED.json'),FINAL_access=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('scope',choices=('scope','source','pilot','ordinary'))
    stage=p.parse_args().scope
    inspect_scope() if stage=='scope' else recount(stage)
