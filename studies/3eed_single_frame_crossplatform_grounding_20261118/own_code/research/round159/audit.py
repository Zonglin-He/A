"""Independent CPU check against the sealed, already scored scalar rows (no label loader)."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
PARENT = HERE.parent / 'round158'


def csv_rows(p):
    with p.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    reference = {s:{} for s in ('ordinary_A_only','development_A_only','registered_pairs_only')}
    tables = {}
    ordinary = csv_rows(PARENT/'transfer_results/all_rows.csv')
    dev = csv_rows(PARENT/'working_results/all_rows.csv')
    cohorts = {'ordinary_A_only':(ordinary,('old_axis','old_top1','overlap')),
               'development_A_only':(dev,('frozen_A_axis','frozen_A','overlap_A_only')),
               'registered_pairs_only':([r for r in dev if r['has_B']=='True'],
                                        ('frozen_A','overlap_A_only','joint_zero','overlap_reader'))}
    for cohort,(rows,columns) in cohorts.items():
        names = ('A_old_pca','A_r158_pca','AB_old_pca','AB_r158_pca') if len(columns)==4 else ('old_axis','old_pca','r158_pca')
        for name,col in zip(names,columns):
            grouped = defaultdict(list)
            for r in rows:
                w = float(r.get('sampling_weight',1))
                grouped[(r['source'],r['target'])].append((float(r[col]),w))
            tables[cohort,name] = grouped
            if len(grouped)!=6:
                raise ValueError('missing direction')
            for direction,values in grouped.items():
                total = math.fsum(w for _,w in values)
                scores = [100*math.fsum(w*fn(y) for y,w in values)/total
                          for fn in (float, lambda y:y>.25, lambda y:y>.5)]
                reference[cohort][(name,*direction)] = scores
            reference[cohort][(name,'all','six_equal')] = [math.fsum(reference[cohort][(name,*d)][m] for d in grouped)/6 for m in range(3)]
    errs = []
    for file in ('directions.csv','overall.csv'):
        for r in csv_rows(RESULTS/file):
            expected = reference[r['cohort']][r['arm'],r.get('source','all'),r.get('target','six_equal')]
            errs.extend(abs(float(r[k])-n) for k,n in zip(('mIoU','Acc25','Acc50'),expected))
    for r in csv_rows(RESULTS/'paired_changes.csv'):
        prior,after = (tables[r['cohort'],r[k]] for k in ('before','after'))
        keys = list(prior) if r['source']=='all' else [(r['source'],r['target'])]
        diff = []
        for key in keys:
            a,b = prior[key],after[key]
            assert len(a)==len(b)
            for (old,w),(new,v) in zip(a,b):
                assert w==v
                diff.append(new-old)
        assert sum(1 for d in diff if d>1e-12)==int(r['improved'])
        assert sum(1 for d in diff if d< -1e-12)==int(r['worsened'])
        assert sum(1 for d in diff if abs(d)<=1e-12)==int(r['tied'])
        key=(r['source'],r['target'])
        net=reference[r['cohort']][(r['after'],*key)][0]-reference[r['cohort']][(r['before'],*key)][0]
        errs.extend([abs(net-float(r['net_pp'])),abs(net-float(r['positive_pp'])-float(r['negative_pp']))])
    if max(errs)>1e-10:
        raise ValueError('metric mismatch')
    with np.load(HERE.parents[1]/'crossplatform/round158/EXTRA_PRIVATE.npz',allow_pickle=False) as sealed:
        boxes=sealed['boxes']
        assert boxes.shape==(len(ordinary),2,7) and np.isfinite(boxes).all() and (boxes[:,:,3:6]>0).all()
    raw=json.loads((HERE/'RAW_REPLAY.json').read_text())
    cli=json.loads((HERE/'CLI_REPLAY.json').read_text())
    assert raw['status']==cli['status']=='PASS' and raw['n']==14 and raw['default_seed_reproducible']
    audit=dict(status='PASS',scalar_comparisons=len(errs),max_error_percentage_points=max(errs),
               baseline_and_selected_boxes_with_positive_dimensions=2*len(ordinary),
               engineering_direction_camera_cases=raw['n'],public_cli=cli['status'],new_target_GT_access=False,FINAL_access=False)
    (HERE/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':
    main()
