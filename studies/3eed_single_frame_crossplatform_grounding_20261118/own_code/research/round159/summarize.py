"""Report the single-frame A protocol using the previously scored and audited R158 predictions."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'round158'
ROOT = HERE.parents[1]
OUTPUT = HERE / 'results'
PRIMARY = {'old_axis': 'old_axis', 'old_pca': 'old_top1', 'r158_pca': 'overlap'}
SECONDARY = {'old_axis': 'frozen_A_axis', 'old_pca': 'frozen_A', 'r158_pca': 'overlap_A_only'}
PAIRED = {'A_old_pca': 'frozen_A', 'A_r158_pca': 'overlap_A_only', 'AB_old_pca': 'joint_zero',
          'AB_r158_pca': 'overlap_reader'}
METRICS = ('mIoU', 'Acc25', 'Acc50')
PINS = {'working_results': '35376a07909394705ef49605e47b819c9341789a41a3e0175ccce158d572e26e',
        'transfer_results': '3b1ab52461679f023730e5b9c737ab6a86b812b07d43d1cd25af7e2880495847'}


def readj(path):
    return json.loads(path.read_text())


def load_rows(family, n):
    path = PARENT / family / 'all_rows.csv'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[family] or readj(path.parent / 'AUDIT.json')['status'] != 'PASS':
        raise ValueError('The registered R158 per-record scores or audit changed')
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != n or len({row['ordinal'] for row in rows}) != n:
        raise ValueError('Unexpected cohort or duplicate direction-record')
    return rows


def csv_file(name, rows):
    with (OUTPUT/name).open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows):
    by_direction = defaultdict(list)
    for row in rows:
        by_direction[(row['source'], row['target'])].append(row)
    if len(by_direction) != 6:
        raise ValueError('All six source-to-target directions must be present')
    return sorted(by_direction.items())


def summarize(name, rows, arms, comparisons):
    directions, macro, pairs = [], [], []
    grouped = group_rows(rows)
    for (src, tgt), subset in grouped:
        weights = [float(r.get('sampling_weight', 1)) for r in subset]
        scores = {arm: np.array([float(r[column]) for r in subset]) for arm, column in arms.items()}
        meta = dict(cohort=name, source=src, target=tgt, n=len(subset))
        for arm, y in scores.items():
            if not np.isfinite(y).all() or np.min(y)<0 or np.max(y)>1:
                raise ValueError('Invalid per-record IoU')
            values = [100*np.average(v, weights=weights) for v in (y, y>.25, y>.5)]
            directions.append(dict(**meta, arm=arm, **dict(zip(METRICS, values))))
        for prior, new in comparisons:
            d = scores[new]-scores[prior]
            improved, worsened = d>1e-12, d< -1e-12
            pairs.append(dict(**meta, before=prior, after=new, improved=int(improved.sum()),
                              worsened=int(worsened.sum()), tied=int((~improved & ~worsened).sum()),
                              positive_pp=100*np.average(d.clip(min=0),weights=weights),
                              negative_pp=100*np.average(d.clip(max=0),weights=weights),
                              net_pp=100*np.average(d, weights=weights)))
    for arm in arms:
        items = [r for r in directions if r['arm']==arm]
        macro.append(dict(cohort=name, arm=arm, n=len(rows),
                          **{k:statistics.mean(r[k] for r in items) for k in METRICS}))
    for prior, new in comparisons:
        items = [r for r in pairs if (r['before'],r['after'])==(prior,new) and r['source']!='all']
        pairs.append(dict(cohort=name,source='all',target='six_equal',n=len(rows),before=prior,after=new,
                          **{k:sum(r[k] for r in items) for k in ('improved','worsened','tied')},
                          **{k:statistics.mean(r[k] for r in items) for k in ('positive_pp','negative_pp','net_pp')}))
    return directions, macro, pairs


def main():
    OUTPUT.mkdir(exist_ok=True)
    main_rows = load_rows('transfer_results', 5942)
    dev = load_rows('working_results',1798)
    paired = [r for r in dev if r['has_B']=='True']
    if len(paired) != 764:
        raise ValueError('B coverage changed')
    base_pairs = [('old_axis','old_pca'),('old_pca','r158_pca')]
    cohorts = [summarize('ordinary_A_only', main_rows, PRIMARY, base_pairs),
               summarize('development_A_only', dev, SECONDARY, base_pairs),
               summarize('registered_pairs_only', paired, PAIRED,
                         [('A_r158_pca','AB_r158_pca'),('AB_old_pca','AB_r158_pca')])]
    for index, name in enumerate(('directions.csv','overall.csv','paired_changes.csv')):
        csv_file(name, [row for cohort in cohorts for row in cohort[index]])

    # Check how many target scenes are new relative to the R158 working set.
    inputs = readj(ROOT/'crossplatform/round158/EXTRA_REGISTRY_PRIVATE.json')
    earlier = readj(ROOT/'research/round145/INPUT_REGISTRY_PRIVATE.json')
    identity = lambda e: (e['target'],e['row']['scene'])
    scenes = {identity(e) for e in inputs}
    chosen_on = {identity(e) for e in earlier}
    content_groups = {row['group'] for row in inputs}
    # Rows across the two source models can describe the same physical instance.
    expressions = {(e['target'], e['row']['metapath'], e['row']['object_index'], e['row']['anno_index']) for e in inputs}
    summary = dict(ordinary_direction_records=len(inputs), physical_content_groups=len(content_groups),
                   target_expression_entries=len(expressions), target_scenes=len(scenes),
                   scenes_absent_from_R158_selection=len(scenes-chosen_on),
                   records_in_absent_scenes=sum(identity(e) not in chosen_on for e in inputs),
                   rows_with_B=len(paired), rows_without_B=len(dev)-len(paired), historical_all_rows_sha256=PINS,
                   thresholds='IoU > .25 and IoU > .50', ordinary_weights='equal per record within direction',
                   development_weights='same R158 sampling weights normalized within each direction and subset',
                   direction_weights='each of six directions has weight 1/6', new_target_GT_scoring=False, FINAL_access=False)
    (OUTPUT/'scope.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__=='__main__':
    main()
