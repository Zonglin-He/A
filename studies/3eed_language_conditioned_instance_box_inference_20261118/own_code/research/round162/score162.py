"""After seals, score all endpoints, paired changes and oracle diagnostics."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

from manifest162 import (DEV, EXTRA, REC, HERE, ORDER, ROOT, WORK, atomic_json, canonical,
                         prediction_file, read, sha, target_rows)
from run162 import SETTINGS, PROTOCOL
sys.path.insert(0, str(ROOT/'research/round154'))
from decode160 import object_index, object_indices
from score154 import DATA, ground_truth, many_iou

METRICS = ('mIoU', 'Acc25', 'Acc50')
FROZEN = ('official_span', 'prepared_axis', 'prepared_pca', 'r158_full')
HEADLINES = ('official_span', 'prepared_axis', 'prepared_pca', 'r158_full', 'pca_obb',
             'uniform_fixed', 'uniform_fit', 'source_selected_simple', 'fixed_winner_support', 'serial', 'lcib')
PAIRS = (('r158_full','lcib'), ('source_selected_simple','lcib'), ('serial','lcib'),
         ('uniform_fit','fixed_winner_support'), ('r158_full','serial'), ('r158_full','source_selected_simple'))
PANEL_PAIRS = (('lcib','single_fit'), ('lcib','no_feedback'), ('lcib','no_relation'), ('no_rgb_graph','lcib'),
               ('swap_cross','lcib'), ('uniform_cross','lcib'), ('system_no_rgb','lcib'),
               ('system_no_rgb_old','system_no_rgb'), ('wrong_frame_rgb','lcib'),
               ('bounded_origins','expanded_origins'), ('expanded_origins','lcib'),
               ('fixed_dimensions','lcib'), ('visible_surface_only','lcib'), ('missing_as_negative','lcib'))


def put_table(path, rows):
    if not rows:
        raise ValueError(f'Empty table {path.name}')
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix('.tmp')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with tmp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verified(scope):
    seal_path = HERE/(scope+'_SEAL.json')
    seal = read(seal_path)
    index = WORK/(scope+'_FILES_PRIVATE.json')
    assert sha(index) == seal['file_index_sha256'] and sha(PROTOCOL) == seal['protocol_sha256']
    assert sha(SETTINGS) == seal['source_parameters_sha256']
    references = read(index)
    assert len(references) == seal['n']
    for name, digest in references.items():
        if sha(WORK/scope/name) != digest:
            raise ValueError(f'Sealed prediction bytes changed in {scope}')
    return sorted(int(Path(name).stem) for name in references)


def allowed_gt(entry, look):
    r = entry['r']
    root = ROOT/'crossplatform/dataset_full/3eed_dataset'
    meta = Path(r['metapath']).resolve()
    view_path = DATA/meta.relative_to(root)
    assert view_path.samefile(meta)
    return ground_truth(entry['target'], str(view_path), object_index(entry, look), canonical(r['caption']))


def matrix(values):
    a = np.asarray(values, dtype=float)
    return np.stack((a, a > .25, a > .50), axis=1)


def weighted(y, weights):
    return np.average(matrix(y), weights=weights, axis=0) * 100


def threshold_cases(records, first, second):
    d = matrix([r[second] for r in records]) - matrix([r[first] for r in records])
    w = np.asarray([r['weight'] for r in records])
    mean = 100 * np.average(d, weights=w, axis=0)
    return mean, (d > 1e-12).sum(0), (d < -1e-12).sum(0), d


def groups(records):
    out = defaultdict(list)
    for r in records:
        out[r['cohort'], r['source'], r['target']].append(r)
    return out


def summaries(records, arms, pairs, with_scene=True):
    direction = []
    paired = []
    macro = []
    y_by_cohort, p_by_cohort = defaultdict(list), defaultdict(list)
    for (cohort, source, target), group in sorted(groups(records).items()):
        sel = dict(cohort=cohort, source=source, target=target)
        weight = [r['weight'] for r in group]
        factors = {r['recording'] for r in group}
        stamp = dict(**sel, n=len(group), scenes=len(factors))
        for arm in arms:
            value = weighted([r[arm] for r in group], weight)
            row = dict(**stamp, arm=arm, **dict(zip(METRICS, value)))
            y_by_cohort[cohort,arm].append(row)
            direction.append(row)
        for before, after in pairs:
            delta, positive, negative, _ = threshold_cases(group, before, after)
            row = dict(**stamp, before=before, after=after)
            for m, v, p, n in zip(METRICS, delta, positive, negative):
                row.update({m+'_pp':v, m+'_gains':int(p), m+'_losses':int(n)})
            paired.append(row)
            p_by_cohort[cohort,before,after].append(row)
    for (cohort, arm), values in y_by_cohort.items():
        assert len(values) == 6
        macro.append(dict(cohort=cohort, arm=arm, n=sum(r['n'] for r in values),
                          **{m: np.mean([r[m] for r in values]) for m in METRICS}))
    for (cohort, a, b), values in p_by_cohort.items():
        assert len(values)==6
        row = dict(cohort=cohort, source='six_equal', target='six_equal', n=sum(r['n'] for r in values),
                   scenes=sum(len({r['recording'] for r in records if r['cohort']==cohort and r['target']==t})
                              for t in ('waymo','drone','quad')), before=a, after=b)
        for m in METRICS:
            row[m+'_pp'] = np.mean([r[m+'_pp'] for r in values])
            for tail in ('gains','losses'):
                row[m+'_'+tail] = sum(r[m+'_'+tail] for r in values)
        paired.append(row)
    return direction, paired, macro


def oracle_table(records):
    by = groups(records)
    rows = []
    all_keys = sorted({
        k for r in records for k in r
        if k.startswith(('oracle_', 'beyond_', 'generated_', 'old_good_new_failed_', 'new_cover_unselected_'))
    })
    for (cohort,s,t), group in sorted(by.items()):
        weights = np.array([r['weight'] for r in group])
        row = dict(cohort=cohort, source=s, target=t, n=len(group))
        for key in all_keys:
            data = np.asarray([r[key] for r in group])
            if key.startswith('oracle_'):
                for m,v in zip(METRICS,weighted(data,weights)):
                    row[key+'_'+m] = v
            else:
                row[key] = int(data.sum())
                row[key+'_percent'] = 100 * np.average(data,weights=weights)
        rows.append(row)
    for c in sorted({r['cohort'] for r in rows}):
        g = [r for r in rows if r['cohort']==c]
        out = dict(cohort=c,source='six_equal',target='six_equal',n=sum(r['n'] for r in g))
        for k in set(g[0]) - set(out):
            out[k]=(sum(r[k] for r in g) if not k.startswith('oracle_') and not k.endswith('_percent')
                    else np.mean([r[k] for r in g]))
        rows.append(out)
    return rows


def confidence_intervals(records, pairs):
    """Resample scenes jointly for both incoming directions of a target platform."""
    rng = np.random.default_rng(162041)
    n_boot = 2500
    out = []
    by = groups(records)
    for c in sorted({r['cohort'] for r in records}):
        scenes = {}
        for t in ('waymo','drone','quad'):
            ids = sorted({r['recording'] for r in records if r['cohort']==c and r['target']==t})
            samples = rng.multinomial(len(ids), np.full(len(ids),1./len(ids)),size=n_boot)
            scenes[t] = {k:i for i,k in enumerate(ids)}, samples
        for old, new in pairs:
            pooled = []
            for s,t in ORDER:
                group = by[c,s,t]
                _,_,_,d = threshold_cases(group,old,new)
                ids, boot = scenes[t]
                den = np.zeros(len(ids))
                sums = np.zeros((len(ids),3))
                where = [ids[r['recording']] for r in group]
                w = np.array([r['weight'] for r in group])
                np.add.at(den, where, w)
                np.add.at(sums, where, w[:,None]*d)
                denom = boot @ den
                vals = np.divide(100 * (boot @ sums), denom[:,None],
                                 out=np.full((n_boot,3),np.nan),where=denom[:,None]>0)
                pooled.append(vals)
                independent = int(np.count_nonzero(den))
                if independent > 1:
                    ok = vals[np.isfinite(vals).all(1)]
                    append_interval(out,c,s,t,old,new,ok,independent,False)
            pooled = np.stack(pooled).mean(0)
            valid = pooled[np.isfinite(pooled).all(1)]
            append_interval(out,c,'six_equal','six_equal',old,new,valid,
                            sum(len(v[0]) for v in scenes.values()),any(len(v[0])==1 for v in scenes.values()))
    return out


def append_interval(out, cohort, s, t, old, new, data, cluster_count, conditional):
    lo, hi = np.quantile(data, [.025,.975], axis=0)
    row = dict(cohort=cohort,source=s,target=t,before=old,after=new,scene_clusters=cluster_count,
               fixed_one_scene_strata=conditional,bootstrap_valid_draws=len(data))
    for k,a,b in zip(METRICS,lo,hi):
        row[k+'_lo'],row[k+'_hi']=a,b
    out.append(row)


def main(scope):
    started = time.perf_counter()
    indices = verified(scope)
    existing = table(ROOT/'research/round160/results/all_records.csv')
    refs = {(r['cohort'], int(r['ordinal'])): r for r in existing}
    rows = target_rows()
    locators = object_indices()
    parameters = read(SETTINGS)['sources']
    eval_rows, panel_rows, per_cycle = [], [], []
    dest = HERE/'results'/scope
    dest.mkdir(exist_ok=True,parents=True)
    for n,i in enumerate(indices,1):
        entry = rows[i]
        cohort, src, tgt, ordinal = (entry[k] for k in ('cohort','source','target','ordinal'))
        prior = refs[cohort,ordinal]
        assert (prior['source'],prior['target'])==(src,tgt)
        gt = allowed_gt(entry,locators)
        weight = entry['sampling_weight']
        assert float(prior['sampling_weight'])==weight
        out = dict(cohort=cohort, ordinal=ordinal, source=src, target=tgt, weight=weight,
                   recording=int(prior['recording']), point_group=int(prior['point_group']),
                   **{k:float(prior[k]) for k in FROZEN})
        with np.load(prediction_file(i),allow_pickle=False) as previous:
            old = np.c_[previous['raw'],np.zeros(256)]
            old[:,3:6] = np.maximum(old[:,3:6],1e-6)
        with np.load(WORK/scope/f'{i:05d}.npz',allow_pickle=False) as p:
            out.update(zip(p['arms'].tolist(),many_iou(p['boxes'],gt)))
            out['source_selected_simple'] = out[parameters[src]['simple_control']]
            out['oracle_axis'] = many_iou(old,gt).max()
            old[:,6] = p['all_pca']
            out['oracle_pca'] = many_iou(old,gt).max()
            new = many_iou(p['generated'],gt)
            out['oracle_lcib'] = new.max(initial=0.)
            out['oracle_union'] = max(out['oracle_pca'],out['oracle_lcib'])
            is_scene = p['origins'] < 0
            out['oracle_scene_origin'] = new[is_scene].max(initial=0.)
            assert abs(float(many_iou(np.atleast_2d(p['old_pca']),gt)[0])-out['r158_full']) < 1e-7
            for h in (.25,.50):
                suffix=str(int(h*100))
                prior_good=out['oracle_pca']>h
                new_good=out['lcib']>h
                added = not prior_good and out['oracle_lcib']>h
                out['beyond_axis_'+suffix] = int(out['oracle_axis']<=h and new_good)
                out['beyond_pca_'+suffix] = int(not prior_good and new_good)
                out['beyond_pca_scene_'+suffix] = int(not prior_good and new_good and int(p['selected_origin'])==-1)
                out['generated_cover_'+suffix] = int(added)
                out['new_cover_unselected_'+suffix] = int(added and not new_good)
                out['old_good_new_failed_'+suffix] = int(prior_good and not new_good)
            for name in ('selected_origin','fallback','visibility','roles','known_edges','swapped_seeds',
                         'enough_wrong_frame','role_unknown','forward_seconds','retained','fit_calls','sweeps',
                         'serial_sweeps','serial_fit_calls','solve_seconds','serial_seconds'):
                out[name] = p[name].item()
            if 'lcib_role_boxes' in p:
                panel_rows.append(out.copy())
                for mode in ('lcib','serial','no_feedback','no_relation'):
                    trajectory = p[mode+'_trace']
                    boxes = p[mode+'_trace_boxes']
                    scores = many_iou(boxes,gt)
                    for (step,energy,shift,moved,selected),iou in zip(trajectory,scores):
                        per_cycle.append(dict(cohort=cohort,ordinal=ordinal,source=src,target=tgt,
                                              arm=mode,cycle=int(step),objective=energy,support_L1=shift,
                                              center_motion=moved,ioU=iou))
        eval_rows.append(out)
        if n % 1024 == 0:
            print('Shared scorer',scope,n,len(indices),flush=True)
    d,p,m = summaries(eval_rows,HEADLINES,PAIRS)
    put_table(dest/'rows.csv',eval_rows)
    put_table(dest/'directions.csv',d)
    put_table(dest/'paired.csv',p)
    put_table(dest/'overall.csv',m)
    put_table(dest/'oracle.csv',oracle_table(eval_rows))
    put_table(dest/'paired_scene_ci.csv',confidence_intervals(eval_rows,PAIRS[:3]))
    if panel_rows:
        panel_arms = tuple(dict.fromkeys(a for pair in PANEL_PAIRS for a in pair))
        sd,sp,so = summaries(panel_rows,panel_arms,PANEL_PAIRS)
        for name, data in [('panel_directions',sd),('panel_paired',sp),('panel_overall',so),('panel_trace',per_cycle)]:
            put_table(dest/(name+'.csv'),data)
    atomic_json(dest/'SCORE.json',dict(rows=len(eval_rows), retained_negative_and_missing_rows=True,
                 inspected_target_labels_after_seal=True, target_labels_used_for_configuration=False,
                 prediction_seal_sha256=sha(HERE/(scope+'_SEAL.json')), panel_rows=len(panel_rows),
                 seconds=time.perf_counter()-started, FINAL_access=False))
    print('Scored',scope,len(eval_rows),'panel',len(panel_rows),flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
