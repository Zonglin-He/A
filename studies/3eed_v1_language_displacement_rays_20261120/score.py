"""Score sealed predictions, including abstentions, against oriented boxes."""
import argparse
import csv
import gzip
import importlib.util
import io
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from capture import (ARMS, CFG, HERE, PLATFORMS, PRIVATE, ROOT, R181, R183, atomic,
                     checked, digest, key, ordinary, read, sealed_arrays, source_inputs)
from rays import energies
from roles import inventory
from scope183 import match_pattern
from evaluate import score_boxes, to_sensor

OUT = HERE / 'results'
NAMES = ('origin','ray','reverse','mismatch')
METRICS = ('Acc25','Acc50','mIoU')


def csvwrite(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def hungarian(full, logits, offsets, row, gt, category, mapping):
    """Single-target final-head assignment under the captured full-prompt tokenization."""
    caption = row['caption'].strip()
    span = None
    for word in mapping.get(category,[category]):
        match = re.search(r'\b'+re.escape(word)+r'\b',caption,re.I)
        if match:
            span = match.span()
            break
    if span is None:
        return None
    hit = np.flatnonzero((offsets[:,1]>offsets[:,0]) & (offsets[:,0]<span[1]) & (offsets[:,1]>span[0]))
    if not len(hit):
        return None
    prob = torch.zeros((1,logits.shape[-1]))
    prob[0,hit] = 1 / len(hit)
    assert abs(float(prob.sum()) - 1.) < 1e-6
    with torch.inference_mode():
        matched = MATCHER(dict(pred_boxes=torch.from_numpy(full[None]),pred_logits=torch.from_numpy(logits[None])),
                          [dict(labels=torch.zeros(1,dtype=torch.long),positive_map=prob,
                                boxes=torch.from_numpy(gt[None,:6].astype(np.float32)))])
    return matched[0][0].item()


def comparisons(scores, good):
    if not good.any() or good.all():
        return None
    diff = scores[~good][None,:] - scores[good][:,None]
    return float(np.mean((diff > 0) + .5 * (diff == 0)))


def inspect(z, ious, gt, row, category):
    full, origins = z['full'][-1], z['reduced'][-1,:,:3].astype(np.float64)
    v = full[:,:3].astype(np.float64) - origins
    mags = (v*v).sum(1)
    fields = dict(valid=bool(z['valid']),reason=str(z['reason']),response=bool(mags.any()),
                  changed_layers=int(np.any(z['full']!=z['reduced'],axis=(1,2)).sum()),
                  median_displacement=float(np.median(np.sqrt(mags))),
                  max_displacement=float(np.sqrt(mags.max())),extra_seconds=float(z['extra_seconds']))
    if not mags.any():
        return fields
    w = mags / mags.sum()
    target = gt[:3]
    gt_energy = energies(target[None], origins, v, z['permutation'])[0]
    for j, tag in enumerate(NAMES):
        fields['gt_rmse_'+tag] = float(np.sqrt(gt_energy[j]))
        for threshold in (25,50):
            good = ious > threshold / 100
            val = comparisons(z['energy'][:,j],good)
            if val is not None:
                fields[f'auc{threshold}_{tag}'] = val
            if (~good).any():
                bad = z['energy'][~good,j]
                fields[f'gt_vs_fail{threshold}_{tag}'] = float(np.mean((gt_energy[j] < bad)+.5*(gt_energy[j]==bad)))
    fields.update(effective_queries=float(1/(w@w)),top_weight=float(w.max()),
                  lambda_min=float(np.linalg.eigvalsh(np.eye(3) - v.T @ v / mags.sum())[0]),
                  forward_weight_at_gt=float(w[(v*(target-origins)).sum(1) > 0].sum()))
    mapping = MAPS['WAYMO_SYNONYMS' if row['platform']=='waymo' else 'M3ED_SYNONYMS']
    match = hungarian(full,z['sem_logits'],z['offsets'],row,gt,category,mapping)
    if match is not None:
        fields['matched_weight'] = float(w[match])
        fields['largest_is_matched'] = bool(match == int(w.argmax()))
        fields['matched_points_forward'] = bool(v[match]@(target-origins[match]) > 0)
    return fields


def labelled(stage):
    dest = PRIVATE / stage
    seal = read(dest/'SEALED.json')
    lock = checked(dest/'LOCK.json',seal['lock'])
    assert not seal['annotations_used'] and not seal['FINAL_access']
    assert all(digest(ROOT / k) == h for k,h in lock['code'].items())
    assert all(digest(dest / p) == h for p,h in seal['files'].items())
    ordinary_rows = ordinary() if stage == 'ordinary' else None
    rows = []
    max_iou_error = max_reference_error = 0.
    for source in PLATFORMS:
        targets = [source] if stage == 'source' else [t for t in PLATFORMS if t!=source]
        if stage == 'source':
            archived = sealed_arrays(R181,'SEALED.json','source_scores',source,['ious'])['ious']
        for target in targets:
            inputs = source_inputs(source) if stage == 'source' else ordinary_rows[target]
            direction = source if stage=='source' else f'{source}_to_{target}'
            for i in lock['roster'][target]:
                row = inputs[i]
                gt_boxes, classes = inventory(row)
                reference = gt_boxes[0].astype(float)
                path = dest / direction / f'{i:05}.npz'
                assert str(path.relative_to(dest)) in seal['files']
                with np.load(path,allow_pickle=False) as z:
                    if stage == 'source':
                        ious, err = score_boxes(z['boxes'].astype(float),reference)
                        max_reference_error = max(max_reference_error,float(np.abs(ious-archived[i]).max()))
                        assert max_reference_error < 1e-5
                        for t in (.25,.5):
                            np.testing.assert_array_equal(ious>t,archived[i]>t)
                    else:
                        from score183 import truth
                        gt = truth(row['metapath'],row['object_index'],row['caption'],target,row['suffix'])
                        ious, err = score_boxes(to_sensor(z['boxes'],target,int(z['view'])),gt)
                    max_iou_error = max(max_iou_error,err)
                    picks = z['choices']
                    r = dict(direction=direction,target=target,recording=row['recording'],
                             ordinal=i,pool_best=float(ious.max()),pilot=bool(z['parity']),
                             choices=picks,ious=ious[picks],**inspect(z,ious,reference,row,str(classes[0])))
                    rows.append(r)
    assert len(rows) == seal['n']
    return rows,dict(iou_two_implementations_max_error=max_iou_error,
                     source_k16_archived_iou_max_error=max_reference_error)


def values(rows):
    ious = np.stack([r['ious'] for r in rows])
    return np.stack((ious>.25,ious>.50,ious),axis=2).astype(float)


def intervals(rows):
    """Recording-level paired draws within target; both sources use the same draw."""
    rng = np.random.default_rng(6147)
    outputs = []
    for target in PLATFORMS:
        on = [r for r in rows if r['target']==target]
        names = sorted({r['recording'] for r in on})
        draws = rng.integers(0,len(names),size=(2000,len(names)))
        for d in sorted({r['direction'] for r in on}):
            parts = [values([r for r in on if r['direction']==d and r['recording']==n]) for n in names]
            counts = np.asarray([len(p) for p in parts])
            sums = np.stack([p.sum(0) for p in parts])
            outputs.append(sums[draws].sum(1) / counts[draws].sum(1)[:,None,None])
    sampled = np.mean(outputs,axis=0) * 100.
    summary = {}
    for a, arm in enumerate(ARMS):
        diff = sampled[:,ARMS.index('ray'),:] - sampled[:,a,:]
        idx = np.concatenate((diff,diff[:,:2].mean(1,keepdims=True)),axis=1)
        summary[arm] = {k:np.quantile(idx[:,j],[.025,.975]).tolist()
                        for j,k in enumerate((*METRICS,'hit_utility'))}
    return summary


def score(stage):
    rows, audit = labelled(stage)
    OUT.mkdir(exist_ok=True)
    ds = list(dict.fromkeys(r['direction'] for r in rows))
    tab, paired, mechanism, pools = [], [], [], []
    extra_keys = set().union(*(r.keys() for r in rows))
    diagnostic_keys = sorted(k for k in extra_keys if k.startswith(('gt_', 'auc','matched_')) or k in
        ('median_displacement','max_displacement','effective_queries','top_weight','lambda_min',
         'forward_weight_at_gt','largest_is_matched'))
    for d in [*ds,'equal']:
        selected = rows if d == 'equal' else [r for r in rows if r['direction']==d]
        val = values(selected)
        # All platforms/directions were sampled in exactly equal counts.
        means = val.mean(0)*100
        for j, arm in enumerate(ARMS):
            tab.append(dict(group=d,arm=arm,n=len(selected),**dict(zip(METRICS,means[j]))))
        for k,reference in enumerate(ARMS):
            for a,arm in enumerate(ARMS):
                if a==k or (arm != 'ray' and reference not in ('v1','frozen_joint')):
                    continue
                before,after = val[:,k],val[:,a]
                pair = dict(group=d,arm=arm,reference=reference,n=len(selected))
                for j, threshold in enumerate((25,50)):
                    pair[f'rescue{threshold}'] = int(((after[:,j]>0)&(before[:,j]==0)).sum())
                    pair[f'harm{threshold}'] = int(((after[:,j]==0)&(before[:,j]>0)).sum())
                pair.update(iou_better=int((after[:,2]>before[:,2]+1e-12).sum()),
                            iou_worse=int((after[:,2]<before[:,2]-1e-12).sum()),
                            switched=int(sum(r['choices'][a]!=r['choices'][k] for r in selected)))
                paired.append(pair)
        active = [r for r in selected if r['response']]
        pool = dict(group=d,total=len(selected),recordings=len({(r['target'],r['recording']) for r in selected}),
                    category_available=sum(r['valid'] for r in selected),responding=len(active),
                    zero_after_valid=sum(r['valid'] and not r['response'] for r in selected),
                    parity=sum(r['pilot'] for r in selected),
                    oracle25=sum(r['pool_best']>.25 for r in selected), oracle50=sum(r['pool_best']>.5 for r in selected),
                    seven_stages_changed=sum(r['changed_layers']==7 for r in selected),
                    response_change_timemean_s=float(np.mean([r['extra_seconds'] for r in active])) if active else None)
        pools.append(pool)
        for field in diagnostic_keys:
            arr = [r[field] for r in active if field in r]
            if arr:
                mechanism.append(dict(group=d,stat=field,n=len(arr),mean=float(np.mean(arr)),median=float(np.median(arr))))
        for tag in ('reverse','mismatch','origin'):
            diff = np.array([r['gt_rmse_'+tag]-r['gt_rmse_ray'] for r in active])
            if diff.size:
                mechanism.append(dict(group=d,stat='paired_gt_rmse_'+tag+'_minus_ray',n=len(diff),
                                      mean=float(diff.mean()),median=float(np.median(diff))))
    uncertainty = intervals(rows)
    for name,table in [('metrics',tab),('paired',paired),('mechanism',mechanism),('coverage',pools)]:
        csvwrite(OUT / f'{stage}_{name}.csv',table)
    atomic(OUT / f'{stage}_intervals.json',uncertainty)
    # Release every row, with outcome and scalar diagnostics only. Private IDs never leave run_PRIVATE.
    public_fields = ['direction','valid','reason','response','changed_layers','pool_best',*diagnostic_keys]
    public = []
    for r in sorted(rows,key=lambda r:key('anonymous',stage,r['direction'],r['ordinal'])):
        entry={k:r.get(k,'') for k in public_fields}
        entry.update({a+'_iou':v for a,v in zip(ARMS,r['ious'])})
        public.append(entry)
    with (OUT/f'{stage}_all_rows.csv.gz').open('wb') as f, gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as g, io.TextIOWrapper(g,encoding='utf-8',newline='') as s:
        w = csv.DictWriter(s,fieldnames=public[0]); w.writeheader(); w.writerows(public)
    atomic(OUT / f'{stage}_scoring_audit.json',dict(n=len(rows),counts_by_parse_reason=dict(Counter(r['reason'] for r in rows)),
           sealed_predictions=digest(PRIVATE/stage/'SEALED.json'),**audit))
    if stage == 'source':
        stats = {r['arm']: r for r in tab if r['group']=='equal'}
        u = {arm:(r['Acc25']+r['Acc50'])/2 for arm,r in stats.items()}
        ray = stats['ray']
        relative = {k:u['ray']-u[k] for k in ('seed','origin','reverse','mismatch')}
        net = {m:ray[m]-stats['frozen_joint'][m] for m in METRICS[:2]}
        truth_gain = {k:next((r['median'] for r in mechanism if r['group']=='equal' and
                      r['stat']=='paired_gt_rmse_'+k+'_minus_ray'), None) for k in ('reverse','mismatch')}
        checks = dict(beat_simple_and_wrong_controls=all(v>0 for v in relative.values()),
                      retained_both_task_thresholds=bool(min(net.values())>=0 and max(net.values())>0),
                      true_center_prefers_correct=all(x is not None and x>0 for x in truth_gain.values()),
                      positive_cluster_lower_bound=uncertainty['frozen_joint']['hit_utility'][0]>0)
        decision = dict(expand=all(checks.values()),source_seal=digest(PRIVATE/'source/SEALED.json'),
                        checks=checks,utility_leads_pp=relative,ray_minus_frozen_pp=net,
                        paired_true_center_rmse_control_minus_ray_m=truth_gain,uncertainty=uncertainty['frozen_joint'],
                        source_records=len(rows),FINAL_access=False)
        atomic(OUT/'source_gate.json',decision)
        print(json.dumps(decision,indent=2))
    print([r for r in tab if r['group']=='equal'])


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('source','ordinary'))
    stage=p.parse_args().stage
    torch.set_num_threads(2)
    native=ROOT/'crossplatform/round3/p0/upstream/models/losses.py'
    spec=importlib.util.spec_from_file_location('native_hungarian_loss', native)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MATCHER=module.HungarianMatcher(1,0,2,True)
    MAPS=match_pattern()
    score(stage)
