"""Evaluate sealed choices on original single-target boxes and independently reconstruct every arm."""
import argparse
import csv
import gzip
import hashlib
from pathlib import Path

import numpy as np
from scipy.special import softmax

from collect_probes import ARMS, CAMERA, HERE, PUBLIC, WORK, digest, inputs, read, roster_id, save
from calibrate import csv_file
from assess import measure, paired, record_intervals, score_boxes, sensor_truth


def reconstruct(base, totals):
    # Independent, probability-space computation of the same update. No use of probes.evidence.
    original, isolated, opened, early, shuffled = totals
    known = (original[0] > 0) & (original[1] > 0)
    p = softmax(base)
    predictions = np.tile(p, (len(ARMS), 1))
    if sum(known) < 2:
        return predictions
    d, w = original[:, known]
    multipliers = []
    for x in (isolated, opened, early, shuffled):
        extra_density, extra_word = x[:, known]
        multipliers.append((w+extra_word)/(d+extra_density)/(w/d))
    mass = (w+isolated[1,known])/w
    density = (d+isolated[0,known])/d
    np.testing.assert_allclose(multipliers[0]*density, mass, rtol=3e-13)
    for i, mult in enumerate([*multipliers[:3], mass, density, multipliers[3]],1):
        scores = p[known]*mult
        predictions[i,known] = scores*(p[known].sum()/scores.sum())
    return predictions


def historical_rows(stage):
    if stage == 'ordinary':
        return {}
    with gzip.open(CAMERA/'results'/f'{stage}_rows.csv.gz','rt') as f:
        return {r['row']:float(r['v1_iou']) for r in csv.DictReader(f)}


def probability_win(difference):
    return np.mean((difference>1e-10) + .5*(abs(difference)<=1e-10))


def assess(stage):
    root = WORK/stage
    seal = read(root/'SEALED.json')
    lock = read(root/'LOCK.json')
    assert seal['lock'] == digest(root/'LOCK.json') and not seal['task_labels_opened'] and not seal['FINAL_access']
    assert lock['arms'] == list(ARMS) and all(digest(HERE/n)==h for n,h in lock['code'].items())
    old_iou = historical_rows(stage)
    roster = inputs(stage)
    assert lock['roster'] == {f'{s}_to_{t}':roster_id(rows) for (s,t),rows in roster.items()}
    details, diagnostics, groups, tables = [], [], {}, {}
    max_iou_error, max_replay_error, prior_error = 0., 0., 0.
    expected = sum(map(len,roster.values()))
    for (source,target),rows in roster.items():
        direction = f'{source}_to_{target}'
        metric, picks, recs, per_frame = [], [], [], []
        for off in range(0,len(rows),64):
            name = f'{direction}/{off:05d}.npz'
            part = seal['files'][name]
            assert digest(root/name)==part['sha256'] and part['lock']==seal['lock']
            with np.load(root/name, allow_pickle=False) as batch:
                z = dict(batch)
            assert len(z['choices'])==part['n']
            for j, chosen in enumerate(z['choices']):
                row = rows[off+j]
                key = hashlib.sha256(f'{stage}|{direction}|{off+j}'.encode()).hexdigest()[:16]
                rec = row.get('recording') or Path(row['metapath']).parent.parent.name
                rec_key = hashlib.sha256(f'{target}|{rec}'.encode()).hexdigest()[:12]
                gt = sensor_truth(target, row['metapath'], row['caption'], row['suffix'])
                ious, error = score_boxes(z['boxes'][j],gt)
                max_iou_error = max(max_iou_error, error)
                if old_iou:
                    prior_error = max(prior_error, abs(old_iou[key]-ious[chosen[0]]))
                verified = reconstruct(z['base'][j], z['totals'][j])
                max_replay_error = max(max_replay_error,np.max(abs(verified-softmax(z['logits'][j],axis=1))))
                np.testing.assert_array_equal(verified.argmax(1), chosen)
                assert np.array_equal(z['logits'][j,0], z['base'][j])
                detail = dict(row=key,direction=direction,recording=rec_key,
                              opportunity25=int(ious[chosen[0]]<=.25<ious.max()),
                              opportunity50=int(ious[chosen[0]]<=.50<ious.max()),
                              supported_slots=int(z['valid'][j].sum()),
                              reserve_overlap_slots=int((z['totals'][j,1,0]>0).sum()))
                for k,arm in enumerate(ARMS):
                    detail[f'{arm}_iou'] = float(ious[chosen[k]])
                    detail[f'{arm}_switch'] = int(chosen[k]!=chosen[0])
                details.append(detail)
                good = ious>.50
                joint = z['logits'][j]
                delta = joint[1]-joint[0]
                loss = z['totals'][j]
                pair_correct = None
                pair_margin = None
                if good.any() and (~good).any():
                    pair_correct = probability_win(delta[good,None]-delta[None,~good])
                    pair_margin = max(joint[1,good])-max(joint[1,~good])-max(joint[0,good])+max(joint[0,~good])
                diag = dict(row=key,direction=direction,opportunity50=detail['opportunity50'],
                            present_slots=detail['supported_slots'],reserve_slots=detail['reserve_overlap_slots'],
                            active_slots=int(((loss[0,0]>0)&(loss[1,0]>0)&(loss[0,1]>0)).sum()),
                            probe_overlap_sum=float(loss[1,0].sum()), native_overlap_sum=float(loss[0,0].sum()),
                            effective_logit_spread=float(np.ptp(delta)),pair_win=pair_correct,pair_margin_shift=pair_margin,
                            private_native_box=float(z['native_error'][j,0]),private_native_word=float(z['native_error'][j,1]),
                            open_native_box=float(z['native_error'][j,2]), open_native_word=float(z['native_error'][j,3]),
                            singleton_box=float(z['single_error'][j,0]), singleton_word=float(z['single_error'][j,1]),
                            reserve_center_shift=float(z['reserve_center_shift'][j]), reserve_word_shift=float(z['reserve_word_change'][j]),
                            anchor_error=float(z['anchor_error'][j]),private_ms=1000*float(z['seconds'][j,0]),
                            open_ms=1000*float(z['seconds'][j,1]))
                per_frame.append(diag)
                metric.append(ious[chosen]); picks.append(chosen); recs.append(rec_key)
        assert len(metric)==len(rows)
        diagnostics.extend(per_frame)
        groups[source,target]=dict(ious=np.array(metric),choices=np.array(picks),recording=np.array(recs))
        tables[source,target]=measure(np.array(metric))
    assert len(details)==expected==seal['n'] and len({r['row'] for r in details})==expected
    assert max_replay_error < 1e-10 and prior_error < 1e-9 and max_iou_error < 1e-7
    mats={k:g['ious'] for k,g in groups.items()}
    macro=np.mean(list(tables.values()),axis=0)
    measurements, changes, intervals, coverage = [], [], [], []
    compare = [(i,0) for i in range(1,len(ARMS))]+[(1,i) for i in range(2,len(ARMS))]
    for key in [*groups,'macro']:
        if key == 'macro':
            name='macro'
            counts = np.concatenate([g['choices'] for g in groups.values()])
            mat = np.concatenate(list(mats.values()))
            met = macro
            diag = diagnostics
        else:
            name='_to_'.join(key)
            mat=mats[key]
            counts=groups[key]['choices']
            met=tables[key]
            diag=[d for d in diagnostics if d['direction']==name]
        for arm,values in zip(ARMS,met):
            measurements.append(dict(direction=name,arm=arm,n=len(mat),Acc25=values[0],Acc50=values[1],mIoU=values[2]))
        for a,b in compare:
            delta=met[a]-met[b]
            changes.append(dict(direction=name,arm=ARMS[a],baseline=ARMS[b],Acc25_pp=delta[0],
                                Acc50_pp=delta[1],mIoU_pp=delta[2],**paired(mat,counts,a,b)))
        checks=[d for d in diag if d['singleton_box']>=0]
        mixed=[d for d in diag if d['pair_win'] is not None]
        opportunities=[d for d in mixed if d['opportunity50']]
        coverage.append(dict(direction=name,n=len(diag),at_least_two_active=sum(d['active_slots']>=2 for d in diag),
            reserve_touches_any=sum(d['reserve_slots']>0 for d in diag),mixed_good_bad50=len(mixed),
            mean_pair_win=float(np.mean([d['pair_win'] for d in mixed])) if mixed else None,
            opportunities50=len(opportunities),positive_margin_opportunities=sum(d['pair_margin_shift']>1e-10 for d in opportunities),
            negative_margin_opportunities=sum(d['pair_margin_shift']< -1e-10 for d in opportunities),
            singleton_checks=len(checks),max_singleton_box=max(d['singleton_box'] for d in checks),
            max_singleton_word=max(d['singleton_word'] for d in checks),
            max_native_box=max(d['private_native_box'] for d in diag),max_native_word=max(d['private_native_word'] for d in diag),
            median_open_native_max_box=float(np.median([d['open_native_box'] for d in diag])),
            median_reserve_center_shift=float(np.median([d['reserve_center_shift'] for d in diag])),
            median_private_ms=float(np.median([d['private_ms'] for d in diag])),median_open_ms=float(np.median([d['open_ms'] for d in diag]))))
    for a,b in compare:
        lo,hi=record_intervals(groups,mats,a,b)
        intervals.append(dict(arm=ARMS[a],baseline=ARMS[b],delta50=macro[a,1]-macro[b,1],
                              descriptive_lower95=lo, descriptive_upper95=hi))
    for stem,records in [('metrics',measurements),('paired',changes),('coverage',coverage),('intervals',intervals)]:
        csv_file(PUBLIC/f'{stage}_{stem}.csv',records)
    for stem,records in [('rows',details),('mechanism',diagnostics)]:
        with gzip.open(PUBLIC/f'{stage}_{stem}.csv.gz','wt',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=records[0])
            writer.writeheader(); writer.writerows(records)
    save(PUBLIC/f'{stage}_audit.json',dict(n=expected,seal=digest(root/'SEALED.json'),scoring_code=digest(__file__),
         distinct_rows=len({r['row'] for r in details}),recalculated_choices_all=True,max_probability_error=max_replay_error,
         max_iou_error=max_iou_error,max_previous_baseline_iou_error=prior_error,seconds=seal['seconds'],FINAL_access=False))
    if stage=='pilot':
        primary = next(r for r in changes if r['direction']=='macro' and r['arm']==ARMS[1] and r['baseline']==ARMS[0])
        expand = all(macro[1,1]>macro[ARMS.index(a),1] for a in ('v1','open_mean','shuffled_mean'))
        save(PUBLIC/'pilot_decision.json',dict(expand=bool(expand and primary['rescue50']>primary['harm50']),
             primary=primary, conditions=read(HERE/'PROTOCOL.json')['cohorts'],FINAL_access=False))
    for r in measurements:
        if r['direction']=='macro':
            print(stage,r,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('source','pilot','ordinary'))
    assess(p.parse_args().stage)
