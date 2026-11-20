"""Score all sealed source records and retain every rescue, harm and fallback."""
import csv
import gzip
import io
from collections import defaultdict

import numpy as np

from audit_native import PRIVATE, RESULTS, validated_predictions
from predict import OUT, PROTOCOL, atomic, digest, prior, read
from seed_code import ARMS, FIELDS, code_identity, focal_costs, key

MAIN = 'native_focal_abstain'
BASE = 'frozen_joint'
PLATFORMS = tuple(PROTOCOL['source'])
METRICS = ('Acc25','Acc50','mIoU')


def csvout(path, rows):
    buffer = io.StringIO(newline='')
    columns = dict.fromkeys(k for row in rows for k in row)
    writer = csv.DictWriter(buffer,fieldnames=columns,lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    raw = buffer.getvalue().encode()
    path.write_bytes(gzip.compress(raw,mtime=0) if path.suffix=='.gz' else raw)


def rank_measures(f,good):
    if not good.any() or good.all():
        return None
    a,b = f[good],f[~good]
    largest_a,largest_b = a.max(),b.max()
    margin = 0. if largest_a==largest_b else float(largest_a-largest_b)
    win = (a[:,None]>b).mean()+.5*(a[:,None]==b).mean()
    return margin,float(win)


def inspect(z,ious,gt):
    joint,codes,fields,choices = z['joint'],z['codes'],z['fields'],z['choices']
    top,perm = z['top'],z['permutation']
    count = codes.sum(1)
    current = int(choices[ARMS.index(BASE)])
    best = int(choices[ARMS.index(MAIN)])
    f = fields[FIELDS.index('native_focal')]
    active = bool(f.max()>0)
    assert np.array_equal(count,top.sum(1)) and np.array_equal(count,count[perm])
    same = (codes[:,None]==codes).all(-1)
    np.testing.assert_array_equal(same,same[perm][:,perm])
    np.testing.assert_array_equal(np.sort(f),np.sort(fields[FIELDS.index('mismatch')]))
    assert f.max()==fields[-1].max()
    changed = (codes!=codes[perm]).any(1)
    ids = [code_identity(mask) for mask in codes]
    groups = {identity:np.array([i for i,k in enumerate(ids) if k==identity]) for identity in dict.fromkeys(ids)}
    reps = [slots[np.argmax(joint[slots])] for slots in groups.values()]
    gt_id = code_identity(gt)
    pos,neg = focal_costs(z['logits'])
    w = neg-pos
    equal_gt = np.array([i==gt_id for i in ids])
    row = dict(active=active,switched=bool(best!=current),class_count=len(groups),
               gt_count=int(gt.sum()),gt_in_pool=bool(equal_gt.any()),
               gt_chosen=bool(equal_gt[best]),gt_class_choice_literal=bool(equal_gt[choices[ARMS.index('native_focal')]]),
               gt_sign_separation=bool((w[gt]>0).all() and (w[~gt]<0).all()),
               gt_positive_weights=int((w[gt]>0).sum()),all_positive_weights=int((w>0).sum()),
               exchangeable=int(changed.sum()),exchangeable_positive=int((changed&(count>0)).sum()),
               positive_codes=int((count>0).sum()),distinct_all_negative=int((count==0).sum()),
               literal_mismatch_winner=bool(choices[ARMS.index('mismatch')]!=choices[ARMS.index('native_focal')]),
               active_mismatch_winner=bool(choices[ARMS.index('mismatch_abstain')]!=best),
               native_top_same_sets=int(np.all(codes==top,axis=1).sum()),
               logged_native_seconds=float(z['native_seconds']),logged_checked_readout_seconds=float(z['readout_and_checks_seconds']))
    for n in range(5):
        row[f'candidates_{n}_seeds']=int((count==n).sum())
    for t in (25,50):
        good = ious>t/100
        mixed_mask = good[:,None]&~good[None,:]
        bad_frozen = not good[current]
        row.update({f'has_good{t}':bool(good.any()),f'all_good{t}':bool(good.all()),
                    f'recoverable{t}':bool(good.any() and bad_frozen),
                    f'class_representative_good{t}':bool(good[reps].any()),
                    f'shared_codes{t}':bool((same&mixed_mask).any()),
                    f'good_bad_pairs{t}':int(mixed_mask.sum()),
                    f'good_bad_same_code_pairs{t}':int((mixed_mask&same).sum()),
                    f'gt_class_has_good{t}':bool((equal_gt&good).any()),
                    f'gt_representative_good{t}':bool(equal_gt.any() and good[groups[gt_id][np.argmax(joint[groups[gt_id]])]])})
    r = {(name,t):rank_measures(fields[j],ious>t/100) for j,name in enumerate(FIELDS) for t in (25,50)}
    return row,r


def gather():
    roster = validated_predictions()
    post = read(PRIVATE/'SEALED.json')
    assert post['predictions']==digest(OUT/'SEALED.json') and post['native_fidelity']==digest(RESULTS/'native_fidelity.json')
    rows=[]
    for platform in PLATFORMS:
        name = platform+'.npz'
        assert post['files'][name]==digest(PRIVATE/name)
        original = prior.source_inputs(platform)
        with np.load(PRIVATE/name,allow_pickle=False) as bank:
            np.testing.assert_array_equal(bank['ordinal'],roster[platform])
            for k,ordinal in enumerate(roster[platform]):
                with np.load(OUT/platform/f'{ordinal:05}.npz',allow_pickle=False) as z:
                    ious,gt=bank['ious'][k],bank['gt_codes'][k]
                    diagnostic,ranking=inspect(z,ious,gt)
                    choices=z['choices']
                    rows.append(dict(platform=platform, ordinal=ordinal, recording=original[ordinal]['recording'],
                                     picked=choices, iou=ious[choices],oracle=ious.max(),features=diagnostic,ranking=ranking))
    return rows


def metric_tensor(rows):
    ious=np.stack([r['iou'] for r in rows])
    return np.stack([ious>.25,ious>.50,ious],axis=2).astype(float)


def pair(rows,arm,ref):
    a,b = ARMS.index(arm),ARMS.index(ref)
    table=dict(arm=arm,reference=ref,n=len(rows),slot_switch=0,iou_better=0,iou_worse=0,iou_equal=0)
    table.update({f'{c}{t}':0 for c in ('rescue','harm') for t in (25,50)})
    for r in rows:
        after,before=r['iou'][[a,b]]
        table['slot_switch']+=int(r['picked'][a]!=r['picked'][b])
        table['iou_better']+=int(after>before+1e-12)
        table['iou_worse']+=int(after<before-1e-12)
        table['iou_equal']+=int(abs(after-before)<=1e-12)
        for t in (25,50):
            table[f'rescue{t}']+=int(after>t/100>=before)
            table[f'harm{t}']+=int(before>t/100>=after)
    return table


def summarized_ranks(rows,field,t):
    values=[r['ranking'][field,t] for r in rows if r['ranking'][field,t] is not None]
    margins=np.asarray([x[0] for x in values])
    finite=margins[np.isfinite(margins)]
    return dict(field=field,threshold=t,n=len(rows),mixed=len(values),
                no_good=sum(not r['features'][f'has_good{t}'] for r in rows),
                all_good=sum(r['features'][f'all_good{t}'] for r in rows),
                good_best_wins=int((margins>0).sum()),ties=int((margins==0).sum()),
                bad_best_wins=int((margins<0).sum()),macro_pair_win=float(np.mean([x[1] for x in values])) if values else None,
                median_margin=float(np.median(margins)) if len(margins) and np.isfinite(np.median(margins)) else None,
                mean_finite_margin=float(finite.mean()) if len(finite) else None,
                plus_infinite=int(np.isposinf(margins).sum()),minus_infinite=int(np.isneginf(margins).sum()))


def coverage(rows,t):
    fs=[r['features'] for r in rows]
    def total(k):return sum(f[k] for f in fs)
    return dict(threshold=t,n=len(rows),oracle_hit=total(f'has_good{t}'),
                frozen_miss_recoverable=total(f'recoverable{t}'),
                any_class_representative_hit=total(f'class_representative_good{t}'),
                unrecoverable_due_to_class_representative=sum(f[f'recoverable{t}'] and not f[f'class_representative_good{t}'] for f in fs),
                recoverable_with_any_good_bad_collision=sum(f[f'recoverable{t}'] and f[f'shared_codes{t}'] for f in fs),
                rows_with_good_bad_collision=total(f'shared_codes{t}'),good_bad_pairs=total(f'good_bad_pairs{t}'),
                same_code_good_bad_pairs=total(f'good_bad_same_code_pairs{t}'),gt_code_present=total('gt_in_pool'),
                gt_code_nonempty=sum(f['gt_count']>0 for f in fs),
                nonempty_gt_code_present=sum(f['gt_count']>0 and f['gt_in_pool'] for f in fs),
                gt_code_has_hit=total(f'gt_class_has_good{t}'),gt_class_representative_hit=total(f'gt_representative_good{t}'),
                gt_sign_theorem_satisfied=total('gt_sign_separation'),GT_positive_seed_bits=total('gt_count'),
                GT_positive_bits_with_positive_weight=total('gt_positive_weights'),all_bits_with_positive_weight=total('all_positive_weights'))


def bootstrap(rows):
    rng=np.random.default_rng(4287)
    samples=[]
    for platform in PLATFORMS:
        by=defaultdict(list)
        for r in rows:
            if r['platform']==platform:
                by[r['recording']].append(r)
        groups=[metric_tensor(by[g]) for g in sorted(by)]
        draw=rng.integers(0,len(groups),size=(2000,len(groups)))
        totals=np.stack([g.sum(0) for g in groups])
        counts=np.array([len(g) for g in groups])
        samples.append(totals[draw].sum(1)/counts[draw].sum(1)[:,None,None])
    res=np.mean(samples,axis=0)*100
    return {arm:{m:np.quantile(res[:,ARMS.index(MAIN),j]-res[:,a,j],[.025,.975]).tolist()
                 for j,m in enumerate(METRICS)} for a,arm in enumerate(ARMS)}


def write_results(rows):
    tasks,pairs,ranks,collisions,gates,controls=[],[],[],[],[],[]
    for group in (*PLATFORMS,'equal'):
        selected=[r for r in rows if group=='equal' or r['platform']==group]
        values=metric_tensor(selected)
        means=values.mean(0)*100
        for j,arm in enumerate(ARMS):
            tasks.append(dict(group=group,arm=arm,n=len(selected),hits25=int(values[:,j,0].sum()),
                              hits50=int(values[:,j,1].sum()),**dict(zip(METRICS,means[j]))))
            for ref in ARMS:
                if ref!=arm and (arm==MAIN or ref in (BASE,'v1')):
                    pairs.append(dict(group=group,**pair(selected,arm,ref)))
        for t in (25,50):
            collisions.append(dict(group=group,**coverage(selected,t)))
            ranks.extend(dict(group=group,**summarized_ranks(selected,f,t)) for f in FIELDS)
        for name,sub in [('all',selected),('abstain',[r for r in selected if not r['features']['active']]),
                         ('active_same',[r for r in selected if r['features']['active'] and not r['features']['switched']]),
                         ('active_switch',[r for r in selected if r['features']['switched']])]:
            for arm in (MAIN,'native_focal'):
                gates.append(dict(group=group,state=name,**pair(sub,arm,BASE)))
        fs=[r['features'] for r in selected]
        controls.append(dict(group=group,n=len(selected),recordings=len({(r['platform'],r['recording']) for r in selected}),
                         slots=len(selected)*16,positive_codes=sum(f['positive_codes'] for f in fs),
                         exchangeable_slots=sum(f['exchangeable'] for f in fs),
                         exchanged_positive_slots=sum(f['exchangeable_positive'] for f in fs),
                         unchanged_slots=sum(16-f['exchangeable'] for f in fs),exchangeable_rows=sum(f['exchangeable']>0 for f in fs),
                         literal_vs_wrong_different_winner=sum(f['literal_mismatch_winner'] for f in fs),
                         guarded_vs_wrong_different_winner=sum(f['active_mismatch_winner'] for f in fs),
                         native_vs_top_identical_codes=sum(f['native_top_same_sets'] for f in fs),
                         **{f'slots_{n}_positive_seeds':sum(f[f'candidates_{n}_seeds'] for f in fs) for n in range(5)}))
    for name,tab in [('metrics',tasks),('paired',pairs),('ranking',ranks),('identifiability',collisions),
                     ('fallback',gates),('mismatch',controls)]:
        csvout(RESULTS/f'source_{name}.csv',tab)
    anon=[]
    for k,r in enumerate(sorted(rows,key=lambda x:key('public',x['platform'],x['ordinal'])),1):
        f=r['features']
        row=dict(row_id=f'S{k:03}',platform=r['platform'],**{n:v for n,v in f.items() if not n.startswith('logged_')},pool_best_iou=r['oracle'])
        for i,arm in enumerate(ARMS):
            row[arm+'_iou']=r['iou'][i]
            row[arm+'_switch']=int(r['picked'][i]!=r['picked'][ARMS.index(BASE)])
        for (field,t),value in r['ranking'].items():
            margin,auc=value if value is not None else (None,None)
            row[f'{field}_margin{t}']=margin
            row[f'{field}_pairwin{t}']=auc
        anon.append(row)
    csvout(RESULTS/'source_all_rows.csv.gz',anon)
    atomic(RESULTS/'source_intervals.json',bootstrap(rows))
    checks={c:pair(rows,MAIN,c) for c in (BASE,'mismatch_abstain','top_focal_abstain','box_max')}
    outcomes={c:v['rescue50']-v['harm50'] for c,v in checks.items()}
    switched=[r for r in rows if r['features']['switched']]
    actual=pair(switched,MAIN,BASE)
    active_loss=actual['harm50']>actual['rescue50']
    # A predeclared stop on damaging actual switches is decisive in this experiment.
    # In other runs a non-damaging result would still need the separate mechanism review.
    atomic(RESULTS/'source_decision.json',dict(n=len(rows),primary=MAIN,paired_net_hits50=outcomes,
                active_switch_rescues50=actual['rescue50'],active_switch_harms50=actual['harm50'],
                source_pass=False if active_loss else None,expand_to_confirmation=False if active_loss else None,
                decision='stop_native_seed_decoder' if active_loss else 'review_predeclared_mechanism_conditions',
                source_recording_bootstrap_replicates=2000,
                prediction_seal=digest(OUT/'SEALED.json'),postseal_audit=digest(PRIVATE/'SEALED.json'),
                replay_forward_seconds_total=sum(r['features']['logged_native_seconds'] for r in rows),
                additional_readout_with_fidelity_checks_seconds_total=sum(r['features']['logged_checked_readout_seconds'] for r in rows)))
    print('Paired Acc50 net counts of main minus comparator',outcomes)
    for t in tasks:
        if t['group']=='equal':print(t['arm'],t['hits25'],t['hits50'],round(t['mIoU'],3))


if __name__=='__main__':
    write_results(gather())
