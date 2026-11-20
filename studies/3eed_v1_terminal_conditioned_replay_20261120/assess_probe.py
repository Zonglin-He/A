"""Score the immutable 192-row probe and its prespecified competition test."""
import csv
import gzip
import hashlib
import io

from replay import (HERE, PUBLIC, SCHEDULES, WORK, archive, atomic, digest, load_arrays, np, read)


def write_rows(path, rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    raw = buffer.getvalue().encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == '.gz' else raw)


def rank_stats(score, ious, threshold):
    good = ious > threshold
    if not good.any() or good.all():
        return '', ''
    diff = score[good, None]-score[None, ~good]
    return float(((diff > 0)+.5*(diff == 0)).mean()), float(score[good].max()-score[~good].max())


def short_id(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def summarize(items):
    actual = np.array([r['iou'] for r in items])
    frozen = np.array([r['frozen_iou'] for r in items])
    out = dict(n=len(items), mIoU=100*actual.mean(), switches=sum(r['changed'] for r in items))
    across = []
    for t in (25, 50):
        rank = [r[f'pair{t}'] for r in items if r[f'pair{t}'] != '']
        margins = [r[f'margin{t}'] for r in items if r[f'margin{t}'] != '']
        shifts = [r[f'margin_delta{t}'] for r in items if r[f'margin_delta{t}'] != '']
        across.extend(rank)
        out.update({f'Acc{t}':100*(actual>t/100).mean(),
                    f'rescue{t}':int(((actual>t/100)&(frozen<=t/100)).sum()),
                    f'harm{t}':int(((frozen>t/100)&(actual<=t/100)).sum()),
                    f'eligible{t}':len(rank), f'pair_win{t}':100*np.mean(rank) if rank else '',
                    f'mean_margin{t}':np.mean(margins) if margins else '',
                    f'median_margin_shift{t}':np.median(shifts) if shifts else ''})
    out['balanced_pair_win'] = 100*np.mean(across) if across else ''
    return out


def records():
    folder = WORK/'probe'
    seal = read(folder/'SEALED.json')
    lock = read(folder/'LOCK.json')
    assert not seal['labels_read'] and digest(folder/'LOCK.json') == seal['lock']
    assert lock['code']['replay.py'] == digest(HERE/'replay.py')
    seed = HERE.parent/'3eed_v1_native_seed_code_decoding_20261120/run_PRIVATE/postseal'
    seedseal = read(seed/'SEALED.json')
    case_root = HERE.parent/'3eed_v1_k16_gt_error_attribution_20261120'
    with (case_root/'results/identity_review.csv').open() as stream:
        labels = {v['case_id']:v['identity'] for v in csv.DictReader(stream)}
    reviewed = {}
    for code, identity in labels.items():
        content = read(case_root/'review_PRIVATE'/(code+'.json'))
        reviewed[content['platform'],content['ordinal']] = (code, identity, content['cohort'])
    records, examples, timings, maxparity = [], [], [], 0.
    for src in ('waymo','drone','quad'):
        path = seed/(src+'.npz')
        assert digest(path) == seedseal['files'][path.name]
        target = load_arrays(path)
        bank = archive.sealed_arrays(archive.R181,'SEALED.json','source_scores',src,['ious'])['ious']
        source = archive.source_inputs(src)
        for ordinal, ious in zip(target['ordinal'], target['ious']):
            i = int(ordinal)
            np.testing.assert_allclose(ious, bank[i], rtol=0, atol=1e-12)
            key = f'{src}/{i:05}.npz'
            path = folder/key
            assert digest(path) == seal['files'][key]['sha256']
            item = load_arrays(path)
            j, inc, p = item['joint'], int(item['v1']), item['picks']
            base = int(p[0])
            assert base == int(j.argmax())
            maxparity = max(maxparity, float(item['parity'].max()))
            rates = item['timing']
            frame = short_id(src+source[i]['point_sha256'])
            rec = short_id('rec|'+src+source[i]['recording'])
            timings.append(dict(row=frame,source=src,**{k:float(x)*1000 for k,x in zip(SCHEDULES,rates)},
                                pooled_empty=int((item['point_counts']==0).sum()),
                                avg_hidden_change=float(np.linalg.norm(item['features'][1],axis=1).mean()),
                                avg_geometry_change=float(item['terminal_delta_norm'].mean())))
            arms = {'v1':(inc,None),'frozen':(base,j)}
            for k,name in enumerate(SCHEDULES):
                if k != 0:
                    arms[name] = int(p[k]), item['incremental'][k]
                arms[name+'_odds'] = int(item['odds'][k].argmax()), item['odds'][k]
            for arm,(selected,score) in arms.items():
                row = dict(row=frame,source=src,recording_group=rec,arm=arm,iou=float(ious[selected]),
                           frozen_iou=float(ious[base]),changed=int(selected!=base))
                for t in (25,50):
                    auc,margin = rank_stats(score,ious,t/100) if score is not None else ('','')
                    orig = rank_stats(j,ious,t/100)[1]
                    row.update({f'pair{t}':auc, f'margin{t}':margin,
                                f'margin_delta{t}':margin-orig if margin!='' else ''})
                records.append(row)
                if (src,i) in reviewed and arm in ('frozen','terminal','initial','shuffled','last_only'):
                    code,identity,cohort = reviewed[src,i]
                    examples.append(dict(case=code,identity=identity,cohort=cohort,**row))
    return records,examples,timings,maxparity


def paired_intervals(records):
    arms = {r['arm'] for r in records}
    rng = np.random.default_rng(6147)
    groups = {}
    for src in ('waymo','drone','quad'):
        groups[src] = sorted({r['recording_group'] for r in records if r['source']==src})
    sums = {}
    for arm in sorted(arms):
        out = []
        for src,recs in groups.items():
            nums = []
            for rec in recs:
                items = [r for r in records if r['recording_group']==rec and r['arm']==arm]
                v = np.array([r['iou'] for r in items])
                b = np.array([r['frozen_iou'] for r in items])
                nums.append([len(v), ((v>.25).astype(int)-(b>.25)).sum(), ((v>.5).astype(int)-(b>.5)).sum()])
            out.append(np.array(nums))
        sums[arm] = out
    draws = [rng.integers(0,len(r),size=(2000,len(r))) for r in groups.values()]
    bounds = []
    for arm,out in sums.items():
        stats = np.zeros((2000,2))
        for group,picks in zip(out,draws):
            sums_drawn = group[picks].sum(1)
            stats += 100*sums_drawn[:,1:]/sums_drawn[:,0,None]/3
        for j,t in enumerate((25,50)):
            low,high = np.quantile(stats[:,j],[.025,.975])
            bounds.append(dict(arm=arm,metric='Acc'+str(t),delta_pp_low=low,delta_pp_high=high))
    return bounds


def main():
    items,examples,timings,parity = records()
    scopes = ['all',*sorted({r['source'] for r in items}),*sorted({r['recording_group'] for r in items})]
    arms = list(dict.fromkeys(r['arm'] for r in items))
    table = []
    for scope in scopes:
        for arm in arms:
            rows = [r for r in items if r['arm']==arm and (scope=='all' or scope in (r['source'],r['recording_group']))]
            table.append(dict(scope=scope,arm=arm,**summarize(rows)))
    write_rows(PUBLIC/'probe_rows.csv.gz',items)
    write_rows(PUBLIC/'probe_metrics.csv',table)
    write_rows(PUBLIC/'probe_cases.csv',examples)
    write_rows(PUBLIC/'probe_timing.csv',timings)
    write_rows(PUBLIC/'probe_recording_intervals.csv',paired_intervals(items))
    top = {r['arm']:r for r in table if r['scope']=='all'}
    diff = top['terminal']['balanced_pair_win']-top['frozen']['balanced_pair_win']
    decision = dict(probe=192,source_parity_max=parity,balanced_pair_win_delta_pp=diff,
                    train_per_protocol=bool(diff>0),rescue50=top['terminal']['rescue50'],
                    harm50=top['terminal']['harm50'],prediction_seal=digest(WORK/'probe/SEALED.json'),
                    assessment=digest(HERE/'assess_probe.py'),FINAL_access=False)
    atomic(PUBLIC/'probe_decision.json',decision)
    print(decision)
    for arm in ('frozen','terminal','initial','shuffled','last_only'):
        print(arm, {k:top[arm][k] for k in ('Acc25','Acc50','balanced_pair_win','rescue50','harm50')})


if __name__=='__main__':
    main()
