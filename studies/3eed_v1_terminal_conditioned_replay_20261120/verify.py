"""Rebuild winners, paired ranks and crossing bounds directly from immutable caches."""
from collections import Counter, defaultdict
import csv
import gzip

from assess_probe import short_id, write_rows
from linear_head import comparison_weights, corrected_scores, loss_gradient
from replay import HERE, PUBLIC, SCHEDULES, WORK, archive, atomic, digest, load_arrays, np, read


def rows(path):
    with (gzip.open if path.suffix=='.gz' else open)(path,'rt',newline='') as file:
        return list(csv.DictReader(file))


def auc_by_ranks(s, positive):
    from scipy.stats import rankdata
    n = positive.sum()
    m = len(positive)-n
    if not n or not m:
        return None
    return (rankdata(s)[positive].sum()-n*(n+1)/2)/(n*m)


def main():
    folder = WORK/'probe'
    seal = read(folder/'SEALED.json')
    gate = read(PUBLIC/'probe_decision.json')
    assert gate['prediction_seal'] == digest(folder/'SEALED.json')
    assert len(seal['files'])==192 and not seal['labels_read']
    public = {(r['row'],r['arm']):r for r in rows(PUBLIC/'probe_rows.csv.gz')}
    metrics = {(r['scope'],r['arm']):r for r in rows(PUBLIC/'probe_metrics.csv')}
    known = {r['row']:(r['case'], r['identity'], r['cohort']) for r in rows(PUBLIC/'probe_cases.csv')}
    cases, summary, diag = [], defaultdict(list), []
    raw_crossings, records, bit_errors = [], defaultdict(set), []
    for s in ('waymo','drone','quad'):
        inputs = archive.source_inputs(s)
        source = archive.sealed_arrays(archive.R181,'SEALED.json','source_scores',s,['ious','scores'])
        for k,v in seal['files'].items():
            domain, name = k.split('/')
            if domain != s:
                continue
            i = int(name.removesuffix('.npz'))
            path = folder/k
            assert digest(path)==v['sha256']
            native = load_arrays(archive.PRIVATE/'source'/k)
            z = load_arrays(path)
            for column in ('slate','boxes','joint'):
                np.testing.assert_array_equal(z[column],native[column])
            np.testing.assert_array_equal(z['joint'],source['scores'][i,:,0])
            bit_errors.extend(z['parity'].tolist())
            odds = z['odds'].astype(float)
            j = z['joint'].astype(float)
            vals = source['ious'][i].astype(float)
            np.testing.assert_array_equal(corrected_scores(j,z['features'][1],np.zeros(288)),j)
            corrections = odds-odds[0]
            modified = j[None, :]+corrections
            np.testing.assert_array_equal(modified,z['incremental'])
            np.testing.assert_array_equal(modified.argmax(axis=1),z['picks'])
            row = short_id(s+inputs[i]['point_sha256'])
            base = int(np.argmax(j))
            paths = {'v1':(None,int(z['v1'])), 'frozen':(j,base)}
            for ix, arm in enumerate(SCHEDULES):
                if ix:
                    paths[arm] = modified[ix],int(modified[ix].argmax())
                paths[arm+'_odds'] = odds[ix],int(odds[ix].argmax())
            records[s].add(inputs[i]['recording'])
            for arm,(scores,winner) in paths.items():
                pub = public[row,arm]
                assert float(pub['iou'])==vals[winner] and int(pub['changed'])==int(base!=winner)
                for t in (25,50):
                    pos = vals>t/100
                    if scores is not None:
                        auc = auc_by_ranks(scores,pos)
                        if auc is not None:
                            assert abs(auc-float(pub[f'pair{t}']))<1e-14
                            summary[arm,t].append(auc)
            terminal = SCHEDULES.index('terminal')
            e = corrections[terminal]
            span = float(np.ptp(e))
            runner_up = np.max(np.delete(j,base))
            diag.append(dict(row=row,source=s,score_range=span,runner_margin=j[base]-runner_up,
                             range_below_runner=int(j[base]-runner_up>span),
                             ratio_hidden=float(np.linalg.norm(z['features'][1])/np.linalg.norm(z['features'][0])),
                             shuffled_hidden=float(np.linalg.norm(z['features'][3])/np.linalg.norm(z['features'][0]))))
            if row in known:
                code,identity,cohort = known[row]
                if cohort=='frozen_miss':
                    good = vals>.5
                    assert good.any() and not good[base]
                    best = int(np.argmax(np.where(good,j,-np.inf)))
                    margin = j[base]-j[best]
                    help_good = e[best]-e[base]
                    cases.append(dict(case=code,source=s,identity=identity,initial_deficit=margin,
                                      strongest_passing_gain=help_good,remaining_deficit=margin-help_good,
                                      correction_range=span,range_smaller_than_deficit=int(margin>span)))
            v, zt = vals[paths['native_odds'][1]],vals[paths['terminal_odds'][1]]
            if (v>.5)!=(zt>.5):
                raw_crossings.append(dict(row=row,source=s,native_odds_iou=float(v),
                                          terminal_odds_iou=float(zt),frozen_joint_iou=float(vals[base])))
    for arm in (a for a in paths if a!='v1'):
        obs = 100*np.mean(summary[arm,25]+summary[arm,50])
        assert abs(obs-float(metrics['all',arm]['balanced_pair_win']))<1e-10
    delta = float(metrics['all','terminal']['balanced_pair_win'])-float(metrics['all','frozen']['balanced_pair_win'])
    assert abs(delta-gate['balanced_pair_win_delta_pp'])<1e-10 and (delta>0)==gate['train_per_protocol']
    assert max(bit_errors)==0 and len(bit_errors)==1152
    assert len(cases)==18 and len(diag)==192
    write_rows(PUBLIC/'recoverable_margin_bounds.csv',sorted(cases,key=lambda x:x['case']))
    write_rows(PUBLIC/'correction_ranges.csv.gz',diag)
    # This artificial three-row example tests differentiation and weighting only.
    # No weight fitting or source annotation enters this numerical check.
    rng = np.random.default_rng(42)
    base = rng.normal(size=(3, 16))
    x = rng.normal(size=(3, 16, 288))
    artificial_iou = rng.uniform(size=(3,16))
    artificial_iou[0] = .75
    pw = comparison_weights(artificial_iou)
    assert abs(pw.sum()-1)<1e-12 and not pw[0].any()
    v = rng.normal(size=288)
    d = rng.normal(size=288)
    d /= np.linalg.norm(d)
    eps = 1e-5
    analytical = loss_gradient(v,base,x,pw,1.)[1] @ d
    numerical = (loss_gradient(v+eps*d,base,x,pw,1.)[0]-loss_gradient(v-eps*d,base,x,pw,1.)[0])/(2*eps)
    assert abs(numerical-analytical)<1e-7
    report = dict(native_layers_verified=len(bit_errors), maximum_native_replay_error=max(bit_errors),
                  number_of_verified_public_rows=len(public), source_recordings={k:len(v) for k,v in records.items()},
                  recoverable_by_identity=dict(Counter(x['identity'] for x in cases)),
                  insufficient_range_for_runner=sum(r['range_below_runner'] for r in diag),
                  insufficient_range_for_passing=sum(r['range_smaller_than_deficit'] for r in cases),
                  median_hidden_relative_norm=float(np.median([r['ratio_hidden'] for r in diag])),
                  median_shuffled_hidden_relative_norm=float(np.median([r['shuffled_hidden'] for r in diag])),
                  native_only_threshold_crossings=raw_crossings, exact_zero_update_winners=192,
                  synthetic_linear_gradient_error=float(abs(numerical-analytical)),
                  probe_model_training_steps=0, target_records_opened=0, FINAL_access=False)
    atomic(PUBLIC/'verification.json',report)
    print(report)


if __name__=='__main__':
    main()
