"""Read audited scalar rows only; current-kappa0 clean/corrupt paired reports."""
from ns_common import *
import argparse

def tails(rows, a, b):
    dt = [100 * (r['values'][a]['t'] - r['values'][b]['t']) for r in rows]
    dv = [100 * (r['values'][a]['v'] - r['values'][b]['v']) for r in rows]
    return dict(severe_t=sum(x < -5 for x in dt), severe_v=sum(x < -5 for x in dv),
                severe_union_once=sum(x < -5 or y < -5 for x, y in zip(dt, dv)),
                rescue_union_once=sum(x > 5 or y > 5 for x, y in zip(dt, dv)))

def run(ds):
    validate()
    ap = D / f'corruption_{ds}_reporting_receipt.json'
    audit = read(ap)
    sp = D / f'corruption_{ds}_score_receipt.json'
    assert audit['status'] == 'verified_whole_domain'
    assert audit['all_arrays_cold_rederived_exact'] and audit['all_means_recomputed']
    assert audit['score_sha256'] == sha(sp)
    sealpath = D / f'corruption_{ds}_sealed.json'
    assert read(sp)['seal_sha256'] == sha(sealpath)
    seal = read(sealpath)
    cleanpath = PREV / 'temporal_v2_v2' / ds / 'metrics.json'
    ca = read(D / f'TA_{ds}_adoption_receipt.json')
    assert ca['parent_metrics_sha256'] == sha(cleanpath)
    assert ca['parent_reporting_sha256'] == sha(PREV / f'temporal_v2_{ds}_reporting_receipt.json')
    clean = {r['id']: r for r in read(cleanpath)['private_rows']}
    rows = read(D / 'corruption' / ds / 'metrics.json')['private_rows']
    assert len(rows) == 384 and len({r['parent'] for r in rows}) == 128
    folder = P / 'corruption' / ds
    comparisons = [('FullMAP', 'Source'), ('FullMAP', 'CleanFullMAP'), ('Source', 'CleanSource')]
    summaries, costs = {}, {}
    def ci(x):
        return f"{x['mean']:+.6f} [{x['CI95'][0]:.6f}, {x['CI95'][1]:.6f}]"
    text = f'# Current no-shrinkage controlled views: {ds}\n\n'
    text += (f"Complete 128-query/128-parent panel × three views, sealed and independently officially scored "
             f"({read(sp)['checks']} dense checks, max error {read(sp)['max_error']:.17g}), then "
             f"{audit['prediction_receipt_SHA_checks']} new prediction/receipt SHA checks plus bound original "
             "Source/compact/teacher hashes, all-array exact rederivation and all means verified. "
             "This root export reads already scored scalar rows only; zero new model, Source, expert or GT scoring. "
             "Kappa0 was selected by the user after exposed full results, not independent unseen confirmation.\n\n")
    text += ('|View|Native Source t/v %|FullMAP t/v %|Full−Source Δv pp [95% CI]|'
             'Full−CleanFull Δv pp [95% CI]|Native-source severe t/v/union|\n'
             '|---|---:|---:|---:|---:|---|\n')
    for view in ['Blur', 'JPEG', 'Dim']:
        joined = []
        for r in rows:
            if r['view'] != view:
                continue
            key = r['id'].rsplit('.', 1)[0]
            c = clean[key]
            assert c['parent'] == r['parent'] and c['duration_group'] == r['duration_group']
            joined.append({**r, 'values': {**r['values'], 'CleanSource': c['values']['Source'],
                                         'CleanFullMAP': c['values']['NoShrink']}})
        assert len(joined) == 128 and len({r['parent'] for r in joined}) == 128
        dest = folder / view / 'matched_clean'
        report = suite().parent_report(joined, ['Source', 'FullMAP', 'CleanSource', 'CleanFullMAP'], comparisons, dest)
        suite().signed_tails(joined, comparisons, dest)
        ts = {a + '-' + b: tails(joined, a, b) for a, b in comparisons}
        atomic(dest / 'all_paired_tails.json', ts)
        summaries[view] = dict(means=report['means'], paired=report['paired'], tails=ts, timegroups=report['timegroups'])
        m, z = report['means'], ts['FullMAP-Source']
        text += (f"|{view}|{100*m['Source']['t']:.6f}/{100*m['Source']['v']:.6f}|"
                 f"{100*m['FullMAP']['t']:.6f}/{100*m['FullMAP']['v']:.6f}|"
                 f"{ci(report['paired']['FullMAP-Source']['v'])}|{ci(report['paired']['FullMAP-CleanFullMAP']['v'])}|"
                 f"{z['severe_t']}/{z['severe_v']}/{z['severe_union_once']}|\n")
        cost = dict(original_Source_calls=0, original_new_teacher_calls=0, old_teacher_hits=0,
                    original_Source_native_forward_seconds=0., original_current_teacher_lookup_seconds=0.,
                    inherited_original_teacher_forward_seconds=0., original_finish_wall_seconds=0.,
                    original_query_pipeline_seconds=0., original_worker_queries={})
        for r in joined:
            e = seal['files'][r['id']]['provenance']['original_acquisition']
            s, t = e['Source_cost'], e['teacher']
            cost['original_Source_calls'] += 1
            cost['original_new_teacher_calls'] += s['ReferDINO_query_forwards']
            cost['old_teacher_hits'] += int(t['old_acquisition_reused'])
            cost['original_Source_native_forward_seconds'] += s['Source224_seconds']
            cost['original_current_teacher_lookup_seconds'] += s['expert_current_wall_seconds']
            cost['inherited_original_teacher_forward_seconds'] += t['original_cost']['forward_seconds']
            cost['original_finish_wall_seconds'] += e['finish_cost']['finish_wall_seconds']
            cost['original_query_pipeline_seconds'] += e['query_seconds']
            w = f"{e['worker']['pid']}:{e['worker']['startticks']}"
            cost['original_worker_queries'][w] = cost['original_worker_queries'].get(w, 0) + 1
        costs[view] = cost
    text += '\n## Every signed comparison and duration group\n\n'
    text += '|View / comparison / group|Queries|Δt pp [95% CI]|Δv pp [95% CI]|\n|---|---:|---:|---:|\n'
    for view, s in summaries.items():
        groups = {'all': dict(queries=128, paired=s['paired']), **s['timegroups']}
        for group, g in groups.items():
            for key, v in g['paired'].items():
                text += f"|{view} / {key} / {group}|{g['queries']}|{ci(v['t'])}|{ci(v['v'])}|\n"
    text += ('\nEvery signed positive/negative scalar, severe loss (>5pp, t/v union once), rescue, observed/'
             'unobserved effect and short (≤5s)/long interval is retained in each view’s matched_clean report/TSVs. '
             'CIs are pointwise parent-cluster bootstrap, not familywise. The same 128 parents recur across views; '
             '384 views are not 384 independent parents. No blanket robustness or guaranteed non-degradation claim.\n\n'
             'Blur=5×5 Gaussian sigma1.5; JPEG quality30/subsampling2; Dim uint8×0.5. Original acquisition used '
             'actual transformed-RGB Source/grid and exact-context teacher payloads, rebuilt geometry/real ROI pooling. '
             'Current kappa0 replay uses its own posterior MAP interval and actually re-solves final DP, while original '
             'kappa64 array equality remains an exact prerequisite. CleanFullMAP is the audited NoShrink output on '
             'these same queries; Source remains native per-offset MAP+hull. No MBR performance comparison.\n\n'
             'Costs retain actual Source/teacher provenance and original per-PID/start query counts; current cached '
             'CPU rederivation wall is additional research work. Source-forward/request/finish/query durations overlap '
             'and must not be added. Old teacher hits do not erase original expert acquisition. Loading/decode/outerloop '
             'are accounted only by whole original worker wall, not individually instrumented. None of these cached '
             'times substitutes for current uncached deployment measurement. Media/captions/coordinates remain private.\n')
    (folder / 'ANALYSIS.md').write_text(text)
    origin = PREV / 'corruptions' / 'sealed.json'
    old = read(origin)
    atomic(folder / 'cost_segments.json', dict(views=costs, current_worker_CPU_seconds=audit['worker_CPU_seconds'],
        original_whole_worker=old['worker'], original_whole_worker_seconds=old['seconds'],
        original_whole_worker_scope='Both domains/768 views together; not summed with nested components.',
        original_seal_sha256=sha(origin), new_Source=0, new_expert=0))
    atomic(folder / 'matched_clean_summary.json', summaries)
    rec = dict(status='root_complete_matched_clean_export', code_sha256=sha(Path(__file__)),
               audit_sha256=sha(ap), score_sha256=sha(sp), clean_metrics_sha256=sha(cleanpath),
               summary_sha256=sha(folder / 'matched_clean_summary.json'), analysis_sha256=sha(folder / 'ANALYSIS.md'),
               new_Source=0, new_expert=0, new_GT_scoring=0)
    atomic(D / f'corruption_{ds}_root_export_receipt.json', rec)
    atomic(folder / 'root_export_receipt.json', rec)
    print(ds, 'current no-shrinkage corruption root exported', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=COUNTS, required=True)
    run(p.parse_args().dataset)
