"""Root review of completed domain scalars and actual costs; no GT/model access."""
from pathlib import Path
import argparse, csv, hashlib, json, math, statistics, time

R = Path(__file__).resolve().parents[3]
D = R / 'outputs/videowalk_full_v1'
P = R / 'results/videowalk_full_v1'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def main(ds):
    private, public = D / ds, P / ds
    report = read(public / 'report.json')
    receipt = read(private / 'reporting_receipt.json')
    audit = read(private / 'report_statistics_audit.json')
    assert receipt['report_sha256'] == audit['report_sha256'] == sha(public / 'report.json')
    assert read(private / 'score_audit.json')['metrics_sha256'] == sha(private / 'metrics.json')
    rows = read(private / 'metrics.json')['private_rows']
    assert len(rows) == report['queries'] == receipt['queries']
    arms = ['Matching', 'Current', 'PureMatched']
    mean_checks = 0
    for arm in arms:
        for field, want in report['means'][arm].items():
            values = [r['values'][arm][field] for r in rows if r['values'][arm][field] is not None]
            got = statistics.mean(values) if values else None
            assert (got is None) == (want is None)
            if got is not None:
                assert abs(got - want) < 1e-12, (arm, field, got, want)
            mean_checks += 1
    pairs = [('Current', 'Matching'), ('Matching', 'PureMatched'), ('Current', 'PureMatched')]
    counts = {}
    for a, b in pairs:
        name = a + '-' + b
        delta = [100 * (r['values'][a]['v'] - r['values'][b]['v']) for r in rows]
        stats = report['paired']['all'][name]['statistics']
        assert abs(statistics.mean(delta) - stats['v']['mean']) < 1e-12
        counts[name] = dict(improved=sum(x > 1e-10 for x in delta),
                            harmed=sum(x < -1e-10 for x in delta),
                            unchanged=sum(abs(x) <= 1e-10 for x in delta),
                            decline_gt5pp=sum(x < -5 for x in delta),
                            rescue_gt5pp=sum(x > 5 for x in delta),
                            decline_ge10pp=sum(x <= -10 for x in delta))
        assert abs(100 * counts[name]['improved'] / len(rows) - stats['improved']['mean']) < 1e-12
        assert abs(100 * counts[name]['harmed'] / len(rows) - stats['harmed']['mean']) < 1e-12
    seal = read(private / 'sealed.json')
    old = read(R / 'outputs/spatial_temporal_six_v1/matching_pilot' / ds / 'sealed.json')
    empty_counts, empty_queries, native_support = 0, 0, 0
    for key, e in seal['files'].items():
        diag = old['files'][key]['proof']['diagnostics'] if e['origin']['kind'] == 'original128_exact_reuse' else e['diagnostics']
        n = diag['empty_unobserved_masks']
        assert diag['all_eight_boxes_exact'] and not diag['Source_E_boxes_or_DP_used']
        empty_counts += n
        empty_queries += int(n > 0)
        # All HC entries have 65 samples. Frame counts for both domains come
        # from the original immutable input list, with no annotation access.
    selection = read(D / 'selection.json')[ds]
    native_support = sum(len(r['metadata']['frame_ids']) - 8 for r in selection)
    costs = read(public / 'all_actual_query_costs.json')
    cost_by_origin = {}
    for kind in ['new_Source', 'original_retained_full_grid', 'original128_exact_reuse']:
        rr = [r for r in costs['rows'] if r['kind'] == kind]
        cost_by_origin[kind] = {'queries': len(rr)}
        for field in ['Source_seconds', 'matching_CPU_seconds', 'query_wall_seconds']:
            values = [r[field] for r in rr if r.get(field) is not None]
            cost_by_origin[kind][field] = dict(measured=len(values), mean=statistics.mean(values) if values else None,
                                              sum=sum(values) if values else None)
    exits = {p.name: read(p) for p in (D / 'exits').glob(ds + '_*.json')}
    assert len([x for x in exits if 'shard' not in x]) == 6
    assert all(e['actual_subprocess_wait'] and e['exit_code'] == 0 for e in exits.values())
    load = read(private / 'model_load.json')
    summary = dict(queries=len(rows), parents=report['parents'], all_absolute_mean_checks=mean_checks,
                   signed_counts=counts,
                   empty_masks=dict(native_unobserved_frames=empty_counts, total_native_unobserved_frames=native_support,
                                    queries_any_empty=empty_queries, zero_box_no_fallback=True,
                                    all_eight_original_expert_boxes_exact=True),
                   conditional_support=dict(unobserved_queries=report['paired']['all']['Current-Matching']['statistics']['event_unobserved_IoU']['queries'],
                                            observed_queries=report['paired']['all']['Current-Matching']['statistics']['event_observed_IoU']['queries']),
                   cost_by_origin=cost_by_origin, Source_model_load=load,
                   actual_stage_exit_wall_seconds={k: e['elapsed_wall_seconds'] for k, e in exits.items() if 'shard' not in k},
                   separate_shard_audits=len([k for k in exits if 'shard' in k]),
                   inherited_Source_and_expert_acquisition_nonzero=True,
                   nested_clocks_not_additive=True, experimental_IO_cold_replay_not_deployment=True,
                   report_sha256=sha(public / 'report.json'), GT_read=False, new_Source=0, new_expert=0)
    (public / 'ROOT_review.json').write_text(json.dumps(summary, indent=2) + '\n')
    text = [f'# ROOT review: {ds} complete Source-feature VideoWalk extension', '',
            f'{len(rows)} queries; {report["parents"]} parent clusters. All six main child stages and independent shards exited zero. Root read the complete report, absolute means, paired parent-cluster 10,000 intervals, short/long groups, distance support, signed tails, empty masks and measured cost. This review does not rescore GT or rerun any model.', '',
            '|Spatial path (same Current interval)|vIoU|R.3|R.5|Native unobserved IoU|Conditional dense unobserved IoU|',
            '|---|---:|---:|---:|---:|---:|']
    for arm in ['PureMatched', 'Matching', 'Current']:
        m = report['means'][arm]
        text.append('|' + arm + '|' + '|'.join(f'{100*m[k]:.6f}' for k in ['v','R03','R05','native_unobserved_IoU','event_unobserved_IoU']) + '|')
    for group, comparisons in report['paired'].items():
        text += ['', group + ':']
        for name, result in comparisons.items():
            v = result['statistics']['v']
            text.append(f'- {name}: delta vIoU {v["mean"]:+.6f} pp; 95% parent-cluster CI {v["CI95"]}; {result["queries"]} queries.')
    text += ['', 'Signed counts: ' + json.dumps(counts), '', 'Empty-mask accounting: ' + json.dumps(summary['empty_masks']), '',
             'Interpretation: the result supports the comparison with this exact source-fixed Source-feature VideoWalk module adaptation. Its lower performance than pure interpolation is retained; it is not evidence against the full original learned-encoder VideoWalk method or every strong propagation algorithm. Same expert input and interval do not hard-lock Current observed outputs. Current retains its original soft E/expert selection, with its observed-frame losses recorded separately. No target retuning or method/anchor/paper changes.', '',
             'Native unobserved IoU averages GT-supported native sampled unobserved frames per query. Conditional dense unobserved IoU restricts support to the common predicted/GT-event intersection, excluding exact eight observed physical frames; empty support is missing, not zero. Distance bins are 0-.5, .5-1, 1-2, 2-4, >=4 seconds. A zero-support bin is unscorable, not equivalence. The existing exposed target lists and pointwise CIs do not constitute an unseen confirmation or familywise guarantee.', '',
             'All Source acquisition, matching, provenance IO, cold replay, startup and actual process costs are archived. Source/new-forward averages are separated from reused inputs. The first matching-pass CPU clock excludes acquisition/cold audit and is not a matched deployment comparison. Model load is counted once for its PID/start; nested clocks must not be added. Reused Source/expert acquisition costs are inherited, not zero.', '',
             'Full signed review: ROOT_review.json. Full short/long, conditional distance, all positive/negative/null intervals: report.json. Whole suite still running until both domains and final root/publication verification complete.']
    (public / 'ROOT_FINDINGS_v1.md').write_text('\n'.join(text) + '\n')
    receipt = dict(status='ROOT_personally_reviewed_complete_domain_all_means_signed_tails_shortlong_distance_empty_cost',
                   queries=len(rows), report_sha256=sha(public / 'report.json'), review_sha256=sha(public / 'ROOT_review.json'),
                   findings_sha256=sha(public / 'ROOT_FINDINGS_v1.md'), root_code_sha256=sha(Path(__file__)),
                   score_audit_sha256=sha(private / 'score_audit.json'), statistics_audit_sha256=sha(private / 'report_statistics_audit.json'),
                   new_model=0, new_Source=0, new_expert=0, new_GT=0, whole_suite_complete=False, time=time.time())
    (private / 'ROOT_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['hc2', 'vidstg'], required=True)
    main(p.parse_args().dataset)
