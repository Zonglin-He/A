"""Final report-only join of 36 independently audited units; no model or label access."""
import csv
import hashlib
import json
import time
from pathlib import Path

R = Path(__file__).resolve().parents[2]
D = R / 'outputs/corruption_subset_shared_roi_v1'
P = R / 'results/corruption_subset_shared_roi_v1'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2) + '\n')


def run():
    plan = read(D / 'plan.json')
    for field in ['code_sha256', 'input_sha256']:
        for name, h in plan[field].items():
            assert sha(R / name) == h, name
    state = read(D / 'state.json')
    assert state['status'] == 'all36_whole128_units_scored_audited_pending_root_analysis_figures_publication'
    assert len(state['completed_units']) == 36
    who = read(D / 'controller.json')
    proc = Path('/proc') / str(who['pid']) / 'stat'
    assert not proc.exists() or int(proc.read_text().rsplit(')', 1)[1].split()[19]) != who['startticks']
    records, bound, stages = [], [], []
    totals = {}
    for e in state['completed_units']:
        ds, name = e['dataset'], e['condition']
        u, pub = D / 'units' / name / ds, P / name / ds
        audit = read(u / 'reporting_receipt.json')
        assert sha(u / 'reporting_receipt.json') == e['reporting_receipt_sha256']
        assert audit['status'] == 'verified_whole128_all_arrays_inputs_teacher_SHA_means'
        assert audit['score_sha256'] == sha(u / 'score_receipt.json')
        assert audit['seal_sha256'] == sha(u / 'sealed.json')
        root = read(u / 'root_export_receipt.json')
        assert root['audit_sha256'] == sha(u / 'reporting_receipt.json')
        assert root['report_sha256'] == sha(pub / 'report.json')
        assert root['tails_sha256'] == sha(pub / 'all_paired_tails.json')
        supplement = read(u / 'root_supplement_receipt_v2.json')
        assert supplement['status'] == 'complete_actual_all128_input_bindings_nested_cost_export'
        assert supplement['input_audit_sha256'] == sha(u / 'root_input_audit_v2.json')
        assert supplement['cost_sha256'] == sha(pub / 'nested_stage_costs_v2.json')
        strata = read(u / 'root_strata_audit_receipt_v1.json')
        assert strata['status'] == 'verified_all_whole_and_nonempty_hit_identity_stratum_means'
        assert strata['metrics_sha256'] == sha(u / 'metrics.json')
        for stage in ['predict', 'score', 'audit']:
            x = read(u / (stage + '_exit.json'))
            assert x['code'] == 0
            child = x['child']
            q = Path('/proc') / str(child['pid']) / 'stat'
            assert not q.exists() or int(q.read_text().rsplit(')', 1)[1].split()[19]) != child['startticks']
            stages.append(dict(dataset=ds, condition=name, stage=stage, **x))
        report = read(pub / 'report.json')
        coverage = read(pub / 'coverage.json')
        costs = read(pub / 'cost_segments.json')
        tails = read(pub / 'all_paired_tails.json')
        assert report['queries'] == report['parents'] == 128
        for k, v in costs['counts'].items():
            totals[k] = totals.get(k, 0) + v
        records.append(dict(dataset=ds, condition=name, means=report['means'], paired=report['paired'],
                            timegroups=report['timegroups'], tails=tails,
                            coverage={k: v for k, v in coverage.items() if k != 'perquery_transform'},
                            costs=costs, nested_costs=read(pub / 'nested_stage_costs_v2.json')))
        bound.append(dict(dataset=ds, condition=name, **{f + '_sha256': sha(u / f) for f in [
            'reporting_receipt.json', 'root_export_receipt.json', 'root_supplement_receipt_v2.json',
            'root_strata_audit_receipt_v1.json', 'predict_exit.json', 'score_exit.json', 'audit_exit.json']}))
    assert totals['queries'] == 4608
    summary = {}
    for ds in ['hc2', 'vidstg']:
        rr = [x for x in records if x['dataset'] == ds]
        assert len(rr) == 18
        def status(pair, metric):
            vals = [x['paired'][pair][metric]['CI95'] for x in rr]
            return dict(positive=sum(lo > 0 for lo, hi in vals), negative=sum(hi < 0 for lo, hi in vals),
                        crosses_or_touches_zero=sum(lo <= 0 <= hi for lo, hi in vals))
        summary[ds] = dict(native_v=status('FullMAP-Source', 'v'), native_t=status('FullMAP-Source', 't'),
                           clean_v=status('FullMAP-CleanFullMAP', 'v'), clean_t=status('FullMAP-CleanFullMAP', 't'),
                           native_v_gain_range_pp=[min(x['paired']['FullMAP-Source']['v']['mean'] for x in rr),
                                                   max(x['paired']['FullMAP-Source']['v']['mean'] for x in rr)],
                           counts={k: sum(x['costs']['counts'][k] for x in rr) for k in totals})
    out = P / 'final'
    out.mkdir(exist_ok=True)
    write(out / 'all36_summary.json', dict(units=36, views=4608, distinct_parents_per_domain=128,
          summary=summary, actual_counts=totals, all36=records,
          CI='Pointwise paired-parent 10000 bootstrap, not multiplicity corrected. Same parents recur across conditions; no pooled independent-4608 inference.',
          cost='Per-worker construction once/PID/start. Nested Source/expert/request/query/stage times must not be added; conditional research wall is not uncached deployment. Inherited clean/legacy costs are nonzero.'))
    with (out / 'all36_table.tsv').open('w', newline='') as f:
        fields = ['dataset', 'condition', 'Source_t', 'Source_v', 'Current_t', 'Current_v',
                  'native_dt', 'native_dt_lo', 'native_dt_hi', 'native_dv', 'native_dv_lo', 'native_dv_hi',
                  'clean_dt', 'clean_dt_lo', 'clean_dt_hi', 'clean_dv', 'clean_dv_lo', 'clean_dv_hi',
                  'native_severe_union', 'native_rescues', 'clean_severe_union', 'clean_rescues',
                  'sampled_hit', 'pixel_identity', 'new_Source', 'new_appearance_prefix', 'new_expert']
        w = csv.DictWriter(f, fields, delimiter='\t'); w.writeheader()
        for x in records:
            row = dict(dataset=x['dataset'], condition=x['condition'])
            for arm, label in [('Source', 'Source'), ('FullMAP', 'Current')]:
                for m in ['t', 'v']:
                    row[label + '_' + m] = x['means'][arm][m] * 100
            for pair, label in [('FullMAP-Source', 'native'), ('FullMAP-CleanFullMAP', 'clean')]:
                for m in ['t', 'v']:
                    v = x['paired'][pair][m]
                    row.update({label+'_d'+m: v['mean'], label+'_d'+m+'_lo': v['CI95'][0], label+'_d'+m+'_hi': v['CI95'][1]})
                row[label+'_severe_union'] = x['tails'][pair]['severe_union_once']
                row[label+'_rescues'] = x['tails'][pair]['rescue_union_once']
            for k in ['sampled_hit', 'pixel_identity']: row[k] = x['coverage'][k]
            for k in ['new_Source', 'new_appearance_prefix', 'new_expert']: row[k] = x['costs']['counts'][k]
            w.writerow(row)
    text = '''# Corruption subset: all 36 units complete

Fixed 128 queries from 128 distinct parents in each domain; 18 conditions and 4608 query-view rows. These are previously exposed confirmation panels, not the full HC1902/Vid10303 populations or independent unseen validation. The same 128 parents recur across conditions. No pooled 4608-parent confidence interval is claimed.

Current uses shared-corrected boxes for feature pooling, kappa0, eight observations and final interval trajectory DP; first auxiliary DP is absent. Source is native per-offset MAP plus physical hull. Methods and severity settings were unchanged throughout this study.

Five physical burst families use ceil(physical frame count times 2.5/5/10 percent), seeded by source identity; Blur/JPEG/Dim remain three separate single-strength controls. Coverage is not equal physical magnitude across families. Hit means sampled RGB overlap, not target/event hit. Occlusion covers 25 percent of selected frame area; exposure is saturated RGB+100. Neither implies unchanged semantics. No-hit/pixel-identity rows are retained, with exact audited clean reuse and nonzero inherited acquisition cost.

## Evidence and limits

Every 128-query unit sealed before independent official scoring and all-array/input/teacher/means audit. Root independently inspected paired signed outcomes, severe temporal/tube union once above5pp, rescues, short<=5s versus long, observed/unobserved support, boundary errors, hit/identity strata, actual RGB/context SHA bindings, and real PID/start cost. All positive and negative results remain in the per-unit TSV/JSON reports. The all36 table carries both corrupt-Source and matched-clean-current contrasts.

HC retains positive tube gains over the corrupted Source in every condition. Vid gains are smaller and most condition-level tube CIs include zero. In particular, high-coverage frame dropping produces a clear drop relative to clean Current in both domains. HC physical motion blur also has clear clean-relative losses at2.5 and10 percent. Positive gains over an already damaged Source do not show that corruption damage was removed. Across-zero CIs are not equivalence or absence-of-harm evidence. CIs are pointwise paired-parent95 percent, not familywise adjusted. Nonmonotonic means were not used to change settings.

The complete severity curve shows both Current-corruptedSource and Current-cleanCurrent, with pointwise CIs. Private rendered PNG is reviewed separately; only scalar SVG/hash/inspection is public. No media, captions, labels/coordinates, features, masks, weights or credentials are exported.

## Actual counts and cost

'''
    text += '```json\n' + json.dumps(dict(summary=summary, actual_counts=totals), indent=2) + '\n```\n\n'
    text += '''[All36 table](all36_table.tsv) and [complete scalar/paired/tail/timegroup/cost join](all36_summary.json) retain each condition separately. Original per-query signed files are linked from the parent study directories. Actual worker time includes decode, checkpoints, checks, IO and root bridge waits. Source, expert forward/request, geometry/pool/finish and query times overlap; they are not additive or uncached deployment timings. Resident teacher construction and original cached acquisition remain disclosed, not zeroed.

All36 prediction, scoring and array-audit stages returned actual wait code0. The single dispatcher is absent after writing its terminal success state; an unavailable external dispatcher wait code is recorded as missing, not fabricated. No experiment owner is restarted. This completes only the authorized subset. Old full219690 corruption, CG/fullgamma and broad external/generic work remain user-paused; extra CGHC stays cancelled.
'''
    (out / 'ANALYSIS.md').write_text(text)
    rec = dict(status='root_all36_signed_CI_tails_timegroups_coverage_cost_join_verified',
               units=36, views=4608, bound_receipts=bound, actual_stage_exits=stages,
               controller=who, controller_incarnation_absent=True, external_controller_wait_code=None,
               missing_external_wait_not_zero=True, code_sha256=sha(Path(__file__)),
               original_plan_sha256=sha(D/'plan.json'), new_model=0, new_GT=0, new_Source=0, new_expert=0,
               summary_sha256=sha(out/'all36_summary.json'), table_sha256=sha(out/'all36_table.tsv'),
               figures_review_and_publication_still_required=True, time=time.time())
    write(D / 'final_root_export_receipt_v1.json', rec)
    public = {k: v for k, v in rec.items() if k not in ['actual_stage_exits', 'controller']}
    write(out / 'root_export_receipt_v1.json', public)
    print(json.dumps(dict(summary=summary, counts=totals, actual_stage_exit_count=len(stages))))


if __name__ == '__main__':
    run()
