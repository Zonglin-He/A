"""Independently reproduce pipeline counts and subgroup metadata aggregates."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np


def read(p):
    return json.loads(p.read_text())


def audit(result, public_root):
    summary = read(result/'SUMMARY.json')
    metadata = read(result/'ANONYMOUS_METADATA.json')
    checks = 0

    def equal(a, b):
        nonlocal checks
        if isinstance(a, float) or isinstance(b, float):
            assert abs(a-b) < 1e-9, (a, b)
        else:
            assert a == b, (a, b)
        checks += 1

    def stage(rr, s, a, b):
        aa = np.array([r[a] for r in rr]); bb = np.array([r[b] for r in rr]); dd = bb-aa
        equal(s['cells'], len(rr)); equal(s['sources'], len({r['source_id'] for r in rr}))
        equal(s['net_pp'], float(dd.mean()*100))
        equal(s['gross_gain_pp'], float(np.maximum(dd, 0).mean()*100))
        equal(s['gross_loss_pp'], float(-np.minimum(dd, 0).mean()*100))
        for t in [.3, .5]:
            good = aa > t; bad = good & (bb <= t)
            expected = dict(correct_before=int(good.sum()), correct_to_wrong=int(bad.sum()),
                            conditional_damage_rate=float(bad.sum()/good.sum()) if good.any() else None,
                            wrong_to_correct=int(((aa <= t) & (bb > t)).sum()))
            for k, v in expected.items(): equal(s[str(t)][k], v)
        for t in [.05, .2]:
            harm = [r for r, d in zip(rr, dd) if d < -t]
            equal(s[f'harm_gt_{t}_cells'], len(harm))
            equal(s[f'harm_gt_{t}_sources'], len({r['source_id'] for r in harm}))

    for ds, d in summary['datasets'].items():
        rows = read(public_root/'results/tastvg_best_quick/2026-10-01'/ds/'ROWS.json')
        refs = read(public_root/'results/tastvg_quick_deep_diagnosis/2026-10-02'/ds/'REFERENCE_ROWS.json')
        meta = metadata[ds]; equal(len(meta), d['queries'])
        equal(d['query_types'], dict(Counter(m['query_type'] for m in meta)))
        assert {r['parent'] for r in rows} == set(range(len(meta)))
        rr = [r for r in rows if r['condition'] != 'clean']
        rf = [r for r in refs if r['condition'] != 'clean']
        for k, a, b in [('inherited_boxes','frozen_v','boxes_only_v'),
                         ('inherited_interval','boxes_only_v','slow_v'),
                         ('temporal_rerank','slow_v','final_v'),('total','frozen_v','final_v')]:
            stage(rr, d['stages'][k], a, b)
        stage([r for r in rr if r['expert_scheduled']], d['expert_temporal'], 'slow_v', 'final_v')
        for t in [.3, .5]:
            lost = [r for r in rr if r['frozen_v'] > t and r['final_v'] <= t]
            z = d['final_correct_loss_path'][str(t)]
            equal(z['final_lost'], len(lost))
            equal(z['first_lost_in_inherited_space'], sum(r['boxes_only_v'] <= t for r in lost))
            equal(z['correct_after_space_but_lost_in_temporal_fast'], sum(r['boxes_only_v'] > t for r in lost))
        for f, groups in d['subgroups'].items():
            for v, s in groups.items():
                gr = [r for r in rr if meta[r['parent']][f] == v]
                gf = [r for r in rf if meta[r['parent']][f] == v]
                equal(s['cells'], len(gr)); equal(s['sources'], len({r['source_id'] for r in gr}))
                equal(s['expert_cells'], len(gf)); equal(s['expert_query_sources'], len({r['parent'] for r in gf}))
                equal(s['empty_reference_cells'], sum(r['valid_frames'] == 0 for r in gf))
                equal(s['nonempty_reference_cells'], sum(r['valid_frames'] > 0 for r in gf))
                equal(s['nonempty_no_GT_overlap_cells'], sum(r['valid_frames'] > 0 and r['inside_GT_frames'] == 0 for r in gf))
                equal(s['frozen_v_mean_pp'], float(np.mean([r['frozen_v'] for r in gr])*100))
                for k, a, b in [('inherited_space','frozen_v','boxes_only_v'),('temporal_fast','slow_v','final_v'),('total','frozen_v','final_v')]:
                    stage(gr, s[k], a, b)
                stage([r for r in gr if r['expert_scheduled']], s['expert_temporal_fast'], 'slow_v', 'final_v')
        for c, s in d['conditions'].items():
            gr = [r for r in rows if r['condition'] == c]
            for k, a, b in [('inherited_space','frozen_v','boxes_only_v'),('temporal_fast','slow_v','final_v'),('total','frozen_v','final_v')]:
                stage(gr, s[k], a, b)
        for k, s in d['input_grid'].items():
            a = np.array([m[k] for m in meta]); equal(s['n'], len(a)); equal(s['mean'], float(a.mean()))
            for name, v in zip(['min','p25','median','p75','max'], np.quantile(a,[0,.25,.5,.75,1])):
                equal(s['quantiles'][name], float(v))
        hit = {str(n): sum(m['uniform5_possible_event_frames'] == n for m in meta) for n in range(6)}
        equal(d['uniform5_geometric_hits'], {k:v for k,v in hit.items() if v})
        equal(d['corrupt_reference']['geometric_zero_event_hits'], sum(meta[r['parent']]['uniform5_possible_event_frames'] == 0 for r in rf))
        equal(d['corrupt_reference']['cells'], len(rf))
        equal(d['corrupt_reference']['empty'], sum(r['valid_frames'] == 0 for r in rf))
        equal(d['corrupt_reference']['nonempty_no_GT_overlap'], sum(r['valid_frames'] > 0 and r['inside_GT_frames'] == 0 for r in rf))
        equal(d['GT_span_exceeds_observed_window_queries'], sum(m['GT_span_exceeds_observed_window'] for m in meta))
        equal(d['GT_span_exceeds_official_input_queries'], sum(m['GT_span_exceeds_official_input'] for m in meta))
        expected = dict(Counter(str(n) for n in Counter(r['source_id'] for r in rr if r['expert_scheduled']).values()))
        equal(d['expert_source_cell_count_distribution'], expected)
        a = np.array([m['unobserved_scorer_event_fraction'] for m in meta]); s = d['unobserved_scorer_event_fraction']
        equal(s['n'], len(a)); equal(s['mean'], float(a.mean()))
        for name, v in zip(['min','p25','median','p75','max'], np.quantile(a,[0,.25,.5,.75,1])):
            equal(s['quantiles'][name], float(v))
        for g, zero in [('geometric_zero',True),('geometric_positive',False)]:
            fr = [r for r in rf if (meta[r['parent']]['uniform5_possible_event_frames'] == 0) == zero]
            expected = dict(empty=sum(r['valid_frames'] == 0 for r in fr),
                            nonempty_no_GT=sum(r['valid_frames'] > 0 and r['inside_GT_frames'] == 0 for r in fr),
                            nonempty_with_GT=sum(r['valid_frames'] > 0 and r['inside_GT_frames'] > 0 for r in fr))
            equal(d['reference_geometry_cross'][g], expected)
    print(json.dumps(dict(status='pass', scalar_checks=checks, GPU_calls=0, GT_annotations_read=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('result', type=Path)
    p.add_argument('--public-root', type=Path, required=True)
    a = p.parse_args()
    audit(a.result, a.public_root)
