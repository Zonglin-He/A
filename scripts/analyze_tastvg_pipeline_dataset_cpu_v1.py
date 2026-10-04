"""Posthoc dataset/pipeline associations; saved scalars and annotation metadata only.

No model imports, prediction generation, box scoring, parameter fitting or online rule.
The private PLAN/pointers/annotations are never exported. Output is anonymous aggregates.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import ijson
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUICK = ROOT / 'artifacts/tastvg_best_quick_v1'
DEEP = ROOT / 'artifacts/tastvg_quick_deep_diagnosis_cpu_v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def distribution(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()),
                quantiles=dict(zip(['min', 'p25', 'median', 'p75', 'max'],
                                   map(float, np.quantile(a, [0, .25, .5, .75, 1])))))


def stage(rows, before, after):
    a = np.asarray([r[before] for r in rows])
    b = np.asarray([r[after] for r in rows])
    d = b - a
    out = dict(cells=len(rows), sources=len({r['source_id'] for r in rows}),
               net_pp=float(d.mean()*100), gross_gain_pp=float(np.maximum(d, 0).mean()*100),
               gross_loss_pp=float(-np.minimum(d, 0).mean()*100))
    for margin in [.05, .2]:
        harm = [r for r, v in zip(rows, d) if v < -margin]
        out[f'harm_gt_{margin}_cells'] = len(harm)
        out[f'harm_gt_{margin}_sources'] = len({r['source_id'] for r in harm})
    for t in [.3, .5]:
        good = a > t
        bad = good & (b <= t)
        out[str(t)] = dict(correct_before=int(good.sum()), correct_to_wrong=int(bad.sum()),
                          conditional_damage_rate=float(bad.sum()/good.sum()) if good.any() else None,
                          wrong_to_correct=int(((a <= t) & (b > t)).sum()))
    return out


def read_metadata(ds, plan, pins):
    exposure = read(QUICK / ds / 'GT_EXPOSURE.json')
    if ds == 'vidstg':
        lp = ROOT / 'artifacts/stvg_fullscale_diagnostics_v1/labels_diagnostic_only.json'
        ap = ROOT / 'external/VidSTG-Dataset/annotations/test_annotations.json'
        for path in [lp, ap]:
            expected = exposure['annotation_sources'][str(path.relative_to(ROOT))]
            assert sha(path) == expected
            pins[str(path.relative_to(ROOT))] = expected
        annotations = read(ap)
        wanted = {r['key'] for r in plan['rows']}
        with lp.open('rb') as f:
            pointers = {k: v['official_annotation'] for k, v in ijson.kvitems(f, '', use_float=True)
                        if k in wanted}
        assert len(pointers) == len(wanted)
    else:
        ap = ROOT / 'data/hcstvg2_official_metadata/val_v2.json'
        expected = exposure['annotation_sources'][str(ap.relative_to(ROOT))]
        assert sha(ap) == expected
        pins[str(ap.relative_to(ROOT))] = expected
        annotations = read(ap)
    result = []
    for parent, r in enumerate(plan['rows']):
        ids = r['frame_ids']
        fps = r['input']['fps']
        if ds == 'vidstg':
            ptr = pointers[r['key']]
            a = annotations[ptr['annotation_index']]
            assert a['vid'] == r['source']
            assert a[ptr['field']][ptr['query_index']]['target_id'] == ptr['target_id']
            assert a[ptr['field']][ptr['query_index']]['description'] == r['input']['caption']
            assert r['query_type'] == ('caption' if ptr['field'] == 'captions' else 'question')
            lo, hi = a['temporal_gt']['begin_fid'], a['temporal_gt']['end_fid']
            label_hi = hi
            used_lo, used_hi = a['used_segment']['begin_fid'], a['used_segment']['end_fid']+1
            assert (r['input']['start_frame'], r['input']['end_frame']) == (used_lo, used_hi)
        else:
            a = annotations[r['annotation_key']]
            assert a['English'].lower() == r['input']['caption']
            lo = int(a['st_frame']) - 1
            hi = lo + len(a['bbox']) - 1  # audited legacy scorer span
            label_hi = hi + 1  # inclusive last box -> annotation half-open extent
            used_lo, used_hi = 0, r['input']['frame_count']
        clip_frames = ids[-1] - ids[0] + 1
        fraction = (hi-lo) / clip_frames
        positions = np.rint(np.linspace(0, len(ids)-1, 5)).astype(int)
        # REFERENCE_ROWS tests presence of a dense GT box, including HC's last
        # annotated frame. It does not test the legacy scorer's exclusive end.
        possible = sum(lo <= ids[i] < label_hi for i in positions)
        result.append(dict(parent=parent, query_type=r['query_type'], grid_frames=len(ids),
                           observed_grid_envelope_seconds=clip_frames/fps,
                           official_input_segment_seconds=(used_hi-used_lo)/fps,
                           original_video_seconds=r['input']['frame_count']/fps,
                           scorer_event_seconds=(hi-lo)/fps,
                           annotation_event_seconds=(label_hi-lo)/fps,
                           event_fraction=fraction,
                           GT_span_exceeds_observed_window=bool(lo < ids[0] or hi > ids[-1]+1),
                           GT_span_exceeds_official_input=bool(lo < used_lo or hi > used_hi),
                           unobserved_scorer_event_fraction=max(0., 1-max(0, min(hi, ids[-1]+1)-max(lo, ids[0]))/(hi-lo)),
                           event_bin='le_25pct' if fraction <= .25 else ('25_to_50pct' if fraction <= .5 else 'gt_50pct'),
                           uniform5_possible_event_frames=possible))
    return result


def group(rows, refs, meta, field, value):
    rr = [r for r in rows if meta[r['parent']][field] == value]
    rf = [r for r in refs if meta[r['parent']][field] == value]
    first_refs = {r['parent'] for r in rf}
    return dict(sources=len({r['source_id'] for r in rr}), cells=len(rr),
                expert_cells=len(rf), expert_query_sources=len(first_refs),
                empty_reference_cells=sum(r['valid_frames'] == 0 for r in rf),
                nonempty_reference_cells=sum(r['valid_frames'] > 0 for r in rf),
                nonempty_no_GT_overlap_cells=sum(r['valid_frames'] > 0 and r['inside_GT_frames'] == 0 for r in rf),
                frozen_v_mean_pp=float(np.mean([r['frozen_v'] for r in rr])*100),
                inherited_space=stage(rr, 'frozen_v', 'boxes_only_v'),
                temporal_fast=stage(rr, 'slow_v', 'final_v'),
                expert_temporal_fast=stage([r for r in rr if r['expert_scheduled']], 'slow_v', 'final_v') if rf else None,
                total=stage(rr, 'frozen_v', 'final_v'))


def run(out):
    pins = {}
    results = {}
    anonymous_metadata = {}
    global_barrier = QUICK / 'GLOBAL_PREDICTION_BARRIER.json'
    assert global_barrier.exists()
    pins[str(global_barrier.relative_to(ROOT))] = sha(global_barrier)
    for ds in ['vidstg', 'hc2']:
        paths = [QUICK/ds/'PLAN.json', QUICK/ds/'ROWS.json', QUICK/ds/'PIPELINE_DIAGNOSIS.json',
                 DEEP/ds/'REFERENCE_ROWS.json', DEEP/ds/'DEEP_STEPS.json']
        for p in paths:
            pins[str(p.relative_to(ROOT))] = sha(p)
        plan, rows, diagnosis, refs, steps = map(read, paths)
        assert len(rows) == plan['total']
        meta = read_metadata(ds, plan, pins)
        anonymous_metadata[ds] = meta
        corrupt = [r for r in rows if r['condition'] != 'clean']
        cr = [r for r in refs if r['condition'] != 'clean']
        expert = [r for r in corrupt if r['expert_scheduled']]
        refkeys = {(r['parent'], r['order'], r['condition'], r['arrival']) for r in cr}
        assert refkeys == {(r['parent'], r['order'], r['condition'], r['arrival']) for r in expert}
        assert len(refkeys) == len(cr) == len(expert)
        # Independent readback against the saved stage counts and aggregate arithmetic.
        stages = {k: stage(corrupt, a, b) for k, a, b in [
            ('inherited_boxes', 'frozen_v', 'boxes_only_v'),
            ('inherited_interval', 'boxes_only_v', 'slow_v'),
            ('temporal_rerank', 'slow_v', 'final_v'),
            ('total', 'frozen_v', 'final_v')]}
        for name, s in stages.items():
            old = diagnosis['full']['corruption'][name]
            for t in ['0.3', '0.5']:
                for k in ['correct_before', 'correct_to_wrong', 'wrong_to_correct']:
                    assert s[t][k] == old[t][k]
            for k in ['gross_gain_pp', 'gross_loss_pp']:
                assert abs(s[k]-old[k]) < 1e-10
        for r in cr:
            assert r['inside_GT_frames'] <= meta[r['parent']]['uniform5_possible_event_frames']
        subgroup = {}
        for f in ['query_type', 'event_bin']:
            subgroup[f] = {v: group(corrupt, cr, meta, f, v) for v in sorted({m[f] for m in meta})}
        condition = {}
        for c in plan['conditions']:
            rr = [r for r in rows if r['condition'] == c]
            condition[c] = {k: stage(rr, a, b) for k, a, b in [
                ('inherited_space', 'frozen_v', 'boxes_only_v'),
                ('temporal_fast', 'slow_v', 'final_v'),
                ('total', 'frozen_v', 'final_v')]}
        final_loss_attribution = {}
        for t in [.3, .5]:
            lost = [r for r in corrupt if r['frozen_v'] > t and r['final_v'] <= t]
            space_first = [r for r in lost if r['boxes_only_v'] <= t]
            temporal_only = [r for r in lost if r['boxes_only_v'] > t]
            assert len(lost) == len(space_first)+len(temporal_only)
            final_loss_attribution[str(t)] = dict(final_lost=len(lost),
                first_lost_in_inherited_space=len(space_first),
                correct_after_space_but_lost_in_temporal_fast=len(temporal_only),
                note='Exhaustive threshold path accounting, not causal interventions or sum of independent effects')
        results[ds] = dict(sources=plan['sources'], queries=plan['queries'],
                           query_types=dict(Counter(m['query_type'] for m in meta)),
                           input_grid={k: distribution([m[k] for m in meta]) for k in
                                       ['grid_frames', 'observed_grid_envelope_seconds', 'official_input_segment_seconds',
                                        'original_video_seconds', 'annotation_event_seconds',
                                        'scorer_event_seconds', 'event_fraction']},
                           uniform5_geometric_hits=dict(Counter(str(m['uniform5_possible_event_frames']) for m in meta)),
                           GT_span_exceeds_observed_window_queries=sum(m['GT_span_exceeds_observed_window'] for m in meta),
                           GT_span_exceeds_official_input_queries=sum(m['GT_span_exceeds_official_input'] for m in meta),
                           unobserved_scorer_event_fraction=distribution([m['unobserved_scorer_event_fraction'] for m in meta]),
                           corrupt_reference=dict(cells=len(cr), empty=sum(r['valid_frames'] == 0 for r in cr),
                                                  nonempty_no_GT_overlap=sum(r['valid_frames'] > 0 and r['inside_GT_frames'] == 0 for r in cr),
                                                  geometric_zero_event_hits=sum(meta[r['parent']]['uniform5_possible_event_frames'] == 0 for r in cr)),
                           reference_geometry_cross={g:{
                                'empty':sum(r['valid_frames'] == 0 for r in cr if (meta[r['parent']]['uniform5_possible_event_frames'] == 0) == zero),
                                'nonempty_no_GT':sum(r['valid_frames'] > 0 and r['inside_GT_frames'] == 0 for r in cr if (meta[r['parent']]['uniform5_possible_event_frames'] == 0) == zero),
                                'nonempty_with_GT':sum(r['valid_frames'] > 0 and r['inside_GT_frames'] > 0 for r in cr if (meta[r['parent']]['uniform5_possible_event_frames'] == 0) == zero)}
                                for g,zero in [('geometric_zero',True),('geometric_positive',False)]},
                           expert_source_cell_count_distribution=dict(Counter(str(n) for n in Counter(r['source_id'] for r in expert).values())),
                           stages=stages, expert_temporal=stage(expert, 'slow_v', 'final_v'),
                           final_correct_loss_path=final_loss_attribution,
                           subgroups=subgroup, conditions=condition)
    output = dict(scope='CPU posthoc saved scalar readback + already-exposed annotation temporal metadata',
                  grouping='Fixed query types and event/input-span fraction bins <=.25, (.25,.5], >.5; descriptive only',
                  aggregation='Full/query-type/event-bin all-stream cells balanced per source: means equal source macro. Expert subgroup means and damage denominators are cell-weighted; repeated expert-source membership differs between orders.',
                  correctness='Strict dense vIoU > .3 or > .5; never sum successive stage destruction counts',
                  HC_span_note='annotation event duration counts inclusive last box; legacy scorer uses last box frame as exclusive span end; both reported',
                  data_boundary=dict(GPU_calls=0, model_forwards=0, new_box_scoring=0, new_predictions=0,
                                     parameter_updates=0, captions_exported=False, GT_coordinates_exported=False,
                                     GT_used_for_online_rule=False, method_promoted=False),
                  limitations=['All sources historically exposed; quick one-query/source panels are not full benchmark',
                               'Associations are posthoc, no subgroup bootstrap or confounder adjustment',
                               'Different checkpoint, query distribution, optimizer and orders confound dataset comparison',
                               'Reference no-overlap is not proof an update was wrong or useless',
                               'Metadata does not measure shot cuts, target identity mistakes or visibility outside event'],
                  datasets=results)
    out.mkdir(parents=True, exist_ok=True)
    target = out/'SUMMARY.json'
    assert not target.exists(), target
    target.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    (out/'ANONYMOUS_METADATA.json').write_text(json.dumps(anonymous_metadata, indent=2, allow_nan=False)+'\n')
    audit = dict(status='pass', input_sha256=pins, summary_sha256=sha(target),
                 stage_readback='All .3/.5 threshold counts and gross gain/loss independently matched',
                 reference_join='Every expert-scheduled corruption cell joined exactly once; GT-valid hits <= geometric hits',
                 input_files=len(pins), code_sha256=sha(Path(__file__)))
    (out/'PRIVATE_INPUT_AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(dict(status='pass', output=str(target), inputs=len(pins))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.out)
