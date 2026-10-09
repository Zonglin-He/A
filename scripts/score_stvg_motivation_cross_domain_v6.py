"""Post-global-seal CPU native T/S diagnostic. No model construction or execution."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.stvg_motivation_cross_domain_common_v6 import *
from vg_tta.stvg_motivation_quadrants_v6 import independent_components, summarize_models


def read_truths(direction, roster):
    target = direction_datasets(direction)[1]
    if target == 'vidstg':
        from scripts.diagnose_tastvg_pipeline_cpu_v1 import truth
        plan = dict(rows=roster['rows'], orders=dict(selected=list(range(128))))
        dense, spans, provenance = truth('P1', plan)
    else:
        path = ROOT/'data/hcstvg2_official_metadata/val_v2.json'
        expected = read(PAPER/'DESIGN_LOCK.json')['hc_annotation_metadata_sha256']
        assert sha(path) == expected
        ann = read(path); dense = {}; spans = {}
        for row in roster['rows']:
            i = row['ordinal']; value = ann[row['annotation_key']]
            assert value['English'].lower() == row['input']['caption']
            start = int(value['st_frame'])-1
            spans[i] = [start, start+len(value['bbox'])-1]
            w, h = row['input']['width'], row['input']['height']
            dense[i] = {start+j: [x, y, min(x+bw, w), min(y+bh, h)] for j, (x, y, bw, bh) in enumerate(value['bbox'])}
        provenance = {str(path.relative_to(ROOT)): sha(path)}
    assert len(dense) == len(spans) == 128 and all(dense.values())
    return dense, spans, provenance


def official_components(z, row, truth, span, target):
    from vg_tta.tastvg_paper48_metrics_v1 import official_functions
    import ast
    scope = official_functions()
    if target == 'hc2':
        p = ROOT/'external/TA-STVG/datasets/evaluation/hcstvg_eval.py'
        nodes = [x for x in ast.parse(p.read_text()).body if isinstance(x, ast.ClassDef) and x.name == 'HCSTVGiouEvaluator']
        assert len(nodes) == 1
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), 'exec'), scope)
        cls = scope['HCSTVGiouEvaluator']
    else:
        cls = scope['VidSTGiouEvaluator']
    result = independent_components(z['boxes'], z['box_frame_ids'], z['interval'], truth, span,
        row['input']['width'], row['input']['height'], hc2=target == 'hc2',
        spatial_valid=z['spatial_valid'] and z['format_valid'], temporal_valid=z['temporal_valid'] and z['format_valid'])
    ev = cls.__new__(cls)
    ev.vid2steds = {0: span}; ev.vid2box = {0: {int(k): [v] for k, v in truth.items()}}
    ev.vid2names = {0: 'anonymous'}; ev.vid2sents = {0: ''}; ev.iou_thresholds = [.3, .5]
    pix = result['pixel_boxes']; ids = z['box_frame_ids']; tube = {}
    if len(pix):
        full = scope['linear_interp']({int(i): [b.tolist()] for i, b in zip(ids, pix)})
        for fid, value in full.items():
            b = np.asarray(value[0])
            if np.isfinite(b).all() and (b[2:] > b[:2]).all():
                tube[fid] = value
    temporal = {0: dict(sted=result['interval'], qtype='declarative')}
    old = ev.evaluate({0: tube}, temporal, {})[0][0] if target == 'hc2' else ev.evaluate({0: tube}, temporal, {}, {})[0][0]
    error = max(abs(float(old['tiou'])-result['tIoU']), abs(float(old['gt_viou'])-result['sIoU']))
    assert error < 1e-10, (target, error)
    result['max_official_independent_error'] = error
    for name in ['pixel_boxes', 'GT_frame_ids', 'per_GT_frame_IoU', 'interval']:
        result.pop(name)
    return result


def run():
    start = time.time()
    g = check_global_seal()
    assert not (BASE/'GT_EXPOSURE.json').exists(), 'Retain a previous score attempt for root review'
    write(BASE/'GT_EXPOSURE.json', dict(status='authorized_read_after_all_512_cross_native_predictions_sealed',
        global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
        global_seal_time=g['time'], purpose='human requested cross-domain Panel B',
        active_P1_prediction_payload_read=False, time=time.time()))
    allrows = []; summary = {}; max_error = 0.; gtproof = {}
    for direction in DIRECTIONS:
        roster = read(BASE/direction/'ROSTER.json')
        _, target = direction_datasets(direction)
        dense, spans, provenance = read_truths(direction, roster)
        gtproof[direction] = provenance
        values = {}
        for model in MODELS:
            rr = []
            for row in roster['rows']:
                i = row['ordinal']; p = BASE/direction/model/'predictions'/f'{i:05}.json'; z = read(p)
                v = official_components(z, row, dense[i], spans[i], target)
                max_error = max(max_error, v['max_official_independent_error'])
                rr.append(dict(model=model, direction=direction, parent=i,
                    source_dataset=direction_datasets(direction)[0], target_dataset=target,
                    **v, native_format_valid=z['format_valid'], native_temporal_valid=z['temporal_valid'],
                    native_spatial_valid=z['spatial_valid'], native_output_sha256=sha(p)))
            values[model] = ([r['tIoU'] for r in rr], [r['sIoU'] for r in rr]); allrows.extend(rr)
        summary[direction] = summarize_models(values, seed=SEED, draws=10000, threshold=THRESHOLD)
    write(BASE/'SCALAR_ROWS.json', allrows)
    write(BASE/'QUADRANT_STATISTICS.json', dict(status='scored_pending_root_review',
        native_only=True, source_training_only=True, cohort_sources_per_direction=128,
        total_native_cells=512, historically_exposed=True, directions=summary,
        definition=dict(temporal='tIoU > .5', spatial='GT-fixed-frame mean IoU > .5',
            uncovered_spatial_support='zero IoU; full GT-frame denominator', strict_greater_than=True),
        bootstrap_draws=10000, seed=SEED, GT_provenance=gtproof,
        global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'), time=time.time()))
    write(BASE/'SCORING_AUDIT.json', dict(status='pass', native_cells=512, component_metric_comparisons=1024,
        independent_geometry_vs_official=True, max_metric_error=max_error,
        all_formats_retained=True, paired_parent_bootstrap=10000,
        GT_after_all_native_predictions=True, source_preprocessing_binding=True,
        no_OPD_prediction_payload_access=True, CPU_seconds=time.time()-start, time=time.time()))
    # Root still has to view the actual image and complete public verification.
    from scripts.render_stvg_motivation_cross_domain_v6 import render_statistics
    render_statistics()
    write(BASE/'CPU_COMPLETION.json', dict(status='CPU_cross_domain_native_quadrants_complete_pending_actual_root',
        native_cells=512, quadrants_sha256=sha(BASE/'QUADRANT_STATISTICS.json'),
        scalar_rows_sha256=sha(BASE/'SCALAR_ROWS.json'), actual_root_visual_review=False,
        verified_GitHub_publication=False, whole_figure_complete=False, time=time.time()))
    status(BASE/'STATUS.json', dict(status='cross_domain_scored_pending_actual_root_case_visual_publication',
        formal_predictions=512, qualification_cells_passed=8, GT_read=True,
        whole_figure_complete=False, whole_paper_complete=False, time=time.time()))


if __name__ == '__main__':
    run()
