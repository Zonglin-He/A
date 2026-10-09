"""Independent CPU audit of sealed cross-domain native figures; portable scalar mode."""
import argparse
import bisect
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/stvg_motivation_cross_domain_v6'
PUBLIC = ROOT / 'results/stvg_motivation_cross_domain/2026-10-09'
DIRECTIONS = ('vidstg_to_hc2', 'hc2_to_vidstg')
MODELS = ('tastvg', 'tubedetr')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(p, d):
    with Path(p).open('x') as f:
        json.dump(d, f, indent=2, allow_nan=False)
        f.write('\n')


def geometry(z, row, truth, span, hc2):
    """Separate scalar interpolation/IoU implementation; no producer metric helper."""
    fmt = z['format_valid']
    a, b = map(int, z['interval']) if fmt and z['temporal_valid'] else (0, 0)
    g, h = map(int, span)
    overlap = max(0, min(b, h) - max(a, g)) if b > a else 0
    union_t = max(0, b-a) + max(0, h-g) - overlap
    tiou = overlap / union_t if union_t else 0.
    ids = z['box_frame_ids']
    boxes = []
    if fmt and z['spatial_valid']:
        assert ids == sorted(set(ids)) and len(ids) == len(z['boxes'])
        scales = [row['input']['width'], row['input']['height']] * 2
        for b0 in z['boxes']:
            box = [float(x)*float(s) for x, s in zip(b0, scales)]
            assert np.isfinite(box).all()
            boxes.append([max(0., x) for x in box] if hc2 else box)
    values = []; covered = 0; invalid = 0
    for fid in sorted(truth):
        value = 0.
        if boxes and ids[0] <= fid <= ids[-1]:
            covered += 1
            k = bisect.bisect_right(ids, fid)-1
            if ids[k] == fid:
                p = boxes[k]
            else:
                ratio = (fid-ids[k]) / (ids[k+1]-ids[k])
                p = [x+ratio*(y-x) for x, y in zip(boxes[k], boxes[k+1])]
            t = truth[fid]
            if p[2] <= p[0] or p[3] <= p[1]:
                invalid += 1
            else:
                inter = max(0., min(p[2], t[2])-max(p[0], t[0])) * max(0., min(p[3], t[3])-max(p[1], t[1]))
                area_p = (p[2]-p[0])*(p[3]-p[1])
                area_t = max(0., t[2]-t[0])*max(0., t[3]-t[1])
                den = area_p+area_t-inter
                value = inter/den if den > 0 else 0.
        values.append(value)
    assert values
    return dict(tIoU=tiou, sIoU=sum(values)/len(values),
        spatial_coverage=covered/len(values), uncovered_GT_frames=len(values)-covered,
        invalid_GT_box_frames=invalid, GT_frames=len(values))


def independent_summary(rows, statistics):
    comparisons = 0
    def equal(a, b):
        nonlocal comparisons
        aa = np.asarray(a, dtype=float); bb = np.asarray(b, dtype=float)
        assert aa.shape == bb.shape and np.allclose(aa, bb, rtol=0, atol=1e-10), (a, b)
        comparisons += int(aa.size)
    for direction in DIRECTIONS:
        categories = {}
        for model in MODELS:
            rr = sorted((x for x in rows if x['direction'] == direction and x['model'] == model), key=lambda x: x['parent'])
            assert len(rr) == 128 and [x['parent'] for x in rr] == list(range(128))
            categories[model] = np.array([0 if x['tIoU'] > .5 and x['sIoU'] > .5 else
                1 if x['tIoU'] > .5 else 2 if x['sIoU'] > .5 else 3 for x in rr])
        sample = np.random.default_rng(20261009).integers(0, 128, size=(10000, 128))
        boot = {}; pct = {}
        for model in MODELS:
            onehot = np.eye(4, dtype=float)[categories[model]]
            boot[model] = onehot[sample].sum(axis=1) / 128 * 100
            counts = onehot.sum(axis=0).astype(int)
            pct[model] = counts / 128 * 100
            saved = statistics['directions'][direction]['models'][model]
            equal(saved['N'], 128); equal(saved['counts'], counts); equal(saved['percent'], pct[model])
            equal(saved['ci95_percent'], np.percentile(boot[model], [2.5, 97.5], axis=0).T)
            equal(saved['contrast_Tplus_Sminus_minus_Tminus_Splus_pp'], pct[model][1]-pct[model][2])
            equal(saved['contrast_ci95_pp'], np.percentile(boot[model][:, 1]-boot[model][:, 2], [2.5, 97.5]))
            assert saved['threshold'] == .5 and saved['strict_greater_than']
        pair = statistics['directions'][direction]['paired_between_backbones']['tastvg minus tubedetr']
        equal(pair['percent_difference'], pct['tastvg']-pct['tubedetr'])
        equal(pair['ci95_paired_parent_bootstrap'], np.percentile(boot['tastvg']-boot['tubedetr'], [2.5, 97.5], axis=0).T)
    assert statistics['bootstrap_draws'] == 10000 and statistics['seed'] == 20261009
    return dict(status='pass', native_rows=512, sources_per_direction=128,
        scalar_comparisons=comparisons, bootstrap_draws=10000,
        all_four_categories_retained=True, paired_parent_unit=True,
        no_model_GT_media_prediction_payload_access=True)


def root():
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.stvg_motivation_cross_domain_common_v6 import check_global_seal, verify, digest, CHECKPOINTS
    from scripts.score_stvg_motivation_cross_domain_v6 import read_truths
    begin = time.perf_counter()
    lock = verify(); barrier = check_global_seal()
    assert sha(BASE/'ROOT_RUNTIME.json')
    for rel, expected in read(BASE/'ROOT_RUNTIME.json')['code'].items():
        assert sha(ROOT/rel) == expected, rel
    exposure = read(BASE/'GT_EXPOSURE.json')
    assert exposure['time'] >= barrier['time']
    assert exposure['global_barrier_sha256'] == sha(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    rows = read(BASE/'SCALAR_ROWS.json'); stats = read(BASE/'QUADRANT_STATISTICS.json')
    indexed = {(r['direction'], r['model'], r['parent']): r for r in rows}
    assert len(indexed) == len(rows) == 512
    portable = independent_summary(rows, stats)
    max_error = 0.; total_gt_frames = 0; native_bytes = 0; videos = {}; checkpoint_bytes = 0
    native_seconds = {}; qualification_seconds = {}; gtproof = {}
    for direction in DIRECTIONS:
        roster = read(BASE/direction/'ROSTER.json'); rr = roster['rows']
        assert len(rr) == 128 and len({r['source'] for r in rr}) == 128
        assert digest(rr[0]['input'])
        dense, spans, provenance = read_truths(direction, roster)
        assert provenance == stats['GT_provenance'][direction]
        gtproof.update(provenance)
        for row in rr:
            p = Path(row['input']['video_path']); p = p if p.is_absolute() else ROOT/p
            videos[str(p)] = row['input']['video_sha256']
        for model in MODELS:
            cell = BASE/direction/model
            q = read(cell/'QUALIFICATION.json')
            assert q['status'] == 'pass' and q['actual_GPU_qualification_cells'] == 2
            assert q['parameter_and_buffer_sha256_before'] == q['parameter_and_buffer_sha256_after']
            assert q['parameter_versions_unchanged'] and q['parameter_updates'] == 0 and not q['optimizer_created']
            assert q['accepted_formal_predictions'] == 0 and not q['GT_read'] and q['native_parity'] and q['repeated_native_outputs']
            assert all(x['repeated_native_bitwise'] and not x['GT_read'] for x in q['cells'])
            source = roster['source_dataset']; relative, expected = CHECKPOINTS[model][source]
            assert sha(ROOT/relative) == expected == q['source_checkpoint_sha256']
            checkpoint_bytes += (ROOT/relative).stat().st_size
            qualification_seconds[direction+'/'+model] = q['seconds']
            elapsed = 0.
            for row in rr:
                p = cell/'predictions'/f"{row['ordinal']:05}.json"; z = read(p)
                saved = indexed[(direction, model, row['ordinal'])]
                assert saved['native_output_sha256'] == sha(p)
                expected_values = geometry(z, row, dense[row['ordinal']], spans[row['ordinal']], direction == 'vidstg_to_hc2')
                for name, value in expected_values.items():
                    err = abs(float(saved[name])-float(value)); max_error = max(max_error, err)
                    assert err < 1e-10, (direction, model, row['ordinal'], name, err)
                total_gt_frames += expected_values['GT_frames']; native_bytes += p.stat().st_size
                elapsed += z['seconds']
            native_seconds[direction+'/'+model] = elapsed
    video_bytes = 0
    for path, expected in videos.items():
        p = Path(path); assert sha(p) == expected, path; video_bytes += p.stat().st_size
    for rel, expected in gtproof.items():
        assert sha(ROOT/rel) == expected, rel
    elapsed = time.perf_counter()-begin
    receipt = dict(status='pass', scope='all sealed cross-domain native sources, scalar geometry, four quadrants and paired bootstrap',
        native_predictions=512, actual_GPU_qualification_cells=8, unique_parent_sources=256,
        native_prediction_bytes=native_bytes, unique_video_files=len(videos), opaque_video_bytes=video_bytes,
        opaque_checkpoint_bytes=checkpoint_bytes, source_rosters_and_common_pixels_checked=True,
        full_GT_frames_evaluated=total_gt_frames, independent_component_comparisons=3072,
        max_independent_geometry_error=max_error, portable_scalar_audit=portable,
        all_formats_and_negative_findings_retained=True, GT_after_all_512_sealed=True,
        CPU_seconds=elapsed, native_prediction_seconds_by_cell=native_seconds,
        qualification_seconds_by_cell=qualification_seconds,
        GPU_timing_scope='saved native worker timings; excludes process launch and initial loading outside each timed native call',
        no_new_GPU_model_calls=True, no_OPD_P1_payload_access=True,
        independent_GT_parser_claim=False, runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),
        root_runtime_sha256=sha(BASE/'ROOT_RUNTIME.json'),
        global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
        scalar_rows_sha256=sha(BASE/'SCALAR_ROWS.json'), quadrants_sha256=sha(BASE/'QUADRANT_STATISTICS.json'),
        GT_read=True, whole_figure_complete=False, whole_paper_complete=False, time=time.time())
    write(BASE/'ROOT_AUDIT.json', receipt); write(BASE/'PORTABLE_SCALAR_AUDIT.json', portable)
    print(json.dumps({k:receipt[k] for k in ['status','native_predictions','unique_parent_sources','full_GT_frames_evaluated','max_independent_geometry_error','CPU_seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', action='store_true'); parser.add_argument('--public', type=Path, default=PUBLIC)
    args = parser.parse_args()
    if args.root:
        root()
    else:
        print(json.dumps(independent_summary(read(args.public/'SCALAR_ROWS.json'), read(args.public/'QUADRANT_STATISTICS.json'))))
