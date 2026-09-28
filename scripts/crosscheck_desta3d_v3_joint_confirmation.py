"""CPU seal/identity audit, then independent saved-score parent reduction.

No model forward, no new labels, and no selection. Run `seal` only after global
completion; run `summary` after the registered scorer. Outputs are write-once.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/desta3d_v3/latent_oracle_v1/joint_learnability_v1'
E = D / 'evaluation'
ARMS = ('B1', 'seed20260928', 'seed20260929')
METRICS = ('vIoU', 'sIoU', 'tIoU')


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(p, x):
    assert not p.exists(), 'Preserve prior audit'
    p.write_text(json.dumps(x, indent=2) + '\n')


def seal():
    import torch
    torch.set_num_threads(4)
    start = time.monotonic()
    out = E / 'ROOT_PRE_SCORE_AUDIT.json'
    assert not out.exists()
    done = read(E / 'COMPLETE.json')
    assert done == dict(queries=447, parents=31, predictions=1341,
                       seal_sha=sha(E / 'PREDICTIONS_SEAL.json'))
    manifest = read(E / 'PREDICTIONS_SEAL.json')
    files = manifest['files']
    assert len(files) == 2235 and manifest['predictions'] == 1341
    assert manifest['source_GT_privileged'] and not manifest['metrics_computed']
    for rel, expected in files.items():
        assert sha(E / rel) == expected, rel
    assert set(files) == {str(p.relative_to(E)) for p in (E / 'episodes').rglob('*') if p.is_file()}
    for lock in (D / 'LOCK.json', E / 'LOCK.json', D / 'SCORER_CPU_PREFLIGHT.json'):
        for name, expected in read(lock)['pins'].items():
            assert sha(Path(name)) == expected, name
    rows = read(D / 'VALIDATION_INPUTS.json')
    assert len(rows) == len({r['key'] for r in rows}) == 447
    assert len({r['source'] for r in rows}) == 31
    assert not {r['source'] for r in rows} & {r['source'] for r in read(D / 'TRAIN_INPUTS.json')}
    finals = {a: sha(D / a / 'FINAL.pt') for a in ARMS[1:]}
    adapter_sha = '4a2ef2cf87e1753fad7c0c5c1f582f499c14b0b2945a839680b296abb0d9c7eb'
    calls = {a: {} for a in ARMS}
    failures = {a: 0 for a in ARMS}
    for i, row in enumerate(rows):
        ep = E / 'episodes' / f'{i:04}'
        inp, complete = read(ep / 'INPUT.json'), read(ep / 'COMPLETE.json')
        assert complete['index'] == i and len(complete['files']) == 4
        for name, expected in complete['files'].items():
            assert expected == files[str((ep / name).relative_to(E))]
        assert inp['key'] == row['key'] and inp['source'] == row['source']
        assert inp['frame_ids'] == row['input']['frame_ids']
        for arm in ARMS:
            p = torch.load(ep / (arm + '.pt'), map_location='cpu', weights_only=False)
            assert (p['key'], p['source'], p['arm']) == (row['key'], row['source'], arm)
            assert p['frame_ids'] == inp['frame_ids']
            assert p['video_sha256'] == row['input']['video_sha256']
            assert p['preprocess'] == inp['preprocess'] and p['support'] == inp['support']
            assert p['adapter_sha'] == adapter_sha and p['mixer_sha'] == finals.get(arm)
            assert p['GT_read'] and not p['decoder_GT_prefix'] and not p['target_read']
            assert p['optimizer_steps'] == 0
            inj = p['injection']
            assert inj['same_field_both_passes'] and inj['common_F_sha'] == inp['support']['visual_grid']
            assert math.isfinite(inj['relative_norm']) and 0 <= inj['relative_norm'] <= read(D / 'CONFIG.json')['radius'] + 2e-6
            if arm == 'B1':
                assert inj['delta_norm'] == inj['relative_norm'] == 0
                assert inj['corrected_F_sha'] == inj['common_F_sha']
            if inj['calls'] == ['event']:
                assert not p['format_ok']
            else:
                assert inj['calls'] == ['event', 'spatial']
            label = '+'.join(inj['calls'])
            calls[arm][label] = calls[arm].get(label, 0) + 1
            failures[arm] += not p['format_ok']
    write(out, dict(status='passed', queries=447, parents=31, predictions=1341,
                    sealed_files=2235, final_hashes=finals, actual_call_counts=calls,
                    format_failures=failures, physical_metadata_and_saved_hashes_checked=True,
                    pixel_redecode=False, scope='Saved identities and hashes; no new inference or GT lookup. Source GT already used as privilege.',
                    seal_sha=sha(E / 'PREDICTIONS_SEAL.json'), auditor_sha=sha(Path(__file__)),
                    CPU_seconds=time.monotonic()-start))
    print(json.dumps(read(out)))


def summary():
    import numpy as np
    start = time.monotonic()
    dest = E / 'independent_readback_v1'
    out = dest / 'ROOT_SUMMARY_CROSSCHECK.json'
    assert not out.exists()
    report = read(dest / 'REPORT.json')
    assert read(dest / 'COMPLETE.json')['report_sha'] == sha(dest / 'REPORT.json')
    assert read(E / 'ROOT_PRE_SCORE_AUDIT.json')['status'] == 'passed'
    rows = {a: report['arms'][a]['rows'] for a in ARMS}
    parents = sorted({r['source'] for r in rows['B1']})
    assert len(parents) == 31
    errors = []

    def close(x, y):
        err = float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
        errors.append(err)
        assert err < 1e-8, (x, y, err)

    matrices = {}
    for arm in ARMS:
        assert [(r['key'], r['source']) for r in rows[arm]] == [(r['key'], r['source']) for r in rows['B1']]
        matrices[arm] = np.asarray([[sum(r['metrics'][m] for r in rows[arm] if r['source'] == p) /
                                    sum(r['source'] == p for r in rows[arm]) for m in METRICS] for p in parents])
        for j, m in enumerate(METRICS):
            close(matrices[arm][:, j].mean(), report['arms'][arm]['summary']['parent_macro'][m])
            close(sum(r['metrics'][m] for r in rows[arm])/447, report['arms'][arm]['summary']['query_macro'][m])
        close(sum(not r['format_ok'] for r in rows[arm]), report['arms'][arm]['summary']['format_failures'])
    parent_deltas, retained, gates = {}, {}, {}
    for arm in ARMS[1:]:
        delta = matrices[arm] - matrices['B1']
        parent_deltas[arm] = {p: {m: float(100*delta[i, j]) for j, m in enumerate(METRICS)} for i, p in enumerate(parents)}
        for j, m in enumerate(METRICS):
            d = delta[:, j]; c = report['comparisons'][arm][m]
            draws = np.random.default_rng(20260927).integers(0, 31, (10000, 31))
            ci = np.quantile(d[draws].mean(axis=1), [.025, .975])*100
            close(d.mean()*100, c['mean_delta_pp']); close(ci, c['bootstrap_ci95_pp'])
            for p, value in zip(parents, d): close(value*100, c['parent_delta_pp'][p])
            close((d > 1e-10).sum(), c['positive_parents']); close((d < -1e-10).sum(), c['negative_parents'])
            close((np.abs(d) <= 1e-10).sum(), c['zero_parents']); close((d < -.05).sum(), c['severe_loss_below_minus5pp'])
            close(d.min()*100, c['worst_delta_pp']); close(d.max()*100, c['best_delta_pp'])
        retained[arm] = {}
        for m in ('vIoU', 'tIoU'):
            good = {r['key'] for r in rows['B1'] if r['metrics'][m] > .5}
            kept = {r['key'] for r in rows[arm] if r['key'] in good and r['metrics'][m] > .5}
            retained[arm][m] = dict(eligible=len(good), retained=len(kept), lost_keys=sorted(good-kept))
            for name in ('eligible', 'retained'): close(retained[arm][m][name], report['retention'][m][arm][name])
        gates[arm] = dict(v_positive=bool(delta[:, 0].mean()>0),
                         v_lower_CI_positive=bool(np.quantile(delta[draws, 0].mean(axis=1), .025)>0),
                         s_nonnegative=bool(delta[:, 1].mean()>=0), t_nonnegative=bool(delta[:, 2].mean()>=0),
                         no_severe_parent_v_harm=bool(not (delta[:, 0]<-.05).any()),
                         native_good_retained=all(r['eligible']==r['retained'] for r in retained[arm].values()))
        assert gates[arm] == report['gate'][arm]
    qualified = all(all(g.values()) for g in gates.values())
    assert qualified == report['qualified']
    write(out, dict(status='passed', checks=len(errors), max_abs=max(errors), parent_deltas=parent_deltas,
                    retention=retained, gate=gates, qualified=qualified, report_sha=sha(dest/'REPORT.json'),
                    auditor_sha=sha(Path(__file__)), CPU_seconds=time.monotonic()-start,
                    scope='Independent saved scalar aggregation; no new label read, inference or selection.'))
    print(json.dumps({k:v for k,v in read(out).items() if k not in ('parent_deltas','retention')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['seal', 'summary'])
    args = parser.parse_args()
    globals()[args.mode]()
