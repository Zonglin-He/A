"""CPU audit of sealed HC64 capacity replay and already-scored outcomes.

This script never imports a model, opens annotations, or scores predictions.
It requires the existing private receipts, state traces and prediction arrays.
Only anonymous aggregate diagnostics are written to the public output.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve()
ARMS = ['Full', 'FullR4', 'PCA4', 'PCA4R4']
METRICS = {'full_vIoU': 'v', 'unobserved_actual_sIoU': 'unobserved_actual_sIoU'}
EPS_PP = 1e-8


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def within(parent, relative):
    path = (Path(parent) / relative).resolve()
    if not path.is_relative_to(Path(parent).resolve()):
        raise ValueError('Input path escapes its declared artifact directory')
    return path


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and len(values) and np.isfinite(values).all()
    return dict(queries=len(values), mean=float(values.mean()), median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()),
                p10=float(np.quantile(values, .1)), p25=float(np.quantile(values, .25)),
                p75=float(np.quantile(values, .75)), p90=float(np.quantile(values, .9)))


def deltas_summary(deltas, denominator):
    deltas = np.asarray(deltas, dtype=np.float64)
    return dict(queries=len(deltas), mean_delta_pp=float(deltas.mean()),
                contribution_to_all64_mean_pp=float(deltas.sum() / denominator),
                improved=int((deltas > EPS_PP).sum()), declined=int((deltas < -EPS_PP).sum()),
                unchanged=int((np.abs(deltas) <= EPS_PP).sum()),
                drops_over5pp=int((deltas < -5).sum()),
                positive_contribution_to_all64_mean_pp=float(deltas[deltas > 0].sum() / denominator),
                negative_contribution_to_all64_mean_pp=float(deltas[deltas < 0].sum() / denominator))


def audit(output):
    import torch
    torch.set_num_threads(1)
    trial = ROOT / 'outputs/state_capacity_hc64'
    parent = ROOT / 'outputs/state_opd_hc128'
    old = ROOT / 'outputs/layout_reward_dev'
    scored_path = ROOT / 'results/state_capacity_hc64/hc64.json'
    scored = read(scored_path)
    seal_path = trial / 'sealed.json'
    seal, lock = read(seal_path), read(trial / 'lock.json')
    parent_lock = read(parent / 'lock.json')
    score_receipt = read(trial / 'score_receipt.json')
    assert seal['GT_read'] is False and seal['queries'] == scored['queries'] == scored['parents'] == 64
    assert seal['fingerprint'] == lock['fingerprint'] == scored['proof']['fingerprint']
    assert seal['lock_sha256'] == sha(trial / 'lock.json') == scored['proof']['lock_sha256']
    assert scored['proof']['seal_sha256'] == sha(seal_path)
    assert lock['parent_fingerprint'] == parent_lock['fingerprint']
    keys = lock['ids']
    assert len(keys) == len(set(keys)) == 64 and keys == [r['id'] for r in parent_lock['inputs'][:64]]
    assert set(seal['files']) == set(keys)
    rows = scored['queries_anonymous']
    assert [r['query'] for r in rows] == ['H%03d' % i for i in range(1, 65)]
    assert score_receipt['proof'] == scored['proof']
    assert score_receipt['metrics_sha256'] == sha(trial / 'metrics.json')
    private_rows = read(trial / 'metrics.json')['rows']
    assert [r['id'] for r in private_rows] == keys
    assert [r['values'] for r in private_rows] == [r['values'] for r in rows]
    del private_rows
    original_inputs = {item['id']: item for item in parent_lock['inputs']}
    norms = {name: [] for name in ARMS}
    energies, outside_energies, radial_energies = [], [], []
    same_projection_states = same_projection_predictions = full_matches = receipt_matches = 0
    geometry_replayed = 0
    max_receipt_error = 0.
    for key in keys:
        entry = seal['files'][key]
        prediction_path = within(trial, entry['path'])
        receipt_path = prediction_path.with_suffix('.json')
        assert sha(prediction_path) == entry['sha256'] and sha(receipt_path) == entry['receipt_sha256']
        receipt = read(receipt_path)
        trace_path = within(trial, receipt['trace_path'])
        assert receipt['sha256'] == entry['sha256'] and sha(trace_path) == receipt['trace_sha256']
        assert receipt['fingerprint'] == seal['fingerprint'] and receipt['GT_read'] is False
        assert receipt['geometry_replay_bit_exact'] is True
        item = original_inputs[key]
        reference_path = within(old, item['reference_path'])
        assert receipt['reference_sha256'] == item['reference_sha256'] == sha(reference_path)
        trace = torch.load(trace_path, map_location='cpu', weights_only=False)
        theta, basis, variants = trace['theta'], trace['basis'], trace['variants']
        assert theta.shape == (8, 256) and basis.shape == (256, 4) and set(variants) == set(ARMS)
        assert torch.equal(trace['state_path'][-1], theta) and torch.count_nonzero(trace['state_path'][0]) == 0
        assert torch.allclose(basis.T @ basis, torch.eye(4, dtype=basis.dtype), atol=1e-10, rtol=0)
        assert hashlib.sha256(basis.numpy().tobytes()).hexdigest() == trace['basis_info']['basis_sha256']
        projection = (theta.double() @ basis @ basis.T).float()
        assert torch.equal(variants['Full'], theta) and torch.equal(variants['PCA4'], projection)
        def cap(value):
            norm = float(value.norm())
            return value * min(1., 4. * (1. - 1e-6) / norm) if norm else value.clone()
        assert torch.equal(variants['FullR4'], cap(theta)) and torch.equal(variants['PCA4R4'], cap(projection))
        norm = float(theta.norm())
        projected_norm = float(projection.norm())
        retained = float(projection.square().sum() / theta.square().sum())
        max_receipt_error = max(max_receipt_error, abs(norm - receipt['theta_L2']),
                                abs(projected_norm - receipt['PCA4_L2']), abs(retained - receipt['retained_state_energy']))
        assert max_receipt_error < 1e-7
        receipt_matches += 1
        for name in ARMS:
            norms[name].append(float(variants[name].norm()))
        energies.append(retained)
        outside_energies.append(float((theta.double() - projection.double()).square().sum() / theta.double().square().sum()))
        radial_energies.append(float(variants['FullR4'].double().square().sum() / theta.double().square().sum()))
        same_projection_states += int(torch.equal(variants['PCA4'], variants['PCA4R4']))
        with np.load(prediction_path, allow_pickle=False) as prediction, np.load(reference_path, allow_pickle=False) as reference:
            assert np.array_equal(prediction['Full'], prediction['GeometryOPD'])
            assert np.array_equal(prediction['Full'], reference['GeometryOPD'])
            assert np.array_equal(prediction['ERefStrong'], reference['E_Ref'])
            full_matches += 1
            same_projection_predictions += int(np.array_equal(prediction['PCA4'], prediction['PCA4R4']))
        geometry_replayed += 1
        del trace, theta, basis, variants, projection
    assert geometry_replayed == full_matches == receipt_matches == 64
    assert not torch.cuda.is_initialized()
    original_v_delta = np.asarray([100 * (r['values']['GeometryOPD']['v'] - r['values']['ERefStrong']['v']) for r in rows])
    masks = dict(positive=original_v_delta > EPS_PP, negative=original_v_delta < -EPS_PP,
                 unchanged=np.abs(original_v_delta) <= EPS_PP)
    assert {k: int(v.sum()) for k, v in masks.items()} == dict(positive=46, negative=17, unchanged=1)
    summaries, groups = {}, {}
    max_saved_mean_error = 0.
    for name in ['GeometryOPD'] + ARMS:
        values = {}
        for label, field in METRICS.items():
            current = np.asarray([r['values'][name][field] for r in rows], dtype=float)
            base = np.asarray([r['values']['ERefStrong'][field] for r in rows], dtype=float)
            reference = np.asarray([r['values']['GeometryOPD'][field] for r in rows], dtype=float)
            assert np.isfinite(current).all() and np.isfinite(base).all()
            mean_key = 'm_vIoU' if field == 'v' else 'm_sIoU_unobserved_actual'
            max_saved_mean_error = max(max_saved_mean_error, abs(float(current.mean()) - scored['means'][name][mean_key]))
            values[label] = dict(mean_percent=float(100 * current.mean()),
                versus_ERef=deltas_summary(100 * (current - base), 64),
                versus_saved_geometry=deltas_summary(100 * (current - reference), 64))
        summaries[name] = values
    assert max_saved_mean_error < 1e-12
    for group, mask in masks.items():
        indices = np.flatnonzero(mask)
        result = dict(queries=len(indices), grouping='posthoc sign of saved GeometryOPD minus ERefStrong full vIoU',
                      full_state_L2=distribution(np.asarray(norms['Full'])[mask]),
                      PCA4_state_L2=distribution(np.asarray(norms['PCA4'])[mask]),
                      retained_state_energy=distribution(np.asarray(energies)[mask]),
                      methods={})
        for name in ['GeometryOPD'] + ARMS:
            result['methods'][name] = {}
            for label, field in METRICS.items():
                delta = np.asarray([100 * (rows[i]['values'][name][field] - rows[i]['values']['ERefStrong'][field]) for i in indices])
                ref_delta = np.asarray([100 * (rows[i]['values']['GeometryOPD'][field] - rows[i]['values']['ERefStrong'][field]) for i in indices])
                entry = deltas_summary(delta, 64)
                entry['saved_geometry_contribution_to_all64_mean_pp'] = float(ref_delta.sum() / 64)
                entry['lost_saved_geometry_contribution_to_all64_mean_pp'] = float((ref_delta - delta).sum() / 64)
                entry['fraction_of_saved_group_signed_gain_retained'] = float(delta.sum() / ref_delta.sum()) if abs(ref_delta.sum()) > 1e-12 else None
                result['methods'][name][label] = entry
        groups[group] = result
    result = dict(status='completed_CPU_audit_of_saved_capacity_replay_and_existing_scores', queries=64,
        scope='Same fixed HC64 parent-video prefix; one query per parent; no new GT scoring or model calls',
        state_norms={k: distribution(v) for k, v in norms.items()},
        capacity=dict(radius=4., clipped_radius=4. * (1. - 1e-6),
                      Full_norm_greater4=int(sum(v > 4 for v in norms['Full'])),
                      PCA4_norm_greater4=int(sum(v > 4 for v in norms['PCA4'])),
                      PCA4_and_PCA4R4_state_bit_exact_queries=same_projection_states,
                      PCA4_and_PCA4R4_prediction_bit_exact_queries=same_projection_predictions,
                      retained_PCA4_squared_state_energy=distribution(energies),
                      outside_PCA4_squared_state_energy=distribution(outside_energies),
                      retained_FullR4_squared_state_energy=distribution(radial_energies)),
        saved_task_metrics=summaries, saved_geometry_posthoc_groups=groups,
        validation=dict(all64_receipts_and_trace_hashes_verified=True,
                        public_rows_equal_existing_scored_private_rows=True,
                        original_geometry_replay_bit_exact_queries=geometry_replayed,
                        Full_arrays_equal_original_geometry_queries=full_matches,
                        state_receipt_scalar_checks=receipt_matches,
                        state_receipt_max_abs_error=max_receipt_error,
                        saved_mean_max_abs_error=max_saved_mean_error,
                        new_GT_scoring=False, GT_annotation_reads=0, model_calls=0, decoder_calls=0,
                        expert_calls=0, CUDA_initialized=False),
        interpretation=[
            'Projection and radius restriction are applied to the same already-fitted geometric state; these are capacity interventions, not newly optimized PCA/limited-radius methods.',
            'State energy is squared Euclidean content-state norm, not task quality or explained visual-feature variance.',
            'Full-versus-PCA4 and Full-versus-FullR4 compare decoder-replayed output consequences. Their losses are not an additive decomposition because the decoder is nonlinear and both restrictions act on the same state.',
            'The posthoc positive/negative/unchanged groups use saved full-vIoU signs; their unobserved-space signs may differ, including a temporal-zero full-vIoU query with a nonzero spatial change.',
            'A zero PCA4-to-PCA4R4 difference means the radius is inactive after projection, not that radius4 has no effect in the full 2048-dimensional state space.'
        ],
        provenance=dict(saved_public_scores_path='results/state_capacity_hc64/hc64.json',
                        saved_public_scores_sha256=sha(scored_path),
                        capacity_seal_sha256=sha(seal_path), capacity_lock_sha256=sha(trial / 'lock.json'),
                        script_sha256=sha(__file__)))
    assert sha(seal_path) == scored['proof']['seal_sha256']
    assert sha(trial / 'lock.json') == scored['proof']['lock_sha256']
    assert score_receipt['proof']['fingerprint'] == seal['fingerprint']
    assert sha(ROOT / 'outputs/formal/state.json') == lock['formal_state_sha256']
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(destination)
    print(json.dumps(dict(status=result['status'], capacity=result['capacity'],
                          saved_task_metrics=result['saved_task_metrics'], validation=result['validation']), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/state_capacity_hc64/diagnostics.json')
    args = parser.parse_args()
    if not args.run:
        parser.error('Explicit --run is required to audit existing private artifacts')
    audit(args.output)


if __name__ == '__main__':
    main()
