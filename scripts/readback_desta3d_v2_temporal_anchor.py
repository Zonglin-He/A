"""CPU readback of all sealed episodes, raw gradients, and existing offline metrics."""
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.desta3d_v2_p0 import read, sha
from scripts.score_desta3d_v2_aux_recovery import save_once

OUT = ROOT / 'artifacts/desta3d_v2/tta_v2/target8_B1_temporal_anchor_v1'
NEW = 'calibration_alignment_temporal_anchor'
BOTH = 'calibration_alignment_output_anchor'
OLD = 'calibration_alignment'


def cosine(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den else None


def extent(values):
    values = np.asarray([x for x in values if x is not None], dtype=np.float64)
    return {'n': len(values), 'min': float(values.min()), 'median': float(np.median(values)),
            'max': float(values.max()), 'mean': float(values.mean())} if len(values) else None


def main():
    torch.set_num_threads(2)
    score = OUT / 'independent_readback_v1'
    assert read(score / 'COMPLETE.json')['report_sha'] == sha(score / 'REPORT.json')
    assert read(OUT / 'ROOT_SUMMARY_CROSSCHECK.json')['status'] == 'passed'
    seal = read(OUT / 'ALL_PREDICTIONS_SEAL.json')
    for name, digest in seal['pins'].items():
        assert sha(Path(name)) == digest, name
    rows = read(OUT / 'INPUTS.json')
    metrics = {(r['key'], r['condition']): r for r in read(score / 'QUERY_METRICS.json')}
    diagnostics = {(r['key'], r['condition'], r['arm']): r for r in read(score / 'OUTPUT_DIAGNOSTICS.json')}
    episodes = []
    max_norm_error = 0.
    for condition in ['clean', 'noise_medium', 'defocus_extreme']:
        for i, row in enumerate(rows):
            path = OUT / 'episodes' / condition / f'{i:02}'
            predictions = {arm: torch.load(path / (arm + '.pt'), map_location='cpu', weights_only=False)
                           for arm in ['Frozen', 'sourcefit_noTTA', OLD, BOTH, NEW]}
            m = metrics[(row['key'], condition)]
            steps = []
            for step in range(1, 4):
                raw = torch.load(path / f'ANCHOR_STEP_{step}.pt', map_location='cpu', weights_only=False)
                vectors = {k: v.double().numpy() for k, v in raw['raw_gradients'].items()}
                assert all(x.shape == (66816,) and np.isfinite(x).all() for x in vectors.values())
                assert 'spatial_weighted' not in vectors and set(raw['cache_audits']) <= {'event'}
                g = vectors['pre_gate']; output = np.zeros_like(g)
                for name in ['event_weighted', 'spatial_weighted']:
                    output += vectors.get(name, np.zeros_like(g))
                residual = np.max(np.abs(g + output - vectors['total_before_clip']))
                assert residual < 1e-7
                for name, vector in vectors.items():
                    err = abs(np.linalg.norm(vector) - raw['history']['gradient_norms'][name])
                    max_norm_error = max(max_norm_error, err)
                    assert err < 1e-10
                steps.append({'step': step, 'output_KL_before': raw['history']['output_KL_before'],
                    'pre_gate_norm': float(np.linalg.norm(g)),
                    'event_to_pre_gate': float(np.linalg.norm(vectors.get('event_weighted', np.zeros_like(g))) / np.linalg.norm(g)),
                    'spatial_to_pre_gate': float(np.linalg.norm(vectors.get('spatial_weighted', np.zeros_like(g))) / np.linalg.norm(g)),
                    'combined_to_pre_gate': float(np.linalg.norm(output) / np.linalg.norm(g)),
                    'event_pre_gate_cos': cosine(vectors.get('event_weighted', np.zeros_like(g)), g),
                    'spatial_pre_gate_cos': cosine(vectors.get('spatial_weighted', np.zeros_like(g)), g),
                    'combined_pre_gate_cos': cosine(output, g),
                    'clip_norm': raw['history']['clip_norm_before'],
                    'raw_sum_max_error': float(residual)})
            episodes.append({'key': row['key'], 'source': row['source'], 'condition': condition,
                'intervals': {a: p['interval'] for a, p in predictions.items()},
                'v': {a: m['arms'][a]['v'] for a in predictions},
                's': {a: m['arms'][a]['s'] for a in predictions},
                't': {a: m['arms'][a]['t'] for a in predictions},
                'new_minus_old_v_pp': 100 * (m['arms'][NEW]['v'] - m['arms'][OLD]['v']),
                'new_minus_B1_v_pp': 100 * (m['arms'][NEW]['v'] - m['arms']['sourcefit_noTTA']['v']),
                'steps': steps,
                'final': {a: diagnostics[(row['key'], condition, a)] for a in [OLD, BOTH, NEW]}})
    summary = {}
    for condition in ['clean', 'noise_medium', 'defocus_extreme']:
        ee = [x for x in episodes if x['condition'] == condition]
        ss = [s for x in ee for s in x['steps'] if s['step'] > 1]
        by_arm = {}
        for arm in [OLD, BOTH, NEW]:
            dd = [x['final'][arm] for x in ee]
            by_arm[arm] = {k: extent([v[k] for v in dd]) for k in ['time_KL_free_reference', 'fixed_prefix_coordinate_KL']}
            by_arm[arm]['event_reference_unchanged'] = sum(x['event_reference_unchanged'] for x in dd)
            by_arm[arm]['interval_changed_vs_B1'] = sum(x['interval_changed'] for x in dd)
        summary[condition] = {'episodes': len(ee), 'post_first_step_gradients': {
            k: extent([x[k] for x in ss]) for k in ['event_to_pre_gate', 'spatial_to_pre_gate', 'combined_to_pre_gate',
                                                   'event_pre_gate_cos', 'spatial_pre_gate_cos', 'combined_pre_gate_cos']},
            'final_distributions': by_arm,
            'new_time_KL_smaller_than_old': sum(x['final'][NEW]['time_KL_free_reference'] < x['final'][OLD]['time_KL_free_reference'] for x in ee),
            'new_coord_KL_smaller_than_old': sum(x['final'][NEW]['fixed_prefix_coordinate_KL'] < x['final'][OLD]['fixed_prefix_coordinate_KL'] for x in ee)}
    result = {'status': 'completed_raw_CPU_readback', 'time': time.time(), 'episodes': episodes, 'summary': summary,
              'all_72_steps_clip_triggered': sum(s['clip_norm'] > 1 for x in episodes for s in x['steps']),
              'max_numpy_gradient_norm_error': max_norm_error,
              'scope': '24 complete cases; saved weighted gradients; no new model inference, optimizer, GT read, or state selection',
              'causal_limit': 'coordinate output loss was removed; removal did not recover noise benefits; shared parameters and nonlinear readout remain possible pathways',
              'pins': {str(Path(__file__)): sha(Path(__file__)), str(score/'QUERY_METRICS.json'): sha(score/'QUERY_METRICS.json'),
                       str(score/'OUTPUT_DIAGNOSTICS.json'): sha(score/'OUTPUT_DIAGNOSTICS.json'),
                       str(OUT/'ALL_PREDICTIONS_SEAL.json'): sha(OUT/'ALL_PREDICTIONS_SEAL.json')}}
    save_once(OUT / 'ROOT_RAW_GRADIENT_AND_CASE_READBACK.json', result)
    print({'summary': summary, 'clip_triggered': result['all_72_steps_clip_triggered'], 'max_norm_error': max_norm_error})


if __name__ == '__main__':
    main()
