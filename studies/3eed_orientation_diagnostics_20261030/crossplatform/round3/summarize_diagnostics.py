"""Descriptive paired and perturbation summaries; consumes sealed offline scores only."""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent


def main():
    d = json.loads((R / 'diagnostics/RESULTS_PRIVATE.json').read_text())
    rows = d['rows']; platforms = ['waymo', 'drone', 'quad']
    by = {(x['id'], x['arm']): x for x in rows}
    comparisons = [('surface_energy_orientationonly_optimization', x) for x in
                   ['PCA_standard_axisaware', 'minimum_area_exact', 'Lshape_area', 'Lshape_closeness', 'Lshape_variance']]
    comparisons += [('GT_yaw_injection', 'PCA_standard_axisaware'), ('finite_yaw_search_best', 'PCA_standard_axisaware')]
    pairs = []
    for a, b in comparisons:
        rr = [x for x in rows if x['arm'] == a]
        values = {p: [] for p in platforms}; scene_rows = []
        for p in platforms:
            for scene in sorted(set(x['scene'] for x in rr if x['platform'] == p)):
                ds = [x['iou'] - by[x['id'], b]['iou'] for x in rr if x['platform'] == p and x['scene'] == scene]
                values[p].append(float(np.mean(ds)))
                scene_rows.append({'platform': p, 'scene': scene, 'n': len(ds), 'delta': float(np.mean(ds))})
        rng = np.random.default_rng(20261030)
        boots = np.mean([rng.choice(v, size=(10000, len(v)), replace=True).mean(1) for v in values.values()], axis=0)
        delta = np.array([x['iou'] - by[x['id'], b]['iou'] for x in rr])
        pairs.append({'a': a, 'b': b, 'scope': 'descriptive fixed comparisons after prediction seal, no selector tuning',
                      'frame_mean_delta': float(delta.mean()),
                      'equalplatform_scene_macro_delta': float(np.mean([np.mean(v) for v in values.values()])),
                      'platformstratified_scene_bootstrap_95ci_descriptive': np.percentile(boots, [2.5, 97.5]).tolist(),
                      'positive': int((delta > 1e-10).sum()), 'negative': int((delta < -1e-10).sum()),
                      'same': int((abs(delta) <= 1e-10).sum()), 'scene_deltas': scene_rows})
    paired = {'n': 54, 'sceneclusters': 23,
              'platformscene_counts': {p: len(values[p]) for p in platforms}, 'pairs': pairs}
    (R / 'diagnostics/PAIRED_COMPARISONS.json').write_text(json.dumps(paired, indent=2) + '\n')
    perturb = d['P2_controlled_perturbation_rows']; output = []
    for group in sorted(set(x['perturbation'] for x in perturb)):
        subset = [x for x in perturb if x['perturbation'] == group]; parents = []
        for sid in sorted(set(x['id'] for x in subset)):
            sr = [x for x in subset if x['id'] == sid]
            parents.append({'id': sid, 'platform': sr[0]['platform'], 'scene': sr[0]['scene'], 'variants': len(sr),
                            'mean_delta': float(np.mean([x['delta_unperturbed_probe'] for x in sr])),
                            'min_delta': min(x['delta_unperturbed_probe'] for x in sr),
                            'max_delta': max(x['delta_unperturbed_probe'] for x in sr),
                            'invalid_variants': sum(not x['valid_PCA_yaw'] for x in sr)})
        output.append({'perturbation': group, 'parents_with_trace': len(parents), 'missing_no_trace_parents': 54 - len(parents),
                       'variants_not_independent_evidence': len(subset),
                       'frame_mean_of_withinparent_variant_deltas': float(np.mean([x['mean_delta'] for x in parents])),
                       'parents': parents, 'negative_tail': sorted(parents, key=lambda x: x['mean_delta'])[:5],
                       'positive_tail': sorted(parents, key=lambda x: -x['mean_delta'])[:5]})
    report = {'scope': 'All perturbation yaws were sealed GT-free; variants are not new independent frames. '
                       'P2 PCA probe min3 differs from P3 classic min8.', 'groups': output}
    (R / 'diagnostics/PERTURBATION_SUMMARY.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'paired_comparisons': len(pairs), 'perturbation_groups': len(output)}))


if __name__ == '__main__':
    main()
