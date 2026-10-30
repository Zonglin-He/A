"""Offline supplemental full-box fitting, a distinct capacity group from yaw-only."""
import hashlib
import json
from pathlib import Path

import numpy as np

from score_diagnostics import metric, target, statistics

R = Path(__file__).resolve().parent
OLD = R.parent / 'round2'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    pred = R / 'p3/supplement/predictions'
    seal = json.loads((pred / 'SEALED.json').read_text())
    assert seal['n'] == 54
    for name, digest in seal['files'].items():
        assert sha(pred / name) == digest
    previous = json.loads((R / 'diagnostics/RESULTS_PRIVATE.json').read_text())
    reference = {x['id']: x for x in previous['rows'] if x['arm'] == 'source'}
    out = R / 'full_refit_diagnostics'; out.mkdir(exist_ok=False)
    (out / 'SCORER_LOCK.json').write_text(json.dumps({'scorer_sha256': sha(Path(__file__)),
        'imported_scorer_sha256': sha(R / 'score_diagnostics.py'), 'prediction_seal_sha256': sha(pred / 'SEALED.json'),
        'previous_result_sha256': sha(R / 'diagnostics/RESULTS_PRIVATE.json'),
        'supplement_plan_after_historical_labels_exposed': True, 'all_supplement_predictions_sealed_before_this_score': True}, indent=2) + '\n')
    rows = []
    for cohort in ['dev', 'confirm']:
        index = OLD / (cohort + '_scoring_index_PRIVATE.json')
        prep = json.loads((OLD / 'INPUT_PREPARATION_RECEIPT.json').read_text())
        assert sha(index) == prep['cohorts'][cohort]['scoring_index_sha256']
        for sample in json.loads(index.read_text()):
            gt = target(sample)
            with np.load(pred / (sample['id'] + '.npz'), allow_pickle=False) as z:
                for arm in seal['arms']:
                    value = metric(z[arm].astype(float), gt)
                    rows.append({'id': sample['id'], 'cohort': cohort, 'platform': sample['platform'], 'scene': sample['scene'],
                                 'arm': arm, 'iou': value, 'source_iou': reference[sample['id']]['iou'],
                                 'delta_source': value - reference[sample['id']]['iou'],
                                 'capability_group': 'classical_full_center_size_yaw_refit'})
    summary, compared = statistics(rows, seal['arms'])
    result = {'scope': 'Historical54 supplementary descriptive fullrefit after initialdiagnostic. No new algorithm/threshold selection. '
                       'Same source query/crop as yaw-only; refits center/size/yaw incl z. Distinct capacity group, not TTA.',
              'n': 54, 'rows': rows, 'summaries': summary, 'comparisons': compared,
              'prediction_seal_sha256': sha(pred / 'SEALED.json')}
    (out / 'RESULTS_PRIVATE.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'n': 54, 'summaries': [s for s in summary if s['platform'] == 'all']}))


if __name__ == '__main__':
    main()
