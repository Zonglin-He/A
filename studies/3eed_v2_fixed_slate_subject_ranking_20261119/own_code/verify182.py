"""Check live single-frame output against the complete sealed scoring path."""

import csv

import numpy as np
import torch

from ranker182 import HERE, ARMS
from ground182 import RankedGrounder
from replay182 import PREDICTIONS, RESULTS, read_predictions
from capture182 import archive_block
from capture181 import load_r180
from evaluate import DIRECTIONS
from scope import PLATFORMS, evaluation_rows, file_sha, save
from train import prepare_device, metrics


def independent_scalar_recount():
    with (RESULTS / 'anonymous_rows.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3072
    with (RESULTS / 'direction.csv').open() as stream:
        for item in csv.DictReader(stream):
            d, arm = item['direction'], item['arm']
            group = [x for x in rows if d == 'six_equal' or f"{x['source']}_to_{x['target']}" == d]
            vals = [float(x[arm]) for x in group]
            assert len(vals) == int(item['n'])
            for metric, expected in metrics(vals).items():
                assert abs(expected - float(item[metric])) < 1e-10
    with (RESULTS / 'transitions.csv').open() as stream:
        pairs = list(csv.DictReader(stream))
    for item in pairs:
        d, a, th = item['direction'], item['arm'], float(item['threshold'])
        group = [x for x in rows if d == 'six_equal' or f"{x['source']}_to_{x['target']}" == d]
        selected = np.array([float(x[a]) > th for x in group])
        prior = np.array([float(x['v1']) > th for x in group])
        outside = np.array([bool(int(x[f'outside_{a}'])) for x in group])
        four_miss = np.array([float(x['original_best']) <= th for x in group])
        rescue = selected & ~prior
        harm = prior & ~selected
        expected = dict(total_rescues=rescue.sum(), total_harms=harm.sum(),
                        rescued_by_outside=(rescue & outside).sum(), harms_by_outside=(harm & outside).sum(),
                        rescued_original_four_miss=(rescue & four_miss).sum())
        for k, n in expected.items():
            assert n == int(item[k])
    return dict(anonymous_rows=len(rows), transition_rows=len(pairs), all_threshold_counts_match=True)


def verify():
    prepare_device()
    recount = independent_scalar_recount()
    rows = evaluation_rows()
    fixed = load_r180(rows)
    prior = read_predictions()
    compared = 0
    from features import observation
    for source in PLATFORMS:
        method = RankedGrounder(source)
        for s, t in DIRECTIONS:
            if s != source:
                continue
            block = fixed[s, t]
            record = archive_block(PREDICTIONS, f'{s}_to_{t}.npz', prior)
            choices = record['chosen']
            changed = np.flatnonzero((choices[:, 4:] != choices[:, :1]).any(axis=1))
            second = next((int(i) for i in changed if i != 0), 1)
            np.testing.assert_array_equal(block['ordinals'], record['ordinals'])
            for j in (0, second):
                result = method.all_boxes(observation(rows[int(block['ordinals'][j])]))
                for i, arm in enumerate(ARMS):
                    chosen = choices[j, i]
                    expected = block['pool'][j, block['slate'][j, 6, chosen]]
                    np.testing.assert_array_equal(np.asarray(result[arm]['box']), expected)
                    assert result[arm]['retains_v1'] == (chosen == choices[j, 0])
                    compared += 1
        del method
        torch.cuda.empty_cache()
    assert compared == 96
    save(RESULTS / 'verify.json', dict(live_outputs_checked=compared, bit_exact=compared,
         recount=recount, entrypoint_sha256=file_sha(HERE / 'ground182.py'),
         predictions_seal=file_sha(PREDICTIONS / 'SEALED.json'), target_labels_read=False,
         FINAL_access=False))


if __name__ == '__main__':
    verify()
