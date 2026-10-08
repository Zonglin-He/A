"""CPU contracts for the audit supplement; no model, media, labels, or CUDA."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vg_tta.decota_spatial_opd_precision_audit_revision002 import softmax_check


def run():
    rng = np.random.default_rng(20261008)
    regimes = [
        np.full((4, 32), 0.5, dtype=np.float32),
        rng.uniform(0, 1, (4, 32)).astype(np.float32),
        (0.9 + rng.uniform(-1e-6, 1e-6, (4, 32))).astype(np.float32),
        np.tile(np.linspace(0, 1, 32, dtype=np.float32), (4, 1)),
    ]
    accepted = rejected = 0
    for tau in (0.05, 0.1, 0.25, 0.5):
        for used in regimes:
            inverse = np.float32(np.float32(1) / np.float32(tau))
            logits = (used * inverse).astype(np.float32)
            weights = torch.softmax(torch.from_numpy(logits), -1).numpy().astype(float)
            assert softmax_check(used, weights, tau)["matched_precision_absolute_threshold"] == 3e-7
            accepted += 1
            # A normalized but wrong distribution must still be rejected.
            bad = weights.copy()
            order = np.argsort(bad[0])
            first, second = int(order[-1]), int(order[-2])
            change = min(1e-4, bad[0, first] / 2)
            assert change > 3e-7
            bad[0, first] -= change
            bad[0, second] += change
            assert np.max(abs(bad.sum(1) - weights.sum(1))) < 1e-15
            try:
                softmax_check(used, bad, tau)
            except AssertionError:
                rejected += 1
            else:
                raise AssertionError("Incorrect normalized probabilities were accepted")
    assert accepted == rejected == 16
    assert not torch.cuda.is_initialized()
    return dict(status="pass", synthetic_CPU_precision_regimes=accepted,
                rejected_wrong_normalized_probabilities=rejected,
                CUDA_initialized=False, model_or_GT_access=False,
                private_GPU_replay_not_claimed=True)


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
