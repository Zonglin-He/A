"""Check the four-end physical calculation independently of predictions and labels."""
from itertools import product
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'research/round6/source_validation_scoring'))
from roles_joint import affinity, decide, direction_tensor, fuse, posterior


def run():
    rng = np.random.default_rng(3287)
    worst = 0.
    for trial in range(12):
        boxes = np.column_stack((rng.normal(0, .6, (16, 3)), rng.uniform(.5, 4., (16, 3)),
                                 rng.uniform(-np.pi, np.pi, 16)))
        s, r, sd, rd = np.split(boxes, 4)
        p, cond, rev = posterior(rng.normal(size=4), rng.normal(size=(4, 4)), rng.normal(size=(4, 4)))
        ss, rr = affinity(s, sd), affinity(r, rd)
        sr, rs = affinity(s, rd), affinity(r, sd)
        np.testing.assert_allclose(ss, affinity(sd, s).T, atol=1e-11)
        post = np.zeros(4)
        for i, j, k, l in product(range(4), repeat=4):
            det = np.linalg.det([[ss[i, k], sr[i, l]], [rs[j, k], rr[j, l]]])
            post[i] += p[i] * cond[i, j] * rev[k, l] * (1 + det)
        post /= post.sum()
        winner, _, box, fraction, result = decide(p, s, r, cond, sd, rd, rev, 'ordered_full')
        np.testing.assert_allclose(result, post, atol=1e-12)
        assert winner == post.argmax() and 0 <= fraction <= 1
        assert winner == decide(p, s, r, cond, sd, rd, rev, 'ordered_raw')[0]
        assert decide(p, s, r, cond, sd, rd, rev, 'ordered_frozen_subject')[0] == p.argmax()
        worst = max(worst, float(np.max(abs(result-post))))

        def move_chart(group):
            angle = .39
            rc = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            new = group.copy()
            new[:, :2] = new[:, :2] @ rc.T + [4., -2.]
            new[:, 2] += 3.
            new[:, 6] += angle
            return new

        other = decide(p, move_chart(s), move_chart(r), cond, move_chart(sd), move_chart(rd), rev, 'ordered_full')
        np.testing.assert_allclose(other[-1], result, atol=1e-11)
        np.testing.assert_allclose(affinity(other[2][None], move_chart(box[None])), [[1]], atol=1e-10)
        swapped = sd.copy()
        swapped[:, [3, 4]] = swapped[:, [4, 3]]
        swapped[:, 6] += np.pi / 2
        np.testing.assert_allclose(direction_tensor(sd), direction_tensor(swapped), atol=1e-12)
        np.testing.assert_allclose(affinity(s, sd), affinity(s, swapped), atol=1e-12)
        degenerate = decide(p, s, r, cond, sd[:1].repeat(4, 0), sd[:1].repeat(4, 0), rev, 'ordered_full')
        np.testing.assert_allclose(degenerate[-1], p, atol=1e-12)
        np.testing.assert_array_equal(degenerate[2], s[p.argmax()])
        np.testing.assert_array_equal(fuse(s[0], sd, 1., np.zeros(4)), s[0])
    return dict(random_trials=12, explicit_determinant_max_error=worst,
                rigid_motion_equivariant=True, role_collapse_restores_input=True, axial_reparameterization=True)


if __name__ == '__main__':
    print(run())
