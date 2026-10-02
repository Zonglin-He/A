"""Analytic tests for rounding, no-op, and cosine/cancellation semantics."""
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from vg_tta.tastvg_saved_write_accumulation_v1 import delta, at_origin, geometry


class Reconstruction(unittest.TestCase):
    def test_saved_float32_prefix_is_exact(self):
        origin = {'w': torch.tensor([1., 1e-8, -1., 1e8], dtype=torch.float32)}
        writes = []
        previous = origin
        for increment in [.001, -.0002, .008, -.07, .01, -.2, .02, .07]:
            post = {'w': previous['w'] + increment}
            writes.append(delta(previous, post))
            self.assertTrue(torch.equal(at_origin(origin, writes)['w'], post['w']))
            previous = post

    def test_no_op_is_source_even_after_prior_writes(self):
        origin = {'w': torch.tensor([1., 2.])}
        first = {'w': torch.tensor([2., 3.])}
        no_op = delta(first, first)
        self.assertTrue(torch.equal(at_origin(origin, [no_op])['w'], origin['w']))

    def test_orthogonal_and_cancelling_writes(self):
        result = geometry([{'w': torch.tensor([1., 0.])},
                           {'w': torch.tensor([0., 1.])},
                           {'w': torch.tensor([-1., 0.])}])
        self.assertEqual(result['cosine_gram'][0][1], 0.)
        self.assertEqual(result['cosine_gram'][0][2], -1.)
        self.assertAlmostEqual(result['prefix_cancellation'][-1], 1/3)

    def test_zero_norms_are_undefined(self):
        result = geometry([{'w': torch.zeros(2)}, {'w': torch.tensor([1., 0.])}])
        self.assertIsNone(result['cosine_gram'][0][0])
        self.assertIsNone(result['consecutive_cosines'][0])
        self.assertIsNone(result['prefix_cancellation'][0])
        self.assertEqual(result['prefix_cancellation'][1], 1.)


if __name__ == '__main__':
    unittest.main()
