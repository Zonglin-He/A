"""CPU-only geometry and learned-pool contracts; no models or GT inputs."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "vg_tta/tastvg_spatial_conditioned_temporal_p1_v1.py"
spec = importlib.util.spec_from_file_location("p1_geometry_pool", MODULE)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class Geometry(unittest.TestCase):
    def test_import_has_no_framework(self):
        code = (
            "import importlib.util,sys;"
            f"s=importlib.util.spec_from_file_location('p1',{str(MODULE)!r});"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "assert not any(x in sys.modules for x in ('torch','tensorflow','jax'))"
        )
        subprocess.run([sys.executable, "-B", "-c", code], check=True)

    def test_fractional_area_and_row_major(self):
        # Half of patch row=2,col=3 on the real 14px grid.
        m = c.fractional_mask([42, 28, 49, 42], 336, 336)
        self.assertEqual(m.shape, (24, 24))
        self.assertEqual(m[2, 3], 0.5)
        self.assertEqual(m.sum(), 0.5)
        self.assertEqual(np.flatnonzero(m.reshape(-1)).tolist(), [2 * 24 + 3])

    def test_fractional_total_area(self):
        b = [10.5, 17.25, 209.75, 143.5]
        m = c.fractional_mask(b, 672, 168)
        self.assertAlmostEqual(m.mean(), (b[2] - b[0]) * (b[3] - b[1]) / (672 * 168))
        self.assertTrue(np.all((m >= 0) & (m <= 1)))

    def test_squash_preserves_off_center_support(self):
        # First source-image quarter remains first grid quarter, not a crop.
        m = c.fractional_mask([0, 0, 200, 100], 800, 100)
        np.testing.assert_array_equal(m[:, :6], 1)
        np.testing.assert_array_equal(m[:, 6:], 0)
        self.assertEqual(m.mean(), 0.25)

    def test_outside_clipping_and_full_mask(self):
        np.testing.assert_array_equal(c.fractional_mask([-10, -9, 336, 400], 336, 336), 1)
        m = c.fractional_mask([-7, -14, 7, 14], 336, 336)
        self.assertEqual(m[0, 0], 0.5)
        self.assertEqual(m.sum(), 0.5)

    def test_invalid_empty_outside_box(self):
        for box in ([0, 0, 0, 1], [3, 2, 1, 4], [0, 0, np.nan, 2],
                    [400, 0, 500, 40], [1, 2], None):
            with self.subTest(box=box):
                np.testing.assert_array_equal(c.fractional_mask(box, 336, 336), 0)

    def test_invalid_metadata_rejected(self):
        for width, height, grid in ((0, 2, 24), (1, np.inf, 24), (1, 2, 0), (1, 2, 2.5)):
            with self.assertRaises(ValueError):
                c.fractional_mask([0, 0, 1, 1], width, height, grid)


class PoolContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from torch import nn

        torch.set_num_threads(2)
        cls.torch = torch

        class ToyPool(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2
                self.probe = nn.Parameter(torch.zeros(1, 1, 4))
                self.attn = nn.MultiheadAttention(4, 2, batch_first=True)
                self.layernorm = nn.LayerNorm(4)
                self.mlp = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
                # Zero Q/K and identity V/out make attention depend only on prior.
                with torch.no_grad():
                    self.attn.in_proj_weight.zero_()
                    self.attn.in_proj_weight[8:].copy_(torch.eye(4))
                    self.attn.in_proj_bias.zero_()
                    self.attn.out_proj.weight.copy_(torch.eye(4))
                    self.attn.out_proj.bias.zero_()
                    for p in self.mlp.parameters():
                        p.zero_()

            def forward(self, x):
                q = self.probe.repeat((len(x), 1, 1)).to(x.dtype)
                y = self.attn(q, x, x, need_weights=False)[0]
                return y + self.mlp(self.layernorm(y))

        class ToyVisual(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool_type = "attn"
                self.use_cls_token = True
                self.attn_pool = ToyPool()
                self.proj = nn.Parameter(torch.diag(torch.tensor([2., 3., 4., 5.])))

            def _pool(self, x):
                return self.attn_pool(x).squeeze(1)

        cls.visual = ToyVisual().eval().requires_grad_(False)
        cls.tokens = torch.arange(1, 21, dtype=torch.float32).reshape(1, 5, 4)

    def test_alpha0_and_fullmask_exact_original_pool(self):
        t = self.torch
        for mask, alpha in (([[[1, 0], [0, 0]]], 0), (np.ones((1, 2, 2)), c.ALPHA)):
            g, s, stats = c.pool_features(self.visual, self.tokens, mask, alpha)
            expected = self.visual._pool(self.tokens) @ self.visual.proj
            self.assertTrue(t.equal(g, expected))
            self.assertTrue(t.equal(s, expected))
            self.assertFalse(stats["guided"].any())

    def test_soft_prior_reuses_attention_projection_and_no_normalize(self):
        t = self.torch
        before = {k: v.clone() for k, v in self.visual.state_dict().items()}
        token_before = self.tokens.clone()
        g, s, stats = c.pool_features(self.visual, self.tokens, [[[1, 0], [0, 0]]])
        weights = t.tensor([1., 1., .5, .5, .5])  # leading CLS is always global
        expected = (self.tokens * weights[None, :, None]).sum(1) / weights.sum()
        expected = expected @ self.visual.proj
        t.testing.assert_close(s, expected, rtol=1e-6, atol=1e-6)
        self.assertFalse(t.equal(g, s))
        self.assertGreater(float(s.norm()), 1)
        self.assertEqual(float(stats["prior_min"][0]), .5)
        self.assertEqual(float(stats["cls_prior"][0]), 1)
        self.assertTrue(stats["background_positive"].all())
        self.assertTrue(t.equal(token_before, self.tokens))
        for k, v in self.visual.state_dict().items():
            self.assertTrue(t.equal(before[k], v))
        self.assertFalse(s.requires_grad)

    def test_empty_fallback_and_mixed_batch_exact_rows(self):
        t = self.torch
        tokens = self.tokens.repeat(3, 1, 1)
        masks = t.tensor([[[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]], [[.25, 0.], [0., 0.]]])
        g, s, stats = c.pool_features(self.visual, tokens, masks)
        self.assertTrue(t.equal(g[:2], s[:2]))
        self.assertEqual(stats["empty_fallback"].tolist(), [True, False, False])
        self.assertEqual(stats["guided"].tolist(), [False, False, True])
        self.assertTrue(stats["background_positive"].all())

    def test_residual_mlp_is_applied_before_visual_projection(self):
        t = self.torch
        visual = type(self.visual)().eval().requires_grad_(False)
        residual = t.tensor([.1, .2, .3, .4])
        with t.no_grad():
            visual.attn_pool.mlp[-1].bias.copy_(residual)
        _, soft, _ = c.pool_features(visual, self.tokens, [[[1, 0], [0, 0]]])
        weights = t.tensor([1., 1., .5, .5, .5])
        expected = (self.tokens * weights[None, :, None]).sum(1) / weights.sum()
        expected = (expected + residual) @ visual.proj
        t.testing.assert_close(soft, expected, rtol=1e-6, atol=1e-6)

    def test_mask_shape_values_and_fixed_alpha(self):
        for mask, alpha in (([[1, 0, 0]], .5), ([[1, 0, np.nan, 0]], .5),
                            ([[1, 0, 0, 2]], .5), ([[1, 0, 0, 0]], .25)):
            with self.assertRaises(ValueError):
                c.pool_features(self.visual, self.tokens, mask, alpha)

    def test_real_num_heads_contract(self):
        old = self.visual.attn_pool.num_heads
        try:
            self.visual.attn_pool.num_heads = old + 1
            with self.assertRaises(ValueError):
                c.pool_features(self.visual, self.tokens, [[[1, 0], [0, 0]]])
        finally:
            self.visual.attn_pool.num_heads = old


if __name__ == "__main__":
    unittest.main()
