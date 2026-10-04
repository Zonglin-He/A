"""CPU-only contracts. No model import, media, expert cache, weights, or GT."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "vg_tta/tastvg_negative_evidence_v1.py"
spec = importlib.util.spec_from_file_location("negative_evidence_math", MODULE)
math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math)


def fixture():
    center = np.array([[.40, .45, .30, .30], [.55, .55, .20, .25], [.45, .40, .25, .20]])
    candidates = np.stack([center, center + [.04, -.02, .01, -.01],
                           center + [-.03, .03, -.01, .01]])
    expert = center.copy()
    return center, candidates, expert, np.array([True, False, True])


class EvidenceContracts(unittest.TestCase):
    def test_import_is_numpy_only(self):
        code = ("import importlib.util,sys; s=importlib.util.spec_from_file_location('n',sys.argv[1]); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "assert 'torch' not in sys.modules")
        subprocess.run([sys.executable, "-I", "-B", "-c", code, str(MODULE)], check=True)

    def test_center_zero_and_raw_iou_units(self):
        _, c, e, v = fixture()
        x = math.negative_evidence(c, e, v)
        np.testing.assert_array_equal(x["valid_positions"], [0, 2])
        np.testing.assert_array_equal(x["e_jk"][:, 0], [0, 0])
        np.testing.assert_allclose(x["framewise_rewards"][:, 0], 1)
        self.assertTrue(((x["e_jk"] >= 0) & (x["e_jk"] <= 1)).all())

    def test_positive_part_precedes_observation_mean(self):
        center = np.array([[.3, .3, .2, .2], [.7, .7, .2, .2]])
        candidate = center[::-1].copy()
        candidates = np.stack([center, candidate])
        expert = np.array([center[0], candidate[1]])
        x = math.negative_evidence(candidates, expert, np.ones(2, bool))
        np.testing.assert_allclose(x["e_jk"], [[0, 1], [0, 0]], rtol=0, atol=1e-15)
        np.testing.assert_allclose(x["global_e"], [0, .5], rtol=0, atol=1e-15)
        self.assertEqual(max(x["framewise_rewards"][:, 0].mean() - x["framewise_rewards"][:, 1].mean(), 0), 0)

    def test_missing_expert_never_becomes_negative(self):
        _, c, e, v = fixture()
        e[1] = np.nan
        self.assertTrue(np.isfinite(math.negative_evidence(c, e, v)["e_jk"]).all())
        x = math.negative_evidence(c, np.full_like(e, np.nan), np.zeros(3, bool))
        self.assertEqual(x["e_jk"].shape, (0, 3))
        np.testing.assert_array_equal(x["global_e"], np.zeros(3))

    def test_invalid_input_rejected(self):
        _, c, e, v = fixture()
        with self.assertRaises(ValueError):
            math.negative_evidence(c, e, v.astype(int))
        e[0, 2] = 0
        with self.assertRaises(ValueError):
            math.negative_evidence(c, e, v)

    def test_independent_block_counterfactuals(self):
        keys = ["spatial.query_residual"] + [f"spatial.layers.5.{n}.{p}" for n in math.BLOCKS[1:] for p in ["weight", "bias"]]
        pre = {key: np.zeros(256) for key in keys}
        post = {key: np.ones(256) for key in keys}
        groups = math.parameter_blocks(pre)
        self.assertEqual(sum(pre[key].size for key in keys), 1792)
        self.assertEqual([sum(pre[k].size for k in groups[b]) for b in math.BLOCKS], [256, 512, 512, 512])
        for block in math.BLOCKS:
            state = math.block_only_state(pre, post, block)
            for key in keys:
                np.testing.assert_array_equal(state[key], post[key] if key in groups[block] else pre[key])
            state[keys[0]][0] = 9
            self.assertEqual(pre[keys[0]][0], 0)
            self.assertEqual(post[keys[0]][0], 1)


class DifferentiableContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        tree = ast.parse((ROOT / "vg_tta/tastvg_spatial_online_opd_s1_v1.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "geometry")
        env = {"torch": torch}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "stock_geometry_only", "exec"), env)
        cls.stock_geometry = staticmethod(env["geometry"])

    def tensors(self):
        b, c, e, v = fixture()
        return self.torch.tensor(b, dtype=self.torch.float64, requires_grad=True), self.torch.tensor(c, dtype=self.torch.float64, requires_grad=True), e, v

    def test_global_geometry_bitwise_stock_arithmetic(self):
        b, c, _, _ = self.tensors()
        for dtype in [self.torch.float32, self.torch.float64]:
            b, c = b.to(dtype), c.to(dtype)
            actual = math.geometry_distances(b, c, (5, 2), mode="global")
            self.assertTrue(self.torch.equal(actual, self.stock_geometry(b, c, 5, 2)))

    def test_lambda_fixed_and_undecided_odds_preserved(self):
        b, c, e, v = self.tensors()
        c = self.torch.cat([c[:1], c[:1], c[1:]], 0)
        anchor = math.make_anchor(b, c, e, v, (5, 2), mode="global")
        self.assertEqual(anchor["lambda_value"], 1)
        self.assertFalse(anchor["logq"].requires_grad)
        self.assertAlmostEqual(float(anchor["logq"][0] - anchor["logq"][1]), float(anchor["logp0"][0] - anchor["logp0"][1]))
        odds_shift = anchor["logq"] - anchor["logp0"]
        self.assertAlmostEqual(float(odds_shift[2] - odds_shift[0]), -float(anchor["e"][2]), places=13)
        anchor["lambda_value"] = .5
        with self.assertRaises(ValueError):
            math.negative_loss(b, anchor)

    def test_no_evidence_initial_exact_zero_and_gradient(self):
        b, c, e, _ = self.tensors()
        for mode in ["global", "local"]:
            anchor = math.make_anchor(b, c, np.full_like(e, np.nan), np.zeros(3, bool), (5, 2), mode=mode)
            out = math.negative_loss(b, anchor)
            self.assertTrue(self.torch.equal(anchor["logq"], anchor["logp0"]))
            self.assertEqual(float(out["loss"]), 0)
            self.assertTrue(self.torch.equal(self.torch.autograd.grad(out["loss"], b)[0], self.torch.zeros_like(b)))

    def test_zero_penalties_exact_initial_noop(self):
        b, c, e, v = self.tensors()
        c = c[:1].repeat(3, 1, 1)
        for mode in ["global", "local"]:
            anchor = math.make_anchor(b, c, e, v, (5, 2), mode=mode)
            out = math.negative_loss(b, anchor)
            self.assertTrue(out["exact_initial_noop"])
            self.assertEqual(float(out["loss"]), 0)
            self.assertTrue(self.torch.equal(self.torch.autograd.grad(out["loss"], b)[0], self.torch.zeros_like(b)))

    def test_kl_equivalence_and_targets_detached(self):
        b, c, e, v = self.tensors()
        for mode in ["global", "local"]:
            anchor = math.make_anchor(b, c, e, v, (5, 2), mode=mode)
            out = math.negative_loss(b, anchor)
            g1 = self.torch.autograd.grad(out["loss"], b, retain_graph=True)[0]
            g2 = self.torch.autograd.grad(out["proximal_plus_penalty"], b)[0]
            self.torch.testing.assert_close(g1, g2, rtol=1e-12, atol=1e-12)
            self.assertIsNone(c.grad)
            for key in ["candidates", "logp0", "logq", "e", "e_jk", "framewise_rewards"]:
                self.assertFalse(anchor[key].requires_grad)

    def test_local_direct_output_gradient_uses_same_observations(self):
        b, c, e, v = self.tensors()
        anchor = math.make_anchor(b, c, e, v, (5, 2), mode="local")
        out = math.negative_loss(b, anchor)
        self.assertEqual(out["distance"].shape, (2, 3))
        gradient = self.torch.autograd.grad(out["loss"], b)[0]
        self.assertTrue(self.torch.equal(gradient[1], self.torch.zeros(4, dtype=b.dtype)))
        self.assertGreater(float(gradient[[0, 2]].norm()), 0)

    def test_shared_parameter_can_move_unobserved_output(self):
        b, c, e, v = self.tensors()
        parameter = self.torch.tensor(0., dtype=b.dtype, requires_grad=True)
        live = b.detach() + parameter * self.torch.tensor([1., 0, 0, 0], dtype=b.dtype)
        anchor = math.make_anchor(live, c, e, v, (5, 2), mode="local")
        gradient = self.torch.autograd.grad(math.negative_loss(live, anchor)["loss"], parameter)[0]
        self.assertNotEqual(float(gradient), 0)
        moved = b.detach() - .01 * gradient * self.torch.tensor([1., 0, 0, 0], dtype=b.dtype)
        self.assertGreater(float((moved[1] - b.detach()[1]).norm()), 0)

    def test_frozen_anchor_after_update(self):
        b, c, e, v = self.tensors()
        anchor = math.make_anchor(b, c, e, v, (5, 2), mode="global")
        original = anchor["logq"].clone()
        moved = (b.detach() + self.torch.tensor([.01, 0, 0, 0], dtype=b.dtype)).requires_grad_()
        out = math.negative_loss(moved, anchor)
        expected = (out["logp"].exp() * (out["logp"] - original)).sum()
        self.assertTrue(self.torch.equal(out["loss"], expected))
        self.assertTrue(self.torch.equal(anchor["logq"], original))

    def test_anchor_is_not_aliased_to_candidate_or_center(self):
        b, c, e, v = self.tensors()
        anchor = math.make_anchor(b, c, e, v, (5, 2), mode="local")
        saved = anchor["candidates"].clone()
        with self.torch.no_grad():
            c.add_(.01)
        self.assertTrue(self.torch.equal(anchor["candidates"], saved))


if __name__ == "__main__":
    unittest.main(verbosity=2)
