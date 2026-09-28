import unittest
import torch
from vg_tta.desta3d_v2_cast_probe import TokenEvidence, cast_difference, serialize_with_guard


class CastProbeTests(unittest.TestCase):
    def test_full_delta_sparse_crossing_and_erasure(self):
        a = torch.tensor([1., 1.0038, -1., 0.], dtype=torch.float32)
        b = a + torch.tensor([.0001, .0002, -.0001, 0.])
        out = cast_difference(a, a.bfloat16(), b, b.bfloat16())
        sp = out['sparse_postcast']; idx = sp['indices'].long()
        reconstructed = a.bfloat16().clone(); reconstructed[idx] = sp['after']
        self.assertTrue(torch.equal(reconstructed, b.bfloat16()))
        self.assertTrue(torch.equal(out['precast_delta'], b-a))
        self.assertEqual(out['stats']['continuous_changed_postcast_unchanged'], 2)
        self.assertEqual(out['stats']['postcast_nonzero'], 1)
        with self.assertRaises(AssertionError):
            cast_difference(a, a.bfloat16(), b, a.bfloat16())

    def test_original_chunk_hook_and_ce_algebra(self):
        torch.manual_seed(3)
        head = torch.nn.Linear(4, 9, bias=False)
        data = {'labels': torch.tensor([[-100] + [i % 9 for i in range(39)]]),
                'ptd_prefix_lengths': torch.tensor([19])}
        ev = TokenEvidence(data); handle = head.register_forward_hook(ev.hook)
        x = torch.randn(39, 4); before = head.weight.clone(); loss = 0
        for i in range(0, 39, 32):
            z = head(x[i:i+32]); loss += torch.nn.functional.cross_entropy(
                z, data['labels'][0, 1+i:1+i+32], reduction='sum')/39
        handle.remove(); evidence = ev.finish()
        ce = torch.cat([c['cross_entropy'] for c in evidence['chunks']])
        algebra = torch.cat([c['logsumexp'] - c['target_logit'] for c in evidence['chunks']])
        self.assertTrue(torch.allclose(ce, algebra, atol=1e-6, rtol=0))
        self.assertAlmostEqual(float(loss), float(ce.mean()), places=6)
        self.assertTrue(torch.equal(before, head.weight))
        self.assertEqual(int(evidence['ntp'].sum()), 18)

    def test_storage_refuses_cap_or_reserve_before_writing(self):
        payload = {'x': torch.arange(3)}
        raw = serialize_with_guard(payload, used_bytes=0, free_bytes=20_000_000,
                                   cap_bytes=2_000_000, reserve_bytes=10_000_000)
        self.assertGreater(len(raw), 0)
        for cap, free in [(100, 20_000_000), (2_000_000, 10_000_001)]:
            with self.assertRaises(AssertionError):
                serialize_with_guard(payload, used_bytes=0, free_bytes=free,
                                     cap_bytes=cap, reserve_bytes=10_000_000)


if __name__ == '__main__': unittest.main()
