import unittest
import torch
from research.native_event_position_v1.core import original_positions

class Table(torch.nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('table',torch.arange(70*8).reshape(70,1,8).float())
    def forward(self,n):return self.table[:n]

class PositionTests(unittest.TestCase):
    def test_noop_and_scoped_removal(self):
        m=Table();old=m(33).clone()
        with original_positions(m,[33,32],[list(range(33)),list(range(32))])as calls:
            for n in [33,33,32,32]:self.assertTrue(torch.equal(m(n),old[:n]))
        self.assertEqual(len(calls),4);self.assertTrue(all(c['same_native_values']for c in calls));self.assertFalse(m._forward_hooks)
    def test_suffix_preserves_indices(self):
        m=Table();old=m(33).clone();ix=[list(range(8,33)),list(range(8,32))]
        with original_positions(m,[33,32],ix)as calls:
            for o,n in [(0,25),(0,25),(1,24),(1,24)]:self.assertTrue(torch.equal(m(n),old[ix[o]]))
        self.assertFalse(any(c['same_native_values']for c in calls));self.assertTrue(torch.equal(m(33),old))
    def test_hook_removed_on_failure(self):
        m=Table()
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with original_positions(m,[33,32],[list(range(25)),list(range(24))]):raise RuntimeError('deliberate')
        self.assertFalse(m._forward_hooks)

if __name__=='__main__':unittest.main()
