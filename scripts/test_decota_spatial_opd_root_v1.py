"""Source weighting and paired-bootstrap checks on synthetic diagnostic rows."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.finalize_decota_spatial_opd_v1 import independent_statistics

class AggregateTests(unittest.TestCase):
 def test_macro_does_not_upweight_duplicate_orders_or_sources(self):
  # Source0 has three cells; source1 has one. Task estimate still gives each
  # source equal weight, rather than assigning source0 three times its weight.
  r=[dict(source_id=0,delta_total_v=.2)]*3+[dict(source_id=1,delta_total_v=-.4)]
  s=independent_statistics(r,'delta_total_v')
  self.assertAlmostEqual(s['mean'],-.1)
  self.assertEqual(s['parents'],2)
  self.assertEqual(s['harm_gt20pp_parents'],1)
  self.assertEqual(s['harm_gt20pp_cells'],1)
  self.assertAlmostEqual(s['gross_gain_pp'],10.)
  self.assertAlmostEqual(s['gross_loss_pp'],20.)
 def test_paired_zero_difference_has_exact_zero_ci(self):
  r=[dict(source_id=j,contrast=0.) for j in range(7) for _ in range(2)]
  s=independent_statistics(r,'contrast')
  self.assertEqual(s['mean'],0.)
  self.assertEqual(s['ci95'],[0.,0.])

if __name__=='__main__':unittest.main()
