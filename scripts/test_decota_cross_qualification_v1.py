"""Cross-stage contracts: isolated scope, exact lifecycle and guard semantics."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from vg_tta.spatial_online_state_v1 import arrival
from vg_tta.decota_actuation_scope_v1 import commit_state
from vg_tta.c1_enabling_tricks_v1 import QUERY
from scripts.tastvg_decota_c1_common_v1 import guard
from scripts.run_spatial_ssl_gpu_v1 import config

class CrossContracts(unittest.TestCase):
    def test_opposite_checkpoints(self):
        self.assertEqual(config('vidstg_test').source_dataset,'hcstvg2')
        self.assertEqual(config('hcstvg1_test').source_dataset,'vidstg')
    def test_query_reset_and_LN_fraction(self):
        src={QUERY:torch.zeros(256),'spatial.layers.5.norm1.weight':torch.ones(256)}
        used={QUERY:torch.ones(256),'spatial.layers.5.norm1.weight':torch.ones(256)*3}
        c=commit_state(src,used);self.assertTrue(torch.equal(c[QUERY],src[QUERY]));self.assertTrue(torch.equal(c['spatial.layers.5.norm1.weight'],torch.ones(256)*1.125))
        nxt=arrival(src,c,'O-split');self.assertTrue(torch.equal(nxt['spatial.layers.5.norm1.weight'],c['spatial.layers.5.norm1.weight']))
        reset=arrival(src,None,'O-split');self.assertTrue(torch.equal(reset['spatial.layers.5.norm1.weight'],src['spatial.layers.5.norm1.weight']))
    def test_GT_and_outcome_guard(self):
        for s in ['/tmp/GT_LABELS_search.json','/tmp/results/demo/ROWS.json','/tmp/annotations/video.json']:
            with self.assertRaises(PermissionError):guard('open',(s,))
        guard('open',('/tmp/cross_domain/PLAN.json',));guard('open',('/tmp/cross_domain/capture/00000.pt',))

if __name__=='__main__':unittest.main()
