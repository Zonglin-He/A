"""Small numerical/authorization contracts without any research input."""
import sys, unittest, tempfile, subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.teacher_purification_import_v1 import core
select,consensus,truth_iou,overlap_matrix=core.select,core.consensus,core.truth_iou,core.overlap_matrix

class PurificationContracts(unittest.TestCase):
    def test_manual_peer_scores_exclude_self(self):
        p=[[0.,2.],[0.,3.],[10.,11.]]
        i,v=consensus(p); self.assertEqual(i,0)
        np.testing.assert_allclose(v,[1/3,1/3,0.],atol=0,rtol=1e-15)
    def test_duplicate_votes_preserved(self):
        r=select([[0,1],[0,1],[10,11]],[0,0,100])
        self.assertEqual(r['proposal_count'],3);self.assertEqual(r['unique_intervals'],2)
        self.assertEqual(r['Consensus_index'],0);self.assertEqual(r['Confidence_index'],2)
        self.assertEqual(r['consensus_scores'],[.5,.5,0])
    def test_confidence_cannot_change_medoid(self):
        p=[[0,2],[.2,2.2],[10,11]]
        a=select(p,[0,0,100]);b=select(p,[100,0,0])
        self.assertEqual(a['Consensus_index'],b['Consensus_index']);self.assertEqual(a['consensus_scores'],b['consensus_scores'])
    def test_first_tie_no_confidence_fallback(self):
        r=select([[0,1],[2,3],[4,5]],[1,2,3])
        self.assertEqual(r['Consensus_index'],0);self.assertEqual(r['Confidence_index'],2)
    def test_fractional_continuous_iou(self):
        np.testing.assert_allclose(truth_iou([[.1,.6],[.6,.8]],[.2,.6]),[.8,0],atol=1e-15)
        np.testing.assert_allclose(overlap_matrix([[.1,.6],[.2,.6]]),[[1,.8],[.8,1]],atol=1e-15)
    def test_singleton_and_invalid_no_fallback(self):
        r=select([[0,2]],[.9]);self.assertTrue(r['singleton_agreement_undefined']);self.assertEqual(r['Consensus_index'],0)
        for p,c in [([],[]),([[2,1]],[.5]),([[0,1]],[float('nan')]),([[0,1]],[])]:
            with self.assertRaises(ValueError):select(p,c)
    def test_guard_prevents_gt_read_allows_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'GT_LABELS_fake.json';path.write_text('{}')
            code='import sys; from scripts.teacher_purification_import_v1 import core; sys.addaudithook(core.read_guard); open(sys.argv[1]).read()'
            r=subprocess.run([sys.executable,'-B','-c',code,str(path)],cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('PermissionError',r.stderr)
            code='import sys; from scripts.teacher_purification_import_v1 import core; sys.addaudithook(core.read_guard); open(sys.argv[1],"w").write("sealed")'
            r=subprocess.run([sys.executable,'-B','-c',code,str(Path(tmp)/'UNLABELED_SELECTION.json')],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
    def test_guard_blocks_scored_control_and_model_input(self):
        read_guard=core.read_guard
        for p in ['/fake/TEACHER_SELECTION_SCORED.json','/fake/checkpoints/model.pth','/fake/target_features/000.pt']:
            with self.assertRaises(PermissionError):read_guard('open',(p,'r',0))
    def test_isolated_core_has_no_deep_learning_framework(self):
        self.assertFalse(any(x in sys.modules for x in ['torch','tensorflow','jax']))
if __name__=='__main__':unittest.main()
