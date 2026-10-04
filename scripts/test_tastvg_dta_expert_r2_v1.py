"""Matched R1 optimizer, fixed selector/tie/coordinate and GT guard checks."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from vg_tta.tastvg_dta_expert_r2_v1 import deploy_choice,oracle_choice,validate_support,read_guard
from vg_tta.tastvg_dta_oracle_v1 import fit_query,gaussian

class Tests(unittest.TestCase):
    def test_deploy_confidence_first_tie(self):
        x=deploy_choice([[0,8],[2,5],[3,7]],[.9,.9,.1]);self.assertEqual(x['index'],0);self.assertFalse(x['GT_used'])
    def test_oracle_support_first_tie(self):
        x=oracle_choice([[0,4],[5,9],[5,9]],[1,.1,.9],[5,9]);self.assertEqual(x['index'],1);self.assertEqual(x['continuous_tIoU'],1.)
    def test_fractional_center_preserved(self):
        x=deploy_choice([[.123,5.765]],[.1]);self.assertEqual(x['interval'],[.123,5.765]);a,_=gaussian([0,2,4,6],x['interval']);b,_=gaussian([0,2,4,6],[0,5]);self.assertFalse(torch.equal(a,b))
    def test_invalid_support_not_silently_skipped(self):
        for p,c in [([],[]),([[2,2]],[1]),([[1,3]],[float('nan')]),([[1,3]],[])]:
            with self.assertRaises(ValueError):validate_support(p,c)
    def test_guard_labels_results_and_writes(self):
        for path in ['a/GT_LABELS_search.json','a/oracle_runs/x.pt','a/ROWS.json']:
            with self.assertRaises(PermissionError):read_guard('open',(path,'rb',0))
        read_guard('open',('a/GT_LABELS_search.json','w',0));read_guard('open',('a/DEPLOY_SELECTION.json','r',0));read_guard('open',('a/HEAD.pt','rb',0))
    def test_same_center_bitwise_R1_and_reset(self):
        torch.manual_seed(7);hidden=torch.randn(8,256);head={'0.weight':torch.randn(256,256)*.03,'0.bias':torch.zeros(256),'1.weight':torch.randn(2,256)*.03,'1.bias':torch.zeros(2)}
        center=oracle_choice([[2,12],[5,9]],[.2,.4],[2,12])['interval'];a=fit_query(hidden,list(range(0,16,2)),head,[2,12],.01);b=fit_query(hidden,list(range(0,16,2)),head,center,.01)
        self.assertEqual(a['after'],b['after'])
        for u,v in zip(a['states'],b['states']):self.assertTrue(torch.equal(u['weight'],v['weight']))
        c=fit_query(hidden,list(range(0,16,2)),head,center,.01);self.assertEqual(c['head_final_sha256'],a['head_final_sha256'])
        self.assertEqual(a['steps'],3);self.assertEqual(a['beta'],1.)

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
