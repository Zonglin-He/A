import math,unittest
from scripts.stvg_semantic_preference_contract_v1 import *
class PreferenceContracts(unittest.TestCase):
    def test_posterior_stability_and_null(self):
        p=categorical_from_logprobs([-1000.,-1001.,-1002.]);self.assertAlmostEqual(sum(p),1);self.assertEqual(p.index(max(p)),0)
    def test_nonfinite_or_not_logprob_rejected(self):
        for v in [[0.,float('nan'),-1.],[.2,-1.,-2.]]:
            with self.assertRaises(ValueError):categorical_from_logprobs(v)
    def test_label_permutation_is_undone(self):
        self.assertEqual(restore_candidate_order([.1,.2,.3,.4],[2,0,1]),[.1,.3,.4,.2])
    def test_nll_has_correct_mean_gradient_and_no_teacher_or_action_gradient(self):
        import torch
        mean=torch.tensor([[.1,-.2]],dtype=torch.float64,requires_grad=True)
        actions=torch.tensor([[[.2,.5],[-.5,.1]]],dtype=torch.float64,requires_grad=True)
        weights=torch.tensor([[.3,.4]],dtype=torch.float64,requires_grad=True);sigma=.25
        loss=weighted_fixed_support_nll(mean,actions,weights,sigma);loss.backward()
        expected=(weights.detach().unsqueeze(-1)*(mean.detach().unsqueeze(-2)-actions.detach())).sum(-2)/(sigma*sigma)
        self.assertTrue(torch.allclose(mean.grad,expected,rtol=0,atol=1e-12));self.assertIsNone(actions.grad);self.assertIsNone(weights.grad)
    def test_abstention_mass_not_renormalized_away(self):
        p=categorical_from_logprobs([math.log(.6),math.log(.3),math.log(.1)])
        self.assertAlmostEqual(sum(p[1:]),.4)
if __name__=='__main__':unittest.main()
