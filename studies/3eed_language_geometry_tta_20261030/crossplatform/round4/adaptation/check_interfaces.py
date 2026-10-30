"""SyntheticJacobian/bridge/reset engineering, NOT a task adaptation loss."""
from pathlib import Path
import json,hashlib
import torch,numpy as np
from interfaces import *
R=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);g=torch.Generator().manual_seed(631);q=torch.randn(8,64,generator=g,dtype=torch.float64);q=q/q.norm(dim=-1,keepdim=True);t=torch.randn(11,64,generator=g,dtype=torch.float64);t=t/t.norm(dim=-1,keepdim=True);logits=q@t.T/.07;mask=torch.tensor([0,1,1,0,0,0,0,0,0,0,0],dtype=torch.bool);prob=logits.softmax(-1);score=(prob*mask[None]).sum(-1)
 rng=torch.get_rng_state().clone();m=Rank4QueryResidual(dtype=torch.float64);assert torch.equal(torch.get_rng_state(),rng)
 assert sum(p.numel() for p in m.parameters())==512
 before=m(q);assert torch.equal(before,q)
 anchored=anchored_span_scores(before,q,t,logits,mask,score);assert torch.equal(anchored,score);assert torch.equal(delta_anchored_logits(before,q,t,logits),logits)
 # A synthetic linearfunctional probes the Jacobian; no targetteacher/newloss.
 coefficients=torch.randn(8,generator=g,dtype=torch.float64);probe=(anchored*coefficients).sum();probe.backward();up_grad=float(m.up.grad.norm());assert up_grad>0;assert m.down.grad.norm()==0
 initial=[p.detach().clone() for p in m.parameters()];opt=torch.optim.SGD(m.parameters(),lr=.01);opt.step();after=m(q);assert not torch.equal(after,q);assert torch.allclose(after.norm(dim=-1),q.norm(dim=-1),atol=1e-14,rtol=1e-14)
 delta=float(m.parameter_delta_l2());m.reset_episode();assert torch.equal(m(q),q) and all(p.grad is None for p in m.parameters());assert m.parameter_delta_l2()==0
 # Reinitialization doesnotcarry optimizerstate; productionmustfreshoptimizer.
 m2=Rank4QueryResidual(dtype=torch.float64);assert all(torch.equal(a,b) for a,b in zip(m.parameters(),m2.parameters()))
 # All-token sum isnot referringcontext. It is mathematicallyconstant.
 allmask=torch.ones(11,dtype=torch.bool);allscore=(prob*allmask[None]).sum(-1);assert float((allscore-1).abs().max())<1e-14
 # Identical tokenvectors provide no contrastivequerysignal; test onlygradient,
 # do notuse tinysoftmax roundoffas an adaptationtarget.
 m3=Rank4QueryResidual(dtype=torch.float64);same=t[:1].expand(11,-1);same_logits=q@same.T/.07;same_score=(same_logits.softmax(-1)*mask[None]).sum(-1);noninfo=anchored_span_scores(m3(q),q,same,same_logits,mask,same_score);(noninfo*coefficients).sum().backward();noninfo_grad=float(m3.up.grad.norm());assert noninfo_grad<1e-12
 # Perrow additive logits cancel under the actualtoken-softmax readout.
 bias=torch.linspace(-.3,.3,8)[:,None];assert torch.allclose(logits.softmax(-1),(logits+bias).softmax(-1),atol=1e-14,rtol=1e-14)
 boxes=torch.tensor([[1.,2.,3.,4.,2.,2.],[4.,3.,2.,2.,1.,1.]],dtype=torch.float64);eligible=torch.tensor([True,False]);pose=SizePriorPoseLatent(boxes,eligible,.05,.1);base=pose();assert torch.equal(base[:,:6],boxes);assert torch.count_nonzero(base[:,6])==0
 weights=torch.arange(14,dtype=torch.float64).reshape(2,7);(pose()*weights).sum().backward();assert pose.raw.grad[0].norm()>0 and pose.raw.grad[1].norm()==0
 with torch.no_grad():pose.raw.fill_(1e6)
 moved=pose();assert torch.equal(moved[:,3:6],boxes[:,3:6]);assert torch.equal(moved[1],base[1]);assert torch.linalg.vector_norm(moved[0,:3]-boxes[0,:3])<=.05+1e-14;assert abs(moved[0,6])<=.1+1e-14
 pose.reset_episode();assert torch.equal(pose(),base)
 arr={'boxes':np.ones((8,6)),'semantic_scores':score.numpy(),'proj_queries':q.numpy(),'proj_tokens':t.numpy(),'exact_logits':logits.numpy(),'token_scores':prob.numpy(),'input_ids':np.arange(11),'attention_mask':np.ones(11),'offset_mapping':np.c_[np.arange(11),np.arange(11)+1],'noun_mask':mask.numpy(),'caption_mask':np.ones(11),'context_mask':~mask.numpy()};contract=validate_cache_arrays(arr,expected_queries=8)
 report={'state':'passed','scope':'syntheticengineeringonly; no realdata/GT/model/checkpoint/GPU read','query_adapter_parameters':512,'rank':4,'embedding_dim':64,'zero_adapter_query_logits_and_readout_bitexact':True,'first_step_nonzero_up_gradient':up_grad,'first_step_down_gradient_zero_expected':True,'synthetic_SGD_probe_parameter_delta':delta,'query_norms_preserved':True,'episodic_reset_exact':True,'constructor_globalRNG_unchanged':True,'all_token_softmax_mass_is_one':True,'identical_tokens_noninformation_gradient':noninfo_grad,'row_additive_calibration_invariance':True,'size_prior_preserved_even_huge_latent':True,'ineligible_pose_unchanged':True,'pose_zero_bridge_and_reset_exact':True,'fixture_physical_budgets':'0.05m/0.1rad engineeringonly, not researchhyperparameters','cache_contract':contract,'interfaces_sha256':hashlib.file_digest((R/'interfaces.py').open('rb'),'sha256').hexdigest(),'research_loss_or_teacher_selected':False}
 (R/'INTERFACE_FIXTURES.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
