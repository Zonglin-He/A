import sys,pathlib,importlib.util,json,hashlib
sys.dont_write_bytecode=True
import numpy as np,torch
ROOT=pathlib.Path(__file__).resolve().parent
from algorithm import native_hm,evaluate_cache,Residual
spec=importlib.util.spec_from_file_location('original_hm',ROOT.parent/'upstream/utils/loss.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_num_threads(4);torch.manual_seed(123)
p=(torch.rand(5,64,1)*.8+.1).requires_grad_(True);target=torch.rand_like(p);original=module.HM_Loss()(p,target);grad1=torch.autograd.grad(original,p,retain_graph=True)[0];ported=native_hm(p,target);grad2=torch.autograd.grad(ported,p)[0];assert torch.equal(original,ported);assert torch.equal(grad1,grad2)
a=torch.full((1,8,1),.3,requires_grad=True);same=native_hm(a,a.detach());selfgrad=torch.autograd.grad(same,a)[0];assert float(selfgrad.abs().max())>0
hidden=np.random.default_rng(4).normal(size=(5,64,64)).astype('float32');logits=np.random.default_rng(5).normal(size=(5,64,1)).astype('float32');neighbors=np.stack([np.roll(np.arange(64),-i)[1:17] for i in range(64)])
rng_before=torch.get_rng_state().clone()
x0,d0=evaluate_cache(hidden,logits,neighbors,steps=0,device='cpu');x1,d1=evaluate_cache(hidden,logits,neighbors,steps=3,device='cpu');x0again,d0again=evaluate_cache(hidden,logits,neighbors,steps=0,device='cpu')
assert torch.equal(torch.get_rng_state(),rng_before),'Adaptation consumed shared RNG'
assert d1['parameter_l2']>0 and all(r['gradient_l2']>0 for r in d1['history']);assert d1['maximum_absolute_logit_residual']<=.5;assert all(np.array_equal(x0[k],x0again[k]) for k in x0)
report={'kind':'synthetic_CPU_engineering_only_not_task_scores','ported_HM_loss_and_gradient_bitexact_with_official':True,'soft_target_self_gradient_nonzero_max_abs':float(selfgrad.abs().max()),'hard_perfect_dice_offset':.5,'zero_head_and_new_episode_reset_bitexact':True,'torch_CPU_RNG_unchanged':True,'parameter_count':129,'nonzero_gradients_and_parameter_change':True,'bounded_residual_max_abs':d1['maximum_absolute_logit_residual'],'no_real_model_inference_or_GT_masks_used':True}
(ROOT/'ENGINEERING_CPU_V2.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
