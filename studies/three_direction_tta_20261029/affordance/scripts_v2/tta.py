"""Label-free episodic adaptation; no scoring, no checkpoint choice from labels."""
import copy, hashlib, json, random, time
import numpy as np
import torch

def set_seed(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def predict(model,img,point,human,obj,seed):
 set_seed(seed);return model(img,point,human,obj)

def configure(model):
 model.eval()
 for p in model.parameters():p.requires_grad_(False)
 names=[]
 for name,p in model.named_parameters():
  if name.startswith('decoder.cmff.cross_atten1.layernorm.'):
   p.requires_grad_(True);names.append(name)
 if not names:raise RuntimeError('No adaptive parameters')
 return names

def adapt(model,img,point,human,obj,steps,lr=1e-4,seed=42,objective="consistency"):
 """Point-order equivariance teacher, source anchor; fixed final-step readout.
 Random point permutation preserves physical geometry and functional labels exactly.
 Each permuted prediction is inverse-restored before consistency loss or averaging.
 Sampling RNG is reset per call; same-index FPS start may still select different
 physical initial points after permutation, so consistency is not purely decoder.
 """
 original_requires={n:p.requires_grad for n,p in model.named_parameters()}
 names=configure(model);device=point.device
 params=[p for p in model.parameters() if p.requires_grad]
 saved={n:p.detach().clone() for n,p in model.named_parameters() if n in names}
 python_rng=random.getstate();numpy_rng=np.random.get_state()
 rng_state=torch.random.get_rng_state();cuda_state=torch.cuda.get_rng_state_all() if device.type=='cuda' else None
 try:
  with torch.no_grad():
   source=predict(model,img,point,human,obj,seed).detach()
   repeat=predict(model,img,point,human,obj,seed).detach()
   repeat_difference=float((source-repeat).abs().max())
  generator=torch.Generator(device='cpu').manual_seed(seed+1)
  permutations=[torch.randperm(point.shape[-1],generator=generator).to(device) for _ in range(2)]
  with torch.no_grad():
   views=[predict(model,img,point[:,:,perm],human,obj,seed)[:,perm.argsort()] for perm in permutations]
   teacher=torch.stack([source,*views]).mean(0).detach()
   disagreement=torch.stack([source,*views]).std(0,unbiased=False)
   # Smooth reliability weighting; no outcome-based rejection or threshold search.
   weight=(1.0-disagreement).clamp(0,1).detach()
  opt=torch.optim.Adam(params,lr=lr);history=[]
  for step in range(steps):
   opt.zero_grad(set_to_none=True)
   perm=permutations[step%len(permutations)]
   pred=predict(model,img,point,human,obj,seed)
   if objective=='consistency':consistency=((pred-teacher).square()*weight).mean()
   elif objective=='entropy':
    p=pred.clamp(1e-6,1-1e-6);consistency=-(p*p.log()+(1-p)*(1-p).log()).mean()
   else:raise ValueError(objective)
   anchor=sum((p-saved[n]).square().mean() for n,p in model.named_parameters() if n in saved)
   loss=consistency+0.01*anchor
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite TTA loss')
   loss.backward();grad=float(torch.sqrt(sum(p.grad.square().sum() for p in params if p.grad is not None)))
   opt.step();history.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':grad})
  with torch.no_grad():after=predict(model,img,point,human,obj,seed).detach()
  delta=float(torch.sqrt(sum((p-saved[n]).square().sum() for n,p in model.named_parameters() if n in saved)))
  return {'source':source.cpu().numpy(),'ensemble':teacher.cpu().numpy(),'adapted':after.cpu().numpy(),
    'diagnostics':{'loss':history,'parameter_delta_l2':delta,'view_disagreement':float(disagreement.mean()),'objective':objective,'mean_prediction':float(after.mean()),'predicted_fraction_above_05':float((after>=.5).float().mean()),'same_input_repeat_max_abs':repeat_difference,'updated_names':names,'updated_count':sum(p.numel() for p in params)}}
 finally:
  with torch.no_grad():
   for n,p in model.named_parameters():
    if n in saved:p.copy_(saved[n])
  model.zero_grad(set_to_none=True);torch.random.set_rng_state(rng_state)
  random.setstate(python_rng);np.random.set_state(numpy_rng)
  for n,p in model.named_parameters():p.requires_grad_(original_requires[n])
  if cuda_state is not None:torch.cuda.set_rng_state_all(cuda_state)
