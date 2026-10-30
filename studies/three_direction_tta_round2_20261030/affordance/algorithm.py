"""129-parameter zero-initialized local residual with task loss and support protection.
No labels, source annotation paths, threshold tuning, or fixed area targets enter this module.
"""
import torch
import torch.nn.functional as F
CONSTANTS={'K':5,'steps':3,'lr':.01,'residual_logit_cap':.5,'knn':16,'feature_std_floor':1e-6,'focal_alpha':.25,'focal_gamma':2.,'hm_epsilon':1e-6,'support_lambda':1.,'weight_decay_lambda':.001,'support_quantile':.25,'teacher_quantile':.5,'quantile_interpolation':'linear','leave_one_view_out':True,'readout':'sigmoid(median_linear_quantile_of_adapted_view_logits)'}
def native_hm(pred,target):
 gamma=2;alpha=.25
 temp1=-(1-alpha)*torch.mul(pred**gamma,torch.mul(1-target,torch.log(1-pred+1e-6)))
 temp2=-alpha*torch.mul((1-pred)**gamma,torch.mul(target,torch.log(pred+1e-6)))
 temp=temp1+temp2
 CELoss=torch.sum(torch.mean(temp,(0,1)))
 intersection_positive=torch.sum(pred*target,1)
 cardinality_positive=torch.sum(torch.abs(pred)+torch.abs(target),1)
 dice_positive=(intersection_positive+1e-6)/(cardinality_positive+1e-6)
 intersection_negative=torch.sum((1.-pred)*(1.-target),1)
 cardinality_negative=torch.sum(2-torch.abs(pred)-torch.abs(target),1)
 dice_negative=(intersection_negative+1e-6)/(cardinality_negative+1e-6)
 temp3=torch.mean(1.5-dice_positive-dice_negative,0)
 DICELoss=torch.sum(temp3)
 return CELoss+1.0*DICELoss
def fixed_features(hidden,neighbors):
 # K,N,64; geometry neighbors are identical across all FPS views.
 local=hidden[:,neighbors,:].mean(2)-hidden
 x=torch.cat([hidden,local],-1)
 mean=x.mean((0,1),keepdim=True);std=x.std((0,1),unbiased=False,keepdim=True).clamp_min(1e-6)
 return ((x-mean)/std).detach()
def peer_targets(logits,neighbors):
 targets=[];lower=[];weights=[]
 for v in range(len(logits)):
  z=torch.cat([logits[:v],logits[v+1:]],0)
  median=torch.quantile(z,.5,dim=0,interpolation='linear');low=torch.quantile(z,.25,dim=0,interpolation='linear')
  mad=torch.quantile((z-median).abs(),.5,dim=0,interpolation='linear')
  neighbor_median=torch.quantile(z[:,neighbors,:],.5,dim=2,interpolation='linear')
  contrast=torch.quantile(z-neighbor_median,.25,dim=0,interpolation='linear').clamp_min(0)
  reliability=contrast/(1+contrast)*torch.exp(-mad)
  targets.append(torch.sigmoid(median));lower.append(low);weights.append(reliability)
 return torch.stack(targets).detach(),torch.stack(lower).detach(),torch.stack(weights).detach()
class Residual(torch.nn.Module):
 def __init__(self):
  super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1,128));self.bias=torch.nn.Parameter(torch.zeros(1))
 def forward(self,x):return .5*torch.tanh(F.linear(x,self.weight,self.bias))
def evaluate_cache(hidden,logits,neighbors,variant='hm_support',steps=3,device=None):
 if device is None:raise ValueError('Declare CPU engineering or root-owned CUDA execution explicitly')
 h=torch.as_tensor(hidden,device=device,dtype=torch.float32);z=torch.as_tensor(logits,device=device,dtype=torch.float32);nb=torch.as_tensor(neighbors,device=device,dtype=torch.long)
 x=fixed_features(h,nb);target,lower,r=peer_targets(z,nb);adapter=Residual().to(device);assert sum(p.numel() for p in adapter.parameters())==129
 optimizer=torch.optim.Adam(adapter.parameters(),lr=.01);history=[]
 with torch.no_grad():
  zero=torch.sigmoid(z+adapter(x));base=torch.sigmoid(z);assert torch.equal(zero,base),'Zero residual changed native view probabilities'
  native_median=torch.sigmoid(torch.quantile(z,.5,dim=0,interpolation='linear'));zero_median=torch.sigmoid(torch.quantile(z+adapter(x),.5,dim=0,interpolation='linear'));assert torch.equal(native_median,zero_median)
 for step in range(steps):
  optimizer.zero_grad(set_to_none=True);delta=adapter(x);p=torch.sigmoid(z+delta);hm=native_hm(p,target)
  support=(r*F.relu(lower-(z+delta)).square()).sum()/(r.sum()+1e-6)
  regularizer=sum(q.square().mean() for q in adapter.parameters());loss=hm+(support if variant=='hm_support' else 0)+.001*regularizer
  assert variant in ['hm_support','hm_only']
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite adaptation loss')
  loss.backward();grad=torch.sqrt(sum(q.grad.square().sum() for q in adapter.parameters() if q.grad is not None));optimizer.step()
  history.append({'step':step+1,'total_loss':float(loss.detach()),'hm_loss':float(hm.detach()),'support_loss':float(support.detach()),'gradient_l2':float(grad.detach())})
 with torch.no_grad():
  delta=adapter(x);adapted=torch.sigmoid(torch.quantile(z+delta,.5,dim=0,interpolation='linear'));p=torch.sigmoid(z)
  out={'native_views':p.cpu().numpy(),'source':p[0].cpu().numpy(),'ensemble':p.mean(0).cpu().numpy(),'median':native_median.cpu().numpy(),'q75':torch.quantile(p,.75,dim=0,interpolation='linear').cpu().numpy(),'adapted':adapted.cpu().numpy()}
  diag={'variant':variant,'steps':steps,'history':history,'parameter_count':129,'parameter_l2':float(torch.sqrt(sum(q.square().sum() for q in adapter.parameters()))),'maximum_absolute_logit_residual':float(delta.abs().max()),'support_weight_sum':float(r.sum()),'support_weight_nonzero_fraction':float((r>0).float().mean()),'zero_head_same_device_views_bitexact':True,'zero_head_same_device_median_bitexact':True,'annotation_targets_used':False,'coordinate_geometry_used_only_for_neighbor_graph':True,'final_readout':CONSTANTS['readout']}
 return out,diag
