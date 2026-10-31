"""Grounding adaptations of Tent/SAR/MEMO, preserving their update mechanisms.
Native readout is token mass. log(mass) is the corresponding candidate logit,
so another softmax does not flatten already-normalized bounded probabilities.
"""
import types,math,importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import ROOT
from geometry import groups,match_boxes

def scores(out,mask):
 q=out['last_proj_queries'];t=out['proj_tokens'];return ((q@t.transpose(-1,-2)/.07).softmax(-1)*mask[None,None,:]).sum(-1)
def rawboxes(out):return torch.cat([out['last_center'],out['last_pred_size']],-1)
def entropy(s):
 log=s.clamp_min(1e-30).log();log=log-log.logsumexp(-1,keepdim=True);return -(log.exp()*log).sum(-1)
def configure_norm(model,sar=False):
 model.eval().requires_grad_(False);selected=[]
 for name,module in model.named_modules():
  if name.startswith('text_encoder'):continue
  if sar and isinstance(module,(nn.BatchNorm1d,nn.BatchNorm2d)):continue
  if isinstance(module,(nn.BatchNorm1d,nn.BatchNorm2d)):
   module.train();module.track_running_stats=False;module.running_mean=None;module.running_var=None
  elif not(sar and isinstance(module,(nn.LayerNorm,nn.GroupNorm))):continue
  for k,p in module.named_parameters(recurse=False):
   if k in ('weight','bias'):p.requires_grad_(True);selected.append((name+'.'+k,p))
 return selected
def grad_summary(selected):
 active=[(n,p) for n,p in selected if p.grad is not None];return {'selected_parameters':sum(p.numel() for n,p in selected),'gradient_parameters':sum(p.numel() for n,p in active),'gradient_L2':float(torch.sqrt(sum(p.grad.detach().square().sum() for n,p in active))) if active else 0.}
def tent(model,pc,text,mask):
 selected=configure_norm(model);optimizer=torch.optim.SGD([p for n,p in selected],lr=.001,momentum=.9);optimizer.zero_grad()
 out=model({'point_clouds':pc,'text':text});pre=detach(out,mask);loss=entropy(scores(out,mask)).mean();loss.backward();diag=grad_summary(selected);optimizer.step();diag.update(loss=float(loss.detach()),steps=1)
 with torch.no_grad():post=detach(model({'point_clouds':pc,'text':text}),mask)
 return pre,post,diag,selected
def detach(out,mask):return {'boxes':rawboxes(out).detach().cpu().numpy()[0],'scores':scores(out,mask).detach().cpu().numpy()[0]}
def sar(model,pc,text,mask,unfiltered=False):
 selected=configure_norm(model,True);spec=importlib.util.spec_from_file_location('R18_official_SAM',ROOT/'crossplatform/round18/official/SAR/sam.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 optimizer=mod.SAM([p for n,p in selected],torch.optim.SGD,lr=.001,momentum=.9,rho=.05);optimizer.zero_grad();out=model({'point_clouds':pc,'text':text});pre=detach(out,mask);h=entropy(scores(out,mask)).mean();margin=.4*math.log(256);diag={'steps':0,'entropy':float(h.detach()),'reliability_margin':margin,'first_reliable':bool(unfiltered or h.detach()<margin),'second_reliable':False,'recovered':False}
 if diag['first_reliable']:
  h.backward();diag.update(grad_summary(selected));optimizer.first_step(zero_grad=True);out2=model({'point_clouds':pc,'text':text});h2=entropy(scores(out2,mask)).mean();diag['second_entropy']=float(h2.detach());diag['second_reliable']=bool(unfiltered or h2.detach()<margin)
  if diag['second_reliable']:
   h2.backward();optimizer.second_step(zero_grad=True);diag['steps']=1
   if h2.detach()<.2:
    for group in optimizer.param_groups:
     for p in group['params']:
      if 'old_p' in optimizer.state[p]:p.data.copy_(optimizer.state[p]['old_p'])
    diag['recovered']=True
  else:
   # Empty second filter is not a NaN loss and must undo the SAM perturbation.
   with torch.no_grad():
    for p in [p for n,p in selected]:
     if 'old_p' in optimizer.state[p]:p.copy_(optimizer.state[p]['old_p'])
 else:diag.update(selected_parameters=sum(p.numel() for n,p in selected),gradient_parameters=0,gradient_L2=0.)
 with torch.no_grad():post=detach(model({'point_clouds':pc,'text':text}),mask)
 return pre,post,diag,selected
def memo_bn(self,x):
 # Official MEMO BN prior16, extended to3D backbone BatchNorm1d as well as2d.
 mean=torch.zeros_like(self.running_mean);var=torch.ones_like(self.running_var);F.batch_norm(x,mean,var,None,None,True,1.,self.eps)
 return F.batch_norm(x,self.running_mean*(16/17)+mean*(1/17),self.running_var*(16/17)+var*(1/17),self.weight,self.bias,False,0.,self.eps)
def memo(model,views,text,mask,anchor,transformer_config=False):
 model.eval().requires_grad_(True)
 for m in model.modules():
  if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):m.forward=types.MethodType(memo_bn,m)
 with torch.no_grad():prior=detach(model({'point_clouds':torch.as_tensor(anchor['points'],device=views.device)[None],'text':text[:1]}),mask)
 selected=list(model.named_parameters());optimizer=(torch.optim.AdamW([p for n,p in selected],lr=.00001,weight_decay=.01) if transformer_config else torch.optim.SGD([p for n,p in selected],lr=.00025));optimizer.zero_grad();out=model({'point_clouds':views,'text':text});ss=scores(out,mask);bb=rawboxes(out).detach().cpu().numpy();g0,rep0=groups(anchor['boxes'],anchor['scores']);aligned=[];mapped=[];diags=[]
 for v in range(len(views)):
  sv=ss[v];gv,rv=groups(bb[v],sv.detach().cpu().numpy());a,b,diag=match_boxes(anchor['boxes'][rep0],bb[v,rv]);diags.append(diag);mapv={int(x):int(y) for x,y in zip(a,b)};mapped.append(mapv)
 common=sorted(set.intersection(*(set(x) for x in mapped)));coverage=[]
 for v in range(len(views)):
  mass=torch.stack([ss[v,g].mean() for g in groups(bb[v],ss[v].detach().cpu().numpy())[0]]);take=[mapped[v][a] for a in common];coverage.append(float(mass[take].sum().detach()/mass.sum().detach()) if take else 0.)
  if take:aligned.append(mass[take]/mass[take].sum().clamp_min(1e-30))
 diag={'augmentations':len(views),'anchor_groups':len(g0),'common_groups':len(common),'coverage':coverage,'geometry_matching':diags,'steps':0,'selected_parameters':sum(p.numel() for n,p in selected),'gradient_parameters':0,'gradient_L2':0.}
 ttaq=int(anchor['scores'].argmax())
 if len(common)>=2 and min(coverage)>=.5:
  avg=torch.stack(aligned).mean(0);loss=-(avg*avg.clamp_min(1e-30).log()).sum();ttaq=int(rep0[common[int(avg.detach().argmax())]]);loss.backward();diag.update(grad_summary(selected));optimizer.step();diag.update(steps=1,loss=float(loss.detach()))
 with torch.no_grad():post=detach(model({'point_clouds':torch.as_tensor(anchor['points'],device=views.device)[None],'text':text[:1]}),mask)
 return post,prior,ttaq,diag,selected
