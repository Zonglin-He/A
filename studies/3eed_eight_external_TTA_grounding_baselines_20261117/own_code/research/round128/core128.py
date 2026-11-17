"""Native 3EED grounding ports of externally published TTA mechanisms.

All candidate identities are fixed spatial anchors, never query indices from a
different forward. No target labels enter this module. Original boxes are readout
endpoints; normalization and backbone features are genuinely recomputed.
"""
from pathlib import Path
import sys, json, hashlib, types, time, math
R=Path(__file__).resolve().parents[2];H=R/'research/round128';B=R/'crossplatform/round128'
sys.path.insert(0,str(R/'research/round97'))
from core97 import base97, strict97, source97, jread, jdump, sample_points
from model51 import SENTINEL
import torch, numpy as np
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def points(row,z):
 p=Path(row['pointpath']);raw=np.load(p,allow_pickle=False) if p.suffix=='.npy' else np.fromfile(p,np.float32).reshape(-1,4)
 ix=z['point_indices'].astype(np.int64);x=np.ascontiguousarray(raw[ix,:3],np.float32)
 if row['platform']=='drone':x[:,2]+=1.8
 if row['platform']=='waymo':
  from utils.transform_waymo import transform_to_front_view
  x,_=transform_to_front_view(x,np.array([[0.,0.,0.,1.,1.,1.,0.]]),['F','FL','FR','SL','SR'][row['suffix']])
 return torch.as_tensor(np.ascontiguousarray(x,dtype=np.float32),device='cuda')[None]

def native(base,pc,row,z):
 out=base._run_backbones({'point_clouds':pc,'text':[row['caption']+SENTINEL]})
 xyz=out['fp2_xyz'];V=out['fp2_features'].transpose(1,2).contiguous();L=out['text_feats'];tp=out['text_attention_mask']
 V,L=base.cross_encoder(vis_feats=V,pos_feats=base.pos_embed(xyz).transpose(1,2).contiguous(),padding_mask=torch.zeros(V.shape[:2],device=V.device,dtype=torch.bool),text_feats=L,text_padding_mask=tp,end_points=out,detected_feats=None,detected_mask=None)
 # Each raw sampling view is registered to the ORIGINAL spatial anchor set.
 anchors=torch.as_tensor(z['query_anchor_xyz'],device=pc.device);cost=torch.cdist(anchors,xyz[0]).detach().cpu().numpy();aa,bb=linear_sum_assignment(cost);assert np.array_equal(aa,np.arange(256));ids=torch.as_tensor(bb,device=pc.device,dtype=torch.int32)[None]
 _,f,_=base.gsample_module(xyz,V.transpose(1,2).contiguous(),ids);Q=base.decoder_query_proj(f).transpose(1,2).contiguous()
 for j,dec in enumerate(base.decoder):Q=dec(Q,V,L,torch.as_tensor(z['position'+str(j)],device=pc.device),None,tp,detected_feats=None,detected_mask=None)
 q=F.normalize(base.contrastive_align_projection_image(Q),dim=-1);u=F.normalize(base.contrastive_align_projection_text(L),dim=-1);a=(q@u.transpose(-1,-2)/.07)[0];m=torch.as_tensor(z['token_mask'],device=pc.device).bool();assert a.shape[1]==len(m)
 s=torch.logsumexp(a.masked_fill(~m[None],-torch.inf),-1)-torch.logsumexp(a.masked_fill(m[None],-torch.inf),-1)
 return s,{'seed_registration_mean_m':float(cost[aa,bb].mean()),'seed_registration_max_m':float(cost[aa,bb].max()),'h':Q,'q':q,'u':u}

def entropy(s):p=s.softmax(-1);return -(p*s.log_softmax(-1)).sum(-1)
def cpu(s):return s.detach().double().softmax(-1).cpu().numpy()
def snapshot(m):return {k:v.detach().clone() for k,v in m.state_dict().items()}
def reset(m,S):
 for mod in m.modules():
  if isinstance(mod,(nn.BatchNorm1d,nn.BatchNorm2d)):
   mod.track_running_stats=True
   if mod.running_mean is None:mod.running_mean=torch.zeros(mod.num_features,device='cuda');mod.running_var=torch.ones(mod.num_features,device='cuda')
   if hasattr(mod,'_r128_original_forward'):mod.forward=mod._r128_original_forward
 m.load_state_dict(S,strict=True);m.eval().requires_grad_(False);m.zero_grad(set_to_none=True)
def norm(m):
 m.eval().requires_grad_(False);ps=[]
 for name,mod in m.named_modules():
  if name.startswith('text_encoder'):continue
  if isinstance(mod,(nn.BatchNorm1d,nn.BatchNorm2d)):
   mod.train();mod.track_running_stats=False;mod.running_mean=None;mod.running_var=None
   for k,p in mod.named_parameters(recurse=False):
    if k in ('weight','bias'):p.requires_grad_(True);ps.append((name+'.'+k,p))
 return ps
def memo_bn(self,x):
 mean=torch.zeros_like(self.running_mean);var=torch.ones_like(self.running_var)
 F.batch_norm(x,mean,var,None,None,True,1.,self.eps)
 return F.batch_norm(x,self.running_mean*(16/17)+mean/17,self.running_var*(16/17)+var/17,self.weight,self.bias,False,0.,self.eps)
def configure_memo(m):
 m.eval().requires_grad_(True)
 for mod in m.modules():
  if isinstance(mod,(nn.BatchNorm1d,nn.BatchNorm2d)):
   if not hasattr(mod,'_r128_original_forward'):mod._r128_original_forward=mod.forward
   mod.forward=types.MethodType(memo_bn,mod)
 return list(m.named_parameters())
def gradinfo(ps):
 active=[p for _,p in ps if p.grad is not None];return {'selected_parameters':sum(p.numel() for _,p in ps),'gradient_parameters':sum(p.numel() for p in active),'gradient_L2':float(torch.sqrt(sum((p.grad.detach().double().square().sum() for p in active),torch.zeros((),device='cuda',dtype=torch.float64))))}
def views(pc,key):
 seed=int(hashlib.sha256((key+'|R128views').encode()).hexdigest()[:16],16);g=torch.Generator(device='cuda');g.manual_seed(seed);out=[]
 for _ in range(3):
  # Small measurement noise, no change to left/right/front/back semantics.
  out.append(pc+torch.randn(pc.shape,device='cuda',generator=g)*.01)
 return out
def fisher(base,src,S):
 rr=[x for x in jread(R/'research/round121/SOURCE_CACHE_REGISTRY_PRIVATE.json') if x['source']==src][:16];ps=norm(base);v={n:torch.zeros_like(p) for n,p in ps};records=[]
 for e in rr:
  z=load(R/e['files']['prefix']['path']);pc=points(e['row'],z);base.zero_grad(set_to_none=True);s,_=native(base,pc,e['row'],z);loss=F.cross_entropy(s[None],s.detach().argmax()[None]);loss.backward()
  for n,p in ps:
   if p.grad is not None:v[n]+=p.grad.detach().square()/len(rr)
  records.append({'source_input_sha256':hashlib.sha256(e['row']['id'].encode()).hexdigest(),'loss':float(loss.detach())})
 reset(base,S);return v,records

def process(base,S,e,fish):
 row=e['row'];z=load(R/e['prefix']);pc=points(row,z);vv=views(pc,row['id']);out={};diag={}
 reset(base,S)
 with torch.no_grad():s,reg=native(base,pc,row,z);out['source']=cpu(s);pure=[cpu(native(base,v,row,z)[0]) for v in vv];out['frozen_three_augment_mean']=np.mean(pure,axis=0)
 cached=z['query']@z['tokens'].T/.07;m=z['token_mask'].astype(bool);cs=torch.as_tensor(np.logaddexp.reduce(cached[:,m],1)-np.logaddexp.reduce(cached[:,~m],1),device='cuda');err=float(np.max(abs(out['source']-cpu(cs))));assert err<2e-5,err
 diag['source_replay_max_probability_error']=err;diag['registration']={k:v for k,v in reg.items() if k.startswith('seed_')}
 for method in ['tent','eata','memo']:
  reset(base,S);t=time.perf_counter();torch.cuda.reset_peak_memory_stats();ps=configure_memo(base) if method=='memo' else norm(base);opt=torch.optim.AdamW([p for _,p in ps],lr=1e-5,weight_decay=.01) if method=='memo' else torch.optim.SGD([p for _,p in ps],lr=.001,momentum=.9);trace=[]
  if method!='memo':
   with torch.no_grad():out['BN_statistics_only' if method=='tent' else 'eata_zero']=cpu(native(base,pc,row,z)[0])
  else:
   with torch.no_grad():out['memo_BNprior_zero']=cpu(native(base,pc,row,z)[0]);out['memo_BNprior_three_mean']=np.mean([cpu(native(base,v,row,z)[0]) for v in vv],0)
  pp=None
  for it in range(3 if method=='eata' else 1):
   opt.zero_grad(set_to_none=True)
   if method=='memo':
    probs=torch.stack([native(base,v,row,z)[0].softmax(-1) for v in vv]);av=probs.mean(0);loss=-(av*av.clamp_min(1e-30).log()).sum();accepted=True;record={'marginal_entropy':float(loss.detach())}
   else:
    s,_=native(base,vv[it],row,z) if method=='eata' else native(base,pc,row,z);ent=entropy(s);loss=ent;accepted=True;record={'entropy':float(ent.detach())}
    if method=='eata':
     prob=s.detach().softmax(-1);reliable=bool(ent.detach()<.4*math.log(256));cos=float(F.cosine_similarity(pp,prob,dim=0)) if pp is not None else None;novel=pp is None or abs(cos)<.05;accepted=reliable and novel;keep=sum((fish[n]*(p-S[n]).square()).sum() for n,p in ps)*2000;loss=ent*torch.exp(.4*math.log(256)-ent.detach())+keep;record.update(reliable=reliable,nonredundant=novel,probability_cosine=cos,Fisher_penalty=float(keep.detach()))
     if accepted:pp=prob.clone() if pp is None else .9*pp+.1*prob
   if accepted:loss.backward();record.update(gradinfo(ps));opt.step()
   else:record.update(selected_parameters=sum(p.numel() for _,p in ps),gradient_parameters=0,gradient_L2=0.)
   record.update(accepted=accepted,objective=float(loss.detach()));trace.append(record)
  with torch.no_grad():
   out[method+'_post']=cpu(native(base,pc,row,z)[0]);out[method+'_post_three_mean']=np.mean([cpu(native(base,v,row,z)[0]) for v in vv],0)
  delta=float(torch.sqrt(sum(((p.detach()-S[n]).double().square().sum() for n,p in ps),torch.zeros((),device='cuda',dtype=torch.float64))))
  diag[method]={'steps':trace,'parameter_delta_L2':delta,'seconds':time.perf_counter()-t,'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30}
 reset(base,S);return out,diag,z['boxes']
