"""Finite real-data external baseline port. Source resets and all rows explicit.
No target annotation containers, RGB, VLM, network request or scoring API.
"""
from pathlib import Path
import sys,time,types,copy,random,hashlib
import numpy as np
import torch
import torch.nn as nn
from common import ROOT,EXP,read,dump,sha,seal,verify,check_plan
sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from run_diagnostic import load_model,mask_for,fingerprint
from runner_helpers import sample_points
from algorithms import scores,rawboxes,detach,tent,sar,memo
from reg_inspired import run_reg,REG_NAMES

class Reset:
 def __init__(self,model):
  self.model=model;self.state={n:v.detach().clone() for n,v in model.state_dict().items()};self.bn={n:(m.forward,m.track_running_stats) for n,m in model.named_modules() if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d))}
 def reset(self,seed=0):
  for n,m in self.model.named_modules():
   if n in self.bn:
    m.forward,m.track_running_stats=self.bn[n]
    for k in ('running_mean','running_var','num_batches_tracked'):
     key=n+'.'+k
     if key in self.state:setattr(m,k,self.state[key].clone())
  self.model.load_state_dict(self.state,strict=True);self.model.eval().requires_grad_(False)
  for p in self.model.parameters():p.grad=None
  torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
 def audit(self,selected):
  names={n for n,p in selected};changed=[];delta=0.
  for n,p in self.model.named_parameters():
   same=torch.equal(p.detach(),self.state[n])
   if not same:
    assert n in names,n;changed.append(n);delta+=float((p.detach()-self.state[n]).square().sum())
  return {'changed_tensor_count':len(changed),'changed_parameter_count':sum(dict(self.model.named_parameters())[n].numel() for n in changed),'parameter_delta_L2':delta**.5,'nonselected_parameters_unchanged':True}
def own_episode(model,row,source,mask):
 pref=read(ROOT/'crossplatform/round17/G_selected64_attempt001/cases_PRIVATE'/(row['id']+'.json'))['preference'];q0=int(source['scores'].argmax());diag={'steps':0,'feedback_accepted':pref['status']=='usable'};selected=[]
 if pref['status']=='usable':
  with np.load(ROOT/'crossplatform/round17/features96_attempt001/samples_PRIVATE'/(row['id']+'.npz')) as z:x=torch.tensor(z['x'],device='cuda');tokens=torch.tensor(z['tokens'],device='cuda');tm=torch.tensor(z['token_mask'],device='cuda')
  layer=model.contrastive_align_projection_image[-1];layer.requires_grad_(True);selected=[('contrastive_align_projection_image.4.'+n,p) for n,p in layer.named_parameters()];base={n:p.detach().clone() for n,p in layer.named_parameters()};s0=torch.tensor(source['scores'],device='cuda');pos=pref['positive_queries'];neg=pref['negative_queries'];opt=torch.optim.AdamW(layer.parameters(),lr=.0001,weight_decay=0.)
  for _ in range(32):
   opt.zero_grad();q=torch.nn.functional.normalize(layer(x),dim=-1);s=((q@tokens.T/.07).softmax(-1)*tm).sum(-1).clamp_min(1e-12);log=s.log();gap=log[pos].mean()-log[neg].mean();p0=(s0/s0.sum()).clamp_min(1e-12);p=(s/s.sum()).clamp_min(1e-12);loss=torch.nn.functional.softplus(1-gap)+.1*(p0*(p0.log()-p.log())).sum()+sum((p-base[n]).square().mean() for n,p in layer.named_parameters());loss.backward();opt.step()
  diag['steps']=32
 return pref['direct_query'],diag,selected
def main(limit=None):
 plan=check_plan();rows=read(EXP.parent/'input/ALL96_PRIVATE.json')
 if limit is not None:rows=[r for p in ('drone','quad') for r in rows if r['platform']==p][:1] if limit==1 else [next(r for r in rows if r['platform']==p) for p in ('drone','quad')]
 out=ROOT/'crossplatform/round18'/('engineering2_attempt001' if limit else 'formal96_attempt001');out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();start=time.perf_counter();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));sourcefp=fingerprint(model);reset=Reset(model);trace=[]
 for i,row in enumerate(rows):
  tick=time.perf_counter();seed=int(hashlib.sha256(('R18|'+row['id']).encode()).hexdigest()[:8],16)%2**31;reset.reset(seed);pc,_,_=sample_points(row);pc=torch.tensor(pc,device='cuda')[None];text=row['caption']+' . not mentioned';mm,_=mask_for(model,text,'all_content_tokens');mask=torch.as_tensor(mm,dtype=torch.bool,device='cuda');torch.cuda.reset_peak_memory_stats()
  with torch.no_grad():original=model({'point_clouds':pc,'text':[text]});source=detach(original,mask)
  with np.load(ROOT/'crossplatform/round17/features96_attempt001/samples_PRIVATE'/(row['id']+'.npz')) as z:assert np.array_equal(source['boxes'],z['boxes']) and np.array_equal(source['scores'],z['scores'])
  names=[];boxes=[];qindices=[];diagnostics={};mode='A' if i<64 and not limit else ('engineering' if limit else 'B')
  def add(name,pred,q=None):
   q=int(np.argmax(pred['scores'])) if q is None else int(q);names.append(name);qindices.append(q);boxes.append(pred['boxes'][q].copy())
  add('source',source)
  reset.reset(seed);pre,post,d,sel=tent(model,pc,[text],mask);d.update(reset.audit(sel));diagnostics['Tent']=d;add('Tent_BN_statistics_only',pre);add('Tent_adapted_post',post)
  reset.reset(seed);pre,post,d,sel=sar(model,pc,[text],mask);d.update(reset.audit(sel));diagnostics['SAR']=d;add('SAR_norm_statistics_only',pre);add('SAR_adapted_post',post)
  reset.reset(seed);rng=np.random.default_rng(seed);rawpath=Path(row['pointpath']);raw=np.load(rawpath,allow_pickle=False) if rawpath.suffix=='.npy' else np.fromfile(rawpath,np.float32).reshape(-1,4);raw=np.asarray(raw[:,:3],np.float32).copy()
  if row['platform']=='drone':raw[:,2]+=np.float32(1.8)
  views=np.stack([raw[rng.choice(len(raw),16384,replace=len(raw)<16384)]+rng.normal(0,.01,(16384,3)).clip(-.03,.03).astype(np.float32) for _ in range(4)]);anchor=dict(source,points=pc[0].cpu().numpy());post,prior,tq,d,sel=memo(model,torch.tensor(views,device='cuda'),[text]*4,mask,anchor);d.update(reset.audit(sel));diagnostics['MEMO']=d;add('MEMO_BN_prior_only',prior);add('MEMO_augmentation_only',source,tq);add('MEMO_adapted_post',post)
  reset.reset(seed);pre,post,student,direct,d,sel,_,_,_=run_reg(model,pc,text,mask,seed);d.update(reset.audit(sel));diagnostics['Reg_inspired']=d;add('Reg_inspired_teacher_post',post);add('Reg_inspired_student_post',student);names.append('Reg_inspired_direct_pseudo');qindices.append(int(source['scores'].argmax()));boxes.append(direct)
  # A64 already has sealed RGB/VLM G evidence; B has no G and is never faked.
  if row['id'] in plan['A_ids_PRIVATE']:
   reset.reset(seed);dq,d,sel=own_episode(model,row,source,mask)
   with torch.no_grad():gpost=detach(model({'point_clouds':pc,'text':[text]}),mask)
   d.update(reset.audit(sel));diagnostics['own_G_episode']=d;add('own_G_direct_no_grounder_update',source,dq);add('own_G_episode_projection32',gpost)
  assert np.isfinite(np.array(boxes)).all()
  np.savez_compressed(out/'predictions_PRIVATE'/(row['id']+'.npz'),sample_id=np.array(row['id']),arm_names=np.array(names),arm_boxes_raw6=np.array(boxes,np.float32),query_indices=np.array(qindices),source_pool=source['boxes'],source_scores=source['scores'])
  trace.append({'id_PRIVATE':row['id'],'alias':f'{mode}_{row["platform"]}_{i:03d}','platform':row['platform'],'subset':mode,'diagnostics':diagnostics,'seconds':time.perf_counter()-tick,'peak_bytes':torch.cuda.max_memory_allocated()});dump(out/'TRACE_PRIVATE.json',trace);dump(out/'PROGRESS.json',{'completed':i+1,'n':len(rows),'last_seconds':trace[-1]['seconds']});print('BASELINES',i+1,len(rows),row['platform'],round(trace[-1]['seconds'],2),flush=True)
 reset.reset();assert fingerprint(model)==sourcefp;dump(out/'registered_plan.json',plan);seal(out,'R18_external_baselines_sealed_before_current_scoring',n=len(rows),source_unchanged=True,seconds=time.perf_counter()-start,programs=plan['programs']);print('BASELINES_SEALED',flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--engineering',action='store_true');a=p.parse_args();main(2 if a.engineering else None)
