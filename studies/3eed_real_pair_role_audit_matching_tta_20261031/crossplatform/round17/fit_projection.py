"""Actual final query-projection nn.Linear adaptation, no boxes/backbone update.
Frozen cached inputs make the boundary explicit. Full-model B closure is separate.
"""
from pathlib import Path
import sys,copy,time,math
import numpy as np
from common import ROOT,EXP,read,dump,sha,arrsha,seal,verify,check_plan

def token_scores(layer,x,tokens,mask):
 import torch
 y=torch.nn.functional.normalize(layer(x),dim=-1)
 return ((y@tokens.T/.07).softmax(-1)*mask).sum(-1)
def tensors(z):
 import torch
 return tuple(torch.tensor(z[k],dtype=torch.float32) for k in ('x','tokens','token_mask'))
def load_z(folder,sid):
 with np.load(folder/'samples_PRIVATE'/(sid+'.npz'),allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def joint_features(z):
 q=np.asarray(z['query'],np.float64);t=np.asarray(z['tokens'],np.float64)[z['token_mask']].mean(0);t/=max(np.linalg.norm(t),1e-12)
 # Query/text interaction, retaining noun-conditioned context in source query.
 return q*t
def run_platform(platform,A,B,features,groot,out,gate,spec):
 import torch
 torch.set_num_threads(4);torch.manual_seed(0)
 source=np.load(features/'projection_source_PRIVATE.npz',allow_pickle=False)
 layer=torch.nn.Linear(288,64);layer.load_state_dict({'weight':torch.from_numpy(source['4.weight'].copy()),'bias':torch.from_numpy(source['4.bias'].copy())})
 original={k:v.detach().clone() for k,v in layer.state_dict().items()};layer.eval();base=copy.deepcopy(layer).requires_grad_(False)
 fit=[];D=[];targets=[]
 for r in A:
  if r['platform']!=platform:continue
  pref=read(groot/'cases_PRIVATE'/(r['id']+'.json'))['preference'];z=load_z(features,r['id'])
  if pref['status']!='usable':continue
  pos=np.array(pref['positive_queries'],int);neg=np.array(pref['negative_queries'],int);x,t,m=tensors(z)
  with torch.no_grad():s=token_scores(base,x,t,m)
  assert np.max(np.abs(s.numpy()-z['scores']))<2e-5
  fit.append((r['id'],x,t,m,pos,neg,s.detach()))
  jf=joint_features(z);D.append(jf[pos].mean(0)-jf[neg].mean(0));logs=np.log(np.maximum(z['scores'],1e-12));targets.append(max(0.,1.-(logs[pos].mean()-logs[neg].mean())))
 eligible=bool(gate['passed']);traces=[];alpha=None;dd=None
 if eligible:
  assert len(fit)>=4
  optimizer=torch.optim.AdamW(layer.parameters(),lr=spec['lr'],weight_decay=0.)
  for step in range(spec['steps']):
   optimizer.zero_grad();losses=[];rank=[]
   for sid,x,t,m,pos,neg,s0 in fit:
    s=token_scores(layer,x,t,m).clamp_min(1e-12);log=s.log();gap=log[pos].mean()-log[neg].mean()
    ranking=torch.nn.functional.softplus(1.-gap)
    p0=(s0/s0.sum()).clamp_min(1e-12);p=(s/s.sum()).clamp_min(1e-12);kl=(p0*(p0.log()-p.log())).sum()
    losses.append(ranking+.1*kl);rank.append(ranking.detach())
   regularization=sum((v-original[k]).square().mean() for k,v in layer.named_parameters())
   loss=torch.stack(losses).mean()+regularization;assert torch.isfinite(loss);loss.backward()
   grad=float(torch.sqrt(sum(v.grad.square().sum() for v in layer.parameters())))
   optimizer.step();traces.append({'step':step+1,'loss_before_step':float(loss.detach()),'ranking':float(torch.stack(rank).mean()),'gradient_L2':grad})
  dd=np.asarray(D,float);kernel=dd@dd.T;lam=1e-3*max(float(np.trace(kernel)/len(dd)),1e-12);alpha=np.linalg.solve(kernel+lam*np.eye(len(dd)),np.asarray(targets))
 delta=float(torch.sqrt(sum((v-original[k]).square().sum() for k,v in layer.named_parameters())))
 assert (delta>0)==eligible
 np.savez_compressed(out/(platform+'_projection_PRIVATE.npz'),weight=layer.weight.detach().numpy(),bias=layer.bias.detach().numpy())
 if eligible:np.savez_compressed(out/(platform+'_preference_memory_PRIVATE.npz'),pair_features=dd,alpha=alpha)
 fit_receipt={'platform':platform,'gate_passed':eligible,'fit_A_samples':len(fit) if eligible else 0,'available_preference_samples':len(fit),'steps':len(traces),'parameter_L2_delta':delta,'trace':traces,'B_G_or_fit_used':False}
 fit_receipt['update_parameters']=sum(v.numel() for v in layer.parameters())
 dump(out/(platform+'_FIT.json'),fit_receipt)
 for r in B:
  if r['platform']!=platform:continue
  z=load_z(features,r['id']);x,t,m=tensors(z)
  with torch.no_grad():s=token_scores(layer,x,t,m).numpy()
  # No A fitting => exact source fallback, rather than numerical CPU tie changes.
  source_scores=z['scores'];qs=int(np.argmax(source_scores));qt=int(np.argmax(s)) if eligible else qs
  memory_log=np.log(np.maximum(source_scores.astype(float),1e-12))
  if eligible:memory_log=memory_log+(joint_features(z)@dd.T)@alpha
  qc=int(np.argmax(memory_log)) if eligible else qs
  indices=np.array([qs,qc,qt],np.int64);names=np.array(['source','same_information_A_memory_direct_rerank','matching_projection_TTA'])
  np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),sample_id=np.array(r['id']),arm_names=names,query_indices=indices,boxes=z['boxes'][indices],source_pool=z['boxes'],source_scores=source_scores,TTA_scores=s,memory_log_scores=memory_log)
 return fit_receipt
def main():
 check_plan();s=read(EXP/'FIT_SPEC_LOCK.json')
 for rel,h in s['programs'].items():assert sha(ROOT/rel)==h
 groot=ROOT/'crossplatform/round17/G_selected64_attempt001';features=ROOT/'crossplatform/round17/features96_attempt001';selected=verify(groot);verify(features)
 gate=read(EXP/'A_scoring_attempt001/GATE.json');assert selected['A_gate_sha256']==sha(EXP/'A_scoring_attempt001/GATE.json') and gate['fit_spec_sha256']==sha(EXP/'FIT_SPEC_LOCK.json')
 d=EXP.parent/'input';A=read(d/'A32_PRIVATE.json');B=read(d/'B32_PRIVATE.json');out=ROOT/'crossplatform/round17/B_predictions_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir()
 start=time.perf_counter();receipts=[run_platform(p,A,B,features,groot,out,gate['platforms'][p],s) for p in ('drone','quad')]
 dump(out/'registered_fit_spec.json',s);dump(out/'GATE.json',gate)
 seal(out,'R17_B32_predictions_sealed_before_current_B_labels',n=32,fit=receipts,boxes_from_frozen_source_pool_only=True,no_B_G_or_fit=True,seconds=time.perf_counter()-start,programs=s['programs'])
 print('B_PREDICTIONS_SEALED',[(r['platform'],r['steps'],r['parameter_L2_delta']) for r in receipts],flush=True)
if __name__=='__main__':main()
