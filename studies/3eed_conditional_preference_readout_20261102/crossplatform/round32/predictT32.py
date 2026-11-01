"""Locked readout families on new observations; no current-target supervision."""
import sys,time,numpy as np,collections
from common32 import *
import scope32 as row_module
import endpoint_scope32 as end_module
import fit24
fit24.feature=feature
from fit24 import module,batch,scores,interact
import torch

def selected_specs():
 s=read(EXP/'SELECTED_SCOPE_LOCK.json');p=s['primary'];return {
 'scoped_TTA':p,
 'same_gate_QP_memory':s['same_gate_memory'],
 'DEVbest_QP_memory':s['best_DEV_memory_same_budget'],
 'global_mean_matched':dict(p,arm='global_mean_TTA'),
 'global_norm_matched':dict(p,arm='global_norm_TTA'),
 'DEVbest_global_amount':s['best_DEV_global_amount_control'],
 'same_family_row_scope':s['same_family_row_support_ablation'],
 'DEVbest_row_scope':s['best_DEV_row_support_ablation'],
 'DEVbest_ungated_QP_memory':s['best_DEV_ungated_memory']}

def old_memory_library(A,feedback):
 keys=np.stack([interact(feature(r)) for r in A]);keys/=np.maximum(np.linalg.norm(keys,axis=-1,keepdims=True),1e-12);values=np.zeros((len(A),256),np.float32);counts=np.zeros_like(values);lookup={r['id']:i for i,r in enumerate(A)}
 for f in feedback:
  if f['id_PRIVATE'] not in lookup:continue
  j=lookup[f['id_PRIVATE']];s=feature(A[j])['scores'];P=f['positive_group_PRIVATE'];N=f['negative_group_PRIVATE'];delta=max(.1-float(s[P].max()-s[N].max()),0)/2;values[j,P]+=delta;values[j,N]-=delta;counts[j,P]+=1;counts[j,N]+=1
 values/=np.maximum(counts,1);active=np.array(sorted({lookup[f['id_PRIVATE']] for f in feedback if f['id_PRIVATE'] in lookup}));return keys,values,active

def old_memory_scores(z,lib):
 keys,values,active=lib;q=interact(z);q/=np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12);out={}
 for name,ix in [('R30_feedbackA_memory',active),('R30_allA_zeroU_memory',np.arange(len(keys)))]:
  sim=torch.einsum('qc,anc->qan',torch.tensor(q,device='cuda'),torch.tensor(keys[ix],device='cuda'));near,which=sim.max(-1);best,obs=near.topk(4,dim=-1);pos=which.gather(1,obs);w=((best-1)/.1).exp();v=torch.tensor(values[ix],device='cuda')[obs,pos];corr=((v*w).sum(-1)/w.sum(-1).clamp_min(1e-12)).cpu().numpy();out[name]=z['scores']+corr
 return out

def build_selected_libraries(A,tokenizer,feedback):
 configs=selected_specs();libs={}
 for spec in configs.values():
  ident=(spec['support'],spec['key'])
  if ident not in libs:
   mod=end_module if spec['support']=='endpoint' else row_module;libs[ident]=mod.build_library(A,spec['key'],tokenizer,feedback)
 return libs

def readouts(r,z,pT,libs,tokenizer,oldlib):
 p0=z['scores'];q0=int(p0.argmax());configs=selected_specs();values={};process={}
 for ident,lib in libs.items():
  support,key=ident;mod=end_module if support=='endpoint' else row_module;knn=mod.neighbors(mod.make_key(z,r,key,tokenizer),lib);vv,stats=mod.score_row(r,p0,pT,lib,knn);values[ident]=vv;process[ident]=stats
 d0=row_module.stable_logodds(p0);d0-=d0[q0];dt=row_module.stable_logodds(pT);dt-=dt[q0];out={'source':d0,'R30_ungated':dt}
 for name,spec in configs.items():out[name]=values[spec['support'],spec['key']][spec['k']][spec['arm']]
 out.update(old_memory_scores(z,oldlib));return out,process

def main():
 from transformers import RobertaTokenizerFast
 verify(BASE/'sourceT128_attempt001');out=BASE/'selectedT128_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();tokenizer=RobertaTokenizerFast.from_pretrained(ROOT/'crossplatform/upstream/data/roberta_base',local_files_only=True);feedback=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;start=time.perf_counter();process=[]
 for pl in ['drone','quad']:
  A=[r for r in rows() if r['platform']==pl];B=[r for r in rows('T128') if r['platform']==pl];b=batch(B);m=module('query_mlp');state=load_npz(ROOT/'crossplatform/round30/screen_attempt001/states_PRIVATE'/(pl+'_b1.npz'));m.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in state.items()});m.eval().requires_grad_(False)
  with torch.no_grad():pT=scores(m,'query_mlp',b).cpu().numpy()
  libs=build_selected_libraries(A,tokenizer,feedback);oldlib=old_memory_library(A,feedback)
  for i,r in enumerate(B):
   z=feature(r);ss,st=readouts(r,z,pT[i],libs,tokenizer,oldlib);names=list(ss);qs=np.array([row_module.group_readout(r,v) for v in ss.values()]);assert qs[0]==int(z['scores'].argmax());assert qs[1]==int(pT[i].argmax());np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(names),queries=qs,boxes=z['boxes'][qs],adapted_probability=pT[i],primary_scores=ss['scoped_TTA'],memory_scores=ss['same_gate_QP_memory']);process.append({'alias':r['alias'],'platform':pl,'selected_gate_stats':[{'support':a,'key':k,'k':n,**v} for (a,k),kv in st.items() for n,v in kv.items()]})
  print('T128_SCOPE32',pl,'seconds',round(time.perf_counter()-start,1),flush=True)
 dump(out/'PROCESS_STATS.json',process);seal(out,'R32_selected_newT128_predictions_before_labels',n=128,selected_scope_sha256=sha(EXP/'SELECTED_SCOPE_LOCK.json'),no_T_feedback_or_labels=True,new_optimizer_steps=0,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated());print('T128_SCOPE32_SEALED')
if __name__=='__main__':main()
