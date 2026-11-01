"""Selected states on untouched S64; same-A information nonparametric control."""
import sys,time,numpy as np
from common30 import *
import torch,fit24
fit24.feature=feature
from fit24 import module,batch,scores,interact

def main():
 verify(BASE/'sourceS64_attempt001');lock=read(EXP/'SELECTED_BETA_LOCK.json');out=BASE/'memoryAllS64_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();start=time.perf_counter();pred={};fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 for pl in ['drone','quad']:
  B=[r for r in rows('S64') if r['platform']==pl];b=batch(B);a=[r for r in rows() if r['platform']==pl];lookup={r['id']:i for i,r in enumerate(a)};ss={}
  for arm,name in [('preference_only','b0'),('keep_selected',lock['selected']['name'])]:
   m=module('query_mlp');state=load_npz(BASE/'screen_attempt001/states_PRIVATE'/(pl+'_'+name+'.npz'));m.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in state.items()})
   with torch.no_grad():ss[arm]=scores(m,'query_mlp',b).cpu().numpy()
  keys=np.stack([interact(feature(r)) for r in a]);keys/=np.maximum(np.linalg.norm(keys,axis=-1,keepdims=True),1e-12);vals=np.zeros((len(a),256),np.float32);counts=np.zeros_like(vals);ff=[f for f in fb if f['platform']==pl]
  for f in ff:
   j=lookup[f['id_PRIVATE']];s=feature(a[j])['scores'];P=f['positive_group_PRIVATE'];N=f['negative_group_PRIVATE'];delta=max(.1-float(s[P].max()-s[N].max()),0)/2;vals[j,P]+=delta;vals[j,N]-=delta;counts[j,P]+=1;counts[j,N]+=1
  vals/=np.maximum(counts,1);active=np.arange(len(a));K=torch.tensor(keys[active]);V=torch.tensor(vals[active])
  for i,r in enumerate(B):
   z=feature(r);q=interact(z);q/=np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12);sim=torch.einsum('qc,anc->qan',torch.tensor(q),K);near,which=sim.max(-1);best,obs=near.topk(min(4,len(active)),dim=-1);pos=which.gather(1,obs);w=((best-1)/.1).exp();correction=(V[obs,pos]*w).sum(-1)/w.sum(-1).clamp_min(1e-12);s={'source':z['scores'],'preference_only':ss['preference_only'][i],'keep_selected':ss['keep_selected'][i],'all_A_zeroU_memory':z['scores']+correction.numpy()};qq=np.array([int(v.argmax()) for v in s.values()]);np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(list(s)),queries=qq,boxes=z['boxes'][qq],**{n+'_scores':v for n,v in s.items()})
 seal(out,'R30_allA_zeroU_memory_before_current_labels',n=64,selected_beta_sha256=sha(EXP/'SELECTED_BETA_LOCK.json'),S_feedback_or_labels_used=False,seconds=time.perf_counter()-start);print('ALLA_MEMORY_SEALED')
if __name__=='__main__':main()
