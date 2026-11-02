"""Twelve fixed main arms and immutable memories, native probability units."""
import sys,time
import numpy as np
from common46 import *
BASE_ARMS=['source','R24_query_mlp_lr1_s16','R24_memory_alpha1','R30_beta1','R30_memory_E_only','R30_memory_allA_zeroU']
def make_key(z):
 t=z['tokens'][z['token_mask']].mean(0);t=t/max(float(np.linalg.norm(t)),1e-12);k=z['query']*t;k/=np.maximum(np.linalg.norm(k,axis=-1,keepdims=True),1e-12);return k
def retrieve(q,K,V,library):
 import torch
 kt=torch.tensor(K[library]);vt=torch.tensor(V[library]);sim=torch.einsum('qc,anc->qan',torch.tensor(q),kt);near,which=sim.max(-1);best,obs=near.topk(min(4,len(library)),dim=-1);pos=which.gather(1,obs);w=((best-1)/.1).exp();corr=(vt[obs,pos]*w).sum(-1)/w.sum(-1).clamp_min(1e-12)
 return corr.numpy(),{'A_row_indices':library[obs.numpy()],'A_query_indices':pos.numpy(),'cosine':best.numpy(),'weights':w.numpy()}
def main():
 a=args(__doc__);plan,rows,out=context(a);verify(out/'source');verify(out/'oldG/bankM64_attempt001');folder=out/'main';folder.mkdir();(folder/'predictions_PRIVATE').mkdir();(folder/'memory_PRIVATE').mkdir();start=time.perf_counter()
 import torch
 torch.set_num_threads(4)
 A=read(ROOT/'research/round24/input/D208_PRIVATE.json');fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');assert len(A)==208 and len(fb)==22
 for folder0 in [ROOT/'crossplatform/round24/capture336_attempt001',ROOT/'crossplatform/round23/bank272_attempt001']:
  registered=read(folder0/'SEALED.json')
  for r in A:
   rel='samples_PRIVATE/'+r['id']+'.npz';assert sha(folder0/rel)==registered['files'][rel]
 for pl in ('drone','quad'):
  aa=[r for r in A if r['platform']==pl];lookup={r['id']:i for i,r in enumerate(aa)};zz=[npz(ROOT/'crossplatform/round24/capture336_attempt001/samples_PRIVATE'/(r['id']+'.npz')) for r in aa];K=np.stack([make_key(z) for z in zz]);V24=[];V30=np.zeros((104,256),np.float32);counts=np.zeros_like(V30);ff=[f for f in fb if f['platform']==pl]
  for r,z in zip(aa,zz):
   b=npz(ROOT/'crossplatform/round23/bank272_attempt001/samples_PRIVATE'/(r['id']+'.npz'));FG=z['scores'].astype(float)+2*b['vision']+2*b['position']+2*b['relation'];V24.append(FG-z['scores'])
  V24=np.asarray(V24,np.float32)
  for f in ff:
   j=lookup[f['id_PRIVATE']];P=f['positive_group_PRIVATE'];N=f['negative_group_PRIVATE'];s=zz[j]['scores'];delta=max(.1-float(s[P].max()-s[N].max()),0)/2;V30[j,P]+=delta;V30[j,N]-=delta;counts[j,P]+=1;counts[j,N]+=1
  V30/=np.maximum(counts,1);E=np.array(sorted({lookup[f['id_PRIVATE']] for f in ff}),int);U=[i for i in range(104) if i not in set(E)];assert len(E)==(8 if pl=='drone' else 11) and np.count_nonzero(V30[U])==0
  np.savez_compressed(folder/'memory_PRIVATE'/(pl+'_library.npz'),A_keys=K,V24_dense_G_probability_increment=V24,V30_preference_probability_increment=V30,E_indices=E,U_indices=np.array(U))
  for r in [r for r in rows if r['platform']==pl]:
   z=npz(out/'source/samples_PRIVATE'/(r['id']+'.npz'));cl=npz(out/'source/closure_PRIVATE'/(r['id']+'.npz'));nm=cl['arm_names'].tolist();key=make_key(z);corrections={}
   for name,V,lib in [('R24_memory_alpha1',V24,np.arange(104)),('R30_memory_E_only',V30,E),('R30_memory_allA_zeroU',V30,np.arange(104))]:
    corr,nn=retrieve(key,K,V,lib);corrections[name]=corr;np.savez_compressed(folder/'memory_PRIVATE'/(r['id']+'_'+name+'.npz'),correction_probability_units=corr,**nn)
   s={'source':z['scores'],'R24_query_mlp_lr1_s16':cl['scores'][nm.index('R24_query_mlp_lr1_s16')],'R24_memory_alpha1':z['scores']+corrections['R24_memory_alpha1'],'R30_beta1':cl['scores'][nm.index('R30_beta1')],'R30_memory_E_only':z['scores']+corrections['R30_memory_E_only'],'R30_memory_allA_zeroU':z['scores']+corrections['R30_memory_allA_zeroU']}
   b=npz(out/'oldG/bankM64_attempt001/samples_PRIVATE'/(r['id']+'.npz'));FG=z['scores'].astype(float)+2*b['vision']+2*b['position']+2*b['relation'];delta=FG-z['scores'].astype(float)
   for name in BASE_ARMS:s[name+'_plus_current_oldG']=FG if name=='source' else s[name].astype(float)+delta
   names=list(s);ss=np.stack(list(s.values()));q=ss.argmax(-1);np.savez_compressed(folder/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(names),scores=ss,queries=q,boxes=z['boxes'][q],source_native_probability=z['scores'],oldG_F64=FG,shared_deltaG_F64=delta,source_delta_addback_F64_diagnostic=z['scores'].astype(float)+delta)
 seal(folder,'R46_TWELVE_MAIN_ARMS_ALL_PREDICTIONS_BEFORE_TASKSCORE',n=len(rows),arms=BASE_ARMS+[n+'_plus_current_oldG' for n in BASE_ARMS],same_A_information_controls=True,one_shared_currentG=True,seconds=time.perf_counter()-start);print('R46_MAIN_SEALED',len(rows),flush=True)
if __name__=='__main__':main()
