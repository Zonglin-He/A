import torch,numpy as np,time
from common24 import *
from fit24 import interact,teacher

def main():
 torch.set_num_threads(4);out=BASE/'memory_selected_scoresM64_attempt001';out.mkdir(exist_ok=False);(out/'samples_PRIVATE').mkdir();selection=read(EXP/'MEMORY_SELECTED_LOCK.json');alpha=float(selection['selected']['name'].split('_a')[1]);start=time.perf_counter()
 for platform in ['drone','quad']:
  A=[r for r in rows('D208') if r['platform']==platform];zz=[feature(r) for r in A];keys=np.stack([interact(z) for z in zz]);keys/=np.maximum(np.linalg.norm(keys,axis=-1,keepdims=True),1e-12);values=np.stack([teacher(r,z)-z['scores'] for r,z in zip(A,zz)]).astype(np.float32);K=torch.tensor(keys);V=torch.tensor(values)
  for r in [r for r in rows('M64') if r['platform']==platform]:
   z=feature(r);q=interact(z);q/=np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12);sim=torch.einsum('qc,anc->qan',torch.tensor(q),K);near,which=sim.max(-1);best,obs=near.topk(4,dim=-1);pos=which.gather(1,obs);y=V[obs,pos];weights=((best-1)/.1).exp();correction=(y*weights).sum(-1)/weights.sum(-1).clamp_min(1e-12);score=z['scores']+alpha*correction.numpy();old=load_npz(BASE/'memory_attempt001/M64_predictions/predictions_PRIVATE'/(r['id']+'.npz'));ix=old['arm_names'].tolist().index(selection['selected']['name']);assert int(score.argmax())==int(old['query_indices'][ix]);np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),scores=score)
 seal(out,'R24_selected_same_A_cache_scores_no_M_feedback',n=64,optimizer_steps=0,seconds=time.perf_counter()-start,selection_sha256=sha(EXP/'MEMORY_SELECTED_LOCK.json'));print('MEMORY_SCORES24_SEALED')
if __name__=='__main__':main()
