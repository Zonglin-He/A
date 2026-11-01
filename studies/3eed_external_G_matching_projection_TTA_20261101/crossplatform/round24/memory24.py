"""Same-A-information nonparametric correction cache; no model updates/current G.
Each unseen candidate gets at most one nearest candidate per A observation,
then four nearest observations are weighted by exp((cos-1)/.1).
"""
import torch,numpy as np,time
from common24 import *
from fit24 import interact,teacher,write_predictions

def main():
 torch.set_num_threads(4);out=BASE/'memory_attempt001';out.mkdir(exist_ok=False);start=time.perf_counter();pred={};metadata=[]
 for platform in ['drone','quad']:
  A=[r for r in rows('D208') if r['platform']==platform];zz=[feature(r) for r in A];keys=np.stack([interact(z) for z in zz]);keys/=np.maximum(np.linalg.norm(keys,axis=-1,keepdims=True),1e-12);values=np.stack([teacher(r,z)-z['scores'] for r,z in zip(A,zz)]).astype(np.float32);K=torch.tensor(keys);V=torch.tensor(values);metadata.append({'platform':platform,'observations':len(A),'keys':list(keys.shape),'float32_cache_bytes':keys.nbytes+values.nbytes})
  for r in [r for r in rows() if r['platform']==platform and r['stage'] in ['K','M']]:
   z=feature(r);q=interact(z);q/=np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12);sim=torch.einsum('qc,anc->qan',torch.tensor(q),K);near,which=sim.max(-1);best,obs=near.topk(4,dim=-1);pos=which.gather(1,obs);y=V[obs,pos];weights=((best-1)/.1).exp();correction=(y*weights).sum(-1)/weights.sum(-1).clamp_min(1e-12);s0=z['scores'];corr=correction.numpy();pred[r['id']]={f'memory_a{alpha}':int((s0+alpha*corr).argmax()) for alpha in [.25,1.]}
 for split in ['V64','M64']:
  rr=rows(split);write_predictions(out/(split+'_predictions'),rr,{k:np.array([pred[r['id']][k] for r in rr]) for k in ['memory_a0.25','memory_a1.0']})
 dump(out/'CACHE_RECEIPT.json',metadata);seal(out,'R24_same_A_feedback_nonparametric_cache',seconds=time.perf_counter()-start,no_M_G=True,learned_model_parameters=0,optimizer_steps=0);print('MEMORY24_SEALED')
if __name__=='__main__':main()
