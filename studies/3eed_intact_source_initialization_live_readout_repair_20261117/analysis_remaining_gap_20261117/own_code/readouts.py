"""Fixed cached-head ablations, sealed before new GT comparison; no model adaptation."""
from common import *
import numpy as np,torch
from torch.nn import functional as F
def main():
 check();torch.set_num_threads(4);out=H/'readouts_PRIVATE';out.mkdir(exist_ok=False);n=0;counts={};files={}
 for src in ['waymo','drone','quad']:
  with np.load(ROOT/'crossplatform/round135'/(src+'_SOURCE_MODEL_PRIVATE.npz')) as f:W={k:torch.from_numpy(f[k].copy()) for k in f.files if k.startswith('match.')}
  for D in sorted(B.glob(src+'_to_*')):
   od=out/D.name;od.mkdir()
   for p in sorted((D/'samples_PRIVATE').glob('*.npz')):
    a=np.load(p);ps=dict(zip(a['arm_names'],a['scores']));ff=np.load(D/'features_PRIVATE'/p.name);x=torch.from_numpy(ff['intact_A_h'].copy())[None]
    for j in [0,2]:x=F.relu(F.linear(x,W['match.'+str(j)+'.weight'],W['match.'+str(j)+'.bias']))
    x=F.linear(x,W['match.4.weight'],W['match.4.bias'])[0];u=torch.from_numpy(ff['intact_A_u'].copy());late=torch.from_numpy(ff['intact_A_late'].copy());mask=torch.from_numpy(ff['intact_A_mask'].copy()).bool()
    def score(q):
     z=F.normalize(q,dim=-1)@u.T/.07;return (z[:,mask].logsumexp(-1)-z[:,~mask].logsumexp(-1)).double()
    native=score(x+late);without=score(x);full=torch.from_numpy(np.log(np.maximum(ps['intact_live_A'],1e-300)));probs=torch.stack([native.softmax(0),without.softmax(0),(full+without-native).softmax(0)]).numpy();assert np.isfinite(probs).all();dest=od/p.name;np.savez_compressed(dest,scores=probs,arm_names=np.array(['base_A_no_competition','base_A_no_late_no_competition','full_A_no_late_fixed_competition']));files[str(dest.relative_to(H))]=sha(dest);n+=1
    if n%256==0:print('READOUT_DIAG',n,flush=True)
 assert n==1798;dump(H/'READOUTS_SEALED.json',{'n':n,'files':files,'GT_read':False,'new_full_model_forward':False,'training_or_updates':False,'posthoc_diagnostic_not_method_selection':True,'note':'noLate removes only final RGB delta; RGB-fused h/u and original competition remain in fixed-comp arm'});print('READOUTS_SEALED',n,flush=True)
if __name__=='__main__':main()
