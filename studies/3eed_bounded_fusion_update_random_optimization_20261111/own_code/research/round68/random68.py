"""Two prespecified repeats; original R67 D256/SA3/anchors unchanged."""
import torch,numpy as np,hashlib,sys
from common68 import ROOT
sys.path.insert(0,str(ROOT/'research/round67'));from sampling67 import suffix
sys.path.insert(0,str(ROOT/'research/round66'));from interface66 import replay

def indices(native,key,repeat):
 assert repeat in [1,2];base=native[:256].astype(np.int64).tolist();assert len(set(base))==256
 rng=np.random.default_rng(int(hashlib.sha256(('R68random_repeat'+str(repeat)+'|'+key).encode()).hexdigest()[:16],16));result=base.copy();seen=set(base)
 for k in rng.permutation(1024):
  if int(k) not in seen:result.append(int(k));seen.add(int(k))
  if len(result)==512:break
 return np.array(result,np.int64)
def predict(model,z,repeat,key):
 ix=indices(z['native_SA3_indices'],key,repeat);xyz=torch.tensor(z['SA2_xyz'],device='cuda');feat=torch.tensor(z['SA2_feat'],device='cuda');vis,_=suffix(model,xyz,feat,ix);out=replay(model,z,alternate_vis0=vis.cpu().numpy());out['SA3_indices']=ix;return out
