"""Two independent tiny modules, no backbone or CLIP update."""
import torch,numpy as np
from common70 import *

class ScoreAdapter(torch.nn.Module):
 def __init__(self):
  super().__init__();self.down=torch.nn.Linear(1089,16);self.up=torch.nn.Linear(16,1);torch.nn.init.zeros_(self.up.weight);torch.nn.init.zeros_(self.up.bias)
 def forward(self,x):return self.up(torch.relu(self.down(x))).squeeze(-1)

class RGBAdapter(torch.nn.Module):
 def __init__(self):
  super().__init__();self.net=torch.nn.Sequential(torch.nn.Conv2d(3,16,3,padding=1),torch.nn.ReLU(),torch.nn.AvgPool2d(2),torch.nn.Conv2d(16,32,3,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(32,16),torch.nn.ReLU(),torch.nn.Linear(16,288))
 def forward(self,x):return torch.nn.functional.normalize(torch.relu(self.net(x)),dim=-1,eps=1e-6)

def snapshot(m):return {k:v.detach().cpu().numpy().copy() for k,v in m.state_dict().items()}
def phash(m):
 import hashlib
 h=hashlib.sha256()
 for k,v in snapshot(m).items():h.update(k.encode());h.update(v.tobytes())
 return h.hexdigest()

def evidence(a,c,kind='subject',joint=True,reliable=True,shuffle=False):
 visual=c['visual'].mean(1) if joint else c['visual'][:,0];t=c['subject_text' if kind=='subject' else 'full_text'];sim=visual@t
 if shuffle:sim=sim[perm(len(sim),'R70Avisualbinding')]
 r=a['reliability'] if reliable else a['project_reliability'];center=float(np.sum(r*sim)/r.sum()) if r.sum()>0 else 0.;e=r*(sim-center)*float(c['logit_scale']);return e[a['query_group']]

def A_input(z,a,c,context=False):
 visual=c['visual'][:,1] if context else c['visual'].mean(1);v=visual[a['query_group']];t=np.broadcast_to(c['subject_text'],v.shape);r=a['reliability'][a['query_group'],None];return np.column_stack([z['query'],v,t,r]).astype(np.float32)

def edit_vis(z,a,pred):
 out=z['vis0'].copy();seeds=a['seed_indices'][a['held_patch_indices']];base=out[0,seeds];unit=base/np.maximum(np.linalg.norm(base,axis=1,keepdims=True),1e-6);delta=pred-unit;delta/=np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1.);out[0,seeds]+=np.float32(.2)*np.linalg.norm(base,axis=1,keepdims=True)*delta;return out

def NW(patches,fit,held,target,wrong=False):
 from scipy.special import softmax
 x=patches.astype(np.float32).reshape(-1,3,4,4,4,4).mean((3,5)).reshape(len(patches),48)/255
 if wrong:x=x[np.roll(np.arange(len(x)),1)]
 dist=np.maximum((x[held]**2).sum(1)[:,None]+(x[fit]**2).sum(1)[None]-2*x[held]@x[fit].T,0);k=min(8,len(fit));ids=np.argsort(dist,axis=1,kind='stable')[:,:k];dd=np.take_along_axis(dist,ids,axis=1);scale=max(float(np.median(dd)),1e-4);w=softmax(-dd/scale,axis=1);pred=np.sum(w[:,:,None]*target[fit][ids],axis=1);return pred/np.maximum(np.linalg.norm(pred,axis=1,keepdims=True),1e-6)
