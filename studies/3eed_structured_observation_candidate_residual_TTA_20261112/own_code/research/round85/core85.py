"""Structured observation -> anchored candidate adaptation; frozen original boxes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round84'))
from core84 import *
from evidence85 import *
H85=ROOT/'research/round85';B85=ROOT/'crossplatform/round85'
ARMS85=['source','frozen_bi','R82_BT','frozen_ABC','R59','direct_constraints','direct_multiview_constraints','mismatch_direct']+[v+str(k) for v in ['full','no_constraints','protected','mismatch'] for k in [1,3]]
class Fusion85(Fusion79):pass
def logits85(base,z,h,u):
 q=F.normalize(base.contrastive_align_projection_image(h.float()),dim=-1);S=q@u.T/.07;mask=z['token_mask'].bool();return (torch.logsumexp(S[:,mask],-1)-torch.logsumexp(S[:,~mask],-1)).double()
def phrase85(base,z,phrase):
 sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import firstnoun_mask
 text=phrase+' . not mentioned';tok=base.tokenizer([text],padding='longest',return_tensors='pt').to('cuda');zz=dict(z);zz['lang0']=base.text_projector(base.text_encoder(**tok).last_hidden_state);zz['textpad']=tok.attention_mask.ne(1).bool();zz['token_mask']=firstnoun_mask(base,text)[0].cuda();h,u=original84(base,zz);return logits85(base,zz,h,u)
def mask85(r,d,region,repeat,protected=False):
 cells=np.flatnonzero(d['depth_valid'].cpu().numpy());local=region['image'].astype(bool);seed=int(hashlib.sha256(('R85localmask|'+r['id']+'|'+str(repeat)).encode()).hexdigest()[:16],16);rg=np.random.default_rng(seed);n=len(cells)//4
 # Half of selected cells preferentially cover candidate hypotheses; the rest global.
 order=[];remaining=cells.copy()
 for count in [n,n]:
  a=rg.permutation(remaining[local[remaining]])[:int(np.ceil(.75*count))];rest=rg.permutation(remaining[~np.isin(remaining,a)])[:count-len(a)];take=np.r_[a,rest];order.append(take);remaining=remaining[~np.isin(remaining,take)]
 im=torch.zeros(1536,device='cuda',dtype=torch.bool);im[torch.as_tensor(order[0],device='cuda')]=True;pm=d['visible']&torch.isin(d['cell'],torch.as_tensor(order[1],device='cuda'))
 if protected:pm[d['query_anchor_indices'].long()]=False
 assert not im[d['cell'][pm]].any()
 return pm,im
def local_weights85(d,pm,im,region):
 out=[]
 for p,i in zip(pm,im):
  one={}
  for k,valid,local in [('P',p&d['visible'],region['point']),('I',i,region['image']),('D',d['depth_valid'],region['image'])]:
   a=valid.float();b=a*torch.as_tensor(local,device='cuda');one[k]=.25*a/a.sum().clamp_min(1)+.75*(b/b.sum().clamp_min(1) if b.any() else a/a.sum().clamp_min(1))
  out.append(one)
 return out
def region85(ev,entity_scores,s0,g):
 labels=ev['labels'];reps=ev['reps'];chosen=[]
 for score in [s0]+list(entity_scores):
  ranked=sorted(range(len(reps)),key=lambda k:float(np.max(score[labels==k])),reverse=True)[:4];chosen += [reps[k] for k in ranked]
 p=ev['support'][np.unique(chosen)].any(0);ii=np.zeros(1536,bool);ii[g['cell'][p&g['visible']]]=True
 return {'point':p,'image':ii}
def auxiliary85(m,z,d,pm,im,ws):return aux82(m,z,d,pm,im,ws)
def anchored85(fused,baseline,s0):return torch.softmax(s0+(fused.clamp_min(1e-300).log()-baseline.clamp_min(1e-300).log()),-1)
def objective85(m,base,z,d,s0,pi,baselines,pm,im,ws):
 full,_,_=m.main(base,z,d,'bidirectional');p=anchored85(full,baselines[0],s0);L=(pi*(pi.clamp_min(1e-300).log()-p.clamp_min(1e-300).log())).sum();masked=[]
 for n,(a,b) in enumerate(zip(pm,im)):
  f,_,_=m.main(base,z,d,'bidirectional',point_mask=a,image_mask=b);pv=anchored85(f,baselines[n+1],s0);masked.append((pi*(pi.clamp_min(1e-300).log()-pv.clamp_min(1e-300).log())).sum())
 klass=L+torch.stack(masked).mean();rec=auxiliary85(m,z,d,pm,im,ws);return klass+rec.double(),p,klass,rec
def patch85(m):return {n:v.detach().cpu().numpy().copy() for n,v in m.tta_parameters()}
