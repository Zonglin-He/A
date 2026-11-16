"""Original native path consuming the exact same geometrically aligned observations."""
from pathlib import Path
import sys
R111=Path(__file__).resolve().parents[2];H111=R111/'research/round111';B111=R111/'crossplatform/round111';sys.path.insert(0,str(R111/'research/round110'));from core110 import *
PRIMARY111='native_real_fusedABC';FAMILIES111=['real','randomA','wrong'];VIEWS111=['B','C','fusedABC','ensemble'];SWAPS111=['sourceA_prepared_head','preparedA_native_head_base','preparedA_prepared_head_base','preparedF_native_head_base','preparedF_prepared_head_base'];REFS111=jread(R111/'crossplatform/round110/MATRIX_SEALED.json')['arms'];ARMS111=REFS111+['native_'+k+'_'+v for k in FAMILIES111 for v in VIEWS111]+SWAPS111

def direct111(head,h,u,mask,late=None,residual=None):
 raw=head(h[None])[0]
 if late is not None:raw=raw+late
 q=F.normalize(raw,dim=-1);s=native_scores(q,u,mask.bool()).double()
 if residual is not None:s=s+residual.double()
 return s.softmax(0),q

def cpu111(h,u,mask,S):
 # Completely separate CPU float32 source projection, GELU and normalized readout.
 import torch.nn.functional as fn
 x=torch.from_numpy(h)
 for j in [0,2]:x=fn.relu(fn.linear(x,torch.from_numpy(S[str(j)+'.weight']),torch.from_numpy(S[str(j)+'.bias'])))
 x=fn.linear(x,torch.from_numpy(S['4.weight']),torch.from_numpy(S['4.bias']));q=fn.normalize(x,dim=-1);t=torch.from_numpy(u);m=torch.from_numpy(mask).bool();s=q@t.T/.07;s=s[:,m].logsumexp(-1)-s[:,~m].logsumexp(-1);return s.double().softmax(0).numpy(),q.numpy()

def sourceprefix111(base,r,src):
 z0,g,receipt=capture98(base,r,src);old=load(R111/'crossplatform/round110/original_PRIVATE'/(src+'_to_'+r['platform'])/(r['id']+'.npz'));assert np.array_equal(z0['boxes'],old['source_boxes']) and np.array_equal(z0['point_indices'],old['point_indices']);assert np.array_equal(z0['h'],old['h']) and np.array_equal(z0['query'],old['query']) and np.array_equal(z0['tokens'],old['tokens']);return z0,receipt

def original111(base,z):
 # interface66 replay uses source fixedpositions and resamples queries AFTER crossencoder.
 with torch.no_grad():f=replay(base,{k:(v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k,v in z.items()})
 return {k:torch.as_tensor(v,device='cuda') for k,v in f.items() if k in ['h','query','tokens','p']}|{'mask':torch.as_tensor(z['token_mask'],device='cuda').bool()}

def observations111(base,r,z0,ar,pairs,oldrec):
 ob={k:{} for k in FAMILIES111};proof={k:{} for k in FAMILIES111}
 for kind in ['real','randomA']:
  for slot in ['B','C']:
   pairid=ar[slot+'_pair'];v,pf=extra106(base,ar['A'],pairs[pairid],z0,kind,slot) if pairid else (None,{'status':'no_'+slot+'_candidate'});ob[kind][slot]=v;proof[kind][slot]=pf;prior=oldrec['families'][kind]['support'][slot]
   assert (v is not None)==oldrec['families'][kind][slot+'_available']
   if v is not None:assert all(pf[a]==prior[a] for a in ['point_sha256','sample_sha256','registration_sha256','aligned_seed_n'])
 for slot in ['B','C']:ob['wrong'][slot]=wrong106(ob['real'][slot],r,slot)
 return ob,proof

def swaps111(base,m,r,A,cache):
 out={};out['sourceA_prepared_head']=direct111(m.match,A['h'],A['tokens'],A['mask'])[0]
 for v,key in [('A','preparedA'),('fusedABC','preparedF')]:
  h=torch.as_tensor(cache['real_'+v+'_h'],device='cuda');u=torch.as_tensor(cache['real_'+v+'_u'],device='cuda');mask=torch.as_tensor(cache['real_'+v+'_mask'],device='cuda')
  for name,head in [('native',base.contrastive_align_projection_image),('prepared',m.match)]:out[key+'_'+name+'_head_base']=direct111(head,h,u,mask)[0]
 return out
