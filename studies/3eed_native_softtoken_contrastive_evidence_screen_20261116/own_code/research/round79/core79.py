"""Frozen R76 model, decision sensitivity, and fixed weighted physical recovery."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round76'))
from common76 import *
from model76 import *
H79=ROOT/'research/round79';B79=ROOT/'crossplatform/round79'
ARMS79=['source','frozen_bi','frozen_direct3','uniform_sgd1','query_raw_sgd1','query_radius_sgd1','shuffle_radius_sgd1','wrong_binding_radius_sgd1','R76_Adam1','R76_Adam3','frozen_ABC','R59']
class Fusion79(Fusion):
 def __init__(self,p):super().__init__(p);self.analysis_point_gate=None;self.analysis_image_gate=None
 def features(self,P,d,mode,image=None,point_mask=None,image_mask=None,reverse_off=False,wrong=False):
  x=d['image'] if image is None else image
  if image_mask is not None:
   mm=image_mask.reshape(1,1,32,48).float().repeat_interleave(8,2).repeat_interleave(8,3).bool();x=x.masked_fill(mm,.5)
  I=self.image_adapter(self.rgb((x-.5)*2).flatten(2).transpose(1,2)[0]);P=self.point_adapter(P)
  if self.analysis_point_gate is not None:P=P*self.analysis_point_gate[:,None]
  if self.analysis_image_gate is not None:I=I*self.analysis_image_gate[:,None]
  if point_mask is not None:P=P.masked_fill(point_mask[:,None],0)
  p2i=d['p2i'];i2p=d['i2p'];iw=d['p2i_weight'];rw=d['i2p_weight']
  if wrong:
   perm=d['wrong_permutation'];p2i=p2i[perm];iw=iw[perm];inverse=torch.argsort(perm);i2p=torch.where(i2p>=0,inverse[i2p.clamp_min(0)],i2p)
  if point_mask is not None:rw=rw*(~point_mask[i2p.clamp_min(0)])*(i2p>=0)
  II=I
  if mode=='bidirectional' and not reverse_off:II=I+torch.tanh(self.gate_I)*self.point_to_image(I,P,i2p,rw)
  PP=P
  if mode!='independent':PP=P+torch.tanh(self.gate_P)*self.image_to_point(P,II,p2i,iw)
  return PP,II,I
 def local_losses(self,z,d,pm,im,wrong=False):
  P=z['vis0'][0];PP,II,Ionly=self.features(P,d,'bidirectional',point_mask=pm,image_mask=im,wrong=wrong);rgb=self.point_image(II,d);rgb=rgb[d['wrong_permutation']] if wrong else rgb;estimate=F.normalize(self.pred_point(PP)+self.pred_point_image(rgb),dim=-1);target=F.normalize(P[:,:64].detach(),dim=-1);lp=(estimate-target).square().sum(-1)/2;trgb=F.avg_pool2d(d['image'],8,8).flatten(2).transpose(1,2)[0].detach();li=F.smooth_l1_loss(self.pred_color(II).sigmoid(),trgb,reduction='none').mean(-1);ld=F.smooth_l1_loss(self.pred_depth(Ionly).squeeze(-1),d['depth_target'].detach(),reduction='none');return lp,li,ld

def pair_set(boxes,xyz,score):
 lo=boxes[:,:3]-boxes[:,3:]/2;hi=boxes[:,:3]+boxes[:,3:]/2;support=((xyz[None]>=lo[:,None])&(xyz[None]<=hi[:,None])).all(-1);rep=[]
 for k in np.argsort(-score,kind='stable'):
  duplicate=False
  for j in rep:
   inter=np.maximum(0,np.minimum(hi[k],hi[j])-np.maximum(lo[k],lo[j])).prod();iou=inter/max(1e-12,np.maximum(0,hi[k]-lo[k]).prod()+np.maximum(0,hi[j]-lo[j]).prod()-inter);shared=int((support[k]&support[j]).sum());union=int((support[k]|support[j]).sum());jac=shared/max(1,union)
   if iou>=.9 or (iou>=.5 and jac>=.5 and shared>=4):duplicate=True;break
  if not duplicate:rep.append(int(k))
  if len(rep)==4:break
 return rep,[(rep[0],j) for j in rep[1:]]
def native_scores(q,u,mask):
 s=q@u.T/.07;return torch.logsumexp(s[:,mask],-1)-torch.logsumexp(s[:,~mask],-1)
def sensitivity(m,base,z,d,boxes):
 m.requires_grad_(False);gp=torch.ones(1024,device='cuda',requires_grad=True);gi=torch.ones(1536,device='cuda',requires_grad=True);m.analysis_point_gate=gp;m.analysis_image_gate=gi
 try:
  p,q,extra=m.main(base,z,d,'bidirectional');s=native_scores(q,extra['projected_text'],z['token_mask'].bool());reps,pairs=pair_set(boxes,z['xyz'][0].detach().cpu().numpy(),s.detach().cpu().numpy());rp=torch.zeros_like(gp);ri=torch.zeros_like(gi)
  for j,(a,b) in enumerate(pairs):
   gradients=torch.autograd.grad(s[a]-s[b],[gp,gi],retain_graph=j+1<len(pairs),allow_unused=True);rp+=torch.zeros_like(gp) if gradients[0] is None else gradients[0].detach().abs()/len(pairs);ri+=torch.zeros_like(gi) if gradients[1] is None else gradients[1].detach().abs()/len(pairs)
  return p.detach(),q.detach(),rp.cpu().numpy(),ri.cpu().numpy(),{'representatives_PRIVATE':reps,'pairs_PRIVATE':pairs,'pair_count':len(pairs),'point_saliency_sum':float(rp.sum()),'image_saliency_sum':float(ri.sum())}
 finally:m.analysis_point_gate=None;m.analysis_image_gate=None
def normalized_weights(r,valid,uniform=False):
 out=np.zeros(len(r),np.float64);ids=np.flatnonzero(valid);n=len(ids)
 if not n:return out,True
 if uniform or np.sum(r[ids])<=0:out[ids]=1/n;return out,not uniform
 a=np.maximum(r[ids].astype(float),0);p=np.zeros(n);left=np.ones(n,bool);mass=1.;cap=min(8/n,1.)
 while left.any():
  ii=np.flatnonzero(left);s=a[ii];proposal=mass*(s/s.sum() if s.sum()>0 else np.ones(len(ii))/len(ii));hot=proposal>cap+1e-15
  if not hot.any():p[ii]=proposal;break
  capped=ii[hot];p[capped]=cap;left[capped]=False;mass-=len(capped)*cap
 out[ids]=.25/n+.75*p;assert abs(out.sum()-1)<1e-10 and out[ids].min()>=.25/n-1e-12 and out.max()<=8/n+1e-10;return out,False
def weight_sets(r,d,pm,im,rp,ri):
 allweights={};fallback=0
 for kind in ['uniform','query','shuffle']:
  ws=[]
  for j,(p,i) in enumerate(zip(pm,im)):
   masks0={'P':(p&d['visible']).cpu().numpy(),'I':i.cpu().numpy(),'D':d['depth_valid'].cpu().numpy()};w={}
   for name,v in masks0.items():
    array,fb=normalized_weights(rp if name=='P' else ri,v,uniform=kind=='uniform');fallback+=int(fb and kind=='query')
    if kind=='shuffle':
     seed=int(hashlib.sha256(('R79shuffle|'+r['id']+'|'+str(j)+'|'+name).encode()).hexdigest()[:16],16);ii=np.flatnonzero(v);array[ii]=np.random.default_rng(seed).permutation(array[ii])
    w[name]=torch.tensor(array,dtype=torch.float32,device='cuda')
   ws.append(w)
  allweights[kind]=ws
 return allweights,fallback
def objective79(m,base,z,d,refp,refq,pm,im,ws,wrong=False):
 pp,q,_=m.main(base,z,d,'bidirectional');keep=(refp*(refp.clamp_min(1e-300).log()-pp.clamp_min(1e-300).log())).sum()+.1*(q-refq).square().sum(-1).mean()/2;terms=[]
 for p,i,w in zip(pm,im,ws):
  lp,li,ld=m.local_losses(z,d,p,i,wrong);terms.append(.1*(w['P']*lp).sum()+.1*(w['I']*li).sum()+.01*(w['D']*ld).sum())
 aux=torch.stack(terms).mean();return aux.double()+keep,aux,keep
def enable79(m):
 m.requires_grad_(False)
 for _,p in m.tta_parameters():p.requires_grad_(True)
def gradient79(m,base,z,d,refp,refq,pm,im,ws,wrong=False):
 enable79(m);L,a,k=objective79(m,base,z,d,refp,refq,pm,im,ws,wrong);ps=[p for _,p in m.tta_parameters()];gg=torch.autograd.grad(L,ps,allow_unused=True);g=torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten() for p,v in zip(ps,gg)]);norm=g.norm();delta=-.5*torch.clamp(1/(norm+1e-6),max=1)*g;return g,delta,{'objective_before':float(L.detach()),'auxiliary_before':float(a.detach()),'preservation_before':float(k.detach()),'gradient_norm':float(norm),'raw_proposal_L2':float(delta.norm())}
def set_step(m,S,delta):
 off=0
 with torch.no_grad():
  for n,p in m.tta_parameters():nn=p.numel();p.copy_(torch.tensor(S[n],device='cuda')+delta[off:off+nn].reshape_as(p));off+=nn
 assert off==len(delta)
def maskpair79(r,d):
 a=[masks(r,d,k) for k in [76001,76002]];return [x[0] for x in a],[x[1] for x in a]
