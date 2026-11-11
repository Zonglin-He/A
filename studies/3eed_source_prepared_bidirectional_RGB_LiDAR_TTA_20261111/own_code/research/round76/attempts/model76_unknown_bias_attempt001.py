"""Small RGB branch + sparse sequential bidirectional fusion, no expert network."""
from common76 import *
import torch,torch.nn as nn,torch.nn.functional as F,copy

class Adapter(nn.Module):
 def __init__(self,d):
  super().__init__();self.down=nn.Linear(d,16);self.up=nn.Linear(16,d);nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
 def forward(self,x):return x+self.up(F.gelu(self.down(x)))
class LocalAttention(nn.Module):
 def __init__(self,dq,dk,out):
  super().__init__();self.q=nn.Linear(dq,64);self.k=nn.Linear(dk,64);self.v=nn.Linear(dk,64);self.o=nn.Linear(64,out)
 def forward(self,q,key,index,weight):
  valid=(index>=0)&(weight>0);idx=index.clamp_min(0);K=self.k(key)[idx];V=self.v(key)[idx];s=(self.q(q)[:,None,:]*K).sum(-1)/8+weight.clamp_min(1e-12).log();s=s.masked_fill(~valid,-1e4);a=s.softmax(-1)*valid;a=a/a.sum(-1,keepdim=True).clamp_min(1e-12);v=(a[:,:,None]*V).sum(1);return self.o(v)*valid.any(1)[:,None]
class Fusion(nn.Module):
 def __init__(self,original_projection):
  super().__init__();self.rgb=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.GroupNorm(4,16),nn.GELU(),nn.Conv2d(16,32,3,2,1),nn.GroupNorm(4,32),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.GELU());self.image_adapter=Adapter(64);self.point_adapter=Adapter(288);self.point_to_image=LocalAttention(64,288,64);self.image_to_point=LocalAttention(288,64,288);self.gate_I=nn.Parameter(torch.zeros(()));self.gate_P=nn.Parameter(torch.zeros(()));self.gate_late=nn.Parameter(torch.zeros(()));self.late=nn.Linear(64,64);self.match=copy.deepcopy(original_projection);self.pred_point=nn.Linear(288,64);self.pred_point_image=nn.Linear(64,64);self.pred_color=nn.Linear(64,3);self.pred_depth=nn.Linear(64,1)
 def features(self,P,d,mode,image=None,point_mask=None,image_mask=None,reverse_off=False,wrong=False):
  x=d['image'] if image is None else image
  if image_mask is not None:
   mm=image_mask.reshape(1,1,32,48).float().repeat_interleave(8,2).repeat_interleave(8,3).bool();x=x.masked_fill(mm,.5)
  I=self.image_adapter(self.rgb((x-.5)*2).flatten(2).transpose(1,2)[0]);P=self.point_adapter(P)
  if point_mask is not None:P=P.masked_fill(point_mask[:,None],0)
  p2i=d['p2i'];i2p=d['i2p'];iw=d['p2i_weight'];rw=d['i2p_weight']
  if wrong:
   # Permute logical point↔pixel bindings, not the depth targets or data labels.
   perm=d['wrong_permutation'];p2i=p2i[perm];iw=iw[perm];inverse=torch.argsort(perm);i2p=torch.where(i2p>=0,inverse[i2p.clamp_min(0)],i2p)
  if point_mask is not None:
   rw=rw*(~point_mask[i2p.clamp_min(0)])*(i2p>=0)
  II=I
  if mode=='bidirectional' and not reverse_off:II=I+torch.tanh(self.gate_I)*self.point_to_image(I,P,i2p,rw)
  PP=P
  if mode!='independent':PP=P+torch.tanh(self.gate_P)*self.image_to_point(P,II,p2i,iw)
  return PP,II,I
 def point_image(self,I,d):
  valid=d['p2i']>=0;ww=d['p2i_weight']*valid;ww=ww/ww.sum(1,keepdim=True).clamp_min(1e-12);return (I[d['p2i'].clamp_min(0)]*ww[:,:,None]).sum(1)
 def main(self,base,z,d,mode,reverse_off=False,image=None,point_mask=None,image_mask=None):
  P=z['vis0'][0];P,I,Ionly=self.features(P,d,mode,image,point_mask,image_mask,reverse_off);V,L=base.cross_encoder(vis_feats=P[None],pos_feats=z['pos0'],padding_mask=torch.zeros((1,len(P)),device=P.device,dtype=torch.bool),text_feats=z['lang0'],text_padding_mask=z['textpad'],end_points={},detected_feats=None,detected_mask=None);ids=z['query_points_sample_inds'].long()[None];_,f,_=base.gsample_module(z['xyz'],V.transpose(1,2).contiguous(),ids.int());Q=base.decoder_query_proj(f).transpose(1,2).contiguous()
  for j,dec in enumerate(base.decoder):Q=dec(Q,V,L,z['position'+str(j)],None,z['textpad'],detected_feats=None,detected_mask=None)
  image_q=self.point_image(I,d)[ids[0]]
  with torch.autocast(device_type='cuda',enabled=False):
   raw=self.match(Q.float())[0]+torch.tanh(self.gate_late)*self.late(image_q.float());q=F.normalize(raw,dim=-1);u=F.normalize(base.contrastive_align_projection_text(L.float()),dim=-1)[0];S=q@u.T/.07;mask=z['token_mask'].bool();odds=torch.logsumexp(S[:,mask],-1)-torch.logsumexp(S[:,~mask],-1)
  return odds.double().softmax(0),q,{'projected_text':u,'query_hidden':Q[0]}
 def auxiliary(self,z,d,mode,pm,im,wrong=False):
  P=z['vis0'][0];PP,II,Ionly=self.features(P,d,mode,point_mask=pm,image_mask=im,wrong=wrong);rgb_at_point=self.point_image(II,d);rgb_at_point=rgb_at_point[d['wrong_permutation']] if wrong else rgb_at_point;estimate=F.normalize(self.pred_point(PP)+self.pred_point_image(rgb_at_point),dim=-1);target=F.normalize(P[:,:64].detach(),dim=-1);ok=pm&d['visible'];lp=(estimate[ok]-target[ok]).square().sum(-1).mean()/2 if ok.any() else estimate.sum()*0;image0=d['image'];targetRGB=F.avg_pool2d(image0,8,8).flatten(2).transpose(1,2)[0].detach();colors=self.pred_color(II).sigmoid();li=F.smooth_l1_loss(colors[im],targetRGB[im]) if im.any() else colors.sum()*0
  # NEVER use II, PP, depth/correspondence features as depth predictor inputs.
  logdepth=self.pred_depth(Ionly).squeeze(-1);okd=d['depth_valid'];ld=F.smooth_l1_loss(logdepth[okd],d['depth_target'][okd]) if okd.any() else logdepth.sum()*0
  loss=.1*lp+.1*li+.01*ld;return loss,{'point':lp,'RGB':li,'depth':ld,'point_targets':int(ok.sum()),'RGB_targets':int(im.sum()),'depth_targets':int(okd.sum())}
 def tta_parameters(self):
  out=[]
  for name,p in self.named_parameters():
   if name.startswith(('image_adapter.','point_adapter.','point_to_image.','image_to_point.')) or name in ['gate_I','gate_P']:out.append((name,p))
  return out

def tensor_prefix(z,device='cuda'):return {k:torch.as_tensor(v,device=device) for k,v in z.items() if k in ['vis0','pos0','lang0','textpad','xyz','query_points_sample_inds','token_mask'] or k.startswith('position')}
def tensor_data(data,r,device='cuda'):
 d={k:torch.as_tensor(v,device=device) for k,v in data.items() if k in ['p2i','p2i_weight','i2p','i2p_weight','visible','cell','depth_target','depth_valid','query_anchor_indices']};d['image']=torch.as_tensor(data['image'].copy(),device=device,dtype=torch.float32).permute(2,0,1)[None]/255;ss=int(hashlib.sha256(('R76wrongpixelbind|'+r['id']).encode()).hexdigest()[:16],16);d['wrong_permutation']=torch.as_tensor(np.random.default_rng(ss).permutation(1024),device=device);return d

def masks(r,d,repeat):
 ss=int(hashlib.sha256(('R76complementarymask|'+r['id']+'|'+str(repeat)).encode()).hexdigest()[:16],16);rng=np.random.default_rng(ss);cells=torch.nonzero(d['depth_valid'],as_tuple=True)[0].cpu().numpy();order=rng.permutation(cells);n=len(order)//4;im=torch.zeros(1536,device=d['image'].device,dtype=torch.bool);im[torch.as_tensor(order[:n],device=im.device)]=True;pm=d['visible']&torch.isin(d['cell'],torch.as_tensor(order[n:2*n],device=im.device));pm[d['query_anchor_indices'].long()]=False;assert not pm[d['query_anchor_indices'].long()].any() and not im[d['cell'][pm]].any();return pm,im

def state(m):return {k:v.detach().cpu().numpy().copy() for k,v in m.state_dict().items()}
def restore(m,s):m.load_state_dict({k:torch.tensor(v,device=next(m.parameters()).device) for k,v in s.items()},strict=True)
