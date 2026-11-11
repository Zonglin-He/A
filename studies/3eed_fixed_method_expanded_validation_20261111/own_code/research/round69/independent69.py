"""Independent fixed R66 controls, evidence algebra, native encoder replay."""
import sys,torch,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round66'))
from independent66 import groups_and_pool,af,probability,replay,mapped

def controls(xyz,regions,indices):
 anchor=np.isin(np.arange(len(xyz)),indices);outside=~regions.any(0);qq=np.where(outside&anchor)[0];nn=np.where(outside&~anchor)[0];out=[]
 for region in regions:
  size=int(region.sum());nq=int((region&anchor).sum());good=size>=4 and size<=len(xyz)-4 and len(qq)>=nq and len(nn)>=size-nq;mask=np.zeros(len(xyz),bool)
  if good:
   pool=qq if nq else nn;origin=xyz[pool[np.linalg.norm(xyz[pool]-xyz[region].mean(0),axis=1).argmax()]];qa=qq[np.argsort(np.linalg.norm(xyz[qq]-origin,axis=1),kind='stable')[:nq]];na=nn[np.argsort(np.linalg.norm(xyz[nn]-origin,axis=1),kind='stable')[:size-nq]];mask[np.r_[qa,na]]=True
  out.append((mask,good))
 return out

def suffix(model,xyz,feat,ids):
 b=model.backbone_net;centers=xyz[:,ids,:];m=b.SA_modules[2];parts=[]
 for group,mlp in zip(m.groupers,m.mlps):parts.append(mlp(group(xyz,centers,feat)).max(dim=3).values)
 f3=torch.cat(parts,dim=1);f3=m.aggregation_layer(f3);x4,f4,_,_=b.SA_modules[3](centers,f3);out=b.fp1(centers,x4,f3,f4);return b.fp2(xyz,centers,feat,out).transpose(1,2).contiguous()

def readout(model,z,vis):
 t=lambda k:torch.as_tensor(z[k],device='cuda');V,L=model.cross_encoder(vis_feats=vis,pos_feats=t('pos0'),padding_mask=torch.zeros((1,1024),dtype=torch.bool,device='cuda'),text_feats=t('lang0'),text_padding_mask=t('textpad'),end_points={},detected_feats=None,detected_mask=None)
 _,features,_=model.gsample_module(t('xyz'),V.transpose(1,2).contiguous(),t('query_points_sample_inds').int()[None]);q=model.decoder_query_proj(features).transpose(1,2).contiguous()
 for j in range(6):q=model.decoder[j](q,V,L,t('position'+str(j)),None,t('textpad'),detected_feats=None,detected_mask=None)
 return {'h':q[0].cpu().numpy(),'query':torch.nn.functional.normalize(model.contrastive_align_projection_image(q),dim=-1)[0].cpu().numpy(),'tokens':torch.nn.functional.normalize(model.contrastive_align_projection_text(L),dim=-1)[0].cpu().numpy()}
