"""Independent manual SA3 and native fixed-anchor encoder/decoder."""
import torch,numpy as np
def suffix(model,xyz,feat,ids):
 b=model.backbone_net;centers=xyz[:,ids,:];m=b.SA_modules[2];parts=[]
 for group,mlp in zip(m.groupers,m.mlps):parts.append(mlp(group(xyz,centers,feat)).max(dim=3).values)
 f3=torch.cat(parts,dim=1);f3=m.aggregation_layer(f3);x4,f4,_,_=b.SA_modules[3](centers,f3);out=b.fp1(centers,x4,f3,f4);return b.fp2(xyz,centers,feat,out).transpose(1,2).contiguous()

def readout(model,z,vis):
 t=lambda k:torch.as_tensor(z[k],device='cuda');V,L=model.cross_encoder(vis_feats=vis,pos_feats=t('pos0'),padding_mask=torch.zeros((1,1024),dtype=torch.bool,device='cuda'),text_feats=t('lang0'),text_padding_mask=t('textpad'),end_points={},detected_feats=None,detected_mask=None)
 _,features,_=model.gsample_module(t('xyz'),V.transpose(1,2).contiguous(),t('query_points_sample_inds').int()[None]);q=model.decoder_query_proj(features).transpose(1,2).contiguous()
 for j in range(6):q=model.decoder[j](q,V,L,t('position'+str(j)),None,t('textpad'),detected_feats=None,detected_mask=None)
 return {'h':q[0].cpu().numpy(),'query':torch.nn.functional.normalize(model.contrastive_align_projection_image(q),dim=-1)[0].cpu().numpy(),'tokens':torch.nn.functional.normalize(model.contrastive_align_projection_text(L),dim=-1)[0].cpu().numpy()}
