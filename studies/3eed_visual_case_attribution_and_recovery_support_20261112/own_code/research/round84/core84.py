"""Read-only selected-case diagnostics; no optimizer or pseudo-label production."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round82'))
from core82 import *
H84=ROOT/'research/round84';B84=ROOT/'crossplatform/round84'
STATES84=['frozen','raw','BT','raw_wrong','BT_wrong']
REF84={'source':'source','frozen':'frozen_bi','raw':'R80_raw_correct','BT':'bt_correct','raw_wrong':'R80_raw_wrong','BT_wrong':'bt_wrong'}
def original84(base,z):
 V,L=base.cross_encoder(vis_feats=z['vis0'],pos_feats=z['pos0'],padding_mask=torch.zeros((1,1024),device='cuda',dtype=torch.bool),text_feats=z['lang0'],text_padding_mask=z['textpad'],end_points={},detected_feats=None,detected_mask=None);ids=z['query_points_sample_inds'].long()[None];_,f,_=base.gsample_module(z['xyz'],V.transpose(1,2).contiguous(),ids.int());Q=base.decoder_query_proj(f).transpose(1,2).contiguous()
 for j,dec in enumerate(base.decoder):Q=dec(Q,V,L,z['position'+str(j)],None,z['textpad'],detected_feats=None,detected_mask=None)
 u=F.normalize(base.contrastive_align_projection_text(L.float()),dim=-1)[0];return Q[0],u
def score84(head,h,u,mask,late=None):
 raw=head(h.float());raw=raw if late is None else raw+late;q=F.normalize(raw,dim=-1);s=q@u.T/.07;v=torch.logsumexp(s[:,mask],-1)-torch.logsumexp(s[:,~mask],-1);return v.double().softmax(0)
def box_iou84(a,b):
 lo=np.maximum(a[:3]-a[3:]/2,b[:3]-b[3:]/2);hi=np.minimum(a[:3]+a[3:]/2,b[:3]+b[3:]/2);inter=np.maximum(hi-lo,0).prod();return float(inter/max(1e-12,np.maximum(a[3:],0).prod()+np.maximum(b[3:],0).prod()-inter))
def in_box84(x,b):return ((x>=b[:3]-b[3:]/2)&(x<=b[:3]+b[3:]/2)).all(-1)
def local_attention84(module,args):
 q,key,idx,weight=args;valid=(idx>=0)&(weight>0);ix=idx.clamp_min(0);K=module.k(key)[ix];logit=(module.q(q)[:,None]*K).sum(-1)/8+weight.clamp_min(1e-12).log();a=logit.masked_fill(~valid,-1e4).softmax(-1)*valid;return a/a.sum(-1,keepdim=True).clamp_min(1e-12)
