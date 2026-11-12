"""Contextual language routes actual native reads; no new semantic weights."""
from pathlib import Path
import sys,contextlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round87'))
from core87 import *
H88=ROOT/'research/round88';B88=ROOT/'crossplatform/round88'
MODES88=['full','no_relation','no_local','wrong_space','role_swap']
BASE88=['source','frozen_bi','frozen_ABC','R59','R87_full','R87_repaired','R87_hard']
ARMS88=BASE88+[v+'_'+r for v in MODES88 for r in ['direct','tta1']]+['full_tta3','physical_only1','no_rank1','no_correspondence1']

class Fusion88(Fusion85):
 def __init__(self,p):super().__init__(p);self.route=None;self.route_mode='off';self.reads=[]
 @contextlib.contextmanager
 def hooks(self,base):
  self.reads=[];hs=[]
  if self.route is not None and self.route_mode not in ['off','no_local']:
   # One fixed encoder interface and one fixed decoder interface, no layer search.
   def lv(module,args,kw,out):
    q=kw['query'];key=kw['key'];val=kw['value'];outv=out[0].clone();weights=out[1];route=self.route
    tm=route['token_roles'];supp=route['supports'];mix=route['weights_'+('plain' if self.route_mode=='no_relation' else 'relation')]
    if self.route_mode=='wrong_space':supp=supp[:,:,route['point_perm']]
    elif self.route_mode=='role_swap' and len(supp)>1:supp=supp.roll(1,0);mix=mix.roll(1,0)
    for e in range(len(tm)):
     ids=torch.where(tm[e])[0]
     if not len(ids):continue
     qq=q[ids];acc=qq*0
     for h in range(supp.shape[1]):
      if float(mix[e,h])==0:continue
      bias=(.05+.95*supp[e,h]).log()[None].expand(len(ids),-1)
      v,a=module(qq,key,val,attn_mask=bias,key_padding_mask=kw.get('key_padding_mask'),need_weights=True)
      acc=acc+mix[e,h]*v
      self.reads.append(F.normalize(v[:,0].mean(0).float(),dim=-1))
     outv[ids]=.5*outv[ids]+.5*acc
    return outv,weights
   # Calling module inside a forward hook would recurse; hook uses native functional MHA.
   def safe_lv(module,args,kw,out):
    original=module.forward
    def functional(q,k,v,**opts):
     return F.multi_head_attention_forward(q,k,v,module.embed_dim,module.num_heads,module.in_proj_weight,module.in_proj_bias,module.bias_k,module.bias_v,module.add_zero_attn,0.,module.out_proj.weight,module.out_proj.bias,training=False,key_padding_mask=opts.get('key_padding_mask'),need_weights=True,attn_mask=opts.get('attn_mask'),average_attn_weights=True)
    class Proxy:
     def __call__(self,*a,**k):return functional(*a,**k)
    return lv(Proxy(),args,kw,out)
   hs.append(base.cross_encoder.layers[-1].cross_layer.cross_lv.register_forward_hook(safe_lv,with_kwargs=True))
   def visual(_module,args,kw):
    b=self.route['decoder_plain' if self.route_mode=='no_relation' else 'decoder_relation']
    if self.route_mode=='wrong_space':b=b[:,self.route['point_perm']]
    elif self.route_mode=='role_swap':b=self.route['decoder_swapped']
    return args,{**kw,'attn_mask':(.25+.75*b).log()}
   hs.append(base.decoder[-1].cross_v.register_forward_pre_hook(visual,with_kwargs=True))
  try:yield
  finally:
   for h in hs:h.remove()
 def main(self,base,z,d,mode,*args,**kwargs):
  with self.hooks(base):
   out=super().main(base,z,d,mode,*args,**kwargs)
   out[2]['local_reads']=torch.stack(self.reads) if self.reads else out[1].new_zeros((0,288))
   return out

def tensor_route88(a):
 return {k:torch.as_tensor(v,device='cuda') for k,v in a.items() if k in ['token_roles','supports','weights_plain','weights_relation','decoder_plain','decoder_relation','decoder_swapped','point_perm','rank_pairs','rank_weights']}

def data88(r):
 z0,z,d=item80(r);a=load(B88/'evidence336_attempt001/samples_PRIVATE'/(r['id']+'.npz'));region={'point':a['region_point'],'image':a['region_image']};ab=[mask86(r,d,region,k) for k in [88001,88002]];pm=[x[0] for x in ab];im=[x[1] for x in ab];ws=local_weights85(d,pm,im,region)
 return z0,z,d,a,pm,im,ws

def objective88(m,base,z,d,refp,refq,refs,pm,im,ws,kind='full'):
 p,q,extra=m.main(base,z,d,'bidirectional');keep=(refp*(refp.clamp_min(1e-300).log()-p.clamp_min(1e-300).log())).sum()+.1*(q-refq).square().sum(-1).mean()/2
 corr=q.sum()*0
 for j,(a,b) in enumerate(zip(pm,im)):
  _,_,e=m.main(base,z,d,'bidirectional',point_mask=a,image_mask=b)
  if e['local_reads'].numel():corr=corr+(e['local_reads']-refs).square().sum(-1).mean()/4
 physical=aux82(m,z,d,pm,im,ws)
 pairs=m.route['rank_pairs'];ww=m.route['rank_weights'];rank=p.sum()*0
 if len(pairs):rank=(F.softplus(-(p[pairs[:,0]].log()-p[pairs[:,1]].log()))*ww).sum()/max(1,len(pairs))
 C=.1*corr if kind!='no_correspondence' else corr*0;R=.01*rank if kind not in ['no_rank','physical_only'] else rank*0
 if kind=='physical_only':C=C*0
 total=physical.double()+keep+C+R
 return total,p,{'physical':physical,'keep':keep,'correspondence':C,'rank':R}
