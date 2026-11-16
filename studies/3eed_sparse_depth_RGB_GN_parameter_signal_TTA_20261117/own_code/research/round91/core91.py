"""Source-safe frozen-base competition scorer; original R80 physical scope retained."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round90'))
from core90 import *
H91=ROOT/'research/round91';B91=ROOT/'crossplatform/round91';BASES91=['R76','R89continue'];HEADS91=['query_disc','full_ce','full_disc','no_ref_disc','no_local_disc']
R89CONT=ROOT/read(B89/'trained_source_attempt001/SELECTED_SOURCE_ONLY.json')['models']['continue']['state_path']

class Interface91(Interface90):
 def forward(self,h,local,own,text,reltext,route,variant='full'):
  # Remove the direct query shortcut in spatial heads, keeping query-conditioned routing.
  # This is an output-input intervention, not an extra network or learned gate.
  if variant=='query_only':return super().forward(h,local,own,text,reltext,route,variant)
  if len(local)==0:return h.sum(-1)*0,{'conditional':h.new_zeros((256,1,64)),'conditional_valid':torch.zeros((256,1),device=h.device,dtype=torch.bool),'subject_weights':h.new_zeros((256,0))}
  hh=self.out.register_forward_pre_hook(lambda _m,args:(torch.cat([args[0][:,:64]*0,args[0][:,64:]],-1),))
  if variant=='no_local':local=local*0;own=own*0;variant='full'
  try:return super().forward(h,local,own,text,reltext,route,variant)
  finally:hh.remove()

class Fusion91(Fusion90):
 def __init__(self,p):super().__init__(p);self.interface=Interface91()
 def tta_parameters(self):return Fusion.tta_parameters(self)
 def main(self,base,z,d,mode,*args,**kw):
  # Fusion90 calls interface with its actual live native inputs, and returns its conditional state.
  return super().main(base,z,d,mode,*args,**kw)

def variant91(head):return {'query_disc':'query_only','no_ref_disc':'no_ref','no_local_disc':'no_local'}.get(head,'full')
def loadbase91(m,name):
 s=load(R76_INIT if name=='R76' else R89CONT);miss,extra=m.load_state_dict({k:torch.as_tensor(v,device='cuda') for k,v in s.items() if not k.startswith('interface.')},strict=False);assert not extra and all(k.startswith('interface.') for k in miss);m.use_interface=True;m.variant='full';m.pool_mode='strict';return state(m)
def data91(r,source=False):
 z0=prefix(r);z=tensor_prefix(z0);d=tensor_data(load(casepath(r)),r);f=B89/'source_features640_attempt001/routes_PRIVATE' if source else B89/'target_evidence336_attempt001/samples_PRIVATE';rt=load(f/(r['id']+'.npz'));route=tensor_route89(rt);pm,im=maskpair79(r,d);ws=weight_sets({'id':'UNUSED_UNIFORM'},d,pm,im,np.zeros(1024),np.zeros(1536))[0]['uniform'];return z0,z,d,rt,route,pm,im,ws
def native_feature91(m,base,z,d,pm=None,im=None):
 m.use_interface=False;m.variant='full';p,q,e=m.main(base,z,d,'bidirectional',point_mask=pm,image_mask=im);return {k:e[j] for k,j in [('h','query_hidden'),('u','projected_text'),('late','late_RGB_delta'),('local','local'),('own','own'),('text','text'),('relation_text','relation_text')]}|{'mask':z['token_mask'],'baseline':p}
def cached91(m,a,route,head):
 q=F.normalize(m.match(a['h'][None])[0]+a['late'],dim=-1);s=native_scores(q,a['u'],a['mask'].bool()).double();delta,e=m.interface(a['h'],a['local'],a['own'],a['text'],a['relation_text'],route,variant91(head));e['projected_query']=q;e['native_scores']=s;e['residual']=delta;score=s+delta.double();return score.softmax(0),e,score
def cached_views91(r):
 a=load(B91/'source_features640_attempt001/samples_PRIVATE'/(r['id']+'.npz'));return [{k:torch.as_tensor(a[str(v)+'_'+k],device='cuda') for k in ['h','u','late','local','own','text','relation_text','mask','baseline']} for v in range(3)]
def source_loss91(m,views,route,y,pairs,head):
 gs=[];ds=[]
 for a in views:
  p,e,s=cached91(m,a,route,head);gs.append(ground89(p,y));ds.append(F.softplus(s[pairs[:,1]]-s[pairs[:,0]]+.2).mean() if len(pairs) else s.sum()*0)
 ground=gs[0]+torch.stack(gs[1:]).mean();disc=ds[0]+torch.stack(ds[1:]).mean();return ground+(disc if head!='full_ce' else 0),{'ground':ground,'disc':disc}
def uniform_aux91(m,z,d,pm,im,ws,wrong=False):
 vals=[]
 for a,b,w in zip(pm,im,ws):
  P,I,D=m.local_losses(z,d,a,b,wrong);vals.append(.1*(w['P']*P).sum()+.1*(w['I']*I).sum()+.01*(w['D']*D).sum())
 return torch.stack(vals).mean()
def freeze91(m):m.requires_grad_(False)
def enable_physical91(m):
 m.requires_grad_(False)
 for _,p in m.tta_parameters():p.requires_grad_(True)
def copy_head91(m,s):
 with torch.no_grad():
  for n,p in m.interface.named_parameters():p.copy_(torch.as_tensor(s['interface.'+n],device=p.device))
def headstate91(m):return {'interface.'+n:p.detach().cpu().numpy().copy() for n,p in m.interface.named_parameters()}
