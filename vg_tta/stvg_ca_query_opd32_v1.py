"""Frozen trained CA anchor-content metric and genuine external-center OPD.

Attention enrichment is a declared working likelihood, not IoU or identity probability.
"""
import math
import numpy as np
import torch
from vg_tta.stvg_anchor_refresh32_v1 import run
from vg_tta.stvg_soft_correspondence_v1 import cell_overlap
from vg_tta.stvg_discriminative_dense32_v1 import candidates,student_logits,noise
ARMS=['E_MBR','Dino_direct','Equery_direct','Dino_fixed','Dino_live','Dino_shuffle','Equery_live','Cosine_like_live']
FIT_ARMS=ARMS[3:]

@torch.no_grad()
def capture(decoder,inputs,n,positions,boxes,condition):
 layer=decoder.layers[-1];assert layer.from_scratch_cross_attn and layer.cross_attn is not None;module=layer.cross_attn;records=[]
 assert module.in_proj_weight is None and module.q_proj_weight is None and module.k_proj_weight is None
 def pre(m,args,kwargs):
  q=kwargs.get('query',args[0]if len(args)>0 else None);k=kwargs.get('key',args[1]if len(args)>1 else None);records.append(dict(q=q.detach().cpu(),k=k.detach().cpu(),heads=m.num_heads,head_dim=m.head_dim))
 def post(m,args,kwargs,out):records[-1]['attention']=out[1].detach().cpu()
 h1=module.register_forward_pre_hook(pre,with_kwargs=True);h2=module.register_forward_hook(post,with_kwargs=True)
 try:
  if condition=='native':
   full=[decoder(**x).cpu()for x in inputs];b=torch.stack([full[i%2].flatten(1,2)[-1][i//2]for i in range(n)]).float();tr=None
  else:b,tr=run(decoder,inputs,n,positions,boxes,True);full=[torch.stack([v['raw_output']for v in t['layers']]).transpose(1,2)for t in tr]
 finally:h1.remove();h2.remove()
 assert len(records)==2
 return dict(boxes=b,full_outputs=full,records=records,traces=tr)

@torch.no_grad()
def reconstruct(rec):
 q=rec['q'].double();k=rec['k'].double();N=q.shape[1];H=rec['heads'];dh=rec['head_dim'];assert q.shape[0]==1 and q.shape[-1]==H*dh
 qq=q[0].reshape(N,H,dh);kk=k.reshape(k.shape[0],N,H,dh);logits=torch.einsum('nhd,snhd->nhs',qq,kk)/math.sqrt(dh);return logits.softmax(-1).mean(1)[:,None]

@torch.no_grad()
def fields(conditions,positions,valid,h,w):
 n=len(valid);V=h*w;out={};keys=[]
 for i in range(n):
  rec=conditions['native']['records'][i%2];dh=rec['head_dim'];H=rec['heads'];assert dh%2==0
  keys.append(rec['k'][:V,i//2].double().reshape(V,H,dh)[...,:dh//2])
 K=torch.stack(keys);fvalid=valid.flatten(1);assert fvalid.any(1).all()
 for name in ['DINO','E']:
  qs=[]
  for p in positions:
   rec=conditions[name]['records'][p%2];qs.append(rec['q'][0,p//2].double().reshape(H,dh)[...,:dh//2])
  Q=torch.stack(qs)if qs else torch.empty(0,H,dh//2,dtype=torch.float64)
  if len(qs):logits=torch.einsum('ahd,nvhd->anhv',Q,K)/math.sqrt(dh);logits=logits.masked_fill(~fvalid[None,:,None],-torch.inf);prob=logits.softmax(-1).mean(2).reshape(len(qs),n,h,w)
  else:prob=torch.empty(0,n,h,w,dtype=torch.float64)
  out[name]=prob
 return out

def normalized_cosine(soft,valid):
 p=soft.double()*valid[None];total=p.sum((-1,-2),keepdim=True);uniform=valid.double()/valid.sum((-1,-2),keepdim=True);return torch.where(total>0,p/total.clamp_min(1e-300),uniform[None])

def feedback(prob,valid,b,eps):
 n,k=b.shape[:2];h,w=valid.shape[-2:];vh=valid.any(2).sum(1).double();vw=valid.any(1).sum(1).double();rectangle=(torch.arange(h)[None,:,None]<vh[:,None,None])&(torch.arange(w)[None,None,:]<vw[:,None,None]);assert torch.equal(valid,rectangle)
 scale=torch.stack((vw/w,vh/h,vw/w,vh/h),-1);area=cell_overlap(b*scale[:,None],h,w)*valid[:,None];mass=area.sum((-1,-2));ok=mass>1e-12;safe=ok.clone();safe[~safe.any(1)]=True;lp=(-.5*eps.square().sum(-1)).masked_fill(~safe,-torch.inf);prior=lp.softmax(-1)
 if not len(prob):L=torch.ones(n,k,dtype=torch.float64);per=torch.empty(0,n,k);pt=prior;info=torch.zeros(n,dtype=torch.bool)
 else:
  per=torch.einsum('anhw,nkhw->ank',prob,area)/(mass/valid.sum((-1,-2))[:,None]).clamp_min(1e-300)[None];L=per.mean(0);positive=ok&(L>0);posterior_safe=positive.clone();posterior_safe[~posterior_safe.any(1)]=safe[~posterior_safe.any(1)];logL=torch.zeros_like(L);logL[positive]=L[positive].log();post=(lp+logL).masked_fill(~posterior_safe,-torch.inf).softmax(-1);info=ok.any(1)&positive.any(1)&((L.masked_fill(~ok,-torch.inf).amax(1)-L.masked_fill(~ok,torch.inf).amin(1))>1e-12);pt=torch.where(info[:,None],post,prior)
 assert torch.isfinite(pt).all()and (pt.sum(-1)-1).abs().max()<1e-12
 return dict(prob=pt,prior=prior,likelihood=L,per_anchor_likelihood=per,valid=ok,informative=info,visible_fraction=mass/(b[:,:,2]*b[:,:,3]*vh[:,None]*vw[:,None]).clamp_min(1e-300))

def direct(E,prob,valid,eps,planned):
 ee=torch.cat((eps[0],eps[1]),1);b=candidates(E,torch.zeros(len(E),2),ee,.25);f=feedback(prob,valid,b,ee);best=f['prob'].argmax(1);use=f['informative'].clone();use[planned]=False;out=E.clone();out[use]=b[torch.arange(len(E))[use],best[use]];return dict(boxes=out,candidates=b,feedback=f,best=best)

def fit(E,fields,valid,eps,planned,mode,permutation):
 prob=fields['E']if mode=='Equery_live'else fields['cosine']if mode=='Cosine_like_live'else fields['DINO'];visible=valid
 if mode=='Dino_shuffle':prob=prob[:,permutation];visible=valid[permutation]
 eta=torch.zeros(len(E),2,dtype=torch.float64);path=[dict(state=eta.clone(),boxes=E.clone())];trace=[];allowed=torch.ones(len(E),dtype=torch.bool);allowed[planned]=False
 for step in range(2):
  center=torch.zeros_like(eta)if mode=='Dino_fixed'else eta.clone();ee=eps[step];b=candidates(E,center,ee,.25);d=center[:,None]+ee;f=feedback(prob,visible,b,ee);safe=f['valid'].clone();safe[~safe.any(1)]=True;lp=(-.5*(d-eta[:,None]).square().sum(-1)).masked_fill(~safe,-torch.inf);sp=lp.softmax(1);g=((sp-f['prob'])[...,None]*d).sum(1);g[~(allowed&f['informative'])]=0;new=eta-.5*g;out=candidates(E,new,torch.zeros(len(E),1,2),.25)[:,0];out[planned]=E[planned]
  assert torch.equal(out[:,2:],E[:,2:])and torch.isfinite(out).all()
  trace.append(dict(step=step,sampling_center=center,candidates=b,standardized_candidates=d,feedback=f,student_prob=sp,gradient=g,before=eta.clone(),after=new.clone(),fixed_step=.5,updated_nodes=int((allowed&f['informative']).sum()),output_step_L2=float((out-path[-1]['boxes']).norm())));eta=new;path.append(dict(state=eta.clone(),boxes=out))
 return dict(mode=mode,path=path,rounds=trace,final=path[-1]['boxes'],selected_step=2,GT_read=False,pretrained_weights_frozen=True)

def properties():
 E=torch.tensor([[.5,.5,.2,.2]]*3,dtype=torch.float64);v=torch.ones(3,5,6,dtype=torch.bool);v[:,:,-1]=False;uniform=v.double()/v.sum((-1,-2),keepdim=True);ep=noise('CA-query-properties',3);cfg={'DINO':uniform[None],'E':uniform[None],'cosine':uniform[None]};f=feedback(cfg['DINO'],v,candidates(E,torch.zeros(3,2),ep[0],.25),ep[0]);assert (f['likelihood']-1).abs().max()<1e-12 and not f['informative'].any()
 for mode in FIT_ARMS:assert torch.equal(fit(E,cfg,v,ep,[0],mode,[0,2,1])['final'],E)
 empty={k:torch.empty(0,3,5,6)for k in cfg}
 for mode in FIT_ARMS:assert torch.equal(fit(E,empty,v,ep,[0],mode,[0,2,1])['final'],E)
 b=candidates(E,torch.zeros(3,2),ep[0],.25);b[:,:,:2]=100;ff=feedback(cfg['DINO'],v,b,ep[0]);assert not ff['informative'].any()and torch.equal(ff['prob'],ff['prior'])
 map=v.double().clone();map[:,:,3:]*=3;map/=map.sum((-1,-2),keepdim=True);ss={k:map[None]for k in cfg};fx=fit(E,ss,v,ep,[0],'Dino_fixed',[0,2,1]);lv=fit(E,ss,v,ep,[0],'Dino_live',[0,2,1]);assert torch.equal(fx['path'][1]['boxes'],lv['path'][1]['boxes']);assert torch.equal(lv['rounds'][1]['sampling_center'],lv['path'][1]['state'])
 return dict(status='pass',uniform_likelihood_one_and_multiround_noop=True,no_reference_multiround_noop=True,all_invisible_noop=True,same_first_fixed_live=True,current_student_second_sampling=True,old_scale_step_and_final2=True,no_reward_temperature_or_trained_head=True)
