"""Frozen position-preserving ROI template feedback; no learned predictor."""
import torch
import torch.nn.functional as F
from vg_tta.stvg_discriminative_dense32_v1 import candidates,student_logits
ARMS=['E_MBR','Layout_readout','Unordered_readout','Fixed_dense','On_policy_dense','Shuffled_OPD','Unordered_OPD','Forward_readout','Forward_Fixed','Forward_On','Forward_Shuffled']
FIT_ARMS=['Fixed_dense','On_policy_dense','Shuffled_OPD','Unordered_OPD']

@torch.no_grad()
def patches(maps,valid,boxes,grid_size=3):
 n,c,h,w=maps.shape;k=boxes.shape[1];v=valid.float();vh=v.amax(2).sum(1);vw=v.amax(1).sum(1)
 assert (vh>0).all()and(vw>0).all()
 rectangle=(torch.arange(h)[None,:,None]<vh[:,None,None])&(torch.arange(w)[None,None,:]<vw[:,None,None]);assert torch.equal(rectangle,valid)
 a=(torch.arange(grid_size,dtype=boxes.dtype)+.5)/grid_size-.5;yy,xx=torch.meshgrid(a,a,indexing='ij');offset=torch.stack((xx,yy),-1).reshape(1,1,grid_size,grid_size,2)
 xy=boxes[:,:,:2]
 grid=xy[:,:,None,None,:]+boxes[:,:,None,None,2:]*offset
 grid=grid*torch.stack((vw/w,vh/h),-1)[:,None,None,None,:];grid=2*grid-1
 sample=F.grid_sample(maps.float()*v[:,None],grid.reshape(n,k*grid_size,grid_size,2).float(),mode='bilinear',padding_mode='zeros',align_corners=False).reshape(n,c,k,grid_size*grid_size).permute(0,2,3,1)
 weight=F.grid_sample(v[:,None],grid.reshape(n,k*grid_size,grid_size,2).float(),mode='bilinear',padding_mode='zeros',align_corners=False).reshape(n,k,grid_size*grid_size)
 return F.normalize(sample,dim=-1,eps=1e-12).double(),weight.double()

@torch.no_grad()
def build(maps,valid,positions,boxes):
 if not positions:return dict(maps=maps,valid=valid,positions=[],templates=torch.empty(0,9,maps.shape[1],dtype=torch.float64),weights=torch.empty(0,9,dtype=torch.float64))
 b=torch.as_tensor(boxes,dtype=torch.float64).reshape(-1,1,4);p,w=patches(maps[positions],valid[positions],b)
 return dict(maps=maps,valid=valid,positions=list(positions),templates=p[:,0],weights=w[:,0])

@torch.no_grad()
def feedback(context,b,eps,tau,permutation=None,pooled=False):
 maps,valid=context['maps'],context['valid']
 if permutation is not None:maps=maps[permutation];valid=valid[permutation]
 p,w=patches(maps,valid,b);n,k=b.shape[:2];ok=w.sum(-1)>1e-6;safe=ok.clone();safe[~ok.any(1)]=True
 lp=(-.5*eps.double().square().sum(-1)).masked_fill(~safe,-torch.inf);prior=lp.softmax(-1)
 if not context['positions']:r=torch.zeros(n,k,dtype=torch.float64);per=torch.empty(0,n,k,dtype=torch.float64);common=torch.empty(0,n,k,9,dtype=torch.float64);pt=prior;informative=torch.zeros(n,dtype=torch.bool)
 else:
  t=context['templates'];tw=context['weights'];common=w[None]*tw[:,None,None,:]
  if pooled:
   pp=(p*w[...,None]).sum(-2)/w.sum(-1).clamp_min(1e-12)[...,None];tt=(t*tw[...,None]).sum(-2)/tw.sum(-1).clamp_min(1e-12)[:,None];per=torch.einsum('nkc,ac->ank',pp,tt)
  else:per=(torch.einsum('nkjc,ajc->ankj',p,t)*common).sum(-1)/common.sum(-1).clamp_min(1e-12)
  r=per.mean(0);pt=(lp+r/tau).softmax(-1);hi=r.masked_fill(~ok,-torch.inf).amax(-1);lo=r.masked_fill(~ok,torch.inf).amin(-1);informative=ok.any(1)&((hi-lo)>1e-6);pt[~informative]=prior[~informative]
 assert torch.isfinite(pt).all()and(pt.sum(-1)-1).abs().max()<1e-12
 return dict(prob=pt,prior=prior,rewards=r,per_anchor_rewards=per,common_valid_bin_mass=common.sum(-1),valid=ok,informative=informative)


def direct(E,context,eps,cfg,planned,pooled=False):
 ee=torch.cat((eps[0],eps[1]),1);b=candidates(E,torch.zeros(len(E),2,dtype=torch.float64),ee,cfg['center_scale']);f=feedback(context,b,ee,cfg['temperature'],pooled=pooled);out=E.clone();use=f['informative'].clone();use[planned]=False;best=f['prob'].argmax(-1);out[use]=b[torch.arange(len(E))[use],best[use]]
 return dict(boxes=out,candidates=b,feedback=f,best=best)


def fit(E,context,eps,cfg,planned,mode,permutation):
 n=len(E);eta=torch.zeros(n,2,dtype=torch.float64);path=[dict(eta=eta.clone(),boxes=E.clone())];rounds=[];allowed=torch.ones(n,dtype=torch.bool);allowed[planned]=False
 for step in range(2):
  center=eta.clone()if mode!='Fixed_dense'else torch.zeros_like(eta);ee=eps[step];b=candidates(E,center,ee,cfg['center_scale']);d=center[:,None]+ee;f=feedback(context,b,ee,cfg['temperature'],permutation if mode=='Shuffled_OPD'else None,pooled=mode=='Unordered_OPD');lp=student_logits(d,eta,f['valid']);sp=lp.softmax(-1);pt=f['prob'].detach();grad=((sp-pt)[...,None]*d).sum(1);grad[~(allowed&f['informative'])]=0;new=eta-cfg['step_size']*grad;out=candidates(E,new,torch.zeros(n,1,2),cfg['center_scale'])[:,0];out[planned]=E[planned];nz=pt>0;term=torch.zeros_like(pt);term[nz]=pt[nz]*(pt[nz].log()-lp.log_softmax(-1)[nz]);KL=float(term[allowed].sum()/max(int(allowed.sum()),1));assert torch.isfinite(new).all()and torch.equal(out[:,2:],E[:,2:])and torch.equal(out[planned],E[planned])
  rounds.append(dict(step=step,sampling_center=center,candidates=b,standardized_candidates=d,feedback=f,student_prob=sp,gradient=grad,eta_before=eta.clone(),eta_after=new.clone(),KL=KL,updated_frames=int((allowed&f['informative']).sum()),nonzero_gradient_frames=int((grad.abs().amax(-1)>1e-12).sum()),output_step_L2=float((out-path[-1]['boxes']).norm())))
  eta=new;path.append(dict(eta=eta.clone(),boxes=out))
 return dict(mode=mode,path=path,rounds=rounds,final=path[-1]['boxes'],selected_step=2,GT_read=False,pretrained_weights_frozen=True)


def properties():
 from vg_tta.stvg_discriminative_dense32_v1 import noise
 torch.manual_seed(20261019);maps=torch.randn(3,12,8,10);valid=torch.ones(3,8,10,dtype=torch.bool);E=torch.tensor([[.5,.5,.4,.4]]*3,dtype=torch.float64);ctx=build(maps,valid,[0],E[:1]);p,w=patches(maps[:1],valid[:1],E[:1,None]);assert torch.equal(p[0,0],ctx['templates'][0]);eps=noise('ROI-layout-properties',3);cfg=dict(center_scale=.25,temperature=.1,step_size=.5);f=feedback(ctx,E[:,None],torch.zeros(3,1,2),.1);assert abs(f['rewards'][0,0]-1)<2e-6
 no=build(maps,valid,[],[])
 for mode in FIT_ARMS:assert torch.equal(fit(E,no,eps,cfg,[0],mode,[0,2,1])['final'],E)
 flat=build(torch.ones_like(maps),valid,[0],E[:1])
 assert not feedback(flat,candidates(E,torch.zeros(3,2),eps[0],.25),eps[0],.1)['informative'].any()
 fixed=fit(E,ctx,eps,cfg,[0],'Fixed_dense',[0,2,1]);live=fit(E,ctx,eps,cfg,[0],'On_policy_dense',[0,2,1]);assert torch.equal(fixed['path'][1]['boxes'],live['path'][1]['boxes']);assert torch.equal(live['rounds'][1]['sampling_center'],live['path'][1]['eta'])
 pv=valid.clone();pv[:,:,8:]=False;xx=maps.clone();xx[:,:,:,8:]=10000*torch.randn_like(xx[:,:,:,8:]);aa,aw=patches(maps,pv,E[:,None]);bb,bw=patches(xx,pv,E[:,None]);assert torch.equal(aa,bb)and torch.equal(aw,bw)
 identity=torch.eye(9).T.reshape(9,3,3);xm=torch.stack((identity,identity.reshape(9,9)[:,torch.arange(9).roll(1)].reshape(9,3,3)));vv=torch.ones(2,3,3,dtype=torch.bool);full=torch.tensor([[.5,.5,1.,1.]]*2,dtype=torch.float64);cc=build(xm,vv,[0],full[:1]);ee=torch.zeros(2,1,2);ordered=feedback(cc,full[:,None],ee,.1);unordered=feedback(cc,full[:,None],ee,.1,pooled=True);assert ordered['rewards'][0,0]>ordered['rewards'][1,0]+.9;assert abs(unordered['rewards'][0,0]-unordered['rewards'][1,0])<1e-6
 return dict(status='pass',self_template_cosine_one=True,constant_features_no_feedback=True,no_reference_multiround_exact_noop=True,first_fixed_live_exact=True,actual_grid3x3=True,no_trained_mapping=True,padding_activations_invariant=True,spatial_order_positive_control=True)
