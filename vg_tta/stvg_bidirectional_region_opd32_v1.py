"""Candidate-conditioned bidirectional region feedback, same frozen coarse maps.

No learned model or expert invocation. Analytic categorical Gaussian KL updates
only per-query per-frame temporary center state, reset between queries.
"""
import torch
import torch.nn.functional as F
from vg_tta.stvg_soft_correspondence_v1 import cell_overlap,soft_iou
from vg_tta.stvg_discriminative_dense32_v1 import candidates,student_logits
ARMS=['E_MBR','Bidir_readout','Fixed_dense','On_policy_dense','Shuffled_OPD','Forward_readout','Forward_Fixed','Forward_On','Forward_Shuffled']
FIT_ARMS=['Fixed_dense','On_policy_dense','Shuffled_OPD']

@torch.no_grad()
def build(maps,valid,positions,boxes,forward_soft):
 n,c,h,w=maps.shape;f=F.normalize(maps.float().flatten(2).transpose(1,2),dim=-1);reverse=[]
 for a in positions:
  logits=(f[a]@f.transpose(1,2))/.07  # N,anchor-cell,target-cell
  # Each direction gets its own key normalization. This is NOT transpose(P).
  logits=logits.masked_fill(~valid.flatten(1)[:,None,:],-torch.inf)
  rv=logits.softmax(-1)*valid[a].flatten()[None,:,None]
  assert torch.isfinite(rv).all();reverse.append(rv)
 return dict(forward_soft=forward_soft,reverse=torch.stack(reverse)if reverse else torch.empty(0,n,h*w,h*w),
  valid=valid,anchor_boxes=torch.as_tensor(boxes,dtype=torch.float64).reshape(-1,4),positions=list(positions))


def feedback(context,b,eps,tau,permutation=None):
 valid=context['valid'];ss=context['forward_soft'];rv=context['reverse']
 if permutation is not None:valid=valid[permutation];ss=ss[:,permutation];rv=rv[:,permutation]
 n,k=b.shape[:2];h,w=valid.shape[-2:];area=cell_overlap(b,h,w)*valid[:,None];ok=area.sum((-1,-2))>1e-12;safe=ok.clone();safe[~ok.any(1)]=True
 lp=-.5*eps.square().sum(-1);lp[~safe]=-torch.inf;prior=lp.softmax(-1)
 if not len(ss):forward=torch.empty(0,n,k);reverse=forward.clone();per=forward.clone();reward=torch.zeros_like(lp);pt=prior.clone();informative=torch.zeros(n,dtype=torch.bool)
 else:
  forward=torch.stack([soft_iou(b,s[:,None])for s in ss])
  # Current candidate's region is projected into the reference grid every call.
  back=torch.einsum('antl,nkl->ankt',rv.double(),area.flatten(-2)).reshape(len(ss),n,k,h,w)
  ab=context['anchor_boxes'][:,None,None,:].expand(-1,n,k,-1)
  reverse=soft_iou(ab,back);assert (forward>=0).all()and(reverse>=0).all()
  per=torch.sqrt(forward*reverse);reward=per.mean(0);pt=(lp+reward/tau).softmax(-1)
  hi=reward.masked_fill(~ok,-torch.inf).amax(-1);lo=reward.masked_fill(~ok,torch.inf).amin(-1);informative=ok.any(1)&((hi-lo)>1e-12);pt[~informative]=prior[~informative]
 assert torch.isfinite(pt).all()and (pt.sum(-1)-1).abs().max()<1e-12
 return dict(prob=pt,prior=prior,rewards=reward,per_anchor_rewards=per,forward_rewards=forward,reverse_rewards=reverse,valid=ok,informative=informative)


def direct(E,context,eps,cfg,planned):
 both=torch.cat((eps[0],eps[1]),1);b=candidates(E,torch.zeros(len(E),2,dtype=torch.float64),both,cfg['center_scale']);f=feedback(context,b,both,cfg['temperature']);out=E.clone();use=f['informative'].clone();use[planned]=False;best=f['prob'].argmax(-1);out[use]=b[torch.arange(len(E))[use],best[use]]
 return dict(boxes=out,candidates=b,feedback=f,best=best)


def fit(E,context,eps,cfg,planned,mode,permutation):
 assert mode in FIT_ARMS;n=len(E);eta=torch.zeros(n,2,dtype=torch.float64);path=[dict(eta=eta.clone(),boxes=E.clone())];rounds=[];allowed=torch.ones(n,dtype=torch.bool);allowed[planned]=False
 for step in range(2):
  center=eta.clone()if mode!='Fixed_dense'else torch.zeros_like(eta);ee=eps[step];b=candidates(E,center,ee,cfg['center_scale']);d=center[:,None]+ee
  f=feedback(context,b,ee,cfg['temperature'],permutation if mode=='Shuffled_OPD'else None);lp=student_logits(d,eta,f['valid']);sp=lp.softmax(-1);pt=f['prob'].detach();grad=((sp-pt)[...,None]*d).sum(1);grad[~(allowed&f['informative'])]=0
  new=eta-cfg['step_size']*grad;out=candidates(E,new,torch.zeros(n,1,2),cfg['center_scale'])[:,0];out[planned]=E[planned];term=torch.zeros_like(pt);nz=pt>0;term[nz]=pt[nz]*(pt[nz].log()-lp.log_softmax(-1)[nz]);KL=float(term[allowed].sum()/max(int(allowed.sum()),1))
  assert torch.isfinite(new).all()and torch.equal(out[:,2:],E[:,2:])and torch.equal(out[planned],E[planned])
  rounds.append(dict(step=step,sampling_center=center,candidates=b,standardized_candidates=d,feedback=f,student_prob=sp,gradient=grad,eta_before=eta.clone(),eta_after=new.clone(),KL=KL,
   updated_frames=int((allowed&f['informative']).sum()),nonzero_gradient_frames=int((grad.abs().amax(-1)>1e-12).sum()),output_step_L2=float((out-path[-1]['boxes']).norm())))
  eta=new;path.append(dict(eta=eta.clone(),boxes=out))
 return dict(mode=mode,path=path,rounds=rounds,final=path[-1]['boxes'],selected_step=2,GT_read=False,pretrained_weights_frozen=True)


def properties():
 from vg_tta.stvg_discriminative_dense32_v1 import noise
 from vg_tta.stvg_soft_correspondence_v1 import propagate
 torch.manual_seed(20261018);x=torch.randn(3,8,4,5);valid=torch.ones(3,4,5,dtype=torch.bool);box=torch.tensor([.5,.5,.5,.5]);ss=propagate(x,valid,[0],[box]);ctx=build(x,valid,[0],[box.tolist()],ss);f=F.normalize(x.flatten(2).transpose(1,2),dim=-1);L=f[1]@f[0].T/.07
 assert torch.allclose(ctx['reverse'][0,1],L.T.softmax(-1),atol=1e-7,rtol=2e-6)
 E=torch.tensor([[.5,.5,.4,.4]]*3,dtype=torch.float64);eps=noise('bidirectional-properties',3);cfg=dict(center_scale=.25,temperature=.1,step_size=.5)
 no=dict(forward_soft=torch.empty(0,3,4,5),reverse=torch.empty(0,3,20,20),valid=valid,anchor_boxes=torch.empty(0,4),positions=[])
 for mode in FIT_ARMS:assert torch.equal(fit(E,no,eps,cfg,[0],mode,[0,2,1])['final'],E)
 fixed=fit(E,ctx,eps,cfg,[0],'Fixed_dense',[0,2,1]);live=fit(E,ctx,eps,cfg,[0],'On_policy_dense',[0,2,1]);assert torch.equal(fixed['path'][1]['boxes'],live['path'][1]['boxes']);assert torch.equal(live['rounds'][1]['sampling_center'],live['path'][1]['eta'])
 return dict(status='pass',reverse_independently_normalized=True,no_ref_exact_noop=True,first_fixed_live_exact=True,second_live_uses_current_state=True)
