"""Only replace score pooling with equal per-anchor probability mixture."""
import torch
from vg_tta.stvg_roi_layout_opd32_v1 import build,feedback as geometric_feedback,patches
from vg_tta.stvg_fine_roi_scoped_opd32_v1 import scope
from vg_tta.stvg_discriminative_dense32_v1 import candidates,student_logits
ARMS=['E_MBR','Center_live','Mix_fixed','Mix_live','Mix_shuffle']

@torch.no_grad()
def feedback(context,b,eps,tau,permutation=None):
 f=geometric_feedback(context,b,eps,tau,permutation)
 old=f['prob'].clone();per=f['per_anchor_rewards'];A=len(per)
 safe=f['valid'].clone();safe[~safe.any(1)]=True
 lp=(-.5*eps.double().square().sum(-1)).masked_fill(~safe,-torch.inf)
 qa=(lp[None]+per/tau).softmax(-1)if A else torch.empty_like(per)
 raw=qa.mean(0)if A else f['prior'].clone()
 pt=raw.clone();pt[~f['informative']]=f['prior'][~f['informative']]
 # Exact singleton contract, including original 1e-6 informative fallback.
 if A==1:assert torch.equal(pt,old)
 f.update(prob=pt,geometric_prob=old,per_anchor_prob=qa,raw_mixture_prob=raw,
          fusion='equal arithmetic mean of per-anchor categorical posteriors',
          informative_gate='unchanged original mean-score range >1e-6',
          effective_anchor_count=A)
 assert torch.isfinite(pt).all()and(pt.sum(-1)-1).abs().max()<1e-12
 return f

def fit(E,context,eps,cfg,allowed,mode,permutation):
 n=len(E);eta=torch.zeros(n,2,dtype=torch.float64);path=[dict(eta=eta.clone(),boxes=E.clone())];rounds=[]
 for step in range(2):
  center=torch.zeros_like(eta)if mode=='Mix_fixed'else eta.clone();ee=eps[step];b=candidates(E,center,ee,cfg['center_scale']);d=center[:,None]+ee
  f=feedback(context,b,ee,cfg['temperature'],permutation if mode=='Mix_shuffle'else None)
  lp=student_logits(d,eta,f['valid']);sp=lp.softmax(-1);pt=f['prob'].detach();g=((sp-pt)[...,None]*d).sum(1);g[~(allowed&f['informative'])]=0
  new=eta-cfg['step_size']*g;out=candidates(E,new,torch.zeros(n,1,2),cfg['center_scale'])[:,0];out[~allowed]=E[~allowed]
  nz=pt>0;term=torch.zeros_like(pt);term[nz]=pt[nz]*(pt[nz].log()-lp.log_softmax(-1)[nz]);KL=float(term[allowed].sum()/max(int(allowed.sum()),1))
  assert torch.isfinite(new).all()and torch.equal(out[:,2:],E[:,2:])and not new[~allowed].count_nonzero()
  rounds.append(dict(step=step,sampling_center=center,candidates=b,standardized_candidates=d,feedback=f,student_prob=sp,gradient=g,eta_before=eta.clone(),eta_after=new.clone(),KL=KL,updated_frames=int((allowed&f['informative']).sum()),output_step_L2=float((out-path[-1]['boxes']).norm())))
  eta=new;path.append(dict(eta=eta.clone(),boxes=out))
 return dict(mode=mode,path=path,rounds=rounds,final=out,selected_step=2,GT_read=False,pretrained_weights_frozen=True)

def properties():
 from vg_tta.stvg_discriminative_dense32_v1 import noise
 torch.manual_seed(20261019);maps=torch.randn(5,12,8,10);valid=torch.ones(5,8,10,dtype=torch.bool);E=torch.tensor([[.5,.5,.4,.4]]*5,dtype=torch.float64);eps=noise('mix-roi-properties',5);cfg=dict(center_scale=.25,temperature=.1,step_size=.5);allow=torch.tensor([False,True,True,True,False])
 one=build(maps,valid,[0],E[:1]);multi=build(maps,valid,[0,4],E[[0,4]]);none=build(maps,valid,[],[])
 for ctx in [one,multi,none]:
  f=fit(E,ctx,eps,cfg,allow,'Mix_fixed',list(range(5)));l=fit(E,ctx,eps,cfg,allow,'Mix_live',list(range(5)));assert torch.equal(f['path'][1]['boxes'],l['path'][1]['boxes'])
  if not ctx['positions']:assert torch.equal(l['final'],E)
  if len(ctx['positions'])==1:
   for mode,oldmode in [('Mix_fixed','Fixed_dense'),('Mix_live','On_policy_dense'),('Mix_shuffle','Shuffled_OPD')]:
    from vg_tta.stvg_fine_roi_scoped_opd32_v1 import scoped_fit
    a=fit(E,ctx,eps,cfg,allow,mode,[0,2,1,4,3]);b=scoped_fit(E,ctx,eps,cfg,[],oldmode,[0,2,1,4,3],allow);assert torch.equal(a['final'],b['final'])
  for t in l['rounds']:
   fb=t['feedback'];qa=fb['per_anchor_prob'];old=fb['geometric_prob'];pr=fb['prior'];safe=fb['valid'].clone();safe[~safe.any(1)]=True
   if len(qa):
    logq=(qa.clamp_min(1e-300).log()).mean(0);logq[~safe]=-torch.inf;geo=logq.softmax(-1);geo[~fb['informative']]=pr[~fb['informative']];assert (geo-old).abs().max()<1e-12
 flat=build(torch.ones_like(maps),valid,[0,4],E[[0,4]]);assert torch.equal(fit(E,flat,eps,cfg,allow,'Mix_live',list(range(5)))['final'],E)
 return dict(status='pass',singleton_all_three_paths_bitexact_original=True,no_reference_multiround_noop=True,constant_maps_noop=True,first_fixed_live_exact=True,normalized_geometric_mean_recovers_old=True,scope_and_sizes_preserved=True)
