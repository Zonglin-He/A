"""Frozen cosine soft-IoU teacher, finite-support center policy and genuine refresh.

No model/GT/data readers. All unobserved actual frames get their own evidence.
"""
import torch
from vg_tta.stvg_soft_correspondence_v1 import cell_overlap,soft_iou
from vg_tta.stvg_discriminative_dense32_v1 import candidates,student_logits
ARMS=['E_MBR','Cosine_readout','Fixed_dense','On_policy_dense','Shuffled_OPD']
FIT_ARMS=ARMS[2:]

def feedback(soft,valid,b,eps,tau):
    area=cell_overlap(b,*valid.shape[-2:])*valid[:,None];ok=area.sum((-1,-2))>1e-12
    usable=ok.any(1);safe=ok.clone();safe[~usable]=True
    lp=-.5*eps.square().sum(-1);lp[~safe]=-torch.inf;prior=lp.softmax(-1)
    if not len(soft):rewards=torch.zeros_like(lp);per=torch.empty(0,*lp.shape);prob=prior.clone();informative=torch.zeros(len(b),dtype=torch.bool)
    else:
        per=torch.stack([soft_iou(b,m[:,None])for m in soft]);rewards=per.mean(0)
        scores=lp+rewards/tau;prob=scores.softmax(-1)
        hi=rewards.masked_fill(~ok,-torch.inf).amax(-1);lo=rewards.masked_fill(~ok,torch.inf).amin(-1)
        informative=usable&((hi-lo)>1e-12);prob[~informative]=prior[~informative]
    assert torch.isfinite(prob).all()and (prob.sum(-1)-1).abs().max()<1e-12
    return dict(prob=prob,prior=prior,rewards=rewards,per_anchor_rewards=per,valid=ok,informative=informative)

def direct(E,soft,valid,eps,cfg,planned):
    both=torch.cat((eps[0],eps[1]),1);eta=torch.zeros(len(E),2,dtype=torch.float64);b=candidates(E,eta,both,cfg['center_scale']);f=feedback(soft,valid,b,both,cfg['temperature']);out=E.clone();use=f['informative'].clone();use[planned]=False
    best=f['prob'].argmax(-1);out[use]=b[torch.arange(len(b))[use],best[use]]
    return dict(boxes=out,candidates=b,feedback=f,best=best,GT_read=False)

def fit(E,soft,valid,eps,cfg,planned,mode,permutation):
    assert mode in FIT_ARMS;n=len(E);eta=torch.zeros(n,2,dtype=torch.float64);path=[dict(eta=eta.clone(),boxes=E.clone())];rounds=[]
    evidence=soft[:,permutation]if mode=='Shuffled_OPD'else soft;visible=valid[permutation]if mode=='Shuffled_OPD'else valid
    allowed=torch.ones(n,dtype=torch.bool);allowed[planned]=False
    for step in range(2):
        center=eta.clone()if mode!='Fixed_dense'else torch.zeros_like(eta);ee=eps[step];b=candidates(E,center,ee,cfg['center_scale']);d=center[:,None]+ee
        f=feedback(evidence,visible,b,ee,cfg['temperature']);lp=student_logits(d,eta,f['valid']);p=lp.softmax(-1);pt=f['prob'].detach()
        gradient=((p-pt)[...,None]*d).sum(1);use=allowed&f['informative'];gradient[~use]=0
        new=eta-cfg['step_size']*gradient;out=candidates(E,new,torch.zeros(n,1,2),cfg['center_scale'])[:,0];out[planned]=E[planned]
        # KL is averaged over unobserved frames. Each independent frame variable
        # gets its own frame-KL gradient (equivalent block learning rate |U|*alpha).
        usemask=pt>0;term=torch.zeros_like(pt);term[usemask]=pt[usemask]*(pt[usemask].log()-lp.log_softmax(-1)[usemask]);KL=float(term[allowed].sum()/max(int(allowed.sum()),1))
        assert torch.isfinite(new).all()and torch.equal(out[:,2:],E[:,2:])and torch.equal(out[planned],E[planned])
        rounds.append(dict(step=step,sampling_center=center,candidates=b,standardized_candidates=d,feedback=f,student_prob=p,gradient=gradient,eta_before=eta.clone(),eta_after=new.clone(),KL=KL,
            updated_frames=int(use.sum()),nonzero_gradient_frames=int((gradient.abs().amax(-1)>1e-12).sum()),output_step_L2=float((out-path[-1]['boxes']).norm()),teacher_target_frames=permutation if mode=='Shuffled_OPD'else list(range(n))))
        eta=new;path.append(dict(eta=eta.clone(),boxes=out))
    return dict(mode=mode,path=path,rounds=rounds,final=path[-1]['boxes'],selected_step=2,variables='per-frame external center mean in standardized E coordinates',pretrained_weights_frozen=True,GT_read=False)

def contract():
    from vg_tta.stvg_discriminative_dense32_v1 import noise
    E=torch.tensor([[.4,.5,.4,.4],[.4,.5,.4,.4],[.4,.5,.4,.4]],dtype=torch.float64);eps=noise('cosine-properties',3);valid=torch.ones(3,8,8,dtype=torch.bool);cfg=dict(center_scale=.25,temperature=.1,step_size=.5);planned=[0];perm=[0,2,1]
    empty=torch.empty(0,3,8,8)
    for mode in FIT_ARMS:
        f=fit(E,empty,valid,eps,cfg,planned,mode,perm);assert torch.equal(f['final'],E)
        f=fit(E,torch.zeros(1,3,8,8),valid,eps,cfg,planned,mode,perm);assert torch.equal(f['final'],E)
    soft=torch.zeros(1,3,8,8);soft[:,:,:,4:]=1
    fixed=fit(E,soft,valid,eps,cfg,planned,'Fixed_dense',perm);live=fit(E,soft,valid,eps,cfg,planned,'On_policy_dense',perm)
    assert torch.equal(fixed['path'][1]['boxes'],live['path'][1]['boxes'])and live['path'][1]['eta'].abs().sum()>0
    assert torch.equal(live['rounds'][1]['sampling_center'],live['path'][1]['eta'])and torch.count_nonzero(fixed['rounds'][1]['sampling_center'])==0
    d=torch.randn(3,17,2,dtype=torch.float64);eta=torch.randn(3,2,dtype=torch.float64,requires_grad=True);pt=torch.randn(3,17,dtype=torch.float64).softmax(-1);lp=student_logits(d,eta,torch.ones(3,17,dtype=torch.bool));g=torch.autograd.grad((pt*(pt.log()-lp.log_softmax(-1))).sum(),eta)[0];expected=((lp.softmax(-1)-pt)[...,None]*d).sum(1);assert (g-expected).abs().max()<1e-12
    return dict(status='pass',no_teacher_exact_noop=True,zero_reward_exact_noop=True,analytic_frame_KL_gradient=True,first_fixed_live_exact=True,second_live_current_center=True,widths_and_observed_exact_E=True)
