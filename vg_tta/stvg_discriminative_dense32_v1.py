"""Per-anchor ridge target/background teacher and external-center KL updates.

No data/GT readers, pretrained parameter mutation or new feature encoder.
"""
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from vg_tta.stvg_soft_correspondence_v1 import cell_overlap,propagate,soft_iou

ARMS=['E_MBR','Teacher_readout','Fixed_dense','On_policy_dense']
FEEDBACK=['ridge','shuffled','cosine']

def pool(maps,valid,boxes):
    # [T,C,H,W], [T,H,W], [T,K,4]; exact fractional-cell average.
    area=cell_overlap(boxes,*maps.shape[-2:])*valid[:,None]
    mass=area.sum((-1,-2));raw=torch.einsum('tkhw,tchw->tkc',area,maps.double())/mass.clamp_min(1e-12)[...,None]
    phi=F.normalize(raw,dim=-1,eps=1e-12)
    return torch.cat((phi,torch.ones_like(phi[...,:1])),-1),mass>1e-12

def backgrounds(positive,all_candidates,grid=9,limit=64):
    b=torch.as_tensor(positive,dtype=torch.float64);wh=b[2:];c=torch.as_tensor(all_candidates,dtype=torch.float64).reshape(-1,4)
    xs=torch.linspace(float(wh[0]/2),float(1-wh[0]/2),grid);ys=torch.linspace(float(wh[1]/2),float(1-wh[1]/2),grid)
    if (wh<=0).any()or(wh>1).any():return torch.empty(0,4,dtype=torch.float64)
    yy,xx=torch.meshgrid(ys,xs,indexing='ij');boxes=torch.cat((torch.stack((xx.flatten(),yy.flatten()),-1),wh.expand(grid*grid,2)),-1).double()
    lo=boxes[:,:2]-boxes[:,2:]/2;hi=boxes[:,:2]+boxes[:,2:]/2;cl=c[:,:2]-c[:,2:]/2;ch=c[:,:2]+c[:,2:]/2
    inter=(torch.minimum(hi[:,None],ch[None])-torch.maximum(lo[:,None],cl[None])).clamp_min(0).prod(-1)
    return boxes[(inter<=0).all(1)][:limit]

def train_teachers(maps,valid,frames,selected,regularization):
    teachers=[];diagnostics=[]
    for frame,k in zip(frames,selected):
        if k<0:continue
        i=frame['position'];positive=torch.as_tensor(frame['boxes'][k],dtype=torch.float64)
        # ALL retained probe candidates are protected, including alternatives.
        bg=backgrounds(positive,frame['boxes']);diag=dict(position=i,candidate_id=int(k),background_count=len(bg),protected_candidates=len(frame['boxes']))
        if not len(bg):diagnostics.append(dict(**diag,usable=False,reason='no_disjoint_background'));continue
        boxes=torch.cat((positive[None],bg));phi,ok=pool(maps[i:i+1],valid[i:i+1],boxes[None]);phi=phi[0];ok=ok[0]
        if not ok[0]:diagnostics.append(dict(**diag,usable=False,reason='positive_outside_valid_features'));continue
        phi=phi[ok];neg=len(phi)-1
        if not neg:diagnostics.append(dict(**diag,usable=False,reason='no_valid_background'));continue
        y=torch.cat((torch.ones(1),-torch.ones(neg))).double();a=torch.cat((torch.tensor([.5],dtype=torch.float64),torch.full((neg,),.5/neg,dtype=torch.float64)))
        # Exact equal class weights. All coordinates, including bias, regularized.
        A=phi.T@(a[:,None]*phi)+regularization*torch.eye(phi.shape[-1],dtype=torch.float64);rhs=phi.T@(a*y)
        w=torch.linalg.solve(A,rhs);err=float((A@w-rhs).abs().max());assert err<1e-10 and torch.isfinite(w).all()
        pred=phi@w
        teachers.append(dict(position=i,positive=positive,w=w,phi=phi,labels=y,weights=a,background_boxes=bg,kept=ok,normal_residual=err))
        diagnostics.append(dict(**diag,usable=True,positive_score=float(pred[0]),background_mean_score=float(pred[1:].mean()),normal_residual=err))
    return teachers,diagnostics

def noise(alias,n,per_round=17):
    assert per_round==17
    seed=int.from_bytes(hashlib.sha256(('ridge-dense32-v1|'+alias).encode()).digest()[:8],'little');rng=np.random.default_rng(seed)
    result=[]
    for _ in range(2):
        z=rng.standard_normal((n,8,2));result.append(torch.from_numpy(np.concatenate((np.zeros((n,1,2)),z,-z),1)))
    return torch.stack(result)

def candidates(E,eta,eps,scale):
    sigma=E[:,2:]*scale;center=E[:,:2]+sigma*eta
    out=E[:,None].expand(-1,eps.shape[1],-1).clone();out[...,:2]=center[:,None]+sigma[:,None]*eps
    return out

def feedback(maps,valid,cands,teachers,prior_logits,tau):
    phi,ok=pool(maps,valid,cands);n,k=ok.shape
    # No visible feature cell -> no observed classification evidence for candidate.
    # The student's finite support is explicitly restricted to visible ROIs.
    usable=ok.any(1);safe=ok.clone();safe[~usable]=True
    lp=prior_logits.clone();lp[~safe]=-torch.inf;prior=lp.softmax(-1)
    if not teachers:return dict(prob=prior,prior=prior,per_anchor=torch.empty(0,n,k),scores=torch.empty(0,n,k),valid=ok,informative=torch.zeros(n,dtype=torch.bool))
    scores=torch.stack([phi@t['w']for t in teachers]);logits=lp[None]+scores/tau;pa=logits.softmax(-1)
    informative=(scores.amax(-1)-scores.amin(-1)).amax(0)>1e-12
    prob=pa.mean(0);prob[~usable|~informative]=prior[~usable|~informative]
    assert torch.isfinite(prob).all()and torch.max(abs(prob.sum(-1)-1))<1e-12
    return dict(prob=prob,prior=prior,per_anchor=pa,scores=scores,valid=ok,informative=usable&informative)

def initial_feedback(maps,valid,E,teachers,eps,cfg,planned,all_E_anchors):
    n=len(E);eta=torch.zeros(n,2,dtype=torch.float64);both=torch.cat((eps[0],eps[1]),1);b=candidates(E,eta,both,cfg['center_scale']);lp=-.5*both.square().sum(-1)
    real=feedback(maps,valid,b,teachers,lp,cfg['temperature']);perm=list(range(n));unobs=[i for i in range(n)if i not in planned]
    if len(unobs)>1:
        for j,i in enumerate(unobs):perm[i]=unobs[(j+1)%len(unobs)]
    shuffled=feedback(maps[perm],valid[perm],b,teachers,lp,cfg['temperature'])
    if teachers:
        ap=[t['position']for t in teachers];ab=[t['positive']for t in teachers];soft=propagate(maps,valid,ap,ab,.07)
        rewards=torch.stack([soft_iou(b,sm[:,None])for sm in soft]).mean(0);cl=lp+rewards/cfg['temperature'];cl[~real['valid']&real['valid'].any(1)[:,None]]=-torch.inf;cosine=cl.softmax(-1)
        cosine[~real['valid'].any(1)]=real['prior'][~real['valid'].any(1)]
    else:soft=torch.empty(0,n,*maps.shape[-2:]);rewards=torch.zeros_like(lp);cosine=real['prior'].clone()
    matched_cosine=cosine.clone()
    if all_E_anchors:
        ap=[a[0]for a in all_E_anchors];ab=[a[1]for a in all_E_anchors];all_soft=propagate(maps,valid,ap,ab,.07)
        all_rewards=torch.stack([soft_iou(b,sm[:,None])for sm in all_soft]).mean(0);cl=lp+all_rewards/cfg['temperature'];cl[~real['valid']&real['valid'].any(1)[:,None]]=-torch.inf;cosine=cl.softmax(-1);cosine[~real['valid'].any(1)]=real['prior'][~real['valid'].any(1)]
    else:all_soft=torch.empty(0,n,*maps.shape[-2:]);all_rewards=torch.zeros_like(lp);cosine=real['prior'].clone()
    out=E.clone();best=real['prob'].argmax(1);eligible=real['informative'].clone();eligible[planned]=False
    out[eligible]=b[torch.arange(n)[eligible],best[eligible]]
    assert torch.equal(out[:,2:],E[:,2:])and torch.equal(out[planned],E[planned])
    return dict(candidates=b,noise=eps,real=real,shuffled=shuffled,cosine=cosine,cosine_rewards=rewards,
        per_anchor_soft_maps=soft,cosine_all_soft_maps=all_soft,cosine_all_rewards=all_rewards,cosine_matched=matched_cosine,cosine_anchor_positions=[a[0]for a in all_E_anchors],permutation=perm,teacher_readout=out,GT_read=False)

def student_logits(d,eta,valid):
    lp=-.5*(d-eta[:,None]).square().sum(-1);ok=valid.clone();ok[~ok.any(1)]=True;lp[~ok]=-torch.inf
    return lp

def distill(maps,valid,E,teachers,eps,cfg,planned,dynamic):
    n=len(E);eta=torch.zeros(n,2,dtype=torch.float64);path=[dict(eta=eta.clone(),boxes=E.clone())];rounds=[];allowed=torch.ones(n,dtype=torch.bool);allowed[planned]=False
    for step in range(2):
        sampler=eta.clone()if dynamic else torch.zeros_like(eta);ee=eps[step];b=candidates(E,sampler,ee,cfg['center_scale']);d=sampler[:,None]+ee;fl=feedback(maps,valid,b,teachers,-.5*ee.square().sum(-1),cfg['temperature'])
        lp=student_logits(d,eta,fl['valid']);p=lp.softmax(-1);pt=fl['prob'].detach();grad=((p-pt)[...,None]*d).sum(1)
        use=allowed&fl['informative'];grad[~use]=0;new=eta-cfg['step_size']*grad
        # Independent per-frame optimization: gradient of frame KL, rather than
        # shrinking each independent variable's update by total video length.
        assert torch.isfinite(new).all();updated=candidates(E,new,torch.zeros(n,1,2),cfg['center_scale'])[:,0];updated[planned]=E[planned]
        mask=pt>0;kl=float((pt[mask]*(pt[mask].log()-lp.log_softmax(-1)[mask])).sum()/max(int(allowed.sum()),1))
        rounds.append(dict(step=step,sampling_center=sampler,candidates=b,standardized_candidates=d,feedback=fl,student_prob=p,gradient=grad,eta_before=eta.clone(),eta_after=new.clone(),KL=kl,updated_frames=int(use.sum()),current_center_used=dynamic))
        eta=new;path.append(dict(eta=eta.clone(),boxes=updated))
    return dict(path=path,rounds=rounds,final=path[-1]['boxes'],dynamic=dynamic,final_step=2,GT_read=False)

def contract():
    maps=torch.ones(2,3,4,4);valid=torch.ones(2,4,4,dtype=torch.bool);E=torch.tensor([[.5,.5,.2,.2],[.5,.5,.2,.2]],dtype=torch.float64);eps=noise('property',2);cfg=dict(center_scale=.25,temperature=.1,step_size=.5)
    for dynamic in [False,True]:
        f=distill(maps,valid,E,[],eps,cfg,[0],dynamic);assert torch.equal(f['final'],E)
    d=torch.randn(3,17,2,dtype=torch.float64);eta=torch.randn(3,2,dtype=torch.float64,requires_grad=True);pt=torch.randn(3,17,dtype=torch.float64).softmax(-1);lp=student_logits(d,eta,torch.ones(3,17,dtype=torch.bool));loss=(pt*(pt.log()-lp.log_softmax(-1))).sum();g=torch.autograd.grad(loss,eta)[0];expected=((lp.softmax(-1)-pt)[...,None]*d).sum(1);assert abs(g-expected).max()<1e-12
    bg=backgrounds(E[0],E[:1]);lo=bg[:,:2]-bg[:,2:]/2;hi=bg[:,:2]+bg[:,2:]/2;assert ((torch.minimum(hi,torch.tensor([.6,.6]))-torch.maximum(lo,torch.tensor([.4,.4]))).clamp_min(0).prod(1)==0).all()
    return dict(status='pass',no_teacher_multiround_exact_noop=True,analytic_categorical_KL_gradient=True,background_disjoint=True,center_only=True)
