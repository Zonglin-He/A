"""Same Fine-Center student; fixed geometric pushforward of candidate probabilities.

Canonical 64x64 ORIGINAL-image grid, plus the user-approved outside-image atom.
No padding cells, learned head, pathwise derivative, or additional visual evidence.
"""
import torch
from vg_tta.stvg_roi_layout_opd32_v1 import feedback
from vg_tta.stvg_discriminative_dense32_v1 import candidates,student_logits
GRID=64

def kernels(boxes,grid=GRID):
    b=boxes.detach().double();assert (b[...,2:]>0).all()
    lo=b[...,:2]-b[...,2:]/2;hi=b[...,:2]+b[...,2:]/2
    edge=torch.arange(grid+1,dtype=b.dtype)/grid
    ax=(torch.minimum(hi[...,0,None],edge[1:])-torch.maximum(lo[...,0,None],edge[:-1])).clamp_min(0)
    ay=(torch.minimum(hi[...,1,None],edge[1:])-torch.maximum(lo[...,1,None],edge[:-1])).clamp_min(0)
    outside=(ax.sum(-1)*ay.sum(-1))==0
    ax=ax/ax.sum(-1,keepdim=True).clamp_min(1e-300);ay=ay/ay.sum(-1,keepdim=True).clamp_min(1e-300)
    K=torch.cat(((ay[..., :,None]*ax[...,None,:]).flatten(-2),outside.double()[...,None]),-1)
    assert not K.requires_grad and (K>=0).all() and (K.sum(-1)-1).abs().max()<2e-14
    return K,dict(x=ax,y=ay,outside=outside,grid=grid,coordinate_system='original_image_normalized_cxcywh; padding excluded',outside_atom_index=grid*grid)

def spatial_loss(logits,q,K):
    """Rowwise stable KL(qK||pK), support fixed; no epsilon mixing/clipping."""
    assert not q.requires_grad and not K.requires_grad
    logp=logits.log_softmax(-1);q=q.detach();K=K.detach()
    # The union of geometric support is fixed within each row. Removing zero
    # columns avoids undefined all-minus-infinity logsumexp gradients.
    losses=[];Qs=[];Ps=[]
    for j in range(len(q)):
        support=((K[j]>0)&torch.isfinite(logits[j])[:,None]).any(0);kk=K[j,:,support];lk=kk.log()
        lP=torch.logsumexp(logp[j,:,None]+lk,0)
        Q=(q[j,:,None]*kk).sum(0);positive=Q>0
        # Only q-positive cells enter KL; logP is finite there because p and q
        # have the identical original candidate validity support.
        assert torch.isfinite(lP[positive]).all()
        loss=(Q[positive]*(Q[positive].log()-lP[positive])).sum()
        P=K.new_zeros(K.shape[-1]);Qfull=P.clone();P[support]=lP.exp();Qfull[support]=Q
        losses.append(loss);Qs.append(Qfull);Ps.append(P)
    return torch.stack(losses),torch.stack(Qs).detach(),torch.stack(Ps).detach()

def fit(E,ctx,eps,cfg,planned,mode,allowed,teacher_cache=None,counts=None):
    import hashlib,time
    cache={}if teacher_cache is None else teacher_cache;counts={}if counts is None else counts
    n=len(E);eta=torch.zeros(n,2,dtype=torch.float64);path=[dict(eta=eta.clone(),boxes=E.clone())];rounds=[]
    for step in range(2):
        center=eta.clone()if mode=='Live'else torch.zeros_like(eta);ee=eps[step].detach();b=candidates(E,center,ee,cfg['center_scale']).detach();d=(center[:,None]+ee).detach()
        key=hashlib.sha256(b.numpy().tobytes()+ee.numpy().tobytes()).hexdigest()
        if key in cache:f=cache[key];counts['teacher_cache_hits']=counts.get('teacher_cache_hits',0)+1
        else:
            tick=time.perf_counter();f=feedback(ctx,b,ee,cfg['temperature']);cache[key]=f;counts['teacher_new_calls']=counts.get('teacher_new_calls',0)+1;counts['teacher_new_candidate_evaluations']=counts.get('teacher_new_candidate_evaluations',0)+b.shape[0]*b.shape[1];counts['teacher_seconds']=counts.get('teacher_seconds',0.)+time.perf_counter()-tick
        active=allowed&f['informative'];v=eta.clone().requires_grad_(True);lp=student_logits(d,v,f['valid']);sp=lp.softmax(-1);q=f['prob'].detach();K,encoding=kernels(b)
        loss,Q,P=spatial_loss(lp,q,K);raw=torch.autograd.grad(loss.sum(),v)[0];grad=raw.detach().clone();grad[~active]=0
        new=eta-cfg['step_size']*grad;out=candidates(E,new,torch.zeros(n,1,2),cfg['center_scale'])[:,0];out[~allowed]=E[~allowed]
        assert torch.isfinite(new).all() and torch.equal(out[:,2:],E[:,2:])and torch.equal(out[~allowed],E[~allowed])
        assert (P.sum(-1)-1).abs().max()<2e-12 and(Q.sum(-1)-1).abs().max()<2e-12
        roundrec=dict(step=step,sampling_center=center,candidates=b,standardized_candidates=d,feedback=f,student_prob=sp.detach(),gradient=grad,raw_unmasked_gradient=raw.detach(),eta_before=eta.clone(),eta_after=new.clone(),spatial_encoding=encoding,spatial_target=Q,spatial_student=P,spatial_KL_per_node=loss.detach(),KL=float(loss[allowed].sum().detach()/max(int(allowed.sum()),1)),updated_frames=int(active.sum()),nonzero_gradient_frames=int((grad.abs().amax(-1)>1e-12).sum()),output_step_L2=float((out-path[-1]['boxes']).norm()),outside_valid_candidates=int((encoding['outside']&f['valid']).sum()),outside_active_candidates=int((encoding['outside']&f['valid']&active[:,None]).sum()),teacher_cache_key=key)
        rounds.append(roundrec);eta=new;path.append(dict(eta=eta.clone(),boxes=out))
    return dict(mode=mode,path=path,rounds=rounds,final=path[-1]['boxes'],selected_step=2,GT_read=False,pretrained_weights_frozen=True,loss='KL(qK||p_eta K)',geometry_stop_gradient=True)

def properties():
    torch.set_num_threads(4)
    b=torch.tensor([[[.43,.5,.35,.4],[.5,.5,.35,.4],[.57,.5,.35,.4],[1.5,.5,.1,.1],[.123,.234,.0001,.0002]]],dtype=torch.float64)
    K,enc=kernels(b);assert enc['outside'][0,3]and K[0,3,-1]==1 and K[0,4].sum()==1
    d=torch.tensor([[[-1.,0],[0.,0],[1.,0],[2.,0],[0.,1.]]],dtype=torch.float64);eta=torch.zeros(1,2,dtype=torch.float64,requires_grad=True);valid=torch.ones(1,5,dtype=torch.bool);lp=student_logits(d,eta,valid);p=lp.softmax(-1);q=p.detach();l,_,_=spatial_loss(lp,q,K);g=torch.autograd.grad(l.sum(),eta)[0];assert abs(float(l))<1e-12 and g.abs().max()<1e-12
    q=torch.tensor([[.12,.25,.55,.03,.05]],dtype=torch.float64)
    def loss(v,spatial):
        lp=student_logits(d,v,valid);return spatial_loss(lp,q,K)[0].sum()if spatial else(q*(q.log()-lp.log_softmax(-1))).sum()
    gs=[];fd=[]
    for spatial in [False,True]:
        v=eta.detach().requires_grad_(True);g=torch.autograd.grad(loss(v,spatial),v)[0];numeric=torch.zeros_like(v)
        for j in range(2):
            h=torch.zeros_like(v);h[0,j]=1e-5;numeric[0,j]=(loss(v.detach()+h,spatial)-loss(v.detach()-h,spatial))/(2e-5)
        assert torch.allclose(g,numeric,rtol=1e-7,atol=1e-9);gs.append(g);fd.append(float((g-numeric).abs().max()))
    assert (gs[0]-gs[1]).abs().max()>1e-3
    from vg_tta.stvg_roi_layout_opd32_v1 import build
    from vg_tta.stvg_discriminative_dense32_v1 import noise
    torch.manual_seed(57);maps=torch.randn(3,8,4,5);vv=torch.ones(3,4,5,dtype=torch.bool);E=torch.tensor([[.5,.5,.4,.4]]*3,dtype=torch.float64);ctx=build(maps,vv,[],[]);cfg=dict(center_scale=.25,temperature=.1,step_size=1.)
    for mode in ['Fixed','Live']:assert torch.equal(fit(E,ctx,noise('spatial-no-reference-test',3),cfg,[0],mode,torch.tensor([False,True,True]))['final'],E)
    return dict(status='pass',K_unit_mass=True,subcell_small_boxes_preserved=True,padding_excluded_by_original_image_grid=True,explicit_outside_atom=True,q_equals_p_zero_loss_and_gradient=True,A_B_overlap_gradients_differ=True,finite_difference_max_errors=fd,no_reference_two_round_exact_noop=True,A_gradient=gs[0].tolist(),B_gradient=gs[1].tolist(),no_epsilon_mixture=True)
