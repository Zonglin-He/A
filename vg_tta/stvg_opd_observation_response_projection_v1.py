"""New candidate operator. Original decoder/OPD sources are not modified.

Private witnesses contain complete Jacobians; public outputs must use anonymous
scalar projections. No GT, teacher forward, temporal change, or stream write.
"""
import math,time
import numpy as np
import torch
from methods.decota_final_simplified_v1.tensors import detached
from vg_tta.c1_enabling_tricks_v1 import TrickReplay
from vg_tta.decota_spatial_opd_tunable_v1 import rollout,likelihood_loss
from vg_tta.decota_spatial_opd_boundary_chart_proposal003 import proposed_coordinates
from vg_tta.decota_optimizer_posterior_r1_v1 import flat
from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
from vg_tta.decota_spatial_opd_matched_gradient_precision006 import check as gradient_check

EPS32=np.finfo(np.float32).eps

def solve(Jo,Ju,e,da,unobserved_frames):
    """Exact finite-dimensional ball LS in the observed null space, float64."""
    from scipy.linalg import eigh,svd
    Jo,Ju,e,da=[np.asarray(a,np.float64) for a in (Jo,Ju,e,da)]
    assert Jo.ndim==Ju.ndim==2 and Jo.shape[1]==Ju.shape[1]==len(da)
    assert len(e)==len(Ju) and all(np.isfinite(x).all() for x in (Jo,Ju,e,da))
    u,s,vt=svd(Jo,full_matrices=False,check_finite=True,lapack_driver='gesvd')
    cutoff=max(1e-12,1e-10*float(s.max(initial=0)))
    rank=int(np.sum(s>cutoff));Q=vt[:rank]
    v=Jo@da;p=Q.T@(Q@da)
    radius2=max(0.,float(da@da-p@p));radius=math.sqrt(radius2)
    W=1/max(1,unobserved_frames)
    C=(Ju-(Ju@Q.T)@Q)*math.sqrt(W);b=(e+Ju@p)*math.sqrt(W)
    # The Moore-Penrose null component lies in the row space of C. Zero
    # eigenvalues do not introduce arbitrary directions.
    gram=C@C.T
    ev,V=eigh(gram,check_finite=True);ev=np.maximum(ev,0.)
    eig_cut=max(1e-24,1e-12*float(ev.max(initial=0)))
    a=V.T@b;keep=ev>eig_cut
    def at(lam):
        coeff=np.zeros_like(a)
        coeff[keep]=a[keep]/(ev[keep]+lam)
        return -C.T@(V@coeff)
    z=at(0.);lam=0.
    if np.linalg.norm(z)>radius:
        low=0.;high=max(1e-16,float(ev.max(initial=0)))
        while np.linalg.norm(at(high))>radius:high*=2
        for _ in range(80):
            middle=(low+high)/2
            if np.linalg.norm(at(middle))>radius:low=middle
            else:high=middle
        lam=high;z=at(lam)
    df=p+z
    station=C.T@(b+C@z)+lam*z
    before=float(np.square(e+Ju@da).sum()*W)
    after=float(np.square(e+Ju@df).sum()*W)
    witness=dict(rank=rank,rank_cutoff=cutoff,original_norm=float(np.linalg.norm(da)),
      P_norm=float(np.linalg.norm(p)),F_norm=float(np.linalg.norm(df)),radius=radius,
      lagrange_multiplier=lam,linear_A_objective=before,linear_F_objective=after,
      equality_P_error=float(np.linalg.norm(Jo@p-v)),equality_F_error=float(np.linalg.norm(Jo@df-v)),
      null_component_observed_error=float(np.linalg.norm(Jo@z)),
      stationarity_norm=float(np.linalg.norm(station)),null_functional_Frobenius=float(np.linalg.norm(C)),
      solver='observed thin SVD / projected output Gram / ball LS',iterations=80 if lam else 0)
    assert witness['F_norm']<=witness['original_norm']+1e-7
    assert after<=before+1e-7*max(1.,before)
    return p,df,witness

def readout(rp,positions):
    outputs=[]
    hook=rp.base.decoder.decoder.bbox_embed.register_forward_hook(lambda module,args,value:outputs.append(value))
    try:result=rp.values()
    finally:hook.remove()
    assert len(outputs)==12
    raw=torch.stack([outputs[5+(i%2)*6][i//2,0] for i in range(len(rp.ids))]).float()
    boxes=result['boxes'];assert torch.equal(raw.sigmoid(),boxes)
    mu,boundary=proposed_coordinates(boxes[positions],raw[positions])
    return boxes,mu,raw,boundary

def restore_vector(rp,origin,delta):
    delta=torch.as_tensor(delta,device=rp.base.delta.device,dtype=torch.float32)
    with torch.no_grad():
        j=0
        for n,p in rp.named:
            p.copy_(origin[n].to(p)+delta[j:j+p.numel()].reshape_as(p));j+=p.numel()
    assert j==1792

def actual_delta(rp,origin):
    return torch.cat([(p.detach()-origin[n].to(p)).flatten() for n,p in rp.named]).cpu().numpy().astype(float)

def output_jacobian(output,params,chunk=8):
    """All rows; bounded vectorized reverse-mode, original native graph."""
    out=output.flatten();rows=[];start=time.perf_counter()
    for first in range(0,len(out),chunk):
        last=min(len(out),first+chunk)
        basis=torch.zeros(last-first,len(out),device=out.device,dtype=out.dtype)
        basis[torch.arange(last-first,device=out.device),torch.arange(first,last,device=out.device)]=1
        grads=torch.autograd.grad(out,params,grad_outputs=basis,is_grads_batched=True,retain_graph=True)
        rows.append(torch.cat([g.flatten(1) for g in grads],1).detach().cpu().double().numpy())
    result=np.concatenate(rows,0) if rows else np.empty((0,1792))
    assert np.isfinite(result).all()
    return result,dict(actual_scalar_VJP_rows=len(out),batched_reverse_calls=len(rows),
      chunk=chunk,seconds=time.perf_counter()-start,complete_parameter_Jacobian=True)

def derivative_checks(rp,positions,unvisited,y,params,J,da,seed):
    """Actual graph VJP, double-backward JVP and real decoder finite differences."""
    origin=rp.state();gen=np.random.default_rng(seed)
    w=gen.normal(size=len(y));w/=np.linalg.norm(w)
    wt=torch.tensor(w,device=y.device,dtype=y.dtype)
    grads=torch.autograd.grad(y,params,grad_outputs=wt,retain_graph=True)
    measured=torch.cat([g.flatten() for g in grads]).detach().cpu().double().numpy()
    expected=J.T@w
    vjp=float(np.linalg.norm(measured-expected));vjp_limit=2e-6+2e-5*np.linalg.norm(expected)
    assert vjp<=vjp_limit,(vjp,vjp_limit)
    directions=[gen.normal(size=1792),np.asarray(da,float)]
    checks=[];pending=[]
    for direction in directions:
        length=np.linalg.norm(direction)
        if not length:continue
        direction=direction/length;predicted=J@direction
        cot=torch.zeros_like(y,requires_grad=True)
        pull=torch.autograd.grad(y,params,grad_outputs=cot,create_graph=True,retain_graph=True)
        pieces=[];i=0
        for p in params:pieces.append(torch.as_tensor(direction[i:i+p.numel()],device=p.device,dtype=p.dtype).reshape_as(p));i+=p.numel()
        jvp=torch.autograd.grad(pull,cot,grad_outputs=pieces,retain_graph=True)[0].detach().cpu().double().numpy()
        err=float(np.linalg.norm(jvp-predicted));limit=2e-6+2e-5*np.linalg.norm(predicted)
        assert err<=limit,(err,limit)
        pending.append((direction,predicted,err,limit))
    # Finish all double-backward calls before finite differences mutate the
    # original leaf parameter version counters.
    for direction,predicted,err,limit in pending:
        fd=[]
        for h in [.002,.001]:
            restore_vector(rp,origin,h*direction)
            with torch.no_grad():bp,mp,_,_=readout(rp,positions);yp=torch.cat((mp.flatten(),bp[unvisited].flatten())).cpu().double().numpy()
            restore_vector(rp,origin,-h*direction)
            with torch.no_grad():bm,mm,_,_=readout(rp,positions);ym=torch.cat((mm.flatten(),bm[unvisited].flatten())).cpu().double().numpy()
            measured=(yp-ym)/(2*h);error=float(np.linalg.norm(measured-predicted))
            limit_fd=1e-4+.02*np.linalg.norm(predicted)
            fd.append(dict(h=h,absolute_error=error,predicted_norm=float(np.linalg.norm(predicted)),
              relative_error=error/max(1e-12,float(np.linalg.norm(predicted))),tolerance=limit_fd,pass_check=error<=limit_fd))
        checks.append(dict(JVP_error=err,JVP_tolerance=limit,finite_differences=fd))
    rp.restore(origin)
    return dict(status='pass' if all(x['pass_check'] for d in checks for x in d['finite_differences']) else 'failed_FD_tolerance',
      VJP_error=vjp,VJP_tolerance=vjp_limit,directions=checks,independent_finite_decoder_replay=True)

def shadows(rp,cfg,adam_before=None):
    pp=[torch.nn.Parameter(p.detach().clone()) for _,p in rp.named]
    opt=torch.optim.Adam(pp,lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
    if adam_before is not None:opt.load_state_dict(detached(adam_before,'cpu'))
    return pp,opt

def proposal(rp,pp,opt,grads):
    prior=[p.detach().clone() for _,p in rp.named]
    before=detached(opt.state_dict(),'cpu')
    with torch.no_grad():
        for shadow,value,g in zip(pp,prior,grads):shadow.copy_(value);shadow.grad=g.detach().clone()
    opt.step()
    da=torch.cat([(p.detach()-a).flatten() for p,a in zip(pp,prior)]).cpu().numpy().astype(float)
    return da,before,detached(opt.state_dict(),'cpu')

def commit_Adam_proposal(rp,pp):
    # The actual Adam result is the state. Subtract/add round-tripping its
    # float32 delta need not reproduce it bitwise.
    with torch.no_grad():
        for (_,p),shadow in zip(rp.named,pp):p.copy_(shadow)

def step(rp,expert,positions,evidence,visited,key,cfg,k,reference,arm,pp,opt,
         lambda_R=None,need_curvature=False,check_derivatives=False,seed=20261013):
    assert arm in ['P','F','R']
    origin=rp.state();params=[p for _,p in rp.named];start=time.perf_counter()
    boxes,mu,raw,boundary=readout(rp,positions)
    unvisited=[i for i in range(len(rp.ids)) if i not in set(visited)]
    entry=dict(step=k,state=detached(origin,'cpu'),boxes=detached(boxes,'cpu'))
    if not positions:
        return entry,dict(round=k,updated=False,no_information=True,empty=True,lambda_R=0.,
          complete_original_no_op=True,seconds=time.perf_counter()-start),lambda_R
    rr=rollout(mu.detach(),evidence,key,k,config=cfg)
    loss=likelihood_loss(mu,rr['mean'],rr['samples'],rr['weights'],cfg['sigma'])
    gmu=torch.autograd.grad(loss,mu,retain_graph=True)[0]
    analytic=((mu.detach()-rr['mean'])/cfg['sigma']**2-
      ((rr['weights']-1/32)[:,:,None]*(rr['samples']-mu.detach()[:,None,:])).sum(1)/cfg['sigma']**2)/len(positions)
    def a(t):return t.detach().cpu().numpy().astype(float)
    checked=gradient_check(a(mu),a(rr['mean']),a(rr['samples']),a(rr['weights']),a(gmu),a(analytic),cfg['sigma'],float((gmu-analytic).abs().max()))
    witness=dict(round=k,rollout=detached(rr,'cpu'),mean=detached(mu,'cpu'),raw_logits=detached(raw[positions],'cpu'),
      boundary=detached(boundary,'cpu'),autograd_mean_gradient=detached(gmu,'cpu'),analytic_gradient=detached(analytic,'cpu'),
      loss=float(loss.detach()),gradient_check=checked,updated=rr['informative'],no_information=not rr['informative'],GT_used=False)
    if not rr['informative']:
        witness.update(complete_original_no_op=True,seconds=time.perf_counter()-start)
        return entry,witness,lambda_R
    gopd=torch.autograd.grad(loss,params,retain_graph=True)
    e=boxes[unvisited]-reference.to(boxes)[unvisited]
    D=e.square().sum()/(2*max(1,len(unvisited)))
    need_Ju=arm=='F' or need_curvature or check_derivatives
    outputs=torch.cat((mu.flatten(),boxes[unvisited].flatten())) if need_Ju else mu.flatten()
    if arm!='R' or need_curvature or check_derivatives:
        J,jstats=output_jacobian(outputs,params);Jo=J[:mu.numel()];Ju=J[mu.numel():]
        witness.update(J_O=torch.from_numpy(Jo),J_U=torch.from_numpy(Ju),Jacobian=jstats)
    else:Jo=Ju=None
    if need_curvature:
        numerator=float(np.square(Jo).sum()/(len(positions)*cfg['sigma']**2))
        denominator=float(np.square(Ju).sum()/max(1,len(unvisited)))
        lambda_R=numerator/denominator if denominator>0 and unvisited else 0.
        witness['curvature']=dict(numerator=numerator,denominator=denominator,lambda_R=lambda_R,
          denominator_defined=bool(denominator>0 and unvisited))
    if arm=='R':
        assert lambda_R is not None
        greg=torch.autograd.grad(D,params,retain_graph=True)
        grads=[g+lambda_R*r for g,r in zip(gopd,greg)]
        witness.update(lambda_R=lambda_R,consistency_gradient=flat(greg).detach().cpu(),consistency_loss=float(D.detach()))
    else:grads=gopd
    # Qualification derivatives precede parameter mutations. Dummy original
    # delta is subsequently checked against the actual Adam proposal.
    if check_derivatives:
        oldmoment=detached(opt.state_dict(),'cpu')
        da_probe,_,_=proposal(rp,pp,opt,gopd);opt.load_state_dict(oldmoment)
        witness['derivative_checks']=derivative_checks(rp,positions,unvisited,outputs,params,J,da_probe,seed)
        # restore() has changed parameter version counters; derivatives were
        # already obtained, and no graph is used after this point.
    da,ab,aa=proposal(rp,pp,opt,grads)
    witness.update(original_OPD_gradient=flat(gopd).detach().cpu(),gradient=flat(grads).detach().cpu(),
      Adam_before=ab,Adam_after=aa,d_A=torch.from_numpy(da),unobserved_positions=unvisited,visited_positions=list(visited))
    commit_Adam_proposal(rp,pp)
    with torch.no_grad():ba,ma,_,_=readout(rp,positions)
    ba=ba.detach().clone();ma=ma.detach().clone()
    DA=float((ba[unvisited]-reference.to(ba)[unvisited]).double().square().sum()/(2*max(1,len(unvisited))))
    candidate=da;solver=None
    if arm in ['P','F']:
        ev=e.detach().cpu().double().flatten().numpy()
        if arm=='P':
            from scipy.linalg import svd
            u,s,vt=svd(Jo,full_matrices=False,lapack_driver='gesvd');rank=int((s>max(1e-12,1e-10*s.max(initial=0))).sum())
            candidate=vt[:rank].T@(vt[:rank]@da)
            solver=dict(rank=rank,P_norm=float(np.linalg.norm(candidate)),original_norm=float(np.linalg.norm(da)))
        else:
            dp,df,solver=solve(Jo,Ju,ev,da,len(unvisited));candidate=dp if arm=='P' else df
        restore_vector(rp,origin,candidate)
        quantized=actual_delta(rp,origin)
        with torch.no_grad():bc,mc,_,_=readout(rp,positions)
        DC=float((bc[unvisited]-reference.to(bc)[unvisited]).double().square().sum()/(2*max(1,len(unvisited))))
        obs_error=float((mc.double()-ma.double()).norm());obs_response=float((ma.double()-mu.detach().double()).norm())
        obs_tol=.10*obs_response+32*EPS32*math.sqrt(mu.numel())
        linear_error=float(np.linalg.norm(Jo@(quantized-da)));linear_tol=2e-6+2e-5*np.linalg.norm(Jo@da)
        norm=float(np.linalg.norm(quantized));normA=float(np.linalg.norm(da))
        d_tol=1e-10+64*EPS32*DA
        tests=dict(observed_response=obs_error<=obs_tol,linear_response=linear_error<=linear_tol,
          parameter_norm=norm<=normA+4e-6)
        if arm=='F':tests.update(actual_functional_nonincrease=DC<=DA+d_tol,actual_functional_improvement=DC<DA-d_tol)
        accepted=all(tests.values());reasons=[n for n,p in tests.items() if not p]
        witness.update(candidate=torch.from_numpy(candidate),actual_candidate=torch.from_numpy(quantized),solver=solver,
          acceptance=tests,accepted_projection=accepted,fallback=not accepted,fallback_reasons=reasons,
          observed_error=obs_error,observed_response_norm=obs_response,observed_tolerance=obs_tol,
          observed_relative_error=obs_error/max(1e-12,obs_response),linear_error=linear_error,linear_tolerance=linear_tol,
          original_norm=normA,candidate_norm=norm,D_A=DA,D_candidate=DC,D_tolerance=d_tol,
          original_proposal_boxes=detached(ba,'cpu'),candidate_boxes=detached(bc,'cpu'),
          original_proposal_mean=detached(ma,'cpu'),candidate_mean=detached(mc,'cpu'))
        if not accepted:commit_Adam_proposal(rp,pp)
    else:witness.update(accepted_projection=None,fallback=False,lambda_R=lambda_R)
    raw_applied=torch.cat([(p.detach()-origin[n].to(p)).flatten() for n,p in rp.named]).cpu()
    entry['update']=dict(gradient=flat(grads).detach().cpu(),raw=raw_applied,optimizer='Adam_proposal_then_'+arm,lr=cfg['lr'],names=[n for n,p in rp.named])
    with torch.no_grad():post,mpost,_,_=readout(rp,positions)
    witness.update(final_boxes=detached(post,'cpu'),final_mean=detached(mpost,'cpu'),
      central_reward_before=aligned_iou(boxes[positions].detach(),evidence).cpu(),
      central_reward_after=aligned_iou(post[positions],evidence).cpu(),actual_raw=raw_applied,
      seconds=time.perf_counter()-start)
    return entry,witness,lambda_R

def fit(base,initial,expert,ids,key,cfg,visited,arm):
    from vg_tta.decota_optimizer_posterior_r1_v1 import Energy
    base.restore(initial);rp=TrickReplay(base,ids,key,{})
    with torch.no_grad():reference=readout(rp,[])[0].detach().clone()
    support=Energy(expert,reference,'top1');positions=[int(x[0]) for x in support.frames]
    evidence=torch.cat([x[1] for x in support.frames],0).detach() if positions else reference.new_empty((0,4))
    pp,opt=shadows(rp,cfg);path=[];rounds=[];lam=None
    try:
        for k in range(cfg['steps']):
            h,w,lam=step(rp,expert,positions,evidence,visited,key,cfg,k,reference,arm,pp,opt,lambda_R=lam,need_curvature=arm=='R' and k==0)
            path.append(h);rounds.append(w)
        with torch.no_grad():final=detached(readout(rp,positions)[0],'cpu')
        state=detached(rp.state(),'cpu');path.append(dict(step=cfg['steps'],state=state,boxes=final))
        return dict(arm=arm,initial=detached(initial,'cpu'),state=state,before=reference.cpu(),final=final,path=path,rounds=rounds,
          config=cfg,selected_step=cfg['steps'],gradient_calls=sum(x['updated'] for x in rounds),empty=not positions,
          positions=positions,frame_metadata=support.frame_metadata,visited=list(visited),active_parameters=1792,lambda_R=lam,GT_used=False)
    finally:rp.restore(initial);base.restore(base.initial)
