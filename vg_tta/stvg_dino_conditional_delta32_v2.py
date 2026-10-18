"""Teacher-relative correction update; the frozen v1 teacher is reused verbatim."""
import numpy as np
from vg_tta.stvg_dino_conditional_opd32_v1 import ConditionalDecoder,conditional_targets


def distill(mu,targets,actions):
    """Known zero-mean sampling noise is a control variate, not supervision.

    This changes the finite-sample estimator. It is not the minimizer of v1's
    empirical KL to the uncentered teacher-output mixture. At a fixed current
    mean, under zero-mean Gaussian sampling it has the same population mean.
    """
    mu=np.asarray(mu,np.float64);targets=np.asarray(targets,np.float64);actions=np.asarray(actions,np.float64)
    assert mu.shape==(4,)and targets.shape==actions.shape and actions.ndim==2 and actions.shape[1]==4 and len(actions)>0
    assert np.isfinite(mu).all()and np.isfinite(targets).all()and np.isfinite(actions).all()
    corrections=targets-actions;shift=corrections.mean(0);new=mu+shift
    centered=mu+corrections
    before=float(np.square(corrections).sum(-1).mean()/2)
    after=float(np.square(centered-new).sum(-1).mean()/2)
    assert after<=before+1e-10
    identity=(corrections==0).all(1)
    if identity.all():assert np.array_equal(new,mu)
    return new,dict(loss_before=before,loss_after=after,
        loss_space='centered correction diagnostic; not v1 empirical KL',
        original_empirical_KL_before=float(np.square(targets-mu).sum(-1).mean()/2),
        original_empirical_KL_after=float(np.square(targets-new).sum(-1).mean()/2),
        mean_step_l2=float(np.linalg.norm(new-mu)),student_mean_before=mu.copy(),student_mean_after=new.copy(),
        teacher_mean_correction=shift.copy(),identity_candidates=int(identity.sum()),
        all_identity_exact_noop=bool(identity.all()),sample_noise_mean=(actions-mu).mean(0))


def contract():
    rng=np.random.default_rng(20261018);checks=[];mu=rng.normal(size=4)
    for k in [1,8,24,32]:
        a=mu+rng.normal(size=(k,4));new,info=distill(mu,a.copy(),a)
        assert np.array_equal(new,mu)and info['all_identity_exact_noop'];checks.append('identity_'+str(k))
    initial=mu.copy()
    for _ in range(100):
        a=mu+rng.normal(size=(8,4));mu,_=distill(mu,a.copy(),a)
    assert np.array_equal(mu,initial);checks.append('100_identity_rounds_exact')
    a=mu+rng.normal(size=(8,4));y=a.copy();y[2]+=[.3,-.2,.1,.4];y[5]+=[-.1,.5,0.,-.2]
    new,info=distill(mu,y,a);expected=mu+((y[2]-a[2])+(y[5]-a[5]))/8
    assert np.allclose(new,expected,rtol=0,atol=1e-15)and info['identity_candidates']==6
    checks.append('partial_feedback_divides_all_K_not_only_accepted')
    # Same correction vectors with arbitrary different sampling locations.
    aa=mu+rng.normal(size=(8,4));yy=aa+(y-a);nn,_=distill(mu,yy,aa)
    assert np.allclose(nn,new,rtol=0,atol=1e-15);checks.append('sampling_mean_does_not_add_motion')
    eps=rng.normal(size=(12,4));a=mu+np.r_[eps,-eps];y=a+rng.normal(size=(24,4))*.1
    new,_=distill(mu,y,a);assert np.allclose(new,y.mean(0),rtol=0,atol=1e-15)
    checks.append('zero_empirical_noise_mean_matches_old_update')
    return dict(status='pass',checks=checks,count=len(checks),new_models=0,GT_read=False)
