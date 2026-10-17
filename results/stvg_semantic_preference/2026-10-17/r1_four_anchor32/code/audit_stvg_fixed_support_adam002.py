"""Independent float32 Adam audit with operation-derived roundoff bounds, unchanged delta gate."""
import math,numpy as np
from scripts.audit_stvg_spatial_lite_fit002 import _arr,_require,_err

def adam_audit(row,names):
    gradient=_arr(row['gradient']).astype(float).reshape(-1);delta=_arr(row['original_adam_delta']).astype(float).reshape(-1)
    before=row['optimizer_before'];proposed=row['optimizer_proposed'];group=proposed['param_groups'][0];b1,b2=group['betas'];lr=float(group['lr']);epsilon=float(group['eps']);pids=group['params'];_require(len(pids)==len(names),'Adam parameter order')
    eps=np.finfo(np.float32).eps;gamma8=8*eps/(1-8*eps);tiny=np.finfo(np.float32).tiny;offset=0;me=ve=de=mr=vr=0.
    for pid,name in zip(pids,names):
        n=_arr(row['state_before'][name]).size;g=gradient[offset:offset+n];old=before['state'].get(pid);new=proposed['state'][pid]
        _require(str(new['exp_avg'].dtype)=='torch.float32' and str(new['exp_avg_sq'].dtype)=='torch.float32','audit requires actual native float32 Adam moments')
        m=np.zeros(n) if old is None else _arr(old['exp_avg']).astype(float).reshape(-1);v=np.zeros(n) if old is None else _arr(old['exp_avg_sq']).astype(float).reshape(-1)
        nm=_arr(new['exp_avg']).astype(float).reshape(-1);nv=_arr(new['exp_avg_sq']).astype(float).reshape(-1)
        expected_m=b1*m+(1-b1)*g;expected_v=b2*v+(1-b2)*g*g
        # Native lerp/subtract/multiply/add (or FMA) uses at most8 rounded operations.
        # First moment uses the stable lerp form m+alpha*(g-m); cancellation is bounded
        # by absolute operands rather than a relative tolerance on a near-zero result.
        mb=gamma8*(np.abs(m)+np.abs(g))+tiny;vb=gamma8*(abs(b2)*np.abs(v)+abs(1-b2)*g*g)+tiny
        em=np.abs(nm-expected_m);ev=np.abs(nv-expected_v)
        _require(np.isfinite(nm).all() and np.isfinite(nv).all() and (nv>=0).all(),'nonfinite/negative actual Adam state')
        _require((em<=mb).all() and (ev<=vb).all(),'Adam moments exceed float32 operation roundoff bounds')
        oldstep=0. if old is None else float(_arr(old['step']));step=float(_arr(new['step']));_require(step==oldstep+1,'Adam step counter')
        # Verified actual moments determine the proposal. Keep the previous strict
        # absolute delta threshold, including float32 rounding at the parameter value.
        expected_delta=(-lr/(1-b1**step)*nm/(np.sqrt(nv/(1-b2**step))+epsilon)).astype(np.float32)
        err=_err(expected_delta,delta[offset:offset+n]);_require(err<=5e-6,'Adam proposal differs from verified moments')
        me=max(me,float(em.max()));ve=max(ve,float(ev.max()));de=max(de,err);mr=max(mr,float((em/mb).max()));vr=max(vr,float((ev/vb).max()));offset+=n
    _require(offset==len(gradient)==len(delta),'Adam flatten size')
    return dict(moment_error=me,variance_error=ve,adam_delta_error=de,max_moment_roundoff_fraction=mr,max_variance_roundoff_fraction=vr,roundoff_model='gamma8 native float32 operations; no update change')
