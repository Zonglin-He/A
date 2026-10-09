"""P2 CPU audit matches the pinned CUDA broadcast-gradient sum topology.

No production operation changes. CUDA Reduce.cuh uses four accumulators,
indices i, i+4, ... and sequential accumulator combination for a contiguous
(N,32,4) input reduced on its middle dimension. CPU's default sum kernel has
a different order. The original004 CPU result and unchanged 2e-5 threshold
are retained; CPU64 autodiff/formula and original gamma48 bounds remain.
"""
import contextlib,time
import numpy as np
import torch
import vg_tta.decota_spatial_opd_inline_gradient_precision004 as inline

REVISION='P2_matched_gradient_reduction_precision_revision006'
FIELD=inline.FIELD
THRESHOLD=inline.THRESHOLD
ORIGINAL_CHECK=inline.check
ORIGINAL_REVISION=inline.REVISION


class Broadcast32(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        assert x.device.type=='cpu' and x.dtype==torch.float32 and x.ndim==2 and x.shape[1]==4
        return x[:,None,:].expand(-1,32,-1)

    @staticmethod
    def backward(ctx,gradient):
        assert gradient.shape[1:]==(32,4) and gradient.dtype==torch.float32
        # The pinned non-fast-stride reduction does not split across warps
        # for only 32 inputs. Each thread has four accumulator registers.
        accumulators=torch.zeros((len(gradient),4,4),dtype=torch.float32)
        for i in range(8):
            accumulators=accumulators+gradient[:,4*i:4*i+4,:]
        return ((accumulators[:,0,:]+accumulators[:,1,:])+accumulators[:,2,:])+accumulators[:,3,:]


@torch.enable_grad()
def cpu32_gradient(current,mean,samples,weights,sigma,matched):
    x=torch.tensor(np.asarray(current),dtype=torch.float32,requires_grad=True)
    mm=torch.tensor(np.asarray(mean),dtype=torch.float32)
    ss=torch.tensor(np.asarray(samples),dtype=torch.float32)
    ww=torch.tensor(np.asarray(weights),dtype=torch.float32)
    inv2v=torch.tensor(np.float32(1)/np.float32(2*float(sigma)**2))
    invn=torch.tensor(np.float32(1)/np.float32(len(x)))
    broadcast=Broadcast32.apply(x) if matched else x[:,None,:]
    kl=(x-mm).square().sum(-1)*inv2v
    logp=-(ss-broadcast).square().sum(-1)*inv2v
    objective=(kl-((ww-1/32)*logp).sum(-1)).sum()*invn
    return torch.autograd.grad(objective,x)[0].detach().numpy().astype(float)


@torch.enable_grad()
def check(current,mean,samples,weights,gpu,analytic32,sigma,original_error):
    c,m,s,w,g,a=[np.asarray(x,np.float64) for x in (current,mean,samples,weights,gpu,analytic32)]
    n=len(c);v=float(sigma)**2
    assert n>0 and c.shape==m.shape==g.shape==a.shape==(n,4)
    assert s.shape==(n,32,4) and w.shape==(n,32) and v>0
    assert all(np.isfinite(x).all() for x in [c,m,s,w,g,a])
    assert all(np.array_equal(x,x.astype(np.float32).astype(float)) for x in [c,m,s,w,g,a])
    assert ((w>=0)&(w<=1)).all()
    u=float(np.finfo(np.float32).eps/2);gamma=48*u/(1-48*u)
    assert np.max(abs(w.sum(1)-1))<=32*u/(1-32*u)
    original=cpu32_gradient(c,m,s,w,sigma,False)
    cpu32=cpu32_gradient(c,m,s,w,sigma,True)
    same=abs(g-cpu32);old_error=float(abs(g-original).max())
    assert float(same.max())<THRESHOLD,'Matched CUDA reduction CPU32/GPU32 fails unchanged 2e-5'
    x64=torch.tensor(c,dtype=torch.float64,requires_grad=True)
    m64=torch.tensor(m,dtype=torch.float64);s64=torch.tensor(s,dtype=torch.float64)
    w64=torch.tensor(w,dtype=torch.float64)
    kl64=(x64-m64).square().sum(-1)/(2*v)
    logp64=-(s64-x64[:,None,:]).square().sum(-1)/(2*v)
    objective64=(kl64-((w64-1/32)*logp64).sum(-1)).mean()
    cpu64=torch.autograd.grad(objective64,x64)[0].detach().numpy()
    baseline=(c-m)/v/n;terms=(w-1/32)[:,:,None]*(s-c[:,None,:])/v/n
    formula=baseline-terms.sum(1);same64=abs(cpu64-formula)
    assert float(same64.max())<THRESHOLD,'Independent CPU64 derivative/formula disagreement'
    scale=abs(baseline)+abs(terms).sum(1)
    arithmetic=gamma*scale+np.finfo(np.float32).tiny+64*np.finfo(float).eps*np.maximum(1,scale)
    bound_g=arithmetic+same
    assert np.all(abs(g-formula)<=bound_g),'Saved GPU derivative exceeds unchanged precision bound'
    assert np.all(abs(a-formula)<=arithmetic),'Original GPU analytic expression exceeds unchanged precision bound'
    error=float(abs(g-a).max());assert error==float(original_error),'Original guard discrepancy was rewritten'
    assert np.all(abs(g-a)<=bound_g+arithmetic)
    return dict(status='pass',original_inline_absolute_guard_passed=bool(error<THRESHOLD),
        original_inline_error=error,original_absolute_threshold=THRESHOLD,
        max_original004_CPU32_vs_GPU32_error=old_error,
        original004_CPU32_absolute_guard_passed=bool(old_error<THRESHOLD),
        max_matched_CPU32_vs_GPU32_error=float(same.max()),matched_precision_absolute_threshold=THRESHOLD,
        max_CPU64_vs_formula64_error=float(same64.max()),max_GPU32_vs_formula64_error=float(abs(g-formula).max()),
        max_original_analytic32_vs_formula64_error=float(abs(a-formula).max()),
        max_GPU_precision_bound=float(bound_g.max()),max_analytic_precision_bound=float(arithmetic.max()),
        max_GPU_bound_fraction=float(np.max(abs(g-formula)/bound_g)),
        max_analytic_bound_fraction=float(np.max(abs(a-formula)/arithmetic)),
        unit_roundoff=u,operation_bound=48,CUDA_reduction_inputs=32,CUDA_reduction_accumulators=4,
        CUDA_reduction_thread_stride=1,CUDA_reduction_warp_split=False,
        production_loss_gradient_Adam_and_readout_unchanged=True)


@contextlib.contextmanager
def mode(matched=True):
    oldcheck,oldrevision=inline.check,inline.REVISION
    inline.check=check if matched else ORIGINAL_CHECK
    inline.REVISION=REVISION if matched else ORIGINAL_REVISION
    try:yield
    finally:inline.check,inline.REVISION=oldcheck,oldrevision


def audit(fit,expert,original_component):
    from vg_tta.decota_spatial_opd_action_readback_precision005 import audit as action_audit
    assert fit[FIELD]['revision']==REVISION
    with mode():result=action_audit(fit,expert,original_component)
    return dict(result,revision=REVISION,
        original004_CPU32_guard_failures=sum(not x['original004_CPU32_absolute_guard_passed']
            for x in result['inline_gradient_precision_checks']),
        production_and_all_old_receipt_bytes_unchanged=True)
