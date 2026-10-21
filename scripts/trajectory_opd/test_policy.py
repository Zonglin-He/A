import torch
from scripts.trajectory_opd.policy import allocations,noise_for,initial_states,conditional_projection,calibrated_temperature


def test_flat_reward_no_move_and_antithetic_exploration():
    counts=allocations(torch.tensor([.98,.01,.01]));assert sum(counts)==32 and min(counts)>=6
    starts=initial_states('synthetic',.31)
    for m,e in zip(starts,noise_for('synthetic',0,counts)):
        a=m[None]+.31*e;u=torch.ones(len(a),dtype=torch.float64)
        p,w=conditional_projection(m,a,u,1.)
        assert torch.allclose(p,m,atol=1e-7,rtol=0)


def test_mixture_modes_are_not_replaced_by_invalid_cross_mode_average():
    m0=torch.full((8,256),-3.);m1=-m0
    e=torch.zeros(4,8,256);e[:2,:,0]=1;e[2:,:,0]=-1
    u=torch.tensor([2.,2.,-2.,-2.],dtype=torch.float64)
    p0,_=conditional_projection(m0,m0[None]+e,u,1.)
    p1,_=conditional_projection(m1,m1[None]-e,u,1.)
    assert p0.mean()<-2.9 and p1.mean()>2.9
    assert torch.allclose((p0+p1)/2,torch.zeros_like(p0))


def test_ess_calibration_constant_and_affine_invariance():
    u=[torch.linspace(-2,2,n,dtype=torch.float64)for n in (10,10,12)]
    tau,m=calibrated_temperature(u,[10,10,12]);assert abs(m['ESS']-16)<1e-8
    tau2,_=calibrated_temperature([v*7+13 for v in u],[10,10,12]);assert abs(tau2/tau-7)<1e-9
    _,m=calibrated_temperature([torch.ones(n)for n in (10,10,12)],[10,10,12]);assert m['flat'] and m['ESS']==32


def test_linear_reward_projection_matches_exponential_gaussian_tilt():
    # This intentionally confirms that the unimodal local-linear special case
    # remains equivalent to a preconditioned gradient; no universal OPD claim.
    x=torch.linspace(-5,5,20001,dtype=torch.float64)
    sigma=.7;g=.3;tau=.8
    w=(-x.square()/(2*sigma**2)+g*x/tau).softmax(0)
    expected=(w*x).sum();assert abs(expected-sigma**2*g/tau)<1e-8
