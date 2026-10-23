import numpy as np
from stvg.prototype import one_step,membership

def test_flat_semantics_exact_native():
    p=np.tile([.02,.1,.7],(16,1));q=np.triu(np.ones((16,16)),1);q/=q.sum()
    assert np.array_equal(one_step(p,q)[0],q)

def test_kappa16_formula_and_opposite_offset():
    p=np.random.default_rng(8).uniform(.01,.9,(16,4));q=np.triu(np.ones((16,16)),1);q/=q.sum();w=membership(q)
    _,s=one_step(p,q)
    for v in s['prototypes']:
        train=np.arange(16)%2!=v['heldout_parity'];mu=p[train].mean(0)
        expected=((w[train,None]*p[train]).sum(0)+16*mu)/(w[train].sum()+16)
        np.testing.assert_allclose(v['positive'],expected,rtol=0,atol=2e-16)
    pp=p.copy();pp[::2]=.99
    assert np.array_equal(s['prototypes'][0]['positive'],one_step(pp,q)[1]['prototypes'][0]['positive'])

def test_class_axis_permutation():
    p=np.random.default_rng(8).uniform(.01,.9,(16,4));q=np.triu(np.ones((16,16)),1);q/=q.sum()
    np.testing.assert_allclose(one_step(p,q)[0],one_step(p[:,::-1],q)[0],atol=3e-17,rtol=1e-13)
