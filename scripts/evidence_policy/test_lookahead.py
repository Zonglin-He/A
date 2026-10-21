"""Synthetic CPU checks only; no saved queries, models or labels are read."""
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from scripts.evidence_policy.lookahead import exact_lookahead,instantaneous_scores


def fixture(T=6,D=5):
    g = torch.Generator().manual_seed(831)
    features = F.normalize(torch.randn(T,3,D,dtype=torch.float64,generator=g),dim=-1)
    valid = torch.ones(T,3,dtype=torch.bool)
    return SimpleNamespace(features=features,
        u=torch.rand(T,3,dtype=torch.float64,generator=g),
        guard=torch.rand(T,3,dtype=torch.float64,generator=g),
        link=torch.rand(max(T-1,0),3,3,dtype=torch.float64,generator=g),
        confidence=torch.rand(T,dtype=torch.float64,generator=g),
        active=torch.ones(T,dtype=torch.bool),valid=valid)


def scalar_scores(b,t,m,p):
    incoming = p@b.link[t-1] if t else torch.zeros(3,dtype=torch.float64)
    return b.u[t]+.25*incoming+.1*(m[None]*b.features[t]).sum(-1)-.1*b.guard[t]


def advance_scalar(b,t,m,a):
    if bool(b.active[t]):
        omega = .2*b.confidence[t]*b.u[t,a]
        if bool(omega.detach() != 0):
            m = F.normalize((1-omega)*m+omega*b.features[t,a],dim=-1,eps=1e-12)
    p = F.one_hot(torch.tensor(a),3).to(torch.float64)
    return m,p


def brute_force(b,t,m,p,h,forced=None):
    """Independent scalar recursion, used solely as a tiny test oracle."""
    valid = b.valid[t] if bool(b.active[t]) else torch.tensor([True,False,False])
    scores = scalar_scores(b,t,m,p)
    rho = scores if bool(b.active[t]) else scores*0
    if forced is not None:
        result = rho[forced]
        if h>1 and t+1<len(b.features):
            mm,pp = advance_scalar(b,t,m,forced)
            result = result+brute_force(b,t+1,mm,pp,h-1)
        return result
    probs = torch.softmax((scores/.2).masked_fill(~valid,-torch.inf),dim=-1)
    result = rho.sum()*0
    for a in range(3):
        if bool(valid[a]):
            result = result+probs[a]*brute_force(b,t,m,p,h,forced=a)
    return result


@pytest.mark.parametrize('horizon',[1,2,4])
def test_batched_exact_sum_matches_independent_enumeration(horizon):
    b = fixture()
    b.valid[2,2] = False
    b.active[3] = False
    nodes = torch.tensor([0,1,2,4,5])
    m = F.normalize(torch.arange(25,dtype=torch.float64).reshape(5,5)+1,dim=-1)
    p = torch.tensor([[.2,.3,.5]],dtype=torch.float64).repeat(5,1)
    out = exact_lookahead(b,nodes,m,p,horizon=horizon)
    expected = torch.zeros_like(out['q'])
    for n,t in enumerate(nodes.tolist()):
        for a in range(3):
            if bool(out['effective_valid'][n,a]):
                expected[n,a] = brute_force(b,t,m[n],p[n],horizon,forced=a)
    torch.testing.assert_close(out['q'],expected,atol=5e-16,rtol=3e-16)
    assert out['exact_discrete_expectation']


def test_h1_reproduces_original_reference_rounding_no_prior_no_extra_confidence():
    b = fixture()
    nodes = torch.arange(6)
    m = b.features[:,0].clone()
    p = torch.tensor([[.13,.31,.56]],dtype=torch.float64).repeat(6,1)
    out = exact_lookahead(b,nodes,m,p,horizon=1)
    expected = torch.stack([scalar_scores(b,t,m[t],p[t]) for t in range(6)])
    torch.testing.assert_close(out['q'],expected,atol=1.2e-16,rtol=0)
    changed = copy.deepcopy(b)
    changed.confidence.zero_()
    other = exact_lookahead(changed,nodes,m,p,horizon=1)
    assert torch.equal(other['q'],out['q'])
    assert int(out['costs']['state_transitions']) == 0


def test_forced_current_action_can_change_future_choice_and_overturn_instant_ranking():
    b = fixture(T=2,D=3)
    b.features.zero_()
    b.u[:] = torch.tensor([.11,.1,0.],dtype=torch.float64)
    b.u[1].zero_()
    b.guard.zero_()
    b.link.zero_()
    b.link[0,1,:] = 1.
    nodes = torch.tensor([0])
    m = torch.zeros((1,3),dtype=torch.float64)
    p = torch.tensor([[1.,0.,0.]],dtype=torch.float64)
    h1 = exact_lookahead(b,nodes,m,p,horizon=1)['q']
    h2 = exact_lookahead(b,nodes,m,p,horizon=2)['q']
    assert int(h1.argmax()) == 0
    assert int(h2.argmax()) == 1
    # A second construction verifies conditional future reference probabilities,
    # not only an action-independent additive bonus.
    b.link[0,1] = torch.tensor([0.,0.,1.],dtype=torch.float64)
    first = F.one_hot(torch.tensor(0),3).double()
    second = F.one_hot(torch.tensor(1),3).double()
    q0 = torch.softmax(scalar_scores(b,1,m[0],first)/.2,dim=-1)
    q1 = torch.softmax(scalar_scores(b,1,m[0],second)/.2,dim=-1)
    assert int(q1.argmax()) == 2 and not torch.equal(q0,q1)


def test_terminal_truncation_is_sum_not_average_and_branches_are_actual():
    b = fixture(T=6)
    b.u.fill_(.5)
    b.features.zero_()
    b.guard.zero_()
    b.link.zero_()
    out = exact_lookahead(b,torch.tensor([0,4,5]),torch.zeros(3,5,dtype=torch.float64),
                          torch.tensor([[1.,0.,0.]],dtype=torch.float64).repeat(3,1),horizon=4)
    expected = torch.tensor([[2.,2.,2.],[1.,1.,1.],[.5,.5,.5]],dtype=torch.float64)
    torch.testing.assert_close(out['q'],expected,atol=5e-16,rtol=0)
    assert torch.equal(out['costs']['continuation_paths'],
                       torch.tensor([[27,27,27],[3,3,3],[1,1,1]]))
    assert torch.equal(out['costs']['branch_states_by_depth'],torch.tensor([3,6,9,27]))
    assert torch.equal(out['costs']['valid_action_branches_by_depth'],torch.tensor([9,18,27,81]))
    assert int(out['costs']['enumerated_continuation_paths']) == 93


def test_inactive_start_zero_but_internal_inactive_advances_E_without_memory_write():
    b = fixture(T=4)
    b.active[1] = False
    nodes = torch.tensor([0,1])
    m = b.features[[0,0],1]
    p = torch.tensor([[0.,1.,0.],[0.,0.,1.]],dtype=torch.float64)
    out = exact_lookahead(b,nodes,m,p,horizon=4)
    assert torch.equal(out['q'][1],torch.zeros(3,dtype=torch.float64))
    assert torch.equal(out['effective_valid'][1],torch.tensor([True,False,False]))
    assert int(out['costs']['excluded_inactive_start_states']) == 1
    for a in range(3):
        expected = brute_force(b,0,m[0],p[0],4,forced=a)
        torch.testing.assert_close(out['q'][0,a],expected,atol=4e-16,rtol=3e-16)
    # Internal inactive node has one continuation, not three, despite stored alts.
    assert torch.equal(out['costs']['branch_states_by_depth'],torch.tensor([2,3,3,9]))


def test_all_valid_h4_full_state_and_reference_probability_gradient():
    b = fixture(T=4,D=3)
    nodes = torch.tensor([0,1])
    m = torch.tensor([[.71,.23,-.12],[.15,.62,.37]],dtype=torch.float64,requires_grad=True)
    # Interior simplex parameterization permits a conventional unconstrained
    # gradcheck while the input probabilities remain normalized.
    logits = torch.tensor([[.3,-.1,.2],[.1,.2,-.2]],dtype=torch.float64,requires_grad=True)
    def fn(mm,ll):
        return exact_lookahead(b,nodes,mm,ll.softmax(-1),horizon=4)['q']
    assert torch.autograd.gradcheck(fn,(m,logits),eps=1e-6,atol=3e-7,rtol=2e-5)
    full = fn(m,logits)
    brute = torch.stack([torch.stack([brute_force(b,int(nodes[i]),m[i],logits[i].softmax(-1),
            4,forced=a) for a in range(3)]) for i in range(2)])
    weight = torch.tensor([[.2,-.5,.7],[-.4,.3,.8]],dtype=torch.float64)
    ga = torch.autograd.grad((full*weight).sum(),(m,logits),retain_graph=True)
    gb = torch.autograd.grad((brute*weight).sum(),(m,logits))
    for a,c in zip(ga,gb):
        torch.testing.assert_close(a,c,atol=8e-16,rtol=4e-15)
        assert bool((a.abs()>1e-8).any())


def test_empty_batch_and_all_inactive_have_no_future_branches():
    b = fixture()
    empty = exact_lookahead(b,torch.empty(0,dtype=torch.long),
        torch.empty(0,5,dtype=torch.float64),torch.empty(0,3,dtype=torch.float64))
    assert empty['q'].shape == (0,3)
    assert int(empty['costs']['branch_state_evaluations']) == 0
    b.active.zero_()
    out = exact_lookahead(b,torch.tensor([0,3]),torch.zeros(2,5,dtype=torch.float64),
                          torch.tensor([[1.,0.,0.]],dtype=torch.float64).repeat(2,1))
    assert torch.equal(out['q'],torch.zeros(2,3,dtype=torch.float64))
    assert int(out['costs']['state_transitions']) == 0
    assert int(out['costs']['enumerated_continuation_paths']) == 0


def test_invalid_actions_do_not_branch_and_zero_memory_write_is_identity():
    b = fixture(T=4,D=3)
    b.valid[:,1:] = False
    b.confidence.zero_()
    m = torch.tensor([[2.,3.,4.]],dtype=torch.float64)
    out = exact_lookahead(b,torch.tensor([0]),m,
                          torch.tensor([[1.,0.,0.]],dtype=torch.float64))
    assert torch.equal(out['q'][:,1:],torch.zeros(1,2,dtype=torch.float64))
    expected = sum(scalar_scores(b,t,m[0],torch.tensor([1.,0.,0.],dtype=torch.float64))[0]
                   for t in range(4))
    torch.testing.assert_close(out['q'][0,0],expected,atol=4e-16,rtol=0)
    assert torch.equal(out['costs']['continuation_paths'],torch.tensor([[1,0,0]]))
    assert torch.equal(out['costs']['branch_states_by_depth'],torch.ones(4,dtype=torch.long))


@pytest.mark.parametrize('what',['emptybank','noE','outofrange','negativeprob','badH','nan'])
def test_invalid_inputs_fail_closed(what):
    b = fixture(T=2,D=3)
    nodes = torch.tensor([0])
    m = torch.ones(1,3,dtype=torch.float64)
    p = torch.tensor([[1.,0.,0.]],dtype=torch.float64)
    h = 4
    if what == 'emptybank': b.features = b.features[:0]
    elif what == 'noE': b.valid[0,0] = False
    elif what == 'outofrange': nodes[0] = 2
    elif what == 'negativeprob': p[0] = torch.tensor([1.1,-.1,0.])
    elif what == 'badH': h = 5
    elif what == 'nan': m[0,0] = float('nan')
    with pytest.raises(ValueError):
        exact_lookahead(b,nodes,m,p,horizon=h)
