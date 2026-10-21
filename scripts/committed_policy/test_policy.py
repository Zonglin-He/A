"""Synthetic CPU tests only: committed routing, exact Q and matched updates."""
import copy
import itertools
from collections import defaultdict

import pytest
import torch

from scripts.committed_policy import policy
from scripts.propagation_v0.test_policy import FakeBank

torch.set_num_threads(1)
CFG=dict(seed=41,rounds=3,paths=2,inner_steps=10,lr=.1,teacher_tau=.2,
    original_prior=.7,prior_weight=.05,gradient_clip_norm=1.,horizon=4,route_strength=2.,initial_raw=[0.]*4)


def synthetic(T=6):
    bank=FakeBank(T);bank.active.fill_(True);bank.valid.fill_(True)
    bank.confidence.fill_(.8);bank.guard[:,0]=0
    du=torch.tensor([0.,.12,.06],dtype=torch.float64)[None].repeat(T,1)
    slots=torch.tensor([0,1,2],dtype=torch.long)[None].repeat(T,1)
    commit=torch.tensor([False,True,True])[None].repeat(T,1)
    compatibility=torch.zeros(T,9,3,dtype=torch.float64)
    compatibility[:,1,1]=1.;compatibility[:,2,2]=1.
    extra=dict(delta_support=du,external_support=du+.2,candidate_trust=torch.full((T,3),.3,dtype=torch.float64),
        slot_routes=slots,commit=commit,route_compat=compatibility,reset_matrices=torch.eye(9,dtype=torch.float64)[None].repeat(T,1,1))
    return bank,extra


def raw():return torch.nn.Parameter(torch.zeros(4,dtype=torch.float64))


def test_positive_weights_initialization_and_no_learned_prior_escape_bias():
    bank,extra=synthetic();theta=raw();path=policy.rollout(bank,extra,theta,CFG)
    torch.testing.assert_close(policy.positive_weights(theta),torch.ones(4,dtype=torch.float64),atol=2e-16,rtol=0)
    assert path['student_features'].shape==(6,3,4)
    expected=path['prior_log_probs']+path['student_features']@policy.positive_weights(theta)+path['route_bias']
    torch.testing.assert_close(path['student_log_probs'],expected.log_softmax(-1),atol=0,rtol=0)
    assert path['state_semantics']=='nine source provenance states; no appearance EMA'
    for values in (-100.,100.):
        w=policy.positive_weights(torch.full((4,),values,dtype=torch.float64))
        assert bool((w>=0).all()) and bool((w<=4).all())


def test_commit_influences_more_than_three_future_nodes_and_survives_E_abstention():
    bank,extra=synthetic(T=9);noise=torch.zeros(9,3,dtype=torch.float64)
    noise[0,1]=100.;noise[2,0]=100.
    path=policy.rollout(bank,extra,raw(),CFG,'student_sample',noise=noise)
    assert path['actions'][0]==1 and path['actions'][2]==0 and path['actions'][8]==1
    assert path['committed_route'][2]==1
    assert path['route_before'][8,1]==1 and path['route_bias'][8,1]==2
    other=copy.deepcopy(extra);other['commit'].zero_()
    no_commit=policy.rollout(bank,other,raw(),CFG,'student_sample',noise=noise)
    assert no_commit['actions'][8]==0
    assert no_commit['route_after'][:,0].eq(1).all()


def test_external_counterevidence_resets_route_and_low_evidence_cannot_commit():
    bank,extra=synthetic(T=9);noise=torch.zeros(9,3,dtype=torch.float64);noise[0,1]=100.
    extra['reset_matrices'][4,1]=0.;extra['reset_matrices'][4,1,0]=1.
    path=policy.rollout(bank,extra,raw(),CFG,'student_sample',noise=noise)
    assert path['reset_event'][4] and path['reset_mass'][4]==1
    assert path['route_after_reset'][4,0]==1 and path['actions'][4:].eq(0).all()
    extra['commit'][0]=False
    low=policy.rollout(bank,extra,raw(),CFG,'student_sample',noise=noise)
    assert not low['commit_event'][0] and low['route_after'][:,0].eq(1).all()


def test_hard_and_probability_mixture_use_exactly_the_same_transition_and_no_detach():
    bank,extra=synthetic(T=2);b=policy._prepare(bank,extra)
    incoming=torch.eye(9,dtype=torch.float64)[3,None]
    nodes=torch.tensor([0]);hard=torch.tensor([[0.,1.,0.]],dtype=torch.float64)
    h,m=policy._advance(b,nodes,incoming,hard)
    assert torch.equal(h,torch.eye(9,dtype=torch.float64)[1,None]) and m==1
    p=torch.tensor([[.3,.4,.3]],dtype=torch.float64,requires_grad=True)
    mixed,mass=policy._advance(b,nodes,incoming,p)
    manual=sum(p[0,k]*policy._advance(b,nodes,incoming,torch.eye(3,dtype=torch.float64)[k,None])[0][0]
               for k in range(3))
    torch.testing.assert_close(mixed[0],manual,atol=1e-16,rtol=0)
    grad=torch.autograd.grad(mixed[0,1],p)[0]
    assert grad[0,1]!=0 and mass.requires_grad


def test_rho_has_no_route_self_reward_and_sums_exactly_to_independent_utility():
    bank,extra=synthetic(T=5);bank.active[2]=False
    extra['commit']&=bank.active[:,None]&bank.valid
    # Fixed random-looking edges distinguish EE from same-history E baselines.
    bank.link=torch.arange(4*9,dtype=torch.float64).reshape(4,3,3)/37
    b=policy._prepare(bank,extra);theta=raw();n=torch.tensor([1]);previous=torch.tensor([[.3,.4,.3]],dtype=torch.float64)
    a=policy._node(b,theta,n,torch.eye(9,dtype=torch.float64)[0,None],previous)
    c=policy._node(b,theta,n,torch.eye(9,dtype=torch.float64)[1,None],previous)
    assert torch.equal(a['rho'],c['rho']) and not torch.equal(a['logp'],c['logp'])
    noise=policy.gumbel_noise('utility',0,0,5)
    path=policy.rollout(bank,extra,theta,CFG,'student_sample',noise=noise)
    chosen=path['rho'][torch.arange(5),path['actions']].sum()
    torch.testing.assert_close(chosen,policy.utility(bank,extra,path['actions']),atol=1e-15,rtol=0)
    assert path['actions'][2]==0 and torch.equal(path['route_before'][2],path['route_after'][2])


def recursive_Q(b,theta,t,h,prev,horizon,forced):
    node=torch.tensor([t]);info=policy._node(b,theta,node,h[None],prev[None])
    rho=info['rho'][0,forced]
    if horizon==1 or t+1==b['T']:return rho
    action=torch.eye(3,dtype=torch.float64)[forced]
    after,_=policy._advance(b,node,info['reset'],action[None])
    next_info=policy._node(b,theta,torch.tensor([t+1]),after,action[None])
    p=next_info['logp'][0].exp();total=rho
    for k in range(3):
        if bool(next_info['valid'][0,k]):
            total=total+p[k]*recursive_Q(b,theta,t+1,after[0],action,horizon-1,k)
    return total


def test_batched_H4_matches_discrete_recursion_and_H1_matches_rho():
    bank,extra=synthetic(T=5);bank.active[2]=False;bank.valid[3,2]=False
    extra['commit']&=bank.active[:,None]&bank.valid
    b=policy._prepare(bank,extra);theta=torch.tensor([.05,-.1,.03,.02],dtype=torch.float64,requires_grad=True)
    nodes=torch.tensor([0,1,2,3,4]);h=torch.eye(9,dtype=torch.float64)[torch.tensor([0,1,1,2,0])]
    previous=torch.tensor([[1.,0.,0.],[.2,.5,.3],[0.,1.,0.],[1.,0.,0.],[.2,.3,.5]],dtype=torch.float64)
    q=policy.exact_q(b,theta,nodes,h,previous,4)
    for i,t in enumerate(nodes):
        for k in range(3):
            if b['active'][t] and b['effective_valid'][t,k]:
                expected=recursive_Q(b,theta,int(t),h[i],previous[i],4,k)
                torch.testing.assert_close(q[i,k],expected,atol=5e-16,rtol=0)
            else:assert q[i,k]==0
    short=policy.exact_q(b,theta,nodes,h,previous,1)
    info=policy._node(b,theta,nodes,h,previous)
    expected=torch.where(b['active'][:,None]&info['valid'],info['rho'],torch.zeros_like(info['rho']))
    torch.testing.assert_close(short,expected,atol=0,rtol=0)


def test_Direct_full_gradient_includes_route_history_Q_branch_policy_and_target():
    bank,extra=synthetic(T=4);bank.valid[2,2]=False
    extra['commit']&=bank.active[:,None]&bank.valid
    theta=torch.tensor([.05,-.1,.03,.02],dtype=torch.float64,requires_grad=True)
    noise=policy.gumbel_noise('full gradient',0,0,4)
    def evaluate(p):
        path=policy.rollout(bank,extra,p,CFG,'direct_soft',noise=noise,collect_only=False)
        return policy._loss([path],p,policy._settings(CFG),False)[0]
    assert torch.autograd.gradcheck(evaluate,(theta,),eps=1e-6,atol=2e-6,rtol=2e-5)
    path=policy.rollout(bank,extra,theta,CFG,'direct_soft',noise=noise,collect_only=False)
    route_grad=torch.autograd.grad(path['route_before'][2,1],theta,retain_graph=True)[0]
    target_grad=torch.autograd.grad(path['teacher_log_probs'][1,1],theta)[0]
    assert route_grad.norm()>1e-8 and target_grad.norm()>1e-8
    assert not path['teacher_target_detached']


def test_prior_in_teacher_and_no_route_self_reward_match_when_rho_zero():
    bank,extra=synthetic(T=3);bank.guard.zero_();bank.link.zero_();extra['delta_support'].zero_();extra['commit'].zero_()
    theta=raw();path=policy.rollout(bank,extra,theta,CFG,collect_only=False)
    torch.testing.assert_close(path['teacherq'],path['prior_log_probs'].exp(),atol=2e-16,rtol=0)


def test_exact_DP_matches_exhaustive_paths_with_inactive_edges_and_invalid_actions():
    bank,extra=synthetic(T=5);bank.active[2]=False;bank.valid[3,2]=False
    extra['commit']&=bank.active[:,None]&bank.valid
    bank.link=torch.arange(4*9,dtype=torch.float64).reshape(4,3,3)/37
    maximum=-float('inf')
    for values in itertools.product(range(3),repeat=5):
        if all(bank.valid[t,k] and (bank.active[t]or k==0)for t,k in enumerate(values)):
            maximum=max(maximum,float(policy.utility(bank,extra,torch.tensor(values,dtype=torch.long))))
    result=policy.fixed_dp(bank,extra,CFG)
    assert abs(result['score']-maximum)<1e-15 and result['actions'][2]==0


@pytest.mark.parametrize('arm',policy.ARMS)
def test_three_outers_thirty_updates_fixedKD_refresh_targets_and_matched_budget(monkeypatch,arm):
    bank,extra=synthetic(T=4)
    monkeypatch.setattr(policy,'utility',lambda *a,**kw:torch.tensor(0.,dtype=torch.float64))
    before_rng=torch.random.get_rng_state().clone()
    out=policy.fit(bank,extra,'actual budget',arm,CFG)
    assert torch.equal(before_rng,torch.random.get_rng_state())
    c=out['costs'];assert c['updates']==c['backward']==c['state_only_backwards']==30
    assert c['outer_iterations']==c['accepted_outer']==3 and c['rejected_outer']==0
    assert len(out['rounds'])==3 and all(len(r['inner'])==10 for r in out['rounds'])
    assert c['heavy_forwards']==c['decoder_calls']==c['encoder_calls']==c['expert_calls']==0
    assert c['exclusive_residual_nanoseconds']==0
    assert len(out['trace']['all_rollouts'])==c['policy_rollouts']==c['total_sequence_rollouts']
    final=out['trace']['final_rollout'];assert final['teacherq'] is not None and final['Q'] is not None
    if arm=='OPD':assert c['q_calls']==4 and c['q_input_states']==7*4 and c['policy_rollouts']==10
    elif arm=='FixedKD':
        assert c['q_calls']==2*4+4 and c['q_input_states']==9*4 and c['policy_rollouts']==6
        fixed=out['trace']['fixed_teacher_paths']
        for outer in out['trace']['outer_histories']:
            for a,b in zip(outer,fixed):
                assert torch.equal(a['actions'],b['actions']) and torch.equal(a['route_before'],b['route_before'])
        assert not torch.equal(out['trace']['outer_histories'][0][0]['Q'],out['trace']['outer_histories'][2][0]['Q'])
    else:assert c['q_calls']==31 and c['q_input_states']==61*4 and c['policy_rollouts']==64
    for outer in range(3):
        for inner in range(10):
            for p,path in enumerate(out['trace']['training_paths'][outer][inner]):
                expected=policy.gumbel_noise('actual budget',0 if arm=='FixedKD'else outer,p,4)
                assert torch.equal(path['noise'],expected)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('arm',policy.ARMS)
def test_full_Adam_outer_rollback_does_not_cancel_later_refresh(monkeypatch,arm):
    bank,extra=synthetic(T=3);schedule=iter([0.,0.,-1.,0.])
    monkeypatch.setattr(policy,'utility',lambda *a,**kw:torch.tensor(next(schedule),dtype=torch.float64))
    out=policy.fit(bank,extra,'outer rollback',arm,CFG);first,middle,last=out['rounds']
    assert first['accepted'] and middle['rolled_back'] and last['accepted']
    assert middle['optimizer_before_sha256']==first['optimizer_after_sha256']
    assert middle['optimizer_after_sha256']==middle['optimizer_before_sha256']
    assert middle['optimizer_proposal_sha256']!=middle['optimizer_after_sha256']
    assert torch.equal(middle['before'],middle['after']) and torch.equal(last['before'],middle['before'])
    assert out['costs']['updates']==30 and out['costs']['accepted_outer']==2 and out['costs']['rejected_outer']==1
    assert len(out['trace']['outer_histories'])==3
