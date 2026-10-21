"""No real assets/GT: causal intervention and common-randomness invariants."""
import copy
import numpy as np
import pytest
import torch
from scripts.correction_value.core import (_prepare,continue_student,intervene,select_points,
    random_alternative,query_noise,verify_branch)
from scripts.evidence_policy.policy import rollout
from scripts.evidence_policy.test_policy import CFG,FakeBank

torch.set_num_threads(1)

def phi():return torch.tensor([.7,.3,.4,.1,.2,-.6,.8,.4],dtype=torch.float64)


def test_continuation_exactly_matches_original_policy_at_nonzero_frozen_state():
    bank=FakeBank(T=8);b=_prepare(bank);noise=query_noise('test',8)
    a=rollout(bank,phi(),CFG,'student_sample',noise=noise,variant='B',collect_only=True)
    c=continue_student(b,phi(),CFG,noise)
    for k in ('features','student_log_probs','effective_valid','memory_before','memory_after','previous_action_probs','previous_features','actions','boxes','omega'):
        assert torch.equal(a[k],c[k]),k


def test_single_action_intervention_has_identical_prefix_but_causal_memory_changes():
    bank=FakeBank(T=8);b=_prepare(bank);noise=query_noise('test',8);base=continue_student(b,phi(),CFG,noise)
    t=1;a=(int(base['actions'][t])+1)%3
    c=continue_student(b,phi(),CFG,noise,t,a)
    proof=verify_branch(base,c,dict(node=t),a)
    assert proof['prefix_and_incoming_state_identical']
    assert not torch.equal(c['memory_after'][t],base['memory_after'][t])
    assert not torch.equal(c['memory_before'][t+1],base['memory_before'][t+1])
    assert torch.equal(c['noise'][t+1:],base['noise'][t+1:])


def test_forcing_original_action_reproduces_every_future_output():
    bank=FakeBank(T=8);b=_prepare(bank);noise=query_noise('test',8);base=continue_student(b,phi(),CFG,noise)
    for t in (0,2,7):
        c=continue_student(b,phi(),CFG,noise,t,int(base['actions'][t]))
        for k in ('actions','boxes','memory_after','features'):assert torch.equal(c[k],base[k])


def test_inactive_does_not_write_memory_and_illegal_force_rejected():
    bank=FakeBank(T=8);b=_prepare(bank);noise=query_noise('test',8)
    c=continue_student(b,phi(),CFG,noise)
    assert c['actions'][3]==0 and torch.equal(c['memory_before'][3],c['memory_after'][3])
    with pytest.raises(AssertionError):continue_student(b,phi(),CFG,noise,3,1)


def sample_for_selection():
    T=17
    return dict(active=np.ones(T,bool),event=np.ones(T,bool),observed=np.zeros(T,bool),
        effective_valid=np.ones((T,3),bool),actions=np.zeros(T,int),
        teacher_scores=np.tile([0.,1.,.5],(T,1)),student_log_probs=np.tile([.5,.2,.1],(T,1)))


def test_quartile_selection_uses_physical_time_not_rewards_and_excludes_last_node():
    path=sample_for_selection();frames=np.arange(17)*10
    points,candidates=select_points(path,frames)
    assert [p['node']for p in points]==[2,6,10,14] and 16 not in candidates
    path['teacher_scores']*=500
    assert select_points(path,frames)[0]==points
    assert all(p['disagreement_type']=='policy_disagreement'for p in points)


def test_quartile_boundaries_use_fixed_event_even_if_confidence_makes_edge_inactive():
    path=sample_for_selection();path['active'][0:2]=False;path['active'][15:]=False
    points,_=select_points(path,np.arange(17)*10)
    assert points[0]['node']==2 and points[0]['time_bin']==0
    assert all(p['node']<14 for p in points) # later active unobserved required


def test_noise_only_category_and_two_action_random_is_teacher():
    path=sample_for_selection();path['student_log_probs'][:]=[.1,.5,.2]
    points,_=select_points(path,np.arange(17))
    assert all(p['disagreement_type']=='noise_only_disagreement'for p in points)
    valid=torch.tensor([True,True,False])
    assert all(random_alternative('x',n,valid,0)==1 for n in range(20))


def test_random_nonstudent_support_and_reproducibility():
    valid=torch.tensor([True,True,True]);a=[random_alternative('x',n,valid,0)for n in range(100)]
    assert set(a)=={1,2}and a==[random_alternative('x',n,valid,0)for n in range(100)]
    assert torch.equal(query_noise('key',8),query_noise('key',8))
    assert not torch.equal(query_noise('key',8),query_noise('other',8))


def test_no_disagreement_or_no_later_unobserved_selects_nothing():
    path=sample_for_selection();path['actions'][:]=1
    assert select_points(path,np.arange(17))[0]==[]
    path=sample_for_selection();path['observed'][:]=True
    assert select_points(path,np.arange(17))[0]==[]


def test_integrated_probe_uses_frozen_phi_and_independent_single_forces():
    bank=FakeBank(T=12);before=phi();out=intervene(bank,before,CFG,'integration',np.arange(12))
    assert out['parameter_updates']==0 and out['GT_read']is False and 0<=len(out['branches'])<=4
    assert torch.equal(before,phi())
    for entry in out['branches']:
        p=entry['point']
        for name,path in entry['paths'].items():
            assert path['forced_node']==p['node'] and torch.equal(path['phi'],before)
        if p['random_action']==p['teacher_action']:
            assert torch.equal(entry['paths']['Random']['boxes'],entry['paths']['Teacher']['boxes'])
