"""CPU synthetic invariants for the locked A/B eight-weight policy."""
import copy

import pytest
import torch

from scripts.evidence_policy import policy
from scripts.propagation_v0.test_policy import FakeBank, CFG as OLD_CFG

torch.set_num_threads(1)
CFG={**OLD_CFG,'inner_steps':10,'initial_weights':[0.]*8}


def weights(values=None):
    return torch.nn.Parameter(torch.zeros(8,dtype=torch.float64) if values is None
                              else torch.tensor(values,dtype=torch.float64))


def test_shared_zero_initialization_features_and_H1_exact_instantaneous_feedback():
    bank=FakeBank();noise=policy.gumbel_noise('shared',0,0,5);phi=weights()
    A=policy.rollout(bank,phi,CFG,'student_sample',noise=noise,variant='A')
    B=policy.rollout(bank,phi,CFG,'student_sample',noise=noise,variant='B')
    assert torch.equal(A['actions'],B['actions'])
    assert torch.equal(A['memory_before'],B['memory_before'])
    assert torch.equal(A['features'],B['features'])
    delta=bank.u-bank.u[:,0,None]
    torch.testing.assert_close(A['features'][...,6],delta,atol=0,rtol=0)
    torch.testing.assert_close(A['features'][...,7],bank.confidence[:,None]*bank.active[:,None]*delta,atol=0,rtol=0)
    torch.testing.assert_close(A['Q'],A['r'],atol=2e-16,rtol=0)
    assert not torch.equal(A['Q'],B['Q'])
    for variant in ('A','B'):
        out=policy.rollout(bank,phi,CFG,variant=variant,collect_only=True)
        assert torch.equal(out['boxes'],bank.E)
        torch.testing.assert_close(out['studentq'],out['priorq'],atol=2e-16,rtol=0)
        assert out['Q'] is None


@pytest.mark.parametrize('variant',['A','B'])
def test_OPD_acquires_two_hard_histories_only_once_per_outer_and_ten_fixed_inner_updates(variant):
    out=policy.fit(FakeBank(T=3),'OPD acquisition','OPD',CFG,variant)
    assert len(out['rounds'])==30
    assert out['costs']['q_calls']==3
    assert out['costs']['q_input_states']==18
    assert out['costs']['outer_student_acquisition_rollouts']==6
    assert out['costs']['cached_state_loss_passes']==60
    assert len(out['trace']['all_rollouts'])==out['costs']['total_sequence_rollouts']==37
    for outer in range(3):
        cached=out['trace']['outer_acquisitions'][outer]
        for inner in range(10):
            paths=out['trace']['training_paths'][outer*10+inner]
            for path,original in zip(paths,cached):
                for name in ('features','teacherq','memory_before','actions','noise'):
                    assert torch.equal(path[name],original[name])
                    assert path[name] is original[name]
                assert path['teacher_target_detached']
        assert out['rounds'][outer*10]['outer']==outer
    assert not torch.equal(out['finaltheta'],out['initialtheta'])
    assert out['costs']['heavy_forwards']==out['costs']['decoder_calls']==0
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('variant',['A','B'])
def test_FixedKD_teacher_histories_are_built_once_and_reused_for_all_thirty_updates(variant):
    out=policy.fit(FakeBank(T=3),'KD fixed','FixedKD',CFG,variant)
    assert len(out['rounds'])==30
    assert out['costs']['teacher_setup_rollouts']==2
    assert out['costs']['q_calls']==6
    assert out['costs']['outer_student_acquisition_rollouts']==0
    assert out['costs']['total_sequence_rollouts']==33
    for paths in out['trace']['training_paths']:
        for path,original in zip(paths,out['trace']['fixed_teacher_paths']):
            for name in ('features','teacherq','memory_before','actions','noise'):
                assert torch.equal(path[name],original[name])
    assert out['costs']['teacher_setup_seconds']>=0


@pytest.mark.parametrize('variant',['A','B'])
def test_Direct_recomputes_every_inner_with_same_outer_noise_and_full_target_gradient(variant):
    bank=FakeBank(T=3)
    out=policy.fit(bank,'Direct recompute','Direct',CFG,variant)
    assert len(out['rounds'])==30
    assert out['costs']['direct_soft_recomputation_rollouts']==60
    assert out['costs']['q_calls']==30
    assert out['costs']['q_input_states']==180
    assert out['costs']['total_sequence_rollouts']==91
    for outer in range(3):
        paths=out['trace']['training_paths'][outer*10:(outer+1)*10]
        for innerpaths in paths:
            for index,path in enumerate(innerpaths):
                assert torch.equal(path['noise'],policy.gumbel_noise('Direct recompute',outer,index,3))
                assert not path['teacher_target_detached']
        assert not torch.equal(paths[0][0]['features'],paths[-1][0]['features'])
    phi=weights([.03,-.02,.01,.02,.01,-.01,.04,-.02])
    noise=policy.gumbel_noise('Direct gradient',0,0,3)
    soft=policy.rollout(bank,phi,CFG,'direct_soft',noise=noise,variant=variant)
    gm=torch.autograd.grad(soft['memory_after'][0,1],phi,retain_graph=True)[0]
    gq=torch.autograd.grad(soft['teacher_log_probs'][1,1],phi)[0]
    assert gm.norm()>1e-8 and gq.norm()>1e-8


def test_Direct_H4_full_soft_history_gradient_matches_finite_differences_with_invalid_actions():
    bank=FakeBank(T=3);bank.valid[1,2]=False
    phi=weights([.03,-.02,.01,.02,.01,-.01,.04,-.02])
    noise=policy.gumbel_noise('H4 gradcheck',0,0,3)
    assert torch.autograd.gradcheck(lambda p:policy.rollout(bank,p,CFG,'direct_soft',
        noise=noise,variant='B')['loss'],(phi,),eps=1e-6,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('arm',policy.ARMS)
def test_score_drop_restores_nonempty_Adam_and_stops_whole_fit(arm):
    bank=FakeBank(T=1);a=bank.features[0,0].clone()
    bank.features[0,1]=-a;bank.initial_memory=-a;bank.valid[0,2]=False
    bank.u[0]=torch.tensor([0.,1.,0.],dtype=torch.float64)
    bank.confidence.fill_(.1);bank.guard.zero_()
    bank.score_unary[0]=torch.tensor([0.,-1.,-2.],dtype=torch.float64)
    out=policy.fit(bank,'rollback',arm,CFG,'B')
    assert 1<len(out['rounds'])<30
    previous,last=out['rounds'][-2:]
    assert previous['accepted'] and last['rolled_back'] and not last['accepted']
    assert last['optimizer_before_sha256']==previous['optimizer_after_sha256']
    assert last['optimizer_after_sha256']==last['optimizer_before_sha256']
    assert last['optimizer_proposal_sha256']!=last['optimizer_after_sha256']
    assert torch.equal(out['finaltheta'],previous['after'])
    assert torch.equal(out['final'],bank.E)
    assert out['costs']['fixed_score_evaluations']==len(out['rounds'])+1
    assert out['costs']['rejected_updates']==1


def test_one_denominator_for_two_paths_and_exact_prior_mean():
    bank=FakeBank(T=3);bank.confidence.fill_(1e-13)
    phi=weights([.2,-.1,.15,.03,-.05,.1,.05,-.02])
    paths=[policy.rollout(bank,phi,CFG,'student_sample',variant='A',
            noise=policy.gumbel_noise('normalization',0,p,3)) for p in range(2)]
    loss,feedback,prior=policy._batch_loss(paths,phi,policy._settings(CFG))
    c=torch.stack([p['confidence']for p in paths]);kl=torch.stack([p['KL_terms']for p in paths])
    pk=torch.stack([p['prior_KL_terms']for p in paths])
    torch.testing.assert_close(feedback,(c*kl).sum()/(c.sum()+1e-12),atol=0,rtol=0)
    torch.testing.assert_close(prior,pk.mean(),atol=0,rtol=0)
    torch.testing.assert_close(loss,feedback+.05*prior,atol=0,rtol=0)


def test_settings_reject_displayed_but_unused_protocol_changes():
    for key,value in [('teacher_tau',.3),('inner_steps',9),('Direct_relaxation_temperature',.5),
                      ('initial_weights',[0.]*6),('feedback_weights',dict(support=1.,link=.3,memory=.1,guard=.1))]:
        with pytest.raises(ValueError):policy._settings({**CFG,key:value})


def test_signed_alternative_margin_separates_learned_evidence_from_prior_barrier():
    bank=FakeBank(T=2);phi=weights()
    path=policy.rollout(bank,phi,CFG,collect_only=True)
    margin=policy._margin([path],phi)
    assert margin['positive_alternative_gap_nodes']==0
    assert bool((margin['best_alternative_minus_E']<0).all())
    torch.testing.assert_close(margin['best_alternative_minus_E'],
        margin['learned_alternative_minus_E']-margin['prior_E_barrier'],atol=0,rtol=0)
    assert torch.equal(margin['learned_alternative_minus_E'],torch.zeros(2,dtype=torch.float64))
