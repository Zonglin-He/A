"""Pure CPU synthetic checks for the frozen v0 policy contract."""
import copy
import itertools

import pytest
import torch

from scripts.propagation_v0 import policy


CFG = dict(seed=41,rounds=3,paths=2,lr=.1,gradient_clip_norm=1.,teacher_tau=.2,
    prior_weight=.05,original_prior=.7,memory_write_max=.2,
    feedback_weights=dict(support=1.,link=.25,memory=.1,guard=.1),
    initial_weights=[0.]*6,Direct_relaxation_temperature=1.)


class FakeBank:
    def __init__(self,T=5):
        basis = torch.eye(256,dtype=torch.float64)
        self.features = torch.stack((basis[0],basis[1],basis[2]))[None].repeat(T,1,1)
        self.E = torch.tensor([.4,.5,.2,.25],dtype=torch.float64)[None].repeat(T,1)
        self.boxes = self.E[:,None].repeat(1,3,1)
        self.boxes[:,1,0] = .6
        self.boxes[:,2,1] = .7
        self.valid = torch.ones(T,3,dtype=torch.bool)
        self.active = torch.ones(T,dtype=torch.bool)
        self.observed = torch.zeros(T,dtype=torch.bool)
        if T>3:
            self.active[3] = False
            self.valid[3,1:] = False
            self.observed[2:4] = True
        self.anchor_features = basis[2][None].repeat(T,1)
        self.initial_memory = basis[0].clone()
        self.u = torch.tensor([.2,.9,.5],dtype=torch.float64)[None].repeat(T,1)
        self.projection = self.u
        self.confidence = torch.ones(T,dtype=torch.float64)*.8
        self.guard = torch.tensor([0.,.1,.2],dtype=torch.float64)[None].repeat(T,1)
        self.link = .7*torch.eye(3,dtype=torch.float64)[None].repeat(max(T-1,0),1,1)
        self.score_unary = torch.zeros(T,3,dtype=torch.float64)
        self.score_pair = torch.zeros(max(T-1,0),3,3,dtype=torch.float64)

    def score_path(self,actions):
        unary = self.score_unary[torch.arange(len(actions)),actions].sum()
        pair = self.score_pair[torch.arange(len(actions)-1),actions[:-1],actions[1:]].sum()
        return unary+pair


def weights(value=None):
    return torch.nn.Parameter(torch.zeros(6,dtype=torch.float64) if value is None
                              else torch.tensor(value,dtype=torch.float64))


def test_zero_initialization_strictly_preserves_E_and_exact_prior_support():
    bank = FakeBank()
    bank.valid[1,2] = False
    out = policy.rollout(bank,weights(),CFG)
    assert torch.equal(out['boxes'],bank.E)
    assert torch.equal(out['actions'],torch.zeros(5,dtype=torch.long))
    torch.testing.assert_close(out['priorq'][0],torch.tensor([.7,.15,.15],dtype=torch.float64),atol=0,rtol=0)
    torch.testing.assert_close(out['priorq'][1],torch.tensor([.7,.3,0.],dtype=torch.float64),atol=0,rtol=0)
    assert torch.equal(out['priorq'][3],torch.tensor([1.,0.,0.],dtype=torch.float64))
    torch.testing.assert_close(out['studentq'],out['priorq'],atol=2e-16,rtol=0)


def test_student_features_exclude_unary_but_teacher_uses_it_without_student_prior():
    bank = FakeBank()
    altered = copy.deepcopy(bank)
    altered.u[0] = torch.tensor([.95,.05,.01],dtype=torch.float64)
    phi = weights([.2,.1,-.1,.3,.05,-.2])
    a = policy.rollout(bank,phi,CFG)
    b = policy.rollout(altered,phi,CFG)
    assert policy.FEATURE_NAMES == ('memory_cosine','previous_selected_feature_cosine',
        'previous_candidate_link','E_feature_cosine','negative_guard','is_E')
    assert torch.equal(a['features'][0],b['features'][0])
    assert torch.equal(a['studentq'][0],b['studentq'][0])
    assert not torch.equal(a['teacherq'][0],b['teacherq'][0])
    expected = torch.softmax(a['r'][0]/.2,dim=-1)
    torch.testing.assert_close(a['teacherq'][0],expected,atol=2e-16,rtol=0)
    assert not torch.allclose(a['teacherq'][0],torch.softmax(a['r'][0]/.2+a['prior_log_probs'][0],dim=-1))


def test_memory_is_confidence_unary_gated_and_observations_never_force_reanchor():
    bank = FakeBank()
    noise = torch.zeros(5,3,dtype=torch.float64)
    noise[:,1] = 30
    out = policy.rollout(bank,weights(),CFG,'student_sample',noise=noise)
    expected_omega = .2*bank.confidence*out['selected_u']*bank.active
    torch.testing.assert_close(out['omega'],expected_omega,atol=0,rtol=0)
    assert not torch.equal(out['memory_after'][2],bank.anchor_features[2])
    assert torch.equal(out['memory_before'][3],out['memory_after'][3])
    no_confidence = copy.deepcopy(bank)
    no_confidence.confidence.zero_()
    unchanged = policy.rollout(no_confidence,weights(),CFG,'student_sample',noise=noise)
    assert torch.equal(unchanged['memory_before'],bank.initial_memory[None].repeat(5,1))
    assert torch.equal(unchanged['memory_after'],unchanged['memory_before'])
    no_unary = copy.deepcopy(bank)
    no_unary.u.zero_()
    unchanged = policy.rollout(no_unary,weights(),CFG,'student_sample',noise=noise)
    assert torch.equal(unchanged['memory_after'],unchanged['memory_before'])


def test_no_memory_teacher_control_has_zero_memory_and_no_write():
    bank = FakeBank()
    out = policy.rollout(bank,weights(),CFG,'no_memory_teacher_greedy')
    assert torch.equal(out['memory_before'],torch.zeros(5,256,dtype=torch.float64))
    assert torch.equal(out['memory_after'],out['memory_before'])
    assert torch.equal(out['omega'],torch.zeros(5,dtype=torch.float64))
    previous_link = torch.zeros(5,3,dtype=torch.float64)
    previous_link[1:] = torch.einsum('ti,tij->tj',out['previous_action_probs'][1:],bank.link)
    torch.testing.assert_close(out['r'],bank.u+.25*previous_link-.1*bank.guard,atol=0,rtol=0)


def test_teacher_and_student_gradients_are_finite_with_invalid_support():
    bank = FakeBank()
    bank.valid[:,2] = False
    phi = weights([.03,-.02,.01,.02,.01,-.01])
    noise = policy.gumbel_noise('gradient',0,0,5)
    out = policy.rollout(bank,phi,CFG,'direct_soft',noise=noise)
    assert not out['teacher_target_detached']
    assert torch.equal(out['action_probs'][:,2],torch.zeros(5,dtype=torch.float64))
    assert bool(torch.isfinite(out['loss']))
    assert torch.autograd.gradcheck(lambda p: policy.rollout(
        bank,p,CFG,'direct_soft',noise=noise)['loss'],(phi,),eps=1e-6,atol=2e-6,rtol=2e-5)
    memory_gradient = torch.autograd.grad(out['memory_after'][0,1],phi,retain_graph=True)[0]
    target_gradient = torch.autograd.grad(out['teacher_log_probs'][1,1],phi)[0]
    assert memory_gradient.norm()>1e-8 and target_gradient.norm()>1e-8


def test_two_path_loss_has_one_confidence_denominator_epsilon_and_all_node_prior_mean():
    bank = FakeBank()
    bank.confidence.fill_(1e-13)
    phi = weights([.2,-.1,.15,.03,-.05,.1])
    paths = [policy.rollout(bank,phi,CFG,'student_sample',noise=policy.gumbel_noise('KL',0,p,5))
             for p in range(2)]
    loss,feedback,prior = policy._batch_loss(paths,phi,policy._settings(CFG))
    c = torch.stack([p['confidence'] for p in paths])
    KL = torch.stack([p['KL_terms'] for p in paths])
    pKL = torch.stack([p['prior_KL_terms'] for p in paths])
    expected_feedback = (c*KL).sum()/(c.sum()+1e-12)
    torch.testing.assert_close(feedback,expected_feedback,atol=0,rtol=0)
    torch.testing.assert_close(prior,pKL.mean(),atol=0,rtol=0)
    torch.testing.assert_close(loss,expected_feedback+.05*pKL.mean(),atol=0,rtol=0)
    assert not torch.allclose(feedback,torch.stack([p['feedback_loss'] for p in paths]).mean())


@pytest.mark.parametrize('arm',policy.ARMS)
def test_equal_score_accepts_three_actual_updates_and_final_is_last_greedy(arm):
    bank = FakeBank()
    result = policy.fit(bank,'fit',arm,CFG)
    assert len(result['rounds'])==3
    assert result['costs']['updates']==result['costs']['backward']==3
    assert result['costs']['accepted_updates']==3
    assert result['costs']['heavy_forwards']==result['costs']['decoder_calls']==0
    assert not torch.equal(result['finalphi'],result['initialphi'])
    assert torch.equal(result['final'],policy.rollout(bank,result['finalphi'],CFG)['boxes'])
    assert len(result['trace']['all_rollouts'])==result['costs']['total_sequence_rollouts']
    assert result['costs']['fixed_score_evaluations']==4
    for step in result['rounds']:
        assert step['accepted'] and step['proposal_score']==step['before_score']==step['after_score']==0
        assert step['gradient'].norm()<=1.+1e-12
    if arm=='FixedKD':
        assert result['costs']['teacher_setup_rollouts']==2
        assert result['costs']['cached_teacher_state_loss_passes']==6
        for paths in result['trace']['training_paths']:
            for path,fixed in zip(paths,result['trace']['fixed_teacher_paths']):
                for name in ('features','teacher_log_probs','memory_before','actions'):
                    assert torch.equal(path[name],fixed[name])


@pytest.mark.parametrize('arm',policy.ARMS)
def test_score_drop_restores_nonempty_Adam_history_and_stops_without_backtracking(arm):
    bank = FakeBank(T=1)
    A = bank.features[0,0].clone()
    bank.features[0,1] = -A
    bank.initial_memory = -A
    bank.valid[0,2] = False
    bank.u[0] = torch.tensor([0.,1.,0.],dtype=torch.float64)
    bank.confidence.fill_(1.)
    bank.guard.zero_()
    bank.score_unary[0] = torch.tensor([0.,-1.,-2.],dtype=torch.float64)
    result = policy.fit(bank,'rollback',arm,CFG)
    assert len(result['rounds'])==2
    first,last = result['rounds']
    assert first['accepted'] and not last['accepted'] and last['rolled_back']
    assert last['proposal_score']<last['before_score']-1e-12
    assert last['optimizer_before_sha256']==first['optimizer_after_sha256']
    assert last['optimizer_after_sha256']==last['optimizer_before_sha256']
    assert last['optimizer_proposal_sha256']!=last['optimizer_after_sha256']
    assert torch.equal(result['finalphi'],first['after'])
    assert torch.equal(result['final'],bank.E)
    assert result['costs']['accepted_updates']==result['costs']['rejected_updates']==1
    assert result['costs']['postupdate_student_greedy_rollouts']==2


def test_exact_additive_evidence_DP_matches_exhaustive_valid_paths():
    bank = FakeBank(T=4)
    generator = torch.Generator().manual_seed(81)
    bank.score_unary = torch.randn(4,3,dtype=torch.float64,generator=generator)
    bank.score_pair = torch.randn(3,3,3,dtype=torch.float64,generator=generator)
    bank.valid[1,2] = False
    possible = []
    for actions in itertools.product(range(3),repeat=4):
        if all(bank.valid[t,a] and (bank.active[t] or a==0) for t,a in enumerate(actions)):
            actions = torch.tensor(actions,dtype=torch.long)
            possible.append(float(bank.score_path(actions)))
    out = policy.evidence_dp(bank)
    assert abs(out['score']-max(possible))<1e-12
    assert out['trace']['actions'][3]==0
    assert out['costs']['total_sequence_rollouts']==0


def test_initial_diagnostics_are_two_fixed_phi0_paths_and_have_deterministic_noise():
    bank = FakeBank()
    before = torch.random.get_rng_state().clone()
    diagnostic = policy.initial_diagnostics(bank,'diagnostic',CFG)
    assert torch.equal(before,torch.random.get_rng_state())
    assert len(diagnostic['diagnostic_paths'])==2
    assert torch.equal(diagnostic['initial_final'],bank.E)
    for index,path in enumerate(diagnostic['diagnostic_paths']):
        assert torch.equal(path['noise'],policy.gumbel_noise('diagnostic',0,index,5))
        assert path['teacher_scores'].shape==path['teacherq'].shape==(5,3)
    opd = policy.fit(bank,'diagnostic','OPD',CFG)
    direct = policy.fit(bank,'diagnostic','Direct',CFG)
    for index,path in enumerate(opd['trace']['training_paths'][0]):
        assert torch.equal(path['actions'],diagnostic['diagnostic_paths'][index]['actions'])
        assert torch.equal(path['noise'],direct['trace']['training_paths'][0][index]['noise'])


@pytest.mark.parametrize('arm',policy.ARMS)
def test_no_active_evidence_keeps_E_memory_and_weights(arm):
    bank = FakeBank()
    bank.active.zero_()
    bank.valid[:,1:] = False
    result = policy.fit(bank,'inactive',arm,CFG)
    assert torch.equal(result['final'],bank.E)
    assert torch.equal(result['finalphi'],result['initialphi'])
    for path in result['trace']['all_rollouts']:
        assert torch.equal(path['memory_before'],path['memory_after'])
        assert torch.equal(path['boxes'],bank.E)


def test_settings_use_canonical_keys_and_reject_conflicting_aliases():
    assert policy._settings(CFG)['memory_rate']==CFG['memory_write_max']
    with pytest.raises(ValueError):
        policy._settings({**CFG,'memory_rate':.1})
    with pytest.raises(ValueError):
        policy._settings({**CFG,'feedback_weights':dict(support=1.,link=.1,memory=.1,guard=.1)})
