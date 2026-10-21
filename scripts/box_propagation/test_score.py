"""Pure saved-tensor history-comparison checks, without assets or labels."""
import copy
import numpy as np
import torch

from scripts.box_propagation.score import _probabilities,_JS,saved_path_comparison,_eligible_valid


def _output(theta):
    T=4
    feature=torch.zeros(T,3,6,dtype=torch.float64);feature[:,1,0]=1;feature[:,2,1]=1
    prior=torch.tensor([.7,.15,.15],dtype=torch.float64).log().repeat(T,1)
    valid=torch.ones(T,3,dtype=torch.bool)
    phi=torch.as_tensor(theta,dtype=torch.float64)
    q=torch.tensor([[.3,.4,.3]],dtype=torch.float64).repeat(T,1)
    path=dict(features=feature,prior_log_probs=prior,effective_valid=valid,
        studentq=torch.from_numpy(_probabilities(feature,prior,valid,phi)),teacherq=q,
        actions=torch.zeros(T,dtype=torch.long),active=torch.ones(T,dtype=torch.bool),
        observed=torch.tensor([True,False,False,True]),memory_before=torch.zeros(T,2,dtype=torch.float64),
        previous_action_probs=torch.tensor([[1.,0.,0.]],dtype=torch.float64).repeat(T,1),noise=torch.zeros(T,3,dtype=torch.float64))
    return dict(rounds=[dict(before=phi)],trace=dict(training_paths=[[path,copy.deepcopy(path)]],
        final_rollout=copy.deepcopy(path)),final=torch.zeros(T,4,dtype=torch.float64),finaltheta=phi)


def test_same_history_uses_updated_KD_weights_not_cached_initial_studentq():
    opd=_output([0.,0.,0.,0.,0.,0.]);kd=_output([3.,0.,0.,0.,0.,0.])
    for path in kd['trace']['training_paths'][0]:path['studentq']=opd['trace']['training_paths'][0][0]['studentq'].clone()
    result=saved_path_comparison(opd,kd)
    assert result['paired_paths']==2 and result['path_rows'][0]['identical_saved_history_nodes']==4
    assert result['path_rows'][0]['same_history_teacher_max_abs_error']==0
    assert result['path_rows'][0]['same_OPD_history_student_argmax_differences']==4
    assert result['path_rows'][0]['same_OPD_history_student_JS_mean']>0


def test_actual_history_differences_are_separate_from_same_state_policy_difference():
    opd=_output([0.]*6);kd=copy.deepcopy(opd)
    for path in kd['trace']['training_paths'][0]:
        path['memory_before'][1]=1;path['actions'][1]=1
    result=saved_path_comparison(opd,kd)
    assert result['path_rows'][0]['own_history_action_differences']==1
    assert result['path_rows'][0]['identical_saved_history_nodes']==3
    assert result['path_rows'][0]['same_OPD_history_student_argmax_differences']==0


def test_JS_handles_zero_probabilities_and_is_symmetric():
    a=np.array([[1.,0.,0.],[.3,.7,0.]])
    b=np.array([[0.,1.,0.],[.6,.4,0.]])
    assert np.isfinite(_JS(a,b)).all()
    assert np.array_equal(_JS(a,a),[0.,0.])
    assert np.allclose(_JS(a,b),_JS(b,a))


def test_oracle_support_excludes_stored_proposals_at_inactive_nodes():
    bank=dict(valid=torch.tensor([[1,1,1],[1,1,0],[1,0,1]],dtype=torch.bool),
        active=torch.tensor([False,True,False]))
    original=bank['valid'].clone()
    assert np.array_equal(_eligible_valid(bank),[[1,0,0],[1,1,0],[1,0,0]])
    assert torch.equal(bank['valid'],original)
