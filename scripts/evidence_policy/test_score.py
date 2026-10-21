"""Pure CPU scoring helpers; no real inputs, GT, predictions or policies."""
import copy
import json
import numpy as np
import torch

from scripts.evidence_policy.score import logit_diagnostics,target_change,final_node_changes,_aggregate_nodes,_focus_pairs,_training_statistics,_history_comparison,_public_margins


def test_margins_use_actual_support_and_include_fallback_counts():
    logits=np.array([[0.,3.,2.],[0.,1.,-1.],[0.,-2.,-3.]])
    prior=np.tile(np.log([.7,.15,.15]),(3,1))
    valid=np.ones((3,3),bool);active=np.array([False,True,True]);obs=np.array([False,False,True])
    result=logit_diagnostics(logits,prior,valid,active,obs)
    assert result['unobserved']['nodes']==2
    assert result['unobserved']['eligible_alternative_nodes']==1
    assert result['unobserved']['E_only_nodes']==1
    assert result['unobserved']['margin_mean']==1
    assert np.isclose(result['unobserved']['prior_barrier_mean'],np.log(.7/.15))


def test_future_target_change_is_same_state_and_inactive_cannot_change():
    scores=np.array([[0.,3.,2.],[0.,1.,3.],[3.,1.,2.]])
    immediate=np.array([[0.,2.,3.],[0.,2.,1.],[3.,2.,1.]])
    result=target_change(scores,immediate,np.ones((3,3),bool),[False,True,True],[False,False,True])
    assert result['all']['nodes']==3 and result['all']['argmax_differences']==1
    assert result['active_unobserved']['argmax_difference_fraction']==1


def test_recovery_damage_denominators_retain_unchanged_fallback():
    result=final_node_changes([.2,.7,.3,np.nan],[.6,.2,.3,np.nan],[False,False,False,True],[True,True,False,False])
    row=result['unobserved']
    assert row['nodes']==3 and row['replaced_nodes']==2 and row['unchanged_nodes']==1
    assert row['initially_below_half_nodes']==2 and row['recovery_fraction']==.5
    assert row['initially_at_least_half_nodes']==1 and row['damage_fraction']==1
    summary=_aggregate_nodes([result,result])['unobserved']
    assert summary['nodes']==6 and summary['harmful_replacement_fraction']==.5


def test_focus_pairs_cover_matched_AB_and_previous_controls():
    pairs=_focus_pairs()
    assert len(pairs)==len(set(pairs))
    for arm in ('OPD','FixedKD','Direct'):
        assert ('B_'+arm,'A_'+arm)in pairs
        assert ('A_'+arm,'Box'+arm)in pairs
    assert ('B_OPD','B_Direct')in pairs and ('B_OPD','B_FixedKD')in pairs


def _saved_output():
    path=dict(teacher_scores=np.array([[0.,2.,1.],[0.,1.,2.]]),r=np.array([[0.,1.,2.],[0.,1.,2.]]),
        effective_valid=np.ones((2,3),bool),active=np.ones(2,bool),observed=np.array([True,False]),
        memory_before=np.zeros((2,2)),previous_action_probs=np.tile([1.,0.,0.],(2,1)),
        previous_features=np.zeros((2,2)),teacherq=np.tile([.2,.3,.5],(2,1)))
    return dict(rounds=[dict(outer=0),dict(outer=0)],
        trace=dict(outer_acquisitions=[[]],training_paths=[[path,copy.deepcopy(path)],[copy.deepcopy(path),copy.deepcopy(path)]]))


def test_direct_empty_acquisitions_use_first_saved_inner_once_per_outer():
    result=_training_statistics(_saved_output())
    assert len(result)==2 and result[0]['Q_versus_immediate']['all']['argmax_differences']==1


def test_same_target_rule_checks_complete_incoming_state():
    a=_saved_output();b=copy.deepcopy(a)
    result=_history_comparison(a,b)
    assert result['rows'][0]['identical_saved_history_nodes']==2
    assert result['rows'][0]['identical_history_target_max_abs_error']==0
    b['trace']['training_paths'][0][0]['previous_features'][1]=1
    b['trace']['training_paths'][0][0]['teacherq'][1]=[.6,.3,.1]
    result=_history_comparison(a,b)
    assert result['rows'][0]['different_saved_history_unobserved_nodes']==1
    assert result['rows'][0]['different_history_target_argmax_differences']==1


def test_real_policy_margin_serializes_as_finite_anonymous_summaries():
    from scripts.evidence_policy.policy import _margin
    features=torch.zeros(3,3,8,dtype=torch.float64);features[:,1,0]=2
    prior=torch.tensor([[.7,.15,.15],[.7,.15,.15],[1.,0.,0.]],dtype=torch.float64).log()
    valid=torch.tensor([[True,True,True],[True,True,True],[True,False,False]])
    private=_margin([dict(features=features,prior_log_probs=prior,effective_valid=valid)],
        torch.tensor([.8,0.,0.,0.,0.,0.,0.,0.],dtype=torch.float64))
    public=_public_margins(private)
    encoded=json.dumps(public,allow_nan=False)
    assert 'Tensor'not in encoded
    for field in ('best_alternative_minus_E','learned_alternative_minus_E','prior_E_barrier'):
        assert public[field]['count']==2 and isinstance(public[field],dict)
    assert public['best_alternative_minus_E']['positive_nodes']==2
    assert public['valid_nodes']==2 and public['positive_alternative_gap_nodes']==2
