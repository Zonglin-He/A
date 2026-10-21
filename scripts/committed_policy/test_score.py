"""Pure CPU reporting checks; no real assets, GT or policy inference."""
import json
import numpy as np
import pytest
import torch

from scripts.committed_policy.score import focus_pairs,replacement_counts,public_statistics,signed_changes,NAMES,route_counts,teacher_suggestions,_validate_fit


def test_replacement_denominators_include_all_unobserved_fallback_nodes():
    E=np.tile([.5,.5,.2,.2],(5,1));final=E.copy();final[1,0]=.6
    result=replacement_counts(final,E,[1,1,0,1,0],[1,0,0,0,0])
    assert result['unobserved']==dict(nodes=4,replaced_nodes=1)
    assert result['inactive']==dict(nodes=2,replaced_nodes=0)
    final[2,0]=.6
    with pytest.raises(AssertionError):replacement_counts(final,E,[1,1,0,1,0],[1,0,0,0,0])


def test_public_statistics_is_native_finite_without_private_vectors():
    result=public_statistics(dict(q=torch.tensor([0.,.25,.75]),events=np.int64(3),reset=True))
    assert result['q']['count']==3 and result['events']==3 and result['q']['zero']==1
    assert isinstance(json.dumps(result,allow_nan=False),str)
    with pytest.raises(AssertionError):public_statistics(torch.tensor([float('nan')]))


def test_signed_controls_include_matched_direct_KD_and_all_old_references():
    pairs=focus_pairs();assert len(pairs)==len(set(pairs))
    for baseline in ('CommitDirect','CommitFixedKD','CommitInitial','CommitDP','B_OPD','GeometryOPD'):
        assert ('CommitOPD',baseline)in pairs
    values={name:dict(v=.5,unobserved_actual_sIoU=.5)for name in NAMES}
    values['CommitOPD']['v']=.4
    signed=signed_changes([dict(values=values)])
    found=[row for row in signed if row['method']=='CommitOPD'and row['baseline']=='CommitDirect'][0]
    assert found['severe_vIoU_drop_over5pp']and np.isclose(found['vIoU_delta_pp'],-10)


def test_route_events_are_distinct_from_box_replacements():
    path=dict(active=np.array([1,1,0,1],bool),observed=np.array([1,0,0,0],bool),
        commit_event=np.array([0,1,0,0]),switch_event=np.array([0,0,0,1]),reset_event=np.array([1,0,0,0]),commit_mass=np.array([0.,.8,.8,.6]))
    row=route_counts(path)['unobserved']
    assert row['nodes']==3 and row['commit_event']==1 and row['switch_event']==1 and row['reset_event']==0


def test_teacher_replacement_damage_is_same_saved_state_and_counts_ties():
    result=teacher_suggestions([.8,.2,.5,np.nan],[.4,.7,.5,np.nan],[.3,.6,.4,np.nan],
        [1,1,2,0],[0,0,1,0],[1,1,1,0],[1,0,0,0])
    row=result['active']
    assert row['suggested_replacement_nodes']==3 and row['harmed_suggested_replacements']==1
    assert np.isclose(row['harmful_replacement_fraction'],1/3)
    assert result['unobserved']['harmed_suggested_replacements']==1


def test_fit_receipt_validation_and_public_serialization_use_three_outer_ten_inner():
    path=dict(active=torch.tensor([True,False]),observed=torch.tensor([True,False]),
        commit_event=torch.zeros(2,dtype=torch.bool),switch_event=torch.zeros(2,dtype=torch.bool),
        reset_event=torch.zeros(2,dtype=torch.bool),commit_mass=torch.zeros(2),teacherq=torch.full((2,3),1/3))
    inner=[dict(gradient=torch.zeros(4),loss=.1)for _ in range(10)]
    rounds=[dict(before_score=0.,after_score=0.,rolled_back=False,inner=inner)for _ in range(3)]
    costs=dict(updates=30,backward=30,state_only_backwards=30,accepted_outer=3,rejected_outer=0,
        heavy_forwards=0,decoder_calls=0,decoder_backwards=0,encoder_calls=0,expert_calls=0)
    output=dict(initialtheta=torch.zeros(4),finaltheta=torch.zeros(4),finalweights=torch.ones(4),
        rounds=rounds,costs=costs,trace=dict(final_rollout=path,training_paths=[[[{},{}]for _ in range(10)]for _ in range(3)]))
    result=_validate_fit(output)
    assert len(result['rounds'])==3 and len(result['rounds'][0]['inner'])==10
    assert json.loads(json.dumps(result,allow_nan=False))['final_effective_weights']['mean']==1
