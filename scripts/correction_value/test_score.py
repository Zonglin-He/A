"""Pure CPU tail-support and aggregation tests; no assets or GT reads."""
import numpy as np

from scripts.correction_value.score import intervention_metrics,query_summary,paired_bootstrap,decision_gate,aggregate_cohort


def test_validation_hashlib_dependency_is_bound_without_reading_assets():
    from scripts.correction_value import score
    assert score.hashlib.sha256(b'abc').hexdigest()=='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'


def test_forced_node_gain_is_excluded_and_inactive_or_observed_tail_not_primary():
    student=np.array([.1,.1,.2,.7,.3,.4]);teacher=student.copy();teacher[1]=1;teacher[2]=.6;teacher[3]=.2;teacher[4]=.8
    values=dict(Student=student,Teacher=teacher,Random=student.copy())
    actions={name:np.zeros(6,int)for name in values};actions['Teacher'][1:3]=1
    row=intervention_metrics(values,actions,1,[1,1,1,0,1,1],[0,0,0,0,1,0])
    primary=row['groups']['active_unobserved_tail']
    assert primary['nodes']==2 and np.isclose(primary['differences']['Teacher-Student'],.2)
    assert row['action_changes']['active_unobserved_tail']['Teacher-Student']==1
    assert row['forced_node_IoU']['Teacher']==1
    assert row['groups']['all_unobserved_tail']['nodes']==3
    assert primary['consequences']['Teacher']['recovered_to_half_nodes']==1


def test_near3_is_strict_future_and_far_starts_at_fourth_node():
    e=np.zeros(7);teacher=e.copy();teacher[1:4]=.3;teacher[4:]=.9
    row=intervention_metrics(dict(Student=e,Teacher=teacher,Random=e),{name:np.zeros(7,int)for name in ('Student','Teacher','Random')},0,np.ones(7,bool),np.zeros(7,bool))
    assert row['groups']['near3_active_unobserved']['nodes']==3
    assert row['groups']['far_active_unobserved']['nodes']==3
    assert np.isclose(row['groups']['near3_active_unobserved']['differences']['Teacher-Student'],.3)
    assert np.isclose(row['groups']['far_active_unobserved']['differences']['Teacher-Student'],.9)


def test_empty_scored_point_and_empty_query_contribute_zero_without_selecting_points():
    actions={name:np.zeros(2,int)for name in ('Student','Teacher','Random')}
    good=intervention_metrics(dict(Student=np.array([0.,.2]),Teacher=np.array([0.,.6]),Random=np.array([0.,.2])),actions,0,[1,1],[0,0])
    empty=intervention_metrics({name:np.array([0.,np.nan])for name in actions},actions,0,[1,1],[0,0])
    query=query_summary([good,empty])['groups']['active_unobserved_tail']
    assert query['eligible']and query['scored_interventions']==1
    assert np.isclose(query['differences']['Teacher-Student'],.2)
    none=query_summary([])['groups']['active_unobserved_tail']
    assert not none['eligible']and none['differences']['Teacher-Student']==0 and none['absolute_sIoU']['Teacher']is None


def test_gate_uses_all64_query_deltas_and_both_strict_positive_bounds():
    positive=paired_bootstrap(np.full(64,.02));zero=paired_bootstrap(np.zeros(64))
    assert decision_gate({'Teacher-Student':positive,'Teacher-Random':positive})['passed']
    failed=decision_gate({'Teacher-Student':positive,'Teacher-Random':zero})
    assert not failed['passed']and failed['status']=='inconclusive'
    negative=paired_bootstrap(np.full(64,-.01))
    assert decision_gate({'Teacher-Student':negative,'Teacher-Random':positive})['status']=='negative'
    eligible=paired_bootstrap([0.,.04],[False,True])
    assert eligible['queries']==1 and eligible['mean_pp']==4


def test_strata_keep_absent_queries_and_two_action_random_teacher_equality():
    actions={name:np.zeros(2,int)for name in ('Student','Teacher','Random')}
    row=intervention_metrics(dict(Student=np.array([0.,.2]),Teacher=np.array([0.,.6]),Random=np.array([0.,.6])),actions,0,[1,1],[0,0])
    row.update(legal_action_count=2,disagreement_type='noise_only_disagreement')
    result=aggregate_cohort([dict(points=[row]),dict(points=[])])
    primary=result['all']['groups']['active_unobserved_tail']
    assert np.isclose(primary['paired']['Teacher-Student']['all_queries']['mean_pp'],20)
    assert np.isclose(primary['paired']['Teacher-Student']['eligible_queries']['mean_pp'],40)
    assert primary['paired']['Teacher-Random']['all_queries']['mean_pp']==0
    policy=result['policy_disagreement']['groups']['active_unobserved_tail']
    assert policy['eligible_queries']==0 and policy['paired']['Teacher-Student']['all_queries']['mean_pp']==0
