import copy,pytest
from vg_tta.desta3d_v3_gap_decision import quadrant,choose_branch

def fixture():
    oracle=dict(v_mean=1.,v_lower=.2,t_mean=.1,s_mean=.1)
    seed=dict(oracle_minus_learned_lower=.1,cos_mean=0.,cos_median=0.,oracle_descent=dict(event=.95,spatial=.95),learned_descent=dict(event=.4,spatial=.4),saturated_fraction=1.,v_mean=-.2)
    return oracle,[seed,copy.deepcopy(seed)]

def test_A_B_and_seed_disagreement():
    o,s=fixture();assert choose_branch(o,s)['DECISION']=='A'
    for x in s:x.update(cos_mean=.5,cos_median=.5,learned_descent=dict(event=.9,spatial=.9))
    assert choose_branch(o,s)['DECISION']=='B'
    s[1]['cos_mean']=0.;assert choose_branch(o,s)['DECISION']=='INCONCLUSIVE'

def test_C_cannot_be_invented_from_one_step_failure():
    o,s=fixture();o['v_lower']=-.1
    r=choose_branch(o,s);assert r['DECISION']=='INCONCLUSIVE' and r['recommendation']=='C'
    with pytest.raises(ValueError):choose_branch(o,s,dict(status='CPU_tested',union_qualified=True,free_qualified=True))
    for u,f,label in [(True,True,'C1'),(False,True,'C2'),(False,False,'C3')]:
        assert choose_branch(o,s,dict(status='independently_audited',union_qualified=u,free_qualified=f))['DECISION']==label

def test_quadrants_keep_all_ties_and_neutral_cases():
    got=[quadrant(x,y) for x in [-1,0,1] for y in [-1,0,1]]
    assert len(set(got))==9 and quadrant(1,-1)=='gain/harm'
    assert quadrant(1e-12,0)=='flat/flat'
