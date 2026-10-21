import pytest
from tools.tubedetr_opd_direct_diagnosis import cluster_difference, needs_research


def test_cluster_bootstrap_preserves_equal_query_estimand_and_missingness():
    m = cluster_difference([.1,.1,-.1,None], ['a','a','b','c'], repeats=200)
    assert m['queries']==3 and m['parents']==2
    assert m['mean_pp']==pytest.approx(10/3)
    assert m['equal_parent_mean_pp']==pytest.approx(0)
    assert m['wins']==2 and m['losses']==1
    assert m['CI95_pp'][0]<=m['mean_pp']<=m['CI95_pp'][1]


def test_near_zero_or_spatial_damage_triggers_mechanistic_research():
    good={'mean_pp':1.,'CI95_pp':[.1,2.]}
    p={'m_vIoU':good,'m_sIoU':good,'unobserved_actual_sIoU':good}
    assert not needs_research(p,.5)['required']
    p['unobserved_actual_sIoU']={'mean_pp':-.1}
    assert needs_research(p,.5)['required']
    p['unobserved_actual_sIoU']=good
    p['m_vIoU']={'mean_pp':.1,'CI95_pp':[-.5,.8]}
    assert len(needs_research(p,.5)['reasons'])==2


def test_undefined_stratum_stays_undefined():
    m=cluster_difference([None],['a'])
    assert m['queries']==0 and m['mean_pp'] is None and m['CI95_pp'] is None
