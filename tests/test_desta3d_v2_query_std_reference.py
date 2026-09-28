import copy
import pytest
import torch
from vg_tta.desta3d_v2_query_std_reference import aggregate_query_std


def row(key,parent,grid):
    x=torch.tensor(grid,dtype=torch.float64).reshape(-1,1)
    moments={'mean':x.mean(0),'second_moment':x.square().mean(0)}
    return {'key':key,'source':parent,'moments':{'referent':moments,'event':copy.deepcopy(moments)}}


def test_unequal_parents_queries_and_grids():
    rows=[row('a1','a',[0,2]),row('a2','a',[2,6]*5),row('b1','b',[8,14])]
    result=aggregate_query_std(rows)
    assert torch.equal(result['weights'],torch.tensor([.25,.25,.5],dtype=torch.float64))
    b=result['branches']['event']
    assert b['mean'].item()==6.75
    assert b['expected_query_std'].item()==2.25
    assert abs(b['within_query_rms_std'].item()-(.25*1+.25*4+.5*9)**.5)<1e-12
    assert b['population_std']>b['within_query_rms_std']>b['expected_query_std']
    repeated=[rows[0],row('a2','a',[2,6]*50),rows[2]]
    assert torch.equal(aggregate_query_std(repeated)['branches']['event']['expected_query_std'],b['expected_query_std'])


def test_zero_variance_uses_same_floor_as_objective():
    b=aggregate_query_std([row('a','a',[7,7,7])])['branches']['spatial']
    assert b['expected_query_std'].item()==1e-6


def test_reject_bad_moments_and_duplicates():
    r=row('a','a',[0,2])
    with pytest.raises(ValueError,match='duplicate'):aggregate_query_std([r,r])
    r['moments']['event']['second_moment']=torch.tensor([-1.])
    with pytest.raises(ValueError,match='negative'):aggregate_query_std([r])
