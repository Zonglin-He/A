import numpy as np
from test_temporal_profile import sample
from stvg.temporal import final_mbr
from stvg.temporal_late_guard import tables, combine, decode, BASE

def test_original_posterior_and_guard_keep_source_feasible():
    ids, records, logits = sample()
    a=tables(records,logits,ids,BASE)
    plain=final_mbr(ids,a.mean(0))
    assert decode(ids,combine(a,ids,BASE),BASE)['interval']==plain['interval']
    p={'mode':'late_bracket'};z=decode(ids,a,p)
    spans=np.array([final_mbr(ids,q)['indices']for q in a[-3:]])
    assert (spans.min(0)<=z['indices']).all()and(z['indices']<=spans.max(0)).all()
    source=final_mbr(ids,a[-1])['indices']
    assert np.isfinite(z['utility_grid'][tuple(source)])

def test_late_consensus_prevents_early_layer_boundary_reversal():
    ids=list(range(8));a=np.zeros((6,8,8));a[:3,0,7]=1;a[3:,2,5]=1
    q=a.mean(0);baseline=final_mbr(ids,q)
    result=decode(ids,a,{'mode':'late_bracket'})
    assert baseline['interval']!=[2,6]
    assert result['interval']==[2,6]
    assert decode(ids,a,{'mode':'early_bracket'})['interval']==[0,8]

def test_identical_layers_are_exact_fixed_point():
    ids,records,logits=sample();a=tables(records,logits,ids,BASE);a=np.repeat(a[-1:],6,axis=0)
    assert decode(ids,a,{'mode':'late_bracket'})['interval']==final_mbr(ids,a.mean(0))['interval']
