"""Synthetic physical-interval pooling checks; no cache, GT or model access."""
import copy

import numpy as np
import pytest

from stvg.temporal import probabilities,mass_table,final_mbr


@pytest.fixture
def core():
    from scripts.interval_pool import core as implementation
    return implementation


def view(ids,p):
    return dict(ids=list(ids),ij=np.asarray(np.triu_indices(len(ids),1)),p=np.asarray(p,dtype=np.float64))


def legal(q,U):
    assert q.shape==(len(U),len(U)) and np.isfinite(q).all() and (q>=0).all()
    assert abs(float(q.sum())-1)<1e-12
    assert not np.tril(q).any()


def expected_length(q,U):
    u=np.asarray(U,dtype=np.float64)
    return float((q*(u[None,:]+1-u[:,None])).sum())


def utility(q,U,interval):
    starts=np.asarray(U,dtype=np.float64)[:,None]
    ends=np.asarray(U,dtype=np.float64)[None,:]+1
    s,e=interval
    intersection=np.maximum(0,np.minimum(e,ends)-np.maximum(s,starts))
    union=(e-s)+(ends-starts)-intersection
    return float((q*np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)).sum())


def test_irregular_physical_mapping_preserves_each_complete_interval_and_mass(core):
    g=view([1,4,10],[.2,.3,.5]);U=[1,2,4,8,10,15]
    q=core.map_view(g,U);legal(q,U)
    expected=np.zeros((6,6),dtype=np.float64)
    expected[0,2]=.2;expected[0,4]=.3;expected[2,4]=.5
    np.testing.assert_array_equal(q,expected)
    assert expected_length(q,U)==pytest.approx(.2*4+.3*10+.5*7,abs=1e-14)
    assert core.choose(U,q,rule='map')==[4,11]
    assert not q[0,1] and not q[1,4]  # No interpolation onto intermediate grid IDs.


def test_single_view_repeated_views_and_identical_distributions_are_invariant(core):
    U=[0,1,2,4,5,9,10]
    a=view([0,2,5,10],[.1,.2,.1,.15,.25,.2])
    b=view([1,4,9],[.25,.35,.4])
    mapped=core.map_view(a,U)
    np.testing.assert_allclose(core.mix_views([a],U),mapped,atol=2e-16,rtol=0)
    pooled=core.mix_views([a,b],U)
    repeated=core.mix_views([a,b]*3,U)
    np.testing.assert_allclose(repeated,pooled,atol=2e-16,rtol=0)
    identical=core.mix_views([copy.deepcopy(a)for _ in range(4)],U)
    np.testing.assert_allclose(identical,mapped,atol=2e-16,rtol=0)
    legal(pooled,U)
    # The two views have different interval counts; each still contributes 1/2.
    first_support=mapped>0
    assert float(pooled[first_support].sum())==pytest.approx(.5,abs=1e-15)
    assert float(pooled[~first_support].sum())==pytest.approx(.5,abs=1e-15)


def test_identical_bimodal_views_keep_modes_under_mix_but_hull_creates_bridge(core):
    U=[1,2,7,8];g=view(U,[.5,0.,0.,0.,0.,.5])
    mix=core.mix_views([g,copy.deepcopy(g)],U)
    hull_U,hull=mass_table([g,copy.deepcopy(g)])
    assert hull_U==U
    assert mix[0,1]==mix[2,3]==.5 and mix[0,3]==0
    assert hull[0,1]==hull[2,3]==.25 and hull[0,3]==.5
    for interval,umix,uhull in (([1,3],.5,.375),([7,9],.5,.375),([1,9],.25,.625)):
        assert utility(mix,U,interval)==pytest.approx(umix,abs=1e-15)
        assert utility(hull,U,interval)==pytest.approx(uhull,abs=1e-15)
    assert core.choose(U,hull,rule='mbr')==[1,9]
    assert core.choose(U,mix,rule='mbr')in([1,3],[7,9])


def test_expected_hull_length_identity_uses_real_physical_endpoint_disagreement(core):
    a=view([1,4,10],[.2,.3,.5]);b=view([2,8,15],[.4,.1,.5])
    U,hull=mass_table([a,b]);mapped=[core.map_view(g,U)for g in(a,b)]
    def endpoints(g):
        ids=np.asarray(g['ids'],dtype=np.float64)
        return ids[g['ij'][0]],ids[g['ij'][1]]+1
    sa,ea=endpoints(a);sb,eb=endpoints(b);weight=a['p'][:,None]*b['p'][None,:]
    start_disagreement=float((weight*np.abs(sa[:,None]-sb[None,:])).sum())
    end_disagreement=float((weight*np.abs(ea[:,None]-eb[None,:])).sum())
    average_length=.5*sum(expected_length(q,U)for q in mapped)
    expected=average_length+.5*(start_disagreement+end_disagreement)
    assert expected_length(hull,U)==pytest.approx(expected,abs=1e-13)
    assert expected_length(core.mix_views([a,b],U),U)==pytest.approx(average_length,abs=1e-13)
    assert expected_length(hull,U)>average_length


def test_MBR_keeps_legal_zero_mass_candidates_instead_of_restricting_to_support(core):
    U=[0,1,2,3,4,5];q=np.zeros((6,6),dtype=np.float64)
    q[0,1]=q[0,3]=q[1,2]=1/3  # [0,2), [0,4), [1,3).
    assert q[0,2]==0
    selected=core.choose(U,q,rule='mbr')
    assert selected==[0,3]==final_mbr(U,q)['interval']
    assert utility(q,U,selected)==pytest.approx(25/36,abs=1e-15)
    assert utility(q,U,selected)>max(utility(q,U,I)for I in([0,2],[0,4],[1,3]))


def test_global_six_layer_outputs_preserve_legacy_hull_and_pool_probabilities_not_logits(core):
    records=[dict(offset=0,flip=False,frame_ids=[1,4,10]),dict(offset=1,flip=False,frame_ids=[2,8,15])]
    U=[1,2,4,8,10,15]
    base=np.asarray([[[.4,-.7],[-.2,.9],[.3,.1]]],dtype=np.float32)
    layers=[np.stack([base+np.asarray([[[k*.07,-k*.03],[0,k*.02],[-k*.04,0]]],dtype=np.float32)
                      for k in range(6)]),
            np.stack([base[:,::-1].copy()+np.asarray([[[0,-k*.05],[k*.03,0],[-k*.01,k*.06]]],dtype=np.float32)
                      for k in range(6)])]
    result=core.global_distributions(records,layers,U)
    assert len(result['views'])==6 and all(len(pair)==2 for pair in result['views'])
    assert np.asarray(result['hull_layers']).shape==(6,6,6)
    assert len(result['statistics'])==6 and all(isinstance(row,dict)for row in result['statistics'])
    legacy=[];flat=[]
    for k in range(6):
        grids=probabilities(records,[layers[o][k]for o in range(2)])
        expected_U,old=mass_table(grids);assert expected_U==U;legacy.append(old)
        np.testing.assert_array_equal(result['hull_layers'][k],old)
        stats=result['statistics'][k]
        assert stats['hull_expected_length_frames']==pytest.approx(expected_length(old,U),abs=1e-12)
        average=.5*sum(expected_length(core.map_view(g,U),U)for g in grids)
        assert stats['mean_view_length_frames']==pytest.approx(average,abs=1e-12)
        assert stats['hull_expected_length_frames']-average==pytest.approx(stats['endpoint_disagreement_frames'],abs=1e-12)
        assert stats['identity_abs_error_frames']<1e-12
        for o in range(2):
            assert result['views'][k][o]['ids']==grids[o]['ids']
            np.testing.assert_array_equal(result['views'][k][o]['p'],grids[o]['p'])
        flat.extend(grids)
    np.testing.assert_array_equal(result['hull'],np.mean(legacy,axis=0))
    np.testing.assert_allclose(result['mix'],core.mix_views(flat,U),atol=2e-15,rtol=0)
    for name in('hull','mix'):legal(result[name],U)
    # Averaging logits first is generally a different calculation.
    averaged=[np.mean(x,axis=0,dtype=np.float32)for x in layers]
    alternative=core.mix_views(probabilities(records,averaged),U)
    assert not np.allclose(result['mix'],alternative,atol=1e-8,rtol=0)


@pytest.mark.parametrize('kind',['unnormalized','negative','missing_endpoint'])
def test_invalid_distribution_or_physical_grid_is_rejected(core,kind):
    g=view([1,4,10],[.2,.3,.5]);U=[1,4,10]
    if kind=='unnormalized':g['p']=g['p']*.8
    elif kind=='negative':g['p']=np.asarray([-.2,.3,.9])
    else:U=[1,4,9]
    with pytest.raises((AssertionError,ValueError,KeyError)):
        core.map_view(g,U)
