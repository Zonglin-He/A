import itertools
import numpy as np
from tools.tubedetr_spatial_search.numeric import batch_dp, finish, dense_scores
from tools.tubedetr_spatial_search.run import profiles
from stvg.metrics import DenseTube


def test_batch_dp_matches_exhaustive_nontrivial_paths():
    rng=np.random.default_rng(51);P,T,K=3,5,3
    unary=rng.normal(size=(P,T,K));pair=rng.normal(size=(P,T-1,K,K))
    valid=np.ones((P,T,K),bool);valid[:,2,1]=False
    boxes=rng.uniform(size=(T,K,4))
    _,actions=batch_dp(boxes,unary,pair,valid)
    for p in range(P):
        def score(a):return sum(unary[p,t,a[t]] for t in range(T))+sum(pair[p,t,a[t],a[t+1]] for t in range(T-1))
        possible=[a for a in itertools.product(range(K),repeat=T) if all(valid[p,t,a[t]] for t in range(T))]
        assert abs(score(actions[p])-max(map(score,possible)))<1e-12


def test_dense_score_matches_reference_including_hc_and_support():
    rng=np.random.default_rng(9);b=rng.uniform(.1,.8,(7,4,4));ids=np.array([2,6,11,15])
    truth={str(t):[5.,8.,44.,65.] for t in range(19)}
    for hc in [False,True]:
        _,iou=dense_scores(b,ids,120,80,truth,hc)
        for i in range(7):
            ref=DenseTube(b[i],dict(frame_ids=ids,input=dict(width=120,height=80)),{int(k):v for k,v in truth.items()},[0,19],clip=hc)
            np.testing.assert_allclose(iou[i],ref.iou,rtol=0,atol=1e-15)


def test_output_zero_and_one_are_exact_and_source_is_immutable():
    rng=np.random.default_rng(10);a=rng.uniform(.1,.9,(10,4));b=rng.uniform(.1,.9,(10,4));original=a.copy();obs=np.arange(10)%2==0
    for alpha,expected in [(0,a),(1,b)]:
        p=dict(alpha_observed=alpha,alpha_unobserved=alpha,width_scale=1.,height_scale=1.)
        assert np.array_equal(finish(a,b,obs,p),expected)
    assert np.array_equal(a,original)


def test_frozen_search_is_deterministic_and_preserves_controls():
    for ds in ['vidstg','hc2']:
        a=profiles(ds);assert a==profiles(ds) and len(a)==2048
        assert a[0]['family']=='source_control' and a[1]['original_baseline']
        assert len({tuple(sorted(p.items())) for p in a})==2048


def test_full_replay_never_substitutes_old_geometry_for_new_parameters():
    from tools.tubedetr_spatial_search.full import cpu_replayable
    p=profiles('vidstg')[1]
    assert not cpu_replayable(p)
    assert cpu_replayable({**p,'alpha_observed':0.,'alpha_unobserved':0.})
    assert cpu_replayable({**p,'family':'sealed_full_blend'})
