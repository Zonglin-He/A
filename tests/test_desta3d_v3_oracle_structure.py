import numpy as np
from vg_tta.desta3d_v3_oracle_structure import field_statistics,torch_reference


def test_constant_rank_one():
    a=np.broadcast_to(np.arange(1.,9),(3,2,4,8)).copy();r=field_statistics(a)
    for k in ('global_energy','temporal_energy','global_cosine','temporal_cosine','rank1_energy'):
        assert abs(r['metrics'][k]-1)<1e-12
    assert all(abs(x['mean']-1)<1e-12 for x in r['neighbors'].values())


def test_temporal_projection_and_full_svd():
    rng=np.random.RandomState(18);a=rng.randn(3,4,2,12);r=field_statistics(a);t=torch_reference(a)
    sv=np.linalg.svd(a.reshape(-1,12),compute_uv=False)**2
    np.testing.assert_allclose(r['singular_value_squared'],sv,atol=1e-12,rtol=1e-12)
    for k,v in r['metrics'].items():assert abs(v-t['metrics'][k])<1e-12
    assert r['metrics']['temporal_energy']>=r['metrics']['global_energy']
    assert abs(r['metrics']['temporal_cosine']**2-r['metrics']['temporal_energy'])<1e-12
    b=np.broadcast_to(rng.randn(3,1,1,12),(3,4,2,12)).copy()
    assert abs(field_statistics(b)['metrics']['temporal_energy']-1)<1e-12


def test_zero_fields_missing_pairs_explicit():
    r=field_statistics(np.zeros((1,2,2,8)))
    assert all(v is None for v in r['metrics'].values())
    assert r['neighbors']['temporal']['pairs']==0
    assert r['neighbors']['horizontal']['defined']==0
    assert r['neighbors']['horizontal']['undefined']==2
