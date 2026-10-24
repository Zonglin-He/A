import numpy as np
from tools.simplepath_llava.adapter import physical_prior


def test_conditional_joint_is_not_an_independent_endpoint_outer_product():
    z=np.full((100,100),-60.);z[1,20]=3;z[1,80]=-10;z[3,20]=-10;z[3,80]=3
    q=physical_prior(z,np.arange(100))
    assert abs(q.sum()-1)<1e-12 and not np.tril(q).any()
    assert q[1,20]>q[1,80]*100 and q[3,80]>q[3,20]*100


def test_duplicate_physical_nodes_aggregate_only_legal_distinct_endpoints():
    ids=np.repeat(np.arange(50),2);q=physical_prior(np.zeros((100,100)),ids)
    assert q.shape==(50,50) and abs(q.sum()-1)<1e-12 and not np.tril(q).any()
    np.testing.assert_allclose(q[np.triu_indices(50,1)],1/1225,atol=1e-15)
