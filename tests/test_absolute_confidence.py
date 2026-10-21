import numpy as np
from tools.universal_trial.absolute_confidence import absolute_confidence_scores


def test_no_evidence_cannot_promote_an_unsupported_change():
    r,e=absolute_confidence_scores(np.array([0.,10.]),np.array([0.,.1]),np.zeros((8,3)),np.array([0,1,2]),[0,3])
    assert e==0 and r.argmax()==0 and r[1]<0


def test_fixed_window_and_same_exposure_apply_to_all_actual_candidates():
    c=np.array([[.2,.4,1.]]*8)
    r,e=absolute_confidence_scores(np.array([1.,2.]),np.array([.1,.2]),c,np.array([0,1,2]),[0,2])
    assert np.isclose(e,.3)
    assert np.allclose(r,np.array([.275,.55]))
