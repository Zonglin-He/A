import numpy as np
from scipy.special import logsumexp
from test_temporal_profile import sample
from stvg.temporal_refinement import correction, tables, combine, BASE
from stvg.temporal_readout import tables as old_tables, BASE as OLD_BASE

def test_source_and_global6_controls_are_exact():
    ids, records, logits = sample()
    expected = old_tables(records, logits, ids, OLD_BASE)
    for mode, truth in [('source', expected[-1]), ('global6', expected.mean(0))]:
        p = {'mode': mode}
        assert np.array_equal(combine(tables(records, logits, ids, p), ids, p), truth)

def test_bound_and_agreement_have_no_logit_gauge_dependence():
    rng = np.random.default_rng(101)
    z = rng.normal(size=(6, 1, 17, 2))
    _, latest, delta = correction(z)
    logp = z[:, 0] - logsumexp(z[:, 0], axis=1, keepdims=True)
    prev = logp[-2] - logp[-3]
    assert np.all(abs(delta) <= np.minimum(abs(prev), abs(latest)) + 1e-14)
    assert np.all(delta[prev * latest <= 0] == 0)
    shifted = z + rng.normal(size=(6, 1, 1, 2)) * 100
    assert np.allclose(correction(shifted)[2], delta, atol=1e-12)

def test_constant_history_is_no_correction_and_all_modes_legal():
    ids, records, logits = sample()
    for z in logits:
        assert np.allclose(correction(np.repeat(z[-1:], 6, axis=0))[2], 0)
    for mode in ['contrast', 'consistent', 'reverse']:
        p = {'mode': mode}
        q = combine(tables(records, logits, ids, p), ids, p)
        assert abs(q.sum()-1) < 1e-12
        assert not q[np.tril_indices(len(ids))].any()
