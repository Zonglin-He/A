import itertools
import numpy as np
import torch
from tools.llava_st.evidence import SourceEvidence
from tools.simplepath_tube.core.spatial import build_geometry, evidence_dp
from tools.simplepath_tube.core.prototype import one_step


def test_new_core_exact_dp_bruteforce_and_removed_old_terms():
    T, H, W = 10, 3, 3
    features = torch.eye(9)[None].repeat(T, 1, 1)
    positions = [0, 1, 2, 3, 5, 6, 8, 9]
    e = SourceEvidence(features, torch.ones(T, 9, dtype=torch.bool), torch.arange(T),
                       positions, torch.ones(8, H, W))
    E = torch.tensor([.4, .5, .2, .3], dtype=torch.float64)[None].repeat(T, 1)
    boxes = torch.tensor([.5, .5, .4, .5], dtype=torch.float64)[None].repeat(8, 1)
    g = build_geometry(e, E, boxes)
    assert g.metadata['cycle'] is False and g.metadata['initialization_guard'] is False
    bank = g.for_interval([0, T])
    chosen = evidence_dp(bank)
    possibilities = [torch.where(v)[0].tolist() for v in bank.valid]
    best = max(float(bank.score_path(torch.tensor(a))) for a in itertools.product(*possibilities))
    assert abs(chosen['score'] - best) < 1e-12


def test_prototype_preserves_native_mass_support_and_constant_identity():
    q = np.triu(np.random.default_rng(3).uniform(size=(8, 8)), 1); q /= q.sum()
    z = np.full((8, 11), .27)
    new, trace = one_step(z, q)
    assert np.array_equal(new, q) and not trace['L'].any()
    assert trace['kappa'] == 64 and trace['beta'] == 1 and trace['rounds'] == 1
    z[3:6, 0] = 1
    new, _ = one_step(z, q)
    assert abs(new.sum() - 1) < 1e-12 and not new[q == 0].any()


def test_incomplete_population_stops_before_any_label_access(tmp_path, monkeypatch):
    import json
    import pytest
    from tools.simplepath_tube import run
    target = tmp_path / 'outputs' / 'vidstg' / 'dev128'
    target.mkdir(parents=True)
    (target / 'sealed.json').write_text(json.dumps(dict(status='complete', queries=0)))
    touched = []
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'OUT', tmp_path / 'outputs')
    monkeypatch.setattr(run, 'checked', lambda: {})
    monkeypatch.setattr(run, 'rows_for', lambda *a: [dict(id='existing-query')])
    monkeypatch.setattr(run, 'sha', lambda path: touched.append(str(path)) or 'hash')
    with pytest.raises(AssertionError):
        run.score('vidstg', False)
    assert touched == []
