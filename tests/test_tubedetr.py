"""Structural checks use the actual upstream decoder, never claim trained accuracy."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from stvg.tubedetr import official_imports, TubeReplay, inject_content, endpoint_posterior
from stvg.evaluation_v2 import aggregate
from stvg.positions import observation_positions


def test_full_grid_content_mapping():
    positions = [0, 2, 3, 5, 7, 9, 12, 14]
    state = torch.arange(8*256).view(8, 256).float()
    q = torch.zeros(15, 1, 256)
    out = inject_content(q, state, positions)
    assert torch.equal(out[positions, 0], state)
    assert out[[1, 4, 6, 8, 10, 11, 13]].count_nonzero() == 0


def test_actual_upstream_decoder_zero_reset_gradient():
    official_imports()
    from models.transformer import TransformerDecoder, TransformerDecoderLayer
    from models.tubedetr import MLP
    torch.manual_seed(41)
    decoder = TransformerDecoder(TransformerDecoderLayer(256, 8, 512, dropout=0), 6,
                                 torch.nn.LayerNorm(256), return_intermediate=True).eval().requires_grad_(False)
    head = MLP(256, 256, 4, 3).eval().requires_grad_(False)
    n = 16
    args = (torch.zeros(n, 1, 256), torch.randn(5, n, 256))
    kwargs = dict(pos=torch.randn(5, n, 256), query_pos=torch.randn(n, 1, 256))
    model = SimpleNamespace(transformer=SimpleNamespace(decoder=decoder), bbox_embed=head)
    with torch.no_grad():
        raw = head(decoder(*args, **kwargs).transpose(1, 2).flatten(1, 2)).sigmoid()[-1]
    positions = observation_positions(list(range(n)), [2, 10])
    rp = TubeReplay(model, dict(decoder_inputs=(args, kwargs), source=raw), raw.double().numpy(), positions)
    before = {k: v.clone() for k, v in decoder.state_dict().items()}
    value = rp.values()['boxes']
    gradient = torch.autograd.grad(value[positions, 0].sum(), rp.state)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    with torch.no_grad():
        rp.state.add_(.01*gradient)
        assert not torch.equal(rp.values()['native'], raw)
        rp.state.zero_()
        assert torch.equal(rp.values()['native'], raw)
    assert all(torch.equal(v, before[k]) for k, v in decoder.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in decoder.parameters())


def test_single_grid_temporal_posterior():
    z = torch.tensor([[[0., 1.], [2., 0.], [0., 4.]]])
    q = endpoint_posterior(z)
    assert np.isclose(q.sum(), 1) and not np.tril(q).any()
    assert divmod(q.argmax(), 3) == (1, 2)


def test_query_estimand_and_cluster_bootstrap():
    rows = []
    for parent, values in [('a', [1., 1., 1.]), ('b', [0.])]:
        for i, value in enumerate(values):
            rows.append(dict(id=f'{parent}{i}', parent=parent,
                values={'Source': {'m_vIoU': 0.}, 'Full': {'m_vIoU': value}}))
    result = aggregate(rows, bootstrap=100)
    assert result['query_means']['Full']['m_vIoU'] == .75
    assert result['equal_parent_means']['Full']['m_vIoU'] == .5
    assert result['paired']['Full-Source']['query_mean_pp'] == 75
    assert result['paired']['Full-Source']['equal_parent_mean_pp'] == 50
