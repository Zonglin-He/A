import copy
import pytest
import torch
from vg_tta.desta3d_v2 import Desta3DAdapterV2
from vg_tta.desta3d_v2_tta_pilot import configure, adapt, forward, validate_prediction


@pytest.mark.parametrize('interface', ['calibration', 'convolution'])
def test_real_v2_three_step_scope_and_reset(interface):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    a = Desta3DAdapterV2(hidden_dim=128, architecture='dual3d', p1_enabled=False)
    before = copy.deepcopy(a.state_dict())
    base = {'visual_grid': torch.randn(1, 3, 2, 2, 2560),
            'query_tokens': torch.randn(1, 4, 2560),
            'query_mask': torch.ones(1, 4, dtype=torch.bool),
            'frame_times': torch.tensor([[0., .5, 1.]])}
    view = dict(base, visual_grid=base['visual_grid'] * .91 + .02)
    moments = {'feature_definition': 'v2_query_conditioned_readers',
               'spatial': {'mean': [0.] * 128, 'std': [1.] * 128},
               'event': {'mean': [0.] * 128, 'std': [1.] * 128}}
    result = adapt(a, base, view, interface=interface, alignment=.01, moments=moments)
    assert result['steps'] == 3 and result['history'][-1]['actual_Adam_steps'] == [3]
    assert result['history'][0]['terms_before']['parameter_anchor'] == 0
    assert result['terms_after']['parameter_anchor'] > 0
    assert any(result['changed_tensors'].values())
    assert torch.equal(a.gate_event, before['gate_event'])
    a.load_state_dict(before)
    assert all(torch.equal(v, before[k]) for k, v in a.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in a.parameters())


def test_prediction_failure_retention_and_physical_bounds():
    p = {'positions': [], 'boxes_cxcywh': torch.empty(0, 4),
         'geometry_valid': torch.empty(0, dtype=torch.bool), 'interval': [1, 2], 'format_ok': False}
    assert validate_prediction(p, 3)
    with pytest.raises(AssertionError):
        validate_prediction(dict(p, interval=[1, 3]), 3)


def test_time_observer_preserves_outputs_and_hooks():
    from types import SimpleNamespace
    from scripts.desta3d_v2_tta8 import observe_time
    logits = torch.arange(60.).reshape(1, 6, 10)
    sentinel = object()
    lm = lambda: (sentinel, logits)
    pg = SimpleNamespace(build_ptd_token_ids=lambda *a, **k: {'ref_end': 9, 'ordered_time_tokens': [3, 5, 7]},
                         _run_language_model=lm)
    def probe(**kw):
        return pg._run_language_model()
    pg._run_cached_ptd_probe = probe
    result, capture = observe_time(pg, SimpleNamespace(tokenizer=None), 3,
        lambda: pg._run_cached_ptd_probe(query_token_ids=torch.tensor([9])))
    assert result[0] is sentinel and result[1] is logits
    assert pg._run_language_model is lm and pg._run_cached_ptd_probe is probe
    assert capture['probe_count'] == 1
    assert torch.equal(capture['endpoint_logits'], logits[0, 1:3, [3, 5, 7]])
