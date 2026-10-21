"""CPU fake-decoder tests; no dataset, checkpoint, labels or GPU execution."""
import copy
import hashlib
import inspect
import textwrap

import pytest
import torch
from torch import nn

from scripts.output_opd.replay import FrozenTrajectoryReplay, expand_knots


class FakeDecoder(nn.Module):
    calls = 0

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(.03), requires_grad=False)
        self.bias = nn.Parameter(torch.tensor(.02), requires_grad=False)
        self.register_buffer('constant_buffer', torch.zeros(1))

    def forward(self, query_tgt=None, pred_boxes=None):
        FakeDecoder.calls += 1
        ref_anchors = []
        for _ in range(2):
            new_pred_boxes = .8 * pred_boxes + self.bias + self.gain * query_tgt[..., :4]
            ref_anchors.append(new_pred_boxes)
            pred_boxes = new_pred_boxes.detach()
        return torch.stack(ref_anchors).transpose(1, 2)


class FakeSource(nn.Module):
    def __init__(self):
        super().__init__()
        self.ground_decoder = nn.Module()
        self.ground_decoder.decoder = FakeDecoder()
        self.unused_encoder_weight = nn.Parameter(torch.ones(2), requires_grad=False)
        self.register_buffer('unused_encoder_buffer', torch.zeros(2))


def fixture():
    model = FakeSource().eval().requires_grad_(False)
    n, positions = 17, [0, 2, 4, 6, 8, 10, 12, 16]
    W = torch.zeros(n, 8)
    for t in range(n):
        if t in positions:
            W[t, positions.index(t)] = 1
        else:
            right = next(i for i, p in enumerate(positions) if p > t)
            left = right - 1
            fraction = (t - positions[left]) / (positions[right] - positions[left])
            W[t, left], W[t, right] = 1 - fraction, fraction
    inputs, offset_boxes = [], []
    with torch.no_grad():
        for view in (0, 1):
            physical = torch.arange(view, n, 2, dtype=torch.float32)
            base = .1 + physical[:, None, None] * .001 + torch.tensor([0., .1, .15, .2])[None, None]
            entry = dict(query_tgt=torch.zeros(len(physical), 1, 256), pred_boxes=base)
            inputs.append(entry)
            offset_boxes.append(model.ground_decoder.decoder(**entry).flatten(1, 2)[-1])
        source = torch.stack([offset_boxes[p % 2][p // 2] for p in range(n)]).float()
    E = source.double().clone()
    E[:, :2] += .01
    E[:, 2:] *= 1.05
    cache = dict(Source=source, E=E, spatial_inputs=inputs,
                 planned_positions=positions, frame_ids=list(range(n)))
    FakeDecoder.calls = 0
    return model, cache, W


def test_constructor_has_no_decoder_calls_and_first_forward_is_charged():
    model, cache, W = fixture()
    replay = FrozenTrajectoryReplay(model, cache, W)
    assert FakeDecoder.calls == 0
    assert replay.counts() == dict(full_trajectory_evals=0, native_offset_decoder_calls=0,
                                  setup_full_trajectory_evals=0, setup_native_offset_decoder_calls=0)
    assert not replay.initial_checks['zero_identity_checked_on_first_charged_forward']
    zero = torch.zeros(8, 256, requires_grad=True)
    out, native = replay.boxes(zero, return_native=True)
    assert FakeDecoder.calls == 2
    assert torch.equal(native.detach(), cache['Source'])
    assert torch.equal(out.detach(), cache['E'])
    assert out.requires_grad and native.requires_grad
    assert replay.counts() == dict(full_trajectory_evals=1, native_offset_decoder_calls=2,
                                  setup_full_trajectory_evals=0, setup_native_offset_decoder_calls=0)
    assert replay.initial_checks['first_identity_full_trajectory_eval'] == 1
    assert replay.initial_checks['original_native_Source_bit_exact']
    assert replay.initial_checks['original_E_bit_exact']


def test_first_nonzero_call_is_rejected_before_any_forward_even_null_mapping():
    model, cache, W = fixture()
    replay = FrozenTrajectoryReplay(model, cache, W * 0)
    with pytest.raises(ValueError, match='first charged'):
        replay.boxes(torch.ones(8, 256))
    with pytest.raises(ValueError, match='first charged'):
        replay.boxes_from_content(torch.ones(17, 256))
    assert FakeDecoder.calls == replay.evals == replay.native_offset_decoder_calls == 0


def test_first_zero_source_mismatch_fails_but_counts_actual_calls():
    model, cache, W = fixture()
    cache['Source'] = cache['Source'].clone()
    cache['Source'][3, 0] += .001
    replay = FrozenTrajectoryReplay(model, cache, W)
    with pytest.raises(AssertionError, match='saved Source'):
        replay.boxes(torch.zeros(8, 256, requires_grad=True))
    assert replay.evals == 1 and replay.native_offset_decoder_calls == 2
    assert not replay.initial_checks['zero_identity_checked_on_first_charged_forward']


def test_invalid_complete_proposal_is_counted_before_finite_validation():
    model, cache, W = fixture()
    replay = FrozenTrajectoryReplay(model, cache, W)
    replay.boxes(torch.zeros(8, 256))
    # Corrupt this fake cached input, rather than any frozen network parameter,
    # to exercise a completed two-offset forward yielding invalid output.
    replay.inputs[0]['pred_boxes'] = torch.full_like(replay.inputs[0]['pred_boxes'], float('nan'))
    with pytest.raises(FloatingPointError, match='Invalid corrected full trajectory'):
        replay.boxes(torch.zeros(8, 256))
    assert replay.evals == 2 and replay.native_offset_decoder_calls == 4
    assert FakeDecoder.calls == 4
    assert replay.setup_evals == 0


def test_first_training_forward_retains_complete_reference_gradient():
    model, cache, W = fixture()
    replay = FrozenTrajectoryReplay(model, cache, W)
    zero = torch.zeros(8, 256, requires_grad=True)
    out, native = replay.boxes(zero, return_native=True)
    ga = torch.autograd.grad(native[7, 0], zero, retain_graph=True)[0]
    gb = torch.autograd.grad(out[7, 0], zero)[0]
    expected = torch.zeros_like(zero)
    expected[:, 0] = .03 * 1.8 * W[7]
    torch.testing.assert_close(ga, expected, atol=1e-9, rtol=1e-6)
    torch.testing.assert_close(gb, expected, atol=1e-9, rtol=1e-6)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert replay.assert_frozen()


def test_dense_physical_mapping_and_offset_order_preserve_exact_old_operations():
    model, cache, W = fixture()
    replay = FrozenTrajectoryReplay(model, cache, W)
    replay.boxes(torch.zeros(8, 256))
    knots = torch.zeros(8, 256)
    knots[3, 0] = .2
    eta = expand_knots(knots, W)
    out, native = replay.boxes(knots, return_native=True)
    assert 7 not in cache['planned_positions'] and eta[7, 0] == .1
    assert native[7, 0] != cache['Source'][7, 0]
    expected_delta = .03 * 1.8 * eta[:, :4]
    torch.testing.assert_close(native - cache['Source'], expected_delta, atol=5e-8, rtol=1e-5)
    b, src = native.double(), cache['Source'].double()
    expected = torch.cat((cache['E'][:, :2] + b[:, :2] - src[:, :2],
                          cache['E'][:, 2:] * torch.exp(torch.log(b[:, 2:]) - torch.log(src[:, 2:]))), -1)
    # Preserve parentheses from the old correction, avoiding reassociation.
    expected[:, :2] = cache['E'][:, :2] + (b[:, :2] - src[:, :2])
    assert torch.equal(out, expected)
    assert torch.equal(out, replay.boxes_from_content(eta))
    assert FakeDecoder.calls == 6 and replay.evals == 3
    assert replay.native_offset_decoder_calls == 2 * replay.evals


def test_full_reference_is_instance_only_and_has_original_source_hash_proof():
    model, cache, W = fixture()
    source = inspect.getsource(FakeDecoder.forward)
    state = copy.deepcopy(model.state_dict())
    replay = FrozenTrajectoryReplay(model, cache, W)
    body = textwrap.dedent(source)
    changed = body.replace('pred_boxes = new_pred_boxes.detach()', 'pred_boxes = new_pred_boxes')
    assert replay.initial_checks['original_forward_sha256'] == hashlib.sha256(body.encode()).hexdigest()
    assert replay.initial_checks['full_reference_forward_sha256'] == hashlib.sha256(changed.encode()).hexdigest()
    assert inspect.getsource(FakeDecoder.forward) == source
    assert model.ground_decoder.decoder.forward.__func__ is FakeDecoder.forward
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
    assert FakeDecoder.calls == 0
    replay.boxes(torch.zeros(8, 256))
    knots = torch.linspace(-.1, .1, 8 * 256).reshape(8, 256)
    _, actual = replay.boxes(knots, return_native=True)
    original_raw = []
    with torch.no_grad():
        eta = W @ knots
        for view, entry in enumerate(cache['spatial_inputs']):
            kw = dict(entry)
            kw['query_tgt'] = entry['query_tgt'] + eta[view::2, None, :]
            original_raw.append(model.ground_decoder.decoder(**kw).flatten(1, 2)[-1])
        expected = torch.stack([original_raw[p % 2][p // 2] for p in range(17)])
    assert torch.equal(actual, expected)
    assert replay.assert_frozen()


def test_network_freeze_and_original_parameter_buffer_version_guards():
    model, cache, W = fixture()
    model.unused_encoder_weight.requires_grad_(True)
    with pytest.raises(ValueError, match='Source network parameter is not frozen'):
        FrozenTrajectoryReplay(model, cache, W)
    model.unused_encoder_weight.requires_grad_(False)
    replay = FrozenTrajectoryReplay(model, cache, W)
    with torch.no_grad():
        model.unused_encoder_buffer.add_(1)
    with pytest.raises(AssertionError, match='Original frozen network buffer changed'):
        replay.assert_frozen()
    assert FakeDecoder.calls == 0


def test_isolated_decoder_parameter_buffer_and_fixed_W_version_guards():
    for changed in ('parameter', 'buffer', 'W'):
        model, cache, W = fixture()
        replay = FrozenTrajectoryReplay(model, cache, W)
        with torch.no_grad():
            if changed == 'parameter': replay.decoder.gain.add_(1)
            elif changed == 'buffer': replay.decoder.constant_buffer.add_(1)
            else: replay.W.mul_(0)
        with pytest.raises(AssertionError):
            replay.assert_frozen()
        assert FakeDecoder.calls == 0


def test_no_truncated_reference_arm_or_bad_weights_or_precision():
    model, cache, W = fixture()
    with pytest.raises(ValueError, match='full_reference=True'):
        FrozenTrajectoryReplay(model, cache, W, full_reference=False)
    with pytest.raises(ValueError, match=r'W\[T,8\]'):
        FrozenTrajectoryReplay(model, cache, W[:, :7])
    invalid = W.clone()
    invalid[0, 0] = 2.
    with pytest.raises(ValueError, match='row mass'):
        FrozenTrajectoryReplay(model, cache, invalid)
    with pytest.raises(ValueError, match='FP32'):
        FrozenTrajectoryReplay(model.double(), cache, W)
    assert FakeDecoder.calls == 0


def test_three_round_six_full_forward_budget_includes_zero_training_call():
    model, cache, W = fixture()
    replay = FrozenTrajectoryReplay(model, cache, W)
    z = torch.zeros(8, 256, requires_grad=True)
    for r in range(3):
        out = replay.boxes(z)
        grad = torch.autograd.grad(out[:, 0].sum(), z)[0]
        z = (z - .01 * grad).detach().requires_grad_()
        with torch.no_grad(): replay.boxes(z)
    assert replay.evals == 6 and replay.native_offset_decoder_calls == 12
    assert replay.setup_evals == replay.setup_native_offset_decoder_calls == 0
    assert replay.assert_frozen()
