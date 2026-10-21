"""CPU-only fake-decoder checks; no checkpoint, dataset, RGB, or GT access."""
import copy
import hashlib
import inspect

import pytest
import torch
from torch import nn

from scripts.trajectory_opd.replay import FrozenTrajectoryReplay, expand_knots


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(.03), requires_grad=False)
        self.bias = nn.Parameter(torch.tensor(.02), requires_grad=False)

    def forward(self, query_tgt=None, pred_boxes=None):
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
    n = 17
    positions = [0, 2, 4, 6, 8, 10, 12, 16]
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
    cache = dict(Source=source, E=E, spatial_inputs=inputs, planned_positions=positions,
                 frame_ids=list(range(n)))
    return model, cache, W


def test_zero_is_saved_Source_E_and_actual_call_counts():
    model, cache, W = fixture()
    replay = FrozenTrajectoryReplay(model, cache, W)
    assert replay.counts() == dict(full_trajectory_evals=0, native_offset_decoder_calls=0,
                                  setup_full_trajectory_evals=1, setup_native_offset_decoder_calls=2)
    boxes, native = replay.boxes(torch.zeros(8, 256), return_native=True)
    assert torch.equal(native, cache['Source'])
    assert torch.equal(boxes, cache['E'])
    assert replay.counts() == dict(full_trajectory_evals=1, native_offset_decoder_calls=2,
                                  setup_full_trajectory_evals=1, setup_native_offset_decoder_calls=2)
    assert replay.assert_frozen()


def test_dense_mapping_directly_edits_nonanchors_and_preserves_physical_order():
    model, cache, W = fixture()
    replay = FrozenTrajectoryReplay(model, cache, W)
    knots = torch.zeros(8, 256)
    knots[3, 0] = .2
    eta = expand_knots(knots, W)
    _, native = replay.boxes(knots, return_native=True)
    assert 7 not in cache['planned_positions'] and eta[7, 0] == .1
    assert native[7, 0] != cache['Source'][7, 0]
    # FakeDecoder has no temporal attention: this change proves direct node
    # injection and cannot be an effect leaking from an observed anchor.
    expected_delta = .03 * (1. + .8) * eta[:, :4]
    torch.testing.assert_close(native - cache['Source'], expected_delta, atol=5e-8, rtol=1e-5)
    assert torch.equal(replay.boxes(knots), replay.boxes_from_content(eta))
    assert replay.native_offset_decoder_calls == 2 * replay.evals


def test_nonanchor_loss_has_interpolation_gradient_to_knots():
    model, cache, W = fixture()
    replay = FrozenTrajectoryReplay(model, cache, W)
    knots = torch.zeros(8, 256, requires_grad=True)
    _, native = replay.boxes(knots, return_native=True)
    gradient = torch.autograd.grad(native[7, 0], knots)[0]
    expected = torch.zeros_like(knots)
    expected[:, 0] = .03 * W[7]
    torch.testing.assert_close(gradient, expected, atol=1e-9, rtol=1e-6)
    assert replay.assert_frozen()


def test_full_reference_patch_is_instance_only_forward_identical_and_restores_gradient():
    model, cache, W = fixture()
    original_source = inspect.getsource(FakeDecoder.forward)
    original_model = copy.deepcopy(model.state_dict())
    native = FrozenTrajectoryReplay(model, cache, W, full_reference=False)
    full = FrozenTrajectoryReplay(model, cache, W, full_reference=True)
    knots = torch.linspace(-.1, .1, 8 * 256).reshape(8, 256).requires_grad_()
    _, a = native.boxes(knots, return_native=True)
    _, b = full.boxes(knots, return_native=True)
    assert torch.equal(a, b)
    ga = torch.autograd.grad(a[7, 0], knots, retain_graph=True)[0]
    gb = torch.autograd.grad(b[7, 0], knots)[0]
    torch.testing.assert_close(gb, 1.8 * ga, atol=1e-9, rtol=1e-6)
    assert full.initial_checks['full_reference_forward_bit_exact']
    assert full.setup_evals == 2 and full.setup_native_offset_decoder_calls == 4
    assert inspect.getsource(FakeDecoder.forward) == original_source
    assert model.ground_decoder.decoder.forward.__func__ is FakeDecoder.forward
    assert all(torch.equal(model.state_dict()[key], value) for key, value in original_model.items())
    assert hashlib.sha256(original_source.encode()).hexdigest()
    assert full.assert_frozen() and native.assert_frozen()


def test_setup_checks_whole_Source_and_version_guard_checks_decoder():
    model, cache, W = fixture()
    model.unused_encoder_weight.requires_grad_(True)
    with pytest.raises(ValueError, match='Source network parameter is not frozen'):
        FrozenTrajectoryReplay(model, cache, W)
    model.unused_encoder_weight.requires_grad_(False)
    replay = FrozenTrajectoryReplay(model, cache, W)
    with torch.no_grad():
        replay.decoder.gain.add_(.001)
    with pytest.raises(AssertionError, match='Isolated decoder parameter changed'):
        replay.assert_frozen()


def test_bad_weight_shape_or_mass_is_rejected():
    model, cache, W = fixture()
    with pytest.raises(ValueError, match=r'W\[T,8\]'):
        FrozenTrajectoryReplay(model, cache, W[:, :7])
    invalid = W.clone()
    invalid[0, 0] = 2.
    with pytest.raises(ValueError, match='row mass'):
        FrozenTrajectoryReplay(model, cache, invalid)

