"""Isolated frozen decoder replay with content injected at every physical node.

The optimized parameter is eight 256-dimensional knots. A fixed evidence-built
matrix W maps those knots to all physical nodes before the two original offset
decoder calls. Importing this module never loads a model, reads inputs, or GT.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import sys
import textwrap
import types

import torch

from stvg.capture import transfer


def expand_knots(knots, weights):
    """Differentiable [8,256] -> [T,256] interpolation; W is frozen evidence."""
    if knots.shape != (8, 256) or weights.ndim != 2 or weights.shape[1] != 8:
        raise ValueError('Expected knots[8,256] and W[T,8]')
    return weights.detach().to(device=knots.device, dtype=knots.dtype) @ knots


def frozen_model_summary(model):
    """Check all network parameters; do not copy or hash the full checkpoint."""
    parameters = list(model.named_parameters())
    if not parameters:
        raise ValueError('The frozen Source model has no registered parameters')
    for name, parameter in parameters:
        if parameter.requires_grad or parameter.grad is not None:
            raise ValueError('Source network parameter is not frozen: ' + name)
    return dict(parameters=len(parameters), scalar_parameters=sum(p.numel() for _, p in parameters),
                all_network_parameters_frozen=True, all_network_parameter_grads_none=True)


class FrozenTrajectoryReplay:
    """Actual two-offset decoder, with fixed dense physical-node state mapping.

    ``boxes(knots)`` expands W internally. ``boxes_from_content(eta)`` exposes the
    same direct all-node injection for callers which have already expanded it.
    CPU operation is supported for fake-decoder engineering tests only; the
    experiment runner owns the actual Source device and frozen asset identity.
    """
    def __init__(self, model, cache, W, full_reference=False, check=None):
        self._model = model
        self.check = check if check is not None else lambda: None
        summary = frozen_model_summary(model)
        self._original_parameters = {name: (parameter, parameter._version)
                                     for name, parameter in model.named_parameters()}
        self._original_buffers = {name: (buffer, buffer._version)
                                  for name, buffer in model.named_buffers()}
        self.decoder = copy.deepcopy(model.ground_decoder.decoder).eval().requires_grad_(False)
        self.device = next(self.decoder.parameters()).device
        self.inputs = transfer(cache['spatial_inputs'], self.device)
        if len(self.inputs) != 2:
            raise ValueError('Expected the original two physical offset inputs')
        self.Source = torch.as_tensor(cache['Source'], device=self.device)
        self.E = torch.as_tensor(cache['E'], device=self.device, dtype=torch.float64)
        if self.Source.ndim != 2 or self.Source.shape[1] != 4 or self.E.shape != self.Source.shape:
            raise ValueError('Expected complete Source and E trajectories [T,4]')
        self.n = len(self.Source)
        self.positions = list(cache['planned_positions'])
        if len(self.positions) != 8 or len(set(self.positions)) != 8 or min(self.positions) < 0 or max(self.positions) >= self.n:
            raise ValueError('Expected eight distinct valid physical knot positions')
        self.W = torch.as_tensor(W, device=self.device, dtype=torch.float32).detach().clone()
        if self.W.shape != (self.n, 8) or not torch.isfinite(self.W).all():
            raise ValueError('Expected finite fixed W[T,8]')
        if bool((self.W < 0).any()) or bool((self.W.sum(-1) > 1. + 1e-6).any()):
            raise ValueError('W must be nonnegative with row mass at most one')
        for view, original in enumerate(self.inputs):
            q = original['query_tgt']
            if q.shape != (len(range(view, self.n, 2)), 1, 256) or q.dtype != torch.float32:
                raise ValueError('Cached query targets must preserve native FP32 offset geometry')
        self.setup_evals = self.evals = 0
        self.setup_native_offset_decoder_calls = self.native_offset_decoder_calls = 0
        self.initial_checks = dict(full_reference=bool(full_reference), **summary,
            physical_nodes=self.n, knot_shape=[8, 256], dense_content_shape=[self.n, 256],
            direct_content_at_all_physical_nodes=True,
            W_sha256=hashlib.sha256(self.W.cpu().numpy().tobytes()).hexdigest())
        zero = torch.zeros(8, 256, device=self.device)
        with torch.no_grad():
            old, native = self.boxes(zero, setup=True, return_native=True)
        if not torch.equal(native.cpu(), torch.as_tensor(cache['Source'])):
            raise AssertionError('Zero-state decoder output differs from saved Source')
        if not torch.equal(old, self.E):
            raise AssertionError('Zero-state corrected trajectory differs from saved E')
        self.initial_checks.update(original_native_Source_bit_exact=True, original_E_bit_exact=True)
        if full_reference:
            original = self.decoder.forward
            body = textwrap.dedent(inspect.getsource(original))
            token = 'pred_boxes = new_pred_boxes.detach()'
            if body.count(token) != 1:
                raise ValueError('Unexpected upstream reference-gradient implementation')
            changed = body.replace(token, 'pred_boxes = new_pred_boxes')
            namespace = dict(vars(sys.modules[original.__module__]))
            exec(compile(changed, '<trajectory-opd-isolated-full-reference>', 'exec'), namespace)
            self.decoder.forward = types.MethodType(namespace['forward'], self.decoder)
            with torch.no_grad():
                restored = self.boxes(zero, setup=True)
            if not torch.equal(old, restored):
                raise AssertionError('Reference-gradient restoration changed forward outputs')
            self.initial_checks.update(full_reference_forward_bit_exact=True,
                original_forward_sha256=hashlib.sha256(body.encode()).hexdigest(),
                full_reference_forward_sha256=hashlib.sha256(changed.encode()).hexdigest())
        self._decoder_versions = {name: (parameter, parameter._version)
                                  for name, parameter in self.decoder.named_parameters()}
        self.assert_frozen()

    def expand(self, knots):
        if knots.device != self.device or knots.dtype != torch.float32:
            raise ValueError('Content knots must use the decoder device and FP32')
        return expand_knots(knots, self.W)

    def boxes(self, knots, setup=False, return_native=False):
        return self.boxes_from_content(self.expand(knots), setup=setup, return_native=return_native)

    def boxes_from_content(self, eta, setup=False, return_native=False):
        if eta.shape != (self.n, 256) or eta.device != self.device or eta.dtype != torch.float32:
            raise ValueError('Expected expanded FP32 content eta[T,256] on decoder device')
        self.check()
        raw = []
        with torch.autocast(self.device.type, enabled=False):
            for view, original in enumerate(self.inputs):
                kwargs = dict(original)
                # This edits every physical node directly, rather than only
                # the eight knots and hoping temporal attention propagates it.
                kwargs['query_tgt'] = original['query_tgt'] + eta[view::2, None, :]
                if setup:
                    self.setup_native_offset_decoder_calls += 1
                else:
                    self.native_offset_decoder_calls += 1
                raw.append(self.decoder(**kwargs).flatten(1, 2)[-1])
            native = torch.stack([raw[p % 2][p // 2] for p in range(self.n)]).float()
            # Same operation order as the frozen spatial correction interface.
            b = native.double()
            src = self.Source.to(b)
            out = torch.cat((self.E[:, :2] + (b[:, :2] - src[:, :2]),
                self.E[:, 2:] * torch.exp(torch.log(b[:, 2:]) - torch.log(src[:, 2:]))), -1)
        if not torch.isfinite(out).all() or not bool((out[:, 2:] > 0).all()):
            raise FloatingPointError('Invalid corrected full trajectory')
        if setup:
            self.setup_evals += 1
        else:
            self.evals += 1
        return (out, native) if return_native else out

    def assert_frozen(self):
        frozen_model_summary(self._model)
        for name, (parameter, version) in self._original_parameters.items():
            if parameter._version != version:
                raise AssertionError('Original frozen network parameter changed: ' + name)
        for name, (buffer, version) in self._original_buffers.items():
            if buffer._version != version:
                raise AssertionError('Original frozen network buffer changed: ' + name)
        for name, (parameter, version) in getattr(self, '_decoder_versions', {}).items():
            if parameter.requires_grad or parameter.grad is not None or parameter._version != version:
                raise AssertionError('Isolated decoder parameter changed: ' + name)
        return True

    def counts(self):
        return dict(full_trajectory_evals=self.evals,
                    native_offset_decoder_calls=self.native_offset_decoder_calls,
                    setup_full_trajectory_evals=self.setup_evals,
                    setup_native_offset_decoder_calls=self.setup_native_offset_decoder_calls)


# The explicit name above is preferred; this alias eases isolated fit adapters.
FrozenContentReplay = FrozenTrajectoryReplay
