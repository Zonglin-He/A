"""Frozen two-offset trajectory replay without unbudgeted setup forwards.

The first *charged* ``boxes`` call must use zero content. That same forward is
the student's initial differentiable prediction and checks the saved Source/E
identity. Construction only copies the frozen decoder and cached tensors. All
arms use the complete reference gradient, restored in this private instance by
removing the one upstream interlayer detach; the Source implementation is never
edited. Importing this module does not load a model or read data or labels.
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
    """Differentiable fixed-W mapping from eight content knots to all nodes."""
    if knots.shape != (8, 256) or weights.ndim != 2 or weights.shape[1] != 8:
        raise ValueError('Expected knots[8,256] and W[T,8]')
    return weights.detach().to(device=knots.device, dtype=knots.dtype) @ knots


def frozen_model_summary(model):
    parameters = list(model.named_parameters())
    if not parameters:
        raise ValueError('The frozen Source model has no registered parameters')
    for name, parameter in parameters:
        if parameter.requires_grad or parameter.grad is not None:
            raise ValueError('Source network parameter is not frozen: ' + name)
    return dict(parameters=len(parameters), scalar_parameters=sum(p.numel() for _, p in parameters),
                all_network_parameters_frozen=True, all_network_parameter_grads_none=True)


class FrozenTrajectoryReplay:
    """One differentiable whole trajectory = two original physical offsets.

    ``evals`` includes the first zero-state training forward. There are no setup
    decoder calls. CPU operation exists only for fake-decoder unit tests; the
    runner owns the actual model device and frozen checkpoint/cache identity.
    """
    def __init__(self, model, cache, W, full_reference=True, check=None):
        if full_reference is not True:
            raise ValueError('Every output-space arm requires full_reference=True')
        self._model = model
        self.check = check if check is not None else lambda: None
        summary = frozen_model_summary(model)
        self._original_parameters = {name: (parameter, parameter._version)
                                     for name, parameter in model.named_parameters()}
        self._original_buffers = {name: (buffer, buffer._version)
                                  for name, buffer in model.named_buffers()}
        self.decoder = copy.deepcopy(model.ground_decoder.decoder).eval().requires_grad_(False)
        self.device = next(self.decoder.parameters()).device
        if any(p.dtype != torch.float32 for p in self.decoder.parameters()):
            raise ValueError('Decoder parameters must preserve native FP32 precision')
        self.inputs = transfer(cache['spatial_inputs'], self.device)
        if len(self.inputs) != 2:
            raise ValueError('Expected the original two physical offset inputs')
        self.Source = torch.as_tensor(cache['Source'], device=self.device).detach()
        self.E = torch.as_tensor(cache['E'], device=self.device, dtype=torch.float64).detach()
        if self.Source.ndim != 2 or self.Source.shape[1] != 4 or self.E.shape != self.Source.shape:
            raise ValueError('Expected complete Source and E trajectories [T,4]')
        if not torch.isfinite(self.Source).all() or not torch.isfinite(self.E).all():
            raise ValueError('Source and E trajectories must be finite')
        if not bool((self.Source[:, 2:] > 0).all()) or not bool((self.E[:, 2:] > 0).all()):
            raise ValueError('Source and E sizes must be strictly positive')
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
        # Identical isolated reference-gradient patch to the already-regressed
        # replay. Removing detach changes derivatives, not forward arithmetic.
        original = self.decoder.forward
        body = textwrap.dedent(inspect.getsource(original))
        token = 'pred_boxes = new_pred_boxes.detach()'
        if body.count(token) != 1:
            raise ValueError('Unexpected upstream reference-gradient implementation')
        changed = body.replace(token, 'pred_boxes = new_pred_boxes')
        namespace = dict(vars(sys.modules[original.__module__]))
        exec(compile(changed, '<output-opd-isolated-full-reference>', 'exec'), namespace)
        self.decoder.forward = types.MethodType(namespace['forward'], self.decoder)
        self._decoder_versions = {name: (parameter, parameter._version)
                                  for name, parameter in self.decoder.named_parameters()}
        self._decoder_buffer_versions = {name: (buffer, buffer._version)
                                         for name, buffer in self.decoder.named_buffers()}
        self._W_version = self.W._version
        self.setup_evals = self.evals = 0
        self.setup_native_offset_decoder_calls = self.native_offset_decoder_calls = 0
        self._zero_checked = False
        self.initial_checks = dict(full_reference=True, **summary,
            physical_nodes=self.n, knot_shape=[8, 256], dense_content_shape=[self.n, 256],
            direct_content_at_all_physical_nodes=True, constructor_decoder_calls=0,
            zero_identity_checked_on_first_charged_forward=False,
            W_sha256=hashlib.sha256(self.W.cpu().numpy().tobytes()).hexdigest(),
            original_forward_sha256=hashlib.sha256(body.encode()).hexdigest(),
            full_reference_forward_sha256=hashlib.sha256(changed.encode()).hexdigest(),
            reference_restore_edit='remove exactly one interlayer reference detach')
        self.assert_frozen()

    def expand(self, knots):
        if knots.device != self.device or knots.dtype != torch.float32:
            raise ValueError('Content knots must use the decoder device and FP32')
        return expand_knots(knots, self.W)

    def _require_first_zero(self, state):
        if not self._zero_checked and bool(torch.count_nonzero(state.detach())):
            raise ValueError('The first charged trajectory forward must use zero content')

    def boxes(self, knots, return_native=False):
        self._require_first_zero(knots)
        return self.boxes_from_content(self.expand(knots), return_native=return_native)

    def boxes_from_content(self, eta, return_native=False):
        if eta.shape != (self.n, 256) or eta.device != self.device or eta.dtype != torch.float32:
            raise ValueError('Expected expanded FP32 content eta[T,256] on decoder device')
        if not bool(torch.isfinite(eta).all()):
            raise FloatingPointError('Nonfinite expanded content')
        self._require_first_zero(eta)
        self.check()
        raw = []
        with torch.autocast(self.device.type, enabled=False):
            for view, original in enumerate(self.inputs):
                kwargs = dict(original)
                kwargs['query_tgt'] = original['query_tgt'] + eta[view::2, None, :]
                self.native_offset_decoder_calls += 1
                raw.append(self.decoder(**kwargs).flatten(1, 2)[-1])
            native = torch.stack([raw[p % 2][p // 2] for p in range(self.n)]).float()
            b = native.double()
            src = self.Source.to(b)
            # Preserve the old correction's exact operation order and dtype.
            out = torch.cat((self.E[:, :2] + (b[:, :2] - src[:, :2]),
                self.E[:, 2:] * torch.exp(torch.log(b[:, 2:]) - torch.log(src[:, 2:]))), -1)
        self.evals += 1
        if not torch.isfinite(out).all() or not bool((out[:, 2:] > 0).all()):
            raise FloatingPointError('Invalid corrected full trajectory')
        if not self._zero_checked:
            # Detach only the identity-check copy. The returned initial
            # prediction retains its complete reference/content autograd graph.
            if not torch.equal(native.detach().cpu(), self.Source.detach().cpu()):
                raise AssertionError('Zero-state decoder output differs from saved Source')
            if not torch.equal(out.detach(), self.E):
                raise AssertionError('Zero-state corrected trajectory differs from saved E')
            self._zero_checked = True
            self.initial_checks.update(original_native_Source_bit_exact=True,
                original_E_bit_exact=True, full_reference_forward_bit_exact=True,
                full_reference_forward_check_scope='first zero-state forward versus saved native Source',
                zero_identity_checked_on_first_charged_forward=True,
                first_identity_full_trajectory_eval=self.evals)
        return (out, native) if return_native else out

    def assert_frozen(self):
        frozen_model_summary(self._model)
        for name, (parameter, version) in self._original_parameters.items():
            if parameter._version != version:
                raise AssertionError('Original frozen network parameter changed: ' + name)
        for name, (buffer, version) in self._original_buffers.items():
            if buffer._version != version:
                raise AssertionError('Original frozen network buffer changed: ' + name)
        for name, (parameter, version) in self._decoder_versions.items():
            if parameter.requires_grad or parameter.grad is not None or parameter._version != version:
                raise AssertionError('Isolated decoder parameter changed: ' + name)
        for name, (buffer, version) in self._decoder_buffer_versions.items():
            if buffer._version != version:
                raise AssertionError('Isolated decoder buffer changed: ' + name)
        if self.W.requires_grad or self.W._version != self._W_version:
            raise AssertionError('Frozen physical interpolation weights changed')
        return True

    def counts(self):
        return dict(full_trajectory_evals=self.evals,
                    native_offset_decoder_calls=self.native_offset_decoder_calls,
                    setup_full_trajectory_evals=0,
                    setup_native_offset_decoder_calls=0)


FrozenContentReplay = FrozenTrajectoryReplay
