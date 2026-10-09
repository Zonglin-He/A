"""Authorized finite-native-logit chart extension, process-local and traceable.

Original fitter and decoder files remain immutable. Interior coordinates use
the exact original logit(box) operation. Only a rounded sigmoid endpoint uses
the finite pre-sigmoid value from the SAME decoder call; no epsilon or clamp.
"""
from contextlib import contextmanager
import sys
import torch
import numpy as np
from vg_tta.decota_spatial_opd_boundary_chart_proposal003 import proposed_coordinates

REVISION = 'native_logit_chart_revision003'


class Installation:
    def __init__(self, qualification_vjp=False, failure_callback=None):
        import vg_tta.decota_spatial_opd_tunable_v1 as core
        self.core = core
        self.old_values = core.TrickReplay.values
        self.old_mean = core.mean_coordinates
        self.old_fit = core.fit
        self.qualification_vjp = qualification_vjp
        self.failure_callback = failure_callback
        self.contexts = []
        self.last_vjp_checks = []
        owner = self

        def values(replay):
            outputs = []
            hook = replay.base.decoder.decoder.bbox_embed.register_forward_hook(
                lambda module, args, output: outputs.append(output))
            try:
                result = owner.old_values(replay)
            finally:
                hook.remove()
            assert len(outputs) == 12, 'Unexpected native decoder head topology'
            raw = torch.stack([outputs[5 + (i % 2) * 6][i // 2, 0]
                               for i in range(len(replay.ids))]).float()
            assert torch.equal(raw.sigmoid(), result['boxes']), 'Raw/value call binding failed'
            replay._chart003_raw = raw
            replay._chart003_boxes = result['boxes']
            replay._chart003_heads = (outputs[5], outputs[11])
            return result

        def mean(boxes):
            frame = sys._getframe(1)
            # The original source-box roundtrip outside fit retains its guard.
            if frame.f_code.co_name != 'fit' or not frame.f_code.co_filename.endswith('decota_spatial_opd_tunable_v1.py'):
                return owner.old_mean(boxes)
            loc = frame.f_locals
            replay = loc['rp']
            positions = loc['positions']
            assert torch.equal(boxes, replay._chart003_boxes[positions])
            raw = replay._chart003_raw[positions]
            value, boundary = proposed_coordinates(boxes, raw)
            phase = 'before' if torch.is_grad_enabled() else 'after'
            assert owner.contexts
            owner.contexts[-1]['trace'].append(dict(round=loc['k'], phase=phase,
                boxes=boxes.detach().cpu(), raw_logits=raw.detach().cpu(),
                mean=value.detach().cpu(), boundary=boundary.detach().cpu(),
                same_native_call_verified=True))
            if owner.qualification_vjp and phase == 'before':
                heads = replay._chart003_heads
                gradients = torch.autograd.grad(value.sum(), heads, retain_graph=True)
                expected = [torch.zeros_like(h) for h in heads]
                for pos in positions:
                    expected[pos % 2][pos // 2, 0] = 1
                error = max(float((g - e).abs().max()) for g, e in zip(gradients, expected))
                assert all(torch.isfinite(g).all() for g in gradients) and error < 2e-6
                owner.contexts[-1]['vjp'].append(dict(round=loc['k'],
                    native_head_VJP_max_error=error,
                    native_head_coordinates=sum(g.numel() for g in gradients),
                    independent_full_decoder_Jacobian=False))
            return value

        def fit(*args, **kwargs):
            context = dict(trace=[], vjp=[])
            owner.contexts.append(context)
            try:
                result = owner.old_fit(*args, **kwargs)
                result['numerical_chart'] = dict(revision=REVISION,
                    interior_formula='original_torch_logit',
                    endpoint_formula='same_call_finite_native_logit', trace=context['trace'])
                owner.last_vjp_checks = context['vjp']
                return result
            except BaseException as exc:
                if owner.failure_callback is not None:
                    owner.failure_callback(exc, context)
                raise
            finally:
                owner.contexts.pop()

        self.values = values
        self.mean = mean
        self.fit = fit
        self.install()

    def install(self):
        self.core.TrickReplay.values = self.values
        self.core.mean_coordinates = self.mean
        self.core.fit = self.fit

    def uninstall(self):
        self.core.TrickReplay.values = self.old_values
        self.core.mean_coordinates = self.old_mean
        self.core.fit = self.old_fit

    @contextmanager
    def original(self):
        self.uninstall()
        try:
            yield self.old_fit
        finally:
            self.install()


def chart_audit(fit):
    """Independent float64 chart/readout binding plus exact stored state links.

CPU elementary sigmoid/log implementations need not be bitwise CUDA kernels.
Same-call GPU equality is checked before each chart construction. Float64
recomputation uses explicit absolute/relative float32 tolerances here.
"""
    chart = fit['numerical_chart']
    assert chart['revision'] == REVISION
    assert chart['interior_formula'] == 'original_torch_logit'
    assert chart['endpoint_formula'] == 'same_call_finite_native_logit'
    positions = fit['positions']
    expected = [(k, phase) for k in range(fit['config']['steps'])
                for phase in ('before', 'after')] if positions else []
    trace = chart['trace']
    assert [(r['round'], r['phase']) for r in trace] == expected
    eps = np.finfo(np.float32).eps
    max_readout = max_interior = 0.
    endpoints = 0
    for r in trace:
        b, raw, mean = [r[k].numpy().astype(np.float64) for k in ('boxes', 'raw_logits', 'mean')]
        assert b.shape == raw.shape == mean.shape == (len(positions), 4)
        assert all(np.isfinite(v).all() for v in (b, raw, mean))
        assert ((b >= 0) & (b <= 1)).all()
        mask = (b == 0) | (b == 1)
        assert np.array_equal(mask, r['boundary'].numpy()) and r['same_native_call_verified'] is True
        assert np.array_equal(mean[mask], raw[mask])
        endpoints += int(mask.sum())
        interior = ~mask
        if interior.any():
            reference = np.log(b[interior]) - np.log1p(-b[interior])
            discrepancy = abs(mean[interior] - reference)
            assert np.all(discrepancy <= 4 * eps * np.maximum(1., abs(reference)))
            max_interior = max(max_interior, float(discrepancy.max()))
        # Stable independent sigmoid handles large finite logits without overflow.
        e = np.exp(-abs(raw))
        decoded = np.where(raw >= 0, 1 / (1 + e), e / (1 + e))
        discrepancy = abs(decoded - b)
        assert np.all(discrepancy <= 2 * eps)
        max_readout = max(max_readout, float(discrepancy.max(initial=0.)))
        k = r['round']; phase = r['phase']
        path = fit['path'][k if phase == 'before' else k + 1]['boxes'][positions]
        bound_mean = fit['rounds'][k]['mean_' + phase]
        assert torch.equal(r['boxes'], path) and torch.equal(r['mean'], bound_mean)
    return dict(status='pass', chart_calls=len(trace), endpoint_coordinates=endpoints,
        max_float64_interior_formula_error=max_interior,
        max_float64_native_sigmoid_readout_error=max_readout,
        interior_logit_relative_absolute_tolerance=4 * eps,
        native_sigmoid_absolute_tolerance=2 * eps,
        same_call_GPU_equality_checked=True, no_clamp_or_box_readout_change=True,
        independent_full_decoder_Jacobian=False)


def audit(fit, expert):
    from vg_tta.decota_spatial_opd_precision_audit_revision002 import audit as previous
    result = previous(fit, expert)
    checked = chart_audit(fit)
    return dict(result, revision=REVISION,
        previous_precision_audit_revision='precision_audit_revision002',
        authorized_chart_extension=checked, fixed_method_hyperparameters_unchanged=True)
