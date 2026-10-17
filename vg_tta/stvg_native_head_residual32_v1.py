"""Zero initialized native head adapters; frozen upstream and no output box writeback."""
from contextlib import contextmanager
import inspect
import numpy as np
import torch
from methods.decota_final_simplified_v1.tensors import ParameterState


class TemporalHeadResidual(torch.nn.Module):
    """Existing 256x2 residual represented at the input of the native temporal head.

    On this backbone this is EXACTLY the previous 512-parameter interface. The
    NumPy float64 matmul preserves the previously sealed fitter/readout ABI;
    differentiable fitting remains the original independently audited fitter.
    The zero-valued autograd term preserves its exact affine derivative while
    retaining the legacy NumPy float64 forward rounding on this backbone.
    """
    def __init__(self, head, theta):
        super().__init__()
        self.head = head
        self.weight = torch.nn.Parameter(torch.as_tensor(theta, dtype=torch.float64,
                                         device=next(head.parameters()).device).clone())
        assert self.weight.shape == (256, 2)

    def forward(self, x):
        y = self.head(x)
        a = x.detach().cpu().double().numpy() @ self.weight.detach().cpu().numpy()
        exact = torch.from_numpy(a).to(y)
        differentiable = (x.double() @ self.weight).to(y)
        return y + exact + (differentiable-differentiable.detach())


class BoxHeadResidualReplay(ParameterState):
    """Residual at final Linear input, only final layer of final decoder pass.

    No bias: z_phi = z_source + h @ A, A=[in_features,4]. The adapter is
    shared across ALL frames. Earlier iterative reference generation is frozen.
    """
    def __init__(self, base):
        self.base = base
        head = base.decoder.decoder.bbox_embed.layers[-1]
        assert isinstance(head, torch.nn.Linear) and head.out_features == 4
        self.weight = torch.nn.Parameter(head.weight.new_zeros((head.in_features, head.out_features)))
        self.named = [('spatial.native_box_head_residual', self.weight)]
        self.initial = self.state()
        self.shape = dict(in_features=head.in_features, out_features=head.out_features,
                          parameters=self.weight.numel(), bias=False,
                          path='ground_decoder.decoder.bbox_embed.layers[-1] input')
        self.last_trace = []

    @contextmanager
    def installed(self, full=False):
        head = self.base.decoder.decoder.bbox_embed.layers[-1]
        layers = len(self.base.decoder.decoder.layers)
        calls = 0
        trace = []
        def output(module, args, y):
            nonlocal calls
            calls += 1
            # Full native replay runs both passes for each original offset.
            active = calls % (2*layers if full else layers) == 0
            if active:
                x = args[0]
                residual = x @ self.weight
                trace.append(dict(call=calls, input_shape=list(x.shape),
                                  residual_norm=float(residual.detach().norm())))
                return y + residual
            return y
        hook = head.register_forward_hook(output)
        try:
            yield
        finally:
            hook.remove()
            assert calls == (4 if full else 2)*layers, calls
            assert len(trace) == 2
            self.last_trace = trace

    def values(self):
        with self.installed():
            return self.base.values()

    def full_values(self):
        with self.installed(full=True):
            return self.base.reference_values(capture=False)


def fit(base, initial, anchors, ids, key, cfg, settings, qualification=False):
    """Only the parameter interface changes; original loss/Adam/response/rollback."""
    from scripts import stvg_spatial_lite_fit_v1 as original
    source = inspect.getsource(original.fit)
    old = 'rp, selected = core._trainable_replay(base, list(ids), key, "joint")'
    assert source.count(old) == 1
    source = source.replace(old, 'rp = BoxHeadResidualReplay(base); selected = rp.named')
    old = 'sum(parameter.numel() for _, parameter in selected) != 1792'
    assert source.count(old) == 1
    source = source.replace(old, 'sum(parameter.numel() for _, parameter in selected) != rp.weight.numel()')
    namespace = dict(original.__dict__, BoxHeadResidualReplay=BoxHeadResidualReplay)
    exec(compile(source, __file__, 'exec'), namespace)
    return namespace['fit'](base, initial, anchors, ids, key, cfg, 'M2', settings,
                            qualification=qualification, recipe='lite_fullstep', response_scope='full_grid')


def cpu_contracts():
    # Independent autograd/finite differences for shared native residual and
    # original Gaussian likelihood, plus exact zero Source recovery.
    g = torch.Generator().manual_seed(20261017)
    h = torch.randn(9, 7, generator=g, dtype=torch.float64)
    z = torch.randn(9, 4, generator=g, dtype=torch.float64)
    a = torch.zeros(7, 4, dtype=torch.float64, requires_grad=True)
    assert torch.equal((z+h@a).sigmoid(), z.sigmoid())
    target = torch.randn(2, 4, generator=g, dtype=torch.float64)
    loss = ((z+h@a)[[1, 6]]-target).square().sum()
    grad = torch.autograd.grad(loss, a)[0]
    expected = 2*h[[1, 6]].T@((z+h@a)[[1, 6]]-target)
    assert torch.allclose(grad, expected, atol=1e-12, rtol=1e-12)
    direction = grad/grad.norm()
    epsilon = 1e-5
    fd = ((((z+h@(epsilon*direction))[[1,6]]-target).square().sum()
           -((z+h@(-epsilon*direction))[[1,6]]-target).square().sum())/(2*epsilon))
    assert abs(float(fd-grad.norm())) < 1e-8
    changed = ((z+h@(-.001*grad)).sigmoid()-z.sigmoid()).norm(dim=-1)
    assert bool((changed[[0,2,3,4,5,7,8]] > 0).all())
    return dict(status='pass', count=4, zero_recovery=True, shared_gradient_exact=True,
                finite_difference=True, nonanchor_influence=True, GPU_used=False, GT_read=False)
