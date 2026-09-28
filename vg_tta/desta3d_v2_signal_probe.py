"""No-update decomposition of the actual pre-gate calibration objective."""
import torch
from vg_tta.desta3d_v2_tta_pilot import configure, forward
from vg_tta.desta3d_v2_tta_objective import calibration_objective, CalibrationWeights


def measure_signals(adapter, observed, student, moments):
    initial, count = configure(adapter, 'calibration')
    assert count == 66816
    params = [p for p in adapter.parameters() if p.requires_grad]
    names = [n for n,p in adapter.named_parameters() if p.requires_grad]
    with torch.no_grad():
        teacher = forward(adapter, observed)
    output = forward(adapter, student)
    weights = CalibrationWeights(alignment=.01)
    total, terms = calibration_objective(adapter, output, teacher, initial,
        weights=weights, source_moments=moments)
    def vector(value, retain):
        grads = torch.autograd.grad(value, params, retain_graph=retain, allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                          for g,p in zip(grads, params)]).detach().cpu()
    raw = {name: vector(getattr(weights,name)*value, True) for name,value in terms.items()}
    raw['total'] = vector(total, False)
    assert all(v.shape == (66816,) and torch.isfinite(v).all() for v in raw.values())
    summed = sum(raw[n] for n in terms)
    assert torch.allclose(summed, raw['total'], atol=1e-7, rtol=2e-4)
    assert all(torch.equal(p, initial[n]) for n,p in adapter.named_parameters() if p.requires_grad)
    assert all(p.grad is None for p in adapter.parameters())
    stats = {}
    for branch in ['event','spatial']:
        x=output['branch_features_'+branch].detach().float().reshape(-1,128)
        stats[branch] = {'mean':x.mean(0).cpu(),'std':x.var(0,unbiased=False).clamp_min(1e-12).sqrt().cpu()}
    result = {'ordered_names':names,'raw_gradients':raw,'loss':float(total.detach()),
        'terms':{n:float(v.detach()) for n,v in terms.items()},'weights':weights.__dict__,
        'gradient_norms':{n:float(v.double().norm()) for n,v in raw.items()},
        'component_sum_max_error':float((summed-raw['total']).abs().max()),
        'branch_channel_moments':stats,'optimizer_steps':0,'parameters_unchanged':True}
    adapter.set_train_stage('frozen')
    return result
