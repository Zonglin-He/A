import torch
from vg_tta.tastvg_decota_critic_p0_v1 import CriticEnergy, aligned_iou


def expert(boxes, scores):
    return {'observations': {('original', 0): {'probe': {
        'boxes': torch.tensor(boxes, dtype=torch.float64).reshape(-1, 4),
        'target_scores': torch.tensor(scores, dtype=torch.float64), 'accepted': False},
        'receipt': {'context_active': True}}}}


def test_single_proposal_is_negative_iou():
    x = torch.tensor([[.46, .52, .28, .3]], dtype=torch.float64, requires_grad=True)
    e = [[.5, .5, .32, .28]]
    objective = CriticEnergy(expert(e, [.6]), x)
    assert torch.allclose(objective(x), -aligned_iou(x[0], torch.tensor(e, dtype=x.dtype))[0])


def test_permutation_and_constant_score_shift():
    x = torch.tensor([[.46, .52, .28, .3]], dtype=torch.float64)
    e = [[.5, .5, .32, .28], [.4, .45, .26, .22]]
    a = CriticEnergy(expert(e, [.2, .4]), x)(x)
    assert torch.allclose(a, CriticEnergy(expert(e[::-1], [.4, .2]), x)(x))
    assert torch.allclose(a, CriticEnergy(expert(e, [.5, .7]), x)(x))


def test_coordinate_gradient_matches_finite_difference():
    x = torch.tensor([[.46, .52, .28, .3]], dtype=torch.float64, requires_grad=True)
    objective = CriticEnergy(expert([[.5, .5, .32, .28], [.4, .45, .26, .22]], [.4, .7]), x)
    assert torch.autograd.gradcheck(objective, (x,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_empty_is_exact_noop():
    x = torch.tensor([[.5, .5, .2, .2]], dtype=torch.float64, requires_grad=True)
    objective = CriticEnergy(expert([], []), x)
    assert objective.empty and float(objective(x)) == 0
    assert torch.count_nonzero(torch.autograd.grad(objective(x), x)[0]) == 0


def test_disjoint_iou_has_zero_gradient():
    x = torch.tensor([[.1, .1, .05, .05]], dtype=torch.float64, requires_grad=True)
    objective = CriticEnergy(expert([[.8, .8, .05, .05]], [.6]), x)
    assert float(objective(x)) == 0
    assert torch.count_nonzero(torch.autograd.grad(objective(x), x)[0]) == 0
    assert objective.diagnostics(x)[0]['all_zero_overlap']


def test_multimodal_objective_stays_finite():
    x = torch.tensor([[.46, .52, .28, .3]], dtype=torch.float64, requires_grad=True)
    objective = CriticEnergy(expert([[.5, .5, .32, .28], [.9, .9, .1, .1]], [.35, .99]), x)
    assert torch.isfinite(objective(x))
    assert torch.isfinite(torch.autograd.grad(objective(x), x)[0]).all()
