"""Bounded post-pilot revision: align temporal variation, not global video means."""
import torch
from .spatial_affine import compute_source_statistics, _fit


def within_video_statistics(source_features, last_weight):
    centered = [z.float() - z.float().mean(0, keepdim=True) for z in source_features]
    stats = compute_source_statistics(centered, last_weight)
    stats['reference'] = 'source within-video centered covariance; source inputs only'
    return stats


def moment_components(calibrated, stats, rank):
    k = min(rank, stats['effective_rank_above_floor'])
    if k < 1:
        zero = calibrated.sum() * 0.
        return zero, zero
    projected = (calibrated.float() - stats['mean'].to(calibrated.device)) @ stats['vectors'][:, :k].to(calibrated.device)
    mean = projected.mean(0)
    target_var = projected.var(0, unbiased=False) + stats['variance_floor']
    source_var = stats['values'][:k].to(calibrated.device) + stats['variance_floor']
    weights = stats['importance'][:k].to(calibrated.device)
    variance = .5 * (target_var/source_var + source_var/target_var - 2.)
    mean_term = .5 * mean.square() * (1./source_var + 1./target_var)
    return (weights*mean_term).sum()/weights.sum(), (weights*variance).sum()/weights.sum()


def fit_variance_affine(head, features, stats, config):
    """Legal loss; hold bias at zero because variance is translation-invariant.

    The shared optimizer implementation expects both vectors. The zero gradient
    hook makes shift immutability explicit and avoids roundoff-sized bias updates.
    Only the 256 scales are effectively adapted; do not count it as 512 updates.
    """
    if stats.get('labels_used') is not False or config.objective != 'subspace':
        raise ValueError('requires label-free source-subspace statistics')
    hook = head.shift.register_hook(lambda grad: torch.zeros_like(grad))
    try:
        result = _fit(head, features, config, lambda: moment_components(head.calibrate(features[-1]), stats, config.rank)[1])
    finally:
        hook.remove()
    assert torch.count_nonzero(head.shift) == 0
    result.update(effective_update_parameters=head.scale_delta.numel(), shift_fixed_zero=True,
                  objective='variance only', gt_used=False)
    return result
