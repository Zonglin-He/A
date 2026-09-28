"""Non-mutating original-head observation and offline full-vocabulary projection."""
import torch
from vg_tta.desta3d_v2_cast_probe import TokenEvidence


class HeadEvidence(TokenEvidence):
    def __init__(self, data):
        super().__init__(data)
        self.hidden = []
        self.logits = []

    def hook(self, module, args, logits):
        assert isinstance(module, torch.nn.Linear) and module.bias is None
        assert args[0].ndim == 2 and args[0].shape[0] == logits.shape[0]
        assert not module.weight.requires_grad and module.weight.grad is None
        self.hidden.append(args[0].detach().cpu().clone())
        self.logits.append(logits.detach().cpu().clone())
        super().hook(module, args, logits)
        # None: no replacement of output or forward arithmetic.

    def finish(self):
        tokens = super().finish()
        return {'tokens': tokens, 'hidden': torch.cat(self.hidden), 'logits': torch.cat(self.logits)}


def full_reference(hidden, actual, targets, weight_chunks):
    """CPU float64 dot, complete vocabulary. Return only token evidence and errors.

    weight_chunks yields contiguous (first_row, BF16 weight rows). The original
    GPU values are never fed back into the model; rounded reference is diagnostic.
    """
    assert hidden.device.type == actual.device.type == targets.device.type == 'cpu'
    assert hidden.dtype == actual.dtype == torch.bfloat16
    assert hidden.ndim == actual.ndim == 2 and hidden.shape[0] == actual.shape[0] == targets.numel()
    assert targets.min() >= 0 and targets.max() < actual.shape[1]
    h = hidden.double(); n, vocab = actual.shape
    lse = torch.full((n,), -torch.inf, dtype=torch.float64)
    lse_round = lse.clone(); lse_actual = lse.clone()
    target = torch.empty(n, dtype=torch.float64); target_round = target.clone()
    target_actual = actual.double().gather(1, targets[:,None]).flatten()
    offset = 0; mismatch = 0; max_abs_round = 0.; max_abs_ref = 0.; squared = 0.
    for first, weight in weight_chunks:
        assert first == offset and weight.dtype == torch.bfloat16 and weight.device.type == 'cpu'
        assert weight.ndim == 2 and weight.shape[1] == h.shape[1] and weight.shape[0] > 0
        end = first + len(weight); assert end <= vocab
        z = h @ weight.double().T
        assert torch.isfinite(z).all()
        rz = z.bfloat16().double(); az = actual[:,first:end].double()
        lse = torch.logaddexp(lse, torch.logsumexp(z, 1))
        lse_round = torch.logaddexp(lse_round, torch.logsumexp(rz, 1))
        lse_actual = torch.logaddexp(lse_actual, torch.logsumexp(az, 1))
        hit = (targets >= first) & (targets < end)
        target[hit] = z[hit, targets[hit]-first]; target_round[hit] = rz[hit, targets[hit]-first]
        mismatch += int((rz != az).sum()); max_abs_round = max(max_abs_round, float((rz-az).abs().max()))
        max_abs_ref = max(max_abs_ref, float((z-az).abs().max())); squared += float(((z-az)**2).sum())
        offset = end
    assert offset == vocab, 'Incomplete vocabulary is not full CE'
    return {'token': {'target_fp64': target, 'lse_fp64': lse, 'ce_fp64': lse-target,
                     'target_round': target_round, 'lse_round': lse_round, 'ce_round': lse_round-target_round,
                     'target_actual': target_actual, 'lse_actual_fp64': lse_actual, 'ce_actual_fp64': lse_actual-target_actual},
            'full_vocab': {'rows': vocab, 'elements': n*vocab, 'round_mismatch_elements': mismatch,
                           'round_maxabs': max_abs_round, 'fp64_vs_actual_maxabs': max_abs_ref,
                           'fp64_vs_actual_squared_error': squared}}
