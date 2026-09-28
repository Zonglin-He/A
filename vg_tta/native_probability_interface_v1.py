"""Native STVG entropy interface, not a DeCoTA objective or complete TTA port.

Both valid endpoint distributions and the actual final actionness *logits*
participate. TA-STVG pipeline.py and criterion.py define the latter as logits.
Padding cannot contribute. MEMO requires identical physical grids between
photometric views; different offsets are independent distributions, not classes
that may be averaged by index. No GT, pseudo boxes, prior or external expert.
"""
from dataclasses import dataclass
import math
import torch
from torch import Tensor


@dataclass(frozen=True)
class NativeOutput:
    sted: Tensor                 # [T,2], FP32/64 logits
    actionness_logits: Tensor    # [T] or [T,1], exactly one sigmoid
    valid: Tensor                # [T], True means valid
    frame_ids: tuple[int, ...]

    def probabilities(self):
        z, a, m = self.sted, self.actionness_logits.reshape(-1), self.valid
        if z.ndim != 2 or z.shape[1] != 2 or a.shape != (z.shape[0],) or m.shape != a.shape:
            raise ValueError("Invalid native probability tensor shape")
        if m.dtype != torch.bool or len(self.frame_ids) != z.shape[0] or not bool(m.any()):
            raise ValueError("Invalid valid mask/grid")
        if tuple(sorted(set(self.frame_ids))) != self.frame_ids:
            raise ValueError("Physical frame IDs must be increasing and unique")
        if not bool(torch.isfinite(z[m]).all() and torch.isfinite(a[m]).all()):
            raise ValueError("Nonfinite valid native logits")
        z, a = z[m].float(), a[m].float()
        return z.softmax(0), a.sigmoid()


def categorical_entropy(p):
    return -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(0).sum()


def binary_entropy(p):
    tiny = torch.finfo(p.dtype).tiny
    return -(p*p.clamp_min(tiny).log()+(1-p)*(1-p).clamp_min(tiny).log()).mean()


def native_entropy(output: NativeOutput):
    """H(start)+H(end)+mean h(actionness), in nats, no hidden 1/2 factor."""
    p, a = output.probabilities()
    return categorical_entropy(p)+binary_entropy(a)


def entropy_ceiling(output: NativeOutput):
    output.probabilities()  # validate
    return 2*math.log(int(output.valid.sum()))+math.log(2)


def memo_native_marginal(views):
    if not views:
        raise ValueError("MEMO needs at least one view")
    base = views[0]
    for view in views[1:]:
        if base.frame_ids != view.frame_ids or not torch.equal(base.valid, view.valid):
            raise ValueError("MEMO view physical correspondence/mask mismatch")
    pp, aa = zip(*(v.probabilities() for v in views))
    return categorical_entropy(torch.stack(pp).mean(0))+binary_entropy(torch.stack(aa).mean(0))


def offsets_entropy(outputs):
    if not outputs:
        raise ValueError("No native offsets")
    return torch.stack([native_entropy(o) for o in outputs]).mean()
