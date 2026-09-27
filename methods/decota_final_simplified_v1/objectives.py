"""The locked spatial C and temporal D algebra; no alternative objectives.

Reduction order and native MAP dtype/tie-breaking are deliberately preserved.
Only static indices/targets are cached. No AMP, batching, or decoder substitution.
"""

import torch
import torch.nn.functional as F


def native_view_indices(z):
    if z.ndim != 3 or z.shape[0] != 1 or z.shape[-1] != 2:
        raise ValueError("Expected [1,T,2] temporal logits")
    if not torch.isfinite(z).all():
        raise ValueError("Nonfinite temporal logits")
    n = z.shape[1]
    if n < 2:
        raise ValueError("The native decoder requires start < end")
    mask = (torch.ones(n, n, dtype=torch.float32) * -1e32).tril(0).to(z.device)
    score = mask + (z[:, :, 0].log_softmax(1).unsqueeze(2)
                    + z[:, :, 1].log_softmax(1).unsqueeze(1))[0]
    return divmod(int(score.flatten().max(0)[1]), n)


def prediction(logits, boxes, records, ids):
    z = [x.detach().cpu().clone() for x in logits]
    raw = [list(native_view_indices(x)) for x in z]
    intervals = [[r["frame_ids"][s], r["frame_ids"][e] + 1]
                 for r, (s, e) in zip(records, raw)]
    extent = [ids.index(min(x[0] for x in intervals)),
              ids.index(max(x[1] for x in intervals) - 1)]
    return dict(logits=z, boxes=boxes.detach().cpu().clone(),
                indices=extent, physical_interval=[ids[extent[0]], ids[extent[1]] + 1],
                raw_offset_indices=raw, raw_physical_intervals=intervals)


def time_cells(frame_ids, device=None):
    f = torch.as_tensor(frame_ids, dtype=torch.float64, device=device)
    if len(f) < 2 or not (f[1:] > f[:-1]).all():
        raise ValueError("Expected increasing physical frame IDs")
    edges = torch.cat((f[:1], (f[:-1] + f[1:]) / 2, f[-1:] + 1))
    widths = edges[1:] - edges[:-1]
    return edges, widths / widths.sum()


def legal_logp(z, ij=None):
    z = z.reshape(-1, 2).double()
    if ij is None:
        ij = torch.triu_indices(len(z), len(z), 1, device=z.device)
    scores = z[ij[0], 0] + z[ij[1], 1]
    return scores - torch.logsumexp(scores, 0), ij


@torch.no_grad()
def project(logits, actions, records, config):
    if len(logits) != 2 or len(actions) != 2 or len(records) != 2:
        raise ValueError("Exactly two original temporal offsets are required")
    offsets = []
    for z, raw, record in zip(logits, actions, records):
        lp, ij = legal_logp(z)
        x = raw.detach().clone().reshape(-1).float().double()
        if len(x) != z.shape[1] or not torch.isfinite(x).all():
            raise ValueError("Invalid final dense actionness")
        median = torch.quantile(x, .5)
        mad = torch.quantile((x - median).abs(), .5)
        u = (x - config.center_fraction * median) / (mad + config.epsilon)
        log_a, log_not_a = F.logsigmoid(u), F.logsigmoid(-u)
        edges, weight = time_cells(record["frame_ids"], lp.device)
        q = weight * (log_a - log_not_a)
        prefix = torch.cat((q.new_zeros(1), q.cumsum(0)))
        cost = -(weight * log_not_a).sum() - (prefix[ij[1] + 1] - prefix[ij[0]])
        score = -cost + config.prior_weight * lp.detach()
        at = int(score.argmax())
        s, e = [int(v) for v in ij[:, at]]
        offsets.append(dict(ij=ij, logp0=lp.detach().clone(), score=score.detach(),
                            cost=cost.detach(), target=at, raw_logits=x,
                            standardized_logits=u, omega=weight, cell_edges=edges,
                            target_indices=[s, e],
                            interval=[record["frame_ids"][s], record["frame_ids"][e] + 1]))
    return dict(offsets=offsets, teacher_frozen=True, GT_online=False)


def temporal_loss(logits, evidence, margin=.2):
    nlls, hinges = [], []
    for z, target in zip(logits, evidence["offsets"]):
        lp, _ = legal_logp(z, target["ij"])
        at = target["target"]
        nlls.append(-lp[at])
        if len(lp) > 1:
            other = lp.detach().clone()
            other[at] = -torch.inf
            competitor = other.argmax()
            hinges.append((margin - (lp[at] - lp[competitor])).clamp_min(0))
        else:
            hinges.append(lp.new_zeros(()))
    # Fixed gamma=1, beta=0. The teacher's prior_weight=.1 is NOT removed.
    return torch.stack(nlls).mean() + torch.stack(hinges).mean()


def xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - .5 * w, cy - .5 * h,
                        cx + .5 * w, cy + .5 * h), -1)


def generalized_iou(pred, target):
    p, t = xyxy(pred), xyxy(target)
    lt, rb = torch.maximum(p[..., :2], t[..., :2]), torch.minimum(p[..., 2:], t[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    pa = (p[..., 2:] - p[..., :2]).clamp(min=0).prod(-1)
    ta = (t[..., 2:] - t[..., :2]).clamp(min=0).prod(-1)
    union = (pa + ta - inter).clamp(min=1e-7)
    enclosing = (torch.maximum(p[..., 2:], t[..., 2:])
                 - torch.minimum(p[..., :2], t[..., :2])).clamp(min=0).prod(-1).clamp(min=1e-7)
    return inter / union - (enclosing - union) / enclosing


class SpatialLoss:
    """Create the same four-observation tensors once, not on every Adam step."""
    def __init__(self, anchors, boxes):
        self.empty = not anchors
        self.positions = torch.tensor([a["position"] for a in anchors],
                                      dtype=torch.long, device=boxes.device)
        self.targets = torch.tensor([a["box"] for a in anchors],
                                    dtype=boxes.dtype, device=boxes.device).reshape(-1, 4)
        if any(a.get("weight", 1.) != 1. for a in anchors):
            raise ValueError("The fixed method uses unit anchor weights")
        self.weights = torch.ones(len(anchors), dtype=boxes.dtype, device=boxes.device)

    def __call__(self, boxes):
        if self.empty:
            return boxes.sum() * 0
        pred = boxes[self.positions]
        d = 5 * (pred - self.targets).abs().sum(-1) + 2 * (1 - generalized_iou(pred, self.targets))
        return (self.weights * d).sum() / 4
