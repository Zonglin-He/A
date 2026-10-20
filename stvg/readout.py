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
