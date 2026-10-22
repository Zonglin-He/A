import torch


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
