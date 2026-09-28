"""Read-only evidence helpers; no alternative injection, optimizer or decoder."""
import io
import torch
from torch.nn import functional as F


class TokenEvidence:
    """Observe original joint_loss lm_head chunks without changing their output."""
    def __init__(self, data):
        labels = data['labels']
        self.keep = torch.nonzero(labels[0, 1:] != -100).flatten()
        self.targets = labels[0, self.keep + 1]
        self.ntp = self.keep + 1 < data['ptd_prefix_lengths'][0]
        self.offset = 0
        self.parts = []

    def hook(self, module, args, logits):
        n = logits.shape[0]
        assert 0 < n <= 32 and self.offset + n <= len(self.targets)
        targets = self.targets[self.offset:self.offset+n]
        z = logits.detach().float()
        assert z.ndim == 2 and torch.isfinite(z).all()
        target = z.gather(1, targets[:, None]).squeeze(1)
        self.parts.append({
            'target_logit': target.cpu(),
            'logsumexp': torch.logsumexp(z, dim=-1).cpu(),
            'cross_entropy': F.cross_entropy(z, targets, reduction='none').cpu(),
            'chunk_size': n,
        })
        self.offset += n
        # Returning None is important: the real output is unchanged.

    def finish(self):
        assert self.offset == len(self.targets)
        return {'positions': (self.keep+1).cpu(), 'targets': self.targets.cpu(),
                'ntp': self.ntp.cpu(), 'chunks': self.parts}


def cast_difference(pre0, post0, pre1, post1):
    assert pre0.dtype == pre1.dtype == torch.float32
    assert post0.dtype == post1.dtype == torch.bfloat16
    pre0, pre1 = pre0.reshape(-1), pre1.reshape(-1)
    post0, post1 = post0.reshape(-1), post1.reshape(-1)
    assert pre0.shape == pre1.shape == post0.shape == post1.shape
    assert torch.isfinite(pre0).all() and torch.isfinite(pre1).all()
    assert torch.equal(pre0.to(post0.dtype), post0)
    assert torch.equal(pre1.to(post1.dtype), post1)
    delta = pre1 - pre0
    idx = torch.nonzero(post0 != post1).flatten()
    assert len(delta) < 2**31
    sparse = {'indices': idx.int(), 'before': post0[idx], 'after': post1[idx],
              'pre_before': pre0[idx], 'pre_after': pre1[idx]}
    pd = (post1[idx].float() - post0[idx].float()).double()
    dd = delta.double()
    stats = {
        'elements': len(delta), 'continuous_nonzero': int((delta != 0).sum()),
        'postcast_nonzero': len(idx),
        'continuous_changed_postcast_unchanged': int(((delta != 0) & (post0 == post1)).sum()),
        'continuous_L2': float(dd.norm()), 'postcast_L2': float(pd.norm()),
        'continuous_maxabs': float(dd.abs().max()), 'postcast_maxabs': float(pd.abs().max()) if len(idx) else 0.,
        'postcast_over_continuous_L2': float(pd.norm()/dd.norm()) if dd.norm() else None,
        'postcast_dot_continuous': float(pd.dot(dd[idx])),
        'endpoint_full_cast_exact': True,
    }
    return {'precast_delta': delta, 'sparse_postcast': sparse, 'stats': stats}


def serialize_with_guard(payload, *, used_bytes, free_bytes, cap_bytes, reserve_bytes):
    """Check actual bytes before any disk write; never truncate support."""
    buf = io.BytesIO()
    torch.save(payload, buf)
    size = buf.tell()
    assert used_bytes + size + 1_000_000 <= cap_bytes, 'new storage cap including metadata buffer'
    assert free_bytes - size - 1_000_000 >= reserve_bytes, '8 GiB reserve including metadata buffer'
    return buf.getvalue()
