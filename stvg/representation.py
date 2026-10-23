"""Fractional-area target-context pooling of original full-caption fused motion memory.

No ASA classifier/TTS score, RGB re-encoding, new learned projection or optimizer.
"""
import numpy as np
def weights(box, height, width, valid):
    """Fractional grid coverage of the fixed 1.5 context; unchanged coordinates."""
    b = np.asarray(box, dtype=np.float64).copy()
    b[2:] *= 1.5
    lo, hi = b[:2] - b[2:] / 2, b[:2] + b[2:] / 2
    xx, yy = np.arange(width) / width, np.arange(height) / height
    dx = np.maximum(0, np.minimum(xx + 1 / width, hi[0]) - np.maximum(xx, lo[0])) * width
    dy = np.maximum(0, np.minimum(yy + 1 / height, hi[1]) - np.maximum(yy, lo[1])) * height
    out = dy[:, None] * dx[None, :] * np.asarray(valid, bool)
    fallback = not bool(out.sum() > 0)
    if fallback:
        out = np.asarray(valid, dtype=np.float64).copy()
    assert out.sum() > 0
    return out, fallback

def pool(features, support):
    a = np.asarray(features, dtype=np.float64)
    w = np.asarray(support, dtype=np.float64)
    assert a.shape[:2] == w.shape and np.isfinite(a).all()
    return np.einsum('hwc,hw->c', a, w) / w.sum()

def extract(cap,aux):
    ids=list(map(int,cap['frame_ids']));out=np.empty((len(ids),256));stats=[]
    cache=cap.get('temporal_cache',cap.get('native_hidden'))
    assert cache is not None, 'Original native time-decoder memory required'
    for off in cache['offsets']:
        h,w=map(int,off['grid_size']);mem=off['inputs']['encoded_feature'];mask=off['inputs']['encoded_mask']
        assert mem.ndim==3 and mem.shape[0]>h*w and mem.shape[2]==256
        visual=mem[-h*w:].permute(1,0,2).numpy().reshape(-1,h,w,256)
        valid=(~mask[:,-h*w:]).numpy().reshape(-1,h,w)
        assert off['frame_ids']==ids[off['offset']::2] and not mask.any(), 'Preserve validated native grid/mask layout'
        for j in range(len(visual)):
            t=off['offset']+2*j;support,fallback=weights(aux[t],h,w,valid[j])
            out[t]=pool(visual[j],support)
            stats.append(dict(frame=t,fallback=bool(fallback),area=float(support.sum()),grid=h*w))
    assert np.isfinite(out).all();return out,stats
