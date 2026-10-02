"""Independent CPU reconstructions of H-lite support and geometry."""
import numpy as np

def weights_from_candidates(candidates, ids):
    w = np.zeros(len(ids), dtype=np.float64)
    for c in candidates:
        first, last = c['indices']
        assert c['physical_interval'] == [ids[first], ids[last]+1]
        assert 0 <= first < last < len(ids)
        for t in range(first, last+1): w[t] += 1/len(candidates)
    return w

def frame_geometry(p, q, coeff):
    p = np.asarray(p, float)[None]; q = np.asarray(q, float)
    a,b = p[...,:2]-p[...,2:]/2, p[...,:2]+p[...,2:]/2
    c,d = q[...,:2]-q[...,2:]/2, q[...,:2]+q[...,2:]/2
    inter = np.maximum(np.minimum(b,d)-np.maximum(a,c),0).prod(-1)
    union = p[...,2:].prod(-1)+q[...,2:].prod(-1)-inter
    enc = (np.maximum(b,d)-np.minimum(a,c)).prod(-1)
    return coeff[0]*abs(p-q).sum(-1)+coeff[1]*(1-inter/union+(enc-union)/enc)

def geometry(p, q, coeff, w=None):
    d = frame_geometry(p, q, coeff)
    return d.mean(-1) if w is None else (d*np.asarray(w)).sum(-1)/np.sum(w)

def weighted_rewards(boxes, expert, valid, weights):
    v = np.asarray(valid, bool); w = np.asarray(weights)[v]
    if w.sum() == 0: return None
    e = np.asarray(expert, dtype=float)[v]; output=[]
    for box in boxes:
        b=np.asarray(box, dtype=float)[v]
        bmin,bmax=b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2
        emin,emax=e[:,:2]-e[:,2:]/2,e[:,:2]+e[:,2:]/2
        # Expert caches and student boxes use normalized cxcywh.
        inter=np.maximum(np.minimum(bmax,emax)-np.maximum(bmin,emin),0).prod(-1)
        union=b[:,2:].prod(-1)+e[:,2:].prod(-1)-inter
        iou=inter/np.maximum(union,1e-12)
        output.append(float(np.dot(w,iou)/w.sum()))
    return np.asarray(output)
