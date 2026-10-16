"""Fixed sampled-grid temporal corruption and candidate-only critic bridge."""
import hashlib
import math
import numpy as np

FAMILIES = ['black', 'motion_blur', 'occlusion', 'overexposure', 'packet_loss']
LEVELS = [1, 5, 10]

def positions(ids, interval, percentage):
    eligible = [i for i, f in enumerate(ids) if interval[0] <= f < interval[1]]
    assert eligible
    count = min(len(eligible), max(1, math.ceil(len(eligible)*percentage/100)))
    start = (len(eligible)-count)//2
    return eligible[start:start+count], len(eligible)

def corrupt_local(frames, selected, family, source):
    """Own deterministic STVG operators; not a byte replica of THUMOS14-C."""
    import cv2
    out = frames.copy(); h,w = frames.shape[1:3]
    seed = int(hashlib.sha256(('C05|'+source+'|'+family).encode()).hexdigest()[:16],16)
    rng = np.random.default_rng(seed)
    length = max(3, int(round(31*min(h,w)/224))) | 1
    kernel = np.zeros((length,length), np.float32);kernel[length//2,:]=1/length
    # One direction/occluder per video, shared across severity levels.
    angle = float(rng.uniform(-45,45))
    kernel = cv2.warpAffine(kernel,cv2.getRotationMatrix2D((length//2,length//2),angle,1),(length,length));kernel/=kernel.sum()
    oh,ow = max(1,h//2),max(1,w//2)
    oy,ox = int(rng.integers(h-oh+1)),int(rng.integers(w-ow+1))
    for i in selected:
        if family=='black':out[i]=0
        elif family=='motion_blur':out[i]=cv2.filter2D(frames[i],-1,kernel,borderType=cv2.BORDER_REFLECT_101)
        elif family=='occlusion':out[i,oy:oy+oh,ox:ox+ow]=0
        elif family=='overexposure':out[i]=np.minimum(frames[i].astype(np.uint16)+100,255).astype(np.uint8)
        elif family=='packet_loss':
            # Fixed seeded block loss simulation; no removal or time remapping.
            rr=np.random.default_rng(seed+i)
            for k in range(20):
                bh,bw=max(1,h//12),max(1,w//3)
                y,x=int(rr.integers(h-bh+1)),int(rr.integers(w-bw+1))
                donor=max(0,i-1) if k%2==0 else min(len(frames)-1,i+1)
                out[i,y:y+bh,x:x+bw]=frames[donor,y:y+bh,x:x+bw]
        else:raise ValueError(family)
    return out

def critic_scores(candidates, proposals, confidence):
    """Score each existing student interval by best confidence-weighted overlap.

    Teacher proposals provide evidence only, never become output candidates.
    Empty proposals yield all-zero scores and native-first tie fallback.
    """
    c=np.asarray(candidates,float);p=np.asarray(proposals,float).reshape(-1,2)
    s=np.asarray(confidence,float)
    assert len(p)==len(s) and np.isfinite(c).all() and np.isfinite(p).all() and np.isfinite(s).all()
    if not len(p):return np.zeros(len(c))
    inter=np.maximum(0,np.minimum(c[:,None,1],p[None,:,1])-np.maximum(c[:,None,0],p[None,:,0]))
    union=np.maximum(c[:,None,1],p[None,:,1])-np.minimum(c[:,None,0],p[None,:,0])
    return (inter/np.maximum(union,1e-12)*s[None,:]).max(1)
