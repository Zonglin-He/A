"""Query/GT-independent physical-video burst; no forced observation/event hit."""
import hashlib,math
import numpy as np
from vg_tta.tastvg_temporal_qualification_v1 import corrupt_local
FAMILIES=['frame_drop','frame_freeze','motion_blur','occlusion','exposure']

def burst_spec(source,frame_count,ids,percentage):
    seed=int(hashlib.sha256(('DeploymentBurst-20260929|'+source).encode()).hexdigest()[:16],16)
    u=float(np.random.default_rng(seed).random());length=max(1,math.ceil(frame_count*percentage/100))
    start=int(u*(frame_count-length+1));end=start+length
    selected=[i for i,f in enumerate(ids) if start<=f<end]
    return dict(physical_start=start,physical_end=end,length=length,total_physical_frames=frame_count,
                positions=selected,actual_physical_fraction=length/frame_count,
                actual_observed_fraction=len(selected)/len(ids),uniform_draw=u)

def apply_burst(frames,query,spec,family,source):
    pos=spec['positions']
    if family=='frame_freeze':
        out=frames.copy()
        if pos:
            from vg_tta.exact_frame_decode_audit_v2 import decode
            donor=max(0,spec['physical_start']-1)
            if donor in query['frame_ids']:image=frames[query['frame_ids'].index(donor)]
            else:image=decode({**query,'frame_ids':[donor]})[0][0]
            out[pos]=image
        return out
    return corrupt_local(frames,pos,{'frame_drop':'black','exposure':'overexposure'}.get(family,family),source)
