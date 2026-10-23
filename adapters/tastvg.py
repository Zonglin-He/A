"""TA-only native cache bridge, no model forward. Not an ASA/TTS-score interface.

Current source is the original full-caption fused motion-memory visual suffix.
Keeping this choice is numerical compatibility, not a framework requirement.
No CG/TubeDETR/MLLM adapter or performance is implied.
"""
import numpy as np
from interfaces.local import LocalFrame,grid_regions,aggregate_sequence

def local_frames(cap):
    ids=list(map(int,cap['frame_ids']));frames=[None]*len(ids)
    cache=cap.get('temporal_cache',cap.get('native_hidden'))
    if cache is None:raise ValueError('Native local memory missing; do not reconstruct from word scores')
    for off in cache['offsets']:
        h,w=map(int,off['grid_size']);memory=off['inputs']['encoded_feature'];mask=off['inputs']['encoded_mask']
        assert memory.ndim==3 and memory.shape[0]>h*w and not mask.any()
        d=memory.shape[-1];visual=memory[-h*w:].permute(1,0,2).numpy().reshape(-1,h*w,d)
        valid=(~mask[:,-h*w:]).numpy();assert off['frame_ids']==ids[off['offset']::2]
        for j in range(len(visual)):
            t=off['offset']+2*j
            frames[t]=LocalFrame(time=ids[t],values=visual[j],regions=grid_regions(h,w),valid=valid[j],grid_shape=(h,w))
    assert all(f is not None for f in frames);return frames

def extract(cap,fulltime_boxes):
    return aggregate_sequence(local_frames(cap),fulltime_boxes,context_scale=1.5)
