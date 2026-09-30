"""HC2 official FFmpeg output timing, then select the frozen frame indices.

The HC dataset's loader decodes raw RGB with FFmpeg's default output timing;
forcing vsync=0 instead indexes encoded frames and can shift/drop its end index.
No per-video fallback, truncation, padding, or frame-ID changes are permitted.
"""
import subprocess
import numpy as np
from vg_tta.unanchored_dense_shift_data_v1 import FFMPEG,FFMPEG_SHA,sha

def decode(query):
    assert sha(FFMPEG)==FFMPEG_SHA and sha(query['video_path'])==query['video_sha256']
    ids=query['frame_ids'];assert ids==sorted(set(ids)) and all(isinstance(i,int) and i>=0 for i in ids)
    cmd=[str(FFMPEG),'-v','error','-i',query['video_path'],'-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    frames=np.frombuffer(result.stdout,np.uint8).reshape(-1,query['height'],query['width'],3)
    assert ids[-1]<len(frames),f"Official HC frame grid missing: requested {ids[-1]}, decoded {len(frames)}"
    return frames[ids].copy(),list(ids)
