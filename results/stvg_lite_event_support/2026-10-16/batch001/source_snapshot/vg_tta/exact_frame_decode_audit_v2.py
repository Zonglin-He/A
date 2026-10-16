"""Metadata-only original-zero-based-frame decoder for alignment sensitivity."""
import subprocess
import numpy as np
from vg_tta.unanchored_dense_shift_data_v1 import FFMPEG,FFMPEG_SHA,sha


def decode(query):
    assert sha(FFMPEG)==FFMPEG_SHA and sha(query['video_path'])==query['video_sha256']
    ids=query['frame_ids'];assert ids==sorted(set(ids)) and all(isinstance(i,int) and i>=0 for i in ids)
    expression='+'.join(f'eq(n\\,{i})' for i in ids)
    cmd=[str(FFMPEG),'-v','error','-i',query['video_path'],'-vf','select='+expression,'-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    frames=np.frombuffer(r.stdout,np.uint8).reshape(-1,query['height'],query['width'],3).copy()
    assert len(frames)==len(ids),'Selected frame missing; never truncate/pad silently'
    return frames,ids
