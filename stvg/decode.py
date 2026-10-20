import subprocess
import numpy as np
from .runtime import ROOT,sha
FFMPEG=ROOT/'.conda/tubedetr/bin/ffmpeg'
FFMPEG_SHA='bfec0e82c8497b0b979b1a7812ba1101da3646e8a3330bcb5537badb840a527e'

def decode_hc(query):
    assert sha(FFMPEG)==FFMPEG_SHA and sha(query['video_path'])==query['video_sha256']
    ids=query['frame_ids'];assert ids==sorted(set(ids)) and all(isinstance(i,int) and i>=0 for i in ids)
    cmd=[str(FFMPEG),'-v','error','-i',query['video_path'],'-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    frames=np.frombuffer(result.stdout,np.uint8).reshape(-1,query['height'],query['width'],3)
    assert ids[-1]<len(frames),f"Official HC frame grid missing: requested {ids[-1]}, decoded {len(frames)}"
    return frames[ids].copy(),list(ids)

def decode_vid(query):
    assert sha(FFMPEG)==FFMPEG_SHA and sha(query['video_path'])==query['video_sha256']
    ids=query['frame_ids'];assert ids==sorted(set(ids)) and all(isinstance(i,int) and i>=0 for i in ids)
    expression='+'.join(f'eq(n\\,{i})' for i in ids)
    cmd=[str(FFMPEG),'-v','error','-i',query['video_path'],'-vf','select='+expression,'-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    frames=np.frombuffer(r.stdout,np.uint8).reshape(-1,query['height'],query['width'],3).copy()
    assert len(frames)==len(ids),'Selected frame missing; never truncate/pad silently'
    return frames,ids

def decode(metadata,dataset):
 return (decode_hc if dataset=='hc2'else decode_vid)(metadata)
