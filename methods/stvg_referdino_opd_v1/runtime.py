"""Local frozen model loading, shared GPU lease and label-free video decoding."""
import os,sys,time,json,hashlib,fcntl,gc,subprocess
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb')as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()

def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
  s=str(args[0])
  if any(x in s for x in ['/annos/','/annotations/','labels_diagnostic','GT_LABEL','test_annotations.json','valv2_proc.json']):raise PermissionError('Inference forbids annotation access: '+s)

def setup():
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 torch.set_num_threads(4);torch.manual_seed(20260920);np.random.seed(20260920)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 sys.addaudithook(guard)

def lease():
 f=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f

def load_source():
 from methods.decota_final_simplified_v1._tastvg_load import official_imports,load_model_on_device
 ck=ROOT/'checkpoints/TASTVG_VidSTG.pth';assert sha(ck)=='5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83'
 official_imports();pipeline=sys.modules['models.pipeline'];old=pipeline.load_json;seen=[]
 def empty(path):
  assert str(path).endswith(('/annos/train.json','/annos/test.json'));seen.append(str(path));return {}
 pipeline.load_json=empty
 try:m,_,_=load_model_on_device('vidstg',ROOT/'artifacts/tastvg_runtime/vidstg',checkpoint=ck,device='cuda',source_dataset='vidstg')
 finally:pipeline.load_json=old
 assert len(seen)==2 and m.verb_label=={} and m.verb_label2=={}
 return m.eval().requires_grad_(False)

def decode(metadata):
 # Match HC official FFmpeg output timing; no vsync=0 alteration.
 FFMPEG=ROOT/'.conda/tubedetr/bin/ffmpeg';FFMPEG_SHA='bfec0e82c8497b0b979b1a7812ba1101da3646e8a3330bcb5537badb840a527e'
 assert sha(FFMPEG)==FFMPEG_SHA
 if metadata.get('video_sha256'):assert sha(metadata['video_path'])==metadata['video_sha256']
 x=subprocess.run([str(FFMPEG),'-v','error','-i',metadata['video_path'],'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
 frames=np.frombuffer(x.stdout,np.uint8).reshape(-1,metadata['height'],metadata['width'],3);ids=metadata['frame_ids'];assert ids==sorted(set(ids)) and ids[-1]<len(frames)
 return frames[ids].copy(),ids

def parse(caption):
 from methods.decota_final_simplified_v1.observations import QuerySubjectParser,parse_context,visual_query
 tick=time.perf_counter();p=QuerySubjectParser(ROOT/'.cache/stanza');v=dict(subject=p(caption)['subject'],old=visual_query(p,caption),context=parse_context(p,caption));return v,time.perf_counter()-tick
