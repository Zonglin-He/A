"""python -m methods.stvg_referdino_opd_v1.run --input request.json --output prediction.pt

Input JSON: metadata with video_path, optional video_sha256, caption, index,
width, height, frame_ids (original HC physical grid); optional query-only parses.
No benchmark ID/roster/annotation is required. Cache is explicit and validated.
"""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from .runtime import setup,lease,decode
from .predictor import ReferDINOOPDPredictor

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--cache-in');p.add_argument('--cache-out');p.add_argument('--seed',type=int,default=41);a=p.parse_args();setup();dst=Path(a.output);assert not dst.exists();req=json.loads(Path(a.input).read_text());md=req['metadata'];key=req.get('key')or hashlib.sha256((md['caption']+str(md['frame_ids'])).encode()).hexdigest();signature=hashlib.sha256(json.dumps(dict(metadata=md,parses=req.get('parses')),sort_keys=True).encode()).hexdigest();guard=lease();t=time.perf_counter()
 try:
  start=time.perf_counter();frames,ids=decode(md);decode_s=time.perf_counter()-start;pred=ReferDINOOPDPredictor();cache=None
  if a.cache_in:
   blob=torch.load(a.cache_in,map_location='cpu',weights_only=False);assert blob['input_signature']==signature and blob['input_metadata']==md and not blob['GT_read'];cache=blob['cache']
  if cache is None:
   cache=pred.capture(frames,ids,md,parses=req.get('parses'))
   if a.cache_out:
    assert not Path(a.cache_out).exists();torch.save(dict(input_signature=signature,input_metadata=md,cache=cache,GT_read=False),a.cache_out)
  z=pred.predict(frames,ids,md,key=key,cache=cache,seed=a.seed);z['cache_reused']=bool(a.cache_in);z['cost']['decode_seconds']=decode_s;z['cost']['command_wall_seconds']=time.perf_counter()-t;z['input_signature']=signature;dst.parent.mkdir(parents=True,exist_ok=True);torch.save(z,dst);print(json.dumps(dict(output=str(dst),GT_read=False,cost=z['cost']),indent=2))
 finally:guard.close()
if __name__=='__main__':main()
