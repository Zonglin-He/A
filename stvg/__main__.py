"""One command for inference, sealed evaluation and offline scoring."""
import argparse,json,os,sys,hashlib,subprocess,time
from pathlib import Path

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def records(path):
 return [json.loads(s)for s in Path(path).read_text().splitlines()if s.strip()]
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def predict(args):
 import numpy as np
 from .runtime import ROOT,setup,lease
 from .decode import decode
 from .inputs import validate_input,validate_metadata
 from .predictor import Predictor
 request=json.loads(Path(args.request).read_text())
 if not isinstance(request,dict) or set(request)-{'dataset','key','metadata'}:raise ValueError('Expected only dataset/key/metadata in a prediction request')
 validate_metadata(request['metadata'])
 ds=request['dataset'];assert ds in ['hc2','vidstg']
 if args.backbone!='tastvg':raise NotImplementedError('Only TA-STVG BoxFixed+RegionKernel is supported')
 setup();lock=lease();start=time.perf_counter()
 try:
  frames,ids=decode(request['metadata'],ds);validate_input(frames,ids,request['metadata'])
  p=Predictor(ds);out=p.predict(frames,ids,request['metadata'],request.get('key','single-query'),temporal_mode='region_kernel');dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
  from .worker import prediction_arrays
  np.savez(dest,**prediction_arrays(out,ids))
  write(dest.with_suffix('.json'),dict(dataset=ds,key=request.get('key','single-query'),configuration=p.config,prediction_sha256=digest(dest),input_sha256=digest(args.request),expert_count=1,total_seconds=time.perf_counter()-start,capture=out['capture_cost'],spatial_and_temporal=out['cost'],temporal_mode=out['temporal_mode'],labels_read=False))
 finally:lock.close()
def evaluate(args):
 from .runtime import setup,lease
 from .worker import run
 from .visual_cache import VisualCache
 setup();lock=lease()
 try:
  run(args.manifest,args.output,args.teacher_cache,args.resume_hash,
      VisualCache(args.visual_cache_mib*2**20)if args.visual_cache_mib else None,args.backbone,temporal_mode='region_kernel')
 finally:lock.close()
def score(args):
 os.environ['CUDA_VISIBLE_DEVICES']=''
 from .evaluation import score as run
 run(args.manifest,args.labels,args.predictions,args.output)
def main():
 parser=argparse.ArgumentParser(prog='python -m stvg',description='BoxFixed + fixed one-round RegionKernel + original physical MBR; one frozen method')
 sub=parser.add_subparsers(dest='command',required=True)
 p=sub.add_parser('predict',help='Predict one label-free video/text request');p.add_argument('--request',required=True);p.add_argument('--output',required=True);p.add_argument('--backbone',choices=['tastvg'],default='tastvg');p.set_defaults(run=predict)
 p=sub.add_parser('evaluate',help='Run a fixed manifest and seal all predictions');p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--backbone',choices=['tastvg'],default='tastvg');p.add_argument('--teacher-cache');p.add_argument('--resume-hash');p.add_argument('--visual-cache-mib',type=int,default=4096);p.set_defaults(run=evaluate)
 p=sub.add_parser('score',help='Score sealed predictions offline; no model inference');p.add_argument('--manifest',required=True);p.add_argument('--labels',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);p.set_defaults(run=score)
 args=parser.parse_args();args.run(args)
if __name__=='__main__':main()
