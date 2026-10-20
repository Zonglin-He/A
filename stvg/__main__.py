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
 ds=request['dataset'];assert ds in ['hc2','vidstg'];setup();lock=lease();start=time.perf_counter()
 try:
  frames,ids=decode(request['metadata'],ds);validate_input(frames,ids,request['metadata']);p=Predictor(ds);out=p.predict(frames,ids,request['metadata'],request.get('key','single-query'));dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
  np.savez(dest,Source=out['Source'],E_Ref=out['E_Ref'],Live_raw=out['Live_raw'],Live=out['Live'],frame_ids=np.asarray(ids),positions8=np.asarray(out['fit']['positions']),Native_interval=out['intervals']['Native'],Strong_interval=out['intervals']['Strong'])
  write(dest.with_suffix('.json'),dict(dataset=ds,key=request.get('key','single-query'),configuration=p.config,prediction_sha256=digest(dest),input_sha256=digest(args.request),expert_count=1,total_seconds=time.perf_counter()-start,capture=out['capture_cost'],adapt=out['cost'],labels_read=False))
 finally:lock.close()
def evaluate(args):
 rows=records(args.manifest);dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
 assert len({r['id']for r in rows})==len(rows),'Repeated evaluation IDs'
 assert not (dest/'sealed.json').exists(),'Evaluation already sealed'
 files={}
 for r in rows:
  key=r['id'];assert key and all(c.isalnum()or c in '_-'for c in key)
  req=dest/'requests'/f'{key}.json';out=dest/'predictions'/f'{key}.npz';assert not out.exists(),'Never overwrite a prediction'
  write(req,dict(dataset=r['dataset'],key=r['key'],metadata=r['metadata']))
  subprocess.run([sys.executable,'-B','-m','stvg','predict','--request',str(req),'--output',str(out)],check=True)
  files[key]=dict(path=str(out.relative_to(dest)),sha256=digest(out))
 write(dest/'sealed.json',dict(status='sealed_before_offline_scoring',queries=len(rows),manifest_sha256=digest(args.manifest),files=files,labels_read=False))
 print('Sealed',len(rows),'predictions')
def score(args):
 os.environ['CUDA_VISIBLE_DEVICES']=''
 from .evaluation import score as run
 run(args.manifest,args.labels,args.predictions,args.output)
def main():
 parser=argparse.ArgumentParser(prog='python -m stvg',description='Single ReferDINO initialization + spatial OPD + Strong time')
 sub=parser.add_subparsers(dest='command',required=True)
 p=sub.add_parser('predict',help='Predict one label-free video/text request');p.add_argument('--request',required=True);p.add_argument('--output',required=True);p.set_defaults(run=predict)
 p=sub.add_parser('evaluate',help='Run a fixed manifest and seal all predictions');p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.set_defaults(run=evaluate)
 p=sub.add_parser('score',help='Score sealed predictions offline; no model inference');p.add_argument('--manifest',required=True);p.add_argument('--labels',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);p.set_defaults(run=score)
 args=parser.parse_args();args.run(args)
if __name__=='__main__':main()
