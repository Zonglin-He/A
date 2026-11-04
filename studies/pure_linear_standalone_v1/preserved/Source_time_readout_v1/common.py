from pathlib import Path
import os,sys,time,json,hashlib,functools,shutil
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/pure_linear_standalone_v1';P=R/'results/pure_linear_standalone_v1';PROVIDER=R.parent/'stvg-opd';RUN=R/'outputs/simplepath_full_v2';CHECK=R/'outputs/spatial_table_recheck_v1'
sys.path.insert(0,str(R));import numpy as np,torch
from pure import interpolate
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest=lambda z:hashlib.sha256(json.dumps(z,sort_keys=True).encode()).hexdigest()
fixed=functools.lru_cache(None)(read)
def atomic(p,z):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(z,indent=2));tmp.replace(p)
def identity():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split()[21]))
COUNTS={'hc2':1902,'vidstg':10303};ARMS=['ExpertLinear_SourceTime','ExpertLinear_FullTime_EvalOnly']
@functools.lru_cache(None)
def rows(ds):
 j=fixed(RUN/'plan.json')['jobs'][ds];p=PROVIDER/j['manifest'];assert sha(p)==j['manifest_sha256'];return [json.loads(x)for x in p.read_text().splitlines()]
def labels(ds):
 j=fixed(RUN/'plan.json')['jobs'][ds];p=PROVIDER/j['labels'];assert sha(p)==j['labels_sha256'];return {r['id']:r for r in map(json.loads,p.read_text().splitlines())if r['dataset']==ds}
def guard(event,args):
 if event=='open'and args and isinstance(args[0],(str,bytes,Path)):
  p=os.fsdecode(args[0]);assert not p.startswith('/dev/nvidia')and Path(p).suffix.lower()not in ['.mp4','.avi','.mov'],p
  if os.environ.get('PREDICTION_ONLY')=='1':assert 'labels.jsonl'not in p and 'evaluation/hc2_labels'not in p and 'evaluation/vidstg_labels'not in p,p

def resources(start):
 assert time.perf_counter()-start<72*3600 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30

def bounds(ds,m):
 a=0 if ds=='hc2'else int(m['start_frame']);b=int(m['frame_count'])if ds=='hc2'else int(m['end_frame']);assert 0<=a<b<=int(m['frame_count']);return a,b

def inputs(ds,row,reference=False):
 e=fixed(CHECK/'teacher_inventory.json')[ds][row['id']];tp=Path(e['payload']);rp=Path(e['receipt']);assert sha(tp)==e['payload_sha256']and sha(rp)==e['receipt_sha256'];receipt=read(rp);ctx=receipt['context'];assert receipt['sha256']==e['payload_sha256']and ctx['dataset']==ds and ctx['caption']==row['metadata']['caption'];assert ctx['frame_ids']==row['metadata']['frame_ids'];raw=torch.load(tp,map_location='cpu',weights_only=False,mmap=True)['boxes'].numpy().astype(np.float64);assert raw.shape==(8,4)
 W,H=row['metadata']['width'],row['metadata']['height'];xy=np.column_stack([raw[:,0]-raw[:,2]/2,raw[:,1]-raw[:,3]/2,raw[:,0]+raw[:,2]/2,raw[:,1]+raw[:,3]/2])*[W,H,W,H];times=np.asarray(ctx['frame_ids'])[np.asarray(ctx['positions'],int)];a,b=bounds(ds,row['metadata']);ids=np.arange(a,b,dtype=np.int64);assert np.all((times>=a)&(times<b))
 if not reference:out=interpolate(times,xy,ids)
 else:
  out=np.empty((len(ids),4),float)
  for j,t in enumerate(ids):
   if t<=times[0]:out[j]=xy[0]
   elif t>=times[-1]:out[j]=xy[-1]
   else:
    left=int(np.searchsorted(times,t,side='right')-1);u=float(t-times[left])/float(times[left+1]-times[left]);out[j]=xy[left]+u*(xy[left+1]-xy[left])
 proof=dict(raw_expert_payload_sha256=e['payload_sha256'],raw_expert_receipt_sha256=e['receipt_sha256'],RGB_context_sha256=ctx['RGB_sha256'],same8physicalpositions=True,video_segment_halfopen=[a,b],old_source_grid_range=[int(ctx['frame_ids'][0]),int(ctx['frame_ids'][-1])],old_uncovered_video_frames=int((ids<ctx['frame_ids'][0]).sum()+(ids>ctx['frame_ids'][-1]).sum()),no_Source_boxes_E_features_posterior_DP_or_time_interval_read=True,GT_read=False)
 return dict(frame_ids=ids,boxes_xyxy=out,anchor_times=times,anchor_xyxy=xy),proof

def validate():
 p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for group in ['code_sha256','input_sha256']:
  for f,h in p[group].items():assert sha(R/f)==h,f
 return p

def evaluation_intervals(ds,key):
 # Evaluation only; this function is NEVER called by prediction/interpolation.
 e=fixed(RUN/ds/'sealed.json')['files'][key];assert sha(RUN/ds/e['path'])==e['sha256']
 with np.load(RUN/ds/e['path'])as z:source=z['Source_interval'].copy()
 e=fixed(CHECK/ds/'sealed.json')['files'][key];assert sha(CHECK/e['path'])==e['sha256']
 with np.load(CHECK/e['path'])as z:full=z['Full_interval'].copy()
 return {ARMS[0]:source,ARMS[1]:full}
