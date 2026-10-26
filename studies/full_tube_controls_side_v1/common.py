"""Isolated, user-requested full-tube controls; frozen parents remain read-only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,hashlib,shutil
from pathlib import Path
from functools import lru_cache
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
sys.path[:0]=[str(R/'studies/shared_corrected_roi_v1'),str(R),str(R/'tools')]
import shared_common as selected
read=selected.read;sha=selected.sha;atomic=selected.atomic;identity=selected.identity;digest=selected.digest;fixed=lru_cache(None)(read)
D=R/'outputs/full_tube_controls_side_v1';P=R/'results/full_tube_controls_side_v1';STRUCT=R/'outputs/shared_corrected_roi_sync_v1';MATCH=R/'outputs/expert_matched_tta_v1';COUNTS={'hc2':1902,'vidstg':10303}
ARMS=['Source','EightReplacementNative','SharedCorrectionNative','ExpertInterpolationNative','NoTemporalRefinement','FullMAP']
PAIRS=[('FullMAP',a)for a in ARMS if a!='FullMAP']+[(a,'Source')for a in ARMS[1:4]]+[('SharedCorrectionNative','EightReplacementNative'),('ExpertInterpolationNative','EightReplacementNative')]
NEW_ARMS=ARMS[1:5]
def records(ds):return selected.records(ds)
def validate():
 p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 selected.validate();return p

def no_GT():
 paths={(selected.PROVIDER/j['labels']).resolve()for j in read(selected.RUN/'plan.json')['jobs'].values()}
 def guard(event,args):
  if event=='open'and args and isinstance(args[0],(str,bytes,Path))and Path(os.fsdecode(args[0])).resolve()in paths:raise PermissionError('Prediction/cold audit never opens labels')
 sys.addaudithook(guard)

def wait_resources(start):
 assert time.perf_counter()-start<259200 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
 # Yield our one CPU slot if more than one unrelated CUDA-hidden study computes.
 while True:
  groups=set()
  for p in Path('/proc').glob('[0-9]*'):
   try:
    v=(p/'stat').read_text().rsplit(')',1)[1].split();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ')
    if v[0]=='Z'or b'python'not in cmd or b'/studies/'not in cmd or b'full_tube_controls_side_v1'in cmd:continue
    env=(p/'environ').read_bytes().split(b'\0')
    if b'CUDA_VISIBLE_DEVICES='in env and any(t in cmd for t in [b'predict',b'score',b'audit',b'export']):groups.add(v[2])
   except (FileNotFoundError,PermissionError,ProcessLookupError):pass
  if len(groups)<=1:return
  atomic(D/'CPU_slot_wait.json',dict(status='yielding_own_slot',external_CPU_groups=len(groups),worker=identity(),time=time.time()));time.sleep(5)
  assert time.perf_counter()-start<259200
