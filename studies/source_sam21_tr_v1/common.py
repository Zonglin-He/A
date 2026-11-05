"""Source-only SAM2.1 tracking + cached Source-box temporal refinement."""
from pathlib import Path
import os,sys,json,time,hashlib,shutil
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/source_sam21_tr_v1';P=R/'results/source_sam21_tr_v1'
V=R/'outputs/review_strengthening_v1/vendor/sam2';WEIGHT=R/'outputs/review_strengthening_v1/vendor/sam2.1_hiera_large.pt';PYTHON=R/'outputs/review_strengthening_v1/sam2_runtime/bin/python'
EXT=R/'outputs/edgetam_comparison_v1/extensions/sam2/_C.so'
sys.path[:0]=[str(R),str(R/'tools')]
from full import read,sha,atomic,identity
import numpy as np,torch
from functools import lru_cache
fixed=lru_cache(None)(read)
COUNTS={'hc2':1902,'vidstg':10303};ARMS=['Source','SourceTR','SAM2Native','SAM2TR','Current'];PAIRS=[('SAM2TR','Source'),('SAM2TR','SourceTR'),('Current','SAM2TR'),('SAM2TR','SAM2Native')]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def source_path(ds,key):return D/'source_inputs'/ds/(key+'.npz')
def roster(ds):return fixed(D/'source_roster.json')[ds]
def rows(stage,ds):
 rr=roster(ds)
 if stage=='pilot':
  keep=fixed(D/'pilot_selection.json')[ds];by={r['id']:r for r in rr};rr=[by[k]for k in keep]
 assert len(rr)==(128 if stage=='pilot'else COUNTS[ds]);return rr
def input_data(ds,row):
 e=fixed(D/'source_inputs_sealed.json')['domains'][ds]['files'][row['id']];assert sha(source_path(ds,row['id']))==e['sha256']
 with np.load(source_path(ds,row['id']))as z:
  assert set(z.files)=={'Source','Source_interval','SourceTR_interval','frame_ids','positions8'};a={k:z[k].copy()for k in z.files}
 assert np.array_equal(a['frame_ids'],row['metadata']['frame_ids']);pos=list(map(int,a['positions8']));assert len(set(pos))==8
 from stvg.positions import observation_positions
 ids=a['frame_ids'].tolist();iv=a['Source_interval'];assert pos==observation_positions(ids,[ids.index(int(iv[0])),ids.index(int(iv[1])-1)])
 return a,e
def data(ds,row):
 from stvg.decode import decode
 a,e=input_data(ds,row);frames,ids=decode(row['metadata'],ds);assert ids==a['frame_ids'].tolist();assert frames.shape[1:3]==(row['metadata']['height'],row['metadata']['width'])
 return frames,a,dict(source_input_sha256=e['sha256'],video_sha256=row['metadata']['video_sha256'],RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),positions_sha256=hashlib.sha256(a['positions8'].tobytes()).hexdigest(),Source_prompt_boxes_sha256=hashlib.sha256(a['Source'][a['positions8']].tobytes()).hexdigest(),GT_read=False,ReferDINO_input=False)
def resource(start):
 assert time.perf_counter()-start<72*3600 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:'))>=8*2**30
 assert sum(f.stat().st_size for f in D.rglob('*')if f.is_file())<160*2**30
def foreign():
 import importlib.util,subprocess
 sp=importlib.util.spec_from_file_location('_source_SAM21_protected_service',R/'studies/expert_subset_followups_v1/protected_service_gate_runtime_v2.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);out=[]
 for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).splitlines():
  if not line.strip():continue
  pid,name=line.split(',',1);pid=int(pid)
  if pid!=os.getpid():out.append(m.identity_from_proc(pid,name.strip()))
 return out
def check(scoring=False):
 p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for group in ['code_sha256','input_sha256']:
  for f,h in p[group].items():assert sha(R/f)==h,f
 if scoring:
  for f,h in p['score_input_sha256'].items():assert sha(R/f)==h,f
 return p
def inference_guard(event,args):
 from stvg.runtime import guard
 guard(event,args)
 if event=='open'and args and isinstance(args[0],(str,bytes,Path)):
  name=os.fsdecode(args[0]);assert not any(x in name for x in ['/teacher_cache/','/new_current_expert_cache/','/expert_payloads/','/outputs/reviewer_key_controls_v1/temporal/','/outputs/shared_corrected_roi_v1/components/']),name
