"""Independent, finite paired deployment timing; legacy studies remain immutable."""
from pathlib import Path
import hashlib,json,os,time,shutil
R=Path(__file__).resolve().parents[2]
S=Path(__file__).parent
D=R/'outputs/matched_pipeline_timing_v1'
P=R/'results/matched_pipeline_timing_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def identity(pid=None):
 pid=pid or os.getpid();return dict(pid=pid,startticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
def validate():
 p=read(D/'plan.json');assert digest({k:v for k,v in p.items()if k!='fingerprint'})==p['fingerprint']
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p
def resources(start):
 assert time.perf_counter()-start<259200
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
 assert shutil.disk_usage(R).free>=80*2**30
 assert sum(p.stat().st_size for p in D.rglob('*')if p.is_file())<160*2**30
def references(ds,rows):
 import numpy as np
 edge=R/'outputs/source_prompt_edgetam_v1';cur=R/'outputs/shared_corrected_roi_v1'
 es=read(edge/'pilot_v1'/ds/'sealed.json');cs=read(cur/f'components_{ds}_sealed.json')
 assert read(edge/'pilot_v1'/ds/'reporting_receipt.json')['status']=='verified_whole128_Source_prompt_only_all_mask_input_arrays_and_means'
 assert read(cur/f'components_{ds}_reporting_receipt.json')['status']=='verified_whole_domain'
 out={}
 for row in rows:
  key=row['id'];e=es['files'][key];c=cs['files'][key]
  assert sha(edge/e['path'])==e['sha256'] and sha(cur/c['path'])==c['sha256']
  with np.load(edge/e['path'])as z,np.load(cur/c['path'])as q:
   assert np.array_equal(z['Source'],q['Source'])
   out[key]=dict(Source=z['Source'].copy(),Source_interval=z['Source_interval'].copy(),EdgeTAM=z['EdgeTAM'].copy(),EdgeTAM_interval=z['EdgeTAM_interval'].copy(),Current=q['FullMAP'].copy(),Current_interval=q['FullMAP_interval'].copy(),frame_ids=z['frame_ids'].copy(),positions8=z['positions8'].copy(),RGB_sha256=e['input_binding']['actual_native_RGB_sha256'],reference_SHA=dict(EdgeTAM=e['sha256'],Current=c['sha256']))
 return out
