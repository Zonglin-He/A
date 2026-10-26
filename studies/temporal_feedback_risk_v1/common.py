"""Frozen temporal risk controls; existing whole-domain evidence remains read-only."""
from pathlib import Path
import os,sys,json,hashlib,time,shutil,csv
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
sys.path.insert(0,str(R/'studies/shared_corrected_roi_v1'))
import shared_common as selected
read=selected.read;sha=selected.sha;atomic=selected.atomic;identity=selected.identity;digest=selected.digest;fixed=selected.fixed
D=R/'outputs/temporal_feedback_risk_v1';P=R/'results/temporal_feedback_risk_v1';COUNTS=selected.COUNTS
ARMS=['Source','SourceJointMAP','NoVisualUpdate','FullMAP','LengthMeanEvidence','TemperatureEvidence2','SmoothPrior3','TemperaturePrior2']
PAIRS=[('FullMAP','Source'),('FullMAP','SourceJointMAP'),('FullMAP','NoVisualUpdate')]+[(a,'NoVisualUpdate')for a in ARMS[4:]]+[('FullMAP',a)for a in ARMS[4:]]
def records(ds):return selected.records(ds)
def validate():
 selected.validate();p=read(D/'plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p
def resources(start):
 assert time.perf_counter()-start<259200 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
 assert sum(f.stat().st_size for f in D.rglob('*')if f.is_file())<160*2**30
def source(ds,row):
 root=selected.PREV/'spatial';e=fixed(root/f'{ds}_sealed.json')['files'][row['id']];c,proof=selected.checked_payload(root,e)
 assert e['input_binding']==selected.binding(row,fixed(selected.RUN/'plan.json')['provider_runtime_fingerprint'])
 assert c['parent_capture_sha256']==fixed(selected.RUN/ds/'sealed.json')['files'][row['id']]['capture_sha256']
 return c,proof
