"""Independent same-observation full-tube comparison; existing science immutable."""
import os,sys,json,hashlib,time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
sys.path[:0]=[str(R/'studies/review_strengthening_v1'),str(R),str(R/'tools')]
import review_common as rc
D=R/'outputs/expert_matched_tta_v1';P=R/'results/expert_matched_tta_v1';T=R/'outputs/tta_full_v1'
COUNTS=rc.COUNTS;read=rc.read;sha=rc.sha;atomic=rc.atomic;identity=rc.identity;records=rc.records
ARMS=['Source','SourceExpertInterpolation','TentExpertInterpolation','SARExpertInterpolation','Ours']
PAIRS=[('Ours',x)for x in ARMS if x!='Ours']+[('TentExpertInterpolation','SourceExpertInterpolation'),('SARExpertInterpolation','SourceExpertInterpolation'),('SourceExpertInterpolation','Source')]
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def validate():
 p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 rc.validate();return p
def resources():
 import shutil
 mem=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
 return mem>=8*2**30 and shutil.disk_usage(R).free>=80*2**30 and sum(p.stat().st_size for p in D.rglob('*')if p.is_file())<160*2**30
def cg_cpu_jobs():
 groups=set()
 for p in Path('/proc').glob('[0-9]*'):
  try:
   v=(p/'stat').read_text().split();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
   if v[2]!='Z'and 'shared_corrected_roi_CG_v1' in cmd and any(x in cmd for x in ['run.py','score.py','audit.py']):groups.add(v[4])
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return len(groups)
