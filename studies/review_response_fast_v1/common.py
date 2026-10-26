"""New standalone retained-cohort review experiments; no old full-Gamma release."""
from pathlib import Path
import os,sys,json,time,hashlib,shutil
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
sys.path[:0]=[str(R/'studies/review_strengthening_v1'),str(R),str(R/'tools')]
from review_common import selected,read,sha,atomic,identity,records,gt_guard,digest
D=R/'outputs/review_response_fast_v1';P=R/'results/review_response_fast_v1';G=R/'outputs/review_strengthening_v1/gamma_retained_v1'
GAMMA=['Source','Gamma0','Gamma025','Gamma05','Gamma075','Gamma1']
ROI=['Source','CorrectedROI','SourceROI','GlobalAppearance','ShiftedROI','RandomROI']
def validate():
 p=read(D/'plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p
def cohort(ds):
 keys=read(D/'cohort_v1.json')[ds];by={r['id']:r for r in records(ds)};return [by[k]for k in keys]
def guard(start):
 assert time.perf_counter()-start<259200 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30

def pairs(task,ds):
 if task=='gamma':
  default='Gamma05'if ds=='hc2'else'Gamma1'
  return [(a,'Source')for a in GAMMA[1:]]+[(a,default)for a in GAMMA[1:]if a!=default]
 return [('CorrectedROI',a)for a in ROI if a!='CorrectedROI']
