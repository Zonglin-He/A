"""Explicit local module loading; no ambiguous legacy controller imports."""
import sys,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
c=load('common',ROOT/'common.py')
original=load('_expert_matched_original_controller',ROOT/'controller.py')
def cg_cpu_jobs():
 total=0
 for ds in c.COUNTS:
  p=c.R/'outputs/shared_corrected_roi_CG_v1'/f'{ds}_CPU_child.json'
  if p.exists():
   w=c.read(p)
   try:
    s=Path('/proc',str(w['pid']),'stat').read_text().split();total+=int(s[2]!='Z'and s[21]==str(w['startticks']))
   except FileNotFoundError:pass
 return total
def validate_execution():
 p=c.read(c.D/'execution_v3_plan.json');assert p['fingerprint']==c.digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert c.sha(c.R/f)==h,f
 c.validate()
if __name__=='__main__':validate_execution();original.cg_cpu_jobs=cg_cpu_jobs;original.main()
