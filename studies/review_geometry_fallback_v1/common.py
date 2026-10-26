from pathlib import Path
import os,sys,json,time,hashlib,shutil
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
sys.path[:0]=[str(R/'studies/shared_corrected_roi_v1'),str(R/'tools'),str(R)]
import shared_common as selected
from shared_common import read,sha,atomic,identity,records,digest
D=R/'outputs/review_geometry_fallback_v1';P=R/'results/review_geometry_fallback_v1'
COUNTS={'hc2':1902,'vidstg':10303};ARMS=['Source','FullMAP','ClipToImage','SourceFallbackFullOutside'];PAIRS=[('ClipToImage','FullMAP'),('SourceFallbackFullOutside','FullMAP'),('FullMAP','Source')]
def validate():
 p=read(D/'plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p
def resources(start):
 assert time.perf_counter()-start<259200 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:'))>=8*2**30
def gt_guard():
 from stvg.runtime import guard
 sys.addaudithook(guard)
