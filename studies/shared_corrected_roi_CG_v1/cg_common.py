"""Independent CG transfer of direct shared-corrected-box ROI pooling."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'studies/shared_corrected_roi_v1'))
import shared_common as selected
from shared_common import *
S=Path(__file__).parent;D=R/'outputs/shared_corrected_roi_CG_v1';P=R/'results/shared_corrected_roi_CG_v1';CGOLD=R/'outputs/cgstvg_full_v1'
ARMS=['Source','SourceJointMAP','HistoricalTwoDP','FullMAP'];PAIRS=[('FullMAP',a)for a in ARMS if a!='FullMAP']
@lru_cache(None)
def legacy():
 sys.path.insert(0,str(R/'studies/cgstvg_full_v1'));m=load_old('_direct_ROI_legacy_CG',R/'studies/cgstvg_full_v1/run.py');sys.path.insert(0,str(S));return m
def validate():
 p=read(D/'plan.json');assert digest({k:v for k,v in p.items()if k!='fingerprint'})==p['fingerprint']
 for f,h in p['code_sha256'].items():assert sha(R/f)==h,f
 for f,h in p['input_sha256'].items():assert sha(R/f)==h,f
 selected.validate();legacy().validate();return p
def save(path,value):
 import torch
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');torch.save(value,temp);os.replace(temp,path)
def resources(start):
 import shutil
 p=fixed(D/'plan.json')['limits'];assert time.perf_counter()-start<p['stage_seconds']
 assert shutil.disk_usage(R).free>=p['minimum_disk_bytes'];assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=p['minimum_RAM_bytes']
def committed(ds,key):
 rp=D/ds/'receipts'/(key+'.json')
 if not rp.exists():return None
 e=read(rp)
 for k,h in [('path','sha256'),('compact_path','compact_sha256')]:assert sha(D/e[k])==e[h]
 return {**e,'receipt_path':str(rp.relative_to(D)),'receipt_sha256':sha(rp)}
