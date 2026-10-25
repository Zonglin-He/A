"""Independent old768-view synchronization using exact native appearance prefixes."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'studies/shared_corrected_roi_v1'))
import shared_common as selected
from shared_common import *
S=Path(__file__).parent;D=R/'outputs/shared_corrected_roi_corruption_v1';P=R/'results/shared_corrected_roi_corruption_v1';CGNEW=R/'outputs/shared_corrected_roi_CG_v1';VIEWS=['Blur','JPEG','Dim'];ARMS=['Source','FullMAP','HistoricalTwoDP'];PAIRS=[('FullMAP','Source'),('FullMAP','HistoricalTwoDP'),('FullMAP','CleanFullMAP'),('Source','CleanSource')]
def validate():
 p=read(D/'plan.json');assert digest({k:v for k,v in p.items()if k!='fingerprint'})==p['fingerprint']
 for f,h in p['code_sha256'].items():assert sha(R/f)==h,f
 for f,h in p['input_sha256'].items():assert sha(R/f)==h,f
 selected.validate();return p
def rows(ds):return fixed(OLD/'selection.json')[ds]['confirm']
def save(path,value):
 import torch
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');torch.save(value,temp);os.replace(temp,path)
def checked_parent(ds,key,view):
 import torch
 e=fixed(PREV/'corruptions/sealed.json')['files'][ds][key][view]
 for k,h in [('path','sha256'),('compact_path','compact_sha256'),('receipt_path','receipt_sha256')]:assert sha(PREV/e[k])==e[h]
 c=torch.load(PREV/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);assert not c['GT_read'];return c,e
def resources(start):
 import shutil
 p=fixed(D/'plan.json')['limits'];assert time.perf_counter()-start<p['stage_seconds']and shutil.disk_usage(R).free>=p['minimum_disk_bytes'];assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=p['minimum_RAM_bytes']
