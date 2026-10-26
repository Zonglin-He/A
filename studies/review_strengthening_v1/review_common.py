"""Independent evidence-only readers for the explicitly authorized review additions."""
import os,sys,json,time,hashlib
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
sys.path[:0]=[str(R),str(R/'tools'),str(R/'studies/shared_corrected_roi_v1')]
import shared_common as selected
from shared_common import RUN,PROVIDER,read,sha,atomic,identity,binding,records
D=R/'outputs/review_strengthening_v1';P=R/'results/review_strengthening_v1'
COUNTS={'hc2':1902,'vidstg':10303}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def validate():
 p=read(D/'matched_cpu_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 selected.validate();return p
def gt_guard():
 labels={(PROVIDER/j['labels']).resolve()for j in read(RUN/'plan.json')['jobs'].values()}
 def guard(event,args):
  if event=='open'and args and isinstance(args[0],(str,bytes,Path))and Path(os.fsdecode(args[0])).resolve()in labels:raise PermissionError('No labels in prediction or cold replay')
 sys.addaudithook(guard)
def paths(ds,key):
 parent=read(RUN/ds/'sealed.json')['files'][key];current=read(selected.D/f'components_{ds}_sealed.json')['files'][key]
 for base,e in [(RUN/ds,parent),(selected.D,current)]:
  for k,h in [('path','sha256'),('receipt_path','receipt_sha256')]:assert sha(base/e[k])==e[h]
 return parent,current
