"""Isolated GT-free native-instance experiment primitives; imports are inert."""
import json,hashlib,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stvg_native_instance_spatial_opd_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,d,mutable=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not mutable and p.exists():raise FileExistsError(p)
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
  s=os.fsdecode(args[0]).lower()
  if '/site-packages/'in s and s.endswith(('.py','.pyc','.so')):return
  blocked=('/annos/','/annotations/','/anno_v2/','private_case_bindings','gt_label','gt_subset','test_annotations.json','valv2_proc.json','val_v2.json','train_v2.json','/rows.json','/summary.json','/stvg_matched_support_reference_v1/','/stvg_local_motion_regions_v1/','grounding_dino','/checkpoints/grounding-dino')
  if any(x in s for x in blocked):raise PermissionError('native instance inference forbids GT, prior oracle artifacts and DINO')
def verify(stage='tracking'):
 p=BASE/(stage.upper()+'_RUNTIME.json');d=read(p)
 assert d['status']=='locked' and d['GT_read'] is False
 for rel,h in {**d['pins'],**d['protected']}.items():assert sha(ROOT/rel)==h,rel
 return d
def row(case):return next(r for r in read(BASE/'ROSTER_LOCK.json')['records']if r['case_id']==case)
