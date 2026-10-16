"""Inert, GT-free state bindings for the final bounded evidence-constrained study."""
import sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
BASE=ROOT/'artifacts/stvg_evidence_constrained_v1'
E=ROOT/'artifacts/stvg_weightfree_tuning_v1/confirmation003'
C=ROOT/'artifacts/stvg_weightfree_coordinate_v1'
S=ROOT/'artifacts/stvg_final_table1_v1'
def roster(cohort='development'):
 p=BASE/('DEVELOPMENT_ROSTER_LOCK.json'if cohort=='development'else'CONFIRMATION_ROSTER_LOCK.json');return read(p)['records']
def load_case(rec):
 import torch
 from scripts.run_stvg_table1_source_v1 import content_sha256
 p=ROOT/rec['source_case_path'] if 'source_case_path'in rec else S/'source_cases'/(rec['probe_alias']+'.pt')
 receipt=read(p.with_suffix('.receipt.json'));assert sha(p)==receipt['sha256'];c=torch.load(p,map_location='cpu',weights_only=False)
 assert c['query_key_sha256']==rec['query_key_sha256']and content_sha256(c)==c['source_case_sha256']and not c['GT_read'];return c,p

def save_pt(p,x):
 import torch
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();torch.save(x,p);return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
def verify_runtime():
 r=read(BASE/'RUNTIME.json');assert r['status']=='locked'and not r['GT_read']
 for p,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/p)==h,p
 return r

def native_arm(case,interval=None):
 import numpy as np
 return dict(boxes=np.asarray(case['native_boxes_xyxy_normalized']).copy(),interval=list(interval if interval is not None else case['Native_interval']))
def fixed_dev_spatial(rec):
 import torch
 lock=read(C/'05_lr_refine/SELECTION_LOCK.json');p=C/'05_lr_refine/student'/(rec['probe_alias']+'.pt');bar=read(C/'05_lr_refine/STUDENT_GLOBAL_BARRIER.json');item=next(x for x in bar['files']if ROOT/x['path']==p);assert sha(p)==item['sha256'];z=torch.load(p,map_location='cpu',weights_only=False);return copy.deepcopy(z['arms'][lock['selected'][rec['dataset']]]),p,z
