"""Frozen Paper48 plans, user-authorized execution policy and immutable implementation verification."""
import sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
BASE=ROOT/'artifacts/tastvg_paper48_v1';OLD=ROOT/'artifacts/tastvg_full_b1_v1'
FAMILIES=['frame_drop','frame_freeze','motion_blur','occlusion','exposure']
CONDS=['clean']+[f'{f}_5' for f in FAMILIES]

def budget(tick=None):
 policy=read(BASE/'EXECUTION_POLICY.json')
 assert policy['total_deadline_unix'] is None and policy['P5_required'] is True
 assert shutil.disk_usage(ROOT).free>8*2**30,'Paper48 free disk floor'

def verify(panel=None):
 p=read(BASE/'IMPLEMENTATION_LOCK.json')
 pins=dict(p['pins'])
 for revision in sorted(BASE.glob('IMPLEMENTATION_REVISION_*.json')):
  r=read(revision);assert r['original_lock_sha256']==sha(BASE/'IMPLEMENTATION_LOCK.json');pins.update(r['pin_overrides'])
 for f,h in pins.items():assert sha(ROOT/f)==h,f
 for f,h in p['locks'].items():assert sha(BASE/f)==h,f
 from scripts.run_tastvg_full_b1_experts_v1 import verify as old_verify
 old_verify() # verifies original frozen method/dependencies/roster, no inference
 return read(BASE/('EXPERT_PLAN.json' if panel is None else panel+'/PLAN.json'))

def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)) and any(x in str(args[0]) for x in ['labels_diagnostic','GT_SUBSET','/ROWS.json','/SUMMARY.json','test_annotations.json','vidstd-test-anno']):raise PermissionError('Paper48 inference forbids GT/metrics')
