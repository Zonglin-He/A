"""Frozen diagnostic runtime pins, no GT/model construction on import."""
import sys,time,shutil,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha,status
BASE=ROOT/'artifacts/stvg_native_support_fig1_v1'
OUT=BASE/'uniform64_v2'
MODELS=['tastvg','tubedetr','ptd']
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)):
  s=str(args[0])
  if any(x in s for x in ['/annotations/','/annos/','GT_LABEL','labels_diagnostic','GT_SUBSET','test_annotations.json','valv2_proc.json','/ROWS.json','/SUMMARY.json']):raise PermissionError('Fig1 capture cannot open GT/score assets: '+s)
def set_status(x):
 status(OUT/'STATUS.json',x)
 status(BASE/'STATUS.json',dict(**x,active_namespace='uniform64_v2',original_smoke_predictions_excluded=5))
def verify():
 p=read(OUT/'RUNTIME_LOCK.json');d=read(BASE/'DESIGN_LOCK.json')
 assert sha(BASE/'ROSTER.json')==d['roster_sha256']==p['original_roster_sha256']
 assert sha(OUT/'ROSTER.json')==p['roster_sha256']
 assert sha(ROOT/'protocols/stvg_native_support_fig1_v1.md')==d['protocol_sha256']
 assert sha(ROOT/'protocols/stvg_native_support_uniform64_v2.md')==p['sampling_protocol_sha256']
 pins=dict(p['pins'])
 for f in sorted((OUT/'revisions').glob('*.json')):pins.update(read(f)['pins'])
 for f,h in pins.items():assert sha(ROOT/f)==h,f
 assert read(ROOT/'artifacts/tastvg_optuna_v1/FINAL_COMPLETION.json')['status']=='completed_verified_and_published'
 assert read(OUT/'USER_AUTHORIZATION.json')['uniform_max64_approved']
 assert shutil.disk_usage(ROOT).free>8*2**30
 return p
