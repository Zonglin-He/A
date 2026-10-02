"""Isolated reference-only online comparison; no change to old experiment code."""
import sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
BASE=ROOT/'artifacts/tastvg_routed_online_token_v1'
PRIOR=ROOT/'artifacts/tastvg_event_support_v1'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
QUAL=ROOT/'artifacts/tastvg_reference_selection_v1'
PUBLIC=ROOT/'results/tastvg_routed_online_token/2026-10-02'
DATASETS=['hc2','vidstg']
from scripts.tastvg_extended_common_v3 import guard
def budget():assert shutil.disk_usage(ROOT).free>8*2**30,'free disk below8GiB'
def verify(ds=None):
 l=read(BASE/'RUNTIME_LOCK.json');pins=dict(l['pins'])
 for rev in sorted((BASE/'revisions').glob('*.json')):pins.update(read(rev)['pin_overrides'])
 for f,h in pins.items():assert sha(ROOT/f)==h,f
 for f,h in l['inputs'].items():assert sha(ROOT/f)==h,f
 return read(BASE/ds/'PLAN.json') if ds else l
