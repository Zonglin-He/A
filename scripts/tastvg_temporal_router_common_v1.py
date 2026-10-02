"""Separate CPU diagnostic namespace; all prior experiments are read-only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys, time, collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
BASE=ROOT/'artifacts/tastvg_temporal_router_t0_v1'
PRIOR=ROOT/'artifacts/tastvg_event_support_v1'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
DATASETS=['vidstg','hc2']

def verify():
    lock=read(BASE/'RUNTIME_LOCK.json')
    for name,h in lock['pins'].items():assert sha(ROOT/name)==h,name
    for name,h in lock['inputs'].items():assert sha(ROOT/name)==h,name
    assert read(PRIOR/'FINAL_COMPLETION.json')['status']=='completed'
    return lock

def scalar_summary(rows, fields):
    from scripts.score_tastvg_best_quick_v1 import source_summary
    # Undefined mass/quantiles of an unavailable expert are not silently
    # fabricated. The matched subset and its denominator are reported.
    available=[r for r in rows if all(r.get(k) is not None for k in fields)]
    z=source_summary(available,fields)
    z.update(requested_cells=len(rows),unavailable_cells=len(rows)-len(available))
    return z

def group(rows, condition, subset='all'):
    return [r for r in rows if (r['condition']!='clean')==(condition=='corruption')
            and (subset=='all' or r['expert_scheduled']==(subset=='expert'))]
