"""Isolated small frame-acquisition qualification and immutable prior assets."""
import sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
BASE=ROOT/'artifacts/tastvg_reference_selection_v1'
PRIOR=ROOT/'artifacts/tastvg_event_support_v1'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
PUBLIC=ROOT/'results/tastvg_reference_selection/2026-10-02'
DATASETS=['vidstg','hc2']

def verify():
    l=read(BASE/'RUNTIME_LOCK.json')
    for group in ['pins','inputs']:
        for f,h in l[group].items():assert sha(ROOT/f)==h,f
    return l

def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        s=str(args[0])
        if any(x in s for x in ['GT_LABELS','GT_EXPOSURE','labels_diagnostic','GT_SUBSET','/ROWS.json','/SUMMARY.json','test_annotations.json','valv2_proc.json','vidstd-test-anno']):
            raise PermissionError('Reference-acquisition inference forbids labels and scored outcomes')

def budget():assert shutil.disk_usage(ROOT).free>8*2**30,'free disk below8GiB'
