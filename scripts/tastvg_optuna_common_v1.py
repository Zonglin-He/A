"""Separate finite tuning task; the published Paper48 recipe remains immutable."""
import sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
BASE=ROOT/'artifacts/tastvg_optuna_v1'
DEFAULT=dict(lr=.005,rho=.05,teacher_temperature=1.)
SPACE=dict(lr=[1e-5,.5],rho=[.001,.5],teacher_temperature=[.05,20.])
DATASETS=['vidstg','hc2']
def budget():assert shutil.disk_usage(ROOT).free>8*2**30,'free disk below 8 GiB'
def verify(dataset=None):
 lock=read(BASE/'LOCK.json')
 for f,h in lock['pins'].items():assert sha(ROOT/f)==h,f
 for f,h in lock['plans'].items():assert sha(BASE/f)==h,f
 assert sha(ROOT/'artifacts/tastvg_paper48_v1/FINAL_COMPLETION.json')==lock['paper48_completion_sha256']
 return read(BASE/dataset/'PLAN.json') if dataset else lock

def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)):
  s=str(args[0])
  if any(v in s for v in ['GT_LABELS','labels_diagnostic','GT_SUBSET','/ROWS.json','/SUMMARY.json','test_annotations.json','valv2_proc.json','vidstd-test-anno']):raise PermissionError('Tuning model worker cannot read labels or scored outcomes')

def bind_decode(dataset):
 from vg_tta import exact_frame_decode_audit_v2 as binding
 if dataset=='hc2':
  from vg_tta.tastvg_paper48_hc2_decode_v1 import decode
  binding.decode=decode
 return binding.decode

def archive(event):
 import subprocess
 p=ROOT/'docs/RESEARCH_HISTORY.md';s=p.read_text();e='**Optuna独立调参事件：'+event+'。** 两数据集分别32搜索源+16复验源，48trial宽范围三参数；Paper48原件/生产/旧队列不改，所有来源有历史曝光。见[执行状态](</home/wwww/visual grounding/artifacts/tastvg_optuna_v1/STATUS.json>)。'
 s=s.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+e+'\n',1);s+='\n\n### Optuna tuning event\n\n'+e+'\n';p.write_text(s)
 with (BASE/'ARCHIVE.log').open('a') as log:
  for cmd in ['check','snapshot','check']:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',cmd],stdout=log,stderr=subprocess.STDOUT,check=True)
