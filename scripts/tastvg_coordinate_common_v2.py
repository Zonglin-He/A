"""Isolated sequential tuning state; old Optuna pins and outputs remain read-only."""
import sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
BASE=ROOT/'artifacts/tastvg_coordinate_tuning_v2'
DEFAULT=dict(lr=.005,rho=.05,teacher_temperature=1.)
DATASETS=['vidstg','hc2']
def budget():assert shutil.disk_usage(ROOT).free>8*2**30,'free disk below8GiB'
def verify(dataset=None):
 lock=read(BASE/'RUNTIME_LOCK.json');design=read(BASE/'DESIGN_LOCK.json')
 assert sha(ROOT/'protocols/tastvg_coordinate_tuning_v2.md')==design['protocol_sha256']
 assert sha(BASE/'GRIDS.json')==design['grids_sha256']
 for rel,h in design['plans'].items():assert sha(BASE/rel)==h,rel
 pins=dict(lock['pins'])
 for rev in sorted((BASE/'revisions').glob('*.json')):pins.update(read(rev)['pins'])
 for rel,h in pins.items():assert sha(ROOT/rel)==h,rel
 for rel,h in lock['metadata'].items():assert sha(BASE/rel)==h,rel
 assert sha(ROOT/'artifacts/stvg_native_support_fig1_v1/uniform64_v2/FINAL_COMPLETION.json')==lock['fig1_completion_sha256']
 assert read(ROOT/'artifacts/stvg_native_support_fig1_v1/uniform64_v2/FINAL_COMPLETION.json')['status']=='completed_verified_and_published'
 return read(BASE/dataset/'PLAN.json') if dataset else lock
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)):
  s=str(args[0])
  if any(v in s for v in ['GT_LABELS','GT_EXPOSURE','labels_diagnostic','GT_SUBSET','/ROWS.json','/SUMMARY.json','test_annotations.json','valv2_proc.json','vidstd-test-anno']):raise PermissionError('Coordinate model worker cannot read labels or scored outcomes')
def bind_decode(dataset):
 from vg_tta import exact_frame_decode_audit_v2 as binding
 if dataset=='hc2':
  from vg_tta.tastvg_paper48_hc2_decode_v1 import decode
  binding.decode=decode
 return binding.decode
def archive(event):
 import subprocess
 p=ROOT/'docs/RESEARCH_HISTORY.md';s=p.read_text();e='**顺序调参执行事件：'+event+'。** Vid/HC分别32搜索+16新复验来源，双序六条件；lr先选定再温度，rho.05/25%专家/K1/1792参数固定，corrupt全部到达选参；历史曝光，旧结果和生产不改。见[状态](</home/wwww/visual grounding/artifacts/tastvg_coordinate_tuning_v2/STATUS.json>)。'
 p.write_text(s.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+e+'\n',1)+'\n\n### Sequential tuning event\n\n'+e+'\n')
 with (BASE/'ARCHIVE.log').open('a') as log:
  for cmd in ['check','snapshot','check']:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',cmd],stdout=log,stderr=subprocess.STDOUT,check=True)
