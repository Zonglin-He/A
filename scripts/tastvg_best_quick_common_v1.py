"""Frozen full-evaluation namespace; no historical run is resumed."""
import sys,time,shutil,gzip,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,save,load
BASE=ROOT/'artifacts/tastvg_best_quick_v1'
DATASETS=['vidstg','hc2']
CONDS=['clean','frame_drop_5','frame_freeze_5','motion_blur_5','occlusion_5','exposure_5']
def budget(*args):assert shutil.disk_usage(ROOT).free>8*2**30,'free disk below8GiB'
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)):
  s=str(args[0])
  if any(t in s for t in ['GT_LABEL','GT_EXPOSURE','labels_diagnostic','GT_SUBSET','/ROWS.json','/SUMMARY.json','test_annotations.json','valv2_proc.json','val_v2.json','vidstd-test-anno']):raise PermissionError('Full evaluation model worker may not read GT or scores')
def bind_decode(dataset):
 from vg_tta import exact_frame_decode_audit_v2 as binding
 if dataset=='hc2':
  from vg_tta.tastvg_paper48_hc2_decode_v1 import decode
  binding.decode=decode
 return binding.decode
def verify(dataset=None):
 lock=read(BASE/'RUNTIME_LOCK.json')
 pins=dict(lock['pins'])
 for rev in sorted((BASE/'revisions').glob('*.json')):pins.update(read(rev)['pins'])
 for rel,h in pins.items():assert sha(ROOT/rel)==h,rel
 for rel,h in lock['metadata'].items():assert sha(BASE/rel)==h,rel
 return read(BASE/dataset/'PLAN.json') if dataset else lock
def savez(p,x):
 import torch
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p
 temp=p.with_suffix(p.suffix+'.tmp')
 with temp.open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0) as f:torch.save(x,f)
 temp.replace(p)
def loadz(p):
 import torch
 with gzip.open(p,'rb') as f:return torch.load(f,map_location='cpu',weights_only=False)
def archive(event):
 import subprocess
 p=ROOT/'docs/RESEARCH_HISTORY.md';s=p.read_text();e='**快速最佳配置对照：'+event+'。** 旧P1 Vid670源/双序8040与P5 HC128源/单序768，均一query每源/六条件/25%专家，v3各自封存参数不变；历史曝光来源内外单列，全部8808预测封存后GT pipeline诊断；全query队列暂停保存、不自动恢复，不改生产。见[状态](</home/wwww/visual grounding/artifacts/tastvg_best_quick_v1/STATUS.json>)。'
 p.write_text(s.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+e+'\n',1)+'\n\n### Full best-configuration evaluation event\n\n'+e+'\n')
 with (BASE/'ARCHIVE.log').open('a') as log:
  for cmd in ['check','snapshot','check']:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',cmd],stdout=log,stderr=subprocess.STDOUT,check=True)
