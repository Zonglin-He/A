"""Isolated fixed-method full evaluation; metadata and atomic receipts only."""
import sys, os, time, json, hashlib, shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha
BASE=ROOT/'artifacts/decota_paper_experiments_v1'
PUB=ROOT/'results/decota_paper_experiments/2026-10-05'
DATASETS=['vidstg','hc2']
CONDS=['clean']
JOBS=[('t1_ours_hc2','hc2','vidstg'),('t1_ours_vid','vidstg','hcstvg2')]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def budget():assert shutil.disk_usage(ROOT).free>8*2**30,'free disk below 8 GiB; preserve outputs and report'
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
  s=str(args[0])
  if any(t in s for t in ['GT_LABEL','GT_EXPOSURE','labels_diagnostic','GT_SUBSET','/ROWS.json','/SUMMARY.json','test_annotations.json','valv2_proc.json','val_v2.json','vidstd-test-anno']):
   raise PermissionError('Full inference cannot read diagnostic labels or results: '+s)
def verify():
 p=read(BASE/'RUNTIME_LOCK.json');pins=dict(p['pins'])
 for f in sorted((BASE/'revisions').glob('*.json')):pins.update(read(f)['pin_overrides'])
 for f,h in {**pins,**p['inputs'],**p['protected']}.items():assert sha(ROOT/f)==h,f
 return p
def save_npz(p,arrays,metadata):
 import numpy as np
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p
 assert 'metadata' not in arrays
 tmp=p.with_suffix('.npz.tmp')
 with tmp.open('wb') as f:
  np.savez_compressed(f,metadata=np.frombuffer(json.dumps(metadata,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode(),dtype=np.uint8),**arrays)
  f.flush();os.fsync(f.fileno())
 tmp.replace(p)
 rc=dict(sha256=sha(p),bytes=p.stat().st_size,GT_read=False,runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),time=time.time())
 write(p.with_suffix('.json'),rc);return rc
def load_npz(p):
 import numpy as np
 p=Path(p);rc=read(p.with_suffix('.json'));assert not rc['GT_read'] and rc['sha256']==sha(p)
 assert rc['runtime_lock_sha256']==sha(BASE/'RUNTIME_LOCK.json')
 with np.load(p,allow_pickle=False) as z:
  md=json.loads(z['metadata'].tobytes());a={k:z[k].copy() for k in z.files if k!='metadata'}
 return a,md,rc
def archive(event):
 BASE.mkdir(parents=True,exist_ok=True)
 f=ROOT/'docs/RESEARCH_HISTORY.md'
 line='**2026-10-05｜冻结方法正式paper实验：'+event+'。** 最新附件8edaf940取代旧四方向全query corruption，880个旧partial保存user_paper_scope_pause不评分；Table1 clean跨域全部Vid10303/HC2 3482query、三固定序。Table2每官方视频一query：Vid732视频，HC3482片段/237父来源（不是237视频），同域clean+五类2.5/5/10%burst；组件/监督/匹配temporal诊断/预算与alpha/成本及全seal GT pipeline依次接续。固定Native-WHEN/Uniform4/单DINO admitted Top1/joint1792 Adam.03/10步/own-loss/LN1/16，不按正式结果改方法，不恢复旧队列；未实现/未测行明确pending。见[正式协议](</home/wwww/visual grounding/protocols/decota_paper_experiments_v1.md>)、[当前状态](</home/wwww/visual grounding/artifacts/decota_paper_experiments_v1/STATUS.json>)。'
 s=f.read_text();f.write_text(s.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+line+'\n',1)+'\n\n### Frozen method paper experiments update\n\n'+line+'\n')
 with (BASE/'ARCHIVE.log').open('a') as log:
  for a in ['check','snapshot','check']:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',a],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
