"""New bounded diagnostics; all historical assets remain read-only."""
import sys,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
BASE=ROOT/'artifacts/tastvg_transfer_aligned_v1'
OLD=ROOT/'artifacts/tastvg_routed_online_token_v1'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
QUAL=ROOT/'artifacts/tastvg_reference_selection_v1'
PUBLIC=ROOT/'results/tastvg_transfer_aligned/2026-10-02'
from scripts.tastvg_extended_common_v3 import guard
def budget():assert shutil.disk_usage(ROOT).free>8*2**30,'free disk below8GiB'
def verify(stage):
 l=read(BASE/(stage+'_RUNTIME_LOCK.json'));pins=dict(l['pins'])
 for r in sorted((BASE/'revisions').glob(stage+'*.json')):pins.update(read(r)['pin_overrides'])
 for f,h in pins.items():assert sha(ROOT/f)==h,f
 for f,h in l['inputs'].items():assert sha(ROOT/f)==h,f
 return l
def commit(f,x):
 save(f,x);write(f.with_suffix('.json'),dict(sha256=sha(f),GT_read=False,time=time.time()))
def receipt(f):
 r=read(f.with_suffix('.json'));assert sha(f)==r['sha256'];return r
def archive(event):
 import subprocess
 p=ROOT/'docs/RESEARCH_HISTORY.md';s=p.read_text();entry='**2026-10-02｜'+event+'。** 用户57b87788授权两项机制诊断：HC既有A/R各96保存写入、self/next/near/far目标，冻结text-only上下文与原后缀重放；另原60固定cell/540候选冻结CLIP共同投影逐token/patch inside-outside资格。零新Sa2VA/训练/online方法，旧资产与CURRENT不改；历史曝光、全部预测先seal再GT、源聚类bootstrap；普通CLIP不宣称FILIP局部训练。见[协议](</home/wwww/visual grounding/protocols/tastvg_transfer_aligned_v1.md>)、[状态](</home/wwww/visual grounding/artifacts/tastvg_transfer_aligned_v1/STATUS.json>)。'
 p.write_text(s.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+entry+'\n',1)+'\n\n### Transfer and aligned token diagnostic event\n\n'+entry+'\n')
 with (BASE/'ARCHIVE.log').open('a') as out:
  for action in ['check','snapshot','check']:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',action],stdout=out,stderr=subprocess.STDOUT,check=True)
