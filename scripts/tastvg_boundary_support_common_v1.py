"""Independent execution namespace for the two-support temporal experiment."""
import sys, time, shutil, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, load, sha
from scripts import tastvg_temporal_quality_common_v1 as previous
BASE = ROOT/'artifacts/tastvg_temporal_boundary_support_v1'
PUBLIC = ROOT/'results/tastvg_temporal_boundary_support/2026-10-03'
PRIOR = previous.BASE
OLD = previous.OLD; POOL = previous.POOL
DATASETS = ['vidstg', 'hc2']; SPLITS = ['search', 'confirm']
ARMS = ['A8', 'B8', 'D8', 'A32', 'B32', 'D32']
plan=previous.plan; key=previous.key; prefix=previous.prefix
acell=previous.acell; donor=previous.donor; donorfile=previous.donorfile
oldfile=previous.oldfile; evidence=previous.evidence; budget=previous.budget
guard=previous.guard; summary=previous.summary

def prior_prediction(c): return PRIOR/c['dataset']/'predictions'/f'{prefix(c)}.json'
def verify(inputs=False, labels=False):
    r=read(BASE/'RUNTIME_LOCK.json')
    for f in sorted((BASE/'revisions').glob('revision_*.json')):
        z=read(f);assert z['runtime_lock_sha256']==sha(BASE/'RUNTIME_LOCK.json')
        assert z['science_changed'] is False and set(z['pin_overrides'])<=set(r['pins'])
        r['pins'].update(z['pin_overrides'])
    for f,h in {**r['pins'],**r['protected_metadata']}.items(): assert sha(ROOT/f)==h,f
    if inputs:
        for f,h in r['inputs'].items(): assert sha(ROOT/f)==h,f
    if labels:
        for f,h in r['label_hashes_from_predecessor_receipt'].items(): assert sha(ROOT/f)==h,f
    return r

def archive(message):
    f=ROOT/'docs/RESEARCH_HISTORY.md';s=f.read_text()
    e=('**2026-10-03｜Old8/Expanded32纯时间边界质量：'+message+'。** '
       '用户明确移除Spatial-support C；固定A全空间框/Uniform持久Rank-RKL/1792参数/像素/25%专家，'
       '同两集32开发+16确认/两序六条件1152到达/288专家；Old8完整保留并确定性maximin补24网格区间，'
       'A UVTG/B旧语义对比/D局部双边界min，共同支持；w1秒与缺侧zero预锁、不调参、不混B+D。'
       'CPU零新模型/专家/GPU，所有候选评分选择先封存，再GT测容量/实际读出/regret/严重负尾；'
       '历史曝光，source-disjoint不称fresh；不晋升生产、不恢复历史队列。'
       '见[协议](</home/wwww/visual grounding/protocols/tastvg_temporal_boundary_support_v1.md>)、'
       '[状态](</home/wwww/visual grounding/artifacts/tastvg_temporal_boundary_support_v1/STATUS.json>)。')
    f.write_text(s.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+e+'\n',1)
        +'\n\n### Temporal boundary support execution update\n\n'+e+'\n')
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'ARCHIVE.log').open('a') as log:
        for cmd in ['check','snapshot','check']:
            subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',cmd],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
