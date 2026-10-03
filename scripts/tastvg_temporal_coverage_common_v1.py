"""An isolated matched readout task; no historical writer is changed."""
import sys, time, shutil, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, load, sha
from scripts import tastvg_correction_views_common_v1 as old
BASE = ROOT/'artifacts/tastvg_temporal_candidate_coverage_v1'
PUBLIC = ROOT/'results/tastvg_temporal_candidate_coverage/2026-10-03'
OLD = old.BASE; POOL = old.POOL
PREVIOUS = ROOT/'artifacts/tastvg_oracle_event5_v1'
DATASETS = ['vidstg', 'hc2']; SPLITS = ['search', 'confirm']

def plan(ds): return read(OLD/ds/'PLAN.json')
def key(c): return '/'.join(str(c[k]) for k in ['dataset','split','condition','order','arrival'])
def prefix(c): return f'{c["split"]}_{c["condition"]}_{c["order"]}_{c["arrival"]:05}'
def oldfile(c): return old.oldfile(c['dataset'],c['split'],c['condition'],c['order'],c['arrival'])
def acell(c): return old.oldcell(c['dataset'],c['split'],c['condition'],c['order'],c['arrival'])
def donorfile(c): return OLD/c['dataset']/'round1'/c['split']/'predictions'/c['condition']/c['order']/f'{c["arrival"]:05}.pt'
def donor(c):
    f=donorfile(c); assert sha(f)==read(f.with_suffix('.json'))['sha256']; return load(f)
def capture(c):
    f=POOL/c['dataset']/'capture'/c['condition']/f'{c["parent"]:05}.json'; r=read(f)
    cf=POOL/c['dataset']/r['cache']; assert sha(cf)==r['sha256']
    return load(cf), r, cf
def budget(): assert shutil.disk_usage(ROOT).free > 8*2**30
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        if any(s in str(args[0]) for s in ['GT_LABELS','GT_EXPOSURE','labels_diagnostic',
            'test_annotations.json','valv2_proc.json','EVENT_PLAN.json','ORACLE_ROWS.json',
            '/results/','INTERVENTION_ROWS.json','TA_ORACLE_EVENT5_REVIEW.md']):
            raise PermissionError('Candidate generation is forbidden to read GT or scores')
def verify(inputs=False, labels=False):
    r=read(BASE/'RUNTIME_LOCK.json')
    for f,h in r['pins'].items(): assert sha(ROOT/f)==h, f
    for f,h in r['protected_metadata'].items(): assert sha(ROOT/f)==h, f
    if inputs:
        for f,h in r['inputs'].items(): assert sha(ROOT/f)==h, f
    if labels:
        for f,h in r['label_hashes_from_predecessor_receipt'].items(): assert sha(ROOT/f)==h, f
    return r
def summary(rows, fields):
    # Shared, already-public bootstrap, independently reconstructed by public auditor.
    from scripts.tastvg_oracle_event5_common_v1 import source_summary
    return source_summary(rows, fields)
def archive(message):
    import subprocess
    f=ROOT/'docs/RESEARCH_HISTORY.md'; s=f.read_text()
    e=('**2026-10-03｜固定八候选时间覆盖对照：'+message+'。** '
       '用户授权当前两集32开发+16历史曝光确认/一query/两序/clean+五5%/25%专家，'
       '1152固定A轨迹和288专家候选对照；只改native+三位置两长度+长区间的候选分配，'
       '用原缓存source-head起止先验，不改原critic、Uniform持续状态/空间、offset物理映射或预算。'
       '新池全seal后CPU GT与完整采样网格oracle，确认不选择配置，零新GPU/专家/反向。'
       '生产CURRENT和全部旧暂停队列保持原样。'
       '见[协议](</home/wwww/visual grounding/protocols/tastvg_temporal_candidate_coverage_v1.md>)、'
       '[状态](</home/wwww/visual grounding/artifacts/tastvg_temporal_candidate_coverage_v1/STATUS.json>)。')
    f.write_text(s.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+e+'\n',1)
       +'\n\n### Temporal fixed-budget coverage execution update\n\n'+e+'\n')
    with (BASE/'ARCHIVE.log').open('a') as log:
        for command in ['check','snapshot','check']:
            subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',command],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
