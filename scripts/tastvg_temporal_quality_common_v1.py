"""Isolated fixed-Old8 temporal-quality experiment and immutable input access."""
import sys, time, shutil, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, load, sha
from scripts import tastvg_correction_views_common_v1 as old
BASE = ROOT/'artifacts/tastvg_temporal_quality_old8_v1'
PUBLIC = ROOT/'results/tastvg_temporal_quality_old8/2026-10-03'
PREVIOUS = ROOT/'artifacts/tastvg_temporal_candidate_coverage_v1'
OLD = old.BASE; POOL = old.POOL
DATASETS = ['vidstg', 'hc2']; SPLITS = ['search', 'confirm']
def plan(ds): return read(OLD/ds/'PLAN.json')
def key(c): return '/'.join(str(c[k]) for k in ['dataset','split','condition','order','arrival'])
def prefix(c): return f'{c["split"]}_{c["condition"]}_{c["order"]}_{c["arrival"]:05}'
def oldfile(c): return old.oldfile(c['dataset'],c['split'],c['condition'],c['order'],c['arrival'])
def acell(c): return old.oldcell(c['dataset'],c['split'],c['condition'],c['order'],c['arrival'])
def donorfile(c): return OLD/c['dataset']/'round1'/c['split']/'predictions'/c['condition']/c['order']/f'{c["arrival"]:05}.pt'
def donor(c):
    f=donorfile(c); assert sha(f)==read(f.with_suffix('.json'))['sha256']; return load(f)
def evidence(c): return old.expert(c['dataset'],'temporal',c['parent'],c['condition'],c['pixel_sha256'])
def budget(): assert shutil.disk_usage(ROOT).free > 8*2**30
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        if any(s in str(args[0]) for s in ['GT_LABELS','GT_EXPOSURE','labels_diagnostic',
            'test_annotations.json','valv2_proc.json','EVENT_PLAN.json','ORACLE_ROWS.json',
            '/results/','INTERVENTION_ROWS.json','/SUMMARY.json','_REVIEW.md']):
            raise PermissionError('Unlabelled temporal scoring cannot read GT or measured outcomes')
def verify(inputs=False, labels=False):
    r=read(BASE/'RUNTIME_LOCK.json')
    for f,h in {**r['pins'],**r['protected_metadata']}.items(): assert sha(ROOT/f)==h,f
    if inputs:
        for f,h in r['inputs'].items(): assert sha(ROOT/f)==h,f
    if labels:
        for f,h in r['label_hashes_from_predecessor_receipt'].items(): assert sha(ROOT/f)==h,f
    return r
def summary(rows,fields):
    from scripts.tastvg_oracle_event5_common_v1 import source_summary
    return source_summary(rows,fields)
def archive(message):
    f=ROOT/'docs/RESEARCH_HISTORY.md'; s=f.read_text()
    e=('**2026-10-03｜固定Old8时间定位质量信号：'+message+'。** '
       '用户授权同两集各32开发+16确认/一query/两序/clean+五5%/25%专家，1152固定A读出/288专家；'
       '唯一新变量为缓存PE pooled query与逐时刻视频cosine的区间内−外平均对比，alpha .25预锁；'
       'Old8/A框/Uniform持久K1 K8/1792参数/原像素与专家预算不变，零新模型/专家/GPU。'
       '全部预测封存后离线GT同支持oracle regret/实际vIoU/正确结果损害，确认不选公式；'
       '全部历史曝光，不称IoU校准头、无监督训练版AutoLoc或fresh test，不晋升生产或恢复旧队列。'
       '见[协议](</home/wwww/visual grounding/protocols/tastvg_temporal_quality_old8_v1.md>)、'
       '[状态](</home/wwww/visual grounding/artifacts/tastvg_temporal_quality_old8_v1/STATUS.json>)。')
    f.write_text(s.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+e+'\n',1)
        +'\n\n### Fixed Old8 temporal quality execution update\n\n'+e+'\n')
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'ARCHIVE.log').open('a') as log:
        for command in ['check','snapshot','check']:
            subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',command],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
