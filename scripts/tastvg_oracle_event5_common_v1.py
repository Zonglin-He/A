"""Isolated GT-assisted diagnosis; sealed A and old observations are read-only."""
import sys,time,shutil,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,load,save,sha
from scripts import tastvg_correction_views_common_v1 as previous
BASE=ROOT/'artifacts/tastvg_oracle_event5_v1'
OLD=previous.BASE
POOL=previous.POOL
PUBLIC=ROOT/'results/tastvg_oracle_event5/2026-10-03'
DATASETS=['vidstg','hc2'];SPLITS=['search','confirm'];BRANCHES=['U','U2','R','GT_event']
def budget():assert shutil.disk_usage(ROOT).free>8*2**30
def plan(ds):return read(OLD/ds/'PLAN.json')
def prefix(c):return f'{c["split"]}_{c["condition"]}_{c["order"]}_{c["arrival"]:05}'
def cell_key(c):return '/'.join(str(c[k]) for k in ['dataset','split','condition','order','arrival'])
def payload_path(c,stage='round1'):
    return OLD/c['dataset']/stage/c['split']/'predictions'/c['condition']/c['order']/f'{c["arrival"]:05}.pt'
def cached_payload(c,stage='round1'):
    f=payload_path(c,stage);assert sha(f)==read(f.with_suffix('.json'))['sha256'];return load(f)
def old_a(c):return previous.oldcell(c['dataset'],c['split'],c['condition'],c['order'],c['arrival'])
def cached_evidence(c,branch):
    x=cached_payload(c,'round2' if branch=='U2' else 'round1')
    r=x['evidence'][branch];assert sha(ROOT/r['cache'])==r['cache_sha256']
    return load(ROOT/r['cache']),r
def verified(include_GT=False):
    r=read(BASE/'RUNTIME_LOCK.json');pins=dict(r['pins'])
    for f in sorted((BASE/'revisions').glob('*.json'),key=lambda f:int(re.search(r'_(\d+)\.json$',f.name).group(1))):pins.update(read(f)['pin_overrides'])
    for f,h in pins.items():assert sha(ROOT/f)==h,f
    for f,h in r['protected_metadata'].items():
        if include_GT or 'GT_LABELS' not in f:assert sha(ROOT/f)==h,f
    return r
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        if any(s in str(args[0]) for s in ['GT_LABELS','GT_EXPOSURE','labels_diagnostic','test_annotations.json',
              'valv2_proc.json','val_v2.json','/results/','/ORACLE_ROWS.json','/INTERVENTION_ROWS.json']):
            raise PermissionError('GT-event model may read planned positions but not raw GT or scores')
def commit(f,value):
    save(f,value);write(f.with_suffix('.json'),dict(sha256=sha(f),raw_GT_read=False,
        GT_assisted_observation=True,runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),time=time.time()))
def checked(f):assert sha(f)==read(f.with_suffix('.json'))['sha256'];return load(f)
def source_summary(rows,fields):
    import numpy as np,collections
    if not rows:return dict(sources=0,cells=0,metrics={})
    groups=collections.defaultdict(list)
    for r in rows:groups[r['source_id'],r['order'],r['condition']].append([r[f] for f in fields])
    orders=collections.defaultdict(list);so=collections.defaultdict(list)
    for (s,o,c),v in groups.items():so[s,o].append(np.mean(v,axis=0))
    sources=collections.defaultdict(list)
    for (s,o),v in so.items():q=np.mean(v,axis=0);sources[s].append(q);orders[o].append(q)
    ids=sorted(sources);matrix=np.array([np.mean(sources[s],axis=0) for s in ids]);rng=np.random.default_rng(20261003)
    boots=np.concatenate([matrix[rng.integers(0,len(matrix),(100,len(matrix)))].mean(1) for _ in range(100)])
    ci=np.quantile(boots,[.025,.975],axis=0);ov=np.array([np.mean(orders[o],axis=0) for o in sorted(orders)])
    metrics={}
    for j,f in enumerate(fields):
        a=matrix[:,j];loo=(a.sum()-a)/(len(a)-1) if len(a)>1 else a
        metrics[f]=dict(mean=float(a.mean()),ci95=ci[:,j].tolist(),order_values=ov[:,j].tolist(),
            order_sample_SD=float(ov[:,j].std(ddof=1)) if len(ov)>1 else None,
            source_values={str(s):float(v) for s,v in zip(ids,a)},
            leave_one_out_range=[float(loo.min()),float(loo.max())],
            largest_influence_source=int(ids[int(np.argmax(abs(loo-a.mean())))]),
            cell_macro=float(np.mean([r[f] for r in rows])))
    return dict(sources=len(ids),cells=len(rows),metrics=metrics,bootstrap_draws=10000,seed=20261003)
def archive(message):
    import subprocess
    f=ROOT/'docs/RESEARCH_HISTORY.md';s=f.read_text()
    entry=('**2026-10-03｜固定A剩余误差与GT-event5诊断：'+message+'。** '
      '用户授权先1152到达CPU GT时空替换及288专家已有8×9候选上限，再最多288个GT-event5观察；'
      '两集各原32开发+16确认/双序clean+五5%/历史曝光/25%专家，VidK1 HC K8持久A固定；'
      '干预仅匹配一步临时SGD，立即丢弃，GT仅离线评分与明确oracle取帧，不改生产或旧队列。'
      '见[协议](</home/wwww/visual grounding/protocols/tastvg_oracle_event5_v1.md>)、'
      '[状态](</home/wwww/visual grounding/artifacts/tastvg_oracle_event5_v1/STATUS.json>)。')
    f.write_text(s.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+entry+'\n',1)
        +'\n\n### Fixed-A oracle/event5 execution update\n\n'+entry+'\n')
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'ARCHIVE.log').open('a') as log:
        for a in ['check','snapshot','check']:
            subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',a],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
