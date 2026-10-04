"""Conditional cached stages; separate receipts from completed R1."""
import sys,time,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
from scripts import decota_optimizer_posterior_common_v1 as r1
BASE=ROOT/'artifacts/decota_actuation_scope_v1'
PUB=ROOT/'results/decota_actuation_scope/2026-10-05'
DATASETS=r1.DATASETS

def verify():
    p=read(BASE/'RUNTIME_LOCK.json');pins=dict(p['pins'])
    for f in sorted((BASE/'revisions').glob('*.json')):pins.update(read(f)['pin_overrides'])
    for f,h in {**pins,**p['inputs']}.items():assert sha(ROOT/f)==h,f
    r1.verify();return p

def commit(f,x):
    save(f,x);write(f.with_suffix('.json'),dict(sha256=sha(f),GT_read=False,runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),time=time.time()))

def checked(f):
    p=read(f.with_suffix('.json'));assert sha(f)==p['sha256'] and not p['GT_read'] and p['runtime_lock_sha256']==sha(BASE/'RUNTIME_LOCK.json');return load(f)

def budget():assert shutil.disk_usage(ROOT).free>8*2**30

def archive(event):
    date=__import__('datetime').datetime.now(__import__('zoneinfo').ZoneInfo('Asia/Shanghai')).date().isoformat()
    f=ROOT/'docs/RESEARCH_HISTORY.md';line=('**'+date+'｜DeCoTA actuation/scope条件接续：'+event+'。** '
        '附件8cf6bbae完整路线授权，R1先判定，不重选确认；原两集32开发+16历史曝光来源/双序/clean+五5%保持。'
        'R3精确track posterior与FrameSum尺度控制；R4 joint/u-only/小LN；R5实际LN1/16流；R6冻结25/50/100专家及新cross-domain资格须逐阶段锁定。'
        '代码/计划不等于完成；CURRENT与旧C1/P0/P1保持，旧队列不恢复。'
        '见[协议](</home/wwww/visual grounding/protocols/decota_actuation_scope_v1.md>)、[状态](</home/wwww/visual grounding/artifacts/decota_actuation_scope_v1/STATUS.json>)。')
    t=f.read_text();f.write_text(t.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+line+'\n',1)+'\n\n### DeCoTA actuation/scope update\n\n'+line+'\n')
    with (BASE/'ARCHIVE.log').open('a') as out:
        for a in ['check','snapshot','check']:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',a],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
