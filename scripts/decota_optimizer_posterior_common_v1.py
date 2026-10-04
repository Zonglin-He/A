"""Isolated R1 stage locks, prediction receipts, and archive transitions."""
import sys,time,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
from scripts import tastvg_decota_critic_ln_common_v1 as prior
BASE=ROOT/'artifacts/decota_optimizer_posterior_v1'
PUB=ROOT/'results/decota_optimizer_posterior/2026-10-04'
DATASETS=['vidstg','hc2']

def verify():
    lock=read(BASE/'RUNTIME_LOCK.json');pins=dict(lock['pins'])
    for f in sorted((BASE/'revisions').glob('*.json')):pins.update(read(f)['pin_overrides'])
    for f,h in {**pins,**lock['inputs'],**lock['protected']}.items():assert sha(ROOT/f)==h,f
    return lock

def commit(f,x):
    save(f,x);write(f.with_suffix('.json'),dict(sha256=sha(f),GT_read=False,
        runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),time=time.time()))

def checked(f):
    r=read(f.with_suffix('.json'));assert not r['GT_read'] and sha(f)==r['sha256']
    assert r['runtime_lock_sha256']==sha(BASE/'RUNTIME_LOCK.json');return load(f)

def verify_seal():
    verify();b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert b['status']=='sealed' and not b['GT_read']
    for f,h in b['files'].items():assert sha(BASE/f)==h,f
    return b

def budget():assert shutil.disk_usage(ROOT).free>8*2**30,'Free disk below 8GiB'

def archive(event):
    f=ROOT/'docs/RESEARCH_HISTORY.md';entry=('**2026-10-04｜DeCoTA optimizer/posterior R1：'+event+'。** '
        '用户附件8cf6bbae授权完整执行有条件路线；首轮保持原两集32开发+16历史曝光来源/一query/双序/clean+五5%，1152匹配到达。'
        '空间固定P1到达前LN状态/1792接口/四旧DINO观察/原native时间，完整Direct/All/Admit/Top1×Adam/SGD及All两authority臂；'
        'SGD仅32开发clean第一步功能位移校准，无GT选LR。所有对照使用相同P1 prestate，不冒充各自持续流；'
        'P1慢LN路径保留，后续独立online验证按条件接续。时间纯CPU复用原actionness/合法双offset跨度，beta1预锁，比较Native/Hard/Full/Extent/历史PM；'
        'Extent保持物理中心概率边际但MAP中心仍可变。全预测seal后GT官方dense/source配对10000，不根据确认GT重选参数。'
        'CURRENT/C1/P0/P1冻结，旧暂停队列不恢复。后续track/scope/online/budget/cross-domain为条件阶段，不凭本轮代码称完成。'
        '见[协议](</home/wwww/visual grounding/protocols/decota_optimizer_posterior_v1.md>)、'
        '[状态](</home/wwww/visual grounding/artifacts/decota_optimizer_posterior_v1/STATUS.json>)。')
    txt=f.read_text();f.write_text(txt.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+entry+'\n',1)+'\n\n### DeCoTA optimizer/posterior update\n\n'+entry+'\n')
    with (BASE/'ARCHIVE.log').open('a') as out:
        for a in ['check','snapshot','check']:
            subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',a],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
