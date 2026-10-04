"""P1 isolated runtime, immutable payloads and archive continuity."""
import sys, shutil, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, save, load, sha
from scripts import tastvg_decota_critic_common_v1 as p0
BASE = ROOT/'artifacts/tastvg_decota_critic_ln_p1_v1'
PUB = ROOT/'results/tastvg_decota_critic_ln_p1/2026-10-04'
OLD, POOL, DATASETS = p0.OLD, p0.POOL, p0.DATASETS


def verify():
    lock = read(BASE/'RUNTIME_LOCK.json'); pins = dict(lock['pins'])
    for f in sorted((BASE/'revisions').glob('*.json')): pins.update(read(f)['pin_overrides'])
    for f,h in {**pins,**lock['inputs']}.items(): assert sha(ROOT/f)==h, f
    for f,h in lock['protected_registries'].items(): assert sha(ROOT/f)==h, f
    return lock


def budget():
    assert shutil.disk_usage(ROOT).free>8*2**30, 'Free disk below 8 GiB'


def guard(event,args):
    p0.guard(event,args)


def commit(path,value):
    save(path,value)
    write(path.with_suffix('.json'),dict(sha256=sha(path),GT_read=False,
        runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),time=time.time()))


def checked(path):
    r=read(path.with_suffix('.json'))
    assert not r['GT_read'] and sha(path)==r['sha256']
    assert r['runtime_lock_sha256']==sha(BASE/'RUNTIME_LOCK.json')
    return load(path)


def verify_seal():
    verify(); b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    assert b['status']=='sealed' and not b['GT_read'] and len(b['files'])==1152
    for f,h in b['files'].items(): assert sha(BASE/f)==h, f
    return b


def archive(event):
    f=ROOT/'docs/RESEARCH_HISTORY.md'
    entry=('**2026-10-04｜Critic-DeCoTA LN继承P1：'+event+'。** '
        '用户在P0读回后明确授权进入LN继承；P0 Vid不确定性/严重负例保留，不是自动晋升。'
        '固定原两集32开发+16历史曝光来源/一query/双序/clean+五5%/官方同域EMA；'
        '只新增空间LN位移1/16写回，query256与Adam逐到达重置，1536LN持续，链按数据集/面板/condition/order重置。'
        '复用P0专家支持/温度1/1/Adam.03十步/自身能量最优0..10/源native I0；零新专家/完整backbone/时间更新。'
        '1152 online到达全部实际运行，不复用第二顺序；episodic critic读出按P0收据复用。'
        '主要比较InheritedBefore−Frozen、OnlineAfter−Episodic及OnlineAfter−Frozen，source-macro/paired10000。'
        '每query仍四观察，Before不是实际25%专家非专家流；全seal后GT、不按GT改名单参数或选择状态。'
        'CURRENT/原C1锁不改，旧队列不恢复。见[协议](</home/wwww/visual grounding/protocols/tastvg_decota_critic_ln_p1_v1.md>)、'
        '[状态](</home/wwww/visual grounding/artifacts/tastvg_decota_critic_ln_p1_v1/STATUS.json>)。')
    text=f.read_text(); marker='## 1. 当前状态：先读这一节\n'; assert marker in text
    f.write_text(text.replace(marker,marker+'\n'+entry+'\n',1)+'\n\n### Critic-DeCoTA LN P1 update\n\n'+entry+'\n')
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'ARCHIVE.log').open('a') as out:
        for action in ['check','snapshot','check']:
            subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',action],
                cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
