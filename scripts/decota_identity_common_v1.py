"""Isolated experiment authority and immutable IO."""
import sys,time,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
from scripts import decota_optimizer_posterior_common_v1 as r1
from scripts import decota_actuation_scope_common_v1 as previous
from scripts import tastvg_decota_c1_common_v1 as c1
BASE=ROOT/'artifacts/decota_identity_commitment_v1'
PUB=ROOT/'results/decota_identity_commitment/2026-10-05'
DATASETS=['vidstg','hc2']
def verify():
    x=read(BASE/'RUNTIME_LOCK.json');pins=dict(x['pins'])
    for f in sorted((BASE/'revisions').glob('*.json')):pins.update(read(f)['pin_overrides'])
    for f,h in {**pins,**x['inputs'],**x['protected']}.items():assert sha(ROOT/f)==h,f
    return x
def checked(f):
    z=read(f.with_suffix('.json'));assert z['sha256']==sha(f) and not z['GT_read']
    assert z['runtime_lock_sha256']==sha(BASE/'RUNTIME_LOCK.json');return load(f)
def commit(f,x):
    save(f,x);write(f.with_suffix('.json'),dict(sha256=sha(f),GT_read=False,runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),time=time.time()))
def budget():assert shutil.disk_usage(ROOT).free>8*2**30
def archive(event):
    f=ROOT/'docs/RESEARCH_HISTORY.md'
    line=('**2026-10-05｜DeCoTA hard identity接续：'+event+'。** 用户明确授权S-next三臂同P1 prestate/Adam.03/joint1792/旧DINO；Top1沿用旧admission而非无门控raw top1，原四观察位置不改。Track-MAP包含hard path与GIoU contrastive两个变化，不唯一归因identity；tau1锁定。32开发+16历史曝光确认/一query/双序/clean+五5%。独立online以开发封存选择接续，五个nested25/50% schedules与100%/episodic；Native WHEN，query/Adam重置、LN1/16。纯CPU时长bias仅已有匹配UVTG与同域/跨域Native，无新expert/训练；不自动实现slow temporal或u-scout/LN trust region。CURRENT和旧队列不变。见[协议](</home/wwww/visual grounding/protocols/decota_identity_commitment_v1.md>)、[状态](</home/wwww/visual grounding/artifacts/decota_identity_commitment_v1/STATUS.json>)。')
    t=f.read_text();f.write_text(t.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+line+'\n',1)+'\n\n### Hard identity update\n\n'+line+'\n')
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'ARCHIVE.log').open('a') as out:
        for a in ['check','snapshot','check']:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',a],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
