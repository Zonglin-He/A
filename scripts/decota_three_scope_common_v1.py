"""Private authority for a finite three-scope experiment; old assets read only."""
import sys,time,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
from scripts import decota_identity_common_v1 as old
from scripts import tastvg_decota_c1_common_v1 as c1
BASE=ROOT/'artifacts/decota_three_scope_v1'
PUB=ROOT/'results/decota_three_scope/2026-10-05'
DATASETS=['vidstg','hc2']
OWN=['vg_tta/decota_three_scope_v1.py','scripts/decota_three_scope_common_v1.py',
     'scripts/run_decota_three_scope_v1.py','scripts/continue_decota_three_scope_v1.py',
     'scripts/test_decota_three_scope_v1.py','protocols/decota_three_scope_v1.md']

def verify():
    x=read(BASE/'RUNTIME_LOCK.json'); pins=dict(x['pins'])
    for f in sorted((BASE/'revisions').glob('*.json')):pins.update(read(f)['pin_overrides'])
    for f,h in {**pins,**x['inputs'],**x['protected']}.items():assert sha(ROOT/f)==h,f
    return x

def commit(f,x):
    assert not f.exists(),f
    save(f,x);write(f.with_suffix('.json'),dict(sha256=sha(f),GT_read=False,time=time.time(),runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json')))

def checked(f):
    z=read(f.with_suffix('.json'));assert sha(f)==z['sha256'] and not z['GT_read']
    assert z['runtime_lock_sha256']==sha(BASE/'RUNTIME_LOCK.json');return load(f)

def budget():assert shutil.disk_usage(ROOT).free>8*2**30

def archive(event):
    f=ROOT/'docs/RESEARCH_HISTORY.md'
    e=('**2026-10-05｜DeCoTA三scope接续：'+event+'。** 新授权第一优先级Uniform4 vs Native-TTS物理分层4；单frozen DINO、原admission/Top1能量、Adam.03/joint1792/十步own-loss选态/Native时间/query重置/LN1/16不改。各32开发+16历史曝光确认、一query/双序/clean+五同像素5%，576唯一输入；选帧后须新DINO，缓存H上冻结头/空间后缀重放，无视频backbone。双offset physical-grid几何重心P0零梯度，只有双集稳定通过才514头SGD.01三步。已有939首write效用只用到达前RoBERTa/Native空间latent/轨迹与时间摘要，双角色source-heldout ridge固定1；recipient改动/GT不进入key，诊断GT效用监督明确披露。conditional memory仅资格化后执行，旧rank8未通过不外推所有记忆无效；独立episodic/online轨迹与negative tails分别报告。CURRENT/旧队列不改。见[协议](</home/wwww/visual grounding/protocols/decota_three_scope_v1.md>)、[状态](</home/wwww/visual grounding/artifacts/decota_three_scope_v1/STATUS.json>)。')
    t=f.read_text();f.write_text(t.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+e+'\n',1)+'\n\n### Three scope update\n\n'+e+'\n')
    with (BASE/'ARCHIVE.log').open('a') as out:
        for a in ['check','snapshot','check']:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',a],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
