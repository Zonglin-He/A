"""Isolated two-P0 authority; private inputs and completed experiments immutable."""
import sys,time,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
from scripts import decota_identity_common_v1 as old
from scripts import tastvg_decota_c1_common_v1 as c1
BASE=ROOT/'artifacts/decota_transform_p0_v1'
PUB=ROOT/'results/decota_transform_p0/2026-10-05'
DATASETS=['vidstg','hc2']
OWN=['vg_tta/decota_transform_p0_v1.py','scripts/decota_transform_common_v1.py',
     'scripts/run_decota_transform_p0_v1.py','scripts/test_decota_transform_p0_v1.py',
     'protocols/decota_transform_p0_v1.md']

def verify():
    z=read(BASE/'RUNTIME_LOCK.json');pins=dict(z['pins'])
    for f in sorted((BASE/'revisions').glob('*.json')):pins.update(read(f)['pin_overrides'])
    for f,h in {**pins,**z['inputs'],**z['protected']}.items():assert sha(ROOT/f)==h,f
    return z

def checked(f):
    q=read(f.with_suffix('.json'));assert q['sha256']==sha(f) and not q['GT_read']
    assert q['runtime_lock_sha256']==sha(BASE/'RUNTIME_LOCK.json');return load(f)

def commit(f,x):
    save(f,x);write(f.with_suffix('.json'),dict(sha256=sha(f),GT_read=False,time=time.time(),runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json')))

def budget():assert shutil.disk_usage(ROOT).free>8*2**30

def archive(event):
    f=ROOT/'docs/RESEARCH_HISTORY.md'
    e=('**2026-10-05｜DeCoTA真实输入变换两P0：'+event+'。** '+
       '旧dual-offset几何重心负结果不重跑；新时间前补1/8首帧与保留Native支持的半上下文crop，真实encoder/双offset原读出后逆映射，Native+shift+crop端点中位数固定。'+
       '空间新horizontal-flip与亮度.95两输入，复用封存Uniform4/单DINO admitted Top1/Adam.03/joint1792选态，在相同新prefix上Native、episodic和实际online100到达前/选态只做suffix前向；无新专家/优化/写入。'+
       '沿用各32开发+16历史曝光来源、一query/双序clean+五5%，576唯一输入；全新预测与无标签稳定性封存后GT dense/source-macro/10000source-bootstrap及corr/AUC。'+
       '裁剪不是已知GT完整支持，flip对方向文本可能改变语义；完整保留并分组，不把已知坐标律称模型正确性。双P0未资格化则条件更新/gate跳过，研究工作点保留而CURRENT不晋升；不启动全量或旧队列。'+
       '见[协议](</home/wwww/visual grounding/protocols/decota_transform_p0_v1.md>)、[状态](</home/wwww/visual grounding/artifacts/decota_transform_p0_v1/STATUS.json>)。')
    t=f.read_text();f.write_text(t.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+e+'\n',1)+'\n\n### Real transform P0 update\n\n'+e+'\n')
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'ARCHIVE.log').open('a') as out:
        for a in ['check','snapshot','check']:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',a],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
