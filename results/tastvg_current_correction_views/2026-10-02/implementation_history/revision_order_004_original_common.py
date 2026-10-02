"""Isolated two-round research scope; old trajectories are immutable inputs."""
import sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,load,save,sha
BASE=ROOT/'artifacts/tastvg_current_correction_views_v1'
OLD=ROOT/'artifacts/tastvg_routed_online_token_v1'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
PUBLIC=ROOT/'results/tastvg_current_correction_views/2026-10-02'
DATASETS=['vidstg','hc2']
ARMS=['A','U_select','R_select','R_temp','Specific_temp']
def budget():assert shutil.disk_usage(ROOT).free>8*2**30
def verify():
    p=read(BASE/'RUNTIME_LOCK.json');pins=dict(p['pins'])
    for r in sorted((BASE/'revisions').glob('*.json')):pins.update(read(r)['pin_overrides'])
    for f,h in {**pins,**p['inputs']}.items():assert sha(ROOT/f)==h,f
    return p
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        if any(x in str(args[0]) for x in ['GT_LABELS','GT_EXPOSURE','labels_diagnostic',
            'GT_SUBSET','/ROWS.json','/SUMMARY.json','/results/','test_annotations.json',
            'valv2_proc.json','vidstd-test-anno']):
            raise PermissionError('Correction model worker cannot access labels or scores')
def commit(path,value):
    save(path,value);write(path.with_suffix('.json'),dict(sha256=sha(path),
        GT_read=False,time=time.time(),runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json')))
def checked(path):
    assert sha(path)==read(path.with_suffix('.json'))['sha256'];return load(path)
def plan(ds):return read(BASE/ds/'PLAN.json')
def oldfile(ds,split,cond,order,at):
    if split=='search':return OLD/ds/'A/online'/cond/order/f'{at:05}.pt'
    arm='anchor' if ds=='vidstg' else 'selected'
    return POOL/ds/'confirmation'/arm/'online'/cond/order/f'{at:05}.pt'
def oldcell(ds,split,cond,order,at):
    f=oldfile(ds,split,cond,order,at);r=read(f.with_suffix('.json'))
    assert sha(f)==r['sha256'];return load(f)
def expert(ds,stage,parent,cond,pixel):
    f=POOL/ds/'experts'/stage/cond/f'{parent:05}.json';r=read(f);cf=POOL/ds/'experts'/r['cache']
    assert sha(cf)==r['cache_sha256'] and r['pixel_sha256']==pixel
    return load(cf),dict(cache=str(cf.relative_to(ROOT)),cache_sha256=r['cache_sha256'],
        input_sha256=load(cf)['input_sha256'],pixel_sha256=pixel,origin='cached_uniform',
        seconds=0.,new_call=False,positions=load(cf).get('positions'))
def key(ds,split,cond,order,at):return f'{ds}/{split}/{cond}/{order}/{at:05}'
def archive(event):
    import subprocess
    f=ROOT/'docs/RESEARCH_HISTORY.md';s=f.read_text();e=('**2026-10-02｜当前纠正与持续学习分离：'+event+'。** '
        '用户最新授权两轮：同一实际A更新前状态及九probe，对照A/Uniform直接选择/Routed直接选择/'
        'Routed临时SGD/Specific临时SGD，所有持续状态仍为Uniform A；两真实2Hz时间采样相位0/.25秒，'
        'min-view相对native规则，输出与五帧取样分别对照，补第二组Uniform5预算控制。'
        '各32开发源/16本轮源互斥确认、双序clean+五5%/25%专家/1792参数/VidK1 HC K8固定，'
        '全部历史曝光；当前临时纠正仅一query，不继承；分阶段预测seal后GT，确认不重选，'
        '保留成本/负尾/混合结论，不恢复旧fullquery，不改生产。'
        '见[协议](</home/wwww/visual grounding/protocols/tastvg_current_correction_views_v1.md>)、'
        '[状态](</home/wwww/visual grounding/artifacts/tastvg_current_correction_views_v1/STATUS.json>)。')
    f.write_text(s.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+e+'\n',1)
        +'\n\n### Current correction / persistent A execution event\n\n'+e+'\n')
    with (BASE/'ARCHIVE.log').open('a') as log:
        for a in ['check','snapshot','check']:
            subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',a],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
