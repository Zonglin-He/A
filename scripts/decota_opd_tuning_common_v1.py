"""Finite search metadata and narrowly scoped original-baseline verification."""
import os,sys,json,time,hashlib,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,save,load
BASE=ROOT/'artifacts/decota_spatial_opd_tuning_v1'
OLD=ROOT/'artifacts/decota_spatial_opd_v1'
PAPER=ROOT/'artifacts/decota_paper_experiments_v1'
PUB=ROOT/'results/decota_spatial_opd_tuning/2026-10-06'
DEFAULT=dict(lr=.03,sigma=.25,tau=.25,steps=10,writeback=1/16,samples=32)
SPACE=dict(lr=[.001,.003,.01,.03,.1],sigma=[.1,.25,.5],tau=[.1,.25,.5,1.],steps=[3,5,10,20],writeback=[0.,1/32,1/16,1/8])

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def budget():assert shutil.disk_usage(ROOT).free>8*2**30

def verify_paper():
    original=read(PAPER/'RUNTIME_LOCK.json');auth=read(BASE/'USER_SCOPE.json')
    assert sha(PAPER/'RUNTIME_LOCK.json')==auth['original_paper_runtime_sha256']
    pins=dict(original['pins'])
    for f in sorted((PAPER/'revisions').glob('*.json')):pins.update(read(f)['pin_overrides'])
    allfiles={**pins,**original['inputs'],**original['protected']}
    allfiles['methods/CURRENT_METHOD.json']=auth['new_main_registry_sha256']
    for f,h in allfiles.items():assert sha(ROOT/f)==h,f
    return original

def bridge():
    # Preserve original functions and scientific pins. Only the expressly
    # authorized main registry has a new expected hash in this process.
    import scripts.decota_paper_common_v1 as paper
    paper.verify=verify_paper

def verify():
    verify_paper();p=read(BASE/'RUNTIME_LOCK.json')
    assert sha(BASE/'DESIGN_LOCK.json')==p['design_sha256']
    for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
    return p

def archive(event):
    f=ROOT/'docs/RESEARCH_HISTORY.md';s=f.read_text()
    line='**2026-10-06｜用户登记空间OPD为主方法并限定调参→baseline顺序：'+event+'。** Native-WHEN/原Uniform4/单DINO/1792/on-policy Gaussian detached反馈不改结构；各数据集统一参数，16个单因素筛查+12次敏感坐标开发搜索，旧128确认不用于选参。其他方法实验暂停；旧paper只接续baseline，原HC源媒体依赖保留。用户登记不等于HC已测稳定；旧配置Vid+3.667pp、HC+0.522pp且HC current -1.220pp的确认负尾保留。清理仅旧大payload，报告/配置/匿名科学摘要和baseline状态保留。见[协议](</home/wwww/visual grounding/protocols/decota_spatial_opd_tuning_v1.md>)、[状态](</home/wwww/visual grounding/artifacts/decota_spatial_opd_tuning_v1/STATUS.json>)。'
    f.write_text(s.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+line+'\n',1)+'\n\n### Spatial OPD main-method tuning update\n\n'+line+'\n')
    with (BASE/'ARCHIVE.log').open('a') as log:
        for a in ['check','snapshot','check']:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',a],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def prepare():
    if (BASE/'DESIGN_LOCK.json').exists():return read(BASE/'DESIGN_LOCK.json')
    old=read(OLD/'DESIGN_LOCK.json');datasets={}
    for ds in ['vidstg','hc2']:
        dev=old['stages']['dev_'+ds];screen=sorted(dev['parents'],key=lambda p:digest(['opd-sensitivity-screen-v1',ds,p]))[:16]
        datasets[ds]=dict(dataset=ds,source=dev['source'],development_parents=dev['parents'],
            screen_parents=screen,screen_orders={k:[x for x in v if x in screen] for k,v in dev['orders'].items()},
            refine_orders=dev['orders'],old_confirmation_used_for_selection=False,historical_exposure=True)
    configs=[dict(config=DEFAULT,factor='reference')]
    for factor,vals in SPACE.items():
        for val in vals:
            if val!=DEFAULT[factor]:configs.append(dict(config={**DEFAULT,factor:val},factor=factor))
    assert len(configs)==16
    d=dict(version='decota_spatial_opd_tuning_v1',datasets=datasets,space=SPACE,reference=DEFAULT,
           screen_configs=configs,refine_trials_per_dataset=12,seed=20261006,
           selection='development source-macro delta_v; exact ties harm20 then cost',
           execution_order='tuning_then_saved_baselines_only',new_expert=False,new_method_experiments=False,time=time.time())
    write(BASE/'DESIGN_LOCK.json',d);return d

def lock():
    prepare()
    if (BASE/'RUNTIME_LOCK.json').exists():return verify()
    files=['scripts/decota_opd_tuning_common_v1.py','scripts/run_decota_opd_tuning_v1.py',
           'scripts/score_decota_opd_tuning_v1.py','scripts/continue_decota_opd_tuning_v1.py',
           'scripts/continue_decota_paper_baselines_only_v2.py','scripts/decota_baseline_bridge_v2.py',
           'scripts/test_decota_opd_tuning_v1.py','protocols/decota_spatial_opd_tuning_v1.md',
           'vg_tta/decota_spatial_opd_tunable_v1.py','vg_tta/decota_spatial_opd_tunable_audit_v1.py',
           'methods/decota_spatial_opd_v1/__init__.py','methods/decota_spatial_opd_v1/predictor.py',
           'methods/decota_spatial_opd_v1/README.md']
    write(BASE/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in files},design_sha256=sha(BASE/'DESIGN_LOCK.json'),
          original_OPD_runtime_sha256=sha(OLD/'RUNTIME_LOCK.json'),original_baseline_runtime_sha256=sha(PAPER/'baselines/RUNTIME_LOCK.json'),time=time.time()))
    return verify()
