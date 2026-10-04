"""Isolated negative-evidence research; old A assets are immutable inputs."""
import sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,load,save,sha
BASE=ROOT/'artifacts/tastvg_negative_evidence_v1'
PUB=PUBLIC=ROOT/'results/tastvg_negative_evidence/2026-10-04'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
VIEW=ROOT/'artifacts/tastvg_current_correction_views_v1'
ARMS=['rank_native','rank_cross','negative_global','negative_local']
DATASETS=['vidstg','hc2']
BUNDLES={'vidstg':dict(lr=.033761698432507946,teacher_temperature=.34902548789596055,steps=1),
         'hc2':dict(lr=.006097133675874025,teacher_temperature=1.,steps=8)}
def key(c):return '/'.join(str(c[x]) for x in ['dataset','split','condition','order'])+f"/{c['arrival']:05}"
def local_payload_path(c):return BASE/'local'/key(c)
def reset_payload_path(c):return BASE/'reset_u'/key(c)
def plan(ds):return read(VIEW/ds/'PLAN.json')
def budget():assert shutil.disk_usage(ROOT).free>8*2**30
def verify():
    p=read(BASE/'RUNTIME_LOCK.json');pins=dict(p['pins'])
    for f in sorted((BASE/'revisions').glob('*.json')):pins.update(read(f)['pin_overrides'])
    for f,h in {**pins,**p['inputs']}.items():assert sha(ROOT/f)==h,f
    assert sha(ROOT/'methods/CURRENT_METHOD.json')==p['CURRENT_METHOD_sha256']
    return p
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        if any(s in str(args[0]) for s in ['GT_LABELS','GT_EXPOSURE','GT_SUBSET','labels_diagnostic','/ROWS.json','/SUMMARY.json','/results/','/annos/','/annotations/','test_annotations.json','valv2_proc.json','vidstd-test-anno']):
            raise PermissionError('Negative-evidence prediction worker cannot access GT or scores')
def commit(path,value):
    path=Path(str(path)+'.pt');save(path,value);write(path.with_suffix('.json'),dict(sha256=sha(path),GT_read=False,time=time.time(),runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json')))
def checked(path):
    path=Path(str(path)+'.pt');r=read(path.with_suffix('.json'));assert sha(path)==r['sha256'] and r['GT_read'] is False;return load(path)
def verify_seal():
    verify();b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert b['status']=='sealed' and b['GT_read'] is False
    for f,h in b['files'].items():assert sha(BASE/f)==h,f
    return b
def oldcell(c):
    p=ROOT/c['old_payload'];assert sha(p)==read(p.with_suffix('.json'))['sha256'];return load(p)
def expert(c):
    f=POOL/c['dataset']/'experts/spatial'/c['condition']/f"{c['parent']:05}.json";r=read(f);cf=f.parents[2]/r['cache']
    assert r['pixel_sha256']==c['pixel_sha256'] and sha(cf)==r['cache_sha256'];return load(cf),dict(path=str(cf.relative_to(ROOT)),sha256=r['cache_sha256'],pixel_sha256=r['pixel_sha256'])
def archive(event):
    f=ROOT/'docs/RESEARCH_HISTORY.md';s=f.read_text();e=('**2026-10-04｜负证据保留分布与参数包匹配：'+event+'。** '
        '附件5ea0fbf授权：沿用各32开发+16来源互斥确认/一query/双序六条件/25%专家/历史曝光/官方同域checkpoint/原Paper48像素和1792空间参数。'
        '288同A更新前专家位置比较原参数Rank、互换lr-T-K参数包Rank、固定本包的负证据fullclip与同观察帧目标；lambda1有限惩罚不扫参，逐有效帧相对中心IoU正差再取均值，无证据显式no-op。'
        '每K步固定source半径9probe刷新，一次写后下一非专家同pre隔离transfer不冒充真实长期；另1152原Rank独立reset-query-residual/LN持续在线删除对照，不与loss捆绑。'
        '全部部署预测封存后GT诊断/官方dense/source配对bootstrap，保存输出梯度/7参数块/4block-only反事实/观察与未观察效用；不新专家/backbone/temporal更新/新选择器，不恢复旧队列不改CURRENT。'
        '见[协议](</home/wwww/visual grounding/protocols/tastvg_negative_evidence_v1.md>)、[状态](</home/wwww/visual grounding/artifacts/tastvg_negative_evidence_v1/STATUS.json>)。')
    f.write_text(s.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+e+'\n',1)+'\n\n### Negative evidence matched execution update\n\n'+e+'\n')
