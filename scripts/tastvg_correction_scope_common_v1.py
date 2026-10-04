"""A-only correction/lifecycle audit; prior artifacts are immutable inputs."""
import sys, time, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, load, save, sha
from scripts.tastvg_negative_evidence_common_v1 import (
    BUNDLES, POOL, VIEW, plan, key, oldcell, expert,
    checked as oldchecked, local_payload_path,
)
BASE = ROOT / 'artifacts/tastvg_correction_scope_v1'
PUB = ROOT / 'results/tastvg_correction_scope/2026-10-04'
DATASETS = ('vidstg', 'hc2')
BLOCKS = ('query', 'norm1', 'norm3', 'norm4', 'full')
SCOPES = ('self', 'same_video_other_query', 'different_video_same_corruption',
          'semantic_near', 'semantic_far', 'different_corruption')

def budget():
    assert shutil.disk_usage(ROOT).free > 8 * 2**30

def guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        if any(s in str(args[0]) for s in ['GT_LABELS', 'GT_EXPOSURE', 'GT_SUBSET',
            'labels_diagnostic', '/ROWS.json', '/SUMMARY.json', '/results/',
            '/annos/', '/annotations/', 'test_annotations.json', 'valv2_proc.json']):
            raise PermissionError('Correction-scope prediction worker cannot read GT/outcomes')

def verify():
    lock = read(BASE / 'RUNTIME_LOCK.json')
    pins = dict(lock['pins'])
    for f in sorted((BASE / 'revisions').glob('*.json')):
        pins.update(read(f)['pin_overrides'])
    for f, h in {**pins, **lock['inputs']}.items():
        assert sha(ROOT / f) == h, f
    assert sha(ROOT / 'methods/CURRENT_METHOD.json') == lock['CURRENT_METHOD_sha256']
    return lock

def commit(f, value):
    f = Path(f)
    save(f, value)
    write(f.with_suffix('.json'), dict(sha256=sha(f), GT_read=False,
          runtime_lock_sha256=sha(BASE / 'RUNTIME_LOCK.json'), time=time.time()))

def checked(f):
    f = Path(f)
    r = read(f.with_suffix('.json'))
    assert r['GT_read'] is False and sha(f) == r['sha256']
    return load(f)

def capture(ds, parent, cond):
    f = POOL / ds / 'capture' / cond / f'{parent:05}.json'
    bar = read(POOL / ds / 'CAPTURE_BARRIER.json')
    assert sha(f) == bar['files'][str(f.relative_to(POOL / ds))]
    r = read(f); cf = POOL / ds / r['cache']
    assert sha(cf) == r['sha256']
    return load(cf), r

def matrix_path(c):
    return BASE / 'matrix' / (key(c) + '.pt')

def episode_path(ds, parent, cond):
    return BASE / ds / 'episodic' / cond / f'{parent:05}.pt'

def alt_path(parent, cond):
    return BASE / 'vidstg' / 'alternative_capture' / cond / f'{parent:05}.pt'

def seal():
    verify()
    b = read(BASE / 'GLOBAL_PREDICTION_BARRIER.json')
    assert b['status'] == 'sealed' and b['GT_read'] is False
    for f, h in b['files'].items():
        assert sha(BASE / f) == h, f
    return b

def archive(event):
    f = ROOT / 'docs/RESEARCH_HISTORY.md'; s = f.read_text()
    e = ('**2026-10-04｜Episodic Rank-RKL 与 correction×scope：' + event + '。** '
         '附件000d5a5b授权；固定旧Sa2VA Rank-RKL A，非新C1/Scale06。各32开发+16历史曝光确认、'
         '一query/双序/clean+五5%/25%专家/同域checkpoint/1792空间参数/VidK1 HC K8及原lr-T。'
         '重跑source完整空间reset→当前post输出；复用288原A全步写，query/norm1/norm3/norm4/full'
         '分别作用于self、同物理视频其他query、同corruption未来非专家、语义近/远和匹配跨corruption目标；'
         '同donorpre基准与固定时间计分，不相加块效用，不把孤立transfer称长期online。'
         'Vid真实另query以hash选；HC相同媒体无另query，明确不可测。语义锁使用冻结RoBERTa，'
         '零新专家；新增仅Vid另query的冻结H及后缀重放，预测全封存后GT。无GT路由、无新loss/搜索、'
         '无dataset规则晋升，不恢复旧队列不改CURRENT。'
         '见[协议](</home/wwww/visual grounding/protocols/tastvg_correction_scope_v1.md>)、'
         '[状态](</home/wwww/visual grounding/artifacts/tastvg_correction_scope_v1/STATUS.json>)。')
    f.write_text(s.replace('## 1. 当前状态：先读这一节\n',
        '## 1. 当前状态：先读这一节\n\n' + e + '\n', 1)
        + '\n\n### Correction scope execution update\n\n' + e + '\n')
