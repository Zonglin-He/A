"""Isolated, finite evaluation; no changes to earlier experiments or registries."""
import sys, time, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, save, load, sha
BASE = ROOT/'artifacts/tastvg_decota_c1_same_domain_v1'
PUB = ROOT/'results/tastvg_decota_c1_same_domain/2026-10-04'
POOL = ROOT/'artifacts/tastvg_extended_sensitivity_v3'
VIEW = ROOT/'artifacts/tastvg_current_correction_views_v1'
DATASETS = ['vidstg', 'hc2']


def verify():
    p = read(BASE/'RUNTIME_LOCK.json')
    pins = dict(p['pins'])
    for f in sorted((BASE/'revisions').glob('*.json')):
        pins.update(read(f)['pin_overrides'])
    for f, h in {**pins, **p['inputs']}.items():
        assert sha(ROOT/f) == h, f
    assert sha(ROOT/'methods/CURRENT_METHOD.json') == p['CURRENT_METHOD_sha256']
    return p


def budget():
    assert shutil.disk_usage(ROOT).free > 8*2**30, 'free disk below 8 GiB'


def guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        s = str(args[0])
        if any(x in s for x in ['GT_LABELS', 'GT_SUBSET', 'labels_diagnostic', '/ROWS.json', '/SUMMARY.json', '/results/', '/annotations/', '/annos/', 'valv2_proc.json', 'vidstd-test-anno']):
            raise PermissionError('C1 prediction worker forbids GT and scored outcomes')


def commit(path, value):
    save(path, value)
    write(path.with_suffix('.json'), dict(sha256=sha(path), GT_read=False,
        runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'), time=time.time()))


def checked(path):
    r = read(path.with_suffix('.json'))
    assert sha(path) == r['sha256'] and r['GT_read'] is False
    assert r['runtime_lock_sha256'] == sha(BASE/'RUNTIME_LOCK.json')
    return load(path)


def cache(dataset, parent, condition, *, content=True):
    pool = POOL/dataset
    f = pool/'capture'/condition/f'{parent:05}.json'
    r = read(f)
    barrier = read(pool/'CAPTURE_BARRIER.json')
    assert sha(f) == barrier['files'][str(f.relative_to(pool))]
    cf = pool/r['cache']
    assert sha(cf) == r['sha256'] and not r['GT_read']
    assert r['checkpoint_state_sha256'] == barrier['checkpoint_state_sha256']
    return (load(cf) if content else None), r


def verify_seal():
    verify()
    b = read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    assert b['status'] == 'sealed' and not b['GT_read'] and b['arrivals'] == 1152
    for f, h in b['files'].items():
        assert sha(BASE/f) == h, f
    return b


def archive(event):
    import subprocess
    f = ROOT/'docs/RESEARCH_HISTORY.md'
    e = ('**2026-10-04｜C1/Scale06 online DeCoTA同域corruption：'+event+'。** '
         '用户已确认后来锁定的online版本；不是早期coverage-only，也不新造持续temporal head。'
         '官方Vid/HC2同域EMA、各32开发+16历史曝光来源、一query双序clean+五5%共1152到达；独立split/condition/order链。'
         '空间1792参数/原Scale06 Adam.03十步/原参考loss最优步/query与Adam重置/LN1/16持续；原DINO每query最多4帧，非A的25%专家预算。'
         '原temporal66306参数NLL+hinge/AdamW五步/eta.25逐query丢弃，Vid lr.1 center.5、HC lr.001 center1，不重调。'
         '所有预测封存后官方dense/source-macro/10000paired bootstrap与独立算术审计；HC为固定Vid空间规则的新评估，非已选HC冠军。'
         '不恢复旧队列、不晋升CURRENT。见[协议](</home/wwww/visual grounding/protocols/tastvg_decota_c1_same_domain_v1.md>)、'
         '[状态](</home/wwww/visual grounding/artifacts/tastvg_decota_c1_same_domain_v1/STATUS.json>)。')
    s = f.read_text()
    f.write_text(s.replace('## 1. 当前状态：先读这一节\n', '## 1. 当前状态：先读这一节\n\n'+e+'\n', 1)
        +'\n\n### C1/Scale06 within-domain evaluation update\n\n'+e+'\n')
    with (BASE/'ARCHIVE.log').open('a') as log:
        for action in ['check', 'snapshot', 'check']:
            subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'), '-B', 'scripts/research_archive.py', action],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

