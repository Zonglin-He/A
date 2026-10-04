"""Bounded P0 paths, hashes and no-GT worker boundaries."""
import sys, shutil, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, save, load, sha
BASE = ROOT / 'artifacts/tastvg_decota_critic_p0_v1'
OLD = ROOT / 'artifacts/tastvg_decota_c1_same_domain_v1'
POOL = ROOT / 'artifacts/tastvg_extended_sensitivity_v3'
PUB = ROOT / 'results/tastvg_decota_critic_p0/2026-10-04'
DATASETS = ['vidstg', 'hc2']
ARMS = ['direct', 'critic']


def verify():
    lock = read(BASE / 'RUNTIME_LOCK.json')
    pins = dict(lock['pins'])
    for f in sorted((BASE / 'revisions').glob('*.json')):
        pins.update(read(f)['pin_overrides'])
    for f, h in {**pins, **lock['inputs']}.items():
        assert sha(ROOT / f) == h, f
    for f, h in lock['protected_registries'].items():
        assert sha(ROOT / f) == h, f
    return lock


def budget():
    assert shutil.disk_usage(ROOT).free > 8 * 2**30, 'Free disk below 8 GiB'


def guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        p = args[0].decode() if isinstance(args[0], bytes) else args[0]
        if any(s in p for s in ['GT_LABELS', 'GT_SUBSET', 'labels_diagnostic',
            '/ROWS.json', '/SUMMARY.json', '/results/', '/annotations/', '/annos/',
            'valv2_proc.json', 'vidstd-test-anno']):
            raise PermissionError('Critic P0 prediction worker forbids GT/results: ' + p)


def commit(path, value):
    save(path, value)
    write(path.with_suffix('.json'), dict(sha256=sha(path), GT_read=False,
        runtime_lock_sha256=sha(BASE / 'RUNTIME_LOCK.json'), time=time.time()))


def checked(path):
    receipt = read(path.with_suffix('.json'))
    assert receipt['GT_read'] is False and sha(path) == receipt['sha256'], str(path)
    assert receipt['runtime_lock_sha256'] == sha(BASE / 'RUNTIME_LOCK.json')
    return load(path)


def verify_seal():
    verify()
    barrier = read(BASE / 'GLOBAL_PREDICTION_BARRIER.json')
    assert barrier['status'] == 'sealed' and barrier['GT_read'] is False
    assert barrier['unique_inputs'] == 576 and barrier['logical_arrivals'] == 1152
    for f, h in barrier['files'].items():
        assert sha(BASE / f) == h, f
    return barrier


def archive(event):
    f = ROOT / 'docs/RESEARCH_HISTORY.md'
    entry = ('**2026-10-04｜Spatial-DeCoTA Direct/critic P0：' + event + '。** '
        '附件aac11155授权以C1–Scale06为母体，仅本轮去除temporal适应与LN继承，'
        '两集原32开发+16历史曝光来源/一query/双序/clean+五5%，1152逻辑到达；'
        '576唯一输入各Direct/critic两臂源状态独立重置，1792参数/Adam.03十步/0..10自身目标最优步。'
        '复用原C1四观察DINO/NMS支持与Frozen H；Direct保留原接纳与5L1+2GIoU/planned4，'
        'critic用非空有效NMS<=3支持、score-softmax温度1及IoU-logsumexp能量温度1/有效帧均值；'
        'fallback仅有缓存单支持，零新专家，不补造多候选。当前输出固定源native I0。'
        '监督构造同时改变支持/接纳/目标，非仅公式因果对照；energy仍是DINO框代理、不称已成立OPD/摆脱伪监督。'
        '全预测seal后GT；不复用旧online after为episodic成绩、不按GT选步/温度/来源。'
        '旧C1/NLL锁、CURRENT与旧预测不改；不启动P1继承/cross-domain/teacher/gate/新搜索。'
        '见[协议](</home/wwww/visual grounding/protocols/tastvg_decota_critic_p0_v1.md>)、'
        '[状态](</home/wwww/visual grounding/artifacts/tastvg_decota_critic_p0_v1/STATUS.json>)。')
    text = f.read_text()
    assert '## 1. 当前状态：先读这一节\n' in text
    f.write_text(text.replace('## 1. 当前状态：先读这一节\n',
        '## 1. 当前状态：先读这一节\n\n' + entry + '\n', 1)
        + '\n\n### Spatial-DeCoTA critic P0 update\n\n' + entry + '\n')
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / 'ARCHIVE.log').open('a') as log:
        for action in ['check', 'snapshot', 'check']:
            subprocess.run([str(ROOT / '.conda/tubedetr/bin/python'), '-B',
                'scripts/research_archive.py', action], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, check=True)
