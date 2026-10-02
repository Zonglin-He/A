"""Isolated same-state Uniform/Routed gradient diagnostic."""
import sys
import time
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, save, load, sha

BASE = ROOT / 'artifacts/tastvg_ur_write_decomposition_v1'
OLD = ROOT / 'artifacts/tastvg_routed_online_token_v1'
POOL = ROOT / 'artifacts/tastvg_extended_sensitivity_v3'
CONTEXT = ROOT / 'artifacts/tastvg_transfer_aligned_v1/transfer/TEXT_CONTEXT.pt'
PUBLIC = ROOT / 'results/tastvg_ur_write_decomposition/2026-10-02'


def budget():
    assert shutil.disk_usage(ROOT).free > 8 * 2**30, 'free disk below 8 GiB'


def verify():
    lock = read(BASE / 'RUNTIME_LOCK.json')
    pins = dict(lock['pins'])
    for revision in sorted((BASE / 'revisions').glob('*.json')):
        pins.update(read(revision)['pin_overrides'])
    for path, expected in {**pins, **lock['inputs']}.items():
        assert sha(ROOT / path) == expected, path
    return lock


def guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        name = str(args[0])
        forbidden = ['GT_LABELS', 'GT_EXPOSURE', 'labels_diagnostic', 'GT_SUBSET',
                     '/ROWS.json', '/SUMMARY.json', '/results/', 'test_annotations.json',
                     'valv2_proc.json', 'vidstd-test-anno']
        if any(token in name for token in forbidden):
            raise PermissionError('U/R model worker cannot read labels or result scores')


def commit(path, value):
    save(path, value)
    write(path.with_suffix('.json'), dict(sha256=sha(path), GT_read=False,
          runtime_lock_sha256=sha(BASE / 'RUNTIME_LOCK.json'), time=time.time()))


def receipt(path):
    record = read(path.with_suffix('.json'))
    assert sha(path) == record['sha256'] and record['GT_read'] is False
    return record


def archive(event):
    import subprocess
    path = ROOT / 'docs/RESEARCH_HISTORY.md'
    body = path.read_text()
    entry = ('**2026-10-02｜同状态U/R write分解：' + event + '。** '
             '用户58e6d040授权HC原32曝光源/一query、双序clean+五类5%，96专家donor；'
             '每个donor共用历史R更新前1792参数状态与九probe，原rank-RKL/lr.006097133675874025/'
             'teacher1/student1/rho.05/D4，仅匹配第一步梯度U/R及R−U，不将K8轨迹混作同状态。'
             'self/next/text-near/text-far与全部后续nonexpert目标事前冻结，所有预测seal后GT，'
             'donor源聚类配对10000bootstrap；无新专家/backbone/完整online，不改CURRENT，'
             '不将几何兼容分布称原生tube policy，历史任务保持暂停。'
             '见[协议](</home/wwww/visual grounding/protocols/tastvg_ur_write_decomposition_v1.md>)、'
             '[状态](</home/wwww/visual grounding/artifacts/tastvg_ur_write_decomposition_v1/STATUS.json>)。')
    path.write_text(body.replace('## 1. 当前状态：先读这一节\n',
          '## 1. 当前状态：先读这一节\n\n' + entry + '\n', 1)
          + '\n\n### Same-state U/R write decomposition event\n\n' + entry + '\n')
    with (BASE / 'ARCHIVE.log').open('a') as out:
        for action in ['check', 'snapshot', 'check']:
            subprocess.run([str(ROOT / '.conda/tubedetr/bin/python'), '-B',
                'scripts/research_archive.py', action], cwd=ROOT,
                stdout=out, stderr=subprocess.STDOUT, check=True)
