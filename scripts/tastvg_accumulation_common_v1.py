"""Isolated saved-delta audit; historical assets are immutable inputs."""
import sys
import time
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, save, load, sha

BASE = ROOT / 'artifacts/tastvg_accumulation_audit_v1'
OLD = ROOT / 'artifacts/tastvg_routed_online_token_v1'
POOL = ROOT / 'artifacts/tastvg_extended_sensitivity_v3'
PUBLIC = ROOT / 'results/tastvg_accumulation_audit/2026-10-02'


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
        forbidden = ['GT_LABELS', 'GT_EXPOSURE', 'labels_diagnostic',
                     'GT_SUBSET', '/ROWS.json', '/SUMMARY.json', '/results/',
                     'test_annotations.json', 'valv2_proc.json', 'vidstd-test-anno']
        if any(token in name for token in forbidden):
            raise PermissionError('Saved-write model worker cannot read labels or scores')


def commit(path, payload):
    save(path, payload)
    write(path.with_suffix('.json'), dict(sha256=sha(path), GT_read=False,
                                         runtime_lock_sha256=sha(BASE / 'RUNTIME_LOCK.json'),
                                         time=time.time()))


def receipt(path):
    record = read(path.with_suffix('.json'))
    assert sha(path) == record['sha256'], str(path)
    assert record['GT_read'] is False
    return record


def archive(event):
    import subprocess
    path = ROOT / 'docs/RESEARCH_HISTORY.md'
    body = path.read_text()
    entry = ('**2026-10-02｜HC saved-write accumulation：' + event + '。** '
             '用户 b620b0e1 授权原 HC32 A/R 双序六条件、每臂288 future nonexpert '
             'target 的 Source/All/Last 保存写入反事实；FP64差分累加后FP32精确复现旧状态，'
             '零新专家/学习/backbone。所有预测先封存再GT，source-macro配对bootstrap；'
             '历史曝光、不是 exact-reset online 方法，不改 CURRENT。另只核验现有视频token接口，'
             '不以静态patch/pooled temporal向量冒充空间时间对齐。'
             '见[协议](</home/wwww/visual grounding/protocols/tastvg_accumulation_audit_v1.md>)、'
             '[状态](</home/wwww/visual grounding/artifacts/tastvg_accumulation_audit_v1/STATUS.json>)。')
    body = body.replace('## 1. 当前状态：先读这一节\n',
                        '## 1. 当前状态：先读这一节\n\n' + entry + '\n', 1)
    path.write_text(body + '\n\n### Saved-write accumulation event\n\n' + entry + '\n')
    with (BASE / 'ARCHIVE.log').open('a') as log:
        for action in ['check', 'snapshot', 'check']:
            subprocess.run([str(ROOT / '.conda/tubedetr/bin/python'), '-B',
                            'scripts/research_archive.py', action], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
