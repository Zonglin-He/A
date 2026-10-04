"""Isolated P0 receipts and data boundaries; no old queue continuation."""
import sys, time, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, save, load, sha
BASE = ROOT/'artifacts/tastvg_privileged_attention_p0_v1'
PUB = ROOT/'results/tastvg_privileged_attention_p0/2026-10-04'
PARENT = ROOT/'artifacts/tastvg_decota_c1_same_domain_v1'
POOL = ROOT/'artifacts/tastvg_extended_sensitivity_v3'
DATASETS = ['vidstg', 'hc2']
OWN = ['protocols/tastvg_privileged_attention_p0_v1.md',
       'vg_tta/tastvg_privileged_attention_p0_v1.py',
       'scripts/tastvg_privileged_p0_common_v1.py',
       'scripts/run_tastvg_privileged_attention_p0_v1.py',
       'scripts/test_tastvg_privileged_p0_v1.py']


def verify():
    p = read(BASE/'RUNTIME_LOCK.json'); pins = dict(p['pins'])
    for f in sorted((BASE/'revisions').glob('*.json')): pins.update(read(f)['pin_overrides'])
    for f, h in {**pins, **p['inputs']}.items(): assert sha(ROOT/f) == h, f
    assert sha(ROOT/'methods/CURRENT_METHOD.json') == p['CURRENT_METHOD_sha256']
    return p


def budget():
    assert shutil.disk_usage(ROOT).free > 8 * 2**30


def commit(f, x):
    save(f, x); write(f.with_suffix('.json'), dict(sha256=sha(f), GT_read=False,
        runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'), time=time.time()))


def checked(f):
    r = read(f.with_suffix('.json')); assert sha(f) == r['sha256'] and not r['GT_read']
    assert r['runtime_lock_sha256'] == sha(BASE/'RUNTIME_LOCK.json')
    return load(f)


def verify_seal():
    verify(); p = read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    assert p['status'] == 'sealed' and p['cells'] == 192 and not p['GT_read']
    for f, h in p['files'].items(): assert sha(BASE/f) == h, f
    return p


def archive(event):
    import subprocess
    f = ROOT/'docs/RESEARCH_HISTORY.md'
    entry = ('**2026-10-04｜Event-scoped privileged spatial attention P0：'+event+'。** '
        '附件f7175e26分阶段授权；各8开发+8历史曝光确认来源/一query/clean+五5%共192输入，官方Vid/HC2同域EMA与原Paper48网格/像素。'
        '每query native区间内最多4原帧，DINO全部phrase>=.35有效框/无topK-NMS-margin伪标签，softmaxT1高斯half-box prior/alpha1/eps1e-6固定。'
        '仅同模型最终6层空间attention logits加bias，特征/时间/权重冻结，无backward/SGD/OPD/LN持续；全部封存后GT官方dense/源配对10000bootstrap。'
        '不是A8、非25%专家全流，不外推online；四面板corrupt均正才进入独立episodic阶段，不按结果改prior。'
        '旧队列与CURRENT不动；见[协议](</home/wwww/visual grounding/protocols/tastvg_privileged_attention_p0_v1.md>)、'
        '[状态](</home/wwww/visual grounding/artifacts/tastvg_privileged_attention_p0_v1/STATUS.json>)。\n\n')
    text = f.read_text(); marker = '## 1. 当前状态：先读这一节'
    pos = text.find('\n', text.index(marker)) + 1
    text = text[:pos] + '\n' + entry + text[pos:]
    text += '\n\n### Privileged spatial attention P0 update\n\n' + entry
    f.write_text(text)
    with open(BASE/'ARCHIVE.log', 'a') as log:
        for op in ['check', 'snapshot', 'check']:
            subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'), '-B',
                'scripts/research_archive.py', op], cwd=ROOT, check=True, stdout=log, stderr=log)
