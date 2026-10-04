"""Isolated clean cross-dataset A qualification; no historical queue is resumed."""
import sys, time, shutil, gzip
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, sha, save, load
BASE = ROOT / 'artifacts/tastvg_cross_domain_qualification_v1'
PUB = ROOT / 'results/tastvg_cross_domain_qualification/2026-10-04'
DIRECTIONS = ('vid_to_hc2', 'hc2_to_vid')
BUNDLES = {
    'vidstg': dict(lr=.033761698432507946, teacher_temperature=.34902548789596055,
                  steps=1, rho=.05, direction_count=4, student_temperature=1.),
    'hc2': dict(lr=.006097133675874025, teacher_temperature=1., steps=8,
                rho=.05, direction_count=4, student_temperature=1.),
}
CHECKPOINTS = {'vidstg': 'checkpoints/TASTVG_VidSTG.pth',
               'hc2': 'checkpoints/TASTVG_HCSTVG2.pth'}
MODEL_SHA = {'vidstg': 'fbb1ed8871d6c2aa093879efefc2ee500bb7de5e0fe1c25b809c5393d010f3c7',
             'hc2': 'ee72f0d9a50c573a115bd7cfc2329c860745a1cf3dab83af1be7787c11b3c218'}

def budget(*args):
    assert shutil.disk_usage(ROOT).free > 8 * 2**30, 'Free disk below 8 GiB'

def guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        s = str(args[0])
        if any(t in s for t in ['GT_LABEL', 'GT_EXPOSURE', 'labels_diagnostic',
            '/GT_SUBSET', '/ROWS.json', '/SUMMARY.json', '/results/', '/scorer_only/',
            '/annos/', '/annotations/', 'val_v2.json', 'vidstd-test-anno']):
            raise PermissionError('Cross-domain prediction worker cannot read GT or outcomes')

def verify(direction=None):
    lock = read(BASE / 'RUNTIME_LOCK.json')
    pins = dict(lock['pins'])
    metadata = dict(lock['metadata'])
    for f in sorted((BASE / 'revisions').glob('*.json')):
        pins.update(read(f)['pin_overrides'])
        metadata.update(read(f).get('metadata_overrides',{}))
    for f, h in pins.items():
        assert sha(ROOT / f) == h, f
    for f, h in metadata.items():
        assert sha(BASE / f) == h, f
    assert sha(ROOT / 'methods/CURRENT_METHOD.json') == lock['CURRENT_METHOD_sha256']
    return read(BASE / direction / 'PLAN.json') if direction else lock

def bind_decode(target):
    from vg_tta import exact_frame_decode_audit_v2 as binding
    if target == 'hc2':
        from vg_tta.tastvg_paper48_hc2_decode_v1 import decode
    else:
        from vg_tta.exact_frame_decode_audit_v2 import decode
    binding.decode = decode
    return decode

def model_load(source):
    from scripts.run_spatial_regression_alignment_v1 import model_load as load_model
    return load_model('hcstvg1_test' if source == 'vidstg' else 'vidstg_test')

def commit(f, x, **receipt):
    save(f, x)
    write(Path(f).with_suffix('.json'), dict(sha256=sha(f), GT_read=False,
          time=time.time(), **receipt))

def checked(f):
    assert sha(f) == read(Path(f).with_suffix('.json'))['sha256']
    return load(f)

def savez(f, x):
    import torch
    f = Path(f); f.parent.mkdir(parents=True, exist_ok=True)
    assert not f.exists(), f
    t = f.with_suffix(f.suffix + '.tmp')
    with t.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as z:
            torch.save(x, z)
    t.replace(f)

def loadz(f):
    import torch
    with gzip.open(f, 'rb') as z:
        return torch.load(z, map_location='cpu', weights_only=False)

def seal():
    verify()
    b = read(BASE / 'GLOBAL_PREDICTION_BARRIER.json')
    assert b['status'] == 'sealed' and b['GT_read'] is False
    for direction in DIRECTIONS:
        out = BASE / direction
        f = out / 'PREDICTION_BARRIER.json'
        assert sha(f) == b['directions'][direction]
        for rel, h in read(f)['files'].items():
            assert sha(out / rel) == h, rel
    return b

def archive(event):
    f = ROOT / 'docs/RESEARCH_HISTORY.md'; s = f.read_text()
    e = ('**2026-10-04｜当前 A clean cross-domain qualification：' + event + '。** '
         '附件6f0ec7a7授权；Vid源→HC2目标135源/135query与HC2源→Vid目标384源/384query，'
         '从旧413/707父名单按输入query hash每源一条，三固定序clean共1557到达。'
         '1792空间参数/Uniform5 Sa2VA/Rank-RKL/原Temporal Fast/25%专家不变；'
         '按source锁Vid K1及HC2 K8原lr/T，非target调参。Frozen/Fast-only/Spatial-only/Full A，'
         '另列target-trained监督参考；当前输出封存后写入，预测全封存才GT诊断。'
         '历史曝光，不称fresh，不恢复从未启动的旧full matrix，不晋升CURRENT。'
         '诊断覆盖候选、专家排序、实际更新、正确样本受损与后续迁移；按source配对10000bootstrap。'
         '见[协议](</home/wwww/visual grounding/protocols/tastvg_cross_domain_qualification_v1.md>)、'
         '[状态](</home/wwww/visual grounding/artifacts/tastvg_cross_domain_qualification_v1/STATUS.json>)。')
    f.write_text(s.replace('## 1. 当前状态：先读这一节\n',
        '## 1. 当前状态：先读这一节\n\n' + e + '\n', 1)
        + '\n\n### Clean cross-domain qualification update\n\n' + e + '\n')
