"""Bounded actual memory-engineering receipts/source export, without task GT."""
import ast
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, sha
from scripts.stage_stvg_opd_memory_engineering_public_v1 import metadata
from scripts.publish_stvg_opd_supplement_v1 import safe, private_attachment_path

BASE = ROOT / 'artifacts/stvg_opd_optimization_geometry_v1'
REC = BASE / 'frozen_visual_host011'
PUB = ROOT / 'results/stvg_opd_optimization_geometry/2026-10-12/memory_engineering009_011'
CHECKOUT = ROOT.parent / 'visual-grounding-public-A'
BRANCH = 'research/stvg-opd-paper-hc2-revision-v2'
NAMES = ('frozen_attention_storage009', 'frozen_attention_lifetime010', 'frozen_visual_host011')


def run():
    for name in NAMES:
        assert read(BASE / name / 'ENGINEERING_ROOT_REVIEW.json')['status'] == 'pass'
        runtime = read(BASE / name / 'REVISION_RUNTIME.json')
        for path, digest in runtime['pins'].items():
            assert sha(ROOT / path) == digest, path

    def git(*args, env=None):
        return subprocess.check_output(['git', *args], cwd=CHECKOUT, env=env, text=True).strip()

    assert git('remote', 'get-url', 'origin') == 'https://github.com/Zonglin-He/A.git'
    subprocess.run(['git', 'fetch', '--quiet', 'origin', 'main', BRANCH], cwd=CHECKOUT, check=True)
    base = git('rev-parse', 'HEAD')
    assert base == git('rev-parse', 'origin/main') == git('rev-parse', 'origin/' + BRANCH)
    assert not git('diff', 'HEAD', '--name-only') and not git('diff', '--cached', '--name-only')
    owned = {record['path'] for record in read(BASE / 'frozen_module_CPU008/PUBLIC_STAGE.json')['files']}
    bindings = {}
    sources = {
        'vg_tta/stvg_opd_frozen_attention_storage009.py',
        'vg_tta/stvg_opd_frozen_attention_lifetime010.py',
        'vg_tta/stvg_opd_frozen_visual_host011.py',
        'scripts/run_stvg_opd_common_state_storage009.py',
        'scripts/run_stvg_opd_common_state_lifetime010.py',
        'scripts/run_stvg_opd_common_state_visual_host011.py',
        'scripts/review_stvg_opd_storage009_prefix_v1.py',
        'scripts/review_stvg_opd_lifetime010_prefix_v1.py',
        'scripts/review_stvg_opd_visual_host011_prefix_v1.py',
        'scripts/stage_stvg_opd_memory009_011_public_v1.py',
        'protocols/stvg_opd_storage009_engineering_v1.md',
        'protocols/stvg_opd_lifetime010_engineering_v1.md',
        'protocols/stvg_opd_visual_host011_engineering_v1.md',
    }
    for rel in sorted(sources):
        original = ROOT / rel
        text = original.read_text()
        assert not private_attachment_path(text)
        if original.suffix == '.py':
            ast.parse(text)
        destination = CHECKOUT / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        bindings[rel] = dict(private_source_sha256=sha(original), public_source_sha256=sha(destination))
        assert bindings[rel]['private_source_sha256'] == bindings[rel]['public_source_sha256']
        owned.add(rel)

    receipts = (
        'CAPTURE_RECEIPT.json', 'DESIGN_LOCK.json', 'REVISION_RUNTIME.json',
        'CPU_CONTRACTS.json', 'GPU_ATTENTION_STORAGE_CONTRACTS.json',
        'GPU_ATTENTION_LIFETIME_CONTRACTS.json', 'GPU_HOST_RESIDENCY_CONTRACTS.json',
        'LAUNCH.json', 'ROOT_MEMORY_QUALIFICATION_READBACK.json',
        'ROOT_RESUMPTION_READBACK.json', 'ENGINEERING_ROOT_REVIEW.json',
        'HALF_HOURLY_MONITOR_RUNTIME_READBACK.json',
        'CPU_dependency_restart006/REVISION_RUNTIME.json', 'CPU_dependency_restart006/LAUNCH.json',
        'CPU_dependency_restart007/REVISION_RUNTIME.json', 'CPU_dependency_restart007/LAUNCH.json',
    )
    for name in (*NAMES, 'frozen_module_CPU008'):
        for rel in receipts:
            original = BASE / name / rel
            if not original.exists():
                continue
            if name == 'frozen_module_CPU008' and rel not in ('ROOT_RESUMPTION_READBACK.json', 'ENGINEERING_ROOT_REVIEW.json'):
                continue
            value = dict(original_private_receipt_sha256=sha(original), projection=metadata(read(original)))
            safe(value)
            write(PUB / 'receipts' / name / rel, value)

    write(PUB / 'SOURCE_PROJECTION_MANIFEST.json', dict(status='pass', sources=bindings,
          scope='exact new engineering sources; anonymous count/integrity receipt projection only'))
    write(PUB / 'FINITE_SCOPE.json', dict(status='bounded_actual_qualification_and_first_formal_readback',
          qualification009_GPU_fits=42, qualification010_GPU_fits=48, qualification011_GPU_fits=54,
          qualification009_rounds=780, qualification010_rounds=840, qualification011_rounds=900,
          each_revision_independently_verified_formal_fits=6, retained_formal_prefix_before011=3003,
          retained_formal_prediction_bytes_before011=8359275779, original_formal_total=5376,
          O_phase_complete=False, Q0_correct_Native_complete=False, all_tasks_complete=False,
          new_task_GT_scoring_complete=False, scientific_configuration_changed=False,
          future_OOM_safety_claim=False, private_raw_fits_and_GT_not_exported=True))
    (PUB / 'REPORT.md').write_text(
        '# Bounded O memory engineering009–011\n\n'
        'Storage009 retains original frozen attention arithmetic while reusing ephemeral storage. '
        'Forty-two full real GPU fits and 780 rounds passed independent root qualification. '
        'Its first six formal fits passed complete math/state/qualified-formal pairs and double SHA checks. '
        'Eighty-one additional formal fits were accepted before another frozen allocation OOM at 2913 records.\n\n'
        'Lifetime010 releases already-dead attention tensor references after their original last use. '
        'Forty-eight complete real GPU fits and 840 rounds, plus the first six formal fits, passed root. '
        'Ninety additional formal fits were accepted before a later 2.14 GiB frozen allocation failed with '
        '2.03 GiB free. All 3003 accepted records (8359275779 bytes) and saved failure sources are preserved. '
        'No complete failed fit was serialized; a bitwise claim about dead failed memory is not made.\n\n'
        'Visual-host011 stages already-computed inactive projected visual tensors, masks and positions '
        'on CPU during the original Video Swin call and restores their original CUDA values before use. '
        'An unused ResNet output reference is released. Fifty-four real GPU fits across both sources, '
        'three arms, two full capture repeats and all five failed cells passed root: 900 rounds and '
        '7527520 independent scalar checks. Its first six new formal fits also passed complete math, '
        'state, qualification/formal and repeated input/prediction SHA readback.\n\n'
        'Original models, weights, frame/window shapes, precision, inputs, rosters, science/runtime, '
        'fit/action/reward/gradient/Adam/state/cap/WHEN remain fixed. The original paused baseline '
        'processes and optimizer memory remain intact. These bounded passes do not guarantee future '
        'OOM safety or measure deployment latency. Full 5376-record global deploy seal, original task '
        'scoring, independent statistics, actual figures/pixel views, all positive and negative results, '
        'verified publication, archive and real experiment FINAL remain required. Correct Native Q0 '
        'and the S/C/N suites retain separate unfinished obligations. No task-efficacy scores, GT, media, '
        'weights, raw predictions, raw fits, input/expert caches or mathematical payloads are exported.\n')

    for original in PUB.rglob('*'):
        if not original.is_file():
            continue
        rel = str(original.relative_to(ROOT))
        destination = CHECKOUT / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        owned.add(rel)
    entries = []
    for rel in sorted(owned):
        path = CHECKOUT / rel
        data = path.read_bytes()
        assert not private_attachment_path(data.decode())
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        entries.append(dict(path=rel, bytes=len(data), sha256=sha(path), blob_sha=blob))
    with tempfile.TemporaryDirectory(prefix='stvg-memory009-011-index-') as folder:
        env = {**os.environ, 'GIT_INDEX_FILE': str(Path(folder) / 'index')}
        git('read-tree', base, env=env)
        for record in entries:
            blob = git('hash-object', '-w', str(CHECKOUT / record['path']))
            assert blob == record['blob_sha']
            git('update-index', '--add', '--cacheinfo', '100644', blob, record['path'], env=env)
        tree = git('write-tree', env=env)
    write(REC / 'PUBLIC_STAGE.json', dict(status='reviewed_staged', repository='Zonglin-He/A',
          branch=BRANCH, base_commit=base, expected_tree=tree, files=entries,
          scope='bounded real memory engineering009-011/first formal readback/source; not O phase or Q0 completion',
          file_count=len(entries), bytes=sum(r['bytes'] for r in entries), GT_payload_exported=False,
          time=time.time()))
    print('MEMORY009_011_PUBLIC_READY', len(entries), sum(r['bytes'] for r in entries), tree)


if __name__ == '__main__':
    run()
