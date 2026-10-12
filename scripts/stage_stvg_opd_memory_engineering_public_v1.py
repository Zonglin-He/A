"""Finite reviewed engineering export; no GT, raw fits, media or commit mutation."""
import ast, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, sha
from scripts.publish_stvg_opd_supplement_v1 import safe, private_attachment_path
BASE = ROOT / 'artifacts/stvg_opd_optimization_geometry_v1'
REC = BASE / 'frozen_module_CPU008'
PUB = ROOT / 'results/stvg_opd_optimization_geometry/2026-10-12/memory_engineering007_008'
CHECKOUT = ROOT.parent / 'visual-grounding-public-A'
BRANCH = 'research/stvg-opd-paper-hc2-revision-v2'


def metadata(v):
    excluded = {'records', 'files', 'prefix', 'accepted_prefix', 'originals', 'original_baseline_processes',
                'protected', 'remote_desktop_process_metadata', 'prior_dependency_failures',
                'preserved_failure_statuses', 'failed_cell', 'private_source_paths'}
    if isinstance(v, dict):
        return {k: metadata(x) for k, x in v.items() if k not in excluded}
    if isinstance(v, list):
        return [metadata(x) for x in v]
    if isinstance(v, str):
        return v.replace(str(ROOT), '<PROJECT_ROOT>')
    return v


def run():
    for namespace in ('frozen_capture_CPU007', 'frozen_module_CPU008'):
        assert read(BASE / namespace / 'ROOT_MEMORY_QUALIFICATION_READBACK.json')['status'] == 'pass'
    assert read(BASE / 'frozen_capture_CPU007/ROOT_RESUMPTION_READBACK.json')['status'] == 'pass'
    def git(*args, env=None):
        return subprocess.check_output(['git', *args], cwd=CHECKOUT, env=env, text=True).strip()
    assert git('remote', 'get-url', 'origin') == 'https://github.com/Zonglin-He/A.git'
    subprocess.run(['git', 'fetch', '--quiet', 'origin', 'main', BRANCH], cwd=CHECKOUT, check=True)
    base = git('rev-parse', 'HEAD')
    assert base == git('rev-parse', 'origin/main') == git('rev-parse', 'origin/' + BRANCH)
    assert not git('diff', 'HEAD', '--name-only') and not git('diff', '--cached', '--name-only')
    owned, bindings = set(), {}
    sources = {
        'vg_tta/stvg_opd_frozen_capture_cpu007.py', 'vg_tta/stvg_opd_frozen_module_cpu008.py',
        'scripts/run_stvg_opd_common_state_capture007.py', 'scripts/run_stvg_opd_common_state_module008.py',
        'scripts/continue_stvg_opd_capture007_dependencies.py',
        'scripts/review_stvg_opd_capture007_prefix_v1.py', 'scripts/review_stvg_opd_module008_prefix_v1.py',
        'scripts/stage_stvg_opd_memory_engineering_public_v1.py',
        'scripts/review_stvg_opd_common_state_postseal_v1.py', 'scripts/review_stvg_opd_common_state_objective002.py',
        'scripts/report_stvg_opd_common_state_postseal_v1.py', 'scripts/continue_stvg_opd_common_state_root_v1.py',
        'scripts/continue_stvg_opd_common_state_objective002.py',
        'protocols/stvg_opd_capture007_engineering_v1.md', 'protocols/stvg_opd_module008_engineering_v1.md',
    }
    for folder, pattern in [('scripts', '*stvg_opd_query_q0*.py'), ('vg_tta', '*stvg_opd_query_q0*.py'),
                            ('protocols', '*stvg_opd_query*q0*.md')]:
        sources.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).glob(pattern))
    for rel in sorted(sources):
        p = ROOT / rel
        text = p.read_text()
        if private_attachment_path(text):
            text = re.sub(r'/home/[^/]+/\.codex/attachments/[^\"\']+', '<LOCAL_AUTHORIZATION_ATTACHMENT>', text)
        assert not private_attachment_path(text)
        if p.suffix == '.py':
            ast.parse(text)
        dest = CHECKOUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        bindings[rel] = dict(private_source_sha256=sha(p), public_source_sha256=sha(dest),
                             local_authorization_path_only_projected=sha(p) != sha(dest))
        owned.add(rel)
    for namespace in ('frozen_capture_CPU007', 'frozen_module_CPU008'):
        for name in ('CAPTURE_RECEIPT.json', 'DESIGN_LOCK.json', 'REVISION_RUNTIME.json', 'CPU_CONTRACTS.json',
                     'LAUNCH.json', 'ROOT_MEMORY_QUALIFICATION_READBACK.json', 'ROOT_RESUMPTION_READBACK.json'):
            p = BASE / namespace / name
            if not p.exists():
                continue
            value = dict(original_private_receipt_sha256=sha(p), projection=metadata(read(p)))
            safe(value)
            out = PUB / 'receipts' / namespace / name
            write(out, value)
    write(PUB / 'SOURCE_PROJECTION_MANIFEST.json', dict(status='pass', sources=bindings,
          scope='exact engineering/science source except local authorization attachment path'))
    write(PUB / 'FINITE_SCOPE.json', dict(status='bounded_real_memory_qualification_and_formal_resumption_only',
          qualification007_GPU_fits=30, qualification008_GPU_fits=36,
          qualification007_complete_rounds=660, qualification008_complete_rounds=720,
          independently_verified007_formal_fits=6, original_preserved_formal_prefix=2769,
          original_formal_total=5376, Q0_correct_Native_qualification_complete=False,
          Q0_correct_Native_formal_complete=False, new_task_GT_scoring_complete=False,
          O_phase_complete=False, all_tasks_complete=False, scientific_configuration_changed=False,
          future_OOM_safety_claim=False, private_raw_fits_and_GT_not_exported=True))
    (PUB / 'REPORT.md').write_text(
        '# O memory engineering007/008\n\n'
        'A real frozen Video Swin allocation failed after 2721 formal records while the paused baseline '
        'retained its original in-memory state. Revision007 staged exact normalized videos and frozen '
        'capture tensors on CPU; thirty complete real qualification fits, full repeated trajectories, '
        'original Full checks and an independent root readback passed. It accepted forty-eight additional '
        'formal records. The first six new formal fits passed complete failed-cell qualification/formal '
        'pairing, math/state and double input/prediction SHA checks.\n\n'
        'A later allocation failed with an additional remote-desktop CUDA context resident. Revision008 '
        'temporarily stages other inactive frozen source modules on CPU only during the unchanged Video '
        'Swin call, restoring them before original subsequent operations. Thirty-six complete real GPU '
        'fits across both sources and both actual failed cells passed repeated trajectories, existing '
        'qualification/original Full comparisons and independent math checks. The 2769 accepted records '
        'were rehashed and preserved. Missing failed-fit memory is not claimed bitwise reproduced.\n\n'
        'This finite export provides code, exact bounded protocols and anonymous integrity/count receipts. '
        'It contains no new task-efficacy scores. Full 5376-record deployment seal, postseal scorer, '
        'independent statistics, actual report pixel views, all positive/negative results, verified public '
        'publication, archive and genuine experiment FINAL are still required. Correct Native-support '
        'Q0 has not yet executed qualification or formal queries. Instrumented and staged timings do '
        'not establish deployment latency; bounded passes do not prove future OOM safety.\n')
    for p in PUB.rglob('*'):
        if not p.is_file():
            continue
        rel = str(p.relative_to(ROOT))
        dest = CHECKOUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        owned.add(rel)
    entries = []
    for rel in sorted(owned):
        p = CHECKOUT / rel
        data = p.read_bytes()
        assert not private_attachment_path(data.decode())
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        entries.append(dict(path=rel, bytes=len(data), sha256=sha(p), blob_sha=blob))
    with tempfile.TemporaryDirectory(prefix='stvg-memory-public-index-') as folder:
        env = {**os.environ, 'GIT_INDEX_FILE': str(Path(folder) / 'index')}
        git('read-tree', base, env=env)
        for record in entries:
            blob = git('hash-object', '-w', str(CHECKOUT / record['path']))
            assert blob == record['blob_sha']
            git('update-index', '--add', '--cacheinfo', '100644', blob, record['path'], env=env)
        tree = git('write-tree', env=env)
    write(REC / 'PUBLIC_STAGE.json', dict(status='reviewed_staged', repository='Zonglin-He/A',
          branch=BRANCH, base_commit=base, expected_tree=tree, files=entries,
          scope='bounded real engineering qualification/resumption and required source only; not O phase or Q0 completion',
          file_count=len(entries), bytes=sum(r['bytes'] for r in entries), GT_payload_exported=False, time=time.time()))
    print('MEMORY_ENGINEERING_PUBLIC_READY', len(entries), sum(r['bytes'] for r in entries), tree)


if __name__ == '__main__':
    run()
