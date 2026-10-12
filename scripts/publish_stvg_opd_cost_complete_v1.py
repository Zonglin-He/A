"""Separate full cost32 export; original S/root006 pinned publisher is untouched."""
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, sha
from scripts.run_stvg_opd_cost_after_N_priority001 import SCIENCE as BASE, verify
from scripts.audit_stvg_opd_cost_complete_public_v1 import REL
from scripts.publish_stvg_opd_actionability_audit_v1 import git, CHECKOUT, BRANCH, put, safe, private_attachment_path, receipt_metadata
OUT = ROOT/REL
SOURCES = ['protocols/stvg_opd_end_to_end_cost_supplement_v1.md', 'protocols/stvg_opd_cost_after_N_priority001.md',
    'scripts/run_stvg_opd_end_to_end_cost_supplement_v1.py', 'scripts/continue_stvg_opd_end_to_end_cost_supplement_v1.py',
    'scripts/continue_stvg_opd_cost_isolation_v1.py', 'scripts/run_stvg_opd_cost_after_N_priority001.py',
    'scripts/continue_stvg_opd_cost_after_N_priority001.py', 'scripts/review_stvg_opd_cost_complete_v1.py',
    'scripts/audit_stvg_opd_cost_complete_public_v1.py', 'scripts/publish_stvg_opd_cost_complete_v1.py',
    'scripts/close_stvg_opd_cost_complete_v1.py']


def run(closing=False):
    verify()
    assert read(BASE/'ROOT_COMPLETE_TIMED_FIT_READBACK.json')['complete_matched_fits'] == 32
    view = read(BASE/'ROOT_VISUAL_REVIEW.json')
    assert view['status'] == 'pass' and view['actual_pixel_view']
    assert git('remote', 'get-url', 'origin') == 'https://github.com/Zonglin-He/A.git'
    subprocess.run(['git', 'fetch', '--quiet', 'origin', 'main', BRANCH], cwd=CHECKOUT, check=True)
    base = git('rev-parse', 'HEAD')
    assert base == git('rev-parse', 'origin/main') == git('rev-parse', 'origin/'+BRANCH)
    assert not git('diff', 'HEAD', '--name-only') and not git('diff', '--cached', '--name-only')
    unrelated = git('ls-files', '--others', '--exclude-standard').splitlines()
    owned = set()
    def copy(source, relative):
        data = source.read_bytes()
        if source.suffix == '.json':
            safe(read(source))
        elif source.suffix in {'.py', '.md'}:
            assert not private_attachment_path(data.decode())
            if source.suffix == '.py':
                ast.parse(data.decode())
        dest = CHECKOUT/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        assert sha(source) == sha(dest)
        owned.add(relative)
    for n in SOURCES:
        copy(ROOT/n, n)
    for n in ['hc2/QUERY_TIMINGS.json', 'vidstg/QUERY_TIMINGS.json', 'END_TO_END_SUMMARY.json',
              'ALL_PATH_COUNTS.json', 'DESIGN_PUBLIC.json', 'PUBLIC_SCALAR_AUDIT.json', 'REPORT.md']:
        copy(OUT/n, str((OUT/n).relative_to(ROOT)))
    if not closing:
        for n in ['ROOT_COMPLETE_TIMED_FIT_READBACK.json', 'ROOT_FRESH_EXPERT_AND_TIMING_READBACK.json',
                  'ROOT_INDEPENDENT_TIMING_STATISTICS.json', 'ROOT_VISUAL_REVIEW.json', 'FIGURE_EXPORT.json']:
            p = BASE/n
            put(OUT/'receipts'/n, dict(original_private_receipt_sha256=sha(p), projection=receipt_metadata(read(p))))
        for ds in ['hc2', 'vidstg']:
            p = BASE/ds/'COMPLETION.json'
            put(OUT/'receipts'/ds/'COMPLETION.json', dict(original_private_receipt_sha256=sha(p), projection=receipt_metadata(read(p))))
        for n in ['SERIAL_HANDOFF_SCOPE.json', 'CPU_STORAGE_CONTRACTS.json', 'REVISION_RUNTIME.json', 'BASELINE_STANDBY_IDENTITY_RECEIPT.json',
                  'LAUNCH.json', 'CPU_COMPLETION.json', 'ARCHIVE_HANDOFF_RECEIPT.json']:
            p = BASE/'after_N_priority001'/n
            put(OUT/'receipts/after_N_priority001'/n, dict(original_private_receipt_sha256=sha(p), projection=receipt_metadata(read(p))))
        p = BASE/'preGPU_timing_scope_engineering001/REVISION_RUNTIME.json'
        put(OUT/'receipts/preGPU_timing_scope_engineering001/REVISION_RUNTIME.json', dict(original_private_receipt_sha256=sha(p), projection=receipt_metadata(read(p))))
        for relative, binding in view['PDF_files'].items():
            p = ROOT/relative
            assert sha(p) == binding['sha256'] and p.stat().st_size == binding['bytes']
            copy(p, str((OUT/'figures'/p.name).relative_to(ROOT)))
        p = subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'), '-B', str(CHECKOUT/'scripts/audit_stvg_opd_cost_complete_public_v1.py'), str(CHECKOUT)],
            capture_output=True, text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        assert p.returncode == 0, p.stdout+p.stderr
        result = json.loads(p.stdout)
        assert result == read(OUT/'PUBLIC_SCALAR_AUDIT.json') and result['status'] == 'pass'
        write(BASE/'PUBLIC_SCALAR_AUDIT.json', dict(result, actual_public_checkout=True, time=time.time()))
    else:
        assert read(BASE/'FINAL_COMPLETION.json')['status'] == 'complete'
        for n in ['ROOT_CLOSING.json', 'FINAL_COMPLETION.json', 'ARCHIVE_VERIFICATION_RECEIPT.json', 'FINAL_GITHUB_RECEIPT.json']:
            p = BASE/n
            put(OUT/'closing'/n, dict(original_private_receipt_sha256=sha(p), projection=receipt_metadata(read(p))))
        previous = read(BASE/'PUBLIC_STAGE.json')
        for e in previous['files']:
            assert sha(CHECKOUT/e['path']) == e['sha256']
        owned.update(e['path'] for e in previous['files'])
    for folder in ['receipts', 'closing']:
        if (OUT/folder).exists():
            for p in (OUT/folder).rglob('*.json'):
                copy(p, str(p.relative_to(ROOT)))
    relative = str((OUT/'SOURCE_MANIFEST.json').relative_to(ROOT))
    manifest = OUT/('SOURCE_MANIFEST_CLOSING.json' if closing else 'SOURCE_MANIFEST.json')
    write(manifest, dict(status='pass', scope='original all32 audited full paths/all anonymous stage timings/no-information and slow paths/source/storage/dependency receipts/actual PDF view; excludes self hash',
        files={n: dict(sha256=sha(CHECKOUT/n), bytes=(CHECKOUT/n).stat().st_size) for n in sorted(owned) if n != relative}))
    copy(manifest, relative)
    entries = []
    for n in sorted(owned):
        p = CHECKOUT/n; data = p.read_bytes()
        entries.append(dict(path=n, bytes=len(data), sha256=sha(p), blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
    with tempfile.TemporaryDirectory(prefix='cost-complete-index-') as folder:
        env = {**os.environ, 'GIT_INDEX_FILE': str(Path(folder)/'index')}
        git('read-tree', base, env=env)
        for e in entries:
            h = git('hash-object', '-w', str(CHECKOUT/e['path'])); assert h == e['blob_sha']
            git('update-index', '--add', '--cacheinfo', '100644', h, e['path'], env=env)
        tree = git('write-tree', env=env)
    write(BASE/('CLOSING_PUBLIC_STAGE.json' if closing else 'PUBLIC_STAGE.json'), dict(status='reviewed_staged', repository='Zonglin-He/A', branch=BRANCH,
        base_commit=base, expected_tree=tree, files=entries, file_count=len(entries), bytes=sum(e['bytes'] for e in entries),
        unrelated_untracked_preserved=unrelated, scope='genuine original32 audited cost complete source/all actual anonymous outcomes/independent root/actual PDF view; no cancelled formal baseline completion or whole followup claim', time=time.time()))
    print(json.dumps(dict(status='reviewed_staged', closing=closing, files=len(entries), bytes=sum(e['bytes'] for e in entries), base_commit=base, expected_tree=tree)))


if __name__ == '__main__':
    run(len(sys.argv) > 1 and sys.argv[1] == 'closing')
