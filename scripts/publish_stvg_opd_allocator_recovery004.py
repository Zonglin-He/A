"""Explicit-path public export of the bounded allocator repair and resumption."""
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stvg_opd_p1_allocator_recovery004 import REC, verify
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE, read, write, sha
CHECKOUT = ROOT.parent / 'visual-grounding-public-A'
BRANCH = 'research/stvg-opd-paper-hc2-revision-v2'
PUB = ROOT / 'results/stvg_opd_p1_allocator_recovery004/2026-10-09'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=CHECKOUT).decode().strip()


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={
            'User-Agent': 'STVG-allocator-root-readback'}), timeout=35) as response:
        return response.read()


def stage():
    verify()
    root = read(REC / 'ROOT_RESUME_READBACK.json')
    assert root['status'] == 'actual_missing_only_continuation_verified'
    assert root['old_prefix_files_opaque_verified'] == 12488
    assert not root['P1_GT_read'] and not root['P1_or_paper_complete']
    assert git('remote', 'get-url', 'origin') == 'https://github.com/Zonglin-He/A.git'
    assert git('branch', '--show-current') == BRANCH
    assert git('rev-parse', 'HEAD') == git('rev-parse', 'origin/main') == git('rev-parse', 'origin/' + BRANCH)
    # Other untracked historical authoring files remain outside this export.
    assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
    assert not PUB.exists()
    PUB.mkdir(parents=True)
    owned = set(read(REC / 'REVISION_RUNTIME.json')['pins']) | {
        'scripts/readback_stvg_opd_allocator_recovery004.py',
        'scripts/publish_stvg_opd_allocator_recovery004.py',
        'docs/STVG_OPD_P1_ALLOCATOR_RECOVERY004.md',
    }
    capture = read(REC / 'CAPTURE_RECEIPT.json')
    original = read(REC / 'REVISION_RUNTIME.json')
    write(PUB / 'CAPTURE_BINDING.json', dict(status=capture['status'], scope=capture['scope'],
        stage=capture['stage'], prefix_arrivals=capture['done'], per_order=capture['per_order'],
        prefix_files=len(capture['prefix_files']), prefix_bytes=capture['prefix_bytes'],
        private_capture_receipt_sha256=sha(REC / 'CAPTURE_RECEIPT.json'),
        preserved_original_files=capture['originals'],
        original_failure='CUDA out of memory during full native Video Swin attention',
        requested_GiB=2.14, available_GiB=2.01, reserved_unused_GiB=3.51,
        GT_read=False, original_failed_native_capture_not_serialized=True,
        scientific_protocol_changed=False))
    for name in ('REVISION_RUNTIME.json', 'CPU_CONTRACTS.json',
                 'ROOT_QUALIFICATION_READBACK.json', 'ROOT_RESUME_READBACK.json'):
        shutil.copy2(REC / name, PUB / name)
    for name in ('GPU_QUALIFICATION.json', 'FIRST_FORMAL_FIT_BITWISE.json'):
        obj = read(REC / name)
        obj['input'] = {k: v for k, v in obj['input'].items() if k != 'path'}
        write(PUB / name, obj)
    write(PUB / 'CODE_BINDING.json', dict(code={f: sha(ROOT / f) for f in sorted(owned)},
        original_science_runtime_sha256=sha(BASE / 'RUNTIME_LOCK.json'),
        engineering_runtime_sha256=sha(REC / 'REVISION_RUNTIME.json'),
        previous_chart_runtime_sha256=original['previous_chart_runtime_sha256'],
        actual_GPU_qualification=True, actual_missing_suffix_resumption=True,
        original_prefix_unchanged=True, HC2_not_rerun=True,
        GT_read=False, P1_or_paper_complete=False, raw_private_payload_exported=False))
    (PUB / 'README.md').write_text('''# P1 allocator-only recovery004

Original CUDA OOM is preserved, six actual GPU fits passed, and the first
missing formal prediction matches qualification bitwise before acceptance.
Root re-hashed all12,488 original predictions/receipts and checked the accepted
fit, original predecessor, LN writeback and resets. Only the allocator setting
changes; original scientific parameters, full input frames and GT barrier stay.

This export closes bounded repair/qualification/resumption only. P1 efficacy
has not been scored; P1 and the original P2–P6 paper suite are not complete.
EATA and historical queues stay paused. No future OOM guarantee or comparison
with the lost failed-process capture is claimed. The measured fit peak counters
do not cover full native-encoder memory. Dataset pixels, query text, GT,
weights, raw predictions, optimizer/gradient arrays and binary fits are excluded.

See docs/STVG_OPD_P1_ALLOCATOR_RECOVERY004.md for actual counts and limits.
The scripts require the original private locked workspace and its checkpoints.
''')
    owned |= {str(p.relative_to(ROOT)) for p in PUB.iterdir() if p.is_file()}
    for rel in sorted(owned):
        assert Path(rel).suffix in {'.py', '.md', '.json'}
        raw = (ROOT / rel).read_bytes()
        if rel.endswith('.json'):
            for key in (b'"caption":', b'"video_path":', b'"GT_xyxy":', b'"gradient":',
                        b'"optimizer_state":', b'"boxes":', b'"raw_logits":', b'"committed":'):
                assert key not in raw, (rel, key)
        dest = CHECKOUT / rel
        assert not dest.exists(), dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    subprocess.run(['git', 'add', '--', *sorted(owned)], cwd=CHECKOUT, check=True)
    assert set(git('diff', '--cached', '--name-only').splitlines()) == owned
    files = [dict(path=f, bytes=(CHECKOUT / f).stat().st_size, sha256=sha(CHECKOUT / f),
                  blob_sha=git('hash-object', f)) for f in sorted(owned)]
    write(REC / 'PUBLIC_STAGE.json', dict(status='reviewed_explicit_paths_staged',
        repository='Zonglin-He/A', branch=BRANCH, base_commit=git('rev-parse', 'HEAD'),
        base_tree=git('rev-parse', 'HEAD^{tree}'), expected_tree=git('write-tree'),
        files=files, file_count=len(files), bytes=sum(f['bytes'] for f in files),
        other_untracked_files_preserved=True, P1_or_paper_complete=False, time=time.time()))
    print(json.dumps(dict(status='reviewed_explicit_paths_staged', files=len(files),
        bytes=sum(f['bytes'] for f in files))))


def verify_remote(commit):
    stage = read(REC / 'PUBLIC_STAGE.json')
    for branch in ('main', BRANCH):
        ref = json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/' + branch))
        assert ref['object']['sha'] == commit
    info = json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/commits/' + commit))
    assert info['parents'][0]['sha'] == stage['base_commit']
    assert info['tree']['sha'] == stage['expected_tree']
    def one(f):
        content = fetch('https://raw.githubusercontent.com/Zonglin-He/A/' + commit + '/' + urllib.parse.quote(f['path']))
        assert content == (ROOT / f['path']).read_bytes() == (CHECKOUT / f['path']).read_bytes()
        assert len(content) == f['bytes'] and hashlib.sha256(content).hexdigest() == f['sha256']
        assert hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest() == f['blob_sha']
        return dict(f, remote_bytes_identical=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(one, stage['files']))
    verify()
    assert git('rev-parse', 'HEAD') == commit
    assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
    write(REC / 'FINAL_GITHUB_RECEIPT.json', dict(status='pass', repository='Zonglin-He/A',
        commit=commit, branches=['main', BRANCH], files=files, file_count=len(files),
        bytes=sum(f['bytes'] for f in files), original_science_runtime_unchanged=True,
        actual_bounded_allocator_repair_qualification_resumption_only=True,
        P1_GT_scoring=False, P1_or_paper_complete=False,
        other_untracked_authoring_preserved=True, time=time.time()))
    print(json.dumps(dict(status='remote_contents_verified', commit=commit,
        file_count=len(files), bytes=sum(f['bytes'] for f in files))))


if __name__ == '__main__':
    stage() if sys.argv[1] == 'stage' else verify_remote(sys.argv[2])
