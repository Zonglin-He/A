"""Explicit safe export and remote-byte verification of bounded reward audit repair."""
import concurrent.futures, hashlib, json, shutil, subprocess, sys, time
import urllib.parse, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_stvg_opd_p1_reward_precision005 import REC, verify
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE, read, write, sha
CHECKOUT = ROOT.parent/'visual-grounding-public-A'
BRANCH = 'research/stvg-opd-paper-hc2-revision-v2'
PUB = ROOT/'results/stvg_opd_p1_reward_precision005/2026-10-09'


def git(*args): return subprocess.check_output(['git', *args], cwd=CHECKOUT).decode().strip()


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'STVG-reward-root-readback'}), timeout=35) as response:
        return response.read()


def stage():
    verify(); root = read(REC/'ROOT_RESUME_READBACK.json')
    assert root['status'] == 'actual_missing_only_continuation_verified' and root['old_prefix_files_opaque_verified'] == 16070
    assert not root['P1_GT_read'] and not root['P1_or_paper_complete']
    assert read(REC/'ROOT_RECEIPT_DISPATCH_CONTRACTS.json')['status'] == 'pass'
    assert git('remote', 'get-url', 'origin') == 'https://github.com/Zonglin-He/A.git'
    assert git('branch', '--show-current') == BRANCH
    assert git('rev-parse', 'HEAD') == git('rev-parse', 'origin/main') == git('rev-parse', 'origin/'+BRANCH)
    assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
    assert not PUB.exists(); PUB.mkdir(parents=True)
    owned = set(read(REC/'REVISION_RUNTIME.json')['pins']) | {
        'scripts/readback_stvg_opd_reward_precision005.py', 'scripts/publish_stvg_opd_reward_precision005.py',
        'docs/STVG_OPD_P1_REWARD_PRECISION005.md'}
    cap = read(REC/'CAPTURE_RECEIPT.json')
    write(PUB/'CAPTURE_BINDING.json', dict(status=cap['status'], scope=cap['scope'],
        prefix_arrivals=cap['done'], prefix_files=len(cap['prefix_files']), prefix_bytes=cap['prefix_bytes'],
        actual_failed_fit_serialized_before_exit=True, failed_fit_sha256=cap['failed_fit_sha256'],
        first_missing_order=cap['first_missing_order'], first_missing_arrival=cap['first_missing_arrival'],
        first_missing_query_ordinal=cap['first_missing_query_ordinal'],
        private_capture_receipt_sha256=sha(REC/'CAPTURE_RECEIPT.json'),
        original_failure='Detached IoU GPU32-versus-float64 absolute audit failed at original line21',
        original_failure_preserved=True, scientific_protocol_changed=False, GT_read=False))
    for name in ('REVISION_RUNTIME.json', 'CPU_CONTRACTS.json', 'ROOT_QUALIFICATION_READBACK.json',
                 'ROOT_RECEIPT_DISPATCH_CONTRACTS.json', 'ROOT_RESUME_READBACK.json', 'ROOT_STATE_CHAIN.json'):
        shutil.copy2(REC/name, PUB/name)
    for name in ('GPU_QUALIFICATION.json', 'FIRST_FORMAL_FIT_BITWISE.json'):
        obj = read(REC/name); obj['input'] = {k: v for k, v in obj['input'].items() if k != 'path'}
        write(PUB/name, obj)
    write(PUB/'CODE_BINDING.json', dict(code={f: sha(ROOT/f) for f in sorted(owned)},
        original_science_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'), engineering_runtime_sha256=sha(REC/'REVISION_RUNTIME.json'),
        actual_GPU_qualification=True, actual_missing_suffix_resumption=True,
        original_prefix_unchanged=True, HC2_not_rerun=True, GT_read=False,
        P1_or_paper_complete=False, raw_private_payload_exported=False))
    (PUB/'README.md').write_text('''# P1 detached reward precision audit revision005

The original detached-IoU absolute float64 failure is preserved. Independent
CPU float32 IoU keeps the3e-6 absolute threshold, and per-action interval
arithmetic encloses measured sigmoid rounding and float32 box geometry.
Neither the actual GPU reward, actions, likelihood, optimizer nor predictions
changes. Six real GPU fits passed. The full failed-process serialization,
qualification repeats and first accepted missing formal fit agree bitwise.
Root checked all16,070 original prediction bytes/receipts, the accepted state
writeback/resets, six saved audit dispatch contracts and old math dictionaries.

This closes bounded repair/qualification/resumption only. P1 efficacy and the
original P2–P6 paper suite remain incomplete. No GT has been read for P1, and
EATA and historical queues stay paused. An independent complete decoder
Jacobian or CUDA transcendental-kernel proof is not claimed. Figure1's CPU
waiter was restored after the dependency failure and still waits for global
P1 GPU seal. Pixels, query text, annotations, weights, raw predictions,
gradient/Adam arrays and binary fits are excluded from this public export.
''')
    owned |= {str(p.relative_to(ROOT)) for p in PUB.iterdir() if p.is_file()}
    for rel in sorted(owned):
        assert Path(rel).suffix in {'.py', '.md', '.json'}
        raw = (ROOT/rel).read_bytes()
        if rel.endswith('.json'):
            for key in (b'"caption":', b'"video_path":', b'"GT_xyxy":', b'"gradient":',
                        b'"optimizer_state":', b'"boxes":', b'"raw_logits":', b'"committed":'):
                assert key not in raw, (rel, key)
        dest = CHECKOUT/rel; assert not dest.exists(), dest
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/rel, dest)
    subprocess.run(['git', 'add', '--', *sorted(owned)], cwd=CHECKOUT, check=True)
    assert set(git('diff', '--cached', '--name-only').splitlines()) == owned
    files = [dict(path=f, bytes=(CHECKOUT/f).stat().st_size, sha256=sha(CHECKOUT/f), blob_sha=git('hash-object', f)) for f in sorted(owned)]
    write(REC/'PUBLIC_STAGE.json', dict(status='reviewed_explicit_paths_staged', repository='Zonglin-He/A',
        branch=BRANCH, base_commit=git('rev-parse', 'HEAD'), base_tree=git('rev-parse', 'HEAD^{tree}'),
        expected_tree=git('write-tree'), files=files, file_count=len(files), bytes=sum(f['bytes'] for f in files),
        other_untracked_files_preserved=True, P1_or_paper_complete=False, time=time.time()))
    print(json.dumps(dict(status='reviewed_explicit_paths_staged', files=len(files), bytes=sum(f['bytes'] for f in files))))


def verify_remote(commit):
    stage = read(REC/'PUBLIC_STAGE.json')
    for branch in ('main', BRANCH):
        ref = json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/'+branch))
        assert ref['object']['sha'] == commit
    info = json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/commits/'+commit))
    assert info['parents'][0]['sha'] == stage['base_commit'] and info['tree']['sha'] == stage['expected_tree']
    def one(f):
        content = fetch('https://raw.githubusercontent.com/Zonglin-He/A/'+commit+'/'+urllib.parse.quote(f['path']))
        assert content == (ROOT/f['path']).read_bytes() == (CHECKOUT/f['path']).read_bytes()
        assert len(content) == f['bytes'] and hashlib.sha256(content).hexdigest() == f['sha256']
        assert hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest() == f['blob_sha']
        return dict(f, remote_bytes_identical=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: files = list(pool.map(one, stage['files']))
    verify(); assert git('rev-parse', 'HEAD') == commit
    assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
    write(REC/'FINAL_GITHUB_RECEIPT.json', dict(status='pass', repository='Zonglin-He/A', commit=commit,
        branches=['main', BRANCH], files=files, file_count=len(files), bytes=sum(f['bytes'] for f in files),
        original_science_runtime_unchanged=True, actual_bounded_reward_audit_repair_qualification_resumption_only=True,
        P1_GT_scoring=False, P1_or_paper_complete=False, other_untracked_authoring_preserved=True, time=time.time()))
    print(json.dumps(dict(status='remote_contents_verified', commit=commit, file_count=len(files), bytes=sum(f['bytes'] for f in files))))


if __name__ == '__main__': stage() if sys.argv[1] == 'stage' else verify_remote(sys.argv[2])
