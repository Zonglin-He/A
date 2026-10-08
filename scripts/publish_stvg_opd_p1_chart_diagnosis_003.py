"""Export the diagnosed numerical failure and a clearly uninstalled draft."""
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
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE, read, write, sha, PYTHON
from scripts.run_stvg_opd_p1_softmax_recovery_002 import verify_revision
REC = BASE / 'recovery/P1_box_chart_003'
PUB = ROOT / 'results/stvg_opd_p1_chart_failure/2026-10-09'
CHECKOUT = ROOT.parent / 'visual-grounding-public-A'
BRANCH = 'research/stvg-opd-paper-hc2-revision-v2'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=CHECKOUT).decode().strip()


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'STVG-OPD-root-integrity'}), timeout=35) as response:
        return response.read()


def stage():
    verify_revision()
    root = read(REC / 'ROOT_DIAGNOSIS_RECEIPT.json')
    assert root['status'] == 'diagnosis_complete_revision_authorization_pending'
    assert not root['GT_read'] and not root['new_prediction_accepted']
    assert git('remote', 'get-url', 'origin') == 'https://github.com/Zonglin-He/A.git'
    assert git('branch', '--show-current') == BRANCH
    assert git('rev-parse', 'HEAD') == git('rev-parse', 'origin/main') == git('rev-parse', 'origin/' + BRANCH)
    assert not git('status', '--porcelain') and not PUB.exists()
    PUB.mkdir(parents=True)
    owned = {
        'scripts/audit_stvg_opd_single_direction_bytes_v2.py',
        'scripts/check_stvg_opd_opaque_readback_contracts_v2.py',
        'protocols/stvg_opd_p1_single_direction_integrity_v2.md',
        'scripts/preserve_stvg_opd_p1_chart_failure_003.py',
        'scripts/reproduce_stvg_opd_p1_chart_failure_003.py',
        'scripts/diagnose_stvg_opd_p1_chart_failure_003.py',
        'scripts/test_stvg_opd_boundary_chart_proposal003.py',
        'vg_tta/decota_spatial_opd_boundary_chart_proposal003.py',
        'protocols/stvg_opd_p1_chart_failure_003.md',
        'docs/STVG_OPD_P1_CHART_FAILURE_DIAGNOSIS.md',
        'scripts/publish_stvg_opd_p1_chart_diagnosis_003.py',
    }
    for source, output in [
        (REC / 'CPU_DIAGNOSIS_AND_DRAFT_CONTRACTS.json', 'CPU_DIAGNOSIS_AND_DRAFT_CONTRACTS.json'),
        (REC / 'ROOT_DIAGNOSIS_RECEIPT.json', 'ROOT_DIAGNOSIS_RECEIPT.json'),
        (REC / 'REPRODUCTION_RUNTIME.json', 'REPRODUCTION_RUNTIME.json'),
        (BASE / 'P1_HC2_ROOT_BYTE_READBACK.json', 'HC2_INITIAL_OPAQUE_READBACK.json'),
        (BASE / 'P1_HC2_ROOT_BYTE_READBACK_SNAPSHOT_VERIFIED.json', 'HC2_SNAPSHOT_OPAQUE_READBACK.json'),
        (BASE / 'P1_OPAQUE_READBACK_CONTRACTS.json', 'OPAQUE_INTEGRITY_CONTRACTS.json'),
        (BASE / 'integrity_readback_versions/single_direction_auditor_initial.py', 'preserved_initial_integrity_auditor.py'),
    ]:
        shutil.copy2(source, PUB / output)
    capture = read(REC / 'CAPTURE_RECEIPT.json')
    manifest = hashlib.sha256(json.dumps(capture['prefix_files'], sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    write(PUB / 'PREFIX_INTEGRITY.json', dict(status='pass', stage=capture['stage'],
        receipted_arrivals=capture['done'], bytes=capture['prefix_bytes'],
        ordered_complete_manifest_sha256=manifest, payload_and_receipt_SHA256_bytes_runtime_verified=True,
        raw_arrays_not_exported=True, capture_receipt_sha256=sha(REC / 'CAPTURE_RECEIPT.json'),
        GT_read=False, whole_P1_or_paper_complete=False))
    reproduction = read(REC / 'REPRODUCTION_RECEIPT.json')
    safe = {k: v for k, v in reproduction.items() if k not in {'input', 'bad_coordinates'}}
    safe['input_payload_sha256'] = reproduction['input']['sha256']
    safe['new_DINO_calls'] = reproduction['input']['new_DINO_calls']
    write(PUB / 'REPRODUCTION_RECEIPT.json', safe)
    test = json.loads(subprocess.check_output([str(PYTHON), '-B',
        str(ROOT / 'scripts/test_stvg_opd_boundary_chart_proposal003.py')], cwd=ROOT))
    assert test['status'] == 'pass' and test['proposal_not_installed']
    write(PUB / 'PORTABLE_DRAFT_CPU_CONTRACTS.json', test)
    write(PUB / 'CODE_BINDING.json', dict(code={p: sha(ROOT / p) for p in sorted(owned)},
        original_science_runtime_sha256=sha(BASE / 'RUNTIME_LOCK.json'),
        original_selected_config_file_sha256=sha(ROOT / 'methods/decota_spatial_opd_v1/configs.json'),
        reproduction_runtime_sha256=sha(REC / 'REPRODUCTION_RUNTIME.json'),
        draft_changes_locked_endpoint_failure_rule=True, draft_authorized_or_deployed=False,
        public_export_contains_private_payloads=False, P1_GT_effectiveness_not_measured=True))
    (PUB / 'README.md').write_text('''# P1 native-box chart failure: diagnosed, revision decision pending

See docs/STVG_OPD_P1_CHART_FAILURE_DIAGNOSIS.md for the actual failure and scope.
HC2 has sealed all 10,446 arrivals; the opaque integrity proof is included here.
VidSTG failed after 1,882 outputs. The global P1 seal and P1 GT scores are absent.
Original code, scientific locks and all receipted predictions remain preserved.

One actual GPU replay from the full saved prefix reproduced a finite native
logit whose float32 sigmoid output rounds to 1.0. The original inverse-chart
guard correctly rejects the endpoint under the locked protocol. The dead
worker did not serialize its fit; this is a reproduction, not a comparison
with its lost memory. The first eight completed rounds and all nine Adam
updates were independently checked on CPU, with no claim of a complete fit
or an independent decoder Jacobian. No GT or new DINO observation was used.

The finite-logit endpoint representation is a concrete DRAFT, not a deployed
repair or a new qualification. It changes the protocol's endpoint-failure
behavior and is awaiting explicit human authorization. Portable CPU contracts:

```bash
python -B scripts/test_stvg_opd_boundary_chart_proposal003.py
```

The preserved_initial_integrity_auditor.py file is an exact archived source
snapshot, not an executable entrypoint from this results directory. The current
integrity auditor writes immutable receipts in the private research workspace.
Its synthetic contract checker also writes a new private receipt; it is not a
GT evaluation or a model run.

Private frames, captions, annotations, source weights, raw predictions, live
optimizer/gradient tensors and the reproduced binary fit are excluded. Scalar
numerical diagnostics and aggregate byte digests are provided instead. The
reproduced GPU fit was not separately timed; no total recovery GPU time or
completed-run throughput is inferred. EATA remains explicitly user paused;
P1 scientific completion and P2-P6 remain pending.
''')
    owned |= {str(p.relative_to(ROOT)) for p in PUB.rglob('*') if p.is_file()}
    for rel in sorted(owned):
        assert rel.startswith(('scripts/', 'vg_tta/', 'protocols/', 'docs/', 'results/'))
        assert Path(rel).suffix in {'.py', '.md', '.json'}
        raw = (ROOT / rel).read_bytes()
        if rel.endswith('.json'):
            for forbidden in [b'"caption":', b'"video_path":', b'"GT_box":',
                              b'"gradient":', b'"optimizer_state":', b'"committed":']:
                assert forbidden not in raw, (rel, forbidden)
        dest = CHECKOUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    subprocess.run(['git', 'add', '--', *sorted(owned)], cwd=CHECKOUT, check=True)
    assert set(git('diff', '--cached', '--name-only').splitlines()) == owned
    files = [dict(path=p, sha256=sha(CHECKOUT / p), bytes=(CHECKOUT / p).stat().st_size,
                  blob_sha=git('hash-object', p)) for p in sorted(owned)]
    write(REC / 'PUBLIC_STAGE.json', dict(status='reviewed_staged', repository='Zonglin-He/A',
        branch=BRANCH, base_commit=git('rev-parse', 'HEAD'), base_tree=git('rev-parse', 'HEAD^{tree}'),
        expected_tree=git('write-tree'), files=files, file_count=len(files),
        bytes=sum(v['bytes'] for v in files), whole_P1_or_paper_complete=False,
        draft_authorized_or_deployed=False, time=time.time()))
    print(json.dumps(dict(status='reviewed_staged', files=len(files), bytes=sum(v['bytes'] for v in files))))


def verify_remote(commit):
    meta = read(REC / 'PUBLIC_STAGE.json')
    for branch in ['main', BRANCH]:
        ref = json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/' + branch))
        assert ref['object']['sha'] == commit
    info = json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/commits/' + commit))
    assert info['parents'][0]['sha'] == meta['base_commit'] and info['tree']['sha'] == meta['expected_tree']

    def one(row):
        content = fetch('https://raw.githubusercontent.com/Zonglin-He/A/' + commit + '/' + urllib.parse.quote(row['path']))
        assert content == (CHECKOUT / row['path']).read_bytes()
        assert len(content) == row['bytes'] and hashlib.sha256(content).hexdigest() == row['sha256']
        assert hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest() == row['blob_sha']
        return dict(row, remote_bytes_identical=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(one, meta['files']))
    test = json.loads(subprocess.check_output([str(PYTHON), '-B',
        'scripts/test_stvg_opd_boundary_chart_proposal003.py'], cwd=CHECKOUT))
    assert test['status'] == 'pass' and test['proposal_not_installed']
    verify_revision()
    assert git('rev-parse', 'HEAD') == commit and not git('status', '--porcelain')
    write(REC / 'FINAL_GITHUB_RECEIPT.json', dict(status='pass', repository='Zonglin-He/A',
        commit=commit, branches=['main', BRANCH], files=files, file_count=len(files),
        bytes=sum(v['bytes'] for v in files), public_CPU_contracts=test,
        bounded_diagnosis_only=True, P1_or_paper_completion_not_claimed=True,
        draft_authorized_or_deployed=False, time=time.time()))
    print(json.dumps(dict(status='remote_contents_verified', commit=commit, files=len(files))))


if __name__ == '__main__':
    stage() if sys.argv[1] == 'stage' else verify_remote(sys.argv[2])
