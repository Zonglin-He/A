"""Public export of authorized chart qualification/resumption, no raw assets."""
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
from scripts.run_stvg_opd_p1_chart_revision003 import REC, verify_revision
from scripts.publish_stvg_opd_p1_chart_diagnosis_003 import git, fetch, CHECKOUT, BRANCH
PUB = ROOT / 'results/stvg_opd_p1_chart_revision003/2026-10-09'


def stage():
    verify_revision()
    q = read(REC / 'GPU_QUALIFICATION.json')
    resumed = read(REC / 'ROOT_RESUME_READBACK.json')
    assert q['status'] == 'pass' and resumed['status'] == 'actual_missing_only_continuation_verified'
    assert not resumed['P1_GT_read'] and not resumed['P1_or_paper_complete']
    assert git('remote', 'get-url', 'origin') == 'https://github.com/Zonglin-He/A.git'
    assert git('branch', '--show-current') == BRANCH
    assert git('rev-parse', 'HEAD') == git('rev-parse', 'origin/main') == git('rev-parse', 'origin/' + BRANCH)
    assert not git('status', '--porcelain') and not PUB.exists()
    PUB.mkdir(parents=True)
    owned = set(read(REC / 'REVISION_RUNTIME.json')['pins']) | {
        'scripts/readback_stvg_opd_chart_revision003.py',
        'scripts/publish_stvg_opd_chart_revision003.py',
        'docs/STVG_OPD_P1_CHART_REVISION003.md',
    }
    for name in ['REVISION_RUNTIME.json', 'CPU_CONTRACTS.json',
                 'ROOT_QUALIFICATION_READBACK.json', 'ROOT_RESUME_READBACK.json',
                 'ROOT_RECEIPT_DISPATCH_CONTRACTS.json']:
        shutil.copy2(REC / name, PUB / name)
    auth = read(REC / 'AUTHORIZATION_RECEIPT.json')
    write(PUB / 'AUTHORIZATION_BINDING.json', dict(status=auth['status'], scope=auth['scope'],
        original_private_authorization_receipt_sha256=sha(REC / 'AUTHORIZATION_RECEIPT.json'),
        changes_original_exact_endpoint_failure_behavior=True,
        original_failure_and_predictions_preserved=True, GT_barrier_unchanged=True,
        method_parameters_unchanged=True, EATA_remains_user_paused=True))
    for name in ['GPU_QUALIFICATION.json', 'FIRST_FORMAL_FIT_BITWISE.json']:
        obj = read(REC / name)
        obj['input'] = {k: v for k, v in obj['input'].items() if k != 'path'}
        write(PUB / name, obj)
    (PUB / 'README.md').write_text('''# Authorized native-logit chart revision003: qualified and resumed

See docs/STVG_OPD_P1_CHART_REVISION003.md for the actual qualification,
fixed scientific scope and limitations. The earlier original failure remains
published under results/stvg_opd_p1_chart_failure/2026-10-09.

Only the rounded endpoint inverse-coordinate representation changes.
Interior values/gradients and native box readouts remain unchanged. No clamp,
hyperparameter tuning, skipped query, altered exploration or GT selection.
Six actual GPU fits cover the failed query twice and two ordinary controls
twice (one no-op and one informative 10-round fit). Full repaired fits are
bitwise identical; pre-failure rounds/updates match the retained reproduction.
A native-head-output VJP and independent full 10-round arithmetic pass.
The original dead worker did not serialize its fit, so no equality with its
lost memory or independent full decoder Jacobian is claimed.

Qualification accepted zero predictions. Formal first missing arrival1882
matches the actual qualified fit bitwise before being accepted, with exact
old-prefix linkage and independent writeback/reset checks. HC2's sealed
10,446 predictions are not rerun. VidSTG's suffix continues with the original
fixed configuration and three orders. Progress is a timestamped snapshot,
not full P1 or paper completion. Both directions and all 41,355 arrivals must
seal before P1 GT scoring; EATA remains paused.

Portable synthetic chart/gradient/trace contracts:

```bash
python -B scripts/test_stvg_opd_chart_revision003.py
```

The GPU/controller/readback scripts require the original private locked
research workspace; this export is inspectable code and safe scalar/hash
evidence, not a bundled dataset or model. Private frames, captions, GT,
weights, raw predictions, optimizer/gradient tensors and binary fits are
excluded. Timed qualification fit segments do not include every capture,
model load or replay overhead and are not total recovery or full-run costs.
''')
    write(PUB / 'CODE_BINDING.json', dict(code={f: sha(ROOT / f) for f in sorted(owned)},
        original_science_runtime_sha256=sha(BASE / 'RUNTIME_LOCK.json'),
        selected_config_sha256=sha(ROOT / 'methods/decota_spatial_opd_v1/configs.json'),
        authorized_runtime_sha256=sha(REC / 'REVISION_RUNTIME.json'),
        actual_GPU_qualification=True, actual_formal_resumption=True,
        P1_GT_efficacy_unmeasured=True, full_P1_or_paper_complete=False,
        private_payload_exported=False))
    owned |= {str(p.relative_to(ROOT)) for p in PUB.iterdir() if p.is_file()}
    for rel in sorted(owned):
        assert Path(rel).suffix in {'.py', '.md', '.json'}
        raw = (ROOT / rel).read_bytes()
        if rel.endswith('.json'):
            for forbidden in [b'"caption":', b'"video_path":', b'"GT_box":',
                              b'"gradient":', b'"optimizer_state":', b'"committed":',
                              b'"raw_logits":', b'"boxes":']:
                assert forbidden not in raw, (rel, forbidden)
        dest = CHECKOUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    subprocess.run(['git', 'add', '--', *sorted(owned)], cwd=CHECKOUT, check=True)
    changed = set(git('diff', '--cached', '--name-only').splitlines())
    assert changed <= owned and changed
    files = [dict(path=f, sha256=sha(CHECKOUT / f), bytes=(CHECKOUT / f).stat().st_size,
                  blob_sha=git('hash-object', f)) for f in sorted(owned)]
    write(REC / 'PUBLIC_STAGE.json', dict(status='reviewed_staged', repository='Zonglin-He/A',
        branch=BRANCH, base_commit=git('rev-parse', 'HEAD'), base_tree=git('rev-parse', 'HEAD^{tree}'),
        expected_tree=git('write-tree'), files=files, file_count=len(files),
        bytes=sum(f['bytes'] for f in files), full_P1_or_paper_complete=False, time=time.time()))
    print(json.dumps(dict(status='reviewed_staged', files=len(files), bytes=sum(f['bytes'] for f in files))))


def verify_remote(commit):
    meta = read(REC / 'PUBLIC_STAGE.json')
    for branch in ['main', BRANCH]:
        ref = json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/' + branch))
        assert ref['object']['sha'] == commit
    info = json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/commits/' + commit))
    assert info['parents'][0]['sha'] == meta['base_commit'] and info['tree']['sha'] == meta['expected_tree']
    def one(f):
        content = fetch('https://raw.githubusercontent.com/Zonglin-He/A/' + commit + '/' + urllib.parse.quote(f['path']))
        assert content == (CHECKOUT / f['path']).read_bytes()
        assert len(content) == f['bytes'] and hashlib.sha256(content).hexdigest() == f['sha256']
        assert hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest() == f['blob_sha']
        return dict(f, remote_bytes_identical=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(one, meta['files']))
    contracts = json.loads(subprocess.check_output([str(PYTHON), '-B',
        'scripts/test_stvg_opd_chart_revision003.py'], cwd=CHECKOUT))
    assert contracts['status'] == 'pass'
    verify_revision()
    assert git('rev-parse', 'HEAD') == commit and not git('status', '--porcelain')
    write(REC / 'FINAL_GITHUB_RECEIPT.json', dict(status='pass', repository='Zonglin-He/A',
        commit=commit, branches=['main', BRANCH], files=files, file_count=len(files),
        bytes=sum(f['bytes'] for f in files), public_CPU_contracts=contracts,
        actual_bounded_repair_and_resumption_only=True, P1_or_paper_complete=False, time=time.time()))
    print(json.dumps(dict(status='remote_contents_verified', commit=commit, files=len(files))))


if __name__ == '__main__':
    stage() if sys.argv[1] == 'stage' else verify_remote(sys.argv[2])
