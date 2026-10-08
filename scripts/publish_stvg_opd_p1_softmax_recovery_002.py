"""Public audit-code/evidence export, excluding every private prediction payload."""
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
from scripts.run_stvg_opd_p1_softmax_recovery_002 import REC, BASE, verify_revision
from scripts.stvg_opd_paper_hc2_revision_common_v2 import PYTHON, read, write, sha

CHECKOUT = ROOT.parent / 'visual-grounding-public-A'
PUB = ROOT / 'results/stvg_opd_p1_precision_recovery/2026-10-08'
BRANCH = 'research/stvg-opd-paper-hc2-revision-v2'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=CHECKOUT).decode().strip()


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'STVG-OPD-byte-audit'}), timeout=35) as response:
        return response.read()


def stage():
    verify_revision()
    resume = read(REC / 'ROOT_RESUME_READBACK.json')
    assert resume['status'] == 'actually_resumed_after_requalification' and not resume['GT_read']
    assert git('remote', 'get-url', 'origin') == 'https://github.com/Zonglin-He/A.git'
    assert git('branch', '--show-current') == BRANCH
    assert git('rev-parse', 'HEAD') == git('rev-parse', 'origin/main') == git('rev-parse', 'origin/' + BRANCH)
    assert not git('status', '--porcelain')
    assert not PUB.exists()
    PUB.mkdir(parents=True)
    files = set(read(REC / 'REVISION_RUNTIME.json')['pins']) | {
        'scripts/test_stvg_opd_softmax_precision_revision002.py',
        'scripts/reproduce_stvg_opd_p1_softmax_failure_002.py',
        'scripts/publish_stvg_opd_p1_softmax_recovery_002.py',
    }
    for name in ['REVISION_RUNTIME.json', 'REQUALIFICATION_ROOT_READBACK.json', 'ROOT_RESUME_READBACK.json', 'TORCH_SOURCE_REFERENCE.json']:
        shutil.copy2(REC / name, PUB / name)
    test = json.loads(subprocess.check_output([str(PYTHON), '-B', str(ROOT / 'scripts/test_stvg_opd_softmax_precision_revision002.py')], cwd=ROOT))
    assert test['status'] == 'pass'
    write(PUB / 'CPU_CONTRACTS.json', test)
    write(PUB / 'CODE_BINDING.json', dict(code={f: sha(ROOT / f) for f in sorted(files)},
        engineering_runtime_sha256=sha(REC / 'REVISION_RUNTIME.json'),
        original_science_runtime_sha256=sha(BASE / 'RUNTIME_LOCK.json'),
        actual_recovery_requalification_and_resume=True,
        public_export_contains_private_payloads=False, P1_task_scores_available=False))
    (PUB / 'README.md').write_text('''# Fixed P1: softmax audit precision supplement 002

This is a completed engineering recovery and an actual running-P1 launch record,
not a completed P1 experiment or evidence about grounding efficacy. Both-direction
P1 deployment predictions remain unsealed; no P1 GT was read during this recovery.

The independent audit failed after 3,630 receipted HC2 predictions (4,524,043,386
opaque bytes). Their prediction and receipt SHA256 hashes were verified and
preserved. Original code, scientific locks, and the original failure remain intact.
The scalar CUDA division implementation in the installed PyTorch 2.7.0 git revision
uses a float32 reciprocal and multiplication. A direct float64 reference has different
intermediate rounding. See the linked primary source in TORCH_SOURCE_REFERENCE.json
and protocols/stvg_opd_p1_softmax_precision_revision002.md.

The original float64/GPU32 probability discrepancy was 3.477426682718665e-7 and
failed the original 3e-7 absolute check. The supplement retains that failed result,
the unchanged 3e-7 matched-operation threshold, and the original gradient/Adam/state
checks. Matched-operation CPU32 and independent float64 normalization of rounded
CUDA logits differed by 2.9802322387695312e-8 and 1.968405555219377e-8, respectively.
An explicit gamma3 logit-rounding bound and componentwise softmax sensitivity bound
certify the cross-precision discrepancy. No fitting or scientific parameter changed.

Validation includes 16 synthetic CPU regimes, 16 deliberately wrong normalized
probability distributions rejected, all 40 rounds of the reproduced failed fit,
12 already-closed GPU qualification fits audited on CPU (not 12 new GPU fits),
and 9 historical prediction receipts whose saved audit dictionaries were reproduced
exactly. One actual GPU qualification replay matched the serialized first reproduction:
948 tensors, 270,844 coordinates, and 1,186 scalars. The first formal missing arrival
was repeated and matched again before acceptance. The old worker did not serialize
its original failing fit; bitwise equality is explicitly between deterministic actual
replays from the saved full prefix, not against lost dead-process memory.

HC2 remains lr=.01, sigma=.025, tau=.05, steps=40, LN writeback=1/16, M=32;
VidSTG remains .03/.1/.25/10/1/8/M32. The full inherited LN prefix is restored by
the unchanged runner; query residual and Adam reset per query. The finite controller
continues HC2 then VidSTG and only permits postseal GT after all 41,355 P1 arrivals.
The CPU bridge dispatches by the saved original/revision001/revision002 audit metadata
and does not rewrite older predictions, locks, or receipts. P1 postseal root audits,
views, all anonymous positive/negative results, and P2-P6 execution remain pending.
EATA and its media/Fisher preparation remain explicitly user paused.

CPU-only public contracts (PyTorch and NumPy required):

```bash
python -B scripts/test_stvg_opd_softmax_precision_revision002.py
```

The private failing fit, actions, weights, gradients, optimizer states, predictions,
source checkpoints, captions, GT, and media are excluded. Private replay requires
the separately obtained original research inputs. The requalification fit measured
1.508919641s of GPU fit wall time; the first reproduction was not separately timed,
so this does not claim total recovery cost or normal P1 throughput.
''')
    files |= {str(p.relative_to(ROOT)) for p in PUB.rglob('*') if p.is_file()}
    for f in sorted(files):
        assert f.startswith(('scripts/', 'vg_tta/', 'protocols/', 'results/'))
        assert Path(f).suffix in ('.py', '.md', '.json')
        raw = (ROOT / f).read_bytes()
        if f.endswith('.json'):
            for forbidden in [b'"caption":', b'"video_path":', b'"GT_box":', b'"gradient":', b'"optimizer_state":', b'"committed":']:
                assert forbidden not in raw, (f, forbidden)
        dest = CHECKOUT / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest)
    subprocess.run(['git', 'add', '--', *sorted(files)], cwd=CHECKOUT, check=True)
    assert set(git('diff', '--cached', '--name-only').splitlines()) == files
    records = [dict(path=f, sha256=sha(CHECKOUT/f), bytes=(CHECKOUT/f).stat().st_size,
                    blob_sha=git('hash-object', f)) for f in sorted(files)]
    write(REC / 'PUBLIC_STAGE.json', dict(status='reviewed_staged', repository='Zonglin-He/A',
        branch=BRANCH, base_commit=git('rev-parse', 'HEAD'), base_tree=git('rev-parse', 'HEAD^{tree}'),
        expected_tree=git('write-tree'), files=records, file_count=len(records),
        bytes=sum(r['bytes'] for r in records), whole_P1_or_paper_complete=False, time=time.time()))
    print(json.dumps(dict(status='reviewed_staged', files=len(records), bytes=sum(r['bytes'] for r in records))))


def verify_remote(commit):
    meta = read(REC / 'PUBLIC_STAGE.json')
    for branch in ['main', BRANCH]:
        ref = json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/' + branch))
        assert ref['object']['sha'] == commit
    info = json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/commits/' + commit))
    assert info['parents'][0]['sha'] == meta['base_commit'] and info['tree']['sha'] == meta['expected_tree']
    def one(row):
        raw = fetch('https://raw.githubusercontent.com/Zonglin-He/A/' + commit + '/' + urllib.parse.quote(row['path']))
        assert raw == (CHECKOUT / row['path']).read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        assert hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() == row['blob_sha']
        return dict(row, remote_bytes_identical=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(one, meta['files']))
    verify_revision()
    assert git('rev-parse', 'HEAD') == commit and not git('status', '--porcelain')
    write(REC / 'FINAL_GITHUB_RECEIPT.json', dict(status='pass', repository='Zonglin-He/A',
        commit=commit, branches=['main', BRANCH], files=records, file_count=len(records),
        bytes=sum(r['bytes'] for r in records), P1_task_results_not_claimed=True, time=time.time()))
    print(json.dumps(dict(status='remote_contents_verified', commit=commit, files=len(records))))


if __name__ == '__main__':
    stage() if sys.argv[1] == 'stage' else verify_remote(sys.argv[2])
