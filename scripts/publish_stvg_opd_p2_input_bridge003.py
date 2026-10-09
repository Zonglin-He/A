"""Stage exact anonymous P2 qualification and strict input recovery evidence."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stvg_opd_p2_input_bridge003 import REC, verify
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE, read, write, sha

OUT = ROOT / 'results/stvg_opd_p2_input_recovery/2026-10-09'
CHECKOUT = ROOT.parent / 'visual-grounding-public-A'
BRANCH = 'research/stvg-opd-paper-hc2-revision-v2'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=CHECKOUT, text=True).strip()


def run():
    verify()
    prior = read(BASE / 'P2_engineering_revision002/PUBLIC_STAGE.json')
    assert prior['status'] == 'reviewed_staged'
    for name in ['ROOT_QUALIFICATION_READBACK.json', 'ROOT_RESUME_READBACK.json']:
        assert read(REC / name)['status'] == 'pass'
    root = read(REC / 'ROOT_RESUME_READBACK.json')
    assert root['accepted_snapshot_predictions'] > 0 and root['GT_read'] is False
    subprocess.run(['git', 'fetch', '--quiet', 'origin', 'main', BRANCH], cwd=CHECKOUT, check=True)
    assert git('remote', 'get-url', 'origin') == 'https://github.com/Zonglin-He/A.git'
    assert git('rev-parse', 'HEAD') == prior['base_commit']
    assert git('rev-parse', 'origin/main') == git('rev-parse', 'origin/' + BRANCH) == prior['base_commit']
    prior_files = {r['path'] for r in prior['files']}
    for r in prior['files']:
        assert sha(CHECKOUT / r['path']) == r['sha256']
        assert (CHECKOUT / r['path']).stat().st_size == r['bytes']
    assert set(git('diff', 'HEAD', '--name-only').splitlines()) <= prior_files
    assert set(git('diff', '--cached', '--name-only').splitlines()) <= prior_files
    untracked = git('ls-files', '--others', '--exclude-standard').splitlines()
    assert untracked == prior['preserved_unrelated_untracked']
    mapped = {}

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        mapped[str(source.relative_to(ROOT))] = dict(
            public_path=str(target.relative_to(ROOT)), sha256=sha(source), bytes=source.stat().st_size)

    for name in ['CAPTURE_RECEIPT.json', 'REVISION_RUNTIME.json',
                 'CPU_ALL_LEGACY_ROOT_READBACK.json', 'ROOT_QUALIFICATION_READBACK.json',
                 'ROOT_RESUME_READBACK.json', 'ENGINEERING_ROOT_REVIEW.json', 'LAUNCH.json']:
        copy(REC / name, OUT / 'receipts' / name)
    copy(REC / 'root_resume_helper_001/CAPTURE_RECEIPT.json',
         OUT / 'receipts/CPU_ROOT_HELPER_CAPTURE.json')
    for stage in ['P2_hc2_cross_clean', 'P2_vidstg_cross_clean']:
        for name in ['GPU_QUALIFICATION.json', 'ROOT_READBACK.json']:
            copy(REC / 'qualification' / stage / name, OUT / 'qualification' / stage / name)
        first = REC / 'qualification' / stage / 'FIRST_FORMAL_FIT_BITWISE.json'
        if first.exists():
            copy(first, OUT / 'qualification' / stage / first.name)
    pins = set(read(REC / 'REVISION_RUNTIME.json')['pins']) | {
        'scripts/review_stvg_opd_p2_input_bridge003.py',
        'scripts/publish_stvg_opd_p2_input_bridge003.py'}
    write(OUT / 'CODE_BINDING.json', dict(
        status='actual_qualified_metadata_repair_and_formal_resumption',
        scope='bounded P2 input-schema repair only; no P2 efficacy or full-paper completion',
        code={f: sha(ROOT / f) for f in sorted(pins)}, exact_metadata_projection=mapped,
        prior_full_numerical_qualification_fits=192, new_input_bridge_GPU_fits=4,
        accepted_snapshot_predictions=root['accepted_snapshot_predictions'],
        original_science_unchanged=True, GT_read=False, P2_global_seal=False,
        paper_suite_complete=False))
    (OUT / 'README.md').write_text('''# Fixed P2: strict legacy input compatibility and real continuation

P1 has completed its actual root, visual, anonymous publication and archive gate.
All P1 results, including the inconclusive VidSTG contrast against DINO Refine,
remain in `results/stvg_opd_p1_complete/2026-10-09`.

The original P2 four arms and all source checkpoints, rosters, parameters,
physical conditions, order resets, inherited LN, query/Adam resets and final
readout are unchanged. The historical engineering002 launch and its full
96-arrival/192-real-fit qualification remain in
`results/stvg_opd_p2_launch/2026-10-09`.

That controller subsequently failed before accepting any formal P2 prediction:
the older P0 clean-input header omitted `source_model_state_sha256`, although
it already referenced the original baseline NPZ by its exact cache hash. The
failure occurred before fitting. Original code, locks, logs, qualification
receipts and pre-repair status snapshots are preserved; there is no dead
failed-fit serialization to compare.

The additive input-schema003 bridge verifies the original hash-bound cache,
its original runtime receipt, exact pixels/frame IDs/interval/native boxes and
the complete admitted expert evidence before reading its recorded source hash.
No input bytes or old receipts are rewritten. All 256 legacy inputs were read
back, four old/current pairs passed and 48 deliberately wrong variants failed.
Unknown legacy schemas fail closed; source provenance is not guessed and no
input equality check is relaxed.

Both first-formal Direct fits were executed and fully repeated on the real GPU:
four new fits, exact equality against the previous real qualifications, original
math/state/reset/writeback readback, zero qualification predictions accepted,
zero new DINO calls and no GT. Formal acceptance then required exact equality
against this new qualified first fit. A new single finite controller actually
resumed the original formal stream. The bounded root receipt records an actual
accepted immutable prefix, not a projected future total or phase seal.
An initial unsealed CPU root-helper namespace initialization error was also
preserved and repaired; the subsequent complete prefix readback passed without
changing any GPU fit, prediction or receipt.

This completes a metadata recovery and resumption gate only. All P2 arms and
both directions must globally seal before GT scoring. Actual root statistics,
math/state/dense/cost/cases, plot inspection, anonymous publication and archive
closing are still required before P3. P2 and the paper remain incomplete. EATA
and unrelated old queues remain paused. The old controller PID and old running
text are historical; use the latest schema003 launch and dynamic stage status.
Private media, query/caption, GT geometry, weights, box/actions, qualified or
formal fit payloads, gradients and Adam arrays are excluded.
''')
    additions = pins | {str(f.relative_to(ROOT)) for f in OUT.rglob('*') if f.is_file()}
    files = sorted(prior_files | additions)
    forbidden = [b'"caption":', b'"video_path":', b'"native_boxes":',
                 b'"GT_box":', b'"committed":', b'"gradient":',
                 b'"optimizer_state":', b'"raw_logits":']
    for rel in files:
        assert rel.startswith(('scripts/', 'vg_tta/', 'protocols/', 'results/'))
        assert Path(rel).suffix in {'.py', '.md', '.json'}
        if rel.endswith('.json'):
            raw = (ROOT / rel).read_bytes()
            for token in forbidden:
                assert token not in raw, (rel, token)
        dest = CHECKOUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    subprocess.run(['git', 'add', '--', *files], cwd=CHECKOUT, check=True)
    changed = git('diff', '--cached', '--name-only').splitlines()
    assert changed and set(changed) <= set(files)
    assert git('ls-files', '--others', '--exclude-standard').splitlines() == untracked
    records = [dict(path=f, sha256=sha(CHECKOUT / f), bytes=(CHECKOUT / f).stat().st_size,
                    blob_sha=git('hash-object', f)) for f in files]
    write(REC / 'PUBLIC_STAGE.json', dict(
        status='reviewed_staged', scope='actual P1 closing, 192 P2 qualifications and 4-fit strict input recovery/resumption',
        repository='Zonglin-He/A', branch=BRANCH,
        base_commit=git('rev-parse', 'HEAD'), base_tree=git('rev-parse', 'HEAD^{tree}'),
        expected_tree=git('write-tree'), files=records, changed_files=changed,
        file_count=len(records), bytes=sum(v['bytes'] for v in records),
        preserved_unrelated_untracked=untracked, GT_read=False,
        paper_suite_complete=False, time=time.time()))
    print(json.dumps(dict(status='ready', files=len(records), changed=len(changed),
                          bytes=sum(v['bytes'] for v in records))))


if __name__ == '__main__':
    run()
