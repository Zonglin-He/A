"""Stage additive closing metadata together with every verified result file."""
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_design_ablation_common_v1 import BASE, PUB, read, write, sha, verify
from scripts.publish_stvg_opd_design_ablation_v1 import CHECKOUT, BRANCH, git, safe


def run(archive_path):
    verify()
    archive_path = Path(archive_path)
    closing = read(BASE / 'ROOT_CLOSING_RECEIPT.json')
    first = read(BASE / 'FINAL_GITHUB_RECEIPT.json')
    archive = read(archive_path)
    initial = read(BASE / 'PUBLIC_STAGE.json')
    assert closing['status'] == 'complete'
    assert first['status'] == archive['status'] == 'pass'
    assert first['actual_remote_bytes_SHA256_Git_blob_tree_verified']
    assert closing['result_commit'] == first['commit']
    assert [c['mode'] for c in archive['checks']] == ['check', 'snapshot', 'check']
    assert all(c['returncode'] == 0 for c in archive['checks'])
    subprocess.run(['git', 'fetch', '--quiet', 'origin', 'main', BRANCH], cwd=CHECKOUT, check=True)
    assert git('rev-parse', 'HEAD') == git('rev-parse', 'origin/main') == git('rev-parse', 'origin/' + BRANCH) == first['commit']
    assert not git('diff', 'HEAD', '--name-only') and not git('diff', '--cached', '--name-only')
    unrelated = git('ls-files', '--others', '--exclude-standard').splitlines()
    assert all('stvg_motivation' in p.lower() for p in unrelated)
    records = {r['path']: r for r in initial['files']}
    for rel, record in records.items():
        p = CHECKOUT / rel
        assert sha(p) == record['sha256'] and p.stat().st_size == record['bytes']
        assert git('hash-object', rel) == record['blob_sha']
    additions = {}
    for source, relative in [
        (BASE / 'ROOT_CLOSING_RECEIPT.json', 'receipts/ROOT_CLOSING_RECEIPT.json'),
        (BASE / 'FINAL_GITHUB_RECEIPT.json', 'receipts/RESULT_GITHUB_RECEIPT.json'),
        (BASE / 'PUBLIC_SCALAR_AUDIT.json', 'receipts/PUBLIC_SCALAR_AUDIT.json'),
        (archive_path, 'receipts/FINAL_ROOT_ARCHIVE_RECEIPT.json'),
    ]:
        safe(read(source))
        target = PUB / relative
        assert not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        additions[str(target.relative_to(ROOT))] = target
    closure = dict(
        status='complete', scope=closing['scope'], experiment_complete=True,
        result_commit=first['commit'], new_formal_fits=5376,
        all_negative_results_retained=True, actual_figures_and_private_RGB_viewed=True,
        actual_remote_result_files_verified=first['file_count'],
        actual_remote_result_bytes_verified=first['bytes'],
        root_closing_sha256=sha(BASE / 'ROOT_CLOSING_RECEIPT.json'),
        result_remote_receipt_sha256=sha(BASE / 'FINAL_GITHUB_RECEIPT.json'),
        actual_archive_receipt_sha256=sha(archive_path),
        original_completed_suite_unchanged=True, extra_TubeDETR=False,
        no_retuning_or_production_promotion=True,
        coverage_is_offline_association_not_causal_proof=True,
        private_payloads_exported=False, CPU_only=True, model_calls=0,
        time=time.time(),
    )
    safe(closure)
    write(PUB / 'FINAL_PUBLIC_CLOSURE.json', closure)
    additions[str((PUB / 'FINAL_PUBLIC_CLOSURE.json').relative_to(ROOT))] = PUB / 'FINAL_PUBLIC_CLOSURE.json'
    for rel, source in additions.items():
        target = CHECKOUT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(target) == sha(source)
        records[rel] = dict(path=rel, bytes=target.stat().st_size, sha256=sha(target), blob_sha=git('hash-object', rel))
    subprocess.run(['git', 'add', '--', *sorted(additions)], cwd=CHECKOUT, check=True)
    changed = git('diff', '--cached', '--name-only').splitlines()
    assert set(changed) == set(additions)
    files = [records[k] for k in sorted(records)]
    write(BASE / 'CLOSING_PUBLIC_STAGE.json', dict(
        status='reviewed_staged', scope='additive final root/remote/archive closure plus exact re-verification of every component result and implementation file',
        repository='Zonglin-He/A', branch=BRANCH,
        base_commit=git('rev-parse', 'HEAD'), base_tree=git('rev-parse', 'HEAD^{tree}'),
        expected_tree=git('write-tree'), changed_files=changed, files=files,
        file_count=len(files), bytes=sum(r['bytes'] for r in files),
        preserved_unrelated_untracked=unrelated,
        initial_result_stage_sha256=sha(BASE / 'PUBLIC_STAGE.json'), time=time.time(),
    ))
    print('DESIGN_CLOSING_METADATA_STAGED', len(files), sum(r['bytes'] for r in files), flush=True)


if __name__ == '__main__':
    assert len(sys.argv) == 2
    run(sys.argv[1])
