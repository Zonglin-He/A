"""Exact user-authorized retirement of unused official model weight files.

Never traverse experimental outputs for deletion. Retain download receipts,
configs, model indexes, source code, and all trained research checkpoints.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/storage_cleanup_20261001_models'
QUICK = ROOT / 'artifacts/tastvg_best_quick_v1'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def identity(path):
    stat = path.stat()
    return dict(dev=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, allocated=stat.st_blocks * 512,
                links=stat.st_nlink)


def candidate_files():
    qualification = ROOT / 'artifacts/ptd_pretrained_teacher_qualification_v1'
    reg = read(qualification / 'REGISTRATION.json')
    finished = read(qualification / 'DOWNLOAD_COMPLETE.json')
    specs = []
    for name, key in [('TaRO-8B', 'taro'), ('STVG-R1-7B', 'stvg')]:
        model = reg['models'][key]
        receipt = qualification / 'DOWNLOAD_COMPLETE.json'
        expected = {Path(row['path']).name: row for row in finished['files']
                    if row['model'] == key}
        specs.append((name, model['repo'], model['revision'], receipt, expected))
    for name, rel in [
        ('LLaVA-ST-Qwen2-7B', 'artifacts/desta3d_v3/external_privileged_opd_v1'),
        ('llava_official_siglip-so400m-patch14-384',
         'artifacts/desta3d_v3/external_privileged_opd_v2/siglip_download')]:
        folder = ROOT / rel
        model = read(folder / 'MODEL_DOWNLOAD_REGISTRATION.json')
        receipt = folder / 'MODEL_DOWNLOAD_COMPLETE.json'
        expected = {row['path']: row for row in read(receipt)['files']}
        specs.append((name, model['repo'], model['revision'], receipt, expected))
    rows = []
    for name, repo, revision, receipt, expected in specs:
        weights = sorted((ROOT / 'checkpoints' / name).glob('*.safetensors'))
        assert len(weights) == (1 if name.startswith('llava_official') else 4), name
        for path in weights:
            assert not path.is_symlink()
            stat = identity(path)
            old = expected[path.name]
            assert stat['links'] == 1
            assert stat['bytes'] == old.get('bytes', old.get('size'))
            assert stat['mtime_ns'] <= receipt.stat().st_mtime_ns
            digest = old.get('sha256', old.get('actual_sha256'))
            assert digest and len(digest) == 64
            rows.append(dict(path=str(path.relative_to(ROOT)), **stat,
                category=name, reason='Official frozen teacher weights of a paused old route; absent from active TA-STVG and deployed DeCoTA dependencies',
                repo=repo, revision=revision, historical_verified_sha256=digest,
                historical_receipt=str(receipt.relative_to(ROOT)),
                restore_url=f'https://huggingface.co/{repo}/resolve/{revision}/{path.name}'))
    partial = ROOT / 'checkpoints/Qwen3-VL-32B-Instruct-GGUF/mmproj-Qwen3VL-32B-Instruct-F16.gguf.sequential'
    if partial.is_file():
        assert not partial.is_symlink()
        rows.append(dict(path=str(partial.relative_to(ROOT)), **identity(partial),
            category='cancelled_download_fragment',
            reason='Incomplete sequential fragment of cancelled Qwen32B download; no active route'))
    pip = Path('/home/wwww/.cache/pip')
    for path in sorted(pip.rglob('*')):
        if path.is_file() and not path.is_symlink():
            rows.append(dict(path=str(path), **identity(path), category='pip_download_cache',
                reason='Rebuildable installer download cache; installed packages untouched'))
    return rows


def resolve(row):
    path = Path(row['path'])
    if not path.is_absolute():
        path = ROOT / path
    allowed = [ROOT / 'checkpoints' / name for name in [
        'TaRO-8B', 'STVG-R1-7B', 'LLaVA-ST-Qwen2-7B',
        'llava_official_siglip-so400m-patch14-384', 'Qwen3-VL-32B-Instruct-GGUF']]
    allowed.append(Path('/home/wwww/.cache/pip'))
    assert any(path.is_relative_to(folder) for folder in allowed), path
    assert path.resolve() == path and not path.is_symlink(), path
    return path


def runtime_readback():
    lock = read(QUICK / 'RUNTIME_LOCK.json')
    pins = dict(lock['pins'])
    for revision in sorted((QUICK / 'revisions').glob('*.json')):
        pins.update(read(revision)['pins'])
    for rel, digest in pins.items():
        assert sha(ROOT / rel) == digest, rel
    for rel, digest in lock['metadata'].items():
        assert sha(QUICK / rel) == digest, rel
    state = read(QUICK / 'STATUS.json')
    if state['status'] == 'running':
        for field in ['controller_pid', 'worker_pid']:
            pid = state.get(field)
            if pid:
                cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
                assert 'tastvg_best_quick' in cmd, (field, pid, cmd)
    return dict(status=state, verified_pin_count=len(pins),
                runtime_lock_sha256=sha(QUICK / 'RUNTIME_LOCK.json'))


def protected_state():
    roots = ['checkpoints/Sa2VA-4B', 'checkpoints/universalvtg',
             'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B', 'checkpoints/tastvg_model_zoo',
             'checkpoints/STVG-R1-tracking', '.cache/grounding_dino_http_v1',
             '.cache/huggingface', '.cache/torch', '.cache/stanza']
    paths = []
    for rel in roots:
        paths.extend(path for path in (ROOT / rel).rglob('*') if path.is_file())
    paths.extend(path for path in (ROOT / 'checkpoints').glob('*.pth') if path.is_file())
    return {str(path.relative_to(ROOT)): identity(path) for path in sorted(set(paths))}


def open_file_check(rows):
    targets = {(row['dev'], row['inode']) for row in rows}
    opened = []
    inaccessible = 0
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    stat = fd.stat()
                    if (stat.st_dev, stat.st_ino) in targets:
                        opened.append(str(fd))
                except (OSError, PermissionError):
                    pass
        except (OSError, PermissionError):
            inaccessible += 1
        try:
            for line in (proc / 'maps').read_text().splitlines():
                fields = line.split(maxsplit=5)
                major, minor = (int(part, 16) for part in fields[3].split(':'))
                if (os.makedev(major, minor), int(fields[4])) in targets:
                    opened.append(str(proc / 'maps'))
        except (OSError, PermissionError):
            pass
    assert not opened, opened
    return dict(open_or_mapped_targets=opened,
                permission_limited_processes=inaccessible,
                scope='Accessible process fds and maps; not a claim of full-system observability')


def main(execute):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / 'DELETE_MANIFEST.json'
    if not execute:
        assert not manifest.exists()
        rows = candidate_files()
        protected = protected_state()
        assert not {str(resolve(row)) for row in rows}.intersection(str(ROOT / rel) for rel in protected)
        write(manifest, dict(time=time.time(), explicit_user_authorization='Delete unused disk assets including unused model weights',
            files=rows, file_count=len(rows), logical_bytes=sum(row['bytes'] for row in rows),
            allocated_bytes=sum(row['allocated'] for row in rows),
            sha_policy='Model SHA identities come from preserved verified official download receipts; no fresh bulk weight rehash',
            runtime_before=runtime_readback(), protected_before=protected,
            untouched=['all predictions/reports/negative evidence/trained checkpoints', 'datasets/media/annotations',
                       'active TA-STVG/Sa2VA/UniversalVTG and deployed DeCoTA assets', 'PTD Fig1 weights',
                       'model configs/tokenizers/indexes/official code/download manifests', 'all installed environments']))
        print(json.dumps(dict(status='planned', files=len(rows), allocated_GiB=sum(row['allocated'] for row in rows)/2**30)))
        return
    assert not (OUT / 'COMPLETE.json').exists()
    plan = read(manifest)
    rows = plan['files']
    before_runtime = runtime_readback()
    assert protected_state() == plan['protected_before']
    process_check = open_file_check(rows)
    free_before = shutil.disk_usage(ROOT).free
    with (OUT / 'DELETED.jsonl').open('x') as log:
        for row in rows:
            path = resolve(row)
            assert identity(path) == {key: row[key] for key in identity(path)}, path
            path.unlink()
            log.write(json.dumps(dict(path=row['path'], bytes=row['bytes'], time=time.time())) + '\n')
            log.flush()
        os.fsync(log.fileno())
    free_after = shutil.disk_usage(ROOT).free
    assert all(not resolve(row).exists() for row in rows)
    assert protected_state() == plan['protected_before']
    after_runtime = runtime_readback()
    categories = {}
    for row in rows:
        categories[row['category']] = categories.get(row['category'], 0) + row['allocated']
    result = dict(status='completed', time=time.time(), files_deleted=len(rows),
        logical_bytes_deleted=sum(row['bytes'] for row in rows),
        allocated_bytes_deleted=sum(row['allocated'] for row in rows),
        categories_allocated_bytes=categories, free_before=free_before, free_after=free_after,
        measured_net_free_increase=free_after-free_before,
        protected_assets_stat_unchanged=True, active_runtime_pins_and_metadata_unchanged=True,
        experiment_before=before_runtime['status'], experiment_after=after_runtime['status'],
        process_check=process_check, research_outputs_deleted=False, datasets_deleted=False,
        trained_checkpoints_deleted=False,
        replay_scope='Retired old teacher weights must be restored from the recorded official revisions before replay; historical results and metadata remain preserved.')
    write(OUT / 'COMPLETE.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    main(parser.parse_args().execute)
