"""Retire exact obsolete download fragments, installers, and VS Code servers."""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from cleanup_unused_model_weights_20261001 import (
    ROOT, identity, open_file_check, protected_state, read, runtime_readback, write,
)

OUT = ROOT / 'artifacts/storage_cleanup_20261001_other'
SERVERS = Path('/home/wwww/.vscode-server/cli/servers')


def selections():
    files = []
    folders = []
    specs = [
        ('checkpoints/STCAT_hc_res416.partial', 'cancelled_download_fragment', 'Abandoned incomplete checkpoint of an inactive historical model'),
        ('checkpoints/UniVG-R1/tokenizer.json.partial', 'cancelled_download_fragment', 'Abandoned incomplete tokenizer of a retired model'),
        ('downloads/dataset_intake_20260910/jhmdb/datasets/JHMDB/Rename_Images.tar.gz.partial',
         'cancelled_download_fragment', 'Incomplete old intake archive; inactive download and unusable as complete media'),
        ('.runtime/nvidia-580.159.03/libnvidia-compute-580-server_580.159.03_amd64.deb',
         'extracted_installer', 'Original deb installer; extracted runtime libraries retained'),
        ('external/runtime_libs/nvidia_580_159_03/libnvidia-compute-580-server_580.159.03-0ubuntu0.25.10.1_amd64.deb',
         'extracted_installer', 'Original deb installer; extracted runtime libraries retained'),
        ('artifacts/decota_paper_execution_20260917/runtime/libnvidia-compute-580-server_580.159.03_amd64.deb',
         'extracted_installer', 'Archived installer copy; retained extracted runtime and all scientific outputs'),
        ('external/llama_cpp_b11158/llama-b11158-bin-ubuntu-cuda-12.8-x64.tar.gz',
         'extracted_installer', 'Release tarball already extracted; binaries and release provenance retained'),
    ]
    for rel, category, reason in specs:
        path = ROOT / rel
        if path.is_file():
            assert not path.is_symlink()
            files.append(dict(path=str(path), **identity(path), category=category, reason=reason))
    for name in ['model-00001-of-00004.safetensors', 'model-00002-of-00004.safetensors',
                 'model-00003-of-00004.safetensors']:
        final = ROOT / 'checkpoints/Sa2VA-4B' / name
        partial = final.with_suffix(final.suffix + '.partial')
        if partial.is_file():
            assert final.is_file() and final.stat().st_size > partial.stat().st_size
            receipt = read(final.parent / 'DOWNLOAD_RECEIPT.json')
            assert name in {row['file'] for row in receipt['files']}
            files.append(dict(path=str(partial), **identity(partial), category='redundant_download_fragment',
                reason='Leftover partial beside qualified complete Sa2VA shard; final shard and verified download receipt retained'))
    for name in ['Qwen3-VL-32B-Instruct-GGUF', 'ParallelTubeDecoding-Qwen3-VL-4B']:
        for path in (ROOT / 'checkpoints' / name / '.cache/huggingface/download').glob('*.incomplete'):
            if path.stat().st_size == 0:
                files.append(dict(path=str(path), **identity(path), category='zero_byte_download_fragment',
                    reason='Stale zero-byte incomplete placeholder; complete model or retired-route metadata retained'))
    installer = Path('/home/wwww/下载/NVIDIA-Linux-x86_64-570.207.run')
    if installer.is_file():
        driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
        assert driver.startswith('580.'), driver
        files.append(dict(path=str(installer), **identity(installer), category='obsolete_driver_installer',
            reason=f'Obsolete570 driver installer; actual running driver{driver} and all installed files retained'))
    versions = []
    for path in SERVERS.glob('Stable-*'):
        package = path / 'server/package.json'
        product = path / 'server/product.json'
        assert package.is_file() and product.is_file() and not path.is_symlink()
        version = read(package)['version']
        commit = read(product)['commit']
        assert path.name == 'Stable-' + commit
        versions.append((tuple(int(part) for part in version.split('.')), path, version))
    newest = max(versions)[1]
    for _, folder, version in versions:
        if folder == newest:
            continue
        folders.append(dict(path=str(folder), category='obsolete_vscode_server', version=version,
                            reason='Older inactive reproducible server installation; newest server and all user data/extensions retained'))
        for path in sorted(folder.rglob('*')):
            if path.is_file() and not path.is_symlink():
                files.append(dict(path=str(path), **identity(path), category='obsolete_vscode_server',
                    reason='File of older inactive VS Code server installation'))
    for folder in ['.runtime/nvidia-580.159.03/extracted', 'external/runtime_libs/nvidia_580_159_03/extracted',
                   'artifacts/decota_paper_execution_20260917/runtime/nvidia580159',
                   'external/llama_cpp_b11158/llama-b11158']:
        assert any((ROOT / folder).rglob('*')), folder
    return files, folders, newest


def protected(rows, newest):
    selected = {row['path'] for row in rows}
    state = {str(ROOT / path): value for path, value in protected_state().items()
             if str(ROOT / path) not in selected}
    for folder in [newest, ROOT / '.runtime/nvidia-580.159.03/extracted',
                   ROOT / 'external/runtime_libs/nvidia_580_159_03/extracted',
                   ROOT / 'artifacts/decota_paper_execution_20260917/runtime/nvidia580159',
                   ROOT / 'external/llama_cpp_b11158/llama-b11158']:
        for path in folder.rglob('*'):
            if path.is_file():
                state[str(path)] = identity(path)
    return state


def check_inactive(folders):
    ancestors = {os.getpid(), os.getppid()}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if proc.stat().st_uid != os.getuid() or int(proc.name) in ancestors:
                continue
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            exe = str((proc / 'exe').resolve())
            for folder in folders:
                assert not exe.startswith(folder['path'] + '/'), (proc.name, exe)
                assert folder['path'] not in cmd, (proc.name, cmd)
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            pass


def main(execute):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / 'DELETE_MANIFEST.json'
    if not execute:
        assert not manifest.exists()
        files, folders, newest = selections()
        check_inactive(folders)
        write(manifest, dict(time=time.time(), authority='User explicitly authorized deleting other unused files as well',
            files=files, directories=folders, newest_server=str(newest),
            runtime_before=runtime_readback(), protected_before=protected(files, newest),
            file_count=len(files), allocated_bytes=sum(row['allocated'] for row in files)))
        print(json.dumps(dict(status='planned', files=len(files), old_server_versions=len(folders),
                             allocated_GiB=sum(row['allocated'] for row in files)/2**30)))
        return
    assert not (OUT / 'COMPLETE.json').exists()
    plan = read(manifest)
    rows, folders = plan['files'], plan['directories']
    newest = Path(plan['newest_server'])
    assert protected(rows, newest) == plan['protected_before']
    before_runtime = runtime_readback()
    check_inactive(folders)
    process_check = open_file_check(rows)
    free_before = shutil.disk_usage(ROOT).free
    for row in rows:
        path = Path(row['path'])
        assert not path.is_symlink()
        assert identity(path) == {key: row[key] for key in identity(path)}, path
    with (OUT / 'DELETED.jsonl').open('x') as log:
        for row in rows:
            path = Path(row['path'])
            assert identity(path) == {key: row[key] for key in identity(path)}, path
            path.unlink()
            log.write(json.dumps(dict(path=row['path'], bytes=row['bytes'], time=time.time())) + '\n')
        log.flush()
        os.fsync(log.fileno())
    for folder in folders:
        path = Path(folder['path'])
        assert path.parent == SERVERS and not path.is_symlink()
        assert all(not entry.is_file() or entry.is_symlink() for entry in path.rglob('*'))
        shutil.rmtree(path)
    free_after = shutil.disk_usage(ROOT).free
    assert all(not Path(row['path']).exists() for row in rows)
    assert protected(rows, newest) == plan['protected_before']
    after_runtime = runtime_readback()
    categories = {}
    for row in rows:
        categories[row['category']] = categories.get(row['category'], 0) + row['allocated']
    result = dict(status='completed', time=time.time(), files_deleted=len(rows),
        old_server_versions_deleted=len(folders), newest_server_retained=str(newest),
        allocated_bytes_deleted=sum(row['allocated'] for row in rows),
        categories_allocated_bytes=categories, free_before=free_before, free_after=free_after,
        measured_net_free_increase=free_after-free_before, process_check=process_check,
        active_runtime_pins_and_metadata_unchanged=True, protected_assets_stat_unchanged=True,
        experiment_before=before_runtime['status'], experiment_after=after_runtime['status'],
        scientific_predictions_deleted=False, datasets_deleted=False, trained_checkpoints_deleted=False,
        installed_environments_and_user_editor_data_untouched=True,
        reproducibility='Deleted fragments require a new download if needed; deleted installers or old server versions require re-fetch, while installed runtime binaries remain available.')
    write(OUT / 'COMPLETE.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    main(parser.parse_args().execute)
