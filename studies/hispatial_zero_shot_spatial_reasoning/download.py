"""Fetch only the official benchmark assets at the pinned revisions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.parse import quote
from urllib.request import urlopen


STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
PROTOCOL = STUDY / 'OFFICIAL_BASELINE.json'
DATA = ROOT / '.cache' / 'official-datasets'
OUTPUT = ROOT / 'outputs' / 'official_baseline_20261121'


def read_json(path):
    return json.loads(Path(path).read_bytes())


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def download(url, target, size, digest):
    partial = target.with_suffix(target.suffix + '.part')
    if not target.exists():
        for attempt in range(4):
            try:
                uri = f'{url}?download=true&baseline_fetch={time.time_ns()}'
                with urlopen(uri, timeout=90) as response, partial.open('wb') as stream:
                    shutil.copyfileobj(response, stream, length=4 << 20)
                if partial.stat().st_size != size or sha256(partial) != digest:
                    raise ValueError(f'incomplete or mismatched download for {target.name}')
                partial.replace(target)
                break
            except (OSError, ValueError) as error:
                print(f'{target.name}: retry {attempt + 1}: {error}', flush=True)
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
    if target.stat().st_size != size or sha256(target) != digest:
        raise ValueError(f'unexpected cached asset {target}')


def fetch(name):
    meta = read_json(PROTOCOL)['datasets'][name]
    source, version = meta['repository'], meta['revision']
    with urlopen(f'https://huggingface.co/api/datasets/{source}/tree/{version}', timeout=30) as handle:
        entries = {row['path']: row for row in json.load(handle)}
    directory = DATA / name
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    verified = []
    for path in meta['files']:
        entry = entries[path]
        nbytes, digest = entry['size'], entry['lfs']['oid']
        print('fetch', name, path, nbytes, flush=True)
        url = f'https://huggingface.co/datasets/{source}/resolve/{version}/{quote(path)}'
        download(url, directory / path, nbytes, digest)
        verified.append({'file': path, 'size': nbytes, 'sha256': digest})
        print('verified', name, path, digest, flush=True)
    write_json(directory / 'source.json', {'repository': source, 'revision': version, 'files': verified})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmarks', nargs='+', choices=read_json(PROTOCOL)['order'])
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=len(args.benchmarks)) as executor:
        for _ in executor.map(fetch, args.benchmarks):
            pass


if __name__ == '__main__':
    main()
