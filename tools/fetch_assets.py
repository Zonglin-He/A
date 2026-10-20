"""Validated, resumable ranged downloads. Private payloads stay outside Git.

Only public, caller-provided URLs are used. Every range must have the exact
Content-Range and byte length. HTML/quota responses are rejected before writes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def drive_url(file_id):
    class Form(HTMLParser):
        def __init__(self):
            super().__init__()
            self.url, self.fields = None, {}

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == 'form':
                self.url = a.get('action')
            if tag == 'input' and 'name' in a:
                self.fields[a['name']] = a.get('value', '')
    # This is the provider's standard large-file confirmation form.
    with urlopen('https://drive.google.com/uc?' + urlencode({'id': file_id}), timeout=45) as r:
        if 'text/html' not in r.headers.get('Content-Type', ''):
            return r.url
        body = r.read(1 << 20).decode()
    parser = Form()
    parser.feed(body)
    if not parser.url or parser.fields.get('id') != file_id:
        raise RuntimeError('Drive did not return its public large-file confirmation form')
    return parser.url + '?' + urlencode(parser.fields)


def get_range(url, start, end, expected_total=None):
    request = Request(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
    with urlopen(request, timeout=90) as r:
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', r.headers.get('Content-Range', ''))
        if r.status != 206 or not match:
            raise RuntimeError(f'Range refused: status={r.status}, type={r.headers.get("Content-Type")}')
        a, b, total = map(int, match.groups())
        if (a, b) != (start, end) or (expected_total is not None and total != expected_total):
            raise RuntimeError('Wrong Content-Range; refusing to append cached or mismatched bytes')
        if 'text/html' in r.headers.get('Content-Type', ''):
            raise RuntimeError('HTML payload; refusing to save as an asset')
        data = r.read(end-start+2)
        if len(data) < end-start+1:
            raise IOError('Truncated range response; no bytes from this block were committed')
        if len(data) > end-start+1:
            raise RuntimeError('Oversized range response')
    return data, total


def download(url, dest, expected_sha=None, workers=8, chunk=8 << 20):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.resume.part')
    state_path = dest.with_suffix(dest.suffix + '.download.json')
    prefix, total = get_range(url, 0, 63)
    if dest.exists():
        if dest.stat().st_size != total or (expected_sha and sha(dest) != expected_sha):
            raise RuntimeError(f'Existing destination is not the requested complete asset: {dest}')
        return str(dest)
    identity = dict(size=total, sha256=expected_sha, chunk=chunk)
    complete = {}
    if state_path.exists() and part.exists():
        saved = json.loads(state_path.read_text())
        if saved['identity'] != identity:
            raise RuntimeError('Resume identity changed')
        with part.open('rb') as handle:
            for k, h in saved['complete'].items():
                start = int(k)*chunk
                handle.seek(start)
                data = handle.read(min(chunk, total-start))
                if hashlib.sha256(data).hexdigest() == h:
                    complete[k] = h
    fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, total)
    lock = threading.Lock()
    tick = time.monotonic()

    def snapshot(status='downloading'):
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(status=status, identity=identity, complete=complete,
            bytes_complete=sum(min(chunk, total-int(k)*chunk) for k in complete)), indent=2)+'\n')
        tmp.replace(state_path)

    def job(k):
        nonlocal tick
        start, end = k*chunk, min(total, (k+1)*chunk)-1
        for attempt in range(3):
            try:
                data, _ = get_range(url, start, end, total)
                break
            except RuntimeError:
                raise  # never retry a wrong range or HTML response
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1+attempt)
        if start == 0 and data[:64] != prefix:
            raise RuntimeError('Asset changed after probe')
        written = 0
        while written < len(data):
            written += os.pwrite(fd, data[written:], start+written)
        with lock:
            complete[str(k)] = hashlib.sha256(data).hexdigest()
            snapshot()
            if time.monotonic()-tick > 15:
                print(dest.name, len(complete), '/', (total+chunk-1)//chunk, flush=True)
                tick = time.monotonic()

    try:
        snapshot()
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(job, k) for k in range((total+chunk-1)//chunk) if str(k) not in complete]
            for f in as_completed(futures):
                f.result()
        os.fsync(fd)
    finally:
        os.close(fd)
    digest = sha(part)
    if expected_sha and digest != expected_sha:
        raise RuntimeError(f'Complete asset hash mismatch: {dest}')
    part.replace(dest)
    snapshot('complete')
    print('COMPLETE', dest, total, digest, flush=True)
    return str(dest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url')
    p.add_argument('--drive-id')
    p.add_argument('--output', required=True)
    p.add_argument('--sha256')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if bool(args.url) == bool(args.drive_id):
        p.error('Supply exactly one URL or public Drive ID')
    url = drive_url(args.drive_id) if args.drive_id else args.url
    download(url, args.output, args.sha256, args.workers)


if __name__ == '__main__':
    main()
