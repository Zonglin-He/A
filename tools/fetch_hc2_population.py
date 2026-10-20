"""Acquire only the complete, user-confirmed 1902-query HC2 media population.

Public mirror ZIP members are range-checked, decompressed, length/CRC-checked,
and atomically committed. Resumption verifies existing files; no query filtering.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import struct
import sys
import threading
import time
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.remote_archive import SplitHTTPFile, HC_PARTS
from tools.build_manifests import HC_SHA, build
from stvg.runtime import sha
from stvg.tube_runner import write

REVISION = '4d26c230bbdba17e57a2e9965d15b584cb69a33e'
PARTS = [(url.replace('/resolve/main/', '/resolve/'+REVISION+'/'), size) for url, size in HC_PARTS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    ann = ROOT/'data/annotations/tastvg_release/data/hc-stvg2/annos/test.json'
    if sha(ann) != HC_SHA:
        raise ValueError('The confirmed annotation release changed')
    names = set(json.loads(ann.read_text()))
    with zipfile.ZipFile(SplitHTTPFile(PARTS)) as z:
        entries = {}
        for info in z.infolist():
            name = Path(info.filename).name
            if name in names:
                if name in entries:
                    raise ValueError('Ambiguous ZIP member')
                entries[name] = info
    if set(entries) != names or len(names) != 1902:
        raise ValueError('Mirror does not cover the complete confirmed population')
    dest = ROOT/'data/hc2/videos'
    dest.mkdir(parents=True, exist_ok=True)
    status = ROOT/'runtime/hc2_population_download.json'
    total_bytes = sum(i.file_size for i in entries.values())
    completed, byte_count = 0, 0
    lock = threading.Lock()

    def acquire(name):
        nonlocal completed, byte_count
        info, path = entries[name], dest/name
        if path.exists():
            crc = 0
            with path.open('rb') as f:
                for block in iter(lambda: f.read(8 << 20), b''):
                    crc = zlib.crc32(block, crc)
            if path.stat().st_size != info.file_size or crc != info.CRC:
                raise ValueError('Existing media differs from pinned ZIP member')
        else:
            for attempt in range(3):
                try:
                    remote = SplitHTTPFile(PARTS)
                    remote.seek(info.header_offset)
                    header = remote.read(30)
                    fields = struct.unpack('<4s5H3I2H', header)
                    if fields[0] != b'PK\x03\x04' or fields[2] & 1 or fields[3] not in (0, 8):
                        raise ValueError('Unsupported or encrypted ZIP member')
                    remote.seek(info.header_offset+30)
                    actual_name = remote.read(fields[-2]).decode('utf-8' if fields[2] & 0x800 else 'cp437')
                    if actual_name != info.filename or fields[3] != info.compress_type:
                        raise ValueError('ZIP member identity mismatch')
                    remote.seek(info.header_offset+30+fields[-2]+fields[-1])
                    decoder = zlib.decompressobj(-15) if info.compress_type == 8 else None
                    crc, size, remaining = 0, 0, info.compress_size
                    temp = path.with_suffix(path.suffix+'.part')
                    with temp.open('wb') as f:
                        while remaining:
                            block = remote.read(min(remaining, 8 << 20))
                            remaining -= len(block)
                            data = decoder.decompress(block) if decoder else block
                            f.write(data); crc = zlib.crc32(data, crc); size += len(data)
                        if decoder:
                            data = decoder.flush()
                            f.write(data); crc = zlib.crc32(data, crc); size += len(data)
                            if not decoder.eof or decoder.unused_data:
                                raise ValueError('Invalid compressed member boundary')
                    if size != info.file_size or crc != info.CRC:
                        raise ValueError('ZIP media length/CRC mismatch')
                    temp.replace(path)
                    break
                except (RuntimeError, ValueError):
                    raise  # Wrong ranges/HTML/integrity failures are never retried.
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt+1)
        with lock:
            completed += 1
            byte_count += info.file_size
            write(status, dict(status='downloading', annotation_sha256=HC_SHA, mirror_revision=REVISION,
                completed=completed, total=1902, verified_media_bytes=byte_count, total_media_bytes=total_bytes))
            if completed % 25 == 0:
                print('HC2 CRC verified', completed, '/ 1902', flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(acquire, name) for name in sorted(names)]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    summary = build('hc2', ann, dest, ROOT/'data/evaluation')
    write(status, dict(status='complete', completed=1902, total=1902, verified_media_bytes=byte_count,
        annotation_sha256=HC_SHA, mirror_revision=REVISION, intake=summary))
    print('COMPLETE all HC2 media and full manifest', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        status = ROOT/'runtime/hc2_population_download.json'
        value = json.loads(status.read_text()) if status.exists() else {}
        value.update(status='failed', error_type=type(error).__name__, error=str(error))
        write(status, value)
        raise
