"""Execution services. These do not choose evidence, steps, or predictions."""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(os.environ.get('STVG_WORKSPACE_ROOT', Path(__file__).resolve().parents[1]))
BASE = ROOT / 'artifacts/stvg_main_qwen_v1'

def interpreter(stage):
    """Keep incompatible Teacher and Student libraries in separate processes."""
    environment = '.venv/stvg_qwen35_geometry_v1' if stage in ('temporal_teacher','spatial_teacher') else '.conda/tubedetr'
    executable = ROOT / environment / 'bin/python'
    if not executable.is_file():
        raise FileNotFoundError(executable)
    return str(executable)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def put(path, obj, *, mutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not mutable:
        raise FileExistsError(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)

class PinVerifier:
    """Hash unchanged inode/size/mtime once per worker; fully rehash at seal."""
    def __init__(self, pins):
        self.pins = dict(pins)
        self.checked = {}
    def verify(self, *, full=False):
        for rel, digest in self.pins.items():
            p = ROOT / rel
            s = p.stat()
            stamp = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if full or self.checked.get(rel) != stamp:
                if sha(p) != digest:
                    raise RuntimeError('frozen dependency changed: ' + rel)
                self.checked[rel] = stamp

class AtomicWriter:
    """Bounded writer; caller passes immutable CPU snapshots, never live tensors."""
    def __init__(self):
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.pending = []
    def save(self, path, payload):
        if len(self.pending) >= 2:
            self.pending.pop(0).result()
        def task():
            import torch
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                raise FileExistsError(p)
            tmp = p.with_name(p.name + '.tmp')
            torch.save(payload, tmp)
            os.replace(tmp, p)
            put(p.with_suffix('.receipt.json'), dict(status='sealed', sha256=sha(p), bytes=p.stat().st_size))
        f = self.pool.submit(task)
        self.pending.append(f)
        return f
    def close(self):
        try:
            for f in self.pending:
                f.result()
        finally:
            self.pool.shutdown(wait=True)

def resource_guard():
    info = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if value.strip().endswith('kB'):
            info[key] = int(value.split()[0]) * 1024
    disk = shutil.disk_usage(ROOT).free
    if disk < 30 * 2**30 or info['MemAvailable'] < 6 * 2**30:
        raise RuntimeError('resource floor reached before next query')
    return dict(disk_free=disk, host_available=info['MemAvailable'])

def status(stage, done, total, **extra):
    put(BASE / 'STATUS.json', dict(status='running', stage=stage, done=done,
        total=total, timestamp_epoch=time.time(), **extra), mutable=True)
