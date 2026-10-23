"""Lossless numerical cache, independent of every currently bound worker.

No eviction, dtype reduction, pickle, model loading or GPU operations. Callers
must bind the exact input, representation, code and checkpoint identities.
"""
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


class CapacityError(RuntimeError):
    pass


class ArrayStore:
    def __init__(self, root, *, budget_bytes=64 << 30, reserve_bytes=25 << 30):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget = budget_bytes
        self.reserve = reserve_bytes

    @staticmethod
    def key(context):
        for name in ['backbone', 'checkpoint_sha256', 'implementation_sha256',
                     'input_sha256', 'sampling_sha256', 'representation', 'scope']:
            if not context.get(name):
                raise ValueError('Missing cache binding: ' + name)
        if context['scope'] == 'query':
            if not context.get('caption_sha256') or 'seed' not in context:
                raise ValueError('Query-conditioned cache requires caption and seed')
        elif context['scope'] == 'video':
            if not context.get('query_independence_proof_sha256'):
                raise ValueError('Video-level reuse requires a validated independence proof')
        else:
            raise ValueError('Explicit query/video scope required')
        return digest(context)

    def put(self, context, arrays, *, required=False):
        key = self.key(context)
        values = {name: np.asarray(value) for name, value in sorted(arrays.items())}
        if not values or any(not name.isidentifier() for name in values):
            raise ValueError('Named numerical arrays required')
        if any(a.dtype.hasobject or a.dtype.kind not in 'biufc' for a in values.values()):
            raise ValueError('Only numerical, non-object arrays are accepted')
        if any(not np.isfinite(a).all() for a in values.values()):
            raise ValueError('Nonfinite cached numerical evidence')
        descriptors = {name: dict(dtype=a.dtype.str, shape=list(a.shape),
                                  sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
                       for name, a in values.items()}
        with (self.root / '.writer.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            receipt = self.root / (key + '.json')
            payload = self.root / (key + '.npz')
            if receipt.exists():
                old = json.loads(receipt.read_text())
                if old['context'] != context or old['arrays'] != descriptors:
                    raise ValueError('Never replace a cache identity with different evidence')
                self.get(context)
                return dict(status='hit', key=key, bytes=payload.stat().st_size)
            if payload.exists():
                raise ValueError('Uncommitted payload retained; manual audit required')
            # Reserve an uncompressed upper bound plus header margin before encoding.
            upper = sum(a.nbytes for a in values.values()) + (1 << 20)
            used = sum(p.stat().st_size for p in self.root.glob('*.npz'))
            if used + upper > self.budget or shutil.disk_usage(self.root).free - upper < self.reserve:
                if required:
                    raise CapacityError('Cache budget/reserve reached; existing evidence retained')
                return dict(status='not_cached_capacity', key=key)
            fd, name = tempfile.mkstemp(prefix='.pending-', dir=self.root)
            tmp = Path(name)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    np.savez_compressed(stream, **values)
                    stream.flush()
                    os.fsync(stream.fileno())
                size = tmp.stat().st_size
                if used + size > self.budget or shutil.disk_usage(self.root).free < self.reserve:
                    if required:
                        raise CapacityError('Cache capacity changed during encoding')
                    return dict(status='not_cached_capacity', key=key)
                info = dict(schema='lossless_numeric_cache_v1', context=context,
                            arrays=descriptors, payload_sha256=sha(tmp), bytes=size)
                os.link(tmp, payload)
                # Atomic metadata commit; readers ignore payloads without receipts.
                meta = tmp.with_suffix('.json')
                try:
                    with meta.open('x') as stream:
                        json.dump(info, stream, sort_keys=True, allow_nan=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.link(meta, receipt)
                finally:
                    meta.unlink(missing_ok=True)
                return dict(status='stored', key=key, bytes=size)
            finally:
                tmp.unlink(missing_ok=True)

    def get(self, context):
        key = self.key(context)
        info = json.loads((self.root / (key + '.json')).read_text())
        payload = self.root / (key + '.npz')
        if info['context'] != context or sha(payload) != info['payload_sha256']:
            raise ValueError('Cache context or payload hash mismatch')
        # Parse exactly the bytes whose hash was checked; no unsafe pickle loading.
        raw = payload.read_bytes()
        if hashlib.sha256(raw).hexdigest() != info['payload_sha256']:
            raise ValueError('Cache changed during read')
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            if set(data.files) != set(info['arrays']):
                raise ValueError('Cache array set changed')
            out = {name: data[name].copy() for name in data.files}
        for name, a in out.items():
            expected = info['arrays'][name]
            if (a.dtype.str != expected['dtype'] or list(a.shape) != expected['shape'] or
                    hashlib.sha256(a.tobytes(order='C')).hexdigest() != expected['sha256']):
                raise ValueError('Cache numerical payload mismatch')
        return out
