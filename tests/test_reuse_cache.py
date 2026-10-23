import numpy as np
import pytest
from tools.reuse_cache.store import ArrayStore, CapacityError
from tools.reuse_cache.catalog import Catalog
from tools.reuse_cache.store import sha


def context():
    return dict(backbone='synthetic', checkpoint_sha256='model', implementation_sha256='code',
                input_sha256='rgb', sampling_sha256='grid', representation='pretext',
                scope='query', caption_sha256='caption', seed=41)


def test_lossless_roundtrip_and_conflicting_identity(tmp_path):
    store = ArrayStore(tmp_path, budget_bytes=8 << 20, reserve_bytes=0)
    a = np.random.default_rng(4).normal(size=(20, 7)).astype(np.float32)
    value = {'features': a, 'ids': np.arange(20, dtype=np.int64)}
    assert store.put(context(), value)['status'] == 'stored'
    old = {p.name: (p.stat().st_mtime_ns, sha(p)) for p in tmp_path.glob('*.npz')}
    assert store.put(context(), value)['status'] == 'hit'
    assert old == {p.name: (p.stat().st_mtime_ns, sha(p)) for p in tmp_path.glob('*.npz')}
    loaded = store.get(context())
    assert loaded['features'].dtype == a.dtype and np.array_equal(loaded['features'], a)
    with pytest.raises(ValueError, match='different evidence'):
        store.put(context(), {'features': a + 1})
    payload = next(tmp_path.glob('*.npz'))
    payload.write_bytes(payload.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        store.get(context())


def test_quota_does_not_evict_or_silently_quantize(tmp_path):
    store = ArrayStore(tmp_path, budget_bytes=1, reserve_bytes=0)
    assert store.put(context(), {'a': np.ones(5)})['status'] == 'not_cached_capacity'
    with pytest.raises(CapacityError):
        store.put(context(), {'a': np.ones(5)}, required=True)
    assert not list(tmp_path.glob('*.npz'))
    c = context(); c['scope'] = 'video'
    with pytest.raises(ValueError, match='independence proof'):
        ArrayStore.key(c)


def test_catalog_lazy_verification_detects_tampering(tmp_path):
    p = tmp_path / 'payload'; p.write_bytes(b'cached')
    r = tmp_path / 'receipt'; r.write_text('bound')
    entry = dict(path=str(p), receipt=str(r), sha256=sha(p), receipt_sha256=sha(r),
                 backbone='tube', kind='capture', dataset='hc', identity='query', bytes=6)
    cat = Catalog(tmp_path / 'index.sqlite')
    cat.add(entry); cat.add(entry)
    assert cat.find('tube', 'capture', 'hc', 'query') == [entry]
    assert cat.verified_path(entry) == p
    r.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        cat.verified_path(entry)
    cat.db.close()


def test_teacher_bridge_requires_exact_rgb_caption_and_fingerprint(tmp_path):
    import hashlib
    import json
    import torch
    frames = np.zeros((8, 2, 2, 3), dtype=np.uint8)
    ctx = dict(RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
               shape=list(frames.shape), caption='same caption', teacher_fingerprint='frozen')
    key = hashlib.sha256(json.dumps(ctx, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    payload = tmp_path / (key + '.pt'); receipt = payload.with_suffix('.json')
    torch.save(dict(RGB_sha256=ctx['RGB_sha256'], boxes=torch.ones(8, 4)), payload)
    receipt.write_text(json.dumps(dict(context=ctx, sha256=sha(payload))))
    cat = Catalog(tmp_path / 'index.sqlite')
    cat.add(dict(path=str(payload), receipt=str(receipt), receipt_sha256=sha(receipt),
                 sha256=sha(payload), bytes=payload.stat().st_size, backbone='llava_st',
                 kind='teacher', dataset='content_bound', identity=key))
    value, trace = cat.teacher('llava_st', frames, 'same caption', 'frozen')
    assert value['boxes'].shape == (8, 4) and trace['new_teacher_calls'] == 0
    with pytest.raises(KeyError):
        cat.teacher('llava_st', frames, 'different caption', 'frozen')
    with pytest.raises(KeyError):
        cat.teacher('llava_st', frames, 'same caption', 'different teacher')
    different = frames.copy(); different[0, 0, 0, 0] = 1
    with pytest.raises(KeyError):
        cat.teacher('llava_st', different, 'same caption', 'frozen')
    cat.db.close()
