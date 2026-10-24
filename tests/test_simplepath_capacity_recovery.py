import numpy as np
import pytest
from tools.reuse_cache.store import ArrayStore, sha
from tools.simplepath_tube.capacity_recovery import RetentionOnlyStore


def context():
    return dict(backbone='synthetic',checkpoint_sha256='frozen',implementation_sha256='code',
                input_sha256='rgb',sampling_sha256='grid',representation='pretext',scope='query',
                caption_sha256='text',seed=41)


def test_recovery_budget_does_not_drop_in_memory_or_append(tmp_path):
    store=RetentionOnlyStore(tmp_path,budget_bytes=1,reserve_bytes=0)
    values={'features':np.arange(60,dtype=np.float32).reshape(10,6)}
    old=values['features'].copy()
    receipt=store.put(context(),values,required=True)
    assert receipt['status']=='not_cached_capacity_recovery_retention_only'
    assert np.array_equal(values['features'],old)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError,match='Finite numerical'):
        store.put(context(),{'features':np.array([np.nan])},required=True)


def test_recovery_keeps_exact_hit_and_detects_identity_conflict(tmp_path):
    ordinary=ArrayStore(tmp_path,budget_bytes=8<<20,reserve_bytes=0)
    values={'features':np.arange(6,dtype=np.float32)}
    ordinary.put(context(),values,required=True)
    before={p.name:sha(p) for p in tmp_path.iterdir() if p.is_file()}
    store=RetentionOnlyStore(tmp_path,budget_bytes=1,reserve_bytes=0)
    assert store.put(context(),values,required=True)['status']=='hit'
    assert np.array_equal(store.get(context())['features'],values['features'])
    assert before=={p.name:sha(p) for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError,match='different evidence'):
        store.put(context(),{'features':values['features']+1})
