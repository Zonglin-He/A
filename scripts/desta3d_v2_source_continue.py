"""Engineering-only continuation: preserve per-epoch adapter weights.

The original locked runner, optimizer, query order, RNG, losses and selection
remain unchanged. An epoch snapshot is model weights plus metadata, not a full
optimizer rewind point. The original LATEST remains the training resume state.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_weights(base, state, adapters, torch, cpu_copy):
    if state['cursor'] != 618:
        return
    directory = base / 'epoch_weights'
    directory.mkdir(exist_ok=True)
    path = directory / f"E{state['epoch']}.pt"
    from scripts.desta3d_v2_p0 import adapter_sha256
    hashes = {a: adapter_sha256(m) for a, m in adapters.items()}
    identity = {'epoch':state['epoch'], 'cursor':618, 'adapter_hashes':hashes,
                'base_lock_sha256':sha(base/'LOCK.json')}
    index = path.with_suffix('.json')
    if index.exists():
        saved = json.loads(index.read_text())
        assert all(saved[k] == v for k,v in identity.items())
        assert sha(path) == saved['weights_sha256']
        return
    assert not path.exists(), 'orphan snapshot requires explicit review'
    if shutil.disk_usage(base).free < 8*2**30 + 64*2**20:
        raise RuntimeError('epoch snapshot disk reserve')
    temporary = path.with_suffix('.tmp.pt')
    torch.save({**identity, 'adapters':{a:cpu_copy(m.state_dict()) for a,m in adapters.items()},
                'optimizer_included':False, 'purpose':'immutable epoch weights for decode diagnostics'}, temporary)
    os.replace(temporary, path)
    index.write_text(json.dumps({**identity, 'weights_sha256':sha(path),
                                 'bytes':path.stat().st_size},indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--allocation',required=True)
    parser.add_argument('--phase-seconds',type=int,default=3600)
    args=parser.parse_args()
    assert args.phase_seconds >= 120
    base=ROOT/'artifacts/desta3d_v2/source_fit'
    amendment=json.loads((base/'REVIEW_AMENDMENT_20260927.json').read_text())
    assert amendment['continuation_wrapper_sha256'] == sha(__file__)
    start=time.monotonic()
    from scripts import desta3d_v2_source_fit as original
    assert amendment['original_runner_sha256'] == sha(original.__file__)
    receipt=base.parent/'receipts'/f'source_wrapper_{args.allocation}.json'
    assert not receipt.exists()
    original_save=original.save_state
    def save_with_epoch_weights(state,adapters,optimizers):
        original_save(state,adapters,optimizers)
        snapshot_weights(base,state,adapters,original.torch,original.cpu_copy)
    original.save_state=save_with_epoch_weights
    error=None
    try:
        original.fit(args.allocation,args.phase_seconds)
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        raise
    finally:
        original.save_state=original_save
        elapsed=time.monotonic()-start
        inner=base.parent/'receipts'/f'source_{args.allocation}.json'
        inside=json.loads(inner.read_text())['seconds'] if inner.exists() else 0.0
        receipt.write_text(json.dumps({'stage':'v2_source_continuation_overhead',
            'allocation':args.allocation,'status':'completed' if error is None else 'failed',
            'seconds':max(0.0,elapsed-inside),'wrapper_wall_seconds':elapsed,
            'inner_receipt':str(inner),'inner_seconds':inside,'failure':error,
            'cumulative_cap_seconds':None,'scientific_changes':False},indent=2)+'\n')


if __name__=='__main__':
    main()
