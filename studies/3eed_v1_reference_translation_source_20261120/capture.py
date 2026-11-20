"""Replay source-own observations without labels and seal the unaligned and transported boxes."""
import fcntl
from functools import lru_cache
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'3eed_v1_reciprocal_physical_roles_20261120'
sys.path.insert(0,str(OLD))
import capture_roles as previous
from capture_roles import Grounder, PROXY, ROOT, SOURCES, archive_npz, digest, frame, read, save
from transport import ARMS,translate_readout

PRIVATE=HERE/'run_PRIVATE'
OUT=HERE/'results'


def observe(model,record,old_path):
    parts={}
    registered=previous.decide
    def tap(*args):
        if args[-1]=='v1':
            parts.update(dict(zip(('p','s','r','conditional','rs','rr','reverse'),args[:-1],strict=True)))
        return registered(*args)
    with patch.object(previous,'decide',tap):
        replay=previous.observe(model,record)
    with np.load(old_path,allow_pickle=False) as prior:
        assert replay['active'] and bool(prior['active'])
        for key in ('scores','boxes','raw','refs','slate','choices'):
            np.testing.assert_allclose(replay[key],prior[key],rtol=0,atol=1e-7)
        np.testing.assert_array_equal(replay['choices'],prior['choices'])
    args=tuple(parts[k] for k in ('p','s','r','conditional','rs','rr','reverse'))
    results=[]
    for arm in ARMS:
        if arm=='v1':
            pick=int(replay['choices'][0])
            result=pick,replay['boxes'][0],parts['p']
        else:
            result=translate_readout(*args,arm)
        results.append(result)
    np.testing.assert_allclose(results[1][1],replay['boxes'][-1],rtol=0,atol=1e-9)
    np.testing.assert_allclose(results[1][2],replay['scores'][-1],rtol=0,atol=1e-10)
    assert results[1][0]==replay['choices'][-1] and results[2][0]==results[-1][0]
    return dict(**parts,choices=np.asarray([v[0] for v in results]),boxes=np.stack([v[1] for v in results]),
                scores=np.stack([v[2] for v in results]))


def run():
    PRIVATE.mkdir(mode=0o700,exist_ok=True)
    OUT.mkdir(exist_ok=True)
    config=read(HERE/'PROTOCOL.json')
    assert tuple(config['controls'])==ARMS and digest(PROXY)==previous.CONFIG['panels']['source_proxy']['source_inventory_sha256']
    old=read(previous.PRIVATE/'SEAL.json')
    assert old['n']==501 and not old['target_updates'] and not old['FINAL_access']
    lock=dict(design=digest(HERE/'PROTOCOL.json'),code={f:digest(HERE/f) for f in ('capture.py','transport.py')},
              original_seal=digest(previous.PRIVATE/'SEAL.json'),roster=digest(PROXY))
    if (PRIVATE/'LOCK.json').exists():
        assert lock==read(PRIVATE/'LOCK.json')
    else:
        save(PRIVATE/'LOCK.json',lock)
    stamp=digest(PRIVATE/'LOCK.json')
    entries=read(PROXY)
    assert len(entries)==111
    files={}
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    start=perf_counter()
    with (ROOT/'research/GPU_RESOURCE.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for source in SOURCES:
            model=None
            for case,entry in enumerate(entries):
                if entry['row']['platform']!=source:
                    continue
                tag=f'source_proxy_{case:03d}_{source}.npz'
                path=PRIVATE/tag
                if not path.exists():
                    if model is None:
                        model=Grounder(source)
                        model._frontend.parser=lru_cache(maxsize=512)(model._frontend.parser)
                    assert entry['rewrite']['status']=='supported'
                    inp=frame(entry['row'])
                    assert inp['target']==source
                    record=dict(frame=inp,views={k:entry['rewrite'][k] for k in ('B','D')})
                    old_path=previous.PRIVATE/tag
                    assert digest(old_path)==old['files'][tag]
                    archive_npz(path,**observe(model,record,old_path),lock=stamp)
                with np.load(path,allow_pickle=False) as f:
                    assert str(f['lock'])==stamp
                files[tag]=digest(path)
            del model
            torch.cuda.empty_cache()
            print(source,len(files),'of 111',round(perf_counter()-start,1),'s',flush=True)
    assert len(files)==111
    save(PRIVATE/'SEAL.json',dict(lock=stamp,n=111,files=files,only_source_own=True,training_steps=0,
         FINAL_access=False,labels_used_for_outputs=False,seconds=perf_counter()-start))


if __name__=='__main__':
    run()
