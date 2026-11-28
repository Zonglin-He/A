"""Recover audited family membership from the frozen Waymo candidate cache."""
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V2 = HERE.parent/'3eed_relative_critic_v2_20261128'
sys.path.insert(0,str(V2))
from compact_pool import CAPACITY, ROOT, digest, read, write
from source import SOURCE as V2_SOURCE, WAYMO, runtime
import source_features

RUN = ROOT/'runs/minimal-verifier-20261128'
SOURCE = RUN/'source'
PLAN = HERE/'PROTOCOL_ZH.md'


def source_guard():
    """Allow prior source caches and current source results only."""
    def audit(event,args):
        if event != 'open' or not isinstance(args[0],(str,bytes,Path)):
            return
        p = Path(os.path.abspath(os.fsdecode(args[0])))
        if (p.is_relative_to(ROOT/'runs') and not any(p.is_relative_to(d) for d in (SOURCE,V2_SOURCE))
                or p.is_relative_to(ROOT/'assets/data') and not p.is_relative_to(WAYMO)
                or p.is_relative_to(ROOT/'assets/archives')
                or p.is_relative_to(ROOT/'assets/wildrefer-source')):
            raise PermissionError(f'Waymo-only source process denied {p}')
    sys.addaudithook(audit)


def retain(pool):
    """Preserve a deduplicated box if any retained family generated it."""
    quantile = lambda tag: tag.startswith('quantile/')
    reference = lambda tag: tag.startswith('pose/') and not tag.endswith('_projected')
    keep = np.zeros((CAPACITY,2),bool)
    for k,entry in enumerate(pool):
        keep[k,0] = any(not quantile(t) for t in entry['sources'])
        keep[k,1] = any(not quantile(t) and not reference(t) for t in entry['sources'])
    assert keep[0].all() and keep[:,0].sum()<=17 and keep[:,1].sum()<=13
    return keep


def describe_tags(case,pool):
    return retain(pool), None, np.arange(CAPACITY)<len(pool)


def source_job(task):
    with patch.object(source_features,'describe',describe_tags), patch.object(source_features,'SOURCE',SOURCE):
        return source_features.job(task)


def main():
    source_guard()
    runtime()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    path = SOURCE/'membership'
    path.mkdir(parents=True,exist_ok=True)
    (SOURCE/'scratch').mkdir(exist_ok=True)
    begin = perf_counter()
    bank = read(V2_SOURCE/'features-seal.json')
    decoded = read(V2_SOURCE/'qwen-seal.json')
    roster = read(V2_SOURCE/'roster-seal.json')
    assert bank['decoded']==digest(V2_SOURCE/'qwen-seal.json') and bank['source_only'] and decoded['source_only']
    assert all(digest(V2/name)==checksum for name,checksum in bank['code'].items())
    files = {}
    seen = {}
    with ProcessPoolExecutor(12) as workers:
        for split in ('val','train'):
            roster_path = V2_SOURCE/f'{split}-rows-private.json'
            assert digest(roster_path)==roster['population'][split]['bank']
            rows = read(roster_path)
            for name,checksum in bank['banks'][split]['files'].items():
                start = int(Path(name).stem.split('-')[-1])
                block = rows[start:start+96]
                replies = list(source_features.verified_batches(split,block,start,seen))
                source = V2_SOURCE/'feature-parts'/name
                assert digest(source)==checksum
                tagged = [view for group in workers.map(source_job,replies,chunksize=3) for view in group]
                keep,_,valid,iou,branches,modes = zip(*tagged,strict=True)
                with np.load(source,allow_pickle=False) as control:
                    for key,array in (('valid',valid),('labels',iou),('branches',branches),('modes',modes),
                                      ('uid',np.repeat([r['uid'] for r in block],5))):
                        np.testing.assert_array_equal(control[key],array,err_msg=f'{name}/{key}')
                np.save(path/Path(name).with_suffix('.npy'),np.asarray(keep))
                file = path/Path(name).with_suffix('.npy')
                files[name] = digest(file)
                if start%576==0 or start+len(block)==len(rows):
                    print('Candidate provenance',split,start+len(block),'/',len(rows),flush=True)
    assert seen==decoded['checksums']
    write(SOURCE/'membership-seal.json',dict(files=files, features=digest(V2_SOURCE/'features-seal.json'),
        qwen=digest(V2_SOURCE/'qwen-seal.json'), code=digest(__file__), protocol=digest(PLAN),
        verified_views=sum(v['views'] for v in bank['banks'].values()),seconds=perf_counter()-begin))


if __name__=='__main__':
    main()
