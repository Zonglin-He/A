"""Rebuild the paired proposal banks from sealed Waymo replies and LiDAR."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np
import torch

from pool_ablation import HERE,POOLS,generate_all,tensors
from provenance import ROOT,V2_SOURCE,digest,read,runtime,write
from source import WAYMO
import source_features as old

RUN = ROOT/'runs/s5-simplification-20261128'
BANKS = RUN/'waymo'
FILES = ('PROTOCOL_ZH.md','pool_ablation.py','source_banks.py')
FIELDS = ('nodes','edges','valid','labels')


def stamp(names=FILES):
    return {name:digest(HERE/name) for name in names}


def source_guard():
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,Path)):
            return
        p = Path(os.path.abspath(os.fsdecode(args[0])))
        roots = (V2_SOURCE,BANKS,ROOT/'runs/minimal-verifier-20261128/source')
        if (p.is_relative_to(ROOT/'runs') and not any(p.is_relative_to(r) for r in roots)
                or p.is_relative_to(ROOT/'assets/data') and not p.is_relative_to(WAYMO)
                or p.is_relative_to(ROOT/'assets/archives') or p.is_relative_to(ROOT/'assets/wildrefer-source')):
            raise PermissionError(f'Waymo-only stage: {p}')
    sys.addaudithook(audit)


def compute(frame,replies,box,shift,mode):
    datasets = {name:record for pools,case,_ in generate_all(frame,replies,shift)
                for name,record in tensors(pools,case,box).items()}
    return {name:datasets[name] for name in POOLS},mode


def job(value):
    row,replies = value
    frame = row['frame']
    assert frame['target']=='waymo'
    points = np.load(frame['points'],allow_pickle=False)
    points = points[np.isfinite(points[:,:3]).all(1)]
    band = old.target_band(frame,replies,points)
    rng = np.random.default_rng(int(old.order(row['uid']+'|degrade')[:16],16))
    result = [compute(frame,replies,row['box'],0.,0)]
    with TemporaryDirectory(prefix='view-',dir=BANKS/'scratch') as directory:
        for index in range(4):
            mode = rng.choice(old.VIEW_NAMES[1:],p=old.PROBABILITIES)
            cloud,shift = old.perturbed(points,band,rng,mode)
            path = str(Path(directory)/f'{index}.npy')
            np.save(path,cloud,allow_pickle=False)
            result.append(compute(dict(frame,points=path),old.perturbed_reply(replies,rng),
                                  row['box'],shift,old.VIEW_NAMES.index(mode)))
    return result


def materialize(part,workers,bank,decoded,stage):
    roster = V2_SOURCE/f'{part}-rows-private.json'
    assert digest(roster)==read(V2_SOURCE/'roster-seal.json')['population'][part]['bank']
    rows = read(roster)
    files,seen = {},{}
    for start in range(0,len(rows),96):
        block = list(old.verified_batches(part,rows[start:start+96],start,seen))
        assert all(decoded['checksums'][k]==v for k,v in seen.items())
        name = f'{part}-{start:04d}.npz'
        file,receipt = BANKS/name,BANKS/f'{name}.json'
        if receipt.is_file() and file.is_file():
            previous = read(receipt)
            assert previous['stage']==stage and previous['payload']==digest(file)
        else:
            records = [r for group in workers.map(job,block,chunksize=2) for r in group]
            arrays = {f'{key}_{field}':np.stack([r[0][key][j] for r in records])
                      for key in POOLS for j,field in enumerate(FIELDS)}
            arrays['modes'] = np.asarray([r[1] for r in records])
            original = V2_SOURCE/'feature-parts'/name
            assert bank['banks'][part]['files'][name]==digest(original)
            with np.load(original,allow_pickle=False) as old_pack:
                np.testing.assert_array_equal(arrays['modes'],old_pack['modes'])
                arrays['recording'] = old_pack['recording'].copy()
                arrays['uid'] = old_pack['uid'].copy()
            tmp = file.with_suffix('.tmp')
            with tmp.open('wb') as fp:
                np.savez(fp,**arrays)
            tmp.replace(file)
            write(receipt,dict(stage=stage,payload=digest(file)))
        files[name] = digest(file)
        print('Waymo',part,min(start+96,len(rows)),'/',len(rows),flush=True)
    return files


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--workers',type=int,default=12)
    args = cli.parse_args()
    source_guard()
    runtime()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    (BANKS/'scratch').mkdir(parents=True,exist_ok=True)
    started = perf_counter()
    bank_file,decoded_file = (V2_SOURCE/name for name in ('features-seal.json','qwen-seal.json'))
    bank,decoded = read(bank_file),read(decoded_file)
    assert bank['source_only'] and decoded['source_only'] and bank['decoded']==digest(decoded_file)
    stage = dict(code=stamp(),bank=digest(bank_file),decoded=digest(decoded_file))
    with ProcessPoolExecutor(args.workers) as workers:
        files = {part:materialize(part,workers,bank,decoded,stage) for part in ('val','train')}
    write(BANKS/'features-seal.json',dict(stage=stage,files=files,source_only=True,seconds=perf_counter()-started))


if __name__=='__main__':
    main()
