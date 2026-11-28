"""Replay and seal the previous external panel's two frozen Qwen reads."""
from collections import Counter
import os
from pathlib import Path
import sys
from time import perf_counter

import torch

from provenance import ROOT, RUN, digest, read, runtime, write
from source import BATCH, MODEL

OLD = ROOT/'runs/relative-critic-v2-20261128/target/mono'
OUT = RUN/'mono-inputs'


def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,Path)):
        return
    p = Path(os.fsdecode(args[0]))
    if p.name in ('val.parquet','train.parquet','test.parquet','label-join-private.json','data_object_label_2.zip',
                  'mono-paired-index-private.json'):
        raise PermissionError(f'No target annotations during Qwen replay: {p.name}')


def main():
    sys.addaudithook(guard)
    runtime()
    from run_full_qwen import batch_read,load_model

    source = OLD/'input-seal.json'
    pin = read(source)
    inputs = OLD/'inputs-private.json'
    checks = OLD/'sensor-hashes-private.json'
    assert not pin['labels_read'] and pin['inputs']==digest(inputs) and pin['sensor_hashes']==digest(checks)
    assert all(digest(Path(p))==sha for p,sha in read(checks).items())
    frames = read(inputs)
    assert len(frames)==pin['n']==148
    OUT.mkdir(parents=True,exist_ok=True)
    started = perf_counter()
    processor, model, limit = load_model(MODEL)
    replies = {}
    batch_log = Counter()
    with torch.inference_mode():
        for i in range(0,len(frames),BATCH):
            group = frames[i:i+BATCH]
            values = {k:batch_read(processor,model,limit,group,k,batch_log) for k in ('2d','3d')}
            for f,a,b in zip(group,values['2d'],values['3d'],strict=True):
                replies[f['uid']] = {'2d':a,'3d':b}
            print('Frozen Mono replies',len(replies),'/',len(frames),flush=True)
    file = OUT/'qwen-private.json'
    write(file,replies)
    write(OUT/'decode-seal.json',dict(inputs=digest(source),payload=digest(file),n=len(replies),labels_read=False,
          code=digest(__file__),seconds=perf_counter()-started,batches=dict(batch_log)))


if __name__=='__main__':
    main()
