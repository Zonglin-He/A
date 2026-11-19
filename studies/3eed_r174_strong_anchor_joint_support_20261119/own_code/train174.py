"""Train the four anchored conditions with the same source data and optimizer budget."""
from collections import defaultdict
from pathlib import Path
import gc
import os
import time

import numpy as np
import torch

from scope174 import (CELLS, HERE, OLD_CACHE, OLD_PRIVATE, PLATFORMS, PRIVATE, SYNTH_CACHE,
                      check_locked, json_atomic, read, sha, state)
from model174 import StrongAnchorDecoder, objective

FEATURE = ('points', 'xyz', 'text', 'valid', 'role_mask', 'has_reference', 'left_right', 'uv', 'in_image', 'visible')
CANDIDATE = ('subject_boxes', 'reference_boxes', 'base_logit', 'ref_logit', 'gt', 'known_ref', 'quality')
TOKEN_DIM = {'text': 1, 'valid': 1, 'role_mask': 2}
TRAIN_PIN = ('model174.py', 'train174.py')


def load_split(source, split, device='cuda'):
    roster = read(OLD_PRIVATE/source/'FEATURE_SEAL.json')
    anchors = read(PRIVATE/source/'ANCHORS_SEAL.json')
    assert anchors['input'] == sha(PRIVATE/'INPUT_LOCK.json')
    n = roster['counts'][split]
    paths = sorted(p for p in roster['files'] if p.startswith(split+'_'))
    width = roster['widths'][split]
    for name in paths:
        if name in anchors['synthetic']:
            p = SYNTH_CACHE/source/name
            assert sha(p) == anchors['synthetic'][name]
            with np.load(p, allow_pickle=False) as z:
                width = max(width, *(len(z[k]) for k in z.files if k.endswith('/text')))
    data, offset = {}, 0
    for name in paths:
        old = OLD_CACHE/source/name
        new = PRIVATE/source/name
        assert sha(old) == roster['files'][name] and sha(new) == anchors['files'][name]
        with np.load(old, allow_pickle=False) as z:
            arrays = {k:z[k] for k in FEATURE}
        with np.load(new, allow_pickle=False) as z:
            arrays.update({k:z[k] for k in CANDIDATE})
        for k, axis in TOKEN_DIM.items():
            expand = [(0, 0)]*arrays[k].ndim
            expand[axis] = (0, width-arrays[k].shape[axis])
            arrays[k] = np.pad(arrays[k], expand)
        if name in anchors['synthetic']:
            with np.load(SYNTH_CACHE/source/name, allow_pickle=False) as z:
                for key in z.files:
                    raw, field = key.split('/')
                    row = arrays[field][int(raw):int(raw)+1]
                    value = z[key]
                    if field in TOKEN_DIM:
                        axis = TOKEN_DIM[field]
                        sub = [slice(None)]*row.ndim
                        sub[axis] = slice(0, value.shape[axis-1])
                        row.fill(0)
                        row[tuple(sub)] = value
                    else:
                        row[:] = value
        count = len(arrays['xyz'])
        for k, a in arrays.items():
            t = torch.from_numpy(a)
            if k not in data:
                data[k] = torch.empty((n,*t.shape[1:]), dtype=t.dtype, device=device)
            data[k][offset:offset+count].copy_(t)
        offset += count
    assert offset == n
    assert (data['known_ref'] <= data['has_reference']).all()
    return data


def slice_fields(data, index):
    return {k: v[index] for k,v in data.items()}


@torch.inference_mode()
def monitor(engine, dataset, batch_size):
    sums = defaultdict(float)
    for j in range(0,len(dataset['xyz']),batch_size):
        p = slice_fields(dataset, slice(j,j+batch_size))
        q = engine(p)['choices']
        iou = p['quality'].gather(-1,q[...,None]).squeeze(-1)
        refs = p['known_ref']
        for k,v in (('n',len(q)), ('ref_n',refs.sum()), ('anchor_iou_sum',p['quality'][:,0,0].sum()),
                    ('subject_iou_sum',iou[:,0].sum()), ('subject25',(iou[:,0]>.25).sum()),
                    ('subject50',(iou[:,0]>.50).sum()), ('ref25',(refs & (iou[:,1]>.25)).sum()),
                    ('pair25',(refs & (iou>.25).all(-1)).sum()), ('pair50',(refs & (iou>.50).all(-1)).sum()),
                    ('winner_changes',(q[:,0]!=0).sum())):
            sums[k] += float(v)
    return dict(sums)


def fit(source, cell, data, held, options):
    folder = PRIVATE/source
    ckpt = folder/f'{cell}.pth'
    ledger = folder/f'{cell}_TRAINING.json'
    inputs = {p:sha(HERE/p) for p in TRAIN_PIN}
    inputs.update(lock=sha(PRIVATE/'INPUT_LOCK.json'), features=sha(folder/'ANCHORS_SEAL.json'))
    if ledger.exists():
        memo = read(ledger)
        assert memo['inputs'] == inputs and sha(ckpt) == memo['checkpoint']
        return memo
    seed = options['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device='cuda').manual_seed(seed+1)
    model = StrongAnchorDecoder(cell).cuda()
    initial = [p.detach().clone() for p in model.parameters()]
    batch_size, epochs = options['batch_size'], options['epochs']
    lr = options['learning_rate']
    optim = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=options['weight_decay'],fused=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim,epochs,eta_min=lr*.1)
    n = len(data['xyz'])
    history = []
    start = time.perf_counter()
    for e in range(epochs):
        order = torch.randperm(n, generator=generator, device='cuda')
        loss_sum = torch.zeros((), device='cuda')
        for k in range(0,n,batch_size):
            part = slice_fields(data,order[k:k+batch_size])
            optim.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss = objective(part,model(part))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),options['gradient_clip'],error_if_nonfinite=True)
            optim.step()
            loss_sum += loss.detach()*len(part['xyz'])
        history.append(dict(epoch=e+1,train_loss=(loss_sum/n).item(),lr=sched.get_last_lr()[0]))
        sched.step()
        print('SOURCE_TRAIN',source,cell,e+1,epochs,round(history[-1]['train_loss'],6),
              round(time.perf_counter()-start,1),flush=True)
    stage = ckpt.with_suffix('.tmp')
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},stage)
    os.replace(stage,ckpt)
    shift = torch.stack([(a-b).norm().square() for a,b in zip(model.parameters(),initial,strict=True)]).sum().sqrt()
    record = dict(source=source,cell=cell,n_train=n,steps=epochs*((n+batch_size-1)//batch_size),
                  parameters=sum(v.numel() for v in model.parameters()),history=history,
                  parameter_delta_l2=shift.item(),inputs=inputs,checkpoint=sha(ckpt),
                  source_monitor=monitor(model.eval(),held,batch_size),
                  source_monitor_used_for_selection=False,
                  seconds=time.perf_counter()-start)
    json_atomic(ledger,record)
    return record


def train_source(source):
    contract = check_locked()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    training, held = (load_split(source,split) for split in ('train','monitor'))
    for arm in CELLS:
        fit(source,arm,training,held,contract['train'])
    expected = PRIVATE/source
    json_atomic(expected/'MODEL_SEAL.json',dict(source=source,lock=sha(PRIVATE/'INPUT_LOCK.json'),
                train_options=contract['train'],files={c:sha(expected/f'{c}_TRAINING.json') for c in CELLS}))
    del training, held
    gc.collect()
    torch.cuda.empty_cache()


if __name__=='__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('sources',nargs='+',choices=PLATFORMS)
    args = p.parse_args()
    state('source_training','Training four matched R158-anchored decoder conditions.')
    for source in args.sources:
        train_source(source)
