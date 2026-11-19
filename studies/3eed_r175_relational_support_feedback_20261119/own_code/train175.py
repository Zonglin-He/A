"""Equal-budget source training with a recording-held checkpoint selection."""
import argparse
from collections import defaultdict
import gc
import os
import time

import numpy as np
import torch

from scope175 import (CELLS,HERE,NEW_CACHE,OLD_CACHE,OLD_PRIVATE,PLATFORMS,PRIVATE,SPEC,
                      check_locked,json_atomic,read,sha,state)
from features175 import SHARD,STATIC,SUPERVISION
from model175 import RelationalDecoder,source_loss
from decoder import gather_slots,in_boxes

TRAIN_PIN=('model175.py','train175.py','features175.py')


def load_split(source,split,device='cuda'):
    parent=OLD_PRIVATE/source
    folder=PRIVATE/source
    old=read(parent/'FEATURE_SEAL.json')
    seal=read(folder/'FEATURE_SEAL.json')
    assert seal['source']==source and seal['entry']==sha(HERE/'features175.py')
    assert seal['roster']==sha(folder/'ROSTER_SEAL.json')
    numbers=seal['sizes'][split]
    n=sum(numbers.values())
    data={}

    def write(fields,lower,upper):
        for k,v in fields.items():
            x=torch.from_numpy(v)
            if k not in data:
                data[k]=torch.empty((n,*x.shape[1:]),dtype=x.dtype,device=device)
            data[k][lower:upper].copy_(x)

    for a in range(0,numbers['natural'],SHARD):
        nm=f'{split}_{a//SHARD:04d}.npz'
        original=OLD_CACHE/source/nm
        label=folder/f'natural_{nm}'
        assert sha(original)==old['files'][nm] and sha(label)==seal['files'][label.name]
        b=min(a+SHARD,numbers['natural'])
        with np.load(original,allow_pickle=False) as first,np.load(label,allow_pickle=False) as next_:
            fields={k:first[k][:b-a] for k in STATIC}
            fields.update({k:next_[k] for k in SUPERVISION})
            replace=next_['replace_role_mask']
            fields['role_mask'][replace]=next_['new_roles'][replace]
            write(fields,a,b)
    for i in range(0,numbers['synthetic'],SHARD):
        path=NEW_CACHE/source/f'synthetic_{split}_{i//SHARD:04d}.npz'
        assert sha(path)==seal['files'][path.name]
        a=numbers['natural']+i
        b=min(a+SHARD,n)
        with np.load(path,allow_pickle=False) as z:
            write({k:z[k] for k in (*STATIC,*SUPERVISION)},a,b)
    assert len(data['xyz'])==n
    assert (data['relation'][data['known_ref']]>0).all()
    assert data['has_reference'][data['known_ref']].all()
    return data


def select(data,inds):
    return {k:v[inds] for k,v in data.items()}


@torch.inference_mode()
def summary(model,data,batch,intervention=None):
    result=defaultdict(float)
    for start in range(0,len(data['xyz']),batch):
        d=select(data,slice(start,start+batch))
        out=model(d,intervention=intervention)
        q=d['quality']
        iou=q.gather(2,out['choices'][...,None]).squeeze(-1)
        natural=~d['synthetic']
        relation=d['known_ref']
        both=relation & (q.amax(-1)>.1).all(-1)
        masks=in_boxes(d['xyz'],d['gt'])
        mass=(out['support']*masks).sum(-1)
        g=q.argmax(-1)
        first=(gather_slots(out['passes'][0][2],g)*masks).sum(-1)
        last=(gather_slots(out['passes'][-1][2],g)*masks).sum(-1)
        for k,v in (
            ('n',len(iou)),('natural_n',natural.sum()),('synthetic_n',d['synthetic'].sum()),
            ('natural_iou',(iou[:,0]*natural).sum()),('natural_acc25',(natural&(iou[:,0]>.25)).sum()),
            ('natural_acc50',(natural&(iou[:,0]>.5)).sum()),('all_iou',iou[:,0].sum()),
            ('n_explicit_reference',relation.sum()),('n_both_covered',both.sum()),
            ('ref25',(relation&(iou[:,1]>.25)).sum()),('pair25',(relation&(iou>.25).all(-1)).sum()),
            ('both_golden_mass_s_first',(both*first[:,0]).sum()),
            ('both_golden_mass_r_first',(both*first[:,1]).sum()),
            ('both_golden_mass_s_last',(both*last[:,0]).sum()),
            ('both_golden_mass_r_last',(both*last[:,1]).sum()),
            ('natural_subject_support',(mass[:,0]*natural).sum()),
            ('mean_field_sum',out['field_strength'].mean(1).sum())):
            result[k]+=float(v)
    return dict(result)


def fit(source,cell,train,monitor,settings):
    folder=PRIVATE/source
    end=folder/f'{cell}.pth'
    ledger=folder/f'{cell}_TRAINING.json'
    pins={k:sha(HERE/k) for k in TRAIN_PIN}
    pins.update(roster=sha(folder/'ROSTER_SEAL.json'),features=sha(folder/'FEATURE_SEAL.json'),
                protocol=sha(SPEC),input=sha(PRIVATE/'INPUT_LOCK.json'))
    if ledger.exists():
        a=read(ledger)
        assert a['inputs']==pins and a['checkpoint']==sha(end)
        return a
    seed=settings['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng=torch.Generator(device='cuda').manual_seed(seed+1)
    model=RelationalDecoder(cell).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'],
                                fused=True)
    epochs,batch=settings['epochs'],settings['batch_size']
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.00006)
    selection, best, observations=None,None,[]
    n=len(train['xyz'])
    init=time.perf_counter()
    for e in range(1,epochs+1):
        model.train()
        order=torch.randperm(n,device='cuda',generator=rng)
        total=torch.zeros((),device='cuda')
        for a in range(0,n,batch):
            piece=select(train,order[a:a+batch])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=model(piece)
                loss=source_loss(model,piece,result)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
            optimizer.step()
            total+=loss.detach()*len(piece['xyz'])
        item=dict(epoch=e,loss=float(total/n),lr=sched.get_last_lr()[0])
        sched.step()
        if e in settings['checkpoint_epochs']:
            candidate=summary(model.eval(),monitor,batch)
            item['source_monitor']=candidate
            key=(candidate['natural_iou'],candidate['natural_acc50'],-e)
            if selection is None or key>selection:
                selection=key
                best=e
                dest=end.with_suffix('.tmp')
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},dest)
                os.replace(dest,end)
        observations.append(item)
        print('R175_TRAIN',source,cell,e,round(item['loss'],5),round(time.perf_counter()-init,1),flush=True)
    assert best in settings['checkpoint_epochs']
    chosen=torch.load(end,weights_only=True,map_location='cpu')
    model.load_state_dict(chosen,strict=True)
    interventions={}
    if cell=='full':
        for cond in ('off','shuffled'):
            interventions[cond]=summary(model.eval(),monitor,batch,cond)
    r=dict(source=source,cell=cell,inputs=pins,n_train=n,steps=epochs*((n+batch-1)//batch),
           epochs=observations,selected_epoch=best,checkpoint=sha(end),
           source_monitor=observations[best-1]['source_monitor'],
           source_interventions=interventions,params=sum(w.numel() for w in model.parameters()),
           source_only_selection=True,wall_seconds=time.perf_counter()-init)
    json_atomic(ledger,r)
    return r


def train_one(source):
    check_locked()
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    folder=PRIVATE/source
    settings=read(SPEC)['training']
    train,monitor=(load_split(source,s) for s in ('train','monitor'))
    for cell in CELLS:
        fit(source,cell,train,monitor,settings)
    files={cell:sha(folder/f'{cell}_TRAINING.json') for cell in CELLS}
    json_atomic(folder/'MODEL_SEAL.json',dict(source=source,files=files,training=sha(SPEC),
                    source_only_selection=True,target_labels_read=False,FINAL_access=False))
    del train,monitor
    gc.collect()
    torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('sources',nargs='+',choices=PLATFORMS)
    options=p.parse_args()
    state('source_training','Train full and matched controls on the same platform-specific data.')
    for source in options.sources:
        train_one(source)
