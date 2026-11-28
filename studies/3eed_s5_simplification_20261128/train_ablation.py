"""Train only the Waymo verifier using matched frozen front-end tensors."""
import argparse
from dataclasses import asdict
from time import perf_counter

import numpy as np
from safetensors.torch import load_file,save_file
import torch

from source_banks import BANKS,FIELDS,V2_SOURCE,digest,read,source_guard,stamp,write
from provenance import SOURCE
from fit import (BATCH,DEVICE,EPOCHS,SEED,calibrate,load_bank,read_masks,source_panel,values)
from models import Arm,project_tensors,raw_choice
from rankers import CONTROL,DESIGNS,ORIGINAL,PROPOSAL,equivalence,model_for,risk_loss,transform

FILES = ('rankers.py','train_ablation.py')


@torch.inference_mode()
def margins(model,data):
    nodes,edges,valid,_ = data
    scores = []
    for start in range(0,len(nodes),BATCH):
        sl = slice(start,start+BATCH)
        logits = model(nodes[sl],edges[sl],valid[sl])
        scores.append(raw_choice(logits-logits[:,:1],valid[sl],True).cpu().numpy())
    return np.concatenate(scores)


def load_new(part,pool,seal):
    rows = {k:[] for k in FIELDS+('modes','recording','uid')}
    for name,checksum in seal['files'][part].items():
        file = BANKS/name
        assert digest(file)==checksum
        with np.load(file,allow_pickle=False) as source:
            for f in rows:
                rows[f].append(source[f'{pool}_{f}' if f in FIELDS else f])
    arrays = {k:np.concatenate(v) for k,v in rows.items()}
    data = tuple(torch.as_tensor(arrays.pop(k),device=DEVICE) for k in FIELDS)
    return data,arrays


def load_original(part):
    bank = read(V2_SOURCE/'features-seal.json')
    membership = read(SOURCE/'membership-seal.json')
    data,meta = load_bank(part,bank)
    masks = read_masks(part,bank,membership)
    return project_tensors(data,masks,Arm(13,utility=True,lean=True))[0],meta


def panels(dataset,selected,use,mask):
    chosen = selected
    truth = dataset[-1].cpu().numpy()
    metric = (chosen,use,truth.max(1))
    return dict(clean=source_panel(metric,truth[:,0],~mask),degraded=source_panel(metric,truth[:,0],mask),
                avg_candidates=float(dataset[-2].sum(1).float().mean()))


def fit(name,design,training,validation,degraded):
    training,validation = (transform(data,design) for data in (training,validation))
    torch.manual_seed(SEED)
    model = model_for(design).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),lr=3.e-4,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=3.e-5)
    truth = validation[-1].cpu().numpy()
    winner = None
    history = []
    start = perf_counter()
    for epoch in range(1,EPOCHS+1):
        model.train()
        losses = []
        for block in torch.randperm(len(training[0]),device=DEVICE).split(BATCH):
            nodes,edges,valid,labels = (v[block] for v in training)
            optimizer.zero_grad(set_to_none=True)
            loss = risk_loss(model(nodes,edges,valid),labels,valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(loss.detach())
        scheduler.step()
        model.eval()
        outputs = margins(model,validation)
        threshold,gain = calibrate(outputs,truth,degraded,True)
        selected,use = values(outputs,truth,threshold,True)
        key = gain,-int((use&degraded).sum())
        history.append(dict(epoch=epoch,threshold=threshold,gain_pp=100*gain,loss=float(torch.stack(losses).mean())))
        if winner is None or key>winner[0]:
            winner = key,epoch,threshold,selected,use,{k:v.cpu().clone() for k,v in model.state_dict().items()}
    _,epoch,threshold,selected,use,weights = winner
    file = BANKS/f'{name}.safetensors'
    save_file(weights,file)
    return dict(design=asdict(design),params=sum(w.numel() for w in weights.values()),
                epoch=epoch,threshold=threshold,seconds=perf_counter()-start,checkpoint=digest(file),history=history,
                **panels(validation,selected,use,degraded))


def frozen_card(validation,degraded):
    original = read(SOURCE/'passed-stack-seal.json')['card']
    file = SOURCE/f'{CONTROL}.safetensors'
    assert digest(file)==original['checkpoint']
    model = model_for(DESIGNS[CONTROL]).to(DEVICE)
    model.load_state_dict(load_file(file))
    model.eval()
    raw = margins(model,validation)
    out = values(raw,validation[-1].cpu().numpy(),original['threshold'],True)
    metrics = panels(validation,*out,degraded)
    for scope in ('degraded','clean'):
        assert all(abs(metrics[scope][k]-original[scope][k])<1.e-6 for k in ('miou','acc50','harm')),metrics
    return dict(design=asdict(DESIGNS[CONTROL]),threshold=original['threshold'],epoch=original['epoch'],
                params=original['params'],checkpoint=original['checkpoint'],**metrics)


def compare_geometry(original,rebuilt):
    # Waymo source membership and score parity before measuring any new pool.
    x,e,v,y = original
    a,b,m,t = rebuilt
    assert torch.equal(v,m)
    for name,old,new in (('nodes',x,a),('edges',e,b),('labels',y,t)):
        torch.testing.assert_close(old,new,rtol=4.e-5,atol=4.e-5,msg=name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('original','proposal'))
    phase = parser.parse_args().phase
    source_guard()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    BANKS.mkdir(parents=True,exist_ok=True)
    bank_stamp = stamp(FILES)
    report = {}
    math = equivalence()
    old_cards = (load_original('train'),load_original('val'))
    original = [data for data,_ in old_cards]
    modes = old_cards[1][1]['modes']!=0
    reference = frozen_card(original[1],modes)
    plans = ORIGINAL if phase=='original' else PROPOSAL
    dependencies = dict(original=digest(V2_SOURCE/'features-seal.json'),checkpoint=reference['checkpoint'],
                        membership=digest(SOURCE/'membership-seal.json'))
    datasets = {'base':original}
    if phase=='proposal':
        bank_file = BANKS/'features-seal.json'
        seal = read(bank_file)
        assert seal['stage']['code']==stamp() and seal['source_only']
        dependencies['generated'] = digest(bank_file)
        for i,part in enumerate(('train','val')):
            rebuilt,meta = load_new(part,'base',seal)
            assert all(np.array_equal(meta[k],old_cards[i][1][k]) for k in ('modes','uid','recording'))
            compare_geometry(original[i],rebuilt)
        for pool in sorted({d.pool for d in plans.values()}):
            datasets[pool] = [load_new(part,pool,seal)[0] for part in ('train','val')]
    for name,design in plans.items():
        card = fit(name,design,*datasets[design.pool],modes)
        report[name] = card
        print(name,card['params'],card['epoch'],card['threshold'],card['degraded'],flush=True)
    write(BANKS/f'{phase}-training.json',
          dict(source_only=True,reference=reference,cards=report,code=bank_stamp,inputs=dependencies,
               softmax_equivalence=math))


if __name__=='__main__':
    main()
