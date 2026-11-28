"""Train and select the predeclared simplifications using Waymo alone."""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from safetensors.torch import load_file,save_file
import torch

from provenance import HERE, PLAN, SOURCE, V2_SOURCE, digest, read, runtime, source_guard, write
from models import ARMS,S0,Arm,model_for,project_tensors,raw_choice,risk_loss
from train import BATCH,DEVICE,EPOCHS,SEED,choose_threshold,decision,load_bank
from verifier import objective

TOLERANCE = 1.


@torch.inference_mode()
def predict(model,data,arm):
    scores = []
    for i in range(0,len(data[0]),BATCH):
        x,e,m = [a[i:i+BATCH] for a in data[:3]]
        scores.append(raw_choice(model(x,e,m),m,arm.utility).cpu().numpy())
    return np.concatenate(scores)


def switches(raw,threshold,single):
    if threshold is None:
        return np.zeros(len(raw),bool)
    return (raw[:,1]>threshold) if single else decision(raw,threshold)


def calibrate(raw,truth,eligible,single):
    if not single:
        return choose_threshold(raw,truth,eligible)
    candidates = raw[:,0].astype(int)
    before,after = truth[:,0],truth[np.arange(len(truth)),candidates]
    scores = raw[:,1]
    accepted = np.flatnonzero(eligible&np.isfinite(scores)&(scores>0.))
    ids = accepted[np.argsort(-scores[accepted],kind='stable')]
    total = eligible.sum()
    if not len(ids):
        return None,0.
    harms = ((before[ids]>.5)&(after[ids]<=.5)).cumsum()
    gain = (after[ids]-before[ids]).astype(float).cumsum()
    ends = np.r_[np.flatnonzero(np.diff(scores[ids])!=0),len(ids)-1]
    safe = ends[(harms[ends]<=.01*total)&(gain[ends]>0.)]
    if not len(safe):
        return None,0.
    end = safe[np.argmax(gain[safe])]
    threshold = float(scores[ids[end+1]]) if end+1<len(ids) else 0.
    return threshold,float(gain[end]/total)


def values(raw,labels,threshold,single):
    use = switches(raw,threshold,single)
    selected = np.where(use,raw[:,0].astype(int),0)
    return labels[np.arange(len(labels)),selected],use


def read_masks(part,bank,mask):
    arrays = []
    for name in sorted(bank['banks'][part]['files']):
        file = SOURCE/'membership'/Path(name).with_suffix('.npy')
        assert mask['files'][name]==digest(file)
        arrays.append(np.load(file,allow_pickle=False))
    return torch.as_tensor(np.concatenate(arrays),device=DEVICE)


def fit(name,arm,training,validation,masks,degraded):
    train = project_tensors(training,masks[0],arm)[0]
    val = project_tensors(validation,masks[1],arm)[0]
    torch.manual_seed(SEED)
    model = model_for(arm).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),lr=3.e-4,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=3.e-5)
    truth = val[-1].cpu().numpy()
    winner = None
    started = perf_counter()
    history = []
    for epoch in range(1,EPOCHS+1):
        model.train()
        losses = []
        for i in torch.randperm(len(train[0]),device=DEVICE).split(BATCH):
            x,e,m,y = (v[i] for v in train)
            optimizer.zero_grad(set_to_none=True)
            z = model(x,e,m)
            loss = risk_loss(z,y,m) if arm.utility else objective(*z,y,m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(loss.detach())
        scheduler.step()
        model.eval()
        outputs = predict(model,val,arm)
        threshold,gain = calibrate(outputs,truth,degraded,arm.utility)
        chosen,use = values(outputs,truth,threshold,arm.utility)
        key = gain,-int((use&degraded).sum())
        history.append(dict(epoch=epoch,gain_pp=100*gain,threshold=threshold,switched=-key[1],
                            loss=float(torch.stack(losses).mean())))
        if winner is None or key>winner[0]:
            winner = key,epoch,threshold,chosen,use,{k:v.cpu().clone() for k,v in model.state_dict().items()}
    _,epoch,threshold,chosen,use,weights = winner
    file = SOURCE/f'{name}.safetensors'
    save_file(weights,file)
    card = dict(arm=asdict(arm),checkpoint=digest(file),epoch=epoch,threshold=threshold,history=history,
                params=sum(p.numel() for p in model.parameters()),seconds=perf_counter()-started)
    return card,(chosen,use,truth.max(1))


def source_panel(arr,anchor,scope):
    chosen,use,oracle = arr
    a,b = anchor[scope]>.50,chosen[scope]>.50
    return dict(n=int(scope.sum()),miou=100*float(chosen[scope].mean(dtype=np.float64)),
                acc50=100*float(b.mean()), oracle_miou=100*float(oracle[scope].mean(dtype=np.float64)),
                replaced=int((use&scope).sum()),rescue=int((~a&b).sum()),harm=int((a&~b).sum()),
                harm_rate=float((a&~b).mean()))


def passes(card,baseline):
    a,b = card['degraded'],baseline['degraded']
    return all(a[m]>=b[m]-TOLERANCE for m in ('miou','acc50')) and a['harm_rate']<=.01


def register(name,card,record,arms,arr,anchor,degraded):
    card['degraded'] = source_panel(record,anchor,degraded)
    card['clean'] = source_panel(record,anchor,~degraded)
    arms[name],arr[name] = card,record
    print(name,card['params'],card['threshold'],card['degraded'],flush=True)


def s0_card(validation):
    stamp = read(V2_SOURCE/'training-seal.json')
    control = V2_SOURCE/'verifier.safetensors'
    file = V2_SOURCE/'validation-scores-private.npz'
    assert digest(control)==stamp['checkpoint'] and stamp['validation_scores']==digest(file)
    model = model_for(S0)
    model.load_state_dict(load_file(control))
    model.to(DEVICE).eval()
    raw = predict(model,validation,S0)
    labels = validation[-1].cpu().numpy()
    chosen,use = values(raw,labels,stamp['thresholds'],False)
    with np.load(file,allow_pickle=False) as old:
        np.testing.assert_array_equal(chosen,old['verifier'])
        np.testing.assert_array_equal(use,old['replaced'])
    return (dict(arm=asdict(S0),checkpoint=stamp['checkpoint'],epoch=stamp['epoch'],
                 threshold=stamp['thresholds'],params=stamp['parameters'],seconds=0),
            (chosen,use,labels.max(1)))


def main():
    source_guard()
    runtime()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    bank = read(V2_SOURCE/'features-seal.json')
    receipt = SOURCE/'membership-seal.json'
    mask = read(receipt)
    assert mask['features']==digest(V2_SOURCE/'features-seal.json') and mask['protocol']==digest(PLAN)
    assert mask['code']==digest(HERE/'provenance.py')
    training,meta_train = load_bank('train',bank)
    validation,meta_val = load_bank('val',bank)
    assert not set(meta_train['recording'])&set(meta_val['recording'])
    masks = tuple(read_masks(k,bank,mask) for k in ('train','val'))
    degraded = meta_val['modes']!=0
    anchor = validation[-1][:,0].cpu().numpy()
    arms,arr = {},{}
    card,record = s0_card(validation)
    register('S0',card,record,arms,arr,anchor,degraded)
    for name,arm in ARMS.items():
        card,record = fit(name,arm,training,validation,masks,degraded)
        register(name,card,record,arms,arr,anchor,degraded)
    passed = {n:passes(v,arms['S0']) for n,v in arms.items() if n!='S0'}
    selected = next((n for n in ('S5-full13','S5-full17') if passed[n]),None)
    if selected is None:
        cap = 13 if passed['S1-13'] else 17 if passed['S1'] else 21
        combined = Arm(cap,passed['S2'],passed['S3'],passed['S4'])
        selected = next((n for n,a in ARMS.items() if a==combined),None)
        if selected is None and combined!=S0:
            card,record = fit('S5-combined',combined,training,validation,masks,degraded)
            register('S5-combined',card,record,arms,arr,anchor,degraded)
            selected = 'S5-combined'
            passed[selected] = passes(card,arms['S0'])
        if not selected or not passed[selected]:
            options = [n for n in ('S1','S1-13','S2','S3','S4') if passed[n]]
            selected = max(options,key=lambda n:arms[n]['degraded']['miou']) if options else 'S0'
    receipt_data = dict(arms=arms,passed=passed,selected=selected,source_only=True,
                        protocol=digest(PLAN),membership=digest(receipt),
                        code={n:digest(HERE/n) for n in ('provenance.py','models.py','fit.py')})
    metrics_file = SOURCE/'validation-private.npz'
    np.savez_compressed(metrics_file,uid=meta_val['uid'],recording=meta_val['recording'],mode=meta_val['modes'],
                        anchor=anchor,**{f'{k}_{key}':value for k,t in arr.items()
                                          for key,value in zip(('iou','switched','oracle'),t,strict=True)})
    receipt_data['predicted']=digest(metrics_file)
    write(SOURCE/'selection-seal.json',receipt_data)
    print('Waymo choice',selected,passed,flush=True)


if __name__=='__main__':
    main()
