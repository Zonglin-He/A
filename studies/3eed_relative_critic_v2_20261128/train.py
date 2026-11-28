"""Train and gate the candidate verifier using only real Waymo train/val tensors."""
from itertools import product
from time import perf_counter

import numpy as np
from safetensors.torch import save_file
import torch

from compact_pool import HERE, digest, read, write
from source import SOURCE, gate, runtime, source_guard
from verifier import RelativeVerifier, objective, top_nonanchor

SEED = 20261128
EPOCHS = 12
BATCH = 96
DEVICE = 'cuda'
SCORE_GRID = (0.,.05,.125,.25,.5,1.,2.)
PROB_GRID = (.50,.55,.60,.65,.70,.75,.80,.85,.90,.95)
GAIN_GRID = (0.,.01,.02,.03,.04,.05,.075,.10)


def load_bank(part, seal):
    members = seal['banks'][part]
    roster_path = SOURCE/f'{part}-rows-private.json'
    assert digest(roster_path) == read(SOURCE/'roster-seal.json')['population'][part]['bank']
    roster = read(roster_path)
    expected_ids = np.repeat([x['uid'] for x in roster],5)
    expected_records = np.repeat([x['recording'] for x in roster],5)
    text_types = {'uid':expected_ids.dtype,'recording':expected_records.dtype,
                  'branches':np.asarray(['C_ray','H','A','missing_3d']).dtype}
    arrays = {}
    files = sorted(members['files'])
    for name in files:
        path = SOURCE/'feature-parts'/name
        assert members['files'][name] == digest(path)
        with np.load(path,allow_pickle=False) as contents:
            if not arrays:
                arrays = {key:np.empty((members['views'],*contents[key].shape[1:]),
                                       text_types.get(key,contents[key].dtype)) for key in contents.files}
                cursor = 0
            end = cursor+len(contents['valid'])
            for field in contents.files:
                arrays[field][cursor:end] = contents[field]
            cursor = end
    assert cursor == members['views']
    attrs = {k:arrays.pop(k) for k in ('uid','recording','modes','branches')}
    assert np.array_equal(attrs['uid'],expected_ids)
    assert np.array_equal(attrs['recording'],expected_records)
    assert np.array_equal(attrs['modes'] == 0,np.arange(cursor)%5 == 0)
    assert arrays['valid'][:,0].all()
    data = tuple(torch.as_tensor(arrays[k],device=DEVICE) for k in ('nodes','edges','valid','labels'))
    return data,attrs


@torch.inference_mode()
def predict(model, data):
    nodes,edges,valid,labels = data
    all_rows = []
    for i in range(0,len(nodes),BATCH):
        sl = slice(i,i+BATCH)
        available = valid[sl]
        outputs = model(nodes[sl],edges[sl],available)
        selected,s,p,d = top_nonanchor(*outputs,available)
        all_rows.append(torch.stack((selected,s,p,d),1).cpu().numpy())
    return np.concatenate(all_rows)


def choose_threshold(raw, truth, eligible):
    selected,s,p,d = raw.T
    rows = np.arange(len(raw))
    before, after = truth[:,0],truth[rows,selected.astype(int)]
    change = after-before
    harm = (before>.50) & (after<=.50)
    assert np.isfinite(change).all()
    best = 0.,0
    thresholds = None
    for score,prob,gain in product(SCORE_GRID,PROB_GRID,GAIN_GRID):
        use = eligible & (s>score) & (p>prob) & (d>gain)
        if np.count_nonzero(use&harm) > .01*eligible.sum():
            continue
        criterion = (float(change[use].sum()),-int(use.sum()))
        if criterion > best:
            thresholds,best = (score,prob,gain),criterion
    use = decision(raw,thresholds)
    return thresholds,float(change[use&eligible].sum()/eligible.sum())


def decision(raw, thresholds):
    if thresholds is None:
        return np.zeros(len(raw),bool)
    _,s,p,d = raw.T
    score,prob,gain = thresholds
    return (s>score) & (p>prob) & (d>gain)


def outcomes(raw, tensor, thresholds):
    nodes,_,valid,labels = tensor
    truth = labels.cpu().numpy().astype(np.float64)
    visual = nodes[:,:,20].masked_fill(~valid,-torch.inf).argmax(1).cpu().numpy()
    idx = np.arange(len(truth))
    candidate = raw[:,0].astype(int)
    use = decision(raw,thresholds)
    values = dict(minimal=truth[:,0],aabb=truth[idx,visual],oracle=truth.max(1),
                  rank=truth[idx,np.where(raw[:,1]>0.,candidate,0)],
                  verifier=truth[idx,np.where(use,candidate,0)])
    return values,use


def report(values, use, meta, subset, loss):
    from evaluate_structure import stat_panel

    rows = [dict(target='waymo',recording=meta['recording'][i],
                 iou={k:v[i] for k,v in values.items()}) for i in np.flatnonzero(subset)]
    policy = stat_panel(rows,tuple(values),'minimal')
    aabb = stat_panel(rows,('aabb','verifier'),'aabb')['metrics']['verifier']
    before = values['minimal'][subset]
    after = values['verifier'][subset]
    h = np.count_nonzero((before>.50)&(after<=.50))
    metrics = policy['metrics']
    changes = (after-before)[use[subset]]
    diagnostic = dict(improved=int((changes>1.e-8).sum()),worsened=int((changes< -1.e-8).sum()),
                      tied=int((abs(changes)<=1.e-8).sum()),
                      oracle_headroom=int((values['oracle'][subset]>before+.03).sum()))
    return dict(**policy, against_aabb=aabb, replaced=int((use&subset).sum()), loss_rate=h/len(rows),
                baseline_success_loss_rate=h/max(1,np.count_nonzero(before>.50)),
                mean_loss=loss, replacements=diagnostic,
                route_counts={str(k):int((meta['branches'][subset]==k).sum())
                              for k in np.unique(meta['branches'][subset])})


def main():
    source_guard()
    gate()
    runtime()
    from shared import setup
    setup()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision('high')
    bank = read(SOURCE/'features-seal.json')
    assert bank['source_only'] and bank['decoded'] == digest(SOURCE/'qwen-seal.json')
    assert all(digest(HERE/name)==pin for name,pin in bank['code'].items())
    train, training = load_bank('train',bank)
    val, validation = load_bank('val',bank)
    assert not set(training['recording']) & set(validation['recording'])
    assert not set(training['uid']) & set(validation['uid'])
    model = RelativeVerifier().to(DEVICE)
    size = sum(p.numel() for p in model.parameters())
    assert 350000 <= size <= 550000
    opt = torch.optim.AdamW(model.parameters(),lr=3.e-4,weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=3.e-5)
    started = perf_counter()
    best = None
    history = []
    labels = val[-1].cpu().numpy()
    degraded = validation['modes'] != 0
    for epoch in range(1,EPOCHS+1):
        model.train()
        losses = []
        perm = torch.randperm(len(train[0]),device=DEVICE)
        for selected in perm.split(BATCH):
            x,e,m,y = (v[selected] for v in train)
            opt.zero_grad(set_to_none=True)
            loss = objective(*model(x,e,m),y,m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            losses.append(loss.detach())
        schedule.step()
        mean_loss = float(torch.stack(losses).mean().cpu())
        model.eval()
        outputs = predict(model,val)
        thresholds,gain = choose_threshold(outputs,labels,degraded)
        scores,use = outcomes(outputs,val,thresholds)
        result = dict(epoch=epoch,gain_pp=100*gain,thresholds=thresholds,degraded_changes=int((use&degraded).sum()),
                      loss=mean_loss)
        print('Verifier source validation',result,flush=True)
        history.append(result)
        key = gain,-int((use&degraded).sum())
        if best is None or key>best[0]:
            best = key,epoch,thresholds,outputs,mean_loss,{k:v.cpu().clone() for k,v in model.state_dict().items()}

    _,epoch,thresholds,outputs,loss,weights = best
    values,use = outcomes(outputs,val,thresholds)
    summary = {m:report(values,use,validation,subset,loss) for m,subset in
               (('degraded',degraded),('clean',~degraded),
                ('high',validation['modes']==1),('low',validation['modes']==2),('scan',validation['modes']==3))}
    main = summary['degraded']
    advanced = (main['loss_rate']<=.01 and main['metrics']['verifier']['miou_ci'][0]>0.
                and main['against_aabb']['miou_ci'][0]>0.)
    checkpoint = SOURCE/'verifier.safetensors'
    save_file(weights,checkpoint)
    ck = digest(checkpoint)
    file = SOURCE/'validation-scores-private.npz'
    np.savez_compressed(file,uid=validation['uid'],mode=validation['modes'],
                        predicted=outputs,replaced=use,**values)
    pins = {f:digest(HERE/f) for f in ('verifier.py','train.py','SOURCE_PROTOCOL_ZH.md')}
    write(SOURCE/'training-seal.json',dict(
        stage='go' if advanced else 'no_go', source_only=True, features=digest(SOURCE/'features-seal.json'),
        checkpoint=ck, parameters=size, epoch=epoch, thresholds=thresholds, history=history,
        validation=summary, validation_scores=digest(file), seconds=perf_counter()-started, code=pins))
    print('Stage 2', 'GO' if advanced else 'NO-GO',main,flush=True)


if __name__ == '__main__':
    main()
