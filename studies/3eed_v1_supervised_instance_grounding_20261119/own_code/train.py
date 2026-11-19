"""Fit and choose the three instance heads using source recordings only."""

import argparse
from collections import defaultdict
from contextlib import nullcontext
import math
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from features import FEATURES, feature_dependencies
from model import InstanceGroundingHead, NATIVE, objective, ownership, proxy_overlap
from scope import HERE, ROOT, PLATFORMS, file_sha, protocol, read, save

sys.path.insert(0, str(ROOT / 'research/round6/source_validation_scoring'))
from batch_yaw_iou import many_iou

MODELS = HERE / 'weights_PRIVATE'
SELECTION = ('v1', 'fixed_winner_geometry', 'full_pool_unary', 'full_pool_joint', 'guarded_joint')


def prepare_device():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')


def read_features(root, platform, subset):
    seal = read(root/'SEALED.json')
    lock = read(root/'LOCK.json')
    assert seal['lock_sha256'] == file_sha(root/'LOCK.json')
    assert not seal['FINAL_access'] and lock['source_only']
    assert lock['dependencies'] == feature_dependencies()
    assert file_sha(HERE/'inputs_PRIVATE/SEAL.json') == lock['input_seal_sha256']
    name = f'{platform}_{subset}'
    folder = root/name
    pin = read(folder/'SEAL.json')
    assert file_sha(folder/'SEAL.json') == seal['splits'][name]['seal_sha256']
    assert pin['lock_sha256'] == file_sha(root/'LOCK.json')
    assert not pin['FINAL_access']
    assert pin['input_sha256'] == file_sha(HERE/'inputs_PRIVATE'/f'{name}.json')
    cols = defaultdict(list)
    for fname, receipt in sorted(pin['files'].items()):
        path = folder/fname
        assert file_sha(path) == receipt['sha256']
        with np.load(path, allow_pickle=False) as data:
            assert all(len(data[key]) == receipt['n'] for key in data.files)
            for key in data.files:
                cols[key].append(data[key])
    values = {key: torch.as_tensor(np.concatenate(array), device='cuda') for key, array in cols.items()}
    assert len(values['points']) == pin['n'] == seal['splits'][name]['n']
    return values


def take(data, ids):
    return {k:v[ids] for k,v in data.items()}


def decisions(out, data):
    boxes = out['boxes']
    batch = torch.arange(len(boxes), device=boxes.device)
    original = data['v1_index'].long()
    u = out['subject'].argmax(-1)
    j = out['joint'].argmax(-1)
    fixed = torch.where(out['geometry_changed'][batch, original, None],
                        boxes[batch, original], data['v1_box'])
    gap = out['joint'][batch,j] - out['joint'][batch,original]
    quality = out['quality'].amax(-1)
    accept = ((gap >= .90) & (quality[batch,j] >= quality[batch,original]-.02)
              & (out['support_strength'][batch,j] >= .025))
    guard = torch.where(accept[...,None], boxes[batch,j], fixed)
    indices = dict(v1=original, fixed_winner_geometry=original, full_pool_unary=u,
                   full_pool_joint=j, guarded_joint=torch.where(accept,j,original))
    result = dict(v1=data['v1_box'], fixed_winner_geometry=fixed,
                  full_pool_unary=boxes[batch,u], full_pool_joint=boxes[batch,j], guarded_joint=guard)
    references, known = {}, {}
    for name in SELECTION:
        if name in ('v1','fixed_winner_geometry'):
            references[name] = data['v1_reference_box']
            known[name] = data['v1_has_reference']
        else:
            ref = out['pair'][batch,indices[name]].argmax(-1)
            references[name] = boxes[batch,ref.clamp_max(boxes.shape[1]-1)]
            known[name] = (ref < boxes.shape[1]) & (data['relation'] != 0)
    return result, references, known, indices


def metrics(ious):
    x = np.asarray(ious,dtype=np.float64)
    return dict(n=len(x), Acc25=float(np.mean(x>.25)*100),
                Acc50=float(np.mean(x>.50)*100), mIoU=float(x.mean()*100))


@torch.inference_mode()
def validate(model, data, batch_size, detailed=False):
    model.eval()
    values = {key: [] for key in SELECTION}
    identities = {key: defaultdict(list) for key in SELECTION}
    oracle = defaultdict(list)
    region_purity = []
    for start in range(0,len(data['points']),batch_size):
        x = take(data,slice(start,start+batch_size))
        out = model(x)
        pred, refs, known, ids = decisions(out, x)
        box_arrays = torch.stack([pred[key] for key in SELECTION],1).cpu().numpy()
        origin = x['origin'].cpu().numpy()
        ref_arrays = torch.stack([refs[key] for key in SELECTION],1).cpu().numpy()
        ref_known = torch.stack([known[key] for key in SELECTION],1).cpu().numpy()
        targets = x['gt'].cpu().numpy()
        if detailed:
            native = x['anchors'].cpu().numpy()
            proposals = out['proposal'].cpu().numpy()
            refined = out['refined'].cpu().numpy()
            available = out['boxes'].cpu().numpy()
            q = proxy_overlap(out['proposal'],x['gt'],x['gt_valid'])[:,:,0]
            best, idx = q.max(-1)
            row = torch.arange(len(q),device=q.device)
            owner, _, _ = ownership(x['points'],x['gt'],x['gt_valid'])
            member = owner[row[:,None],out['patch'][row,idx]] == 0
            m = out['instance_weight'][row,idx]
            g = out['geometric_weight'][row,idx]
            ok = (best>.25) & member.any(-1) & (~member).any(-1)
            a = (m*member).sum(-1)/m.sum(-1).clamp_min(1e-8)
            b = (g*member).sum(-1)/g.sum(-1).clamp_min(1e-8)
            region_purity.extend(torch.stack((a[ok],b[ok]),-1).cpu().tolist())
        for j in range(len(origin)):
            a = many_iou(box_arrays[j],targets[j,0])
            if int(origin[j])<2:
                for k, name in enumerate(SELECTION):
                    values[name].append(float(a[k]))
                if detailed:
                    na = many_iou(native[j],targets[j,0]).max()
                    raw = many_iou(proposals[j],targets[j,0]).max()
                    local = many_iou(refined[j],targets[j,0]).max()
                    delivered = many_iou(available[j],targets[j,0]).max()
                    for key, result in (('A256',na),('A256_plus_votes',raw),('union_with_refined',max(na,raw,local)),
                                        ('geometry_selected_pool',delivered)):
                        oracle[key].append(float(result))
            if int(origin[j]) and detailed:
                b = many_iou(ref_arrays[j],targets[j,1]) * ref_known[j]
                kind = 'natural_inventory_unique' if origin[j]==1 else 'synthetic'
                for k,name in enumerate(SELECTION):
                    identities[name][kind].append((a[k]>.25,b[k]>.25,ref_known[j,k]))
    summary = {key:metrics(value) for key,value in values.items()}
    if not detailed:
        return summary
    both = {}
    for arm, categories in identities.items():
        both[arm] = {}
        for key, elems in categories.items():
            sub, ref, known = np.asarray(elems,dtype=bool).T
            both[arm][key] = dict(n=int(len(sub)),subject=int(sub.sum()),reference=int(ref.sum()),
                                  dual=int((sub&ref).sum()),unknown=int((~known).sum()))
    purity = np.asarray(region_purity)
    support = dict(n=len(purity),
                   learned_membership=float(purity[:,0].mean()) if len(purity) else None,
                   geometry_only=float(purity[:,1].mean()) if len(purity) else None)
    return dict(natural=summary, both_roles=both, proposal_coverage={k:metrics(v) for k,v in oracle.items()},
                local_support_purity=support)


def utility(row):
    return row['Acc25'] + row['Acc50'] + row['mIoU']


def fit(smoke):
    prepare_device()
    cfg = protocol()
    source_root = HERE/'engineering_features_PRIVATE' if smoke else FEATURES
    target = HERE/'engineering_weights_PRIVATE' if smoke else MODELS
    target.mkdir(mode=0o700)
    lock = dict(source_only=True, FINAL_access=False, fit_files=feature_dependencies(),
                model_sha256=file_sha(HERE/'model.py'),train_sha256=file_sha(Path(__file__)),
                source_feature_seal=file_sha(source_root/'SEALED.json'))
    save(target/'LOCK.json',lock)
    started = perf_counter()
    artifacts = {}
    duration = 1 if smoke else cfg['optimization']['epochs']
    batch_size = 2 if smoke else cfg['optimization']['batch_size']
    for i, platform in enumerate(PLATFORMS):
        seed = cfg['seed'] + i
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model = InstanceGroundingHead().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['optimization']['learning_rate'],
                                       weight_decay=cfg['optimization']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,duration,eta_min=1e-4)
        fitting = read_features(source_root,platform,'fit')
        held = read_features(source_root,platform,'calibration')
        snapshot, choice = None, (-math.inf,0,'v1')
        history = []
        for epoch in range(1,duration+1):
            model.train()
            observed = defaultdict(float)
            order = torch.randperm(len(fitting['points']),generator=generator)
            num = 0
            for chunk in order.split(batch_size):
                data = take(fitting,chunk.to('cuda'))
                output = model(data)
                loss, parts = objective(output,data)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{platform} epoch {epoch} non-finite loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step()
                observed['loss'] += float(loss.detach()) * len(chunk)
                for key, value in parts.items():
                    observed[key] += float(value) * len(chunk)
                observed['preclip_gradient_norm'] += float(norm) * len(chunk)
                num += len(chunk)
                if smoke:
                    break
            scheduler.step()
            entry = dict(epoch=epoch, training_examples=num,
                         **{key: v/num for key,v in observed.items()})
            if epoch in cfg['optimization']['checkpoint_epochs'] or smoke:
                candidates = validate(model,held,batch_size)
                entry['source_holdout'] = candidates
                for name, row in candidates.items():
                    key = (utility(row), row['Acc50'], name=='v1')
                    previous = (choice[0],choice[1],choice[2]=='v1')
                    if key > previous:
                        snapshot = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                        choice = (key[0],key[1],name)
                        which_epoch = epoch
            history.append(entry)
            save(target/f'{platform}_history.json',history)
            print(platform,'epoch',epoch,'loss',round(entry['loss'],4),
                  'source pick',choice[2],'elapsed',round(perf_counter()-started,1),flush=True)
        assert snapshot is not None
        model.load_state_dict(snapshot,strict=True)
        detailed = validate(model,held,batch_size,detailed=True)
        path = target/f'{platform}.pth'
        draft = path.with_suffix('.tmp')
        torch.save(dict(state=snapshot, source=platform, epoch=which_epoch,
                        readout=choice[2], n_parameters=sum(v.numel() for v in snapshot.values()),
                        optimization_seed=seed),draft)
        os.replace(draft,path)
        details = dict(source=platform,epoch=which_epoch,readout=choice[2],weight_sha256=file_sha(path),
                       fit_count=len(fitting['points']),holdout_count=len(held['points']),
                       metrics=detailed)
        save(target/f'{platform}_source_validation.json',details)
        artifacts[platform]=dict(weight_sha256=file_sha(path),epoch=which_epoch,readout=choice[2],
                                 proof_sha256=file_sha(target/f'{platform}_source_validation.json'))
        del model,fitting,held,optimizer
        torch.cuda.empty_cache()
    save(target/'SEALED.json',dict(sources=artifacts,lock_sha256=file_sha(target/'LOCK.json'),
                                  elapsed_s=perf_counter()-started,source_only=True,FINAL_access=False))


def load_head(source, folder=MODELS):
    seal,lock = read(folder/'SEALED.json'),read(folder/'LOCK.json')
    assert seal['lock_sha256'] == file_sha(folder/'LOCK.json') and seal['source_only']
    assert lock['model_sha256'] == file_sha(HERE/'model.py') and not seal['FINAL_access']
    info = seal['sources'][source]
    path = folder/f'{source}.pth'
    assert file_sha(path) == info['weight_sha256']
    saved = torch.load(path,map_location='cpu',weights_only=True)
    assert saved['source'] == source and saved['readout'] == info['readout']
    model = InstanceGroundingHead().cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['state'],strict=True)
    return model,saved['readout']


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--smoke',action='store_true')
    fit(cli.parse_args().smoke)
