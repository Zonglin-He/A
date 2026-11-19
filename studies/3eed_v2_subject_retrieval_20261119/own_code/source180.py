"""Fit only a subject retriever and lock K on separate recordings of each source."""

import argparse
from time import perf_counter

import numpy as np
import torch

from retrieval180 import HERE, PRIOR, TRAINING, K_VALUES, SubjectRetriever, RetrievalGrounder
from scope import PLATFORMS, ROOT, file_sha, read, save, source_rows
from features import FEATURES, observation
from train import prepare_device, read_features, many_iou, metrics


def protocol():
    return read(HERE/'PROTOCOL.json')


def implementation():
    files = [HERE/x for x in ('PROTOCOL.json','retrieval180.py','source180.py')]
    files += [ROOT/'crossplatform'/x for x in ('ground.py','CURRENT_METHOD.json','candidate_frontend.py',
                                               'single_frame_sources.json','v1_evidence.py','v1_decoder.py')]
    files += [HERE.parent/'3eed_v1_retrieval_binding_refinement_20261119'/'mechanisms.py']
    files += [PRIOR/'features.py',PRIOR/'scope.py',PRIOR/'train.py']
    return {str(p.relative_to(ROOT)):file_sha(p) for p in files}


def fit_one(source, seed, cfg):
    cached = read_features(FEATURES,source,'fit')
    natural = cached['origin']<2
    feature = {k:cached[k][natural] for k in ('anchor_query','token_proj','prior','anchors')}
    gt = cached['gt'][natural,0].cpu().numpy()
    boxes = feature['anchors'].cpu().numpy()
    positive = np.stack([many_iou(p,b)>.25 for p,b in zip(boxes,gt)])
    assert len(positive)==cfg['n_fit_per_source']
    y = torch.as_tensor(positive,device='cuda')
    indices = torch.where(y.any(-1))[0]
    if len(indices)==0:
        raise RuntimeError(f'{source} has no labeled positive proposals')
    del cached
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = SubjectRetriever().cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    history = []
    for epoch in range(1,cfg['epochs']+1):
        nav = indices[torch.randperm(len(indices),generator=rng).to(indices.device)]
        weighted = 0.
        for block in nav.split(cfg['batch_size']):
            scores = model({k:v[block] for k,v in feature.items()})
            supported_mass = scores.masked_fill(~y[block],-torch.inf).logsumexp(dim=-1)
            loss = (scores.logsumexp(dim=-1)-supported_mass).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{source} rank fitting loss is not finite')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            weighted += float(loss.detach())*len(block)
        history.append(dict(epoch=epoch,loss=weighted/len(indices)))
    path=TRAINING/f'{source}.pth'
    torch.save(dict(source=source,state={k:v.cpu() for k,v in model.state_dict().items()}),path)
    return dict(weights_sha256=file_sha(path),n_fit=len(positive),positive_rows=len(indices),
                positive_proposals=int(positive.sum()),n_parameters=sum(p.numel() for p in model.parameters()),
                training=history)


def fit():
    cfg=protocol()
    assert file_sha(FEATURES/'SEALED.json')==cfg['source_feature_seal_sha256']
    assert file_sha(PRIOR/'inputs_PRIVATE/SEAL.json')==cfg['source_input_seal_sha256']
    prepare_device()
    TRAINING.mkdir(mode=0o700)
    sources={}
    start=perf_counter()
    for i,source in enumerate(PLATFORMS):
        result=fit_one(source,cfg['training']['seed']+i,cfg['training'])
        sources[source]=result
        print(source,'eligible',result['positive_rows'],'/',result['n_fit'],
              'loss',round(result['training'][-1]['loss'],4),flush=True)
    save(TRAINING/'FITTED.json',dict(implementation=implementation(),sources=sources,
         source_only=True,FINAL_access=False,elapsed_seconds=perf_counter()-start))


def calibrated():
    selection=read(TRAINING/'SELECTION.json')
    assert selection['source_only'] and not selection['target_labels_used'] and not selection['FINAL_access']
    assert selection['implementation']==implementation()
    assert selection['fitted_sha256']==file_sha(TRAINING/'FITTED.json')
    return selection


@torch.inference_mode()
def calibrate():
    fitted=read(TRAINING/'FITTED.json')
    assert fitted['implementation']==implementation() and fitted['source_only'] and not fitted['FINAL_access']
    prepare_device()
    summary={}
    start=perf_counter()
    for source in PLATFORMS:
        model=RetrievalGrounder(source)
        rows=source_rows(source,'calibration')
        cached=read_features(FEATURES,source,'calibration')
        natural=cached['origin']<2
        xyz=cached['anchors'][natural].cpu().numpy()
        truths=cached['gt'][natural,0].cpu().numpy()
        assert len(rows)==len(truths)==protocol()['training']['calibration_counts'][source]
        output, recalls={},{ }
        for row,expected,truth in zip(rows,xyz,truths):
            pool,trials,_,_=model.compare(observation(row))
            np.testing.assert_array_equal(pool,expected)
            measured=many_iou(pool,truth)
            for name,(indices,winner) in trials.items():
                output.setdefault(name,[]).append(measured[winner])
                recalls.setdefault(name,[]).append(measured[indices].max())
        audit={name:dict(**metrics(x),recall25=metrics(recalls[name])['Acc25'],
                         recall50=metrics(recalls[name])['Acc50']) for name,x in output.items()}
        chosen=max(K_VALUES,key=lambda k:(audit[f'trained{k}']['Acc25'],audit[f'trained{k}']['Acc50'],
                                         audit[f'trained{k}']['mIoU'],-k))
        summary[source]=dict(selected_k=chosen,n=len(rows),recordings=len({row['recording'] for row in rows}),
                            heldout=audit)
        print(source,'locked K',chosen,'n',len(rows),'heldout Acc25',audit[f'trained{chosen}']['Acc25'],flush=True)
        del model,cached
        torch.cuda.empty_cache()
    save(TRAINING/'SELECTION.json',dict(implementation=implementation(),sources=summary,source_only=True,
         fitted_sha256=file_sha(TRAINING/'FITTED.json'),target_predictions_used=False,
         target_labels_used=False,FINAL_access=False,elapsed_seconds=perf_counter()-start))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('fit','calibrate'))
    {'fit':fit,'calibrate':calibrate}[parser.parse_args().phase]()
