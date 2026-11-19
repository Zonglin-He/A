"""Fit source-only scales for the frozen subject and reference readout components."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from capture181 import HERE, SOURCE, manifest, protocol
from scope import PLATFORMS, file_sha, read, save
from train import metrics

OUTPUT = HERE/'calibration_PRIVATE.json'


def components(terms,kind):
    p,u,j,r,w,t = np.asarray(terms,dtype=np.float64).transpose(2,0,1)
    values = [p,u]
    if kind=='joint':
        values.append(j-u+r)
    elif kind=='mismatch':
        values.append(w-u+t)
    elif kind!='subject':
        raise ValueError(f'Unknown calibration mode {kind}')
    return np.stack(values,axis=-1)


def prep_loss(x,ious):
    groups = []
    for th,multiplier in ((.25,1.),(.50,.5)):
        mask = ious>th
        eligible = mask.any(axis=1)
        if eligible.any():
            groups.append((x[eligible],mask[eligible],multiplier))
    if not groups:
        raise ValueError('Source slates must contain at least one labeled positive')
    return groups


def fit_scales(x,ious,cfg):
    groups=prep_loss(x,ious)
    identity=np.ones(x.shape[-1])
    reg=cfg['regularization']

    def objective(scales):
        shift=scales-identity
        value=reg*(shift@shift)
        grad=2*reg*shift
        for features,positive,multiplier in groups:
            logits=features@scales
            all_lse=logsumexp(logits,axis=-1,keepdims=True)
            pos_logits=np.where(positive,logits,-np.inf)
            pos_lse=logsumexp(pos_logits,axis=-1,keepdims=True)
            all_mass=np.exp(logits-all_lse)
            pos_mass=np.exp(pos_logits-pos_lse)
            value+=multiplier*float((all_lse-pos_lse).mean())
            grad+=multiplier*((all_mass-pos_mass)[...,None]*features).sum(axis=1).mean(axis=0)
        return value,grad

    bounds=[tuple(v) for v in cfg['bounds'][:len(identity)]]
    fit=minimize(objective,identity,method='L-BFGS-B',jac=True,bounds=bounds,
                 options=dict(maxiter=120,ftol=1e-12,gtol=1e-8))
    if not fit.success:
        raise RuntimeError(f'Source fit failed: {fit.message}')
    return fit.x,dict(initial_loss=float(objective(identity)[0]),fitted_loss=float(fit.fun),
                      objective_rows=[len(y) for _,y,_ in groups])


def select(logit,incumbent,margin):
    winner=logit.argmax(axis=1)
    row=np.arange(len(logit))
    if margin is None:
        return incumbent.copy()
    return np.where(logit[row,winner]>logit[row,incumbent]+margin,winner,incumbent)


def source_margin(x,ious,incumbent,w,options):
    logits=x@w
    row=np.arange(len(x))
    trials=[]
    for margin in options:
        picked=select(logits,incumbent,margin)
        vals=ious[row,picked]
        trials.append(dict(margin=margin,switches=int((picked!=incumbent).sum()),**metrics(vals)))
    best=max(trials,key=lambda v:(v['Acc25'],v['Acc50'],v['mIoU'],-v['switches']))
    return best,trials


def fit():
    source=read(SOURCE/'SEALED.json')
    assert source['implementation']==manifest() and source['source_only'] and not source['target_labels_used']
    assert not source['FINAL_access']
    cfg=protocol()['source_calibration']
    result={}
    for platform in PLATFORMS:
        splits={}
        for name in ('fit','calibration'):
            path=SOURCE/f'{platform}_{name}.npz'
            assert file_sha(path)==source['files'][path.name]['sha256']
            with np.load(path,allow_pickle=False) as z:
                splits[name]={k:z[k] for k in ('terms','ious','incumbent','chosen')}
        training,held=splits['fit'],splits['calibration']
        assert len(training['ious'])==cfg['fit_per_source']
        assert len(held['ious'])==cfg['heldout_source_counts'][platform]
        row=np.arange(len(held['ious']))
        records={}
        for mode in ('subject','joint'):
            w,learning=fit_scales(components(training['terms'],mode),training['ious'],cfg)
            margin,trial=source_margin(components(held['terms'],mode),held['ious'],held['incumbent'],w,
                                       cfg['margin_grid'])
            records[mode]=dict(weights=w.tolist(),selected=margin,validated=trial,fit=learning)
        result[platform]=dict(controls={
            'v1':metrics(held['ious'][row,held['incumbent']]),
            **{method:metrics(held['ious'][row,held['chosen'][:,j]]) for j,method in
               enumerate(('frozen_joint','frozen_subject','frozen_mismatch'))}},
            source_training_rows=len(training['ious']),models=records)
        print(platform,[records[m]['selected']['margin'] for m in ('subject','joint')],flush=True)
    save(OUTPUT,dict(source_only=True,source_seal=file_sha(SOURCE/'SEALED.json'),
         code_sha256=file_sha(HERE/'calibrate181.py'),protocol_sha256=file_sha(HERE/'PROTOCOL.json'),
         models_by_source=result,target_labels_used=False,FINAL_access=False))


def load_calibration():
    value=read(OUTPUT)
    assert value['source_only'] and not value['target_labels_used'] and not value['FINAL_access']
    assert value['source_seal']==file_sha(SOURCE/'SEALED.json')
    assert value['code_sha256']==file_sha(HERE/'calibrate181.py')
    assert value['protocol_sha256']==file_sha(HERE/'PROTOCOL.json')
    return value


def final_choices(terms,frozen,incumbent,source_model):
    learned=[]
    for constraint,representation in (('subject','subject'),('joint','joint'),('joint','mismatch')):
        setting=source_model['models'][constraint]
        logits=components(terms,representation)@np.asarray(setting['weights'])
        learned.append(select(logits,incumbent,setting['selected']['margin']))
    return np.column_stack((incumbent,frozen,*learned)).astype(np.int8)


if __name__=='__main__':
    fit()
