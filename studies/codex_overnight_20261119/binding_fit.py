"""Fit and evaluate the two source-only weights within recording-held-out folds."""
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from binding import alternative_pairs, weights_fit
from binding_collect import HERE, PLATFORMS, PRIVATE, pins, read, sha, atomic_json
from edge_score import csv_atomic


def open_checked(path):
    from predict171 import read_npz

    seal = read(path / 'SEAL.json')
    protocol = pins(path.name)
    assert sha(path / 'LOCK.json') == seal['pin'] and protocol == read(path / 'LOCK.json')
    assert seal['frozen'] and not seal['protected_data_access'] and not seal['target_label_used']
    assert all(sha(path / name) == h for name, h in seal['files'].items())
    return seal, read_npz


def observed(path, proof, npz):
    from labels166 import occupied_owners, physical_labels

    roster = path / 'ROSTER.json'
    assert proof['roster'] == sha(roster)
    content = read(roster)
    cases = []
    for rec in content:
        a = npz(path / f"source/{rec['case']:02d}_{rec['source']}.npz")
        labels, _, seeds, owners = physical_labels(rec['row'], a['seeds'])
        ref = rec['ref_id']
        assert 0 < ref < len(labels)
        # Source annotation proxies remain distinct from prediction-blind target labels.
        s = occupied_owners(a['subject_boxes'], seeds, owners, len(labels))
        r = occupied_owners(a['reference_boxes'], seeds, owners, len(labels))
        rows = alternative_pairs(a, s, r, (0, ref))
        known = (s[:, None] >= 0) & (r[None, :] >= 0)
        id_correct = (s[:, None] == 0) & (r[None, :] == ref)
        cases.append(dict(entry=rec, field=a, training=rows, known=known, correct=id_correct))
    assert len(cases) == len(proof['files'])
    return cases


def measure(case, weights):
    field, correct, known, training = (case[k] for k in ('field','correct','known','training'))
    base = field['baseline'] + field['offset'].sum(0)
    weights = np.asarray(weights)
    adjusted = base + np.einsum('k,ksr->sr',weights,field['edge'])
    i, j = np.unravel_index(adjusted.argmax(), adjusted.shape)
    dual_correct = j < correct.shape[1] and correct[i,j]
    logloss = np.nan
    if training is not None:
        diff, offset = training
        logloss = float(np.logaddexp(0.,-offset-diff@weights).mean())
    return dict(dual_correct=int(dual_correct), eligible=int(training is not None),
                correct_pair_available=int(correct.any()), pair_logloss=logloss)


def run():
    path = PRIVATE / 'source'
    seal, npz = open_checked(path)
    out = PRIVATE / 'SOURCE_FIT.json'
    public = HERE / 'results/binding_source'
    assert not out.exists()
    files = {p.name:sha(p) for p in (Path(__file__),HERE / 'binding.py',HERE / 'binding_protocol.json')}
    atomic_json(PRIVATE / 'SOURCE_FIT_LOCK.json',dict(code=files,source=sha(path / 'SEAL.json')))
    rows = observed(path,seal,npz)
    coefficients, summaries, measurements = {}, [], []
    for p in PLATFORMS:
        block = [r for r in rows if r['entry']['source'] == p]
        matched = [r for r in block if r['training'] is not None]
        kinds = dict(same_info=[], fixed=[], heldout_fit=[])
        fold = sorted({r['entry']['recording'] for r in block})
        for g_index,g in enumerate(fold):
            train = [r['training'] for r in matched if r['entry']['recording'] != g]
            w = weights_fit(train)
            for r in block:
                if r['entry']['recording'] != g:
                    continue
                for method,a in (('same_info',(0.,0.)),('fixed',(1.,1.)),('heldout_fit',w)):
                    value = measure(r,a)
                    kinds[method].append(value)
                    measurements.append(dict(source=p,recording=g_index+1, anonymous_frame=r['entry']['case'],
                                             method=method,**value))
        enough = len(matched)>=8 and len({r['entry']['recording'] for r in matched})>=2
        losses = {name:float(np.nanmean([k['pair_logloss'] for k in val])) if matched else '' for name,val in kinds.items()}
        duo = {name:sum(k['dual_correct'] for k in val) for name,val in kinds.items()}
        chosen = enough and losses['heldout_fit'] < losses['same_info'] and duo['heldout_fit'] >= duo['same_info']
        if chosen:
            w_final = weights_fit([r['training'] for r in matched])
        else:
            w_final = np.ones(2)
        coefficients[p] = dict(weights=list(w_final),source_fit_gate=bool(chosen))
        available = sum(k['correct_pair_available'] for k in kinds['same_info'])
        summaries.append(dict(source=p,frames=len(block),recordings=len(fold),available_pair=available,
                              matched_training_frames=len(matched),
                              matched_recordings=len({r['entry']['recording'] for r in matched}),
                              matched_alt_pairs=sum(len(r['training'][0]) for r in matched),
                              **{f'loss_{k}':v for k,v in losses.items()},
                              **{f'dual_correct_{k}':v for k,v in duo.items()},
                              source_fit_gate=int(chosen),label_kind='source_annotation_proxy'))
    csv_atomic(public/'source_metrics.csv',summaries)
    csv_atomic(public/'source_frames.csv',measurements)
    atomic_json(out,dict(source=sha(path/'SEAL.json'),design=sha(PRIVATE/'SOURCE_FIT_LOCK.json'),
                         coefficients=coefficients,summary=sha(public/'source_metrics.csv'),
                         target_label_used=False,protected_data_access=False))
    print('Source calibration and held-out results', summaries, flush=True)


if __name__ == '__main__':
    run()
