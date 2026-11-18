"""Calibrate an optional reference role from source support rather than a fixed null penalty."""
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from binding_collect import HERE, PLATFORMS, PRIVATE as BINDING, read, sha, atomic_json
from binding_fit import open_checked
from edge_score import csv_atomic

DATA = HERE/'private/abstain'
SPEC = HERE/'abstain_protocol.json'


def witnesses(b):
    n = len(b['reference_boxes'])
    ns = len(b['subject_boxes'])
    f, d = b['f'][ns:], b['d'][ns:]
    a, z = b['f_strength'][ns:],b['d_strength'][ns:]
    physical = np.zeros((n,3))
    common = np.isfinite(f)&np.isfinite(d)
    if common.any():
        vf = f[common]-logsumexp(f[common])
        vd = d[common]-logsumexp(d[common])
        physical[common,0] = np.tanh(vd-vf)
        physical[common,1] = np.minimum(a[common],z[common])
    return np.vstack((physical,[0.,0.,1.]))


def source_examples():
    from labels166 import physical_labels, occupied_owners

    source=BINDING/'source'
    certificate,npz=open_checked(source)
    assert certificate['roster']==sha(source/'ROSTER.json')
    out=[]
    for case in read(source/'ROSTER.json'):
        x=npz(source/f"source/{case['case']:02d}_{case['source']}.npz")
        assert len(x['reference_boxes'])
        names,_,cloud,owners=physical_labels(case['row'],x['seeds'])
        ref=case['ref_id']
        r=occupied_owners(x['reference_boxes'],cloud,owners,len(names))
        enough=(owners==ref).sum()>=3
        good=np.r_[r==ref,not np.any(r==ref)]
        # Isolate per-reference scores and null. The same joint energy will be used when testing.
        old=x['baseline'][:, :-1]
        unary=old[0]-x['sign']*x['left'][0,:-1]
        unary=unary-unary.max()
        offsets=np.r_[unary,0.]
        out.append(dict(case=case['case'],source=case['source'],recording=case['recording'],
                       x=witnesses(x),b=offsets,correct=good,allowed=bool(enough),
                       covered=bool((r==ref).any()),baseline_joint=x['baseline'],
                       offset=x['offset'].sum(0),edge=x['edge'].sum(0)))
    assert len(out)==len(certificate['files'])
    return out


def fit(cases):
    def loss(w):
        values=[]
        derivatives=[]
        for e in cases:
            z=e['b']+e['x']@w
            good=e['correct']
            y=z[good]
            values.append(logsumexp(z)-logsumexp(y))
            derivatives.append(softmax(z)@e['x']-softmax(y)@e['x'][good])
        return float(np.mean(values)+.05*w@w),np.mean(derivatives,axis=0)+.1*w
    if not cases:
        return np.zeros(3)
    fit_result=minimize(loss,np.zeros(3),jac=True,bounds=[(-2.,2.)]*3,method='L-BFGS-B',options={'maxiter':100})
    assert fit_result.success
    return fit_result.x


def evaluate_score(e,w,mode):
    q=e['b']+e['x']@w
    count=len(q)-1
    if mode=='forced':
        idx=int(q[:count].argmax())
    elif mode=='old_forced':
        idx=int(e['b'][:count].argmax())
    else:
        idx=int(q.argmax())
    return dict(correct=int(e['correct'][idx]),abstained=int(idx==count),count_phys=count)


def main():
    DATA.mkdir(parents=True,mode=0o700,exist_ok=True)
    own=DATA/'SOURCE_CALIBRATION.json'
    assert not own.exists()
    lock=dict(protocol=sha(SPEC),code=sha(Path(__file__)),source=sha(BINDING/'source/SEAL.json'))
    atomic_json(DATA/'LOCK.json',lock)
    instances=source_examples()
    conclusions,measurements,internal=[],[],{}
    for s in PLATFORMS:
        block=[c for c in instances if c['source']==s]
        eligible=[c for c in block if c['allowed']]
        folds=sorted({c['recording'] for c in eligible})
        rows=[]
        for ordinal,g in enumerate(folds):
            trials=[c for c in eligible if c['recording']!=g]
            w=fit(trials)
            for x in eligible:
                if x['recording']!=g:
                    continue
                for arm in ('old_forced','forced','null'):
                    o=evaluate_score(x,w,arm)
                    rows.append(dict(source=s,frame=x['case'],recording_alias=ordinal+1,
                                     arm=arm,covered=int(x['covered']),**o))
        measurements.extend(rows)
        n={a:sum(r['correct'] for r in rows if r['arm']==a) for a in ('old_forced','forced','null')}
        right=sum(r['correct'] and not r['abstained'] for r in rows if r['arm']=='null')
        unknown=sum(r['correct'] and r['abstained'] for r in rows if r['arm']=='null')
        gate=len(folds)>=2 and right>0 and unknown>0 and n['null']>max(n['forced'],n['old_forced'])
        internal[s]=dict(assess=bool(gate),weight=fit(eligible).tolist() if gate else [])
        conclusions.append(dict(source=s,n=len(block),eligible=len(eligible),recordings=len(folds),
                                target_gate=int(gate),not_covered=sum(not x['covered'] for x in eligible),
                                correct_old_forced=n['old_forced'],correct_forced=n['forced'],correct_null=n['null'],
                                null_correct_known=right,null_correct_abstained=unknown,
                                null_total_abstained=sum(r['abstained'] for r in rows if r['arm']=='null')))
    out=HERE/'results/abstain_source'
    csv_atomic(out/'source_rows.csv',measurements)
    csv_atomic(out/'summary.csv',conclusions)
    atomic_json(own,dict(lock=sha(DATA/'LOCK.json'),source_only=True,coefficients=internal,
                         result=sha(out/'summary.csv'),protected_data_access=False))
    print(conclusions,flush=True)


if __name__=='__main__':
    main()
