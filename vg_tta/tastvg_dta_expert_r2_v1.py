"""Prelocked raw-proposal center selectors and GT-free deployment read guard."""
import os
import numpy as np


def validate_support(proposals, confidence):
    p=np.asarray(proposals,dtype=np.float64);c=np.asarray(confidence,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=2 or not len(p) or c.shape!=(len(p),):
        raise ValueError('Nonempty aligned expert proposal support is required')
    if not np.isfinite(p).all() or not np.isfinite(c).all() or not (p[:,1]>p[:,0]).all():
        raise ValueError('Invalid expert proposal interval/confidence')
    return p,c


def deploy_choice(proposals, confidence):
    p,c=validate_support(proposals,confidence)
    i=int(np.argmax(c))
    return dict(index=i,interval=p[i].tolist(),confidence=float(c[i]),GT_used=False)


def oracle_choice(proposals, confidence, truth):
    p,c=validate_support(proposals,confidence);s,e=map(float,truth)
    assert np.isfinite([s,e]).all() and e>s
    overlap=np.maximum(0,np.minimum(p[:,1],e)-np.maximum(p[:,0],s))
    v=overlap/(p[:,1]-p[:,0]+e-s-overlap);i=int(np.argmax(v))
    return dict(index=i,interval=p[i].tolist(),confidence=float(c[i]),continuous_tIoU=float(v[i]),GT_used=True)


def read_guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
    path=os.fsdecode(args[0]);mode=args[1] if len(args)>1 else None
    reading=mode is None or (isinstance(mode,str) and ('r' in mode or '+' in mode)) or (isinstance(mode,int) and mode&os.O_ACCMODE!=os.O_WRONLY)
    if reading and any(s in path for s in ['GT_LABELS','SOURCE_GT','GT_EXPOSURE','GT_SUBSET','test_annotations','valv2_proc',
         'vidstd-test-anno','/oracle_runs/','/ORACLE_SELECTION','/SOURCE_VALIDATION_ROWS','/SOURCE_SELECTION.json',
         '/ROWS.json','/SUMMARY.json','/CASES.json','/ADAPTATION_TRACES_SCORED.json']):
        raise PermissionError('E-Deploy cannot read target GT, oracle outputs or scored results: '+path)

