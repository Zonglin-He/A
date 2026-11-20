"""Analytic heading-marginalized overlap and four-subject nomination."""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
ARMS = ('v1','shared_heading','measured_pca','equal_area_disk','crossed_axes',
        'vote_only','diversity_only','axis_45')


@lru_cache(maxsize=1)
def library():
    source = HERE/'overlap.cpp'
    build = HERE/'run_PRIVATE'/'build'
    build.mkdir(parents=True,mode=0o700,exist_ok=True)
    path = build/('kernels_'+hashlib.sha256(source.read_bytes()).hexdigest()+'.so')
    if not path.exists():
        draft = path.with_suffix('.'+str(os.getpid())+'.tmp')
        subprocess.run(['c++','-std=c++17','-O3','-fPIC','-shared',str(source),'-o',str(draft)],check=True)
        draft.replace(path)
    lib = ctypes.CDLL(str(path))
    matrix = np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    lib.kernels.argtypes = (matrix,ctypes.c_int,matrix)
    lib.kernels.restype = None
    return lib


def metrics(boxes):
    boxes = np.ascontiguousarray(boxes,dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1]!=7 or not np.isfinite(boxes).all():
        raise ValueError('Expected finite boxes with native dimensions and measured yaw')
    out=np.empty((5,len(boxes),len(boxes)),np.float64)
    library().kernels(boxes,len(boxes),out)
    return out


def vote(matrix, probability):
    score = np.log(np.maximum(probability,1e-300))-np.log(np.square(matrix).sum(1))
    return matrix@np.exp(score-logsumexp(score))


def shortlist(utility, overlap, budget=4):
    order=np.argsort(-utility,kind='stable')
    kept=[int(order[0])]
    for i in order[1:]:
        if (overlap[i,kept]<=.35).all():
            kept.append(int(i))
            if len(kept)==budget:
                break
    if len(kept)<budget:
        kept.extend(int(i) for i in order if i not in kept)
    ids=np.array(kept[:budget])
    log=np.log(np.maximum(utility[ids],1e-80))
    return ids,(log-log[0]).clip(-20.,0.).astype(np.float32)


def nominate(native, oriented, probability, axis):
    raw=np.column_stack((native,oriented[:,6]))
    shared,measured,disk,crossed,fixed45=metrics(raw)
    raw_vote=vote(axis,probability)
    shared_vote=vote(shared,probability)
    alternatives=[(raw_vote,axis),(shared_vote,shared),(vote(measured,probability),measured),
                  (vote(disk,probability),disk),(vote(crossed,probability),crossed),
                  (shared_vote,axis),(raw_vote,shared),(vote(fixed45,probability),fixed45)]
    ids,priors=zip(*(shortlist(v,k) for v,k in alternatives))
    return np.stack(ids),np.stack(priors)
