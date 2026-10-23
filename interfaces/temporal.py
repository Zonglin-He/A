"""Temporal evidence adapters are separate from local representation feedback.

The native model supplies priors/candidates, temporal mapping and final readout.
A single generated timestamp answer does NOT imply alternative candidate mass.
No artificial candidates or probabilities are invented by these contracts.
"""
from dataclasses import dataclass
from typing import Protocol
import numpy as np

class TemporalEvidence(Protocol):
    times: np.ndarray
    prior: np.ndarray
    def soft_membership(self): ...
    def interval_scores(self,frame_feedback): ...
    def revise(self,frame_feedback): ...
    def readout(self,posterior): ...

def update(prior,scores):
    if np.all(scores[prior>0]==0):return prior.copy()
    take=prior>0;x=np.log(prior[take])+scores[take];x-=x.max();out=np.zeros_like(prior);out[take]=np.exp(x);out/=out.sum();return out

@dataclass
class JointGridPosterior:
    times: np.ndarray
    prior: np.ndarray
    native_readout: object
    def __post_init__(self):
        self.times=np.asarray(self.times);self.prior=np.asarray(self.prior,dtype=np.float64)
        if self.prior.shape!=(len(self.times),len(self.times))or np.any(self.prior<0)or not np.isfinite(self.prior).all()or not np.isclose(self.prior.sum(),1):raise ValueError('Explicit normalized native joint posterior required')
        if not np.all(np.diff(self.times)>0)or np.any(np.tril(self.prior,-1)>0):raise ValueError('Ordered original temporal grid and legal intervals required')
    def soft_membership(self):
        q=self.prior;return np.diag(np.cumsum(np.cumsum(q,axis=0)[:,::-1],axis=1)[:,::-1]).copy()
    def interval_scores(self,L):
        if len(L)!=len(self.times):raise ValueError('Feedback/time grid mismatch')
        c=np.r_[0.,np.cumsum(L)];return c[1:][None,:]-c[:-1][:,None]
    def revise(self,L):return update(self.prior,self.interval_scores(L))
    def readout(self,q):return self.native_readout(self.times,q)

@dataclass
class CandidateListPrior:
    times: np.ndarray
    prior: np.ndarray
    intervals: np.ndarray
    coverage: np.ndarray
    native_readout: object
    def __post_init__(self):
        self.times=np.asarray(self.times);self.prior=np.asarray(self.prior,dtype=np.float64);self.intervals=np.asarray(self.intervals);self.coverage=np.asarray(self.coverage,dtype=np.float64)
        k=len(self.prior)
        if self.prior.ndim!=1 or self.intervals.shape!=(k,2)or self.coverage.shape!=(k,len(self.times)):raise ValueError('Explicit candidates, native priors and per-node candidate coverage required')
        if not np.isfinite(self.prior).all()or np.any(self.prior<0)or not np.isclose(self.prior.sum(),1)or not np.isfinite(self.coverage).all()or np.any((self.coverage<0)|(self.coverage>1)):raise ValueError('Invalid candidate prior or temporal correspondence')
        if not np.all(np.diff(self.times)>0)or np.any(self.intervals[:,1]<=self.intervals[:,0]):raise ValueError('Physical time mapping required')
    def soft_membership(self):return self.prior@self.coverage
    def interval_scores(self,L):return self.coverage@np.asarray(L)
    def revise(self,L):return update(self.prior,self.interval_scores(L))
    def readout(self,q):return self.native_readout(self.intervals,q)
    def require_alternatives(self):
        if np.count_nonzero(self.prior)<=1:raise ValueError('Point-only timestamp output cannot correct boundaries by multiplicative update. Supply and validate explicit candidate-generation/scoring adapter first.')
