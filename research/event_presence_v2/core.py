"""Within-target nuisance calibration, avoiding other-slot extrapolation."""
import numpy as np
from research.event_presence_v1.core import word_logits,full_reweight

def conditional_event(premax,verbs,nouns,selected):
    v=word_logits(premax,verbs)[:,selected];n=word_logits(premax,nouns)[:,selected].mean(-1);out=[];diagnostics=[]
    for t in range(len(v)):
        take=np.arange(len(v))!=t;x=n[take];mu=x.mean();sd=max(float(x.std()),1e-12)
        design=np.c_[np.ones(take.sum()),(x-mu)/sd];coef=np.linalg.lstsq(design,v[take],rcond=None)[0]
        prediction=np.r_[1.,(n[t]-mu)/sd]@coef;out.append(v[t]-prediction)
        diagnostics.append(dict(rank=int(np.linalg.matrix_rank(design)),noun_scale=sd,extrapolation_z=float((n[t]-mu)/sd),slopes=(coef[1]/sd).tolist()))
    return np.stack(out),diagnostics
